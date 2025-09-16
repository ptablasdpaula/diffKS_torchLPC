import torch
import torch.nn as nn
import torch.nn.functional as F
from diffKS import DiffKS
from core import hz_to_samples, sigmoid_valid_gain_range, scale_function, amp_to_impulse_response, fft_convolve
import math

MIN_HZ = 20
MAX_HZ = 8000

# =============================================================
# Small TCN building blocks
# =============================================================
class Conv1dSame(nn.Conv1d):
    """Conv1d with 'same' padding for stride=1; for stride>1 we use manual padding.
    """
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, stride=1):
        pad = (kernel_size - 1) // 2 * dilation if stride == 1 else 0
        super().__init__(in_ch, out_ch, kernel_size, stride=stride, padding=pad, dilation=dilation)

    def forward(self, x):
        if self.stride[0] == 1:
            return super().forward(x)
        # Manual 'same-ish' padding for strided conv (keep length//stride)
        k = self.kernel_size[0]
        d = self.dilation[0]
        pad_total = max(0, (x.shape[-1] - 1) * (self.stride[0] - 1) + (k - 1) * d)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        x = F.pad(x, (pad_left, pad_right))
        return super().forward(x)

class DilatedResBlock(nn.Module):
    def __init__(self, ch, kernel_size=3, dilation=1):
        super().__init__()
        self.conv1 = Conv1dSame(ch, ch, kernel_size, dilation=dilation)
        self.conv2 = Conv1dSame(ch, ch, kernel_size, dilation=1)
        self.norm1 = nn.GroupNorm(1, ch)
        self.norm2 = nn.GroupNorm(1, ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv1(self.act(self.norm1(x)))
        y = self.conv2(self.act(self.norm2(y)))
        return x + y

# =============================================================
# TCN feature extractor for segment-level prediction
# =============================================================
class TCN(nn.Module):
    def __init__(self, in_ch=2, ch=64, n_blocks=6, kernel=3, loop_out=3, n_bands=64):
        super().__init__()
        self.inp = Conv1dSame(in_ch, ch, kernel, dilation=1)
        blocks = []
        for i in range(n_blocks):
            blocks.append(DilatedResBlock(ch, kernel_size=kernel, dilation=2 ** i))
        self.tcn = nn.Sequential(*blocks)
        self.post = Conv1dSame(ch, ch, 1)
        # Heads are applied on segment-pooled features (AttentionPooling outside)
        self.loop_proj = nn.Linear(ch, loop_out)
        assert self.loop_proj.bias is not None and self.loop_proj.bias.numel() >= loop_out, "loop_proj must have a bias of size >= loop_out"
        self.band_proj = nn.Linear(ch, n_bands)
        self.gain_proj = nn.Linear(ch, 1)

        nn.init.constant_(self.loop_proj.bias[0],  2.0)   # g logit → sigmoid ≈ 0.88
        nn.init.constant_(self.gain_proj.bias, 1.4)
        nn.init.constant_(self.band_proj.bias, -6.0)

    def forward(self, pitch_seq, loud_seq):
        """
        Inputs:
          pitch_seq, loud_seq: [B, T, 1]
        Returns:
          h: hidden features [B, C, T]
        """
        x = torch.cat([pitch_seq, loud_seq], dim=-1).transpose(1, 2)  # [B, 2, T]
        h = self.inp(x)
        h = self.tcn(h)
        h = self.post(h)  # [B, C, T]
        return h

# =============================================================
# nnKarplusStrong with TCN on pitch/loudness → DiffKS loop + GEQ
# =============================================================
class nnKarplusStrong(nn.Module):
    def __init__(self,
                 interpolation_type,
                 filter_type,
                 n_noise_bands: int = 16,
                 tcn_ch: int = 64, ):
        super().__init__()
        self.n_noise_bands = n_noise_bands

        self.log2_f0_min = math.log2(MIN_HZ)
        self.log2_f0_max = math.log2(MAX_HZ)
        self.n_noise_bands = int(n_noise_bands)
        self.max_geq_db = 6.0  # GEQ is boost-only up to +6 dB

        self.tcn = TCN(in_ch=2, ch=tcn_ch, n_blocks=6, kernel=3,
                       loop_out=2, n_bands=n_noise_bands)

        self.decoder = DiffKS(
            interp_type=interpolation_type,
            use_double_precision=False,
            loop_filter_kind=filter_type,
        )
        for p in self.decoder.parameters():
            p.requires_grad = False

    def _scale_conditioning(self, pitch_seq: torch.Tensor, loud_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Map inputs to [0,1] for the TCN, matching your dataset:
          • pitch_seq is in Hz  → clamp [E2,E6], log2, min‑max to [0,1]
          • loud_seq already in [0,1] → clamp to [0,1]
        Inputs/outputs: [B, T, 1].
        """
        # Pitch: assume Hz
        f0_hz = pitch_seq.squeeze(-1)
        f0_hz = f0_hz.clamp(min=MIN_HZ, max=MAX_HZ)
        f0_log2 = torch.log2(f0_hz.clamp(min=1e-6))
        denom = (self.log2_f0_max - self.log2_f0_min)
        f0_scaled = (f0_log2 - self.log2_f0_min) / max(1e-12, denom)
        f0_scaled = f0_scaled.clamp(0.0, 1.0).unsqueeze(-1)

        # Loudness: assume already unit-scaled
        L_scaled = loud_seq.clamp(0.0, 1.0)

        return f0_scaled, L_scaled
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, pitch, _loudness, audio, audio_sr, return_parameters=False,):
        B, N = audio.shape
        T_frames = pitch.size(1)

        # --- Low-rate conditioning (DDX7-style scaling) ---
        pitch_scaled, loud_scaled = self._scale_conditioning(pitch, _loudness)

        # --- TCN features at frame-rate ---
        # h: [B, C, T]
        h = self.tcn(pitch_scaled, loud_scaled)
        C = h.shape[1]

        # --- Excitation noise: simple white noise per batch ---
        burst_stream = torch.randn(B, N, device=audio.device, dtype=audio.dtype)

        # --- Frame-level loop/geq/gain parameters (no global pooling) ---
        loop_out = self.tcn.loop_proj.out_features
        K = int(self.n_noise_bands)
        # h: [B, C, T_frames] -> [B, T_frames, C]
        h_t = h.transpose(1, 2)  # [B, T_frames, C]
        # Compute loop_logits, geq_logits, gain_logits at frame level
        loop_logits = self.tcn.loop_proj(h_t)  # [B, T_frames, loop_out]
        geq_logits = self.tcn.band_proj(h_t)   # [B, T_frames, K]

        # --- IRCAM frequency-domain filterbank (vectorized; no Python frame loop) ---
        # Map per-frame band logits -> positive amplitudes
        band_amp_frames = scale_function(geq_logits)            # [B, T_frames, K]
        # Choose block size from audio/frames; last block will be cropped
        block = int(math.ceil(N / max(1, T_frames)))
        # Build per-frame impulse responses and convolve white noise blocks
        impulse = amp_to_impulse_response(band_amp_frames, block)     # [B, T_frames, block]
        noise_blocks = (torch.rand(B, T_frames, block, device=audio.device, dtype=audio.dtype) * 2 - 1)
        shaped_blocks = fft_convolve(noise_blocks, impulse).contiguous()  # [B, T_frames, block]
        # Stitch blocks and crop to exact length
        excitation = shaped_blocks.reshape(B, T_frames * block)
        excitation = excitation[:, :N]

        # --- Map loop logits -> bounded params and upsample to samples ---
        logits = loop_logits  # [B, T_frames, 2]
        g = sigmoid_valid_gain_range(logits[..., 0])        # (0, 1), biased near 1
        p = torch.sigmoid(logits[..., 1])             # (0, 1)
        l_b_frames = torch.stack([g, p], dim=-1)      # [B, T_frames, 2]
        # Upsample per-frame loop params to per-sample
        l_b_samples = F.interpolate(
            l_b_frames.permute(0, 2, 1), size=N, mode="linear", align_corners=False
        ).permute(0, 2, 1)  # [B, N, 2]

        # --- Convert f0 (Hz) frames -> per-sample period in samples ---
        f0_frames_hz = pitch.squeeze(2).clamp(MIN_HZ, MAX_HZ)        # [B, T_frames]
        f0_frames_samples = hz_to_samples(f0_frames_hz, fs=int(audio_sr))  # [B, T_frames]
        f0_samples = F.interpolate(
            f0_frames_samples.unsqueeze(1), size=N, mode="linear", align_corners=False
        ).squeeze(1)  # [B, N]

        # --- DiffKS (pure DSP): per‑sample f0 + per‑sample loop taps ---
        out = self.decoder(
            f0=f0_samples,
            input=excitation,
            l_b=l_b_samples,
            invert=False,
        )  # [B, N]

        if return_parameters:
            return {
                "resonator_excitation": excitation.detach(),
                "band_amp_frames": band_amp_frames.detach(),  # [B, T_frames, K]
                "loop_logits": loop_logits.detach(),
                "loop_params_samples": l_b_samples.detach(), # [B, N, 2] (g, p)
                "f0_samples": f0_samples.detach(),           # [B, N]
            }

        return out