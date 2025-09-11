import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffKS import DiffKS
from core import make_onset_noise, detect_onsets_librosa
import math

# Additional imports for Jordi-style attention pooling
from einops import rearrange, repeat

import torch.fft as fft

MIN_HZ = 20
MAX_HZ = 8000

# ==============================
# FIR noise filterbank utilities
# ==============================
# Bounded dB-range mapping: [min_db, max_db].
def scale_function(x, min_db: float = -40.0, max_db: float = 16.0):
    """
    Map logits x -> linear gain g with a bounded dB range.
    Range: [min_db, max_db] dB (defaults to -80 dB .. +16 dB).
    Use a negative bias at call-site (e.g., x-5) to start near the quiet floor.
    """
    # σ(x)∈(0,1) → scale_db ∈ [min_db, max_db]
    scale_db = min_db + (max_db - min_db) * torch.sigmoid(x)
    # Convert dB to linear
    g = torch.pow(torch.tensor(10.0, device=x.device, dtype=x.dtype), scale_db / 20.0)
    return g


# Helper: build log-frequency triangular filterbank for band-to-linear mapping
def _make_log_tri_filterbank(n_bins_lin: int, fs: int, n_bands: int, fmin: float, fmax: float, device, dtype):
    """
    Returns [n_bins_lin, n_bands] weight matrix for mapping n_bands log-f bands to linear-f spectrum.
    Each row is a linear-f bin; each col is a band; rows sum to 1 where covered.
    """
    # Clamp fmax to < fs/2
    nyq = fs / 2.0
    if fmax >= nyq:
        fmax = nyq - 1e-4
    # Linear-f bin frequencies (including DC and Nyquist)
    freqs = torch.linspace(0, nyq, n_bins_lin, device=device, dtype=dtype)  # [n_bins_lin]
    # Log-space centers
    exponents = torch.linspace(
        math.log10(fmin), math.log10(fmax), n_bands,
        device=device, dtype=dtype
    )
    centers = torch.pow(10.0, exponents)    # Edges: geometric mean between centers

    edges = torch.zeros(n_bands + 1, device=device, dtype=dtype)
    edges[1:-1] = torch.sqrt(centers[:-1] * centers[1:])
    edges[0] = fmin
    edges[-1] = fmax
    # Compute log10 for all freqs/centers/edges
    log_freqs = torch.log10(freqs.clamp(min=1e-8))
    log_centers = torch.log10(centers)
    log_edges = torch.log10(edges)
    # Build triangular weights
    fb = torch.zeros(n_bins_lin, n_bands, device=device, dtype=dtype)
    for k in range(n_bands):
        # For each band: left, center, right in log-f
        left = log_edges[k]
        c = log_centers[k]
        right = log_edges[k+1]
        # Compute rising and falling slopes
        # Mask for bins within band
        in_band = (log_freqs >= left) & (log_freqs <= right)
        # Rising slope
        rise = ((log_freqs - left) / (c - left)).clamp(0, 1)
        # Falling slope
        fall = ((right - log_freqs) / (right - c)).clamp(0, 1)
        w = torch.minimum(rise, fall)
        w = w * in_band
        fb[:, k] = w
    # Normalize so each linear bin sums to 1 across bands (where covered)
    norm = fb.sum(dim=1, keepdim=True).clamp_min(1e-12)
    fb = fb / norm
    fb[torch.isnan(fb)] = 0.0
    return fb  # [n_bins_lin, n_bands]

def amp_to_impulse_response(
    amp: torch.Tensor,
    target_size: int,
    fs: int,
    norm: str = "l2",
    fmin: float = MIN_HZ,
    fmax: float = MAX_HZ,
) -> torch.Tensor:
    """Map per-frame magnitude response (possibly in log-f bands) -> time-domain FIR (length = target_size).
    amp: [B, T, K] real, non-negative magnitudes. If K == target_size//2+1, treated as linear spectrum.
    Otherwise, upsample via log-f triangular filterbank to M = target_size//2+1.
    Returns: [B, T, target_size] real FIR kernels. Optionally normalized per-frame.
    """
    B, T, K = amp.shape
    M = target_size // 2 + 1
    device = amp.device
    dtype = amp.dtype
    # If K == M, treat as linear spectrum
    if K == M:
        amp_lin = amp
    else:
        # K is #log-f bands, upsample to linear spectrum [B, T, M]
        fb = _make_log_tri_filterbank(M, fs, K, fmin, fmax, device, dtype)  # [M, K]
        amp_lin = torch.matmul(amp, fb.t())  # [B, T, K] x [K, M] = [B, T, M]
    # Build complex spectrum with zero imaginary part
    amp_c = torch.view_as_complex(torch.stack([amp_lin, torch.zeros_like(amp_lin)], dim=-1))  # [B, T, M]
    ir = fft.irfft(amp_c, n=target_size)  # [B, T, target_size]
    filter_size = ir.shape[-1]
    # Center the IR, window it, pad/crop to target size, then un-center
    ir = torch.roll(ir, shifts=filter_size // 2, dims=-1)
    win = torch.hann_window(filter_size, dtype=ir.dtype, device=ir.device)
    ir = ir * win
    if filter_size < target_size:
        ir = F.pad(ir, (0, int(target_size) - int(filter_size)))
    elif filter_size > target_size:
        ir = ir[..., :target_size]
    ir = torch.roll(ir, shifts=-filter_size // 2, dims=-1)
    # --- Per-frame normalization (recommended: unit energy) ---
    if norm == "l2":
        d = ir.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        ir = ir / d
    elif norm == "l1":
        d = ir.abs().sum(dim=-1, keepdim=True).clamp_min(1e-8)
        ir = ir / d
    elif norm is None:
        pass
    else:
        raise ValueError("norm must be 'l2', 'l1', or None")
    return ir

def fft_convolve(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Frame-wise FFT convolution along the last dimension.
    signal: [B, T, L], kernel: [B, T, L] -> [B, T, L]
    """
    L = signal.shape[-1]
    x = F.pad(signal, (0, L))
    h = F.pad(kernel, (L, 0))
    y = fft.irfft(fft.rfft(x) * fft.rfft(h))
    return y[..., y.shape[-1] // 2:]

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
# Jordi Shier–style Attention Pooling (query + MHA)
# =============================================================
class AttentionPooling(nn.Module):
    def __init__(self, in_features: int, keep_seq_dim: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.query = nn.Parameter(torch.zeros(1, 1, in_features))
        self.attn = nn.MultiheadAttention(in_features, 1, bias=False)
        self.keep_seq_dim = keep_seq_dim

        # Init the learned query to a small random value for symmetry breaking
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expects shape (batch, channels, time) and returns (batch, channels)
        unless keep_seq_dim=True, in which case returns (batch, channels, 1).
        """
        # [B, C, T] -> [T, B, C]
        x = rearrange(x, "b c t -> t b c")
        x = self.norm(x)
        q = repeat(self.query, "() () c -> () b c", b=x.shape[1])
        attn, _ = self.attn(q, x, x, need_weights=False)
        if self.keep_seq_dim:
            attn = rearrange(attn, "t b c -> b c t")
        else:
            attn = attn.squeeze(dim=0)  # [B, C]
        return attn







# =============================================================
# Low-rate TCN feature extractor for segment-level prediction
# =============================================================
class LowRateTCN(nn.Module):
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

        nn.init.constant_(self.loop_proj.bias[0],  2.0)   # g logit → sigmoid ≈ 0.88

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
# nnKarplusStrong with split controllers:
#  - High-rate TCN on pre-KS excitation → FiLM-conditioned burst shaping
#  - Low-rate TCN on pitch/loudness     → DiffKS loop + GEQ
# =============================================================
class nnKarplusStrong(nn.Module):
    def __init__(self,
                 batch_size,
                 loop_order,
                 internal_sr,
                 interpolation_type,
                 filter_type,
                 timesteps: int = 250,
                 n_noise_bands: int = 16,
                 hi_hop_samples: int = 64,
                 lpc_order: int = 6,
                 lo_tcn_ch: int = 64,):
        super().__init__()
        self.internal_sr = internal_sr
        self.loop_order = loop_order
        self.timesteps = timesteps
        self.n_noise_bands = n_noise_bands
        self.lpc_order = lpc_order
        self.hi_hop = int(hi_hop_samples)


        # --- Conditioning normalization (dataset-aware) ---
        # Pitch comes in Hz; clamp between E2 and E6, then log2 → [0,1]
        self.f0_min_hz = float(MIN_HZ)
        self.f0_max_hz = float(MAX_HZ)
        self.log2_f0_min = math.log2(self.f0_min_hz)
        self.log2_f0_max = math.log2(self.f0_max_hz)
        # Loudness is already z-scored / unit-ranged in [0,1]
        self.loudness_is_unit = True
        self.pitch_in_hz = True
        # (Only used if loudness is provided in dB; kept for completeness)
        self.loud_db_min = -80.0
        self.loud_db_max = 0.0

        # --- n_noise_bands attribute ---
        self.n_noise_bands = int(n_noise_bands)

        # --- Low-rate TCN for feature extraction and heads ---
        self.low_tcn = LowRateTCN(in_ch=2, ch=lo_tcn_ch, n_blocks=6, kernel=3,
                                   loop_out=loop_order + 1, n_bands=n_noise_bands)

        # Attention pooling used *between onsets* (segment-level pooling)
        self.onset_pool = AttentionPooling(in_features=lo_tcn_ch, keep_seq_dim=False)


        # Differentiable KS decoder
        self.decoder = DiffKS(
            batch_size=batch_size,
            internal_sr=internal_sr,
            loop_order=loop_order,
            loop_n_frames=timesteps,
            interp_type=interpolation_type,
            use_double_precision=False,
            min_f0_hz=20,
            loop_filter_kind=filter_type,
        )
        for p in self.decoder.parameters():
            p.requires_grad = False
    def _scale_conditioning(self, pitch_seq: torch.Tensor, loud_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Map inputs to [0,1] for the low‑rate TCN, matching your dataset:
          • pitch_seq is in Hz  → clamp [E2,E6], log2, min‑max to [0,1]
          • loud_seq already in [0,1] → clamp to [0,1]
        Inputs/outputs: [B, T, 1].
        """
        # Pitch
        if self.pitch_in_hz:
            f0_hz = pitch_seq.squeeze(-1)
            f0_hz = f0_hz.clamp(min=self.f0_min_hz, max=self.f0_max_hz)
            f0_log2 = torch.log2(f0_hz.clamp(min=1e-6))
            denom = (self.log2_f0_max - self.log2_f0_min)
            f0_scaled = (f0_log2 - self.log2_f0_min) / max(1e-12, denom)
            f0_scaled = f0_scaled.clamp(0.0, 1.0).unsqueeze(-1)
        else:
            f0_scaled = pitch_seq.clamp(0.0, 1.0)

        # Loudness
        if self.loudness_is_unit:
            L_scaled = loud_seq.clamp(0.0, 1.0)
        else:
            L = loud_seq.squeeze(-1)
            Lc = L.clamp(min=self.loud_db_min, max=self.loud_db_max)
            L_scaled = ((Lc - self.loud_db_min) / (self.loud_db_max - self.loud_db_min + 1e-12)).unsqueeze(-1)

        return f0_scaled, L_scaled

    # -----------------------------
    # Helpers
    # -----------------------------

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(self, pitch, _loudness, audio, audio_sr, return_parameters=False,
                triggers: bool = False, trigger_width_frames: int = 1):
        B, N = audio.shape
        T_frames = pitch.size(1)

        # --- Low-rate conditioning (DDX7-style scaling) ---
        pitch_scaled, loud_scaled = self._scale_conditioning(pitch, _loudness)

        # --- Low-rate TCN features at frame-rate ---
        # h: [B, C, T]
        h = self.low_tcn(pitch_scaled, loud_scaled)
        C = h.shape[1]

        # Helper: sample index -> frame index
        def _sample_to_frame_idx(s: int, N: int, T: int) -> int:
            idx = int(round(s * T / max(1, N)))
            return max(0, min(T - 1, idx))

        # --- Detect onsets (sample indices per batch) ---
        onset_list = []
        for b in range(B):
            on_b = detect_onsets_librosa(audio[b], sr=int(audio_sr))
            if on_b.size == 0:
                on_b = np.array([0], dtype=int)
            onset_list.append(on_b)

        # --- Build bursts-only excitation per batch (noise bursts per onset, length = one period) ---
        burst_rows = []
        for b in range(B):
            burst_b = torch.zeros(1, N, device=audio.device, dtype=audio.dtype)
            for s in onset_list[b].tolist():
                f_idx = _sample_to_frame_idx(s, N, T_frames)
                # Clamp pitch to at least E2 to avoid absurdly long periods
                f0_loc = float(torch.clamp(pitch[b, f_idx, 0], min=MIN_HZ).item())
                L_loc = int(round(float(audio_sr) / max(f0_loc, 1e-6)))
                nb_burst = make_onset_noise(
                    onset_samples=np.array([s], dtype=int),
                    num_samples=N,
                    sample_rate=int(audio_sr),
                    batch_size=1,
                    device=audio.device,
                    dtype=audio.dtype,
                    burst_len_samples=L_loc,
                    impulse_instead=True,
                )  # [1, N]
                # --- Scale burst amplitude linearly from frame loudness (0–1) ---
                L_val = loud_scaled[b, f_idx, 0].clamp(0.0, 1.0)
                amp = L_val.clamp_min(1e-5)
                nb_burst = nb_burst * amp

                burst_b = burst_b + nb_burst
            burst_rows.append(burst_b)
        burst_stream = torch.cat(burst_rows, dim=0)  # [B, N]

        # --- Onset-segment attention pooling over low-rate features ---
        # We create frame boundaries using detected onsets mapped to frame indices.
        loop_out = self.loop_order + 1
        K = int(self.n_noise_bands)
        # Prepare outputs [B, T, *]
        loop_logits = torch.zeros(B, T_frames, loop_out, device=audio.device, dtype=audio.dtype)
        band_gains_frames = torch.zeros(B, T_frames, K, device=audio.device, dtype=audio.dtype)

        for b in range(B):
            # frame boundaries (include 0 and T)
            frame_bounds = [0]
            for s in onset_list[b].tolist():
                fi = _sample_to_frame_idx(int(s), N, T_frames)
                if fi not in frame_bounds:
                    frame_bounds.append(fi)
            frame_bounds = sorted(set([i for i in frame_bounds if 0 <= i < T_frames]))
            if frame_bounds[0] != 0:
                frame_bounds = [0] + frame_bounds
            if frame_bounds[-1] != T_frames:
                frame_bounds.append(T_frames)

            hb = h[b:b+1]  # [1, C, T]
            for si in range(len(frame_bounds) - 1):
                s = frame_bounds[si]
                e = frame_bounds[si + 1]
                if e <= s:
                    continue
                seg = hb[:, :, s:e]  # [1, C, L]
                pooled = self.onset_pool(seg)  # [1, C]
                # Heads on pooled features → piece-wise constant outputs
                ll = self.low_tcn.loop_proj(pooled)  # [1, loop_out]
                band_logits = self.low_tcn.band_proj(pooled)  # [1, K]

                # IRCAM-like quiet init + allow up to +16 dB boost
                # Use the same scale_function and a -5 shift for a very quiet start.
                mags = scale_function(band_logits - 5.0)  # [1, K]

                loop_logits[b, s:e, :]       = ll.expand(e - s, -1)
                band_gains_frames[b, s:e, :] = mags.expand(e - s, -1)

        # --- FIR noise shaping ala IRCAM DDSP (frame-wise time-varying FIR) ---
        # Choose block size so that T_frames * block_size >= N
        block_size = int(math.ceil(N / max(1, T_frames)))
        pad_len = T_frames * block_size - N

        # Build per-frame noise that follows the onset bursts: use the original
        # burst_stream as a mask so we only excite frames that contain burst samples.
        burst_padded = F.pad(burst_stream, (0, pad_len))                      # [B, T*block]
        burst_blocks = burst_padded.view(B, T_frames, block_size)             # [B, T, block]

        # Magnitude bins per frame -> per-frame FIR impulse responses
        mags = band_gains_frames                                              # [B, T, K]
        impulse = amp_to_impulse_response(mags, block_size, fs=int(audio_sr), norm=None)  # no per-frame normalization → magnitude mapping controls loudness

        # Frame-wise convolution and fold back to a 1D excitation
        shaped_blocks = fft_convolve(burst_blocks, impulse)                   # [B, T, block]
        excitation = shaped_blocks.reshape(B, T_frames * block_size)[..., :N] # [B, N]

        # --- DiffKS decode with loop coefficients at frame-rate ---
        out = self.decoder(
            f0_frames=pitch.squeeze(2),
            input=excitation,
            input_sr=audio_sr,
            loop_coefficients=loop_logits,
        )  # [B, N]

        if return_parameters:
            return {
                "loop_logits": loop_logits.detach(),
                "band_gains_frames": band_gains_frames.detach(),
                "burst_stream": burst_stream.detach(),
                "resonator_excitation": excitation.detach(),
                "fir_impulse": impulse.detach(),
            }

        return out