import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import numpy as np
from diffKS import DiffKS
from core import make_onset_noise, detect_onsets_librosa
import math
from torchlpc import sample_wise_lpc
from flamo.auxiliary.eq import eq_freqs, geq as geq_sos

# Additional imports for Jordi-style attention pooling
from einops import rearrange, repeat

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
# SoundStream-style waveform encoder (no FiLM, no quantization)
# Based on Shier et al.'s DrumBlender encoder blocks
# =============================================================
class Pad(nn.Module):
    def __init__(self, kernel_size: int, dilation: int, causal: bool = False):
        super().__init__()
        self.pad = int(dilation) * (int(kernel_size) - 1)
        self.causal = bool(causal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            return F.pad(x, (self.pad, 0))
        left = self.pad // 2
        right = self.pad - left
        return F.pad(x, (left, right))

class _SSResidualUnit(nn.Module):
    def __init__(self, width: int, dilation: int, kernel_size: int = 7, causal: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            Pad(kernel_size, dilation, causal=causal),
            nn.Conv1d(width, width, kernel_size, dilation=dilation, padding=0),
            nn.ELU(),
            nn.Conv1d(width, width, 1),
        )
        self.final_act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return x + self.final_act(y)

class _SSEncoderBlock(nn.Module):
    def __init__(self, width: int, stride: int, kernel_size: int = 7, causal: bool = False):
        super().__init__()
        # Three residual units with dilations 1,3,9 operating at width//2 channels
        self.units = nn.ModuleList([
            _SSResidualUnit(width // 2, 1, kernel_size, causal=causal),
            _SSResidualUnit(width // 2, 3, kernel_size, causal=causal),
            _SSResidualUnit(width // 2, 9, kernel_size, causal=causal),
        ])
        # Strided conv to double channels and downsample time by `stride`
        self.out = nn.Sequential(
            Pad(2 * stride, 1, causal=causal),
            nn.Conv1d(width // 2, width, 2 * stride, stride=stride, padding=0),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for u in self.units:
            x = u(x)
        return self.out(x)

class SoundStreamEncoder(nn.Module):
    """Convolutional waveform encoder (no RVQ).
    Strides default to (2,2,4,4) → overall downsample x64.
    """
    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 32,
        output_channels: int = 64,
        kernel_size: int = 7,
        strides: tuple[int, ...] = (2, 2, 4, 4),
        causal: bool = False,
    ):
        super().__init__()
        self.input = nn.Sequential(
            Pad(kernel_size, 1, causal=causal),
            nn.Conv1d(input_channels, hidden_channels, kernel_size, padding=0),
            nn.ELU(),
        )
        enc_blocks = []
        h = hidden_channels
        for s in strides:
            h = h * 2
            enc_blocks.append(_SSEncoderBlock(h, s, kernel_size=kernel_size, causal=causal))
        self.blocks = nn.ModuleList(enc_blocks)
        self.output = nn.Sequential(
            Pad(3, 1, causal=causal),
            nn.Conv1d(h, output_channels, 3, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C=1, N] → features: [B, output_channels, N/∏strides]
        y = self.input(x)
        for blk in self.blocks:
            y = blk(y)
        return self.output(y)


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
# FiLM and TemporalFiLM1x1
# =============================================================

class FiLM(nn.Module):
    """Feature-wise Linear Modulation (Jordi-style).
    Takes a clip-level embedding and returns gamma/beta for an activation x.
    If used with batch-norm, it normalizes x (affine=False) before applying FiLM.
    """
    def __init__(self, film_embedding_size: int, input_channels: int, use_batch_norm: bool = True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.norm = nn.BatchNorm1d(input_channels, affine=False)
        self.net = nn.Linear(film_embedding_size, input_channels * 2)

    def forward(self, x: torch.Tensor, film_embedding: torch.Tensor):
        film = self.net(film_embedding)
        gamma, beta = film.chunk(2, dim=-1)
        if self.use_batch_norm:
            x = self.norm(x)
        return gamma[..., None] * x + beta[..., None]


# Jordi Shier's DrumBlender-style Temporal Feature-wise Linear Modulation (TFiLM)
class TFiLM(nn.Module):
    """Temporal Feature-wise Linear Modulation layer. Derives affine parameters from a
    decimated version of the input signal, and applies them to the input. Allows the
    model to learn longer-range temporal dependencies.
    """

    def __init__(self, channels: int, block_size: int):
        super().__init__()
        self.block_size = block_size

        self.pool = nn.MaxPool1d(block_size)
        self.block_size = block_size

        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=channels,
            num_layers=1,
        )
        self.proj = nn.Linear(channels, channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *_, length = x.shape
        n_blocks = length // self.block_size
        assert n_blocks > 0, "Input length must be greater than block size."
        assert (
            length == n_blocks * self.block_size
        ), "Input length must be divisible by block size."

        x_decimated = self.pool(x)
        x_decimated = rearrange(x_decimated, "b c t -> t b c")

        affine, _ = self.lstm(x_decimated)
        affine = self.proj(affine)
        affine = rearrange(affine, "t b c -> b c t 1")
        gamma, beta = affine.chunk(2, dim=1)

        x = rearrange(x, "b c (n k) -> b c n k", k=self.block_size)
        x = gamma * x + beta
        x = rearrange(x, "b c n k -> b c (n k)")

        return x




# =============================================================
# Jordi-style GatedActivation, TCN, and SoundStreamAttentionEncoder
# =============================================================

class GatedActivation(nn.Module):
    """Gated activation function for 1D convolutional networks. Expects input of shape
    (batch_size, channels * 2, time).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-2)
        assert x1.shape[-2] == x2.shape[-2], "Input channels must be divisible by 2."
        return torch.tanh(x1) * torch.sigmoid(x2)


class _DilatedResidualBlock(nn.Module):
    """Temporal convolutional network internal block (Jordi Shier, DrumBlender style).
    Includes optional FiLM and optional TFiLM hooks (we keep FiLM only here).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        causal: bool = True,
        norm: str | None = None,
        activation: str = "GELU",
        film_conditioning: bool = False,
        film_embedding_size: int | None = None,
        film_batch_norm: bool = True,
        use_temporal_film: bool = True,
        temporal_film_block_size: int | None = 256,
    ):
        super().__init__()

        if film_conditioning and (film_embedding_size is None or film_embedding_size < 1):
            raise ValueError("FiLM conditioning requires a valid embedding size (int >= 1).")

        net = []
        pre_activation_channels = out_channels * 2 if activation == "gated" else out_channels

        if norm is not None:
            if norm not in ("batch", "instance"):
                raise ValueError("Invalid norm type (must be batch or instance)")
            _Norm = nn.BatchNorm1d if norm == "batch" else nn.InstanceNorm1d
            net.append(_Norm(in_channels))

        net.extend([
            Pad(kernel_size, dilation, causal=causal),
            nn.Conv1d(
                in_channels,
                pre_activation_channels,
                kernel_size,
                dilation=dilation,
                padding=0,
            ),
        ])
        self.net = nn.Sequential(*net)

        self.film = FiLM(film_embedding_size, pre_activation_channels, film_batch_norm) if film_conditioning else None
        self.activation = GatedActivation() if activation == "gated" else getattr(nn, activation)()
        self.residual = nn.Conv1d(in_channels, out_channels, 1)
        self.tfilm = None
        if use_temporal_film:
            if temporal_film_block_size is None or temporal_film_block_size < 1:
                raise ValueError("TFiLM requires a valid block size (int >= 1).")
            self.tfilm = TFiLM(out_channels, temporal_film_block_size)

    def forward(self, x: torch.Tensor, film_embedding: torch.Tensor | None = None):
        activations = self.net(x)
        if self.film is not None:
            activations = self.film(activations, film_embedding)
        y = self.activation(activations)
        if self.tfilm is not None:
            y = self.tfilm(y)
        return y + self.residual(x)


class TCN(nn.Module):
    """Jordi-style Temporal Convolutional Network with (optional) FiLM conditioning.
    Operates at audio rate on 1D sequences.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dilation_base: int = 2,
        num_layers: int = 8,
        kernel_size: int = 3,
        causal: bool = True,
        norm: str | None = None,
        activation: str = "GELU",
        film_conditioning: bool = False,
        film_embedding_size: int | None = None,
        film_batch_norm: bool = True,
        use_temporal_film: bool = True,
        temporal_film_block_size: int | None = 256,
    ):
        super().__init__()
        self.in_projection = nn.Conv1d(in_channels, hidden_channels, 1)
        self.out_projection = nn.Conv1d(hidden_channels, out_channels, 1)

        net = []
        for n in range(num_layers):
            dilation = dilation_base ** n
            net.append(
                _DilatedResidualBlock(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    dilation,
                    causal=causal,
                    norm=norm,
                    activation=activation,
                    film_conditioning=film_conditioning,
                    film_embedding_size=film_embedding_size,
                    film_batch_norm=film_batch_norm,
                    use_temporal_film=use_temporal_film,
                    temporal_film_block_size=temporal_film_block_size,
                )
            )
        self.net = nn.ModuleList(net)

        # Xavier init for stability
        nn.init.xavier_uniform_(self.in_projection.weight)
        nn.init.zeros_(self.in_projection.bias)
        nn.init.xavier_uniform_(self.out_projection.weight)
        nn.init.zeros_(self.out_projection.bias)

    def forward(self, x: torch.Tensor, film_embedding: torch.Tensor | None = None):
        x = self.in_projection(x)
        for layer in self.net:
            x = layer(x, film_embedding)
        x = self.out_projection(x)
        return x


class SoundStreamAttentionEncoder(nn.Module):
    """SoundStream encoder with attention pooling to produce a clip-level embedding
    (as in DrumBlender)."""
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        kernel_size: int = 7,
        strides: tuple[int, ...] = (2, 2, 4, 4),
        causal: bool = False,
    ):
        super().__init__()
        self.encoder = SoundStreamEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            strides=strides,
            causal=causal,
        )
        self.pool = AttentionPooling(output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N] -> emb: [B, E]
        feats = self.encoder(x)          # [B, E, T]
        emb  = self.pool(feats)          # [B, E]
        return emb

# =============================================================
# Low-rate controller TCN (runs at frame-rate over pitch/loudness)
# Emits DiffKS loop logits per frame and clip-global GEQ gains
# =============================================================
class LowRateTCN(nn.Module):
    def __init__(self, in_ch=2, ch=64, n_blocks=6, kernel=3, loop_out=3, n_geq=32):
        super().__init__()
        self.inp = Conv1dSame(in_ch, ch, kernel, dilation=1)
        blocks = []
        for i in range(n_blocks):
            blocks.append(DilatedResBlock(ch, kernel_size=kernel, dilation=2 ** i))
        self.tcn = nn.Sequential(*blocks)
        self.post = Conv1dSame(ch, ch, 1)
        self.loop_head = nn.Conv1d(ch, loop_out, 1)
        # Safe initialization: start with zero logits → mid‑range g/p in current mapping
        nn.init.zeros_(self.loop_head.weight)
        if self.loop_head.bias is not None:
            nn.init.zeros_(self.loop_head.bias)
        self.attn_pool = AttentionPooling(in_features=ch, keep_seq_dim=False)
        self.geq_proj = nn.Linear(ch, n_geq)
        # Neutral GEQ at start (0 dB for all bands)
        nn.init.zeros_(self.geq_proj.weight)
        if self.geq_proj.bias is not None:
            nn.init.zeros_(self.geq_proj.bias)

    def forward(self, pitch_seq, loud_seq):
        """
        pitch_seq, loud_seq: [B, T, 1]
        Returns:
          loop_logits: [B, T, loop_out]
          geq_logits:  [B, n_geq]
        """
        x = torch.cat([pitch_seq, loud_seq], dim=-1).transpose(1, 2)  # [B, 2, T]
        h = self.inp(x)
        h = self.tcn(h)
        h = self.post(h)                     # [B, C, T]
        loop = self.loop_head(h).transpose(1, 2)  # [B, T, loop_out]
        h_pool = self.attn_pool(h)  # [B, C]
        geq = self.geq_proj(h_pool)
        return loop, geq

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
                 n_noise_bands: int = 64,
                 hidden_size: int = 256,
                 hi_hop_samples: int = 64,
                 lpc_order: int = 6,
                 lo_tcn_ch: int = 64,
                 hi_tcn_ch: int = 32,
                 tfilm_block_size: int = 256):
        super().__init__()
        self.internal_sr = internal_sr
        self.loop_order = loop_order
        self.timesteps = timesteps
        self.n_noise_bands = n_noise_bands
        self.lpc_order = lpc_order
        self.hi_hop = int(hi_hop_samples)
        self.tfilm_block_size = int(tfilm_block_size)

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

        # --- GEQ layout (one‑third‑octave bands), applied POST-KS ---
        self.sample_rate = internal_sr
        cf, sh = eq_freqs(interval=3)
        nyq = self.sample_rate * 0.5
        cf = torch.as_tensor(cf, dtype=torch.float32)
        cf = cf[cf <= (nyq * 0.98)]
        self.geq_centers = cf
        sh = torch.as_tensor(sh, dtype=torch.float32)
        sh = torch.stack([torch.clamp(sh[0], min=20.0), torch.clamp(sh[1], max=nyq * 0.98)])
        self.geq_shelves = sh
        self.n_geq = int(self.geq_centers.numel() + 3)
        self.max_gain_db = 12.0

        # Controller: SoundStream + AttentionPooling -> clip embedding for FiLM
        film_size = hi_tcn_ch  # FiLM embedding size
        self.ss_controller = SoundStreamAttentionEncoder(
            input_channels=1,            # controller sees target audio only (Jordi-style)
            hidden_channels=32,
            output_channels=film_size,
            kernel_size=7,
            strides=(2, 2, 4, 4),
            causal=False,
        )

        # Burst shaper: audio-rate TCN on the burst stream, FiLM-conditioned by the controller embedding
        self.burst_tcn = TCN(
            in_channels=1,
            hidden_channels=hi_tcn_ch,
            out_channels=1,
            dilation_base=2,
            num_layers=8,
            kernel_size=3,
            causal=True,
            norm=None,
            activation="GELU",
            film_conditioning=True,
            film_embedding_size=film_size,
            film_batch_norm=True,
            use_temporal_film=True,
            temporal_film_block_size=self.tfilm_block_size,
        )

        self.low_tcn = LowRateTCN(in_ch=2, ch=lo_tcn_ch, n_blocks=6, kernel=3,
                                   loop_out=loop_order + 1, n_geq=self.n_geq)

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
    def _rc_to_lpc(self, rc: torch.Tensor) -> torch.Tensor:
        """Vectorised Levinson step-up: RC (PARCOR) → LPC 'a' coefficients.
        rc: [B, T, P]  → a: [B, T, P]
        """
        Bf, Tf, P = rc.shape
        a = torch.zeros(Bf, Tf, P, device=rc.device, dtype=rc.dtype)
        for m in range(P):
            km = rc[..., m]
            if m == 0:
                a[..., 0] = km
            else:
                prev = a[..., :m].clone()
                rev = torch.flip(prev, dims=[-1])
                a[..., :m] = prev + km.unsqueeze(-1) * rev
                a[..., m] = km
        return a

    def _apply_geq_fd(self, x: torch.Tensor, gains_db: torch.Tensor, sr: int) -> torch.Tensor:
        """Apply FLAMO GEQ as a cascade of SOS in the frequency domain.
        x: [B, N] time‑domain mono
        gains_db: [B, K] where K = 2 + len(self.geq_centers)
        Returns: [B, N]
        """
        B, N = x.shape
        nfft = 1
        while nfft < N:
            nfft <<= 1
        w = 2.0 * math.pi * torch.arange(0, nfft // 2 + 1, device=x.device, dtype=x.dtype) / float(nfft)
        z1 = torch.exp(-1j * w)
        z2 = torch.exp(-2j * w)

        cf = self.geq_centers.to(device=x.device, dtype=x.dtype)
        sh = self.geq_shelves.to(device=x.device, dtype=x.dtype)
        R = torch.tensor(2.7, device=x.device, dtype=x.dtype)

        y_list = []
        K_needed = int(cf.numel() + 3)

        for b in range(B):
            gdb = gains_db[b]
            if gdb.numel() != K_needed:
                if gdb.numel() > K_needed:
                    gdb = gdb[:K_needed]
                else:
                    pad = torch.zeros(K_needed - gdb.numel(), device=gdb.device, dtype=gdb.dtype)
                    gdb = torch.cat([gdb, pad], dim=0)

            b_sos, a_sos = geq_sos(
                center_freq=cf,
                shelving_freq=sh,
                R=R,
                gain_db=gdb,
                fs=float(sr),
                device=x.device,
            )

            b0, b1, b2 = b_sos[0], b_sos[1], b_sos[2]
            a0, a1, a2 = a_sos[0], a_sos[1], a_sos[2]
            a0_safe = a0 + 1e-12
            b0, b1, b2 = b0 / a0_safe, b1 / a0_safe, b2 / a0_safe
            a1, a2 = a1 / a0_safe, a2 / a0_safe

            num = b0.view(-1, 1) + b1.view(-1, 1) * z1 + b2.view(-1, 1) * z2
            den = 1.0 + a1.view(-1, 1) * z1 + a2.view(-1, 1) * z2
            H_sections = num / (den + 1e-30)
            H = torch.prod(H_sections, dim=0)

            X = torch.fft.rfft(x[b], n=nfft)
            H = H.to(dtype=X.dtype)
            Y = X * H
            y_time = torch.fft.irfft(Y, n=nfft).real[:N]
            y_list.append(y_time.to(dtype=x.dtype))

        return torch.stack(y_list, dim=0)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(self, pitch, _loudness, audio, audio_sr, return_parameters=False,
                triggers: bool = False, trigger_width_frames: int = 1):
        B, N = audio.shape
        T_frames = pitch.size(1)

        # --- Low-rate controller on pitch/loudness (frame-rate), with DDX7-style normalization ---
        pitch_scaled, loud_scaled = self._scale_conditioning(pitch, _loudness)
        loop_logits, geq_logits = self.low_tcn(pitch_scaled, loud_scaled)  # [B,T,loop+1], [B,K]
        gains_db = self.max_gain_db * torch.tanh(geq_logits)

        # --- Onset-triggered noise burst excitation (reference logic) ---
        # Map sample index -> frame index helper
        def _sample_to_frame_idx(s: int, N: int, T: int) -> int:
            idx = int(round(s * T / max(1, N)))
            return max(0, min(T - 1, idx))

        # Detect onsets per batch (librosa-based helper), with fallback
        onset_list = []
        for b in range(B):
            on_b = detect_onsets_librosa(audio[b], sr=int(audio_sr))
            if on_b.size == 0:
                on_b = np.array([0], dtype=int)
            onset_list.append(on_b)

        # Build bursts-only excitation per batch (noise bursts per onset, length = one period)
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
                    impulse_instead=False,
                )  # [1, N]
                burst_b = burst_b + nb_burst
            burst_rows.append(burst_b)
        burst_stream = torch.cat(burst_rows, dim=0)  # [B, N]

        # Controller embedding from target audio (clip-level)
        film_emb = self.ss_controller(audio.unsqueeze(1))  # [B, E]

        # Audio-rate FiLM-conditioned TCN directly shapes the burst stream
        # Ensure length divisible by TFiLM block size by right-padding zeros
        bs = getattr(self, "tfilm_block_size", 256)
        burst_in = burst_stream.unsqueeze(1)  # [B,1,N]
        if bs is not None and bs > 1:
            pad = (bs - (N % bs)) % bs
        else:
            pad = 0
        if pad:
            burst_in = F.pad(burst_in, (0, pad))
        shaped = self.burst_tcn(burst_in, film_emb)  # [B,1,N+pad]
        excitation = shaped.squeeze(1)[..., :N]      # trim back to original length

        # --- Synthesize with DiffKS (low-rate loop logits) ---
        out = self.decoder(
            f0_frames=pitch.squeeze(2),
            input=excitation,
            input_sr=audio_sr,
            loop_coefficients=loop_logits,
        )  # [B, N]

        # --- Clip-global GEQ (body) applied POST-KS ---
        out_eq = self._apply_geq_fd(out, gains_db=gains_db, sr=int(audio_sr))

        if return_parameters:
            return {
                "loop_logits": loop_logits,
                "burst_stream": burst_stream.detach(),
                "film_embedding": film_emb.detach(),
                "decoder_out_pre_geq": out.detach(),
                "decoder_out_post_geq": out_eq.detach(),
                "geq_info": {
                    "centers_hz": self.geq_centers.detach().cpu(),
                    "shelves_hz": self.geq_shelves.detach().cpu(),
                    "gains_db": gains_db.detach().cpu(),
                },
            }

        return out_eq