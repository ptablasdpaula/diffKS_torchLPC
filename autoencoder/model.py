import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffKS import DiffKS
from core import make_onset_noise, detect_onsets_librosa
import math
from torchlpc import sample_wise_lpc

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

# ==============================
# Time-domain GEQ utilities (1/3-oct style)
# ==============================

def scale_function_ircam(x: torch.Tensor) -> torch.Tensor:
    """IRCAM-DDSP style amplitude scaling.
    Maps logits -> linear amplitude in ~[~0, 2] (≈ +6 dB max) with a very quiet floor.
    """
    return 2.0 * torch.sigmoid(x) ** (math.log(10.0)) + 1e-7


def _logspace_centers(n_bands: int, fs: int, fmin: float = MIN_HZ, fmax: float = MAX_HZ, device=None, dtype=None) -> torch.Tensor:
    """Return n_bands log-spaced center frequencies between fmin and min(fmax, 0.98*Nyquist)."""
    nyq = fs / 2.0
    fmax_eff = min(float(fmax), float(nyq * 0.98))
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    start = math.log10(float(fmin))
    end = math.log10(float(fmax_eff))
    exps = torch.linspace(start, end, steps=int(n_bands), device=device, dtype=dtype)
    return torch.pow(10.0, exps)


def _design_biquad_peaking(f0: torch.Tensor, Q: float, gain_db: torch.Tensor, fs: int) -> torch.Tensor:
    """Design biquad peaking filters per-band using RBJ Audio EQ Cookbook.
    Args:
      f0: [K] center freqs in Hz
      Q: scalar Q-factor
      gain_db: [K] boost in dB (>=0)
      fs: sampling rate
    Returns: SOS coeffs [K, 6] as (b0, b1, b2, a0, a1, a2) with a0 normalized to 1.
    """
    device = f0.device
    dtype = f0.dtype
    w0 = 2.0 * math.pi * f0 / float(fs)
    A = torch.pow(torch.tensor(10.0, device=device, dtype=dtype), gain_db / 40.0)
    alpha = torch.sin(w0) / (2.0 * Q)
    cosw0 = torch.cos(w0)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cosw0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cosw0
    a2 = 1.0 - alpha / A
    # Normalize by a0
    b0 = b0 / a0
    b1 = b1 / a0
    b2 = b2 / a0
    a1 = a1 / a0
    a2 = a2 / a0
    a0 = torch.ones_like(b0)
    sos = torch.stack([b0, b1, b2, a0, a1, a2], dim=-1)
    return sos


def _design_geq_sos(n_bands: int, fs: int, fmin: float, fmax: float, gains_db: torch.Tensor) -> torch.Tensor:
    """Design a cascade of peaking biquads approximating a 1/3-oct GEQ.
    gains_db: [K] non-negative band boosts in dB.
    Returns: SOS [K,6].
    """
    centers = _logspace_centers(n_bands, fs, fmin=fmin, fmax=fmax, device=gains_db.device, dtype=gains_db.dtype)
    # 1/3-octave equivalent Q ≈ 1 / (2^(1/6) - 2^(-1/6)) ~ 4.318
    Q_const = 1.0 / (2.0 ** (1.0 / 6.0) - 2.0 ** (-1.0 / 6.0))
    return _design_biquad_peaking(centers, Q_const, gains_db, fs)


def _apply_sos_cascade(x: torch.Tensor, sos: torch.Tensor) -> torch.Tensor:
    """Apply a cascade of biquad sections in time domain using TorchLPC for the AR part.
    x: [B, L]
    sos: [K, 6] with (b0,b1,b2,a0(=1),a1,a2).
    Returns: [B, L]
    NOTE: Stateless per call; each band starts with zero state (piece‑wise per segment).
    """
    if sos.numel() == 0:
        return x
    y = x
    B, L = y.shape
    # Ensure contiguous for conv/LPC
    y = y.contiguous()
    for k in range(sos.shape[0]):
        b0, b1, b2, a0, a1, a2 = sos[k]
        # Normalize so a0 == 1 (safety in case of numeric drift)
        if torch.abs(a0 - 1.0) > 1e-12:
            b0 = b0 / a0
            b1 = b1 / a0
            b2 = b2 / a0
            a1 = a1 / a0
            a2 = a2 / a0
        # --- FIR numerator (b0 + b1 z^-1 + b2 z^-2) via causal conv1d ---
        # conv1d computes correlation, so use flipped kernel and left padding for causality
        kernel = torch.stack([b0, b1, b2], dim=0).to(dtype=y.dtype, device=y.device)
        kernel = kernel.flip(0).view(1, 1, 3)
        v = F.conv1d(F.pad(y.unsqueeze(1), (2, 0)), kernel).squeeze(1)  # [B, L]
        # --- AR denominator using TorchLPC (all‑pole): y = v - a1*y_{-1} - a2*y_{-2}
        A = torch.stack([a1, a2], dim=-1).to(dtype=y.dtype, device=y.device)
        # Time‑invariant across the segment, expand to [B, L, 2]
        A = A.view(1, 1, 2).expand(B, L, 2).contiguous()
        y = sample_wise_lpc(v, A)  # [B, L]
    return y


# ==============================
# Batched biquad cascade (vectorized across batch)
# ==============================
def _apply_sos_cascade_batched(x: torch.Tensor, sos: torch.Tensor) -> torch.Tensor:
    """Vectorized biquad cascade over the whole batch using TorchLPC for AR.
    Args:
      x   : [B, L] time-domain input (padded if needed)
      sos : [B, K, 6] biquad coeffs per-item, per-band as (b0,b1,b2,a0(=1),a1,a2).
    Returns:
      y   : [B, L] output after cascading K biquads (sequential across K, vectorized across B).
    Notes:
      • Stateless per call; resets states at segment boundaries.
      • We normalize by a0 defensively in case of small drift.
    """
    if sos.numel() == 0:
        return x
    B, L = x.shape
    K = sos.shape[1]
    y = x.contiguous()
    for k in range(K):
        # Coeffs per batch item for band k
        b0 = sos[:, k, 0]
        b1 = sos[:, k, 1]
        b2 = sos[:, k, 2]
        a0 = sos[:, k, 3]
        a1 = sos[:, k, 4]
        a2 = sos[:, k, 5]
        # Normalize (safety); a0 is expected to be 1 already
        eps = 1e-12
        scale = torch.where(torch.abs(a0) > eps, a0, torch.ones_like(a0))
        b0 = b0 / scale
        b1 = b1 / scale
        b2 = b2 / scale
        a1 = a1 / scale
        a2 = a2 / scale
        # --- FIR numerator via grouped conv1d (one filter per batch item) ---
        # Treat batch as channels so each item uses its own kernel; groups=B
        y_in = y.view(1, B, L)
        kernels = torch.stack([b0, b1, b2], dim=1)  # [B, 3]
        kernels = kernels.flip(1).view(B, 1, 3)     # [B, 1, 3]
        v = F.conv1d(F.pad(y_in, (2, 0)), kernels, groups=B).squeeze(0)  # [B, L]
        # --- AR denominator via TorchLPC ---
        A = torch.stack([a1, a2], dim=-1).to(dtype=v.dtype, device=v.device)  # [B, 2]
        A = A.view(B, 1, 2).expand(B, L, 2).contiguous()                      # [B, L, 2]
        y = sample_wise_lpc(v, A)  # [B, L]
    return y

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
        self.max_geq_db = 6.0  # GEQ is boost-only up to +6 dB

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
                    impulse_instead=False,
                )  # [1, N]
                # --- Scale burst amplitude linearly from frame loudness (0–1) ---
                L_val = loud_scaled[b, f_idx, 0].clamp(0.0, 1.0)
                amp = L_val.clamp_min(1e-5)
                nb_burst = nb_burst * amp

                burst_b = burst_b + nb_burst
            burst_rows.append(burst_b)
        burst_stream = torch.cat(burst_rows, dim=0)  # [B, N]

        # --- Onset-segment attention pooling over low-rate features ---
        loop_out = self.loop_order + 1
        K = int(self.n_noise_bands)
        loop_logits = torch.zeros(B, T_frames, loop_out, device=audio.device, dtype=audio.dtype)
        geq_db_frames = torch.zeros(B, T_frames, K, device=audio.device, dtype=audio.dtype)
        gain_frames = torch.zeros(B, T_frames, 1, device=audio.device, dtype=audio.dtype)

        # Build the shaped excitation directly in time domain (piece-wise per onset)
        excitation = torch.zeros(B, N, device=audio.device, dtype=audio.dtype)

        # Helper to map sample idx -> frame idx
        def _samp2frame(s_samp: int) -> int:
            return _sample_to_frame_idx(int(s_samp), N, T_frames)

        # --- Prepare per-batch segment boundaries ---
        seg_lists = []
        max_segs = 0
        for b in range(B):
            onset_samples = onset_list[b]
            if onset_samples.size == 0:
                onset_samples = np.array([0], dtype=int)
            seg_starts = onset_samples.tolist()
            if 0 not in seg_starts:
                seg_starts = [0] + seg_starts
            seg_starts = sorted(set([s for s in seg_starts if 0 <= s < N]))
            seg_ends = seg_starts[1:] + [N]
            seg_lists.append((seg_starts, seg_ends))
            max_segs = max(max_segs, len(seg_starts))

        # --- Vectorized over batch per segment index (only to write loop logits) ---
        for seg_idx in range(max_segs):
            for b in range(B):
                seg_starts, seg_ends = seg_lists[b]
                if seg_idx >= len(seg_starts):
                    continue
                s_samp = seg_starts[seg_idx]
                e_samp = seg_ends[seg_idx]
                fs_idx = _samp2frame(s_samp)
                fe_idx = _samp2frame(e_samp - 1) + 1
                fe_idx = max(fs_idx + 1, min(T_frames, fe_idx))
                seg_h = h[b:b+1, :, fs_idx:fe_idx]  # [1, C, Lf]
                pooled = self.onset_pool(seg_h)     # [1, C]
                ll = self.low_tcn.loop_proj(pooled) # [1, loop_out]
                # Fill loop logits for these frames (GEQ/gain handled globally below)
                loop_logits[b, fs_idx:fe_idx, :] = ll.expand(fe_idx - fs_idx, -1)

        # === Global (continuous) GEQ & gain per batch item ===
        # Pool across the full sequence once per batch for GEQ/gain (continuous filter; no per-onset resets)
        pooled_all = self.onset_pool(h)  # [B, C]
        geq_logits_all = self.low_tcn.band_proj(pooled_all)  # [B, K]
        gain_logit_all = self.low_tcn.gain_proj(pooled_all)  # [B, 1]

        # Boost-only GEQ in dB (0..+6 dB), initialized quiet via -5 bias
        geq_db = self.max_geq_db * torch.sigmoid(geq_logits_all - 5.0)  # [B, K]
        # IRCAM-like scalar gain (very quiet .. +6 dB), initialized ~-60 dB via gain_proj bias
        gain_lin = scale_function_ircam(gain_logit_all - 5.0)           # [B, 1]

        # For logging: broadcast per-batch constants across all frames
        geq_db_frames[:] = geq_db.unsqueeze(1).expand(B, T_frames, K)
        gain_frames[:]   = gain_lin.unsqueeze(1).expand(B, T_frames, 1)

        # Design SOS per batch item (time-invariant across the clip)
        sos_list = []
        for b in range(B):
            sos_b = _design_geq_sos(K, int(audio_sr), fmin=MIN_HZ, fmax=MAX_HZ, gains_db=geq_db[b])  # [K,6]
            sos_list.append(sos_b)
        sos_batch = torch.stack(sos_list, dim=0)  # [B, K, 6]

        # Apply one cascade per band over the **entire batch sequence** (continuous filtering)
        x_full = burst_stream * gain_lin  # [B, N]
        excitation = _apply_sos_cascade_batched(x_full, sos_batch)  # [B, N]

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
                "geq_db_frames": geq_db_frames.detach(),  # per-frame GEQ boosts (dB)
                "gain_frames": gain_frames.detach(),      # per-frame scalar gain (linear)
                "burst_stream": burst_stream.detach(),
                "resonator_excitation": excitation.detach(),
            }

        return out