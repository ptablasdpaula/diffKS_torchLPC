import math
from typing import Union
import numpy as np
import torch
import torchaudio
import librosa
import torch.nn as nn

class ShelfIIR(nn.Module):
    """RBJ shelving biquad (low/high) via torchaudio.functional.lfilter (differentiable).
    Use `which` in {"low", "high"}.
    """
    def __init__(self, sample_rate: float, which: str, device=None, dtype=None):
        super().__init__()
        assert which in ("low", "high")
        self.which = which
        self.fs = float(sample_rate)
        self._device, self._dtype = device, dtype

    def _to_tensor(self, x):
        t = x if isinstance(x, torch.Tensor) else torch.tensor(x)
        if self._dtype is not None: t = t.to(self._dtype)
        if self._device is not None: t = t.to(self._device)
        return t

    def _design(self, fc, Q, gain_db, N: int):
        def expand1(t):
            t = self._to_tensor(t)
            return t.expand(N) if t.ndim == 0 else t.reshape(N)

        fc  = expand1(fc)
        Q   = expand1(Q)
        GdB = expand1(gain_db)

        w0 = 2 * math.pi * fc / self.fs
        c, s = torch.cos(w0), torch.sin(w0)
        A = torch.pow(torch.tensor(10.0, device=w0.device, dtype=w0.dtype), GdB / 40.0)

        # Q-form (RBJ): alpha = sin(w0) / (2Q)
        alpha = s / (2.0 * (Q + 1e-12))
        beta  = 2.0 * torch.sqrt(A) * alpha

        m = (A + 1.0)
        n = (A - 1.0)

        # Sign pattern distinguishes low vs high shelf
        sgn_c  = 1.0 if self.which == "high" else -1.0
        sgn_b1 = -1.0 if self.which == "high" else 1.0
        sgn_a1 =  1.0 if self.which == "high" else -1.0

        b0 = A * ( m + sgn_c * n * c + beta)
        b1 = 2.0 * sgn_b1 * A * ( n + sgn_c * m * c)
        b2 = A * ( m + sgn_c * n * c - beta)
        a0 =      ( m - sgn_c * n * c + beta)
        a1 = 2.0 * sgn_a1 * ( n - sgn_c * m * c)
        a2 =      ( m - sgn_c * n * c - beta)

        inv_a0 = 1.0 / (a0 + 1e-12)
        b0, b1, b2 = b0 * inv_a0, b1 * inv_a0, b2 * inv_a0
        a1, a2     = a1 * inv_a0, a2 * inv_a0
        a0         = torch.ones_like(a1)

        a = torch.stack([a0, a1, a2], dim=-1)
        b = torch.stack([b0, b1, b2], dim=-1)
        return a, b

    def forward(self, x: torch.Tensor, fc, Q=1.0, gain_db=0.0):
        x = x.to(device=self._device or x.device, dtype=self._dtype or x.dtype)
        *lead, T = x.shape
        N = int(torch.tensor(lead).prod().item()) if lead else 1
        x2 = x.reshape(N, T)
        a, b = self._design(fc, Q, gain_db, N)
        # Always run unclamped to preserve differentiability
        y2 = torchaudio.functional.lfilter(x2, a_coeffs=a, b_coeffs=b, clamp=False)
        return y2.reshape(*lead, T)

class StaticShelf(nn.Module):
    """Learnable shelf with fc (Hz), Q (>0), and gain in dB (bounded)."""
    def __init__(self, which: str, sample_rate: float, init_fc_hz: float,
                 fmin_hz=20.0, fmax_hz=None, init_Q=0.707, init_gain_db=0.0,
                 device=None, dtype=None,
                 max_gain_db: float = 12.0):
        super().__init__()
        assert which in ("low", "high")
        self.which = which
        self.fs = float(sample_rate)
        self.fmin = float(fmin_hz)
        self.fmax = float(fmax_hz or (self.fs/2 - 1.0))
        self._eps = 1e-6
        self.max_gain_db = float(max_gain_db)
        self.filter = ShelfIIR(sample_rate=self.fs, which=which, device=device, dtype=dtype)

        # fc via K=tan(pi f / Fs)
        init_fc = float(min(max(init_fc_hz, self.fmin), self.fmax - 1e-3))
        K0 = math.tan(math.pi * init_fc / self.fs)
        raw_fc0 = math.log(math.expm1(max(K0, 1e-9)))
        self.raw_fc = nn.Parameter(torch.tensor(raw_fc0, device=device, dtype=dtype))

        # Quality factor Q = softplus(raw_Q)
        raw_Q0 = math.log(math.expm1(max(init_Q, 1e-6)))
        self.raw_Q = nn.Parameter(torch.tensor(raw_Q0, device=device, dtype=dtype))

        # Gain dB via tanh to bound within ±max_gain_db
        self.raw_gdb = nn.Parameter(torch.tensor(init_gain_db / max_gain_db, device=device, dtype=dtype))

    # Readouts
    def fc_hz(self):
        with torch.no_grad():
            K = torch.nn.functional.softplus(self.raw_fc) + self._eps
            fc = (self.fs / math.pi) * torch.atan(K)
            return fc.clamp(self.fmin, self.fmax - self._eps)

    def quality_Q(self):
        with torch.no_grad():
            return torch.nn.functional.softplus(self.raw_Q)

    def gain_db(self):
        with torch.no_grad():
            return self.max_gain_db * torch.tanh(self.raw_gdb)

    # Forward values
    def _fc_forward(self):
        K = torch.nn.functional.softplus(self.raw_fc) + self._eps
        return ((self.fs / math.pi) * torch.atan(K)).clamp(self.fmin, self.fmax - self._eps)

    def _Q_forward(self):
        return torch.nn.functional.softplus(self.raw_Q)

    def _gdb_forward(self):
        return self.max_gain_db * torch.tanh(self.raw_gdb)

    def forward(self, x: torch.Tensor):
        fc  = self._fc_forward()
        Q   = self._Q_forward()
        gdb = self._gdb_forward()
        return self.filter(x, fc=fc, Q=Q, gain_db=gdb)


# --------------------------------------------------------------------------
# Onset detection helper (librosa, 50 ms left pad, backtrack)
# --------------------------------------------------------------------------
def detect_onsets_librosa(x: Union[torch.Tensor, np.ndarray],
                          sr: int,
                          pad_ms: float = 50.0,
                          hop_length: int = 512,
                          backtrack: bool = True) -> np.ndarray:
    """Return onset sample indices using librosa with a temporary left pad.
    x: mono audio [N] or [1, N].
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().squeeze().numpy()
    x = x.astype(np.float32)
    pad = int(round((pad_ms / 1000.0) * sr))
    x_pad = np.pad(x, (pad, 0), mode="constant")

    # onset_detect with backtrack produces frame indices; convert to samples
    onset_frames = librosa.onset.onset_detect(y=x_pad, sr=sr,
                                             hop_length=hop_length,
                                             backtrack=backtrack,
                                             units="frames")
    onset_samples = librosa.frames_to_samples(onset_frames, hop_length=hop_length)

    # undo the left pad and keep valid onsets
    onset_samples = onset_samples - pad
    onset_samples = onset_samples[onset_samples >= 0]
    onset_samples = onset_samples[onset_samples < x.shape[-1]]
    return onset_samples.astype(int)


def make_onset_noise(onset_samples: np.ndarray,
                     num_samples: int,
                     sample_rate: int,
                     batch_size: int = 1,
                     device=None,
                     dtype=None,
                     noise_ms: float = 10.0,
                     burst_len_samples: int | None = None) -> torch.Tensor:
    """
    Create [B, N] mostly‑zero signal with uniform noise bursts in [-1, 1].
    If `burst_len_samples` is provided, it overrides `noise_ms` for burst length.
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    if burst_len_samples is not None:
        seg_len = max(1, int(burst_len_samples))
    else:
        seg_len = max(1, int(round((noise_ms / 1000.0) * sample_rate)))

    sig = torch.zeros(batch_size, num_samples, device=device, dtype=dtype)
    if onset_samples.size == 0:
        onset_samples = np.array([0], dtype=int)
    for s in onset_samples:
        start = int(s)
        if start >= num_samples:
            continue
        end = min(start + seg_len, num_samples)
        # zero‑mean uniform noise in [-0.5, 0.5]
        noise = torch.rand(batch_size, end - start, device=device, dtype=dtype) - 0.5
        sig[:, start:end] = noise
    return sig

def scale_noise_bursts_to_target_rms(noise: torch.Tensor,
                                     target: torch.Tensor,
                                     onset_samples: np.ndarray,
                                     burst_len_samples: int,
                                     eps: float = 1e-8,
                                     compensate_delay_len: bool = False) -> torch.Tensor:
    """
    For each onset window [s, s+L), scale the noise burst so its RMS matches
    the RMS of the target audio over the same window. Optionally multiply by
    1/sqrt(L) to keep energy roughly pitch‑invariant.

    noise  : [B, N]
    target : [B, N]
    onset_samples : np.ndarray of onset sample indices
    burst_len_samples : L (length of each burst)
    """
    assert noise.dim() == 2 and target.dim() == 2, "Expected [B, N] tensors"
    B, N = noise.shape
    L = int(burst_len_samples)
    if len(onset_samples) == 0:
        onset_samples = np.array([0], dtype=int)

    out = noise.clone()
    for s in onset_samples:
        start = int(max(0, s))
        end = int(min(start + L, N))
        if end <= start:
            continue
        # Compute RMS per batch over the same window
        seg_tgt = target[:, start:end]
        seg_noi = out[:, start:end]
        rms_tgt = torch.sqrt(torch.clamp((seg_tgt ** 2).mean(dim=-1, keepdim=True), min=eps))  # [B, 1]
        rms_noi = torch.sqrt(torch.clamp((seg_noi ** 2).mean(dim=-1, keepdim=True), min=eps))  # [B, 1]
        gain = rms_tgt / (rms_noi + eps)                                                       # [B, 1]
        if compensate_delay_len and L > 0:
            gain = gain * (1.0 / math.sqrt(L))
        out[:, start:end] = seg_noi * gain
    return out