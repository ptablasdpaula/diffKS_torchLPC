import math
from typing import Union
import numpy as np
import torch
import torchaudio
import librosa
import torch.nn as nn
import torch.nn.functional as F

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

class DualShelfController(nn.Module):
    """Maps AE `shelf_raw` → physically valid shelf parameters with f_high > f_low.

    Input `shelf_raw` is expected as [..., 6] with the layout
        [low_fc_raw, low_Q_raw, low_g_raw, high_fc_raw, high_Q_raw, high_g_raw].

    We convert the two raw fc logits into a *log-frequency center* `c` and a
    *positive distance* `d` using a smooth mapping:
        center_raw = 0.5 * (l_fc_raw + h_fc_raw)
        dist_raw   = (h_fc_raw - l_fc_raw)
        c = L + (U-L) * sigmoid(center_raw)
        d_max = softmin(c-L, U-c) - eps
        d = d_max * sigmoid(dist_raw)
        f_low  = exp(c - d)
        f_high = exp(c + d)

    Q is mapped by softplus; Gains are constrained to be non-positive via -softplus(raw).
    """
    def __init__(self,
                 fs: float,
                 fmin: float = 20.0,
                 fmax: float | None = None,
                 max_gain_db: float = 12.0,
                 tau_softmin: float = 8.0,
                 eps: float = 1e-6):
        super().__init__()
        self.fs = float(fs)
        self.fmin = float(fmin)
        self.fmax = float(self.fs/2 - 200.0 if fmax is None else fmax)
        self.max_gain_db = float(max_gain_db)
        self._L = math.log(self.fmin)
        self._U = math.log(self.fmax)
        self._range = self._U - self._L
        self._tau = float(tau_softmin)
        self._eps = float(eps)

    def _softmin(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # smooth min(a,b) with temperature 1/τ
        stacked = torch.stack([-self._tau * a, -self._tau * b], dim=0)
        return -torch.logsumexp(stacked, dim=0) / self._tau

    def forward(self, shelf_raw: torch.Tensor):
        """Return ((f_low, Q_low, g_low_db), (f_high, Q_high, g_high_db)).

        `shelf_raw`: tensor of shape [..., 6] with layout
            [l_fc_raw, l_Q_raw, l_g_raw, h_fc_raw, h_Q_raw, h_g_raw].
        """
        assert shelf_raw.shape[-1] == 6, f"expected last dim 6, got {shelf_raw.shape}"
        l_fc_r, l_Q_r, l_g_r, h_fc_r, h_Q_r, h_g_r = torch.unbind(shelf_raw, dim=-1)

        # Center+distance in log-frequency with smooth box constraint
        center_raw = 0.5 * (l_fc_r + h_fc_r)
        dist_raw   = (h_fc_r - l_fc_r)

        c = self._L + self._range * torch.sigmoid(center_raw)
        d_max = self._softmin(c - self._L, self._U - c) - self._eps
        d = d_max * torch.sigmoid(dist_raw)

        log_f_low  = c - d
        log_f_high = c + d
        f_low  = torch.exp(log_f_low)
        f_high = torch.exp(log_f_high)

        # Map Q (>0) and gains (±max_gain_db)
        Q_low = F.softplus(l_Q_r)
        Q_high = F.softplus(h_Q_r)
        # Non-positive dB gains (≤ 0), smooth and unclamped:
        g_low_db = -F.softplus(l_g_r)
        g_high_db = -F.softplus(h_g_r)

        return (f_low, Q_low, g_low_db), (f_high, Q_high, g_high_db)

    @torch.no_grad()
    def as_dict(self, shelf_raw: torch.Tensor):
        (fl, Ql, gl), (fh, Qh, gh) = self.forward(shelf_raw)
        get0 = lambda t: float(torch.as_tensor(t).reshape(-1)[0].item())
        return {
            "low_fc_hz": get0(fl),
            "low_Q": get0(Ql),
            "low_gain_db": get0(gl),
            "high_fc_hz": get0(fh),
            "high_Q": get0(Qh),
            "high_gain_db": get0(gh),
        }

class StaticShelf(nn.Module):
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


    # --- Helpers to constrain RAW parameters (for external control) ---------
    def _constrain_fc_raw(self, raw_fc: torch.Tensor | float):
        t = raw_fc if isinstance(raw_fc, torch.Tensor) else torch.tensor(raw_fc, device=self.raw_fc.device if hasattr(self, 'raw_fc') else None, dtype=self.raw_fc.dtype if hasattr(self, 'raw_fc') else None)
        K = torch.nn.functional.softplus(t) + self._eps
        fc = (self.fs / math.pi) * torch.atan(K)
        return fc.clamp(self.fmin, self.fmax - self._eps)

    def _constrain_Q_raw(self, raw_Q: torch.Tensor | float):
        t = raw_Q if isinstance(raw_Q, torch.Tensor) else torch.tensor(raw_Q, device=self.raw_Q.device if hasattr(self, 'raw_Q') else None, dtype=self.raw_Q.dtype if hasattr(self, 'raw_Q') else None)
        return torch.nn.functional.softplus(t)

    def _constrain_gdb_raw(self, raw_gdb: torch.Tensor | float):
        t = raw_gdb if isinstance(raw_gdb, torch.Tensor) else torch.tensor(raw_gdb, device=self.raw_gdb.device if hasattr(self, 'raw_gdb') else None, dtype=self.raw_gdb.dtype if hasattr(self, 'raw_gdb') else None)
        return self.max_gain_db * torch.tanh(t)

    def _to_tensor(self, x):
        # Ensure the value is a tensor on the right device/dtype
        t = x if isinstance(x, torch.Tensor) else torch.tensor(x, device=self.raw_fc.device, dtype=self.raw_fc.dtype)
        return t

    def forward(self,
                x: torch.Tensor,
                fc_hz: torch.Tensor | float | None = None,
                Q: torch.Tensor | float | None = None,
                gain_db: torch.Tensor | float | None = None,
                *,
                from_raw: bool = False,
                invert: bool = False):
        """
        Apply shelf filter to `x`.
        If `fc_hz`, `Q`, or `gain_db` are provided, they override the module's
        internal learnable parameters. Set `from_raw=True` if those overrides are
        *raw* (unconstrained) parameters that should be mapped like the module does
        internally (softplus/tanh/atan mapping).
        """
        # Choose parameter sources
        if from_raw:
            fc  = self._constrain_fc_raw(fc_hz if fc_hz is not None else self.raw_fc)
            q   = self._constrain_Q_raw(Q       if Q      is not None else self.raw_Q)
            gdb = self._constrain_gdb_raw(gain_db if gain_db is not None else self.raw_gdb)
        else:
            fc  = (self._fc_forward() if fc_hz is None else self._to_tensor(fc_hz))
            q   = (self._Q_forward()  if Q     is None else self._to_tensor(Q))
            gdb = (self._gdb_forward() if gain_db is None else self._to_tensor(gain_db))

        # Normal path: apply the learned shelf
        if not invert:
            return self.filter(x, fc=fc, Q=q, gain_db=gdb)

        # Inversion path: apply the UNITY‑NORMALIZED inverse at \omega_ref = 2*pi*fc/fs.
        # Steps:
        #  1) Design the synthesis shelf coefficients (a_s, b_s) for (fc, Q, gdb).
        #  2) Evaluate |H(e^{j\omega_ref})| at \omega_ref corresponding to fc.
        #  3) Build the exact inverse by swapping numerator/denominator.
        #  4) Normalize the inverse so that its magnitude is unity at \omega_ref
        #     by multiplying the inverse numerator by M = |H(e^{j\omega_ref})|.
        #  5) Ensure a_inv[0] == 1 by dividing both sets by a_inv_raw[...,0].

        # Ensure dtype/device and flatten leading dims for torchaudio.lfilter
        x = x.to(device=self.filter._device or x.device, dtype=self.filter._dtype or x.dtype)
        *lead, T = x.shape
        N = int(torch.tensor(lead).prod().item()) if lead else 1
        x2 = x.reshape(N, T)

        # 1) Design synthesis shelf
        a_s, b_s = self.filter._design(fc, q, gdb, N)  # shapes: [N, 3]

        # 2) Evaluate magnitude at \omega_ref associated with fc
        #    Make sure fc is expanded to [N]
        fc_t = self.filter._to_tensor(fc)
        fc_vec = fc_t.expand(N) if fc_t.ndim == 0 else fc_t.reshape(N)
        w_ref = 2.0 * math.pi * fc_vec / self.fs  # [N]

        # Complex dtype matched to real precision
        ctype = torch.complex64 if x2.dtype == torch.float32 else torch.complex128
        z1 = torch.cos(w_ref).to(ctype) - 1j * torch.sin(w_ref).to(ctype)  # e^{-j w}
        z2 = z1 * z1

        b_c = b_s.to(ctype)
        a_c = a_s.to(ctype)
        H_num = b_c[..., 0] + b_c[..., 1] * z1 + b_c[..., 2] * z2
        H_den = a_c[..., 0] + a_c[..., 1] * z1 + a_c[..., 2] * z2
        M = torch.abs(H_num / H_den).to(x2.dtype)  # [N]

        # 3) & 5) Exact inverse (swap) with a0 normalization
        a_inv_raw = b_s  # new denominator
        b_inv_raw = a_s  # new numerator
        eps = torch.finfo(x2.dtype).eps
        inv_a0 = a_inv_raw[..., 0:1]
        a_inv = a_inv_raw / (inv_a0 + eps)
        b_inv = b_inv_raw / (inv_a0 + eps)

        # 4) Unity normalization at w_ref: scale numerator by M
        b_inv = b_inv * M.unsqueeze(-1)

        # Filter with the inverse
        y2 = torchaudio.functional.lfilter(x2, a_coeffs=a_inv, b_coeffs=b_inv, clamp=False)
        return y2.reshape(*lead, T)


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