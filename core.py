from typing import Union
import numpy as np
import torch
import librosa
from torchlpc import sample_wise_lpc
from torch import nn
import math

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
    # Deduplicate identical frame indices
    onset_frames = np.asarray(onset_frames, dtype=int)
    if onset_frames.size:
        onset_frames = np.unique(onset_frames)
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
                     burst_len_samples: int | None = None,
                     seed: int | None = 12345,
                     generator: torch.Generator | None = None,
                     impulse_instead: bool = False,
                     impulse_window: str = "hann",
                     use_fixed_length: bool = False,
                     fixed_length_ms: float = 3.0) -> torch.Tensor:
    """
    Create [B, N] mostly‑zero signal with uniform noise bursts in [-1, 1].

    Impulse mode: if `impulse_instead` is True, ignore noise burst logic and write
    an impulse at each onset sample. `impulse_window` can be "delta" for a single-
    sample Kronecker delta, or "hann" for a 3-sample Hann blip (energy-normalized).

    This keeps excitation energy independent of pitch and makes per-onset gain
    learning easier. In impulse mode, random generator settings are ignored.

    Fixed-length mode: if `use_fixed_length` is True, the noise burst length is
    set to `fixed_length_ms` (converted to samples) for all onsets, ignoring
    `burst_len_samples` and `noise_ms`. This keeps burst duration independent of f0.
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    # Optional deterministic RNG for repeatable bursts
    if generator is None:
        # Create a generator tied to the target device for torch.rand (supports cpu/cuda/mps)
        gen_device = 'cpu'
        if device is not None and hasattr(device, 'type'):
            if device.type in ('cpu', 'cuda', 'mps'):
                gen_device = device.type
        generator = torch.Generator(device=gen_device)
        if seed is not None:
            generator.manual_seed(int(seed))
    else:
        # If a generator was provided, ensure it's seeded if requested
        if seed is not None:
            generator.manual_seed(int(seed))

    if use_fixed_length:
        seg_len = max(1, int(round((fixed_length_ms / 1000.0) * sample_rate)))
    elif burst_len_samples is not None:
        seg_len = max(1, int(burst_len_samples))
    else:
        seg_len = max(1, int(round((noise_ms / 1000.0) * sample_rate)))

    sig = torch.zeros(batch_size, num_samples, device=device, dtype=dtype)
    if onset_samples.size == 0:
        onset_samples = np.array([0], dtype=int)

    # Impulse mode: ignore burst logic and return early
    if impulse_instead:
        if impulse_window not in ("delta", "hann"):
            raise ValueError("impulse_window must be 'delta' or 'hann'")

        if impulse_window == "delta":
            # 1-sample Kronecker delta at each onset
            for s in onset_samples:
                s = int(s)
                if 0 <= s < num_samples:
                    sig[:, s] = 1.0
        else:
            # 3-sample Hann blip [0.25, 0.5, 0.25], energy-normalized
            kernel = torch.tensor([0.25, 0.5, 0.25], device=device, dtype=dtype)
            kernel = kernel / torch.sqrt(torch.sum(kernel * kernel))
            for s in onset_samples:
                center = int(s)
                if center < 0 or center >= num_samples:
                    continue
                # write at indices [center-1, center, center+1] with bounds check
                x_start = center - 1
                k_start = 0
                if x_start < 0:
                    k_start = -x_start
                    x_start = 0
                x_end = center + 2  # exclusive
                k_end = 3
                if x_end > num_samples:
                    k_end -= (x_end - num_samples)
                    x_end = num_samples
                if k_start < k_end:  # valid overlap
                    sig[:, x_start:x_end] = kernel[k_start:k_end]
        return sig
    for s in onset_samples:
        start = int(s)
        if start >= num_samples:
            continue
        end = min(start + seg_len, num_samples)
        # zero‑mean uniform noise in [-1, +1]
        noise = torch.rand(batch_size, end - start, device=device, dtype=dtype, generator=generator) * 2 - 1
        sig[:, start:end] = noise
    return sig

# =============================================================
# IIR Resonator Filterbank (parallel all-pole second-order bands)
# =============================================================
class IIRResonatorBank(nn.Module):
    """Parallel bank of second-order all-pole resonators (biquad denominators only).
    Each band k is defined by a complex pole pair with radius r_k and angle w0_k.
    Filtering is performed with torchlpc.sample_wise_lpc so that it is fully
    differentiable and batched on GPU.

    Args:
        fs (int): sample rate in Hz.
        n_bands (int): number of resonator bands.
        min_hz (float): lowest center frequency.
        max_hz (float): highest center frequency (<= Nyquist).
        q (float): constant-Q of resonators (BW = f0 / q).
    """
    def __init__(self, fs: int, n_bands: int, min_hz: float, max_hz: float, q: float = 8.0):
        super().__init__()
        self.fs = float(fs)
        self.n_bands = int(n_bands)
        nyq = 0.5 * self.fs
        f0 = torch.linspace(min_hz, min(max_hz, nyq * 0.98), steps=self.n_bands)
        # Constant-Q bandwidth -> pole radius r = exp(-pi * BW / fs) with BW = f0 / q
        bw = f0 / float(q)
        r = torch.exp(-math.pi * bw / self.fs)
        w0 = 2.0 * math.pi * f0 / self.fs
        a1 = -2.0 * r * torch.cos(w0)
        a2 = r ** 2
        # Store denominator coefficients as buffers (shape [K])
        self.register_buffer("a1", a1.to(torch.float32))
        self.register_buffer("a2", a2.to(torch.float32))

    def forward(self, noise: torch.Tensor, gains_up: torch.Tensor) -> torch.Tensor:
        """Filter white noise through the bank and apply time-varying gains.
        Args:
            noise: [B, N] white noise.
            gains_up: [B, N, K] linear gains per band at sample-rate.
        Returns:
            y: [B, N] synthesized wideband noise shaped by the filterbank.
        """
        B, N = noise.shape
        K = self.a1.numel()
        assert gains_up.shape == (B, N, K), f"Expected gains_up [B,N,K]={B,N,K}, got {tuple(gains_up.shape)}"
        out_acc = noise.new_zeros(B, N)
        for k in range(K):
            # Build time-constant LPC denominator for this band and expand over time
            a_k = torch.stack([self.a1[k].expand(B, N), self.a2[k].expand(B, N)], dim=-1)
            # y_k = filter(noise) with all-pole denominator 1 + a1 z^-1 + a2 z^-2
            y_k = sample_wise_lpc(noise, a_k)
            # Apply per-sample gain for this band and accumulate
            out_acc = out_acc + y_k * gains_up[:, :, k]
        # Light energy normalization to avoid scale blow-up when many bands active
        return out_acc / math.sqrt(max(1, K))