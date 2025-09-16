from typing import Union
import numpy as np
import torch
import librosa
import math
from torch.nn import functional as F
import torch.fft as fft

MIN_HZ = 20
MAX_HZ = 8000

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

def hz_to_samples(f0_hz: torch.Tensor, fs: int) -> torch.Tensor:
    """Convert fundamental frequency in Hz to samples/period given sample rate fs.
    Keeps dtype/device of f0_hz.
    """
    return torch.as_tensor(float(fs), dtype=f0_hz.dtype, device=f0_hz.device) / f0_hz

def sigmoid_valid_gain_range(gain_logits: torch.Tensor) -> torch.Tensor:
    """Map unconstrained logits to a value g ∈ (0.9, 1)"""
    return torch.sigmoid(gain_logits) * 0.1 + 0.9

def scale_function(x: torch.Tensor) -> torch.Tensor:
    """IRCAM-style positive scaling used for band amps.
    Matches: 2 * sigmoid(x)**log(10) + 1e-7.
    """
    return 2.0 * torch.sigmoid(x) ** (math.log(10)) + 1e-7

def amp_to_impulse_response(amp: torch.Tensor, target_size: int) -> torch.Tensor:
    """Map non-negative band amplitudes to a real impulse response per frame.
    Args:
        amp: [B, T, K] non-negative magnitudes (frequency bands)
        target_size: output impulse length per frame (e.g., block size)
    Returns:
        impulse: [B, T, target_size]
    """
    # Make a real spectrum with zero imaginary part: [B, T, K] -> complex [B,T,K]
    amp_c = torch.view_as_complex(torch.stack([amp, torch.zeros_like(amp)], dim=-1))
    # IFFT to time domain; length inferred from K
    ir = fft.irfft(amp_c)
    Lf = ir.shape[-1]
    # Center the IR and apply Hann window for smoothness
    ir = torch.roll(ir, shifts=Lf // 2, dims=-1)
    win = torch.hann_window(Lf, dtype=ir.dtype, device=ir.device)
    ir = ir * win
    # Pad or crop to target_size and roll back
    if Lf < target_size:
        ir = F.pad(ir, (0, target_size - Lf))
    elif Lf > target_size:
        ir = ir[..., :target_size]
    ir = torch.roll(ir, shifts=-(Lf // 2), dims=-1)
    return ir

def fft_convolve(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """FFT overlap-save style convolution per frame (no state across frames).
    signal: [B, T, L]
    kernel: [B, T, L]
    Returns: [B, T, L]
    """
    # Pad to same length in a simple circular fashion per frame
    s = F.pad(signal, (0, signal.shape[-1]))
    k = F.pad(kernel, (kernel.shape[-1], 0))
    out = fft.irfft(fft.rfft(s) * fft.rfft(k))
    out = out[..., out.shape[-1] // 2:]
    return out