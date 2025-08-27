from typing import Union
import numpy as np
import torch
import librosa

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
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Create [B, N] mostly‑zero signal with uniform noise bursts in [-1, 1].
    If `burst_len_samples` is provided, it overrides `noise_ms` for burst length.
    Determinism: pass a `seed` or a pre‑constructed `generator` to get repeatable bursts.
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
        # zero‑mean uniform noise in [-1, +1]
        noise = torch.rand(batch_size, end - start, device=device, dtype=dtype, generator=generator) * 2 - 1
        sig[:, start:end] = noise
    return sig