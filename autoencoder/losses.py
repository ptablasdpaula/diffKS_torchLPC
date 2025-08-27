import torch
import torch.nn.functional as F
from third_party.auraloss.auraloss.freq import MultiResolutionSTFTLoss

# --- Envelope helper (frame-aligned, strict) ---------------------------------

def _frame_env(x: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Return average |x| per frame so output has exactly `num_frames`.
    Assumes x shape [B,N]. Uses hop = N // num_frames and trims to hop*num_frames.
    """
    assert x.dim() == 2, f"_frame_env expects [B,N], got {tuple(x.shape)}"
    B, N = x.shape
    hop = N // num_frames
    assert hop > 0, f"num_frames ({num_frames}) too large for N={N}"
    cutN = hop * num_frames
    x = x[:, :cutN]
    env = F.avg_pool1d(x.abs().unsqueeze(1), kernel_size=hop, stride=hop, ceil_mode=False).squeeze(1)
    assert env.size(1) == num_frames, f"Envelope frames {env.size(1)} != {num_frames}"
    return env

def build_smooth_mrstft(
        sample_rate: int = 16000,
        scale_invariance: bool = False):
    return MultiResolutionSTFTLoss(
        fft_sizes=[257, 509, 1019, 2039, 4093],
        hop_sizes=[128, 254, 509, 1019, 2046],
        win_lengths=[257, 509, 1019, 2039, 4093],
        window="flattop",         # WF: Flat-top window with low sidelobes
        mag_distance="L2",        # D2: squared-L2 distance
        log_eps=1.0,              # C2: log-compression with ε=1
        w_sc=1.0,                 # spectral-convergence
        w_log_mag=1.0,            # log-magnitude
        w_lin_mag=1.0,            # linear-magnitude
        perceptual_weighting=True,
        scale_invariance=scale_invariance,
        sample_rate=sample_rate,
    )