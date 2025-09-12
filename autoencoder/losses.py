import torch
import torch.nn.functional as F
from third_party.auraloss.auraloss.freq import MultiResolutionSTFTLoss
import torch.nn as nn
from typing import Optional, Union
from kymatio.torch import TimeFrequencyScattering
from data.preprocess import a_weighted_loudness

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
        w_lin_mag=0.0,            # linear-magnitude
        perceptual_weighting=True,
        scale_invariance=scale_invariance,
        sample_rate=sample_rate,
    )


# --- JTFS Loss --------------------------------------------------------------

class JTFSTLoss(nn.Module):
    """
    Joint Time-Frequency Scattering loss wrapper with a fixed target length.
    Inputs are padded/cropped to `shape` so you can call it with arbitrary
    clips and still share a single JTFS transform.
    """
    def __init__(
        self,
        shape: int,
        J: int,
        Q1: int,
        Q2: int,
        J_fr: int,
        Q_fr: int,
        T: Optional[Union[str, int]] = None,
        F: Optional[Union[str, int]] = None,
        format_: str = "joint",
        p: int = 2,
    ):
        super().__init__()
        assert format_ in ("time", "joint")
        self.shape = int(shape)
        self.format = format_
        self.p = p
        self.jtfs = TimeFrequencyScattering(
            shape=(self.shape,),
            J=J,
            Q=(Q1, Q2),
            Q_fr=Q_fr,
            J_fr=J_fr,
            T=T,
            F=F,
            format=format_,
        )

    def forward(self, x: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """x, x_target: [B, 1, N] real waveforms.
        Returns a scalar JTFS distance (mean over batch).
        """
        assert x.ndim == x_target.ndim == 3, "Expect [B, 1, N] tensors"
        assert x.size(1) == x_target.size(1) == 1, "Expect mono audio"
        B, _, N = x.shape

        # Pad/crop to the configured length
        S = self.shape
        if N < S:
            pad = S - N
            x = F.pad(x, (0, pad))
            x_target = F.pad(x_target, (0, pad))
        elif N > S:
            x = x[..., :S]
            x_target = x_target[..., :S]

        # Flatten channel dim
        x = x.view(-1, 1, S)
        x_target = x_target.view(-1, 1, S)

        Sx = self.jtfs(x)
        Sx_t = self.jtfs(x_target)

        if self.format == "time":
            # Drop 0th-order coef along the JTFS order axis
            Sx = Sx[:, :, 1:, :]
            Sx_t = Sx_t[:, :, 1:, :]
            dist = torch.linalg.vector_norm(Sx_t - Sx, ord=self.p, dim=-1)
        else:  # 'joint'
            dist = torch.linalg.vector_norm(Sx_t - Sx, ord=self.p, dim=(-2, -1))

        return dist.mean()


def build_jtfst(shape: int = 64000):
    """Construct a JTFS loss with fixed recommended hyperparameters.

    Hardcoded params: J=12, Q1=8, Q2=2, J_fr=3, Q_fr=2, T=None, F=None,
    format_='joint', p=2.

    Args:
        shape: Number of samples the JTFS is built for. Inputs will be
               zero-padded or center-cropped to this length inside the loss.
    """
    return JTFSTLoss(
        shape=shape,
        J=12,
        Q1=8,
        Q2=2,
        J_fr=3,
        Q_fr=2,
        T=None,
        F=None,
        format_="joint",
        p=2,
    )

# --- A-weighted loudness loss ------------------------------------------------
class ALoudnessLoss(nn.Module):
    """
    Framewise A-weighted loudness distance.

    Uses `data.preprocess.a_weighted_loudness(x)` which returns (B, F) log-power
    loudness over hop-sized frames. Accepts audio as [B, 1, N] or [B, N].

    By default, we crop both inputs to the same number of *samples* (min length)
    before computing loudness, then crop the resulting frame sequences to the
    same number of frames (min) to compare.
    """
    def __init__(self, p: int = 1, reduction: str = "mean", sync_on_samples: bool = True):
        super().__init__()
        assert p in (1, 2), "p must be 1 (L1) or 2 (L2)"
        assert reduction in ("none", "mean", "sum")
        self.p = int(p)
        self.reduction = reduction
        self.sync_on_samples = bool(sync_on_samples)

    def forward(self, x: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        # Expect [B, 1, N] or [B, N]
        assert x.ndim in (2, 3) and x_target.ndim in (2, 3), "Expect [B,N] or [B,1,N]"
        if x.ndim == 3:
            assert x.size(1) == 1, "mono expected"
            x = x.squeeze(1)
        if x_target.ndim == 3:
            assert x_target.size(1) == 1, "mono expected"
            x_target = x_target.squeeze(1)

        # Optionally crop to the same number of samples first
        if self.sync_on_samples and x.size(-1) != x_target.size(-1):
            N = min(x.size(-1), x_target.size(-1))
            x = x[..., :N]
            x_target = x_target[..., :N]

        # Compute A-weighted log-power loudness per frame: (B, F)
        Lx  = a_weighted_loudness(x)
        Lxt = a_weighted_loudness(x_target)

        # Align frame counts by cropping to the minimum
        Fmin = min(Lx.size(-1), Lxt.size(-1))
        if Fmin == 0:
            # Degenerate case; return zero to avoid NaNs
            return torch.zeros((), device=x.device, dtype=x.dtype)
        Lx  = Lx[..., :Fmin]
        Lxt = Lxt[..., :Fmin]

        diff = Lxt - Lx
        if self.p == 1:
            diff = diff.abs()
        else:  # p == 2
            diff = diff.pow(2)

        if self.reduction == "mean":
            return diff.mean()
        elif self.reduction == "sum":
            return diff.sum()
        else:
            return diff


def build_a_loudness_loss(p: int = 1, reduction: str = "mean", sync_on_samples: bool = True) -> ALoudnessLoss:
    """Convenience constructor for A-weighted loudness loss.

    Args:
        p: 1 for L1, 2 for L2.
        reduction: one of {"mean", "sum", "none"}.
        sync_on_samples: if True, crop inputs to the same number of samples
            before computing loudness. If False, compares after loudness by
            cropping to the min number of frames.
    """
    return ALoudnessLoss(p=p, reduction=reduction, sync_on_samples=sync_on_samples)