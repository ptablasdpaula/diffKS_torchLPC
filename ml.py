import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn

class STEPeakPick(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            x: Tensor, # [Batches, Frames]
            threshold: float
    ) -> Tensor:
        # Expect a 2-D tensor: (batch_size, num_frames)
        if x.dim() != 2:
            raise ValueError(f"x must be a 2-D tensor of shape (B, Frames), got shape: {x.shape}")

        # Boolean mask of above-threshold values
        above = x > threshold

        # Detect the start of each contiguous above-threshold region along frames
        above_int = above.int()
        is_start = (above_int - above_int.roll(1, dims=1)) == 1

        # Assign region IDs per batch row
        region_id = torch.cumsum(is_start.long(), dim=1) * above.long()

        # Allocate tensor for per-region maxima, same shape as x
        max_vals = torch.zeros_like(x)

        # Scatter-reduce to compute per-region max along frames
        max_vals.scatter_reduce_(
            dim=1,
            index=region_id,
            src=x,
            reduce="amax",
            include_self=False
        )

        # Gather the max value for each position's region
        peak_vals = max_vals.gather(1, region_id)

        # Mark peaks: positions equal to their region max and above threshold
        peaks = (x == peak_vals) & above

        return peaks.to(x.dtype)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        # Straight-through: dL/dx = dL/dpeaks
        return grad_output, None


class STEUpsampleFirstPeak(torch.autograd.Function):
    """
    Frame-rate → sample-rate trigger resampler that keeps only the *first*
    sample of every above-zero block.  Forward path is binary / non-diff.,
    backward path is identity (straight-through) after a 1-D down-sample.

       trig_f               [B, F]   (input, 0/1)
          │  nearest-nn ↑
          ├───────────────► repeat blocks
          │                 [B, N]
          │  first-peak │
          ▼              ▼
       trig_s               [B, N]   (output, 0/1, one-hot peaks)
    """

    @staticmethod
    def forward(
            ctx,
            trig_f: Tensor,
            n_samples: int
    ) -> Tensor:
        batch, n_frames = trig_f.shape
        device, dtype = trig_f.device, trig_f.dtype

        # ─── 1. nearest-neighbour up-sample to sample rate ────────────────────
        rep = F.interpolate(trig_f.unsqueeze(1).float(),  # [B,1,F] → [B,1,N]
                            size=n_samples,
                            mode="nearest").squeeze(1)     # [B,N]

        # ─── 2. keep only the first 1 of every contiguous block ──────────────
        shifted = F.pad(rep, (1, 0))[:, :-1]               # prepend a zero
        peaks   = (rep > 0.5) & (shifted <= 0.5)           # rising edges

        out = peaks.to(dtype)                              # cast back

        # save shapes for backward
        ctx.n_frames     = n_frames
        ctx.n_samples  = n_samples
        return out

    @staticmethod
    def backward(
            ctx,
            *grad_outputs
    ):
        """
        Straight-through: pass gradients from sample grid → frame grid by
        a simple linear down-sample (area/mean would also work).
        """
        grad_f = F.interpolate(grad_outputs[0].unsqueeze(1),       # [B,1,N] → [B,1,F]
                               size=ctx.n_frames,
                               mode="linear",
                               align_corners=False).squeeze(1)
        return grad_f, None   # no grad for the int argument

def mlp(in_size, hidden_size, n_layers):
    channels = [in_size] + (n_layers) * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)

def gru(n_input, hidden_size):
    return nn.GRU(n_input * hidden_size, hidden_size, batch_first=True)