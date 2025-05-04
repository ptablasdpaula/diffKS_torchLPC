from __future__ import annotations

import torch
from torch import nn
from third_party.auraloss.auraloss.freq import MultiResolutionSTFTLoss as _MultiSTFT

from diffKS import DiffKS

# ───────────────────────────────── STFT ────────────────────────────────────
class STFTLoss(nn.Module):
    def __init__(self,
                 sample_rate: int,
                 scale_invariant: bool = True,
                 perceptual: bool = True):
        super().__init__()
        self.fn = _MultiSTFT(sample_rate        = sample_rate,
                             scale_invariance   = scale_invariant,
                             perceptual_weighting = perceptual)

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor: # i/o: (B, 1, N)
        return self.fn(pred, target) # i/o: (B, 1, N)

# ───────────────────────────── Parameter loss ──────────────────────────────
def _flatten(*tensors: torch.Tensor) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


@torch.no_grad()
def parameter_loss_from_meta(pred_agent: DiffKS, meta: dict) -> torch.Tensor:
    """
    RMS distance between *predicted* constrained parameters and the **constrained
    parameters from metadata JSON** using the same constraining logic as in DiffKS.
    """
    # Get the predicted parameters after constraining
    # These are already constrained by the model during forward pass
    pred_loop_coeffs = pred_agent.get_constrained_l_coefficients(
        pred_agent.loop_coefficients,
        pred_agent.loop_gain
    )
    pred_exc_coeffs = pred_agent.get_constrained_exc_coefficients(
        pred_agent.exc_coefficients
    )

    # Flatten predicted constrained parameters
    pred = _flatten(pred_loop_coeffs, pred_exc_coeffs)

    # Convert metadata to tensors with same dtype/device
    meta_loop_coeffs = torch.tensor(
        meta["loop_coefficients"],
        dtype=pred_agent.loop_coefficients.dtype,
        device=pred_agent.loop_coefficients.device
    )
    meta_loop_gain = torch.tensor(
        meta["loop_gain"],
        dtype=pred_agent.loop_gain.dtype,
        device=pred_agent.loop_gain.device
    )
    meta_exc_coeffs = torch.tensor(
        meta["exc_coefficients"],
        dtype=pred_agent.exc_coefficients.dtype,
        device=pred_agent.exc_coefficients.device
    )

    # Reshape if necessary to match the expected dimensions
    if meta_loop_coeffs.dim() == 1:
        meta_loop_coeffs = meta_loop_coeffs.view(1, 1, -1)
    if meta_loop_gain.dim() == 1:
        meta_loop_gain = meta_loop_gain.view(1, 1, -1)
    if meta_exc_coeffs.dim() == 1:
        meta_exc_coeffs = meta_exc_coeffs.view(1, 1, -1)

    # Apply the same constraining operations as in DiffKS
    meta_loop_coeffs_constrained = pred_agent.get_constrained_l_coefficients(
        meta_loop_coeffs, meta_loop_gain
    )
    meta_exc_coeffs_constrained = pred_agent.get_constrained_exc_coefficients(
        meta_exc_coeffs
    )

    # Flatten meta constrained parameters
    target = _flatten(meta_loop_coeffs_constrained, meta_exc_coeffs_constrained)

    # Compute RMS
    return torch.sqrt(torch.mean((pred - target) ** 2))