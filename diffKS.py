from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlpc import sample_wise_lpc
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from utils import get_device

class DiffKS(nn.Module):
    """
    A differentiable Karplus–Strong model with time-varying fractional delay
    and configurable-order filter with normalized coefficients.
    """

    def __init__(
        self,
        burst: torch.Tensor,
        n_frames: int,
        sample_rate: int,
        lowest_note_in_hz: float,
        l_filter_order: int = 5,
        init_coeffs_frames: Optional[torch.Tensor] = None,
        coeff_range: Tuple[float, float] = (-2, 0),
    ):
        super().__init__()
        assert l_filter_order >= 1, "Filter order must be at least 1"

        self.n_frames = n_frames
        self.sample_rate = sample_rate
        self.lowest_note_in_hz = lowest_note_in_hz
        self.l_filter_order = l_filter_order
        self.coeff_vector_size = int(self.sample_rate // self.lowest_note_in_hz) + self.l_filter_order + 1

        if init_coeffs_frames is None:
            raw_init = torch.empty(self.n_frames, self.l_filter_order).uniform_(*coeff_range)
        else:
            raw_init = torch.special.logit(init_coeffs_frames)

        self.raw_coeff_frames = nn.Parameter(raw_init)

        # The excitation burst is stored as a non-trainable buffer
        self.register_buffer("excitation", burst.float())

    def forward(self, delay_len_frames: torch.Tensor, n_samples: int) -> torch.Tensor:
        delay_interp, coeff_interp = self.get_upsampled_parameters(delay_len_frames, n_samples)

        z_l = torch.floor(delay_interp).long()
        alfa = delay_interp - z_l

        idxs = [z_l + i for i in range(self.l_filter_order + 1)]
        assert torch.all(idxs[-1] < self.coeff_vector_size), "Delay index exceeds the buffer size"

        A = torch.zeros((1, n_samples, self.coeff_vector_size), device=self.excitation.device)
        b = coeff_interp  # shape: (n_samples, l_filter_order)

        # First term: z^L
        A[0, torch.arange(n_samples), idxs[0]] = -(1 - alfa) * b[:, 0]

        # Middle terms with crossfade
        for i in range(1, self.l_filter_order):
            A[0, torch.arange(n_samples), idxs[i]] = -(alfa * b[:, i - 1] + (1 - alfa) * b[:, i])

        # Last term: z^(L + l_filter_order)
        A[0, torch.arange(n_samples), idxs[-1]] = -alfa * b[:, -1]

        x = torch.zeros(n_samples, device=self.excitation.device)
        x[: self.excitation.shape[0]] = self.excitation
        x = x.unsqueeze(0)

        y_out = sample_wise_lpc(x, A)
        return y_out.squeeze(0)

    def get_constrained_coefficients(self, for_plotting: bool = False) -> torch.Tensor:
        """
        Generate coefficient trajectories over time using a 1-layer linear network.

        Each coefficient is modeled as: coeff(t) = weight * t + bias
        with optional normalization so the filter remains stable.
        """
        def compute_coeffs():
            sigmoid_b = torch.sigmoid(self.raw_coeff_frames)
            sum_b = sigmoid_b.sum(dim=-1, keepdim=True)
            return sigmoid_b / sum_b

        if for_plotting:
            with torch.no_grad():
                return compute_coeffs().detach().cpu()
        else:
            return compute_coeffs()

    def get_upsampled_parameters(
        self,
        delay_len_frames: torch.Tensor,
        num_samples: int,
        for_plotting: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        coeff_frames = self.get_constrained_coefficients(for_plotting=for_plotting)

        def interpolate_fn():
            if self.n_frames == 1:
                return delay_len_frames.repeat(num_samples), coeff_frames.repeat(num_samples, 1)

            t_in = torch.linspace(0, 1, steps=self.n_frames,
                                  device=delay_len_frames.device)
            t_out = torch.linspace(0, 1, steps=num_samples,
                                   device=delay_len_frames.device)

            delay_input = delay_len_frames.view(1, self.n_frames, 1)
            coeff_input = coeff_frames.view(1, self.n_frames, self.l_filter_order)

            delay_coeffs = natural_cubic_spline_coeffs(t_in, delay_input)
            coeffs_coeffs = natural_cubic_spline_coeffs(t_in, coeff_input)

            delay_interp = NaturalCubicSpline(delay_coeffs).evaluate(t_out).squeeze(0).squeeze(-1)
            coeff_interp = NaturalCubicSpline(coeffs_coeffs).evaluate(t_out).squeeze(0)
            return delay_interp, coeff_interp

        if for_plotting:
            with torch.no_grad():
                delay, coeffs = interpolate_fn()
                return delay.detach().cpu(), coeffs.detach().cpu()
        return interpolate_fn()


def noise_burst(
    sample_rate: int,
    length_s: float,
    burst_width_s: float,
    return_1D: bool = True
) -> torch.Tensor:
    """
    Generate a single-channel noise burst and zero-pad it.
    """
    burst_width_n = int(sample_rate * burst_width_s)
    total_length_n = int(sample_rate * length_s)

    if total_length_n < burst_width_n:
        raise ValueError(
            f"Requested total length {length_s:.3f}s < noise burst width {burst_width_s:.3f}s."
        )

    burst = torch.rand((1, burst_width_n, 1), device=get_device()) - 0.5

    # Zero-pad up to total_length_n
    pad_amount = total_length_n - burst_width_n
    padded_burst = F.pad(
        burst,
        pad=(0, 0, 0, pad_amount),
        mode="constant",
        value=0.0,
    )

    if return_1D:
        return padded_burst.squeeze()
    else:
        return padded_burst
