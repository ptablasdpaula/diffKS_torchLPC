from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlpc import sample_wise_lpc
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from utils import get_device

class DiffKS(nn.Module):
    """
    A differentiable Karplus–Strong model that supports time-varying
    fractional delay and time-varying reflection coefficients.
    """

    def __init__(
        self,
        burst: torch.Tensor,
        n_frames: int,
        sample_rate: int,
        lowest_note_in_hz: float,
        init_coeffs_frames: Optional[torch.Tensor] = None,
        coeff_range: Tuple[float, float] = (-2, 0),
    ):
        super().__init__()

        # Store or compute needed values
        self.n_frames = n_frames
        self.sample_rate = sample_rate
        self.lowest_note_in_hz = lowest_note_in_hz
        self.coeff_vector_size = int(self.sample_rate // self.lowest_note_in_hz) + 2

        # Initialize reflection coefficients
        if init_coeffs_frames is None:
            raw_init = torch.empty(self.n_frames, 2).uniform_(*coeff_range)
        else:
            # Convert to logit space
            raw_init = torch.special.logit(init_coeffs_frames)

        self.raw_coeff_frames = nn.Parameter(raw_init)

        # The excitation burst is stored as a non-trainable buffer
        self.register_buffer("excitation", burst.float())

    def forward(self, delay_len_frames: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Compute the time-domain output of the Karplus–Strong waveguide.
        """
        time_varying_delay, time_varying_coeffs = self.get_upsampled_parameters(
            delay_len_frames, n_samples
        )

        # Integer / fractional part
        z_L = torch.floor(time_varying_delay).long()
        z_Lminus1 = z_L + 1
        z_Lminus2 = z_L + 2
        alfa = time_varying_delay - z_L

        # Make sure we don't exceed size
        assert torch.all(z_Lminus2 < self.coeff_vector_size), (
            f"Delay index exceeds the buffer size ({self.coeff_vector_size})."
        )

        # Construct matrix A for sample_wise_lpc
        A = torch.zeros(
            (1, n_samples, self.coeff_vector_size),
            device=self.excitation.device
        )

        b1 = time_varying_coeffs[:, 0]
        b2 = time_varying_coeffs[:, 1]

        # Place reflection coefficients
        A[0, torch.arange(n_samples), z_L] = -(b1 * (1 - alfa))
        A[0, torch.arange(n_samples), z_Lminus1] = -(b1 * alfa + b2 * (1 - alfa))
        A[0, torch.arange(n_samples), z_Lminus2] = -(b2 * alfa)

        # Prepare the excitation burst
        x = torch.zeros(n_samples, device=self.excitation.device)
        x[: self.excitation.shape[0]] = self.excitation
        x = x.unsqueeze(0)

        # Run the waveguide update
        y_out = sample_wise_lpc(x, A)
        return y_out.squeeze(0)

    def get_constrained_coefficients(self, for_plotting: bool = False) -> torch.Tensor:
        """
        Compute b1, b2 in [0, 1], ensuring b1 + b2 <= 1 at each frame.
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
        """
        Upsample the delay lengths and reflection coefficients
        from frame rate to sample rate using cubic splines.
        """
        coeff_frames = self.get_constrained_coefficients(for_plotting=for_plotting)

        def interpolate_fn():
            t_in = torch.linspace(0, 1, steps=self.n_frames,
                                  device=delay_len_frames.device)
            t_out = torch.linspace(0, 1, steps=num_samples,
                                   device=delay_len_frames.device)

            delay_input = delay_len_frames.view(1, self.n_frames, 1)
            coeff_input = coeff_frames.view(1, self.n_frames, 2)

            delay_coeffs = natural_cubic_spline_coeffs(t_in, delay_input)
            coeffs_coeffs = natural_cubic_spline_coeffs(t_in, coeff_input)

            delay_interp = NaturalCubicSpline(delay_coeffs).evaluate(t_out)
            coeff_interp = NaturalCubicSpline(coeffs_coeffs).evaluate(t_out)

            # Remove extra dims
            delay_interp = delay_interp.squeeze(0).squeeze(-1)
            coeff_interp = coeff_interp.squeeze(0)
            return delay_interp, coeff_interp

        if for_plotting:
            with torch.no_grad():
                delay, coeffs = interpolate_fn()
                return delay.detach().cpu(), coeffs.detach().cpu()
        else:
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
