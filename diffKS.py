from os import remove
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlpc import sample_wise_lpc
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from utils import get_device

LAGRANGE_ORDER = 5

def spline_upsample(x: torch.Tensor,  # shape [1, Frames, D]
                    num_samples) -> torch.Tensor:  # shape [1, Samples, D]
    frames = x.size(1)
    t_in = torch.linspace(0, 1, steps=frames, device=x.device)
    t_out = torch.linspace(0, 1, steps=num_samples, device=x.device)
    spline_fit = natural_cubic_spline_coeffs(t_in, x)
    return NaturalCubicSpline(spline_fit).evaluate(t_out)

class InvertLPC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, A, zi):
        B, T = y.shape
        N = A.shape[2]

        if zi is not None:
            initial = zi.flip(dims=[1])
        else:
            initial = y.new_zeros(B, N)

        y_padded = torch.cat([initial, y], dim=1)
        x = y.clone()

        # Precompute all shifted versions in one go
        shifts = torch.stack([y_padded[:, N - k:N - k + T] for k in range(1, N + 1)], dim=2)
        x += (A * shifts).sum(dim=2)

        ctx.save_for_backward(A, shifts, y_padded)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        A, shifts, y_padded = ctx.saved_tensors
        B, T, N = A.shape
        grad_y = grad_A = grad_zi = None

        # Gradient for y (input signal)
        if ctx.needs_input_grad[0]:
            grad_y = grad_output.clone()
            for k in range(1, N + 1):
                grad_shifted = F.pad(grad_output, (k, 0))[:, :-k] * A[:, :, k - 1]
                grad_y += grad_shifted

        # Gradient for A (coefficients)
        if ctx.needs_input_grad[1]:
            grad_A = grad_output.unsqueeze(2) * shifts

        # Gradient for zi (initial conditions)
        if ctx.needs_input_grad[2]:
            grad_zi = torch.zeros_like(y_padded[:, :N])
            for k in range(1, N + 1):
                grad_zi[:, N - k] = (grad_output[:, :k] * A[:, :k, k - 1]).sum(dim=1)
            grad_zi = grad_zi.flip(dims=[1])

        return grad_y, grad_A, grad_zi


def invert_lpc(y: torch.Tensor, A: torch.Tensor, zi: torch.Tensor = None) -> torch.Tensor:
    return InvertLPC.apply(y, A, zi)

class DiffKS(nn.Module):
    """
    A differentiable Karplus–Strong model with time-varying fractional delay
    and configurable-order filter with normalized coefficients.
    """

    def __init__(
        self,
        burst: torch.Tensor,
        loop_n_frames: int,
        sample_rate: int,
        min_f0_hz: float,
        loop_order: int = 5,
        init_loop_b_frames: Optional[torch.Tensor] = None,
        init_loop_b_range: Tuple[float, float] = (-2, 0),
        gain: Optional[float] = None,  # <--- None => learnable; non-None => fixed
        exc_order: int = 5,
        exc_n_frames: int = 100,
        exc_requires_grad : bool = True,
        interp_type: str = "linear",
        use_double_precision: bool = False,
    ):
        super().__init__()
        assert loop_order >= 1, "Filter order must be at least 1"

        self._dtype = torch.float64 if use_double_precision else torch.float32

        self.loop_n_frames = loop_n_frames
        self.exc_n_frames = exc_n_frames

        self.sample_rate = sample_rate

        self.min_f0_hz = min_f0_hz
        self.loop_order = loop_order
        self.loop_n_coefficients = loop_order + 1 # To account for DC coefficient

        self.lagrange_denom = torch.arange(LAGRANGE_ORDER, -1, -1).view(-1, 1) - torch.arange(LAGRANGE_ORDER + 1).view(1, -1)
        self.lagrange_mask = self.lagrange_denom != 0
        self.lagrange_denom = torch.where(self.lagrange_mask, self.lagrange_denom, 1)

        self.interp_type = interp_type

        self.excitation_filter_order = exc_order
        self.exc_requires_grad = exc_requires_grad

        if gain is None:
            self.raw_gain = nn.Parameter(torch.tensor(0.0, dtype=self._dtype))
        else:
            clamped = float(torch.clamp(torch.tensor(gain, dtype=self._dtype), 1e-6, 1 - 1e-6))
            self.register_buffer("raw_gain", torch.logit(torch.tensor(clamped, dtype=self._dtype)))

        if init_loop_b_frames is None:
            raw_init = torch.empty(self.loop_n_frames, self.loop_n_coefficients, dtype=self._dtype).uniform_(*init_loop_b_range)
        else:
            raw_init = torch.special.logit(init_loop_b_frames).to(self._dtype)

        self.raw_coeff_frames = nn.Parameter(raw_init)

        self.register_buffer("excitation", burst.to(self._dtype))
        self.exc_coefficients = nn.Parameter(torch.zeros(self.exc_n_frames, self.excitation_filter_order, dtype=self._dtype))

        self.register_buffer("excitation_filter_out", burst.to(self._dtype)) # Buffer to store the excitation filter out in the last training run
        self.register_buffer("inverse_filtered_signal", torch.zeros(self.sample_rate))

    @property
    def interp_type(self):
        return self._interp_type

    @interp_type.setter
    def interp_type(self, value: str):
        assert value in ["linear", "allpass", "lagrange"], "Invalid interpolation type"
        self._interp_type = value
        if value == "linear":
            self.num_active_indexes = self.loop_n_coefficients + 1
        elif value == "allpass":
            self.num_active_indexes = self.loop_n_coefficients + 1
        elif value == "lagrange":
            self.num_active_indexes = self.loop_n_coefficients + LAGRANGE_ORDER

        self.coeff_vector_size = int(self.sample_rate // self.min_f0_hz) + self.num_active_indexes

    def forward(self,
                delay_len_frames: torch.Tensor,
                n_samples: int,
                target: torch.Tensor = None,) -> torch.Tensor:
        delay_interp, coeff_interp, exc_filt_interp = self.get_upsampled_parameters(delay_len_frames, n_samples)

        A = torch.zeros((1, n_samples, self.coeff_vector_size), device=self.excitation.device, dtype=self._dtype)
        b = coeff_interp  # shape: (n_samples, loop_n_coefficients)

        omega = 2 * torch.pi / delay_interp
        zs = torch.exp(1j * omega.view(-1, 1)) ** -torch.arange(self.loop_n_coefficients, device=self.excitation.device).view(1, -1)
        p_a = -torch.angle(torch.sum(b * zs, dim=-1, keepdim=False)) / omega

        delay_interp = delay_interp - (1 + p_a)

        z_l = torch.floor(delay_interp).long()
        alfa = delay_interp - z_l

        idxs = [z_l + i for i in range(self.num_active_indexes)]
        assert torch.all(idxs[-1] < self.coeff_vector_size), "Delay index exceeds the buffer size"


        x = torch.zeros(n_samples, device=self.excitation.device)
        x[: self.excitation.shape[0]] = self.excitation

        if self.interp_type == "linear":
            # First term: z^L
            A[0, torch.arange(n_samples), idxs[0]] = -(1 - alfa) * b[:, 0]

            # Middle terms with crossfade
            for i in range(1, self.loop_n_coefficients):
                A[0, torch.arange(n_samples), idxs[i]] = -(alfa * b[:, i - 1] + (1 - alfa) * b[:, i])

            # Last term: z^(L + loop_order)
            A[0, torch.arange(n_samples), idxs[-1]] = -alfa * b[:, -1]
        elif self.interp_type == "allpass":
            C = (1 - alfa) / (1 + alfa)
            x = torch.cat([x[0:1], x[1:] + x[:-1] * C[1:]])

            A[0, torch.arange(n_samples), 0] = C

            A[0, torch.arange(n_samples), idxs[0]] = -C * b[:, 0]

            for i in range(1, self.loop_n_coefficients):
                A[0, torch.arange(n_samples), idxs[i]] = -(b[:, i - 1] + C * b[:, i])

            A[0, torch.arange(n_samples), idxs[-1]] = -b[:, -1]
        elif self.interp_type == "lagrange":
            lagrange_coeffs = alfa.view(-1, 1) - torch.arange(LAGRANGE_ORDER + 1, device=delay_interp.device, dtype=self._dtype).view(1, -1)

            lagrange_denom = self.lagrange_denom.to(delay_interp.device).unsqueeze(0)
            lagrange_coeffs = lagrange_coeffs.unsqueeze(1)

            lagrange_coeffs = torch.where(self.lagrange_mask, lagrange_coeffs, 1)

            lagrange_coeffs = lagrange_coeffs / lagrange_denom
            lagrange_coeffs = lagrange_coeffs.prod(dim=-1)

            b = torch.nn.functional.conv1d(b.unsqueeze(0), lagrange_coeffs.unsqueeze(1), padding=LAGRANGE_ORDER, groups=len(b)).squeeze(0)

            for i, idx in enumerate(idxs):
                A[0, torch.arange(n_samples), idx] = -b[:, i]

        else:
            raise NotImplementedError

        burst_length = self.excitation.shape[0]
        burst_only = x[:burst_length]

        if self.exc_requires_grad:
            a_in = exc_filt_interp.unsqueeze(0)
            filtered_burst = sample_wise_lpc(burst_only.unsqueeze(0), a_in)
        else:
            filtered_burst = burst_only

        # Now zero-pad the filtered result to full length
        x = torch.zeros(n_samples, device=self.excitation.device)
        x[:burst_length] = filtered_burst
        x = x.unsqueeze(0)

        self.excitation_filter_out = x

        # Now we obtain plucking signal through inverse filtering of the
        # predicted filters:
        if target is not None:
            y_target = target.squeeze(0)

            # inversed loop filter
            x_est = invert_lpc(y_target, A)
            x_est = x_est[:, :burst_length] # cut to make sure we conserve only plucking

            pluck_est = invert_lpc(x_est, a_in)
            self.inverse_filtered_signal = pluck_est.squeeze() # Store for plotting

            # Refiltering
            x_refiltered = sample_wise_lpc(pluck_est, a_in)

            x = torch.zeros(n_samples, device=self.excitation.device) # Pad to total length
            x[:burst_length] = x_refiltered

            y_out = sample_wise_lpc(x.unsqueeze(0), A)
        else:
            y_out = sample_wise_lpc(x, A)

        return y_out.squeeze(0).to(torch.float32)

    def get_inverse_filtered_signal(self):
        return self.inverse_filtered_signal

    def get_excitation_filter_out(self):
        return self.excitation_filter_out

    def get_gain(self):
        return torch.sigmoid(self.raw_gain)

    def get_constrained_coefficients(self, for_plotting: bool = False) -> torch.Tensor:
        """
        Generate coefficient trajectories over time using a 1-layer linear network.

        Each coefficient is modeled as: coeff(t) = weight * t + bias
        with optional normalization so the filter remains stable.
        """
        def compute_coeffs():
            sigmoid_b = torch.sigmoid(self.raw_coeff_frames)
            sum_b = sigmoid_b.sum(dim=-1, keepdim=True)
            return (sigmoid_b / sum_b) * self.get_gain()

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Upsample:
          1) delay_len_frames -> delay_interp ( shape: [f0_frames] )
          2) self.raw_coeff_frames -> coeff_interp ( shape: [loop_frames, self.loop_n_coefficients] )
          3) self.raw_excitation_filter_frames -> exc_filt_interp ( shape: [exc_frames, excitation_filter_order] )

        Returns (delay_interp, coeff_interp, exc_filt_interp).
        """
        def interpolate_fn():
            coeff_frames = self.get_constrained_coefficients(for_plotting=for_plotting)

            if self.loop_n_frames == 1:
                coeff_i = coeff_frames.repeat(num_samples, 1)
            else:
                coeff_input = coeff_frames.view(1, self.loop_n_frames, self.loop_n_coefficients).to(dtype=self._dtype)
                coeff_i = spline_upsample(coeff_input, num_samples).squeeze()

            f0_n_frames = delay_len_frames.size(0)

            if f0_n_frames == 1:
                delay_i = delay_len_frames.repeat(num_samples)
            else:
                # reshape => [1, F, D]
                delay_input = delay_len_frames.view(1, f0_n_frames, 1).to(dtype=self._dtype)
                delay_i = spline_upsample(delay_input, num_samples).squeeze()

            # Interpolation for excitation filter
            exc_filter_frames_ = self.exc_coefficients
            burst_num_samples = self.excitation.shape[0]

            if self.exc_n_frames == 1:
                exc_filt_i = exc_filter_frames_.repeat(burst_num_samples, 1)
            else:
                # shape => [1, exc_frames, exc_filter_order]
                exc_input = exc_filter_frames_.unsqueeze(0)
                exc_filt_i = spline_upsample(exc_input, burst_num_samples).squeeze(0) # => [burst_num_samples, exc_filter_order]

            return delay_i, coeff_i, exc_filt_i

        if for_plotting:
            with torch.no_grad():
                d, c, e = interpolate_fn()
                return d.detach().cpu(), c.detach().cpu(), e.detach().cpu()
        else:
            return interpolate_fn()


def noise_burst(
    sample_rate: int,
    length_s: float,
    burst_width_s: float,
    return_1D: bool = True,
    normalize: bool = False,
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

    if normalize:
        burst = burst - burst.mean()
        burst = burst / burst.abs().max()

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
