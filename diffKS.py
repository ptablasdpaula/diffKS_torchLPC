from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlpc import sample_wise_lpc
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from utils import resize_tensor_dim

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
        batch_size: int = 1,
        sample_rate: int = 16000,
        min_f0_hz: float = 27.5,
        loop_order: int = 1,
        loop_n_frames: int = 1,
        exc_order: int = 5,
        exc_n_frames: int = 100,
        exc_length_s : float = 0.025,
        interp_type: str = "linear",
        use_double_precision: bool = False,
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") ,
    ):
        super().__init__()
        assert loop_order >= 1, "Filter order must be at least 1"

        # ====== General ================================
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.device = device
        self._dtype = torch.float64 if use_double_precision else torch.float32
        self.min_f0_hz = min_f0_hz

        # ====== Excitation Filter ======================
        self.exc_n_frames = exc_n_frames
        self.exc_order = exc_order
        self.exc_length_n = int (exc_length_s * sample_rate)
        self.exc_coefficients = nn.Parameter(torch.rand(batch_size, self.exc_n_frames, self.exc_order, dtype=self._dtype) * 1e-4)

        # ====== Loop Filter ============================
        self.loop_n_frames = loop_n_frames
        self.loop_order = loop_order
        self.loop_n_coefficients = loop_order + 1  # To account for DC coefficient
        self.loop_coefficients = torch.rand(batch_size, self.loop_n_frames, self.loop_n_coefficients,
                                                 dtype=self._dtype).uniform_(-2, 0)
        self.loop_gain = nn.Parameter(torch.rand(batch_size, loop_n_frames, 1, dtype=self._dtype))

        # ====== Interpolation Settings ==================
        self.interp_type = interp_type

        self.lagrange_denom = torch.arange(LAGRANGE_ORDER, -1, -1).view(-1, 1) - torch.arange(LAGRANGE_ORDER + 1).view(1, -1)
        self.lagrange_mask = self.lagrange_denom != 0
        self.lagrange_denom = torch.where(self.lagrange_mask, self.lagrange_denom, 1)

        # ====== Analysis Buffers =======================
        self.register_buffer("excitation_filter_out", torch.empty(batch_size, self.exc_length_n))
        self.register_buffer("ks_inverse_signal", torch.zeros(batch_size, self.exc_length_n))

        # ====== METADATA table for inner shapes (no batch)
        self._param_meta: dict[str, Tuple[Tuple[int, ...], str]] = {
            "exc_coefficients": ((self.exc_n_frames, self.exc_order), "Parameter"),
            "loop_coefficients": ((self.loop_n_frames, self.loop_n_coefficients), "Parameter"),
            "loop_gain": ((self.loop_n_frames, 1), "Parameter"),
        }

    def _expect(self, tensor: torch.Tensor, name: str, shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Validate *shape*, then cast to model dtype / device if necessary."""
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name}: expected shape {shape}, got {tuple(tensor.shape)}")
        return tensor.to(dtype=self._dtype, device=self.device)

    def _prepare(self, name: str, new_value: Optional[torch.Tensor], *, inplace: bool = False,
    ) -> torch.Tensor:
        """Common path used by both setters and forward.

        Parameters
        ----------
        name : str -> One of the keys of ``_param_meta``.
        new_value : Tensor | None
            A tensor supplied externally **or** None to indicate that the stored
            Parameter should be used.
        inplace : bool, default = False
            If *True*, copy the validated value into the Parameter under
            ``torch.no_grad()`` (used by setters).
        """
        inner_shape, _ = self._param_meta[name]
        full_shape = (self.batch_size, *inner_shape)

        if new_value is None:
            return getattr(self, name)  # use the Parameter as‑is

        value = self._expect(new_value, name, full_shape)

        if inplace:
            with torch.no_grad():
                getattr(self, name).data.copy_(value)
        return value

    # setters for manual init
    @torch.no_grad()
    def set_exc_coefficients(self, value: torch.Tensor) -> None:
        self._prepare("exc_coefficients", value, inplace=True)

    @torch.no_grad()
    def set_loop_coefficients(self, value: torch.Tensor) -> None:
        self._prepare("loop_coefficients", value, inplace=True)

    @torch.no_grad()
    def set_loop_gain(self, value: torch.Tensor) -> None:
        self._prepare("loop_gain", value, inplace=True)

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
                f0_frames: torch.Tensor,  # [batch_size, n_frames]
                input: torch.Tensor,  # [batch_size, n_samples]
                direct: bool = False,
                loop_coefficients: Optional[torch.Tensor] = None,  # [batch_size, loop_n_frames, loop_n_coefficients]
                loop_gain: Optional[torch.Tensor] = None,  # [batch_size, loop_n_frames, 1]
                exc_coefficients: Optional[torch.Tensor] = None,  # [batch_size, exc_n_frames, exc_order]
                ) -> torch.Tensor:  # [batch_size, n_samples]

        assert f0_frames.dim() == 2, f"f0_frames must have 2 dimensions, got shape {f0_frames.shape}"
        assert input.dim() == 2, f"target must have 2 dimensions (batch, samples), got shape {input.shape}"

        l_b = self._prepare("loop_coefficients", loop_coefficients)
        l_g = self._prepare("loop_gain", loop_gain)
        exc_b = self._prepare("exc_coefficients", exc_coefficients)

        n_samples = input.size(1)

        l_b_constrained = self.get_constrained_coefficients(l_b=l_b, l_g=l_g)
        f0, l_b, exc_b = self.get_upsampled_parameters(f0_frames, n_samples,
                                                       l_b=l_b_constrained,
                                                       exc_b=exc_b)
        A, x = self.compute_resonator_matrix(f0=f0,
                                             loop_coefficients=l_b,
                                             input=input)

        loop_inv = invert_lpc(x, A)

        ks_inv_signal = invert_lpc(resize_tensor_dim(loop_inv, self.exc_length_n, 1),
                                   exc_b)
        self.ks_inverse_signal = ks_inv_signal

        exc_filter_out = sample_wise_lpc(ks_inv_signal if direct is False else resize_tensor_dim(x, self.exc_length_n, 1),
                                         exc_b)
        self.exc_filter_out = exc_filter_out

        y_out = sample_wise_lpc(resize_tensor_dim(exc_filter_out, n_samples, 1),
                                A)

        return y_out.to(torch.float32)

    def compute_resonator_matrix(
            self,
            f0: torch.Tensor,  # [batch_size, n_samples]
            loop_coefficients: torch.Tensor,  # [batch_size, n_samples, loop_n_coefficients]
            input: torch.Tensor, # [batch_size, n_samples]
    ) -> Tuple[
        torch.Tensor, torch.Tensor]:  # Returns A [batch_size, n_samples, coeff_vector_size], x [batch_size, n_samples]
        """
        Computes the coefficient matrix for a resonator with fractional delay.

        Args:
            f0: Fundamental frequency in samples [batch_size, n_samples]
            loop_coefficients: Filter coefficients [batch_size, n_samples, loop_n_coefficients]
            input: Input excitation signal [batch_size, n_samples]

        Returns:
            A: Coefficient matrix [batch_size, n_samples, coeff_vector_size]
            x: Modified excitation signal [batch_size, n_samples]
        """
        assert f0.dim() == 2, f"f0 must have 2 dimensions, got shape {f0.shape}"
        assert loop_coefficients.dim() == 3, f"loop_coefficients must have 3 dimensions, got shape {loop_coefficients.shape}"

        batch_size, n_samples = f0.shape
        assert loop_coefficients.size(
            0) == batch_size, f"Batch size mismatch: f0 has {batch_size}, loop_coefficients has {loop_coefficients.size(0)}"
        assert loop_coefficients.size(
            1) == n_samples, f"Sample count mismatch: f0 has {n_samples}, loop_coefficients has {loop_coefficients.size(1)}"
        assert loop_coefficients.size(
            2) == self.loop_n_coefficients, f"Coefficient count mismatch: expected {self.loop_n_coefficients}, got {loop_coefficients.size(2)}"
        assert input.size(1) == n_samples, f"Input sample count mismatch: expected {n_samples}, got {input.size(1)}"

        x = input
        b = loop_coefficients  # [batch_size, n_samples, loop_n_coefficients]

        # Calculate phase adjustment
        omega = 2 * torch.pi / f0  # [batch_size, n_samples]
        coeff_range = torch.arange(self.loop_n_coefficients, device=self.device).view(1, 1, -1)
        zs = torch.exp(1j * omega.view(batch_size, n_samples, 1)) ** -coeff_range
        p_a = -torch.angle(torch.sum(b * zs, dim=-1)) / omega
        f0_corrected = f0 - (1 + p_a)

        z_l = torch.floor(f0_corrected).long()  # [batch_size, n_samples]
        alfa = f0_corrected - z_l  # [batch_size, n_samples]

        max_delay_idx = z_l + self.num_active_indexes - 1
        assert torch.all(max_delay_idx < self.coeff_vector_size), "Delay index exceeds the buffer size"

        A = torch.zeros((batch_size, n_samples, self.coeff_vector_size), device=self.device, dtype=self._dtype)

        # Create indexing tensors
        batch_indices = torch.arange(batch_size, device=self.device).view(-1, 1).expand(-1, n_samples)
        sample_indices = torch.arange(n_samples, device=self.device).view(1, -1).expand(batch_size, -1)

        if self.interp_type == "linear":
            indices = z_l
            A[batch_indices, sample_indices, indices] = -(1 - alfa) * b[..., 0]

            for i in range(1, self.loop_n_coefficients):
                indices = z_l + i
                A[batch_indices, sample_indices, indices] = -(alfa * b[..., i - 1] + (1 - alfa) * b[..., i])

            indices = z_l + self.num_active_indexes - 1
            A[batch_indices, sample_indices, indices] = -alfa * b[..., -1]

        elif self.interp_type == "allpass":
            C = (1 - alfa) / (1 + alfa)

            x_processed = torch.zeros_like(x)
            x_processed[:, 0] = x[:, 0]
            x_processed[:, 1:] = x[:, 1:] + x[:, :-1] * C[:, 1:]
            x = x_processed

            A[:, :, 0] = C

            indices = z_l
            A[batch_indices, sample_indices, indices] = -C * b[..., 0]

            for i in range(1, self.loop_n_coefficients):
                indices = z_l + i
                A[batch_indices, sample_indices, indices] = -(b[..., i - 1] + C * b[..., i])

            indices = z_l + self.num_active_indexes - 1
            A[batch_indices, sample_indices, indices] = -b[..., -1]

        elif self.interp_type == "lagrange":
            lag_k = self._lagrange_kernel(alfa)  # (B,N,L+1)
            B, N, M = b.shape

            if b.is_cuda:  # depth‑wise conv
                inp = b.transpose(1, 2).reshape(1, B * N, M)
                kern = lag_k.reshape(B * N, 1, LAGRANGE_ORDER + 1)

                b_processed = F.conv1d(
                    inp, kern, padding=LAGRANGE_ORDER, groups=B * N
                ).reshape(B, N, -1)  # (B,N,M+L)

            else:  # vectorised einsum
                L = LAGRANGE_ORDER  # = 5
                b_unf = F.pad(b, (L, L))  # (B,N,K+2L)   pad left *and* right
                b_unf = b_unf.unfold(-1, L + 1, 1)  # (B,N,K+2L,L+1)
                b_processed = torch.einsum('bnml,bnl->bnm', b_unf, lag_k)

            base_idx = z_l.unsqueeze(-1) + torch.arange(
                self.num_active_indexes, device=self.device
            )

            A[torch.arange(B).view(-1, 1, 1),
            torch.arange(N).view(1, -1, 1),
            base_idx] = -b_processed

        else:
            raise NotImplementedError(f"Interpolation type {self.interp_type} not implemented")

        return A, x

    def get_gain(self,
                 l_g : torch.Tensor = None,): # [batches, loop_n_frames, 1]
        return torch.sigmoid(l_g if l_g is not None else self.loop_gain)

    def get_constrained_coefficients(self,
                                     l_b : Optional[torch.Tensor] = None, # [batches, loop_n_frames, loop_n_coefficients]
                                     l_g : Optional[torch.Tensor] = None, # [batches, loop_n_frames, 1]
                                     ) -> torch.Tensor:
        sigmoid_b = torch.sigmoid(l_b if l_b is not None else self.loop_coefficients)
        sum_b = sigmoid_b.sum(dim=-1, keepdim=True)
        return (sigmoid_b / sum_b) * (self.get_gain(l_g if l_g is not None else None))

    def get_upsampled_parameters(
            self,
            f0: torch.Tensor, # [batches, f_0_frames,]
            num_samples: int,
            l_b: Optional[torch.Tensor] = None, # [batches, loop_n_frames, loop_n_coefficients]
            exc_b: Optional[torch.Tensor] = None, # [batches, exc_n_frames, exc_order]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        loop_b_frames_ = l_b if l_b is not None else self.loop_coefficients
        exc_b_frames_ = exc_b if exc_b is not None else self.exc_coefficients

        batch_size = f0.size(0)
        f0_n_frames = f0.size(1)
        exc_length_n = self.exc_length_n

        if f0_n_frames == 1:
            f0_i = f0.expand(batch_size, num_samples)
        else:
            f0_reshaped = f0.unsqueeze(-1).to(dtype=self._dtype)
            f0_i = spline_upsample(f0_reshaped, num_samples).squeeze(-1)

        if self.loop_n_frames == 1:
            loop_b_i = loop_b_frames_.repeat(1, num_samples, 1)
        else:
            loop_b_i = spline_upsample(loop_b_frames_.to(dtype=self._dtype), num_samples)

        if self.exc_n_frames == 1:
            exc_b_i = exc_b_frames_.repeat(1, exc_length_n, 1)
        else:
            exc_b_i = spline_upsample(exc_b_frames_.to(dtype=self._dtype), exc_length_n)

        return f0_i, loop_b_i, exc_b_i

    def get_inverse_filtered_signal(self):
        return self.ks_inverse_signal

    def get_excitation_filter_out(self):
        return self.excitation_filter_out

    # --- helper: vectorised Lagrange kernel ---------------------------------------
    def _lagrange_kernel(self, alfa: torch.Tensor) -> torch.Tensor:
        """
        alfa : [B, N]  fractional part (0…1)
        returns a per–sample kernel of shape [B, N, L+1]  (here L = 5)
        """
        # α − i   for i = 0…L
        k = torch.arange(LAGRANGE_ORDER + 1,
                         device=alfa.device,
                         dtype=self._dtype)  # (L+1,)
        diff = alfa.unsqueeze(-1) - k  # (B,N,L+1)

        # numerator / denominator,  mask the diagonals that would be “0/0”
        num = diff.unsqueeze(-2)  # (B,N,1,L+1)
        denom = self.lagrange_denom.to(alfa.device)  # (L+1,L+1)
        kernel = torch.where(self.lagrange_mask, num, 1.) / denom  # (B,N,L+1,L+1)

        # Π over the last axis → per‑sample kernel length L+1
        return kernel.prod(-1)  # (B,N,L+1)
