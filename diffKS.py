from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlpc import sample_wise_lpc
from utils.misc import get_device
from utils.dsp import kaiser_resample, invert_lpc, spline_upsample

LAGRANGE_ORDER = 5

class DiffKS(nn.Module):
    """
    A differentiable Karplus–Strong model with time-varying fractional delay
    and configurable-order filter with normalized coefficients.
    """

    def __init__(
        self,
        batch_size: int = 1,
        internal_sr: int = 44100,
        min_f0_hz: float = 27.5,
        loop_order: int = 1,
        exc_order: int = 5,
        interp_type: str = "linear",
        use_double_precision: bool = False,
        device: torch.device = get_device(),
    ):
        super().__init__()
        assert loop_order >= 1, "Filter order must be at least 1"

        # ====== General ================================
        self.batch_size = batch_size
        self.internal_sr = internal_sr
        self.device = device
        self._dtype = torch.float64 if use_double_precision else torch.float32
        self.min_f0_hz = min_f0_hz

        # ====== Excitation Filter ======================
        self.exc_order = exc_order
        self.exc_n_coefficients = exc_order + 1 # To account for exc_g
        self.exc_coefficients = nn.Parameter(torch.rand(batch_size, 1, self.exc_n_coefficients, dtype=self._dtype))

        # ====== Loop Filter ============================
        self.loop_order = loop_order
        self.loop_n_coefficients = loop_order + 1  # To account for DC coefficient
        self.loop_coefficients = nn.Parameter(torch.rand(batch_size, 1, self.loop_n_coefficients,
                                                 dtype=self._dtype).uniform_(-2, 0))
        self.loop_gain = nn.Parameter(torch.rand(batch_size, 1, 1, dtype=self._dtype))

        # ====== Interpolation Settings ==================
        self.interp_type = interp_type

        self.register_buffer("lagrange_denom",
                             torch.arange(LAGRANGE_ORDER, -1, -1).view(-1, 1) -
                             torch.arange(LAGRANGE_ORDER + 1).view(1, -1))
        self.register_buffer("lagrange_mask", self.lagrange_denom != 0)
        self.register_buffer("lagrange_denom_masked",
                             torch.where(self.lagrange_mask, self.lagrange_denom, 1))

        # ====== Analysis Buffers =======================
        self.register_buffer("excitation", None, persistent=False)
        self.register_buffer("resonator_matrix", None, persistent=False)

        # ====== METADATA table for inner shapes (no batch)
        self._param_meta = {
            "exc_coefficients":  ((None, exc_order + 1),  "Parameter"),  # None = any F
            "loop_coefficients": ((None, loop_order + 1), "Parameter"),
            "loop_gain":         ((None, 1),              "Parameter"),
        }

    def _expect(self, tensor: torch.Tensor, name: str, shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Validate *shape*, then cast to model dtype / device if necessary."""
        if tensor.shape[0] != shape[0]:
            raise ValueError(f"{name}: expected first dim {shape[0]}, got {tensor.shape[0]} (batch mismatch)")
        if tensor.shape[-1] != shape[-1]:
            raise ValueError(f"{name}: expected last dim {shape[-1]}, got {tensor.shape[-1]} (order mismatch)")
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
                param = getattr(self, name)
                if param.shape != value.shape:
                    # replace by a fresh Parameter of the right shape
                    setattr(self, name, nn.Parameter(value.clone()))
                else:
                    param.data.copy_(value)

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

    @torch.no_grad()
    def reinit(self) -> None:
        """Reset learnable tensors to their constructor defaults.
        """
        self.loop_coefficients.uniform_(-2, 0)
        self.loop_gain.uniform_(0, 1)
        self.exc_coefficients.uniform_(0, 1)

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

        self.coeff_vector_size = int(self.internal_sr // self.min_f0_hz) + self.num_active_indexes

    def forward(self,
                f0_frames: torch.Tensor,  # [batch_size, n_frames]
                input: torch.Tensor,  # [batch_size, n_samples]
                input_sr: int,
                direct: bool = False,
                loop_coefficients: Optional[torch.Tensor] = None,  # [batch_size, F, loop_n_coefficients]
                loop_gain: Optional[torch.Tensor] = None,  # [batch_size, F, 1]
                exc_coefficients: Optional[torch.Tensor] = None,  # [batch_size, F, exc_order]
                ) -> torch.Tensor:  # [batch_size, n_samples]

        assert f0_frames.dim() == 2, f"f0_frames must have 2 dimensions, got shape {f0_frames.shape}"
        assert input.dim() == 2, f"target must have 2 dimensions (batch, samples), got shape {input.shape}"

        l_b = self._prepare("loop_coefficients", loop_coefficients)
        l_g = self._prepare("loop_gain", loop_gain)
        exc_b = self._prepare("exc_coefficients", exc_coefficients)

        f0_frames = self.internal_sr / f0_frames # Convert from Hz to samples

        if input_sr != self.internal_sr:
            input = kaiser_resample(input, sr_in=input_sr, sr_out=self.internal_sr)

        n_samples = input.size(1)

        f0, l_b, l_g, exc_b = self.get_upsampled_parameters(
            f0_frames, n_samples,
            l_b=l_b, l_g=l_g, exc_b=exc_b,
        )

        l_b = self.get_constrained_l_coefficients(l_b=l_b, l_g=l_g)
        exc_b = self.get_constrained_exc_coefficients(exc_b=exc_b)

        A, x = self.compute_resonator_matrix(f0=f0,
                                             loop_coefficients=l_b,
                                             input=input)

        if not direct:
            loop_inv = invert_lpc(x, A)
            excitation = invert_lpc(loop_inv, exc_b)
            self.excitation = excitation

        exc_filter_out = sample_wise_lpc(excitation if not direct else x, exc_b)

        self.exc_filter_out = exc_filter_out

        y_out = sample_wise_lpc(exc_filter_out, A)

        return kaiser_resample(y_out, sr_in=self.internal_sr, sr_out=input_sr).to(torch.float32)

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
        batch_size, n_samples = f0.shape

        x = input
        b = loop_coefficients  # [batch_size, n_samples, loop_n_coefficients]

        # Calculate phase adjustment
        omega = 2 * torch.pi / f0  # [batch_size, n_samples]
        coeff_range = torch.arange(self.loop_n_coefficients, device=self.device).view(1, 1, -1)

        if self.device == torch.device("mps"):
            cos_k = torch.cos(omega.unsqueeze(-1) * coeff_range)
            sin_k = torch.sin(omega.unsqueeze(-1) * coeff_range)
            real_sum = (b * cos_k).sum(dim=-1)
            imag_sum = -(b * sin_k).sum(dim=-1)
            phase = torch.atan2(imag_sum, real_sum)
            p_a = -phase / omega
        else:
            zs = torch.exp(1j * omega.view(batch_size, n_samples, 1)) ** -coeff_range
            p_a = -torch.angle(torch.sum(b * zs, dim=-1)) / omega

        f0_corrected = f0 - (1 + p_a)

        max_int_delay = self.coeff_vector_size - self.num_active_indexes

        needs_clamp = (f0_corrected < 0.0) | (f0_corrected > max_int_delay - 1e-6)
        f0_corrected = f0_corrected.clamp_(min=0.0, max=max_int_delay - 1e-6)

        if torch.any(needs_clamp):
            print("[KS] some delays were clamped to the valid range "
                  f"[0, {max_int_delay - 1e-6:g}]")

        z_l = torch.floor(f0_corrected).to(dtype=torch.long)  # [B, N] int64
        alfa = f0_corrected - z_l  # [B, N] 0 ≤ α < 1

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
        else:
            raise NotImplementedError(f"Interpolation type {self.interp_type} not implemented")

        self.resonator_matrix = A

        return A, x

    def get_constrained_exc_coefficients(
            self,
            exc_b: Optional[torch.Tensor] = None  # [B, samples, exc_order+1]
    ) -> torch.Tensor:  # [B, samples, exc_order]
        """
        The first slot of *exc_b* is interpreted as a raw gain term (``exc_g``).
        After a sigmoid, that gain scales every AR coefficient, while the implicit
        DC coefficient that `torchlpc.sample_wise_lpc` assumes stays at **+1**.
        """
        raw = exc_b if exc_b is not None else self.exc_coefficients  # [B,F,O+1]

        exc_g_raw = raw[..., :1]  # shape [B, samples, 1]  –– gain parameter
        exc_b_raw = raw[..., 1:]  # shape [B, samples, exc_order]  –– AR coeffs

        exc_g = torch.sigmoid(exc_g_raw)  # (0‥1)

        exc_b = torch.sigmoid(exc_b_raw)  # (0‥exc_g)
        sum_exc = exc_b.sum(dim=-1, keepdim=True)
        exc_b = (exc_b / sum_exc)

        return exc_b * exc_g

    def get_gain(self,
                 l_g : torch.Tensor): # [batches, samples, 1]
        return torch.sigmoid(l_g)

    def get_constrained_l_coefficients(self,
                                          l_b : torch.Tensor,  # [batches, samples, loop_n_coefficients]
                                          l_g : torch.Tensor,  # [batches, samples, 1]
                                          ) -> torch.Tensor:
        sigmoid_b = torch.sigmoid(l_b)
        sum_b = sigmoid_b.sum(dim=-1, keepdim=True)
        result = (sigmoid_b / sum_b) * (self.get_gain(l_g))
        return result.to(self.device)

    def get_upsampled_parameters(
            self,
            f0: torch.Tensor, # [batches, f_0_frames,]
            num_samples: int,
            l_b: Optional[torch.Tensor] = None, # [batches, frames, loop_n_coefficients]
            l_g: Optional[torch.Tensor] = None, # [batches, frames, 1]
            exc_b: Optional[torch.Tensor] = None, # [batches, frames, exc_order]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # ---------- coefficients -------------------------------------------
        l_b = l_b if l_b is not None else self.loop_coefficients
        l_g = l_g if l_g is not None else self.loop_gain
        exc_b = exc_b if exc_b is not None else self.exc_coefficients

        def upsample_coeff(frames: torch.Tensor) -> torch.Tensor:
            # frames: [B, T, D] → permute → interp → permute back
            return (
                F.interpolate(
                    frames.permute(0, 2, 1),
                    size=num_samples,
                    mode="linear",
                    align_corners=False
                )
                .permute(0, 2, 1)
            )

        l_b = upsample_coeff(l_b)
        l_g = upsample_coeff(l_g)
        exc_b = upsample_coeff(exc_b)

        # ---------- F0  -----------------------------------------------------
        B = f0.size(0)

        if f0.size(1) == 1:
            f0_i = f0.expand(B, num_samples)
        else:
            f0_i = (
                spline_upsample(f0.unsqueeze(-1).to(self._dtype), num_samples)
                .squeeze(-1)
            )

        return f0_i.to(self.device), l_b.to(self.device), l_g.to(self.device), exc_b.to(self.device)