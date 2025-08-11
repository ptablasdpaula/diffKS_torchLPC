from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlpc import sample_wise_lpc
from utils.misc import get_device
from utils.dsp import kaiser_resample, invert_lpc, spline_upsample

S_NSYNTH = 4
SR_NSYNTH = 16000

LAGRANGE_ORDER = 5

class DiffKS(nn.Module):
    """
    A differentiable Karplus–Strong model with time-varying fractional delay
    and configurable-order filter with normalized coefficients.
    """
    def __init__(
        self,
        batch_size: int = 1,
        internal_sr: int = 16000,
        min_f0_hz: float = 27.5,
        loop_order: int = 2,
        loop_n_frames: int = 250,
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

        # ====== Loop Filter ============================
        self.loop_n_frames = loop_n_frames
        self.loop_order = loop_order
        self.loop_n_coefficients = loop_order + 1  # To account for DC coefficient

        self.loop_coefficients = nn.Parameter(
            torch.rand(
                batch_size,
                self.loop_n_frames,
                self.loop_n_coefficients,
                dtype=self._dtype
            ).uniform_(-2, 0))

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

    # setters for manual init
    @torch.no_grad()
    def set_loop_coefficients(self, value: torch.Tensor) -> None:
        assert value.size() == (self.batch_size, self.loop_n_frames, self.loop_n_coefficients)
        self.loop_coefficients = nn.Parameter(value.clone())

    @torch.no_grad()
    def reinit(self) -> None:
        """Reset learnable tensors to their constructor defaults.
        """
        self.loop_coefficients.uniform_(-2, 0)

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

    # ------------------- debug helper -------------------
    def _store_buffer(self, tensor: torch.Tensor, name: str, kind: str):
        """
        Save `tensor` to an attribute **only when the module is in eval mode**
        (i.e. self.training is False) to keep training‑time GPU memory low.

        The tensor is:
        - cast to float32 if needed
        - down‑sampled to a fixed length to curb size
            * kind == "sequential"  -> 250 frames  (assumes shape [B, T, ...])
            * kind == "audio"       -> 16000 samples (assumes shape [B, N] or [B, N, 1])
        - detached and moved to CPU

        Parameters
        ----------
        tensor : torch.Tensor
            Data to save.
        name : str
            Attribute name where the tensor will be stored.
        kind : str
            Either "sequential" or "audio".
        """
        if self.training:
            return  # skip during training

        if tensor is None:
            return

        # ensure float32 for compactness
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)

        if kind == "sequential":
            target_len = 250

            # Ensure shape is [B, T, C] where C may be 1
            if tensor.dim() == 2:                 # [B, T]
                tensor = tensor.unsqueeze(-1)     # [B, T, 1]

            # Now tensor is [B, T, C...] – move time to last dim
            tensor = tensor.transpose(1, -1)      # [B, C..., T]

            if tensor.shape[-1] != target_len:
                tensor = F.interpolate(
                    tensor,
                    size=target_len,
                    mode="linear",
                    align_corners=False
                )

            # Move time back to dim 1
            tensor = tensor.transpose(1, -1)      # [B, T', C...]

            # If we added a dummy channel, squeeze it out
            if tensor.dim() == 3 and tensor.shape[2] == 1:
                tensor = tensor.squeeze(-1)
        elif kind == "audio":
            target_len = SR_NSYNTH * S_NSYNTH
            if tensor.dim() == 2:               # [B, N]
                tensor = tensor.unsqueeze(1)    # [B, 1, N]
            if tensor.shape[-1] != target_len:
                tensor = F.interpolate(
                    tensor,
                    size=target_len,
                    mode="linear",
                    align_corners=False
                )
            tensor = tensor.squeeze(1)          # back to [B, N] or [B, N, 1]

        # finally detach, move to CPU and store
        setattr(self, name, tensor.detach().cpu())

    def forward(
        self,
        f0_frames: torch.Tensor,  # [batch_size, n_frames]
        input: torch.Tensor,  # [batch_size, n_samples]
        input_sr: int,
        loop_coefficients: Optional[torch.Tensor] = None,  # [batch_size, F, loop_n_coefficients]
    ) -> torch.Tensor:  # [batch_size, n_samples]

        assert f0_frames.dim() == 2, f"f0_frames must have 2 dimensions, got shape {f0_frames.shape}"
        assert input.dim() == 2, f"target must have 2 dimensions (batch, samples), got shape {input.shape}"

        l_b = loop_coefficients if loop_coefficients is not None else self.loop_coefficients
        assert l_b.shape == (self.batch_size, self.loop_n_frames, self.loop_n_coefficients)

        f0_frames = self.internal_sr / f0_frames # Convert from Hz to samples

        if input_sr != self.internal_sr: input = kaiser_resample(input, sr_in=input_sr, sr_out=self.internal_sr)

        f0, l_b = self.get_upsampled_parameters(
            f0_frames, input.size(1),
            l_b=l_b,
        )

        l_b = self.get_constrained_l_coefficients(l_b=l_b)
        A, x = self.compute_resonator_matrix(f0=f0, loop_coefficients=l_b, input=input)

        y_out = sample_wise_lpc(x, A)
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

        self._store_buffer(A, "resonator_matrix", "sequential")

        return A, x

    def get_constrained_l_coefficients(
            self,
            l_b : torch.Tensor,  # [batches, samples, loop_n_coefficients]
    ) -> torch.Tensor : # [batches, samples, loop_n_coefficients]

        gain_logits = l_b[..., :1]  # [B, N, 1]
        taps_logits = l_b[..., 1:]  # [B, N, loop_order]

        taps = torch.tanh(taps_logits).clone()                      # ∈ (-1, 1)
        gain_target = torch.sigmoid(gain_logits) * 0.1 + 0.9        # [B, N, 1] # ∈ (0.9, 1)
        tap_sum_total = 1.0 + taps.abs().sum(dim=-1, keepdim=True)  # 1 (DC) + sum |b_i|
        gain = gain_target / tap_sum_total                          # 0 < tap_sum_total * gain < 1)

        return torch.cat([torch.ones_like(gain), taps], dim=-1) * gain

    def get_upsampled_parameters(
            self,
            f0: torch.Tensor, # [batches, f_0_frames,]
            num_samples: int,
            l_b: Optional[torch.Tensor] = None, # [batches, frames, loop_n_coefficients]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        l_b = F.interpolate(
            l_b.permute(0, 2, 1),
            size=num_samples,
            mode="linear",
            align_corners=False
        ).permute(0, 2, 1)

        if f0.size(1) == 1:
            f0 = f0.expand(f0.size(0), num_samples)
        else:
            f0 = spline_upsample(
                f0.unsqueeze(-1).to(self._dtype),
                num_samples
            ).squeeze(-1)

        return f0.to(self.device), l_b.to(self.device)