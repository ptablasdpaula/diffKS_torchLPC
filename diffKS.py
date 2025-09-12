from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlpc import sample_wise_lpc
from utils.misc import get_device
from utils.dsp import invert_lpc

LAGRANGE_ORDER = 5

class DiffKS(nn.Module):
    """
    A differentiable Karplus–Strong model with time-varying fractional delay
    and order-1 loop filter (IIR or FIR) in series with Lagrange fractional delay.
    Pure DSP kernel: no internal parameters, no resampling.
    """
    def __init__(
        self,
        interp_type: str = "lagrange",
        use_double_precision: bool = False,
        device: torch.device = get_device(),
        loop_filter_kind: str = "iir",
    ):
        super().__init__()
        assert loop_filter_kind in ("iir", "fir"), "loop_filter_kind must be 'iir' or 'fir'"
        self.loop_filter_kind = loop_filter_kind
        self.device = device
        self._dtype = torch.float64 if use_double_precision else torch.float32
        self.interp_type = interp_type

    @property
    def interp_type(self):
        return self._interp_type

    @interp_type.setter
    def interp_type(self, value: str):
        assert value in ["linear", "allpass", "lagrange"], "Invalid interpolation type"
        self._interp_type = value

    def tune_f0(
        self,
        f0: torch.Tensor,        # [B, N]
        l_b: torch.Tensor,       # [B, N, 2] taps
    ) -> torch.Tensor:
        """Phase‑correct the fractional period based on the loop filter kind.
        Returns f0 (in samples).

        Notes:
          • `c` is defined internally here (empirical alignment constant).
          • We use **phase delay** p_a(ω) = −∠H(ω)/ω for the loop contribution:
              IIR:  H_den(ω) = 1 − a1 e^{−jω}  →  p_a =  ∠H_den / ω
              FIR:  H_num(ω) = b0 + a1 e^{−jω} →  p_a = −∠H_num / ω
          • The corrected delay placed into the index arithmetic is:
              f0_corr = f0 − (c + p_a)
        """
        b0 = l_b[..., 0]
        a1 = l_b[..., 1]

        # Frequency grid at ω = 2π / f0
        omega = 2 * torch.pi / f0
        cosw = torch.cos(omega)
        sinw = torch.sin(omega)

        if self.loop_filter_kind == "iir":
            # p_a = angle(1 - a1 e^{-jω}) / ω
            real1 = 1.0 - a1 * cosw
            imag1 = a1 * sinw
            phase1 = torch.atan2(imag1, real1)
            p_a = phase1 / omega
        elif self.loop_filter_kind == "fir":
            # p_a = -angle(b0 + a1 e^{-jω}) / ω
            fir_real = b0 + a1 * cosw
            fir_imag = -a1 * sinw
            fir_phase = torch.atan2(fir_imag, fir_real)
            p_a = -fir_phase / omega
        else:
            raise NotImplementedError(f"Unknown loop_filter_kind: {self.loop_filter_kind}")

        c = 1.0  # internal alignment constant
        f0_corrected = f0 - (c + p_a)
        return f0_corrected

    def forward(
        self,
        f0: torch.Tensor,         # [B, N] in samples
        input: torch.Tensor,      # [B, N] mono waveform
        l_b: torch.Tensor,        # [B, N, 2] loop coefficients (gain and mix)
        invert: bool = False,
    ) -> torch.Tensor:
        assert f0.dim() == 2 and input.dim() == 2 and l_b.dim() == 3 and l_b.size(2) == 2
        B, N = input.shape
        assert f0.shape == (B, N), "f0 shape must match input (f0 frames must be upsampled to samples)"
        assert l_b.shape[:2] == (B, N), "loop coefficients must match input (frames must be upsampled to samples)"

        l_b = self.design_loop(l_b)
        f0 = self.tune_f0(f0=f0.to(self._dtype), l_b=l_b.to(self._dtype))

        A, x = self.compute_resonator_matrix(f0=f0, loop_coefficients=l_b, input=input.to(self._dtype))
        y = sample_wise_lpc(x, A) if not invert else invert_lpc(x, A)
        return y.to(torch.float32)

    def compute_resonator_matrix(
            self,
            f0: torch.Tensor,  # [batch_size, n_samples] phase-corrected fractional delay (in samples)
            loop_coefficients: torch.Tensor,  # [batch_size, n_samples, 2]
            input: torch.Tensor, # [batch_size, n_samples]
    ) -> Tuple[
        torch.Tensor, torch.Tensor]:  # Returns A [batch_size, n_samples, coeff_vector_size], x [batch_size, n_samples]
        """
        Computes the coefficient matrix for a resonator with **given corrected fractional delay** and order-1 loop filter (IIR or FIR) in series with the Lagrange fractional delay inside the loop.

        Args:
            f0: Phase-corrected delay (samples) [batch_size, n_samples]
            loop_coefficients: Filter coefficients [batch_size, n_samples, 2]
                (interpreted as [b0, a1])
            input: Input excitation signal [batch_size, n_samples]

        Returns:
            A: Coefficient matrix [batch_size, n_samples, coeff_vector_size]
            x: Modified excitation signal [batch_size, n_samples]
        """
        batch_size, n_samples = f0.shape

        # Dynamically compute coeff_vector_size based on max delay + interp
        max_z = torch.floor(f0).to(torch.long).max()
        if self.interp_type == "lagrange":
            coeff_vector_size = int(max_z.item()) + LAGRANGE_ORDER + 3
        else:  # linear / allpass treated as linear here for size
            coeff_vector_size = int(max_z.item()) + 3

        x = input
        b = loop_coefficients  # [batch_size, n_samples, 2]
        b0 = b[..., 0]
        a1 = b[..., 1]

        z_l = torch.floor(f0).to(dtype=torch.long)  # [B, N] int64
        alfa = f0 - z_l  # [B, N] 0 ≤ α < 1

        # Premultiply input only for IIR
        if self.loop_filter_kind == "iir":
            x_prime = x.clone()
            # n=0
            x_prime[:, 0] = x[:, 0] * (1.0 - a1[:, 0])
            # n>=1
            if n_samples > 1:
                x_prime[:, 1:] = x[:, 1:] - a1[:, 1:] * x[:, :-1]
            x = x_prime

        A = torch.zeros((batch_size, n_samples, coeff_vector_size), device=self.device, dtype=self._dtype)

        # Create indexing tensors
        batch_indices = torch.arange(batch_size, device=self.device).view(-1, 1).expand(-1, n_samples)
        sample_indices = torch.arange(n_samples, device=self.device).view(1, -1).expand(batch_size, -1)

        if self.interp_type == "linear":
            if self.loop_filter_kind == "iir":
                # IIR: place b0 at z_l and z_l+1, add AR term at 1
                A[batch_indices, sample_indices, z_l]     = -(1 - alfa) * b0
                A[batch_indices, sample_indices, z_l + 1] = -alfa * b0
                # AR(1) term
                A[batch_indices, sample_indices, 1] += -a1
            elif self.loop_filter_kind == "fir":
                # FIR: classic 2-tap crossfade
                # idxs: [L, L+1, L+2]
                A[batch_indices, sample_indices, z_l]     = -(1 - alfa) * b0
                A[batch_indices, sample_indices, z_l + 1] = -(alfa * b0 + (1 - alfa) * a1)
                A[batch_indices, sample_indices, z_l + 2] = -alfa * a1
        elif self.interp_type == "lagrange":
            # === Lagrange interpolation ===
            z_center = torch.floor(f0).to(torch.long) - (LAGRANGE_ORDER // 2)
            alfa = f0 - z_center.to(dtype=self._dtype)
            u = alfa.view(batch_size, n_samples, 1)
            j = torch.arange(LAGRANGE_ORDER + 1, device=self.device, dtype=self._dtype).view(1, 1, -1)
            num = u - j
            k = torch.arange(LAGRANGE_ORDER, -1, -1, device=self.device, dtype=self._dtype).view(-1, 1)
            j_full = torch.arange(LAGRANGE_ORDER + 1, device=self.device, dtype=self._dtype).view(1, -1)
            denom = (k - j_full).unsqueeze(0).unsqueeze(0)
            num = num.unsqueeze(-2).expand(-1, -1, LAGRANGE_ORDER + 1, -1)
            mask = (k - j_full) != 0
            mask = mask.unsqueeze(0).unsqueeze(0)
            num = torch.where(mask, num, torch.ones_like(num))
            denom = torch.where(mask, denom, torch.ones_like(denom))
            weights = (num / denom).prod(dim=-1)
            BN = batch_size * n_samples
            b0e_flat = b0.reshape(BN, 1).unsqueeze(0)
            w_flat = weights.reshape(BN, LAGRANGE_ORDER + 1).unsqueeze(1)
            if self.device.type == "mps":
                b0e_flat = b0e_flat.to(torch.float32)
                w_flat = w_flat.to(torch.float32)
                b0_w = F.conv1d(b0e_flat, w_flat, padding=LAGRANGE_ORDER, groups=BN).to(self._dtype)
            else:
                b0_w = F.conv1d(b0e_flat, w_flat, padding=LAGRANGE_ORDER, groups=BN)
            b0_w = b0_w.squeeze(0).reshape(batch_size, n_samples, -1)
            if self.loop_filter_kind == "iir":
                # IIR: just b0_w, pad to (M+2)
                b_conv_total = F.pad(b0_w, (0, 1))
                block_len = b_conv_total.size(-1)
                idxs = [z_center + i for i in range(block_len)]
                assert torch.all(idxs[-1] < coeff_vector_size), "Delay index exceeds the buffer size"
                for i in range(block_len):
                    A[batch_indices, sample_indices, idxs[i]] = -b_conv_total[..., i]
                # AR(1) term
                A[batch_indices, sample_indices, 1] += -a1
            elif self.loop_filter_kind == "fir":
                # FIR: build 2-tap result by adding +1 shifted copy scaled by a1
                b_conv_total = F.pad(b0_w, (0, 1)) + (a1.unsqueeze(-1) * F.pad(b0_w, (1, 0)))
                block_len = b_conv_total.size(-1)
                idxs = [z_center + i for i in range(block_len)]
                assert torch.all(idxs[-1] < coeff_vector_size), "Delay index exceeds the buffer size"
                for i in range(block_len):
                    A[batch_indices, sample_indices, idxs[i]] = -b_conv_total[..., i]
        else:
            raise NotImplementedError(f"Interpolation type {self.interp_type} not implemented")

        return A, x

    def design_loop(self, l_b: torch.Tensor) -> torch.Tensor:
        """Converts "gain" and "mix" into coefficient taps [b0, a1].
        l_b_logits: [B, N, 2]
        returns: [B, N, 2] taps
        """
        g = l_b[..., 0]
        p = l_b[..., 1]

        assert p.max().item() <= 1.0
        assert g.max().item() <= 1.0

        if self.loop_filter_kind == "iir":
            a1 = p
            b0 = (1.0 - a1) * g
        else:
            b0 = (1.0 - p) * g
            a1 = p * g

        taps = torch.stack([b0, a1], dim=-1)
        assert torch.all(taps.abs() <= 1.0)
        return taps