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
    and order-1 loop filter (IIR or FIR) in series with Lagrange fractional delay.
    """
    def __init__(
        self,
        batch_size: int = 1,
        internal_sr: int = 16000,
        min_f0_hz: float = 27.5,
        loop_order: int = 1,
        loop_n_frames: int = 250,
        interp_type: str = "linear",
        use_double_precision: bool = True,
        device: torch.device = get_device(),
        loop_filter_kind: str = "fir",
    ):
        super().__init__()
        assert loop_order == 1, "This implementation supports only order-1 loop filters."
        assert loop_filter_kind in ("iir", "fir"), "loop_filter_kind must be 'iir' or 'fir'"
        self.loop_filter_kind = loop_filter_kind

        # ====== General ================================
        self.batch_size = batch_size
        self.internal_sr = internal_sr
        self.device = device
        self._dtype = torch.float64 if use_double_precision else torch.float32
        self.min_f0_hz = min_f0_hz

        # ====== Loop Filter ============================
        self.loop_n_frames = loop_n_frames
        self.loop_order = loop_order
        self.loop_n_coefficients = loop_order + 1  # gain and first tap

        self.loop_coefficients = nn.Parameter(
            torch.rand(
                batch_size,
                self.loop_n_frames,
                self.loop_n_coefficients,
                dtype=self._dtype
            ).uniform_(-2, 0))

        # ====== Interpolation Settings ==================
        self.interp_type = interp_type

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

    def tune_f0(
        self,
        f0: torch.Tensor,        # [B, N] in samples/period
        b0: torch.Tensor,        # [B, N]
        a1: torch.Tensor,        # [B, N]
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
        f0_frames: torch.Tensor,  # [batch_size, n_frames]
        input: torch.Tensor,  # [batch_size, n_samples]
        input_sr: int,
        loop_coefficients: Optional[torch.Tensor] = None,  # [batch_size, F, loop_n_coefficients]
        invert = False,
    ) -> torch.Tensor:  # [batch_size, n_samples]

        assert f0_frames.dim() == 2, f"f0_frames must have 2 dimensions, got shape {f0_frames.shape}"
        assert input.dim() == 2, f"target must have 2 dimensions (batch, samples), got shape {input.shape}"

        l_b = loop_coefficients if loop_coefficients is not None else self.loop_coefficients
        assert l_b.shape == (self.batch_size, self.loop_n_frames, self.loop_n_coefficients), f"Invalid loop coefficients shape: {l_b.shape} when it should be {self.batch_size, self.loop_n_frames, self.loop_n_coefficients}"

        f0_frames = self.internal_sr / f0_frames # Convert from Hz to samples

        if input_sr != self.internal_sr: input = kaiser_resample(input, sr_in=input_sr, sr_out=self.internal_sr)

        f0, l_b = self.get_upsampled_parameters(
            f0_frames, input.size(1),
            l_b=l_b,
        )

        l_b = self.design_loop(
            f0=f0,
            l_b=l_b,
        )

        f0 = self.tune_f0(f0=f0, b0=l_b[..., 0], a1=l_b[..., 1])
        A, x = self.compute_resonator_matrix(f0=f0, loop_coefficients=l_b, input=input)

        y_out = sample_wise_lpc(x, A) if invert is False else invert_lpc(x, A)
        return kaiser_resample(y_out, sr_in=self.internal_sr, sr_out=input_sr).to(torch.float32)

    def compute_resonator_matrix(
            self,
            f0: torch.Tensor,  # [batch_size, n_samples] phase-corrected fractional delay (in samples)
            loop_coefficients: torch.Tensor,  # [batch_size, n_samples, loop_n_coefficients]
            input: torch.Tensor, # [batch_size, n_samples]
    ) -> Tuple[
        torch.Tensor, torch.Tensor]:  # Returns A [batch_size, n_samples, coeff_vector_size], x [batch_size, n_samples]
        """
        Computes the coefficient matrix for a resonator with **given corrected fractional delay** and order-1 loop filter (IIR or FIR) in series with the Lagrange fractional delay inside the loop.

        Args:
            f0: Phase-corrected delay (samples) [batch_size, n_samples]
            loop_coefficients: Filter coefficients [batch_size, n_samples, loop_n_coefficients]
                (interpreted as [b0, a1])
            input: Input excitation signal [batch_size, n_samples]

        Returns:
            A: Coefficient matrix [batch_size, n_samples, coeff_vector_size]
            x: Modified excitation signal [batch_size, n_samples]
        """
        batch_size, n_samples = f0.shape

        x = input
        b = loop_coefficients  # [batch_size, n_samples, loop_n_coefficients]
        assert b.shape[-1] == self.loop_n_coefficients, (
            f"loop_coefficients last dim {b.shape[-1]} != expected {self.loop_n_coefficients}")

        # === Order-1 loop filter: numerator b0, AR/FIR tap a1 ===
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

        A = torch.zeros((batch_size, n_samples, self.coeff_vector_size), device=self.device, dtype=self._dtype)

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
                assert torch.all(idxs[-1] < self.coeff_vector_size), "Delay index exceeds the buffer size"
                for i in range(block_len):
                    A[batch_indices, sample_indices, idxs[i]] = -b_conv_total[..., i]
                # AR(1) term
                A[batch_indices, sample_indices, 1] += -a1
            elif self.loop_filter_kind == "fir":
                # FIR: build 2-tap result by adding +1 shifted copy scaled by a1
                b_conv_total = F.pad(b0_w, (0, 1)) + (a1.unsqueeze(-1) * F.pad(b0_w, (1, 0)))
                block_len = b_conv_total.size(-1)
                idxs = [z_center + i for i in range(block_len)]
                assert torch.all(idxs[-1] < self.coeff_vector_size), "Delay index exceeds the buffer size"
                for i in range(block_len):
                    A[batch_indices, sample_indices, idxs[i]] = -b_conv_total[..., i]
        else:
            raise NotImplementedError(f"Interpolation type {self.interp_type} not implemented")

        self._store_buffer(A, "resonator_matrix", "sequential")
        return A, x

    def design_loop(self,
                    l_b: torch.Tensor,  # [B, N, 2] -> logits for [gain, mix]
                    f0: torch.Tensor,   # [B, N]
                    return_gain: bool = False,
                    ) -> torch.Tensor:  # [B, N, 2] -> [b0, a1]
        """
        Designs an order‑1 loop filter with a numerically safe softcap parameterization.

        • `g` is mapped to (0.9, 1) but never equals 1 in finite precision.
        • `p` is mapped to (eps_p, 1 - eps_p), keeping sums strictly < 1.

        IIR: a1 = p, b0 = g * (1 - a1)
        FIR: taps sum to g via b0 = (1 - p) * g, a1 = p * g

        Returns [b0, a1] (or [b0, a1, g] if return_gain=True).
        """
        gain_logits = l_b[..., 0]
        mix_logits  = l_b[..., 1]

        # --- Helper: smooth map R -> (0, 1 - eps) that never touches 1 ---
        def softcap01(z, eps: float = 1e-6, tau: float = 0.25, power: float = 2.0):
            s = F.softplus(z / tau)
            if power != 1.0:
                s = s.pow(power)
            return (1.0 - eps) * s / (1.0 + s)  # in (0, 1 - eps)

        # --- Explicit g in (0.9, 1) but never 1 ---
        eps_g  = 1e-6
        base   = 0.9
        span   = 0.1 - eps_g                 # so upper bound is 1 - eps_g
        g_unit = softcap01(gain_logits, eps=eps_g, tau=0.25, power=2.0)  # (0, 1 - eps_g)
        g      = base + span * (g_unit / (1.0 - eps_g))                  # (0.9, 1 - eps_g)

        # --- p in a strict open interval (eps_p, 1 - eps_p) ---
        eps_p  = 1e-6
        p_unit = softcap01(mix_logits, eps=eps_p, tau=0.25, power=1.5)   # (0, 1 - eps_p)
        p      = eps_p + (1.0 - 2.0 * eps_p) * (p_unit / (1.0 - eps_p))  # (eps_p, 1 - eps_p)

        if self.loop_filter_kind == "iir":
            a1 = p
            b0 = (1.0 - a1) * g
        elif self.loop_filter_kind == "fir":
            b0 = (1.0 - p) * g
            a1 = p * g
        else:
            raise NotImplementedError("loop_filter_kind must be 'iir' or 'fir'")

        taps = torch.stack([b0, a1], dim=-1)

        # Debug safety check (should never trigger with the parameterization above)
        sum_taps = torch.abs(taps).sum(dim=-1)
        condition = sum_taps >= 1.0
        assert not condition.any(), (
            f"Loop filter |taps| sum must be < 1. Found {condition.sum().item()} instances where the sum is >= 1.0. "
            f"The violating values are:"
            f"\n  b0: {b0[condition].tolist()}"
            f"\n  a1: {a1[condition].tolist()}"
            f"\n  g: {g[condition].tolist()}"
            f"\n  sum: {sum_taps[condition].tolist()}"
        )

        if return_gain is False:
            return taps
        else:
            return torch.stack([b0, a1, g], dim=-1)


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