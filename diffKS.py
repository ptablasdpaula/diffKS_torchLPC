from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlpc import sample_wise_lpc
import torchaudio.functional as TAF
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from utils import get_device

LAGRANGE_ORDER = 5

def kaiser_resample(x, sr_in: int, sr_out: int,
                    width: int = 32, beta: float = 14.0,
                    rolloff: float = 0.9475937167399596):
    """
    Linear-phase Kaiser-windowed sinc resampler, as used in DDSP (Engel., et al).

    Args
    ----
    x        : (..., time) tensor
    sr_in    : original sample-rate
    sr_out   : target   sample-rate
    width    : low-pass filter width (taps per phase)
    beta     : Kaiser β; 14 ≈ >90 dB stop-band
    rolloff  : pass-band edge / Nyquist (DDSP uses 0.94759371674)
    """
    orig_dev = x.device
    # if on MPS, do the actual resampling on CPU
    if orig_dev.type == "mps":
        x = x.cpu()
    y = TAF.resample(
        x, sr_in, sr_out,
        lowpass_filter_width=width,
        rolloff=rolloff,
        resampling_method="sinc_interp_kaiser",
        beta=beta
    )
    # move back to the original device only if we fell back
    return y.to(orig_dev) if orig_dev.type == "mps" else y

def spline_upsample(x: torch.Tensor,  # shape [B, Frames, D]
                    num_samples) -> torch.Tensor:  # shape [B, Samples, D]
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
            initial = y.new_zeros(B, N, device=y.device)

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


def invert_lpc(y: torch.Tensor, # [B, N],
               A: torch.Tensor, # [B, N, D], where D is order
               zi: torch.Tensor = None) -> torch.Tensor:
    return InvertLPC.apply(y, A, zi)

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
        exc_length_s : float = 0.025,
        interp_type: str = "linear",
        use_double_precision: bool = False,
        upsample_mode: str = "zoh",
        soft_zoh_tau: float = 5.0,
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
        # upsampling mode for parameter frames ('zoh', 'spline', or 'soft')
        self.upsample_mode = upsample_mode
        # softness (in *samples*) of the rising/falling sigmoid edges used in soft ZOH
        self.soft_zoh_tau = float(soft_zoh_tau)

        # ====== Excitation Filter ======================
        self.exc_order = exc_order
        self.exc_n_coefficients = exc_order + 1 # To account for exc_g
        self.exc_length_n = int (exc_length_s * internal_sr)

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
        self.register_buffer("excitation_filter_out", torch.empty(batch_size, self.exc_length_n), persistent=False)
        self.register_buffer("ks_inverse_signal", torch.zeros(batch_size, self.exc_length_n), persistent=False)

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
                constrain_coefficients: bool = True,
                triggers: Optional[torch.Tensor] = None, # [batch_size, loop_n_frames]
                ) -> torch.Tensor:  # [batch_size, n_samples]

        assert f0_frames.dim() == 2, f"f0_frames must have 2 dimensions, got shape {f0_frames.shape}"
        assert input.dim() == 2, f"target must have 2 dimensions (batch, samples), got shape {input.shape}"


        if constrain_coefficients:
            l_b = self._prepare("loop_coefficients", loop_coefficients)
            l_g = self._prepare("loop_gain", loop_gain)
            exc_b = self._prepare("exc_coefficients", exc_coefficients)
        else:
            l_b = loop_coefficients
            l_g = loop_gain
            exc_b = exc_coefficients

        f0_frames = self.internal_sr / f0_frames # Convert from Hz to samples

        if input_sr != self.internal_sr:
            input = kaiser_resample(input, sr_in=input_sr, sr_out=self.internal_sr)

        n_samples = input.size(1)

        f0, l_b, l_g, exc_b = self.get_upsampled_parameters(
            f0_frames, n_samples,
            l_b=l_b, l_g=l_g, exc_b=exc_b,
            triggers=triggers,
        )

        if constrain_coefficients:
            l_b = self.get_constrained_l_coefficients(l_b=l_b, l_g=l_g)

            exc_b = self.get_constrained_exc_coefficients(exc_b=exc_b)
        else:
            # make sure interpolation didn't produce out of range values
            l_b = l_b.clamp(min=0.0, max=1.0)
            l_g = l_g.clamp(min=0.0, max=1.0)
            exc_b = exc_b.clamp(min=0.0, max=1.0)

            # Apply gain to the loop coefficients
            l_b = l_b * l_g

        A, x = self.compute_resonator_matrix(f0=f0,
                                             loop_coefficients=l_b,
                                             input=input)

        if not direct:
            loop_inv = invert_lpc(x, A)
            ks_inv_signal = self._inversed_windowed_lpc(loop_inv, exc_b, triggers=triggers)
            self.ks_inverse_signal = ks_inv_signal.detach().clone()

        exc_filter_out = sample_wise_lpc(ks_inv_signal if not direct else x, exc_b)

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

        return A, x

    def _inversed_windowed_lpc(
            self,
            x: torch.Tensor,          # [B, N]
            b: torch.Tensor,          # [B, N, O]
            triggers: torch.Tensor,   # [B, F]
    ):
        n_samples = x.size(1)

        proc = invert_lpc(x, b)                               # ← [B, N]

        idx = torch.arange(n_samples, device=x.device)                # [N]
        idx = idx.view(1, 1, -1)                                      # [1,1,N]

        win_start = triggers.to(torch.long).unsqueeze(-1)             # [B,F,1]
        win_end   = win_start + self.exc_length_n                     # [B,F,1]

        mask = (idx >= win_start) & (idx < win_end)                   # [B,F,N]
        mask = mask.any(dim=1)                                        # [B,N]

        out = torch.where(mask, proc, torch.zeros_like(proc))
        return out

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

    def _upsample_by_triggers(
            self,
            frames: torch.Tensor,        # [B, F, D]
            triggers: torch.Tensor,      # [B, F] (sample indices @ internal‑SR)
            n_samples: int,              # total output length
            mode: str = "soft",        # "spline" (default), "zoh", or "soft"
    ) -> torch.Tensor:                  # [B, n_samples, D]
        """
        Interpolate the *frames* timeline according to *triggers*.

        • **mode="spline"          – natural cubic spline between frames,
          then zero‑order held after the last trigger.

        • **mode="zoh"** – pure zero‑order hold; hard, non‑differentiable w.r.t. trigger times.

        • **mode="soft"** – *differentiable* soft zero‑order hold: each frame owns a
          soft rectangular window defined by adjacent trigger positions; edges are
          smoothed by sigmoids of width `self.soft_zoh_tau` samples so gradients flow
          back to the trigger times.

        Frames and triggers may disagree on length; extra frames are truncated,
        missing ones repeat the last frame.
        """
        if mode not in ("spline", "zoh", "soft"):
            raise ValueError("mode must be 'spline' or 'zoh' or 'soft'")

        B, F_coef, D = frames.shape
        B_t, F_trig  = triggers.shape
        if B != B_t:
            raise ValueError("batch mismatch between frames and triggers")

        # ----- 1. ensure |frames| == |triggers| ------------------------------
        if F_coef > F_trig:                                # too many frames
            frames = frames[:, :F_trig, :]
        elif F_coef < F_trig:                              # too few  → pad
            pad = frames[:, -1:, :].expand(-1, F_trig - F_coef, -1)
            frames = torch.cat([frames, pad], dim=1)

        if mode == "soft":
            # ----- differentiable soft ZOH ------------------------------------
            # We approximate a rectangular ownership window for each trigger with
            # smooth sigmoid edges so gradients can flow to trigger locations.
            # frames: [B,F,D]; triggers: [B,F] (floatable); returns [B,N,D]
            trig_f = triggers.to(dtype=self._dtype)
            device = frames.device
            N = n_samples
            # build "next trigger" tensor by concatenating last index = N-1
            last = torch.full((B,1), float(N-1), device=device, dtype=self._dtype)
            trig_next = torch.cat([trig_f[:,1:], last], dim=1)
            # time axis
            t = torch.arange(N, device=device, dtype=self._dtype).view(1,1,-1)  # [1,1,N]
            # softness
            tau = torch.tensor(self.soft_zoh_tau, device=device, dtype=self._dtype)
            # rising edge at current trigger; falling edge at next trigger
            start = torch.sigmoid((t - trig_f.unsqueeze(-1)) / tau)      # [B,F,N]
            end   = torch.sigmoid((t - trig_next.unsqueeze(-1)) / tau)   # [B,F,N]
            w = (start - end).clamp_min_(0.0)                            # [B,F,N]
            # normalize across frames so weights sum to 1 at each sample
            w_sum = w.sum(dim=1, keepdim=True).clamp_min_(1e-12)
            w = w / w_sum
            # weighted sum of frames
            out = torch.einsum('bfn,bfd->bnd', w, frames.to(self._dtype))  # [B,N,D]
            return out

        if mode == "zoh":
            trig_int = triggers.to(torch.long)
            # --- safety: enforce non‑decreasing, in‑range trigger timeline ----
            # triggers may contain padded or unsorted values (e.g., from batching)
            trig_int = torch.clamp(trig_int, min=0, max=n_samples - 1)
            # ensure monotonic non‑decreasing: cumulative maximum along time axis
            trig_int = torch.cummax(trig_int, dim=1).values
            segs     = []
            for b in range(B):
                tb = trig_int[b]  # [F_trig]
                if F_trig == 1:
                    seg_len = torch.tensor([n_samples], device=frames.device, dtype=torch.long)
                else:
                    first = tb[1]                                       # samples from 0→t1
                    inner = tb[2:] - tb[1:-1]
                    last  = n_samples - tb[-1]
                    seg_len = torch.cat([first.unsqueeze(0), inner, last.unsqueeze(0)])
                # guard against any residual negatives due to numerical drift
                seg_len = torch.clamp(seg_len, min=0)
                segs.append(seg_len)
            out = []
            for b in range(B):
                out_b = torch.repeat_interleave(frames[b], segs[b], dim=0)
                out.append(out_b)
            return torch.stack(out, dim=0)                              # [B,N,D]

        # ------------------- spline mode -------------------------------------
        device   = frames.device
        t_out    = torch.arange(n_samples, device=device,
                                dtype=self._dtype)        # [N]
        outs = []
        for b in range(B):
            t_in     = triggers[b].to(self._dtype)             # [F]
            y_in     = frames[b]                               # [F, D]
            # normalise to 0‑1 for torchcubicspline
            t_norm   = t_in / (n_samples - 1)
            t_out_n  = t_out / (n_samples - 1)

            coeffs   = natural_cubic_spline_coeffs(t_norm, y_in)
            y_interp = NaturalCubicSpline(coeffs).evaluate(t_out_n)  # [N, D]

            # hold last value after final trigger
            after_last = t_out > t_in[-1]
            y_interp[after_last] = y_in[-1]
            outs.append(y_interp)
        return torch.stack(outs, dim=0)                         # [B, N, D]

    def get_upsampled_parameters(
            self,
            f0: torch.Tensor, # [batches, f_0_frames,]
            num_samples: int,
            l_b: Optional[torch.Tensor] = None, # [batches, frames, loop_n_coefficients]
            l_g: Optional[torch.Tensor] = None, # [batches, frames, 1]
            exc_b: Optional[torch.Tensor] = None, # [batches, frames, exc_order]
            triggers: Optional[torch.Tensor] = None, # [B, loop_frames]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        loop_b_frames_ = l_b if l_b is not None else self.loop_coefficients
        loop_g_frames_ = l_g if l_g is not None else self.loop_gain
        exc_b_frames_ = exc_b if exc_b is not None else self.exc_coefficients

        batch_size = f0.size(0)
        f0_n_frames = f0.size(1)

        # ---------- F0  -----------------------------------------------------
        if f0_n_frames == 1:
            f0_i = f0.expand(batch_size, num_samples)
        else:
            f0_reshaped = f0.unsqueeze(-1).to(dtype=self._dtype)
            f0_i = spline_upsample(f0_reshaped, num_samples).squeeze(-1)

        # ---------- coefficients -------------------------------------------
        if triggers is not None:
            loop_b_i = self._upsample_by_triggers(loop_b_frames_.to(dtype=self._dtype),
                                             triggers, num_samples, mode=self.upsample_mode)
            loop_g_i = self._upsample_by_triggers(loop_g_frames_.to(dtype=self._dtype),
                                             triggers, num_samples, mode=self.upsample_mode)
            exc_b_i = self._upsample_by_triggers(exc_b_frames_.to(dtype=self._dtype),
                                            triggers, num_samples, mode=self.upsample_mode)
        else:
            raise NotImplementedError(f"support for no triggers not implemented")

        return f0_i.to(self.device), loop_b_i.to(self.device), loop_g_i.to(self.device), exc_b_i.to(self.device)

    def get_inverse_filtered_signal(self):
        return self.ks_inverse_signal

    def get_excitation_filter_out(self):
        return self.excitation_filter_out