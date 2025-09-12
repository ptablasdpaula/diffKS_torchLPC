import torch
import torch.nn.functional as F
import torchaudio.functional as TAF
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

def midi_to_hz(midi : torch.Tensor) -> torch.Tensor:
    return 440 * 2 ** ((midi - 69) / 12)

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

    AUDIO_S = x.shape[1] / sr_in
    assert y.shape[1] == sr_out * AUDIO_S, f"Expected axis-1 length {sr_out}, got {y.shape[1]}"

    # move back to the original device only if we fell back
    return y.to(orig_dev) if orig_dev.type == "mps" else y


def spline_upsample(
        x: torch.Tensor,  # [B, Frames, D]
        num_samples: int, # [Samples]
) -> torch.Tensor:  # shape [B, Samples, D]
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