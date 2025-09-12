# --------------------------------------------------------------------------
# Differentiable Karplus-Strong: optimise loop logits by gradient descent
# Pure-DSP DiffKS (no internal params) + learnable per-sample logits
# --------------------------------------------------------------------------
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from losses import build_smooth_mrstft
from diffKS import DiffKS
from core import hz_to_samples, one_minus_log_tail
from utils.misc import get_device

SAMPLE_RATE = 16000
LENGTH_AUDIO_S = 4.0
RANDOM_SEED = 1234

TORCH_DTYPE = torch.float32

def save_audio(path: str | Path, tensor: torch.Tensor, sr: int) -> None:
    Path("analysis").mkdir(exist_ok=True)
    p = Path("analysis") / Path(path).name
    t = tensor.detach().cpu()
    t = t.unsqueeze(0)
    torchaudio.save(str(p), t, sr)

# ---------------------------- Simple excitation ----------------------------

def make_single_burst(T: int, L_burst: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """A simple onset burst at t=0 of length L_burst samples (mono, batch=1)."""
    x = torch.zeros(1, T, device=device, dtype=dtype)
    if L_burst > 0:
        burst = torch.randn(L_burst, device=device, dtype=dtype)
        rms = torch.sqrt(torch.mean(burst ** 2) + 1e-12)
        burst = 0.2 * burst / (rms + 1e-12)
        x[:, :L_burst] = burst
    return x

# ----------------------------- Training bits -------------------------------

def train_optimize_logits(
    model: DiffKS,
    target: torch.Tensor,      # [1, T]
    excitation: torch.Tensor,  # [1, T]
    f0_hz: float,
    epochs: int,
    lr: float,
    sample_rate: int,
    loop_n_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = model.device
    B, T = target.shape

    # Fixed f0 (Hz) → samples/period (per-sample constant)
    f0_hz_tensor = torch.full((B, T), float(f0_hz), device=device, dtype=TORCH_DTYPE)
    f0_samples = hz_to_samples(f0_hz_tensor, fs=sample_rate)

    # Learnable per-sample logits: [..., 0]=gain, [..., 1]=mix
    torch.manual_seed(RANDOM_SEED)
    loop_logits = nn.Parameter(torch.rand(B, loop_n_frames, 2, device=device, dtype=TORCH_DTYPE))

    loss_fn = build_smooth_mrstft(sample_rate=sample_rate)
    opt = torch.optim.Adam([loop_logits], lr=lr)

    loss_curve = []
    y_final = None
    pbar = tqdm(range(epochs), desc=f"Optimising logits (lr={lr:g})")
    for _ in pbar:
        # Map logits → constrained params expected by DiffKS.design_loop
        g = one_minus_log_tail(loop_logits[..., 0])                # (0.9, 1)
        p = torch.sigmoid(loop_logits[..., 1])                     # (0, 1)
        l_b = torch.stack([g, p], dim=-1)                          # [B, 1, 2]
        l_b = F.interpolate(l_b.permute(0,2,1), size=T, mode="linear").permute(0,2,1)

        # Synthesise
        y = model(
            f0=f0_samples,
            input=excitation,
            l_b=l_b,
            invert=False,
        )  # [B, T]
        y_final = y.detach()

        loss = loss_fn(y.unsqueeze(1), target.unsqueeze(1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        loss_v = float(loss.detach().cpu())
        loss_curve.append(loss_v)
        pbar.set_postfix(loss=f"{loss_v:.4f}")

    # After training, map final logits to bounded params
    g = one_minus_log_tail(loop_logits[..., 0])
    p = torch.sigmoid(loop_logits[..., 1])
    l_b = torch.stack([g, p], dim=-1)
    gp_traj = torch.stack([g, p], dim=-1)

    return l_b.detach(), gp_traj.detach(), y_final.detach()

# ------------------------------- Main script --------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Differentiable Karplus-Strong optimisation")
    parser.add_argument("--filter-kind", choices=["iir", "fir"], default="iir", help="Loop filter kind (iir or fir)")
    parser.add_argument("--loop-n-frames", type=int, default=1, help="Number of loop coefficient frames (logits)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate")
    parser.add_argument("--use-double-precision", action="store_true", help="Use float64 precision instead of float32")
    parser.add_argument("--interp-type", choices=["linear", "allpass", "lagrange"], default="lagrange", help="Interpolation type for fractional delay")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    sr = int(SAMPLE_RATE)
    target, _ = torchaudio.load(str("data/test.wav"))
    target = target.to(device)
    T = target.shape[-1]
    save_audio("target.wav", target[0], sr)

    # Simple excitation: one noise burst of period length
    f0_hz = 311.13
    L_delay = int(hz_to_samples(torch.tensor(f0_hz, dtype=TORCH_DTYPE, device=device), fs=sr).item())
    excitation = make_single_burst(T, L_delay, device, TORCH_DTYPE).to(device)
    save_audio("excitation.wav", excitation[0], sr)

    # Build pure-DSP model
    model = DiffKS(
        interp_type=args.interp_type,
        use_double_precision=args.use_double_precision,
        device=device,
        loop_filter_kind=args.filter_kind,
    ).to(device)

    # Optimise per-sample logits
    l_b, gp_traj, y_final = train_optimize_logits(
        model=model,
        target=target,
        excitation=excitation,
        f0_hz=f0_hz,
        epochs=int(args.epochs),
        lr=float(args.lr),
        sample_rate=sr,
        loop_n_frames=args.loop_n_frames,
    )

    save_audio("optimised.wav", y_final[0], sr)

    # Plot loss curve and coefficient trajectories (time axis in seconds)
    Path("analysis").mkdir(exist_ok=True)
    frames = gp_traj.shape[1]
    g_np = gp_traj[0, :, 0].detach().cpu().numpy()
    p_np = gp_traj[0, :, 1].detach().cpu().numpy()

    duration_s = T / float(sr)
    # Frame edge times: [0, dur/frames, 2*dur/frames, ...]
    x_edges = np.linspace(0.0, duration_s, num=frames, endpoint=False)
    # Build step arrays that extend to the end time
    x_step = np.concatenate([x_edges, [duration_s]])
    g_step = np.concatenate([g_np, g_np[-1:]])
    p_step = np.concatenate([p_np, p_np[-1:]])

    plt.figure(figsize=(8, 3))
    plt.step(x_step, g_step, where="post", label="g (near 1)")
    plt.step(x_step, p_step, where="post", label="p (mix)")
    plt.xlim(0.0, duration_s)
    plt.ylim(0.0, 1.05)
    plt.xlabel("seconds")
    plt.title("Optimised loop params (g, p)")
    plt.legend()
    plt.tight_layout(); plt.savefig("analysis/params_gp.png")

    print("Saved: analysis/target.wav, analysis/excitation.wav, analysis/optimised.wav, analysis/params_gp.png")

if __name__ == "__main__":
    main()