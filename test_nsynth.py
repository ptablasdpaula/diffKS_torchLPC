# --------------------------------------------------------------------------
# Differentiable Karplus-Strong experiment with extra spectrogram outputs
# --------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Union

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import torch.nn as nn
from itertools import chain

from third_party.auraloss.auraloss.freq import MultiResolutionSTFTLoss as MultiSTFT
from diffKS import DiffKS
from core import make_onset_noise, scale_noise_bursts_to_target_rms, detect_onsets_librosa, StaticShelf
from utils.misc import get_device

hp = {
    "use_A_weighing": True,
    "epochs": 150,
    "lr": 0.1,
}

mp = {
    "loop_order": 2,
    "loop_n_frames": 250,
    "f0_hz": 311.13,
    "min_f0_hz": 82.41,  # MIDI E2 in Hz
    "burst_width_s": 0.03,
    "use_double_precision": True,
    "interp_type": "linear"
}

gs = {
    "sample_rate": 16000,
    "internal_sr": 16000,
    "length_audio_s": 4,
    "random_seed": 1234
}

LENGTH_N = 4 * gs["sample_rate"]
LENGTH_N_UPSAMPLED = 4 * gs["internal_sr"]

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def save_audio(path: str | Path, tensor: torch.Tensor, sr: int) -> None:
    """Save *mono* tensor to WAV (expects shape [1, samples] or [samples])."""
    # Create the analysis directory if it doesn't exist
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)

    # Modify the path to save in the analysis directory
    file_name = Path(path).name
    save_path = analysis_dir / file_name

    tensor = tensor.detach().cpu()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    torchaudio.save(str(save_path), tensor, sr)

def load_guitar(path: str | Path, sr_tgt: int) -> torch.Tensor:
    """Load *audio/guitar.wav* and resample to *sr_tgt* (mono)."""
    wav, sr_in = torchaudio.load(str(path))
    if wav.dim() > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)  # down‑mix
    if sr_in != sr_tgt:
        wav = torchaudio.transforms.Resample(sr_in, sr_tgt)(wav)
    return wav

def build_random_model(seed: int, F: int) -> DiffKS:
    """Create a *DiffKS* with random weights but fixed *seed*."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    loop_order = mp["loop_order"]

    model = DiffKS(
        internal_sr=gs["internal_sr"],
        min_f0_hz=mp["min_f0_hz"],
        loop_order=loop_order,
        use_double_precision=mp["use_double_precision"],
        batch_size=1
    )

    model.set_loop_coefficients(torch.rand(1, F, loop_order + 1))
    return model

# -----------------------------------------------------------------------------
# Composite plotting helpers
# -----------------------------------------------------------------------------

def composite_plot(fig_path: str,
                   signals: Dict[str, torch.Tensor],
                   coeffs: Dict[str, Union[np.ndarray, torch.Tensor]],
                   plot_low_shelf=None, plot_high_shelf=None) -> None:
    """
    Plot a set of waveforms ('signals') and time‑varying filter coefficients
    ('coeffs') on a single canvas and save to *fig_path*.

    Composite plot: 3 subplots only —
      1. Target signal
      2. Optimised output signal
      3. Loop coefficients (all, including b0/gain)
    """
    # Always 3 rows (subplots): Target, Optimised, Loop Coeffs
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    # 1. Target signal
    ax = axs[0]
    wav_np = signals["Target"].squeeze().detach().cpu().numpy()
    wav_samples = len(wav_np)
    t = np.linspace(0, gs['length_audio_s'], wav_samples)
    ax.plot(t, wav_np, color='orange')
    ax.set_xlabel("seconds")
    ax.set_xlim(0, gs['length_audio_s'])
    ax.set_title("Target")
    ax.grid(False)

    # 2. Optimised output signal
    ax = axs[1]
    wav_np = signals["Optimised"].squeeze().detach().cpu().numpy()
    wav_samples = len(wav_np)
    t = np.linspace(0, gs['length_audio_s'], wav_samples)
    ax.plot(t, wav_np, color='orange')
    ax.set_xlabel("seconds")
    ax.set_xlim(0, gs['length_audio_s'])
    ax.set_title("Optimised")
    ax.grid(False)

    # 3. Loop coefficients (all, including b0/gain)
    ax = axs[2]
    # Only one entry in coeffs: "Loop coeffs"
    for name, traj in coeffs.items():
        traj_np = traj if isinstance(traj, np.ndarray) else traj
        traj_np = traj_np if isinstance(traj_np, np.ndarray) else traj_np.cpu().numpy()
        sr_vis      = gs["sample_rate"]
        sr_internal = gs["internal_sr"]
        x_coeff     = np.arange(traj_np.shape[0]) * (sr_vis / sr_internal)
        if traj_np.ndim == 1:
            ax.plot(x_coeff, traj_np, label=name)
        else:
            for k in range(traj_np.shape[1]):
                ax.plot(x_coeff, traj_np[:, k], label=f"{name}-b{k}")
        ax.set_title("Loop coefficients (all taps, including b0/gain)")
        ax.set_xlim(0, gs["sample_rate"] * gs["length_audio_s"])
        ax.set_xlabel("samples (16kHz)")
        ax.grid(False)
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.show()

def _compute_losses(model: DiffKS,
                    low_shelf: nn.Module,
                    high_shelf: nn.Module,
                    raw_noise: torch.Tensor,
                    target: torch.Tensor,
                    f0_frames: torch.Tensor,
                    loss_fn: MultiSTFT) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Forward pass for excitation → KS → STFT loss (single-stage; no reverb/regularisers)."""
    sample_rate = gs["sample_rate"]

    # Excitation shaping: Low-shelf → High-shelf
    excitation = high_shelf(low_shelf(raw_noise))

    # KS synthesis (dry)
    out = model(
        f0_frames=f0_frames,
        input=excitation,
        input_sr=sample_rate,
    )

    # Main reconstruction loss only (no smoothness / whitening)
    loss_stft = loss_fn(out.unsqueeze(1), target.unsqueeze(1))

    loss = loss_stft
    log = {
        "stft": float(loss_stft.detach().cpu()),
    }
    return loss, log

# ----------------------------------------------------------------------------
# Single-stage training: optimise DiffKS + shelves together
# ----------------------------------------------------------------------------

def train_single_stage(model: DiffKS,
                       raw_noise: torch.Tensor,
                       target: torch.Tensor,
                       f0_frames: torch.Tensor,
                       low_shelf: nn.Module,
                       high_shelf: nn.Module) -> None:
    sample_rate = gs["sample_rate"]
    device = model.device
    model.train(); low_shelf.train(); high_shelf.train()

    loss_fn = MultiSTFT(
        fft_sizes=[257, 509, 1019, 2039, 4093],
        hop_sizes=[128, 254, 509, 1019, 2046],
        win_lengths=[257, 509, 1019, 2039, 4093],
        window="flattop",
        mag_distance="L2",
        log_eps=1.0,
        w_sc=1.0,
        w_log_mag=1.0,
        w_lin_mag=0.0,
        perceptual_weighting=hp.get("use_A_weighing", True),
        scale_invariance=False,
        sample_rate=sample_rate,
    )

    raw_noise = raw_noise.to(device)
    target = target.to(device)
    f0_frames = f0_frames.to(device)

    params = list(chain(model.parameters(), low_shelf.parameters(), high_shelf.parameters()))
    opt = torch.optim.Adam(params, lr=float(hp.get("lr", 2.5e-2)))

    n_epochs = int(hp.get("epochs", 1500))
    pbar = tqdm(range(n_epochs), desc=f"Single-stage (@lr={hp.get('lr', 2.5e-2):g})")

    loss_curve = []
    for _ in pbar:
        loss, log = _compute_losses(model, low_shelf, high_shelf,
                                    raw_noise, target, f0_frames, loss_fn)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_curve.append(float(loss.detach().cpu()))
        pbar.set_postfix(loss=f"{loss_curve[-1]:.4f}", stft=f"{log['stft']:.4f}")

    # Plot & save a simple loss curve
    plt.figure(figsize=(8, 3))
    plt.plot(loss_curve)
    plt.title("Training loss (single-stage)")
    plt.xlabel("iteration")
    plt.ylabel("total loss")
    plt.grid(True, alpha=0.3)
    Path("analysis").mkdir(exist_ok=True)
    plt.savefig("analysis/loss_curve_single_stage.png")

# -----------------------------------------------------------------------------
# Entry‑point ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main() -> None:
    sample_rate = gs["sample_rate"]

    # -------------------------------------------------------------------------
    # 3.  Load & prepare guitar -------------------------------------------------
    guitar = load_guitar("data/test.wav", sample_rate).to(device=get_device())
    save_audio("target.wav", guitar, sample_rate)

    # -------------------------------------------------------------------------
    # 4.  Onsets (librosa with 50 ms left pad) + noise bursts of length = delay
    # -------------------------------------------------------------------------
    onset_samples = detect_onsets_librosa(guitar, sr=sample_rate, pad_ms=50.0,
                                          hop_length=512, backtrack=True)

    # Delay length at the *external* SR for excitation construction
    L_delay = max(1, int(round(sample_rate / mp["f0_hz"])) )

    raw_noise = make_onset_noise(
        onset_samples=onset_samples,
        num_samples=LENGTH_N,
        sample_rate=sample_rate,
        batch_size=1,
        device=get_device(),
        dtype=guitar.dtype,
        burst_len_samples=L_delay,
    )
    # Save pre-matched (dry) bursts for reference
    save_audio("raw_onset_noise_pre_match.wav", raw_noise[0], sample_rate)

    # Match burst RMS to target RMS in the same windows [s, s+L_delay)
    raw_noise = scale_noise_bursts_to_target_rms(
        noise=raw_noise,
        target=guitar,
        onset_samples=onset_samples,
        burst_len_samples=L_delay,
        eps=1e-8,
        compensate_delay_len=True,  # set True to also apply 1/sqrt(L) scaling
    )

    save_audio("raw_onset_noise.wav", raw_noise[0], sample_rate)

    # Learnable shelves on excitation (body tilt)
    low_shelf = StaticShelf(
        which="low",
        sample_rate=sample_rate,
        init_fc_hz=120.0,
        fmin_hz=20.0,
        fmax_hz=sample_rate / 2 - 200.0,
        init_Q=0.707,
        init_gain_db=-3.0,  # gentle low cut
    ).to(get_device())

    high_shelf = StaticShelf(
        which="high",
        sample_rate=sample_rate,
        init_fc_hz=3000.0,
        fmin_hz=30.0,
        fmax_hz=sample_rate / 2 - 200.0,
        init_Q=0.707,
        init_gain_db=-1.5,  # a touch off the air band
    ).to(get_device())

    # -------------------------------------------------------------------------
    # 5.  Build model with matching number of frames --------------------------
    seed = gs["random_seed"]
    F = mp["loop_n_frames"]
    model_opt = build_random_model(seed, F)

    # constant f0 per frame
    f0_frames_opt = torch.full((1, F), mp["f0_hz"], dtype=guitar.dtype)

    # 6.  Train (single-stage) ----------------------------------------------
    target_audio = guitar
    train_single_stage(model_opt,
                       raw_noise=raw_noise,
                       target=target_audio,
                       f0_frames=f0_frames_opt,
                       low_shelf=low_shelf,
                       high_shelf=high_shelf)

    # -------------------------------------------------------------------------
    # 7.  Render with the trained model and save ------------------------------
    model_opt.eval()
    low_shelf.eval()
    high_shelf.eval()
    with torch.no_grad():
        excitation_trained = high_shelf(low_shelf(raw_noise))
        save_audio("filtered_excitation.wav", excitation_trained[0].cpu(), sample_rate)
        optim_audio = model_opt(
            f0_frames=f0_frames_opt.to(model_opt.device),
            input=excitation_trained.to(model_opt.device),
            input_sr=sample_rate,
        ).cpu()[0, ...]

    ls_fc = float(low_shelf.fc_hz().item())
    ls_Q = float(low_shelf.quality_Q().item())
    ls_gdb = float(low_shelf.gain_db().item())
    print(f"[LowShelf]  fc={ls_fc:.2f} Hz, Q={ls_Q:.3f}, gain={ls_gdb:+.2f} dB")

    hs_fc = float(high_shelf.fc_hz().item())
    hs_Q = float(high_shelf.quality_Q().item())
    hs_gdb = float(high_shelf.gain_db().item())
    print(f"[HighShelf] fc={hs_fc:.2f} Hz, Q={hs_Q:.3f}, gain={hs_gdb:+.2f} dB")

    assert not any(map(math.isnan, [ls_fc, ls_Q, ls_gdb])), "Low shelf returned NaN!"
    assert not any(map(math.isnan, [hs_fc, hs_Q, hs_gdb])), "High shelf returned NaN!"

    plot_low_shelf = (ls_fc, ls_Q, ls_gdb)
    plot_high_shelf = (hs_fc, hs_Q, hs_gdb)

    # Composite plot including coefficient trajectories (loop & excitation)
    # Compute the number of internal-sr samples used inside the model
    N_samples_vis = guitar.shape[-1]
    N_samples_internal = int(round(N_samples_vis * gs["internal_sr"] / gs["sample_rate"]))

    with torch.no_grad():
        # Upsample parameters to internal-sr length, then apply constraints
        _f0_i, l_b_u = model_opt.get_upsampled_parameters(
            f0=f0_frames_opt.to(model_opt.device),
            num_samples=N_samples_internal,
            l_b=model_opt.loop_coefficients,
        )

        print (f"this is the size of l_b: {model_opt.loop_coefficients.shape}")

        l_b_c_t = model_opt.get_constrained_l_coefficients(f0=_f0_i, l_b=l_b_u)       # [1, T_int, loop_n]

        # For plotting as before
        l_b_c = l_b_c_t[0].detach().cpu().numpy()

    signals = {
        "Target": target_audio[0],        # [N]
        "Optimised": optim_audio,         # [N]
    }
    coeffs = {
        "Loop coeffs": l_b_c,             # [T_int, loop_n_coefficients]
    }

    composite_plot("analysis/composite.png", signals, coeffs, plot_low_shelf=plot_low_shelf, plot_high_shelf=plot_high_shelf)
    save_audio("optimized_model_from_onsets.wav", optim_audio, sample_rate)

if __name__ == "__main__":
    main()