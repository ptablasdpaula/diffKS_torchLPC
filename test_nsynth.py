# diffks_experiment.py
# --------------------------------------------------------------------------
# Differentiable Karplus-Strong experiment with extra spectrogram outputs
# --------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any, Union

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm

from third_party.auraloss.auraloss.freq import MultiResolutionSTFTLoss as MultiSTFT

from diffKS import DiffKS
from utils import resize_tensor_dim

hp = {
    "learning_rate": 0.2,
    "max_epochs": 250,
    "use_A_weighing": True,
}

mp = {
    "exc_order": 5,
    "exc_n_frames": 4,
    "exc_length_s": 0.025,
    "loop_order": 2,
    "loop_n_frames": 4,
    "f0_hz": 311.13,
    "min_f0_hz": 82.41,  # MIDI E2 in Hz
    "burst_width_s": 0.03,
    "use_double_precision": False,
    "normalize_burst": True,
    "interp_type": "linear"
}

gs = {
    "sample_rate": 16000,
    "internal_sr": 41000,
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


def build_random_model(seed: int) -> DiffKS:
    """Create a *DiffKS* with random weights but fixed *seed*."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    loop_order = mp["loop_order"]
    loop_n_frames = mp["loop_n_frames"]
    exc_order = mp["exc_order"]
    exc_n_frames = mp["exc_n_frames"]

    model = DiffKS(
        internal_sr=gs["internal_sr"],
        min_f0_hz=mp["min_f0_hz"],
        loop_order=loop_order,
        loop_n_frames=loop_n_frames,
        exc_order=exc_order,
        exc_n_frames=exc_n_frames,
        exc_length_s=mp["exc_length_s"],
        interp_type=mp["interp_type"],
        use_double_precision=mp["use_double_precision"],
        batch_size=1
    )

    model.set_loop_coefficients(torch.rand(1, loop_n_frames, loop_order + 1))
    model.set_loop_gain(torch.rand((1, loop_n_frames, 1), ))
    model.set_exc_coefficients(torch.rand(1, exc_n_frames, exc_order + 1) * 0.1)

    return model


# -----------------------------------------------------------------------------
# Composite plotting helpers
# -----------------------------------------------------------------------------

def composite_plot(fig_path: str,
                   signals: Dict[str, torch.Tensor],
                   coeffs: Dict[str, Union[np.ndarray, torch.Tensor]]) -> None:
    """
    Plot a set of waveforms ('signals') and time‑varying filter coefficients
    ('coeffs') on a single canvas and save to *fig_path*.

    • Each entry in *signals* is a mono tensor – shape [..., samples].
    • Each entry in *coeffs* is either a 1‑D trajectory (time) or a 2‑D matrix
      (time, n_coeffs).  All taps are plotted if 2‑D.
    """
    n_rows = max(len(signals), len(coeffs))
    fig, axs = plt.subplots(n_rows, 2, figsize=(14, 3 * n_rows))
    axs = axs.ravel()

    # --------------------------------------------------------------------- #
    # 1.  Waveforms
    # --------------------------------------------------------------------- #
    row = 0
    for name, wav in signals.items():
        ax = axs[row * 2]
        wav_np = wav.squeeze().detach().cpu().numpy()

        # Choose color based on signal name
        if name in ["Target", "Optimised", "Inverse filtered"]:
            color = 'orange'
        else:
            color = None  # Use default color

        # Apply time scaling based on signal type
        if name == "Inverse filtered":
            # Inverse filtered: 0 to 0.025s
            wav_samples = len(wav_np)
            t = np.linspace(0, mp['exc_length_s'], wav_samples)
            ax.plot(t, wav_np, color=color)
            ax.set_xlabel("seconds")
            ax.set_xlim(0, mp['exc_length_s'])
        elif name in ["Target", "Optimised"]:
            # Target and Optimised: 0 to 4 seconds
            wav_samples = len(wav_np)
            t = np.linspace(0, gs['length_audio_s'], wav_samples)
            ax.plot(t, wav_np, color=color)
            ax.set_xlabel("seconds")
            ax.set_xlim(0, gs['length_audio_s'])
        else:
            # Other signals: keep sample index
            ax.plot(wav_np, color=color)
            ax.set_xlabel("samples")

        ax.set_title(name)
        ax.grid(False)
        row += 1

    # --------------------------------------------------------------------- #
    # 2.  Coefficient trajectories
    # --------------------------------------------------------------------- #
    row = 0
    for name, traj in coeffs.items():
        ax = axs[row * 2 + 1]
        traj_np = traj if isinstance(traj, np.ndarray) else traj
        traj_np = traj_np if isinstance(traj_np, np.ndarray) else traj_np.cpu().numpy()

        if traj_np.ndim == 1:  # single trajectory
            ax.plot(traj_np, label=name)
        else:  # multiple taps
            for k in range(traj_np.shape[1]):
                ax.plot(traj_np[:, k], label=f"{name}-b{k}")
        ax.set_title(name)
        ax.set_xlabel("samples")
        ax.grid(False)
        ax.legend(loc="upper right")
        row += 1

    plt.tight_layout()
    plt.savefig(fig_path)
    # Keep plot window open
    plt.show()

# -----------------------------------------------------------------------------
# Training loop - Simplified version
# -----------------------------------------------------------------------------
def optimise_model(model: DiffKS,
                   input_signal: torch.Tensor,  # what goes *into* the model
                   target: torch.Tensor,  # waveform we want to match
                   f0_frames: torch.Tensor,  # delay‑trajectory (ctrl‑rate)
                   direct: bool,  # True→noise‑burst path
                   ) -> None:
    """
    Optimise *model* so that model(input_signal) ≈ target under STFT loss.
    """
    # Set up optimizer using Adam
    optimiser = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])

    # Set up loss function (multi-resolution STFT)
    loss_fn = MultiSTFT(scale_invariance=True,
                        perceptual_weighting=hp["use_A_weighing"],
                        sample_rate=gs["sample_rate"])

    loss_curve = []
    pbar = tqdm(range(hp["max_epochs"]), desc="Training")

    for epoch in pbar:
        # Forward pass
        out = model(f0_frames=f0_frames.to(model.device),
                    input=input_signal.to(model.device),
                    input_sr=gs["sample_rate"],
                    direct=direct)

        # Calculate loss
        loss = loss_fn(out.unsqueeze(1),
                       target.to(model.device).unsqueeze(1))

        # Backward pass and optimization
        optimiser.zero_grad()
        loss.backward(retain_graph=True)
        optimiser.step()

        # Training progress tracking
        l = loss.item()
        loss_curve.append(l)
        pbar.set_postfix(loss=f"{l:.4f}")

    # Save loss curve
    plt.figure(figsize=(8, 3))
    plt.plot(loss_curve)
    plt.title("Training loss")
    plt.xlabel("epoch");
    plt.ylabel("STFT loss")
    plt.grid(True);
    plt.tight_layout()
    plt.savefig("analysis/loss_curve.png")

# -----------------------------------------------------------------------------
# Entry‑point ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main() -> None:
    sample_rate = gs["sample_rate"]

    # -------------------------------------------------------------------------
    # 3.  Load & prepare guitar -------------------------------------------------
    guitar = load_guitar("data/test.wav", sample_rate)
    save_audio("target.wav", guitar, sample_rate)

    # -------------------------------------------------------------------------
    # 4.  New random model ------------------------------------------------------
    seed = gs["random_seed"]
    model_opt = build_random_model(seed)

    # -------------------------------------------------------------------------
    # 5–6.  Forward / optimisation --------------------------------------------
    input_sig = guitar          # optimise against test.wav only
    direct_flag = False
    target_audio = guitar
    f0_frames_opt = torch.tensor([[mp['f0_hz']]])

    optimise_model(model_opt,
                   input_signal=input_sig,
                   target=target_audio,
                   f0_frames=f0_frames_opt,
                   direct=direct_flag)

    optim_audio = model_opt(f0_frames=f0_frames_opt.to(model_opt.device),
                            input=input_sig.to(model_opt.device),
                            input_sr=sample_rate,
                            direct=direct_flag).cpu()[0, ...]

    save_audio("optimized_model.wav", optim_audio, sample_rate)

    # -------------------------------------------------------------------------
    # Upsampled coefficients for comparison ------------------------------------
    _, l_b_opt, _, exc_b_opt = model_opt.get_upsampled_parameters(f0=f0_frames_opt,
                                                                  num_samples=LENGTH_N_UPSAMPLED)
    l_b_opt = l_b_opt[0].squeeze().detach().cpu().numpy()
    exc_b_opt = exc_b_opt[0].squeeze().detach().cpu().numpy()

    coeffs_dict = {
        "Loop coeff (opt.)": l_b_opt,  # shape (time, 2)  → DC + z‑1
        "Exc coeff (opt.)": exc_b_opt,  # shape (time, 5)
    }

    signals_dict = {
        "Target": target_audio,
        "Optimised": optim_audio,
    }

    # Calculate the number of samples for exc_length_s
    exc_len_samples = int(mp['exc_length_s'] * sample_rate)

    # Get the signals
    inv_sig = model_opt.get_inverse_filtered_signal().cpu()
    exc_after_opt = model_opt.exc_filter_out.cpu()

    # Only take the first exc_length_s of each signal
    signals_dict["Inverse filtered"] = inv_sig[:, :exc_len_samples]
    signals_dict["After excitation"] = exc_after_opt[:, :exc_len_samples]

    # -------------------------------------------------------------------------
    # Upsampled and constrained coefficients for comparison ---------------------
    # First get upsampled parameters
    _, l_b_opt_up, l_g_opt_up, exc_b_opt_up = model_opt.get_upsampled_parameters(
        f0=f0_frames_opt,
        num_samples=LENGTH_N_UPSAMPLED
    )
    # Then apply constraints to the upsampled parameters
    l_b_opt = model_opt.get_constrained_l_coefficients(l_b=l_b_opt_up, l_g=l_g_opt_up)
    exc_b_opt = model_opt.get_constrained_exc_coefficients(exc_b=exc_b_opt_up)
    l_b_opt = l_b_opt.squeeze().detach().cpu().numpy()
    exc_b_opt = exc_b_opt.squeeze().detach().cpu().numpy()

    coeffs_dict = {
        "Loop coeff (opt.)": l_b_opt,  # Now showing constrained values
        "Exc coeff (opt.)": exc_b_opt,  # Now showing constrained values
    }

    signals_dict = {
        "Target": target_audio,
        "Optimised": optim_audio,
    }

    # Calculate the number of samples for exc_length_s
    exc_len_samples = int(mp['exc_length_s'] * sample_rate)

    # Get the signals
    inv_sig = model_opt.get_inverse_filtered_signal().cpu()
    exc_after_opt = model_opt.exc_filter_out.cpu()

    # Only take the first exc_length_s of each signal
    signals_dict["Inverse filtered"] = inv_sig[:, 225:exc_len_samples + 225]
    signals_dict["After excitation"] = exc_after_opt[:, :exc_len_samples]

    composite_plot("analysis/composite.png", signals_dict, coeffs_dict)

    # -------------------------------------------------------------------------
    # 6.  Extra diagnostics for the trained model ------------------------------
    inv_signal = model_opt.get_inverse_filtered_signal().cpu()
    save_audio("inverse_filtered.wav", inv_sig, sample_rate)
    exc_after_opt = model_opt.exc_filter_out.cpu()

    # Plot only the first exc_length_s seconds
    exc_len_samples = int(mp['exc_length_s'] * sample_rate)

    plt.figure(figsize=(12, 4))
    plt.plot(inv_signal.squeeze().detach().cpu().numpy()[:exc_len_samples],
             label="Inverse filtered")
    plt.plot(exc_after_opt.squeeze().detach().cpu().numpy()[:exc_len_samples],
             label="After excitation", alpha=0.7)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("inverse_excitation_opt.png")

if __name__ == "__main__":
    main()