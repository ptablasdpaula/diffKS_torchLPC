from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Tuple, Dict, Any, Union

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm

import pygad
from auraloss.freq import MultiResolutionSTFTLoss as MultiSTFT

from diffKS import DiffKS
from utils import (noise_burst, load_config, resize_tensor_dim,)

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create output folders declared in the spec."""
    for p in ["plots", "audio/out"]:
        Path(p).mkdir(parents=True, exist_ok=True)


def save_audio(path: str | Path, tensor: torch.Tensor, sr: int) -> None:
    """Save *mono* tensor to WAV (expects shape [1, samples] or [samples])."""
    tensor = tensor.detach().cpu()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    torchaudio.save(str(path), tensor, sr)


def to_samples(f0_hz: float, sr: int) -> float:
    """Convert fundamental (Hz) to delay in *samples* (non‑integer)."""
    return sr / f0_hz


# -----------------------------------------------------------------------------
# Main experiment logic
# -----------------------------------------------------------------------------

def build_usual_ks(cfg_mp: Dict[str, Any], sr: int) -> Tuple[DiffKS, torch.Tensor]:
    """Return a *DiffKS* initialised to “usual” KS values plus an excitation
    burst long enough for *exc_length_s* from the config.
    """
    exc_len_s: float = cfg_mp["exc_length_s"]
    loop_order = cfg_mp["loop_order"]
    exc_order = cfg_mp["exc_order"]
    batch_size = cfg_mp["batch_size"]

    model = DiffKS(
        sample_rate=sr,
        min_f0_hz=cfg_mp["min_f0_hz"],
        loop_order=loop_order,
        loop_n_frames=cfg_mp["loop_n_frames"],
        exc_order=exc_order,
        exc_n_frames=cfg_mp["exc_n_frames"],
        exc_length_s=exc_len_s,
        interp_type=cfg_mp["interp_type"],
        use_double_precision=cfg_mp["use_double_precision"],
        batch_size=batch_size,
    )

    # --- Configure loop filter so it behaves like the classic KS averager -------
    loop_frames = cfg_mp["loop_n_frames"]

    loop_coeffs = torch.full((batch_size, loop_frames, loop_order + 1), 0.5)  # σ(0)=0.5 everywhere
    loop_coeffs[:, :, 0] = 0.2
    loop_coeffs[:, :, 1] = 0.8
    model.set_loop_coefficients(loop_coeffs)

    # overall feedback gain a whisker below 1.0  → very slow decay
    model.set_loop_gain(torch.full((batch_size, loop_frames, 1), 5.3))  # σ(5.3) ≈ 0.995

    # --- Excitation filter: gently varying small reflection coeffs ------------
    exc_frames = cfg_mp["exc_n_frames"]
    t = torch.linspace(0, 1, exc_frames).unsqueeze(-1)
    exc_coeffs = 0.2 * torch.sin(2 * math.pi * (1 + torch.arange(exc_order + 1)) * t)
    exc_coeffs = exc_coeffs.expand(batch_size, -1, -1)  # [batch_size, exc_n_frames, exc_order + 1]
    model.set_exc_coefficients(exc_coeffs)

    _, _, _, gs = load_config()

    # --- Generate noise burst -------------------------------------------------
    burst = noise_burst(
        sample_rate=sr,
        length_s=gs["length_audio_s"],
        burst_width_s=cfg_mp["burst_width_s"],
        normalize=cfg_mp["normalize_burst"],
        batch_size=batch_size,
    )

    return model, burst


def run_usual_ks(model: DiffKS, burst: torch.Tensor, sr: int,
                 cfg_mp: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor,
                                                  torch.Tensor]:
    """Propagate *burst* through *model* with base‑440 Hz plus vibrato."""
    f0_start = cfg_mp['f0_1_Hz']
    f0_end = cfg_mp['f0_2_Hz']
    n_frames = cfg_mp['f0_n_frames']

    batch_size = cfg_mp['batch_size']

    f0_hz = torch.linspace(f0_start, f0_end, n_frames)
    f0_frames = (sr / f0_hz).repeat(batch_size, 1)  # delay in *samples*

    audio = model(f0_frames=f0_frames.to(model.device),
                  input=burst.to(model.device),
                  direct=True)  # in‑domain generation

    exc_out = model.exc_filter_out  # [1, exc_len_samples]
    return audio.cpu(), exc_out.cpu(), f0_frames


def load_guitar(path: str | Path, sr_tgt: int) -> torch.Tensor:
    """Load *audio/guitar.wav* and resample to *sr_tgt* (mono)."""
    wav, sr_in = torchaudio.load(str(path))
    if wav.dim() > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)  # down‑mix
    if sr_in != sr_tgt:
        wav = torchaudio.transforms.Resample(sr_in, sr_tgt)(wav)
    return wav


def build_random_model(cfg_mp: Dict[str, Any], sr: int,
                       seed: int) -> DiffKS:
    """Create a *DiffKS* with random weights but fixed *seed*."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    loop_order = cfg_mp["loop_order"]
    loop_n_frames = cfg_mp["loop_n_frames"]
    exc_order = cfg_mp["exc_order"]
    exc_n_frames = cfg_mp["exc_n_frames"]
    batch_size = cfg_mp["batch_size"]

    model = DiffKS(
        sample_rate=sr,
        min_f0_hz=cfg_mp["min_f0_hz"],
        loop_order=loop_order,
        loop_n_frames=loop_n_frames,
        exc_order=exc_order,
        exc_n_frames=exc_n_frames,
        exc_length_s=cfg_mp["exc_length_s"],
        interp_type=cfg_mp["interp_type"],
        use_double_precision=cfg_mp["use_double_precision"],
        batch_size=batch_size,
    )

    model.set_loop_coefficients(torch.rand(batch_size, loop_n_frames, loop_order + 1))
    model.set_loop_gain(torch.rand((batch_size, loop_n_frames, 1),))
    model.set_exc_coefficients(torch.rand(batch_size, exc_n_frames, exc_order + 1) * 0.1)

    return model

# -----------------------------------------------------------------------------
# Composite plotting helpers
# -----------------------------------------------------------------------------

def composite_plot(fig_path: str,
                   signals: Dict[str, torch.Tensor],
                   coeffs: Dict[str, Union[np.ndarray, torch.Tensor]]) -> None:
    """
    Plot a set of waveforms (‘signals’) and time‑varying filter coefficients
    (‘coeffs’) on a single canvas and save to *fig_path*.

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
        ax.plot(wav.squeeze().detach().numpy())
        ax.set_title(name)
        ax.set_xlabel("samples")
        ax.grid(True)
        row += 1

    # --------------------------------------------------------------------- #
    # 2.  Coefficient trajectories
    # --------------------------------------------------------------------- #
    row = 0
    for name, traj in coeffs.items():
        ax = axs[row * 2 + 1]
        traj_np = traj if isinstance(traj, np.ndarray) else traj
        traj_np = traj_np if isinstance(traj_np, np.ndarray) else traj_np.cpu().numpy()

        if traj_np.ndim == 1:                       # single trajectory
            ax.plot(traj_np, label=name)
        else:                                       # multiple taps
            for k in range(traj_np.shape[1]):
                ax.plot(traj_np[:, k], label=f"{name}-b{k}")
        ax.set_title(name)
        ax.set_xlabel("samples")
        ax.grid(True)
        ax.legend(loc="upper right")
        row += 1

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)



# -----------------------------------------------------------------------------
# Training loop ----------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Training loop - works for BOTH in‑domain and out‑of‑domain cases
# -----------------------------------------------------------------------------
def optimise_model(model: DiffKS,
                   input_signal: torch.Tensor,    # what goes *into* the model
                   target      : torch.Tensor,    # waveform we want to match
                   f0_frames   : torch.Tensor,    # delay‑trajectory (ctrl‑rate)
                   direct      : bool,            # True→noise‑burst path
                   hp         : Dict[str, Any],) -> None:
    """
    Optimise *model* so that model(input_signal) ≈ target under STFT loss.

    Parameters
    ----------
    model : DiffKS
    input_signal : [B, samples] tensor fed to the forward‑pass
    target : [B, samples] reference audio
    f0_frames : [B, n_ctrl_frames] delay lengths in *samples*
    direct : bool
        Passed straight to ``DiffKS.forward``.  True ⇒ use the burst path
        (Karplus‑Strong); False ⇒ inverse‑filter path.
    hp : hyperparameter sub‑dict from config
    """
    _, mp, _, gs = load_config()

    method = hp.get('method', 'gradient')

    loss_curve = []

    if method == 'gradient':
        optimiser = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])

        loss_fn = MultiSTFT(scale_invariance=True,
                            perceptual_weighting=hp["use_A_weighing"],
                            sample_rate=gs["sample_rate"])

        bad_epochs = 0
        pbar = tqdm(range(hp["max_epochs"]), desc="Training")

        for epoch in pbar:
            # forward -------------------------------------------------------------
            out = model(f0_frames=f0_frames.to(model.device),
                        input=input_signal.to(model.device),
                        direct=direct)

            loss = loss_fn(out.unsqueeze(1),
                           target.to(model.device).unsqueeze(1)) * hp["stft_weight"]

            # backward ------------------------------------------------------------
            optimiser.zero_grad()
            loss.backward(retain_graph=True)
            optimiser.step()

            # bookkeeping ---------------------------------------------------------
            l = loss.item()
            loss_curve.append(l)
            pbar.set_postfix(loss=f"{l:.4f}")

            # early‑stopping ------------------------------------------------------
            bad_epochs = bad_epochs + 1 if l < hp["patience_delta"] else 0
            if bad_epochs >= hp["patience_epochs"]:
                break

    elif method == "genetic":
        loop_n_frames = mp["loop_n_frames"]
        exc_n_frames = mp["exc_n_frames"]

        loop_order = mp["loop_order"]
        exc_order = mp["exc_order"]

        use_double_precision = mp["use_double_precision"]

        total_loop_coeffs = loop_n_frames * (loop_order + 1)
        total_exc_coeffs = exc_n_frames * (exc_order + 1)

        num_generations = hp["max_epochs"]
        num_parents_mating = 18
        sol_per_pop = mp["batch_size"]

        loss_fn = MultiSTFT(scale_invariance=True,
                            perceptual_weighting=hp["use_A_weighing"],
                            sample_rate=gs["sample_rate"])

        def solution_to_coeffs(solution: np.ndarray):
            num_sols = solution.shape[0]
            start, end = 0, total_loop_coeffs
            loop_coeffs = solution[:, start:end].reshape((num_sols, loop_n_frames, loop_order + 1))

            start = end
            end = start + 1
            loop_gain = solution[:, start:end].reshape((num_sols, loop_n_frames, 1))

            start = end
            end = start + total_exc_coeffs
            assert end == solution.shape[-1]
            exc_coeffs = solution[:, start:end].reshape((num_sols, exc_n_frames, exc_order + 1))

            return loop_coeffs, loop_gain, exc_coeffs

        def fitness_func(ga_instance, solution, solution_idx):
            with torch.no_grad():
                loop_coeffs, loop_gain, exc_coeffs = solution_to_coeffs(solution)

                out = model(f0_frames=f0_frames.to(model.device),
                            input=input_signal.to(model.device),
                            direct=direct,
                            loop_coefficients=torch.from_numpy(loop_coeffs).to(model.device),
                            loop_gain=torch.from_numpy(loop_gain).to(model.device),
                            exc_coefficients=torch.from_numpy(exc_coeffs).to(model.device),
                            )

                tgt = target.to(model.device).unsqueeze(1)
                stft_loss = []
                for item in out:
                    stft_loss.append(-loss_fn(
                        item.view(1, 1, -1),
                        tgt
                    ).item())
                return stft_loss

        fitness_function = fitness_func

        num_genes = total_loop_coeffs + 1 + total_exc_coeffs

        init_range_low = -1
        init_range_high = 1

        parent_selection_type = "sss"
        keep_parents = 0

        crossover_type = "single_point"

        mutation_type = "random"
        mutation_percent_genes = 0.1

        def on_gen(ga_instance):
            print("Generation : ", ga_instance.generations_completed)

            fitness = ga_instance.best_solution()[1]
            loss_curve.append(fitness)
            print("Fitness of the best solution :", fitness)

        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=fitness_function,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            init_range_low=init_range_low,
            init_range_high=init_range_high,
            parent_selection_type=parent_selection_type,
            keep_parents=keep_parents,
            keep_elitism=0,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            # mutation_percent_genes=mutation_percent_genes,
            random_mutation_min_val=-0.1,
            random_mutation_max_val=0.1,
            mutation_probability=0.5,
            on_generation=on_gen,
            gene_type=np.float64 if use_double_precision else np.float32,
            fitness_batch_size=sol_per_pop,
        )

        ga_instance.run()

        loop_coeffs, loop_gain, exc_coeffs = solution_to_coeffs(ga_instance.best_solution()[0][None, :])

        loop_coeffs = torch.from_numpy(loop_coeffs).to(model.device).repeat(sol_per_pop, 1, 1)
        loop_gain = torch.from_numpy(loop_gain).to(model.device).repeat(sol_per_pop, 1, 1)
        exc_coeffs = torch.from_numpy(exc_coeffs).to(model.device).repeat(sol_per_pop, 1, 1)

        model.set_exc_coefficients(exc_coeffs)
        model.set_loop_coefficients(loop_coeffs)
        model.set_loop_gain(loop_gain)

    # ---------- save loss curve ---------------------------------------------
    plt.figure(figsize=(8, 3))
    plt.plot(loss_curve)
    plt.title("Training loss")
    plt.xlabel("epoch"); plt.ylabel("STFT loss")
    plt.grid(True); plt.tight_layout()
    plt.savefig("plots/loss_curve.png")
    plt.close()

# -----------------------------------------------------------------------------
# Entry‑point ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main() -> None:
    ensure_dirs()
    hp, mp, idp, gs = load_config()
    sample_rate: int = gs["sample_rate"]

    # -------------------------------------------------------------------------
    # 1–2.  In‑domain synthetic reference using “usual” KS ---------------------
    model_id, burst = build_usual_ks(mp, sample_rate)
    in_domain_audio, exc_after, f0_id = run_usual_ks(model_id, burst, sample_rate, mp)

    save_audio("audio/out/in_domain.wav", in_domain_audio, sample_rate)
    save_audio("audio/out/burst.wav", burst, sample_rate)
    save_audio("audio/out/burst_exc.wav", exc_after, sample_rate)

    # -------------------------------------------------------------------------
    # 3.  Load & prepare guitar -------------------------------------------------
    guitar = load_guitar("audio/guitar.wav", sample_rate)
    save_audio("audio/out/guitar_sr.wav", guitar, sample_rate)

    # -------------------------------------------------------------------------
    # 4.  New random model ------------------------------------------------------
    seed = gs.get("random_seed", 1234)
    model_opt = build_random_model(mp, sample_rate, seed)

    batch_size = mp["batch_size"]

    # -------------------------------------------------------------------------
    # 5–6.  Forward / optimisation --------------------------------------------
    use_in_domain = idp["use_in_domain"]
    if use_in_domain:
        input_sig = burst  # noise burst → direct=True
        direct_flag = True
        target_audio = in_domain_audio
    else:
        input_sig = guitar  # real audio → direct=False
        input_sig = input_sig.repeat((batch_size, 1))
        direct_flag = False
        target_audio = guitar

    f0_frames_opt = f0_id

    optimise_model(model_opt,
                   input_signal=input_sig,
                   target=target_audio,
                   f0_frames=f0_frames_opt,
                   direct=direct_flag,
                   hp=hp)

    optim_audio = model_opt(f0_frames=f0_frames_opt.to(model_opt.device),
                            input=input_sig.to(model_opt.device),
                            direct=direct_flag).cpu()[0, ...]

    save_audio("audio/out/optimized_model.wav", optim_audio, sample_rate)

    # -------------------------------------------------------------------------
    # Upsampled coefficients for comparison ------------------------------------
    n_samp = target_audio.shape[-1]
    _, l_b_opt, exc_b_opt = model_opt.get_upsampled_parameters(f0=f0_frames_opt,
                                                              num_samples=n_samp)
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

    if use_in_domain:  # direct = True
        signals_dict["Noise burst"] = resize_tensor_dim(burst, exc_after.size(1), 1)
        signals_dict["Burst after exc"] = exc_after
    else:  # direct = False
        inv_sig = model_opt.get_inverse_filtered_signal().cpu()
        exc_after_opt = model_opt.exc_filter_out.cpu()
        signals_dict["Inverse filtered"] = inv_sig
        signals_dict["After excitation"] = exc_after_opt

    # --- Coefficient comparison: in‑domain vs optimised ----------------------
    _, l_b_id, exc_b_id = model_id.get_upsampled_parameters(f0=f0_id,
                                                            num_samples=n_samp)
    _, l_b_opt, exc_b_opt = model_opt.get_upsampled_parameters(f0=f0_frames_opt,
                                                               num_samples=n_samp)

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # loop coefficients (all taps)
    for k in range(l_b_id.shape[-1]):
        ax[0].plot(l_b_id[0, :, k].cpu(), label=f'In‑domain b{k}', linewidth=1)
        ax[0].plot(l_b_opt[0, :, k].cpu(), label=f'Optimised b{k}', linestyle='--')
    ax[0].set_title('Loop coefficients (upsampled)')
    ax[0].legend();
    ax[0].grid(True)

    # excitation coefficients (all 5 taps)
    exc_len_samples = int(mp['exc_length_s'] * sample_rate)

    ax[1].set_xlim(0, exc_len_samples)  # limit x‑axis

    for k in range(exc_b_id.shape[-1]):
        ax[1].plot(exc_b_id[0, :exc_len_samples, k].detach(),
                   label=f'In‑domain a{k}', linewidth=1)
        ax[1].plot(exc_b_opt[0, :exc_len_samples, k].detach(),
                   label=f'Optimised a{k}', linestyle='--')

    ax[1].set_title('Excitation coefficients (upsampled)')
    ax[1].legend();
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig('plots/in_domain_vs_excitation.png')
    plt.close(fig)

    # add reference coeffs if in‑domain ----------------------------------------
    if use_in_domain:
        _, l_b_id, exc_b_id = model_id.get_upsampled_parameters(f0=f0_id,
                                                                num_samples=n_samp)
        if use_in_domain:  # add references when they exist
            coeffs_dict["Loop coeff(ref.)"] = l_b_id.squeeze().detach().numpy()
            coeffs_dict["Exc coeff (ref.)"] = exc_b_id.squeeze().detach().numpy()

    composite_plot("plots/composite.png", signals_dict, coeffs_dict)

    # -------------------------------------------------------------------------
    # 6.  Extra diagnostics for the trained model ------------------------------
    if not use_in_domain:
        inv_signal = model_opt.get_inverse_filtered_signal().cpu()
        save_audio("audio/out/inverse_filtered.wav", inv_sig, sample_rate)
        exc_after_opt = model_opt.exc_filter_out.cpu()
        plt.figure(figsize=(12, 4))
        plt.plot(inv_signal.squeeze().detach().numpy(), label="Inverse filtered")
        plt.plot(exc_after_opt.squeeze().detach().numpy(), label="After excitation", alpha=0.7)
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig("plots/inverse_excitation_opt.png")
        plt.close()

if __name__ == "__main__":
    main()
