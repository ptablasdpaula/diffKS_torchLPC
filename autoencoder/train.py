from tqdm import tqdm
import numpy as np
import soundfile as sf
import torch, torch.optim as optim, wandb
import torchaudio.functional as TAF
from third_party.auraloss.auraloss.freq import MultiResolutionSTFTLoss
from torch.utils.data import DataLoader
from .model import AE_KarplusModel, MfccTimeDistributedRnnEncoder
import argparse, os
import multiprocessing as mp
import psutil
from ddc_onset.constants import FRAME_RATE

import matplotlib.pyplot as plt

# ─── Training phase schedule ──────────────────────────────────────────────
STAGE1_STEPS = 500    # freeze DDC
STAGE2_STEPS = 2000   # unfreeze DDC, freeze coeffs
LR_MAIN      = 1e-4   # base LR (overridden by CLI)
LR_FINE      = 1e-5   # lower LR for fine‑tuning phases

# Global variable for internal sample rate used by plotting helpers
MODEL_INTERNAL_SR = None  # set in main() after model construction; used by plotting helpers

from paths import NSYNTH_PREPROCESSED_DIR
from data.preprocess import NsynthDataset
from utils import get_device, str2bool

def parse_args():
    p = argparse.ArgumentParser()
    env = os.environ.get

    # ─── Run identification ────────────────────────────────────────────────
    p.add_argument("--name", type=str, default=env("NAME", "exp"),
                   help="Unique experiment name (used for checkpoints & wandb run)")
    p.add_argument("--continue_from_checkpoint", action="store_true",
                   default=str2bool(env("CONTINUE_FROM_CHECKPOINT", "false")),
                   help="Resume training from latest checkpoint for this --name")

    # ─── Optimisation hyper-parameters ─────────────────────────────────────
    p.add_argument("--learning_rate", type=float, default=float(env("LEARNING_RATE", 1e-2)))
    p.add_argument("--max_epochs", type=int, default=int(env("MAX_EPOCHS", 330)))
    p.add_argument("--patience", type=int, default=int(env("PATIENCE", 20)))
    p.add_argument("--min_delta", type=float, default=float(env("MIN_DELTA", 0.001)),
                   help="Minimum relative (fractional) validation-loss improvement to reset patience. 0.001 = 0.1 %")

    # ─── Data-loading ──────────────────────────────────────────────────────
    p.add_argument("--batch_size", type=int, default=int(env("BATCH_SIZE", 1)))
    p.add_argument("--num_workers", type=int, default=int(env("NUM_WORKERS", 2)))

    # ─── Model size ────────────────────────────────────────────────────────
    p.add_argument("--hidden_size", type=int, default=int(env("HIDDEN_SIZE", 512)))

    # ─── DiffKS filter configuration ───────────────────────────────────────
    p.add_argument("--l_order", type=int, default=int(env("L_ORDER", 2)))
    p.add_argument("--exc_order", type=int, default=int(env("EXC_ORDER", 5)))

    # ─── Dataset filters ────────────────────────────────────────────────────
    p.add_argument("--families", type=str, default=env("FAMILIES", "guitar"))
    p.add_argument("--sources", type=str, default=env("SOURCES", "acoustic"))

    # ─── DiffKS decoder settings ───────────────────────────────────────────
    p.add_argument("--interpolation_type", type=str, default=env("INTERPOLATION_TYPE", "linear"))
    p.add_argument("--pitch_mode", type=str, default=env("PITCH_MODE", "meta"))

    # ─── Onset fine‑tune schedule ─────────────────────────────────────────
    p.add_argument("--unfreeze_onset_after", type=int,
                   default=int(env("UNFREEZE_ONSET_AFTER", 500)),
                   help="Global training step after which the ddc_onset sub‑modules will be unfrozen (0 = train from start).")

    p.add_argument("--batches_per_epoch", type=int, default=int(env("BATCHES_PER_EPOCH", 100)))

    # Stage transition steps CLI arguments
    p.add_argument("--stage1_steps", type=int, default=int(env("STAGE1_STEPS", 500)),
                   help="Step to transition from stage 0 (DDC frozen) to stage 1 (DDC unfrozen)")
    p.add_argument("--stage2_steps", type=int, default=int(env("STAGE2_STEPS", 2000)),
                   help="Step to transition from stage 1 (coeff frozen) to stage 2 (all unfrozen, fine LR)")

    return p.parse_args()



# -----------------------------------------------------------------
def build_optimizer(model, phase: int):
    """
    phase 0 : train **coefficient‑prediction network** (everything except placement_cnn) at LR_MAIN
    phase 1 : train **placement_cnn (DDC onset CNN)** only at LR_MAIN
    phase 2 : train **all** (coeff predictor + placement_cnn) jointly at LR_FINE

    Decoder parameters (`decoder.*`) and spectrogram front‑end parameters
    (`spec_extractor.*`, `spec_normalizer.*`) always remain frozen.
    """
    decoder_params     = []
    placement_params   = []
    spec_params        = []   # spec_extractor + spec_normalizer
    other_params       = []
    coeff_params       = []

    for name, p in model.named_parameters():
        if name.startswith("decoder."):
            decoder_params.append(p)
        elif name.startswith("placement_cnn."):
            placement_params.append(p)
        elif name.startswith(("spec_extractor.", "spec_normalizer.")):
            spec_params.append(p)
        elif name.startswith("coefficients."):
            coeff_params.append(p)
        else:
            other_params.append(p)

    # --- 1. Always freeze the decoder and spec_extractor/normalizer ------
    for p in decoder_params + spec_params:
        p.requires_grad = False

    # --- 2. Phase‑specific rules -------------------------------------------
    if phase == 0:
        # Phase 0 – train the whole coefficient‑prediction network
        # (z_encoder + in/out MLPs + GRU + coefficients head).
        # Freeze placement_cnn.
        for p in coeff_params + other_params:
            p.requires_grad = True
        for p in placement_params:
            p.requires_grad = False
        param_groups = [
            {"params": coeff_params + other_params, "lr": LR_MAIN},
        ]

    elif phase == 1:
        # Phase 1 – train the DDC onset CNN only; freeze coefficient network.
        for p in placement_params:
            p.requires_grad = True
        for p in coeff_params + other_params:
            p.requires_grad = False
        param_groups = [
            {"params": placement_params, "lr": LR_MAIN},
        ]

    else:
        # Phase 2 – fine‑tune everything jointly (except decoder/spec) at LR_FINE
        for p in coeff_params + placement_params + other_params:
            p.requires_grad = True
        param_groups = [
            {"params": coeff_params + placement_params + other_params, "lr": LR_FINE},
        ]

    return optim.Adam(param_groups)
# -----------------------------------------------------------------

def main():
    args = parse_args()
    # Override stage transition steps from CLI
    global STAGE1_STEPS, STAGE2_STEPS
    STAGE1_STEPS = args.stage1_steps
    STAGE2_STEPS = args.stage2_steps
    config = {
        "hidden_size": args.hidden_size,
        "loop_order": args.l_order,
        "exc_order": args.exc_order,
        "sample_rate": 16000,
        "ks_sample_rate": 44100,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "eval_interval": 1,
        "save_dir": f"autoencoder/runs/{args.name}",
        "families": [f.strip() for f in args.families.split(",")],
        "sources": [s.strip() for s in args.sources.split(",")],
        "num_workers": args.num_workers,
        "interpolation_type": args.interpolation_type,
        "pitch_mode": args.pitch_mode,
        "unfreeze_onset_after": args.unfreeze_onset_after,
        "batches_per_epoch": args.batches_per_epoch,
    }


    print("\n▶ Running with config:")
    for k, v in vars(args).items():
        print(f"   {k:15}: {v}")

    # ─── device init ───────────────────────────────────────────────────────
    device = get_device()
    print(f"Using device: {device}")

    # ─── WandB init ────────────────────────────────────────────────────────
    wandb_id = None
    latest_ckpt = os.path.join(config["save_dir"], f"latest_model_{args.name}.pth")
    if args.continue_from_checkpoint and os.path.exists(latest_ckpt):
        tmp_ckpt = torch.load(latest_ckpt, map_location="cpu")
        wandb_id = tmp_ckpt.get("wandb_id", None)
        print(f"[INFO] Found checkpoint – will resume run id {wandb_id}")

    autoencoder_dir = os.path.dirname(os.path.abspath(__file__))
    wandb_run = wandb.init(project="diffks-autoencoder", name=args.name, dir=autoencoder_dir,
                           id=wandb_id, resume="allow", config=config)
    if wandb_id is None:
        wandb_id = wandb_run.id  # store for fresh runs

    # ----- WandB logging helpers (single global step; let wandb manage) -----
    def log_train_batch(value: float):
        if wandb.run is not None:
            wandb.log({"train loss per batch": value})

    def log_epoch(train_loss: float, val_loss: float):
        if wandb.run is not None:
            wandb.log({
                "train loss per epoch": train_loss,
                "val loss per epoch":   val_loss,
            })

    # ─── RAM check ────────────────────────────────────────────── #
    process = psutil.Process(os.getpid())
    print(f"[INFO] Memory at start: {process.memory_info().rss / 1024 ** 3:.2f} GB")

    # ─── Create save directories ───────────────────────────────── #
    full_save_path = os.path.abspath(config["save_dir"])
    os.makedirs(full_save_path, exist_ok=True)
    print(f"Using save directory: {full_save_path}")

    # ─── Data ─────────────────────────────────────────────────────────────
    dataset = NsynthDataset(root=NSYNTH_PREPROCESSED_DIR,
                            split="test",
                            pitch_mode=config["pitch_mode"],
                            families=config["families"],
                            sources=config["sources"], )

    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                              drop_last=True, pin_memory=True if device.type != "mps" else False,
                              num_workers=config["num_workers"])

    val_dataset = NsynthDataset(root=NSYNTH_PREPROCESSED_DIR,
                                split="test",
                                pitch_mode=config["pitch_mode"],
                                families=config["families"],
                                sources=config["sources"], )

    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                            drop_last=True, pin_memory=True if device.type != "mps" else False, num_workers=config["num_workers"])

    # ---- Fixed batch for consistent logging across epochs ----
    fixed_audio, fixed_pitch, fixed_loud = next(iter(val_loader))
    n_plot = min(fixed_audio.size(0), 5)

    # ─── Start Model, optimizer & Loss ────────────────────────── #
    model = AE_KarplusModel(
        batch_size=config["batch_size"],
        hidden_size=config["hidden_size"],
        loop_order=config["loop_order"],
        exc_order=config["exc_order"],
        internal_sr=config["ks_sample_rate"],
        interpolation_type=config["interpolation_type"],
        z_encoder=MfccTimeDistributedRnnEncoder(),
    ).to(device)
    # Set global for plotting helpers
    global MODEL_INTERNAL_SR
    MODEL_INTERNAL_SR = model.internal_sr

    current_phase = 0  # start at phase 0 – coefficients only
    optimizer = build_optimizer(model, current_phase)
    global_step = 0

    mr_stft = MultiResolutionSTFTLoss(scale_invariance=True, perceptual_weighting=True,
                                      sample_rate=config["sample_rate"], device=device, )

    # ─── Resume from checkpoint if requested ──────────────────────────────
    start_epoch, best_val_loss = 0, float('inf')
    if args.continue_from_checkpoint and os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device)

        # strip out the two analysis buffers so their old shapes don't conflict
        sd = ckpt["model_state_dict"]
        for key in list(sd):
            if "ks_inverse_signal" in key or "excitation_filter_out" in key:
                sd.pop(key)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)

        print(f"[RESUME] Starting at epoch {start_epoch} (best so far {best_val_loss:.4f})")

    bpe = min(len(train_loader), config["batches_per_epoch"])

    # ─── Early-stopping bookkeeping ───────────────────────────── #
    epochs_since_improve = 0

    # ───────────────────────── training epochs ───────────────────────────
    print(f"[INFO] Training with variable‑length triggers; ddc_onset unfrozen after step {config['unfreeze_onset_after']}.")
    for epoch in range(start_epoch, config["max_epochs"]):
        model.train()
        t_loss = 0
        batches_processed = 0

        torch.autograd.set_detect_anomaly(True)

        # ─── Training step ───────────────────────────────────────────────
        for batch_idx, (audio, pitch, loud) in enumerate(tqdm(train_loader, desc=f"[E{epoch:03d} train]")):
            for name, param in model.named_parameters():
                print(f"{name}: requires_grad={param.requires_grad}")

            if batch_idx >= config["batches_per_epoch"]:
                break
            audio, pitch, loud = audio.to(device), pitch.to(device), loud.to(device)
            recon = model(
                pitch=pitch,
                loudness=loud,
                audio=audio,
                audio_sr=config["sample_rate"],
            )
            # Sanity‑check: recon must come back at the original sample‑rate
            assert recon.shape[1] == audio.shape[1], (
                f"Decoder returned {recon.shape[1]} samples, "
                f"but target has {audio.shape[1]}. "
                "This usually means an incorrect `audio_sr` was passed to the model."
            )
            stft_loss = mr_stft(recon.unsqueeze(1), audio.unsqueeze(1))
            loss = stft_loss
            optimizer.zero_grad()
            loss.backward()
            # Debug: print gradient mean for all parameters after backward
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"[DEBUG TRAIN LOOP] {name}: grad mean {param.grad.abs().mean().item():.6f}")
                else:
                    print(f"[DEBUG TRAIN LOOP] {name}: NO GRAD")
            optimizer.step()
            global_step += 1
            # ---- phase scheduler --------------------------------------------------
            if current_phase == 0 and global_step >= STAGE1_STEPS:
                current_phase = 1
                optimizer = build_optimizer(model, current_phase)
                print(f"[SCHED] Step {global_step}: phase 1 – now training DDC CNN only (coefficients frozen, LR={LR_MAIN})")
            elif current_phase == 1 and global_step >= STAGE2_STEPS:
                current_phase = 2
                optimizer = build_optimizer(model, current_phase)
                print(f"[SCHED] Step {global_step}: phase 2 – joint fine‑tuning (LR={LR_FINE})")
            log_train_batch(loss.item())
            t_loss += loss.item()
            batches_processed += 1

        if batches_processed > 0:
            t_loss /= batches_processed

        # ─── VALID ───────────────────────────────────────────────────────
        if epoch % config["eval_interval"] == 0:
            model.eval()
            v_losses = []
            with torch.no_grad():
                for audio, pitch, loud in val_loader:
                    audio, pitch, loud = audio.to(device), pitch.to(device), loud.to(device)
                    recon = model(
                        pitch=pitch,
                        loudness=loud,
                        audio=audio,
                        audio_sr=config["sample_rate"],
                    )
                    # Sanity‑check: recon must come back at the original sample‑rate
                    assert recon.shape[1] == audio.shape[1], (
                        f"Decoder returned {recon.shape[1]} samples, "
                        f"but target has {audio.shape[1]}. "
                        "This usually means an incorrect `audio_sr` was passed to the model."
                    )
                    stft_v = mr_stft(recon.unsqueeze(1), audio.unsqueeze(1))
                    batch_v = stft_v.item()
                    v_losses.append(batch_v)
            v_loss = float(np.mean(v_losses))
        else:
            v_loss = np.nan

        # ─── LOGGING ───────────────────────────────────────────────────────
        log_epoch(t_loss, v_loss)

        # ─── CHKPTS  ───────────────────────────────────────────────────────
        improved = False
        if not np.isnan(v_loss):
            # Relative improvement wrt best
            if best_val_loss == float('inf') or (best_val_loss - v_loss) / best_val_loss >= config["min_delta"]:
                improved = True
                best_val_loss = v_loss
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
        else:
            epochs_since_improve += 1  # treat missing val as no improvement

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "wandb_id": wandb_id,
        }
        torch.save(ckpt, latest_ckpt)
        if improved:
            torch.save(ckpt, os.path.join(config["save_dir"], f"best_model_{args.name}.pth"))

        # ─── AUDIO LOGS ──────────────────────────────────────────────────
        if epoch % config["eval_interval"] == 0:
            with torch.no_grad():
                # Use the same validation batch every epoch for consistent media logging
                a, p, l = fixed_audio.to(device), fixed_pitch.to(device), fixed_loud.to(device)
                # use model.last_salience computed in forward (already in [B, T_sal] at 100 Hz)
                # forward pass (freeze schedule still respected)
                rec = model(
                    pitch=p,
                    loudness=l,
                    audio=a,
                    audio_sr=config["sample_rate"],
                )
                # Sanity‑check: rec must come back at the original sample‑rate
                assert rec.shape[1] == a.shape[1], (
                    f"Decoder returned {rec.shape[1]} samples, "
                    f"but target has {a.shape[1]}. "
                    "This usually means an incorrect `audio_sr` was passed to the model."
                )

                # grab *active* trigger times (weights > 0.5)
                trig_times_s = model.last_true_trigger_times_s.detach().cpu().numpy()  # [B,K_true]

                # ---- Diagnostic: parameter-only forward to get constrained coeff frames and upsample to per-sample trajectories ----
                with torch.no_grad():
                    # constrained, encoder‑frame‑rate coefficients
                    l_b_frames, exc_b_frames = model(
                        pitch=p,
                        loudness=l,
                        audio=a,
                        audio_sr=config["sample_rate"],
                        return_parameters=True,
                    )

                n_internal = int(round(a.size(1) * model.internal_sr / config["sample_rate"]))

                _, loop_traj, _, exc_traj = model.decoder.get_upsampled_parameters(
                    p.squeeze(-1),
                    n_internal,
                    l_b=l_b_frames.to(model.decoder.device),
                    l_g=torch.ones_like(l_b_frames),
                    exc_b=exc_b_frames.to(model.decoder.device),
                )

                loop_traj_np = loop_traj.detach().cpu().numpy()
                exc_traj_np = exc_traj.detach().cpu().numpy()

                # Helper: interpolate a 1‑D signal onto the 16kHz time grid used by the waveforms
                def _interp_to_t_wave(sig_1d, orig_sr, target_t):
                    t_orig = np.arange(sig_1d.shape[0]) / float(orig_sr)
                    return np.interp(target_t, t_orig, sig_1d)

                # media logging: use the fixed batch, n_plot is already defined above

                media_log = {}
                for idx in range(n_plot):
                    k = idx  # keep original naming for logging
                    # prepare concatenated waveform original||recon
                    wave_orig = a[idx].cpu().numpy()
                    wave_rec  = rec[idx].cpu().numpy()
                    sample = np.concatenate([wave_orig, wave_rec], axis=0)

                    # normalize copy for playback (avoid inaudibly small or clipped audio)
                    peak = float(np.max(np.abs(sample))) if sample.size > 0 else 0.0
                    if peak > 0:
                        sample_play = sample / peak * 0.99
                    else:
                        sample_play = sample.copy()

                    # write wav
                    wav_name = f"sample_e{epoch}_{k}.wav"
                    wav_path = os.path.join(config["save_dir"], wav_name)
                    sf.write(wav_path, sample, config["sample_rate"])
                    print(f"[AUDIO] wrote: {os.path.abspath(wav_path)}")

                    # overlay triggers on both segments
                    if trig_times_s.ndim == 2 and idx < trig_times_s.shape[0]:
                        trig_s = trig_times_s[idx]
                        seg_dur_s = wave_orig.shape[0] / float(config["sample_rate"])
                        # only triggers inside the original segment
                        trig_concat_s = trig_s
                    else:
                        trig_concat_s = None

                    # ---- Composite figure with waveform, reconstruction, inverse signal, and coefficients ----
                    fig, axes = plt.subplots(7, 1, figsize=(10, 16))

                    # 1) Target waveform with salience & triggers
                    t_wave = np.arange(wave_orig.shape[0]) / float(config["sample_rate"])
                    axes[0].plot(t_wave, wave_orig, linewidth=1.0, label="target waveform")
                    axes[0].set_ylim(-1, 1)

                    # twin y‑axis reserved for salience only
                    ax_twin = axes[0].twinx()
                    ax_twin.set_ylim(-1, 1)

                    # === overlay raw post‑tanh salience (upsampled to 16 kHz) ===
                    if hasattr(model, "last_salience"):
                        sal_100 = model.last_salience[idx].cpu().numpy()           # (T_sal,)
                        t_sal   = np.arange(sal_100.shape[0]) / FRAME_RATE         # seconds
                        # use raw tanh output (already in −1..1): no normalization
                        sal_up = np.interp(t_wave, t_sal, sal_100)                 # to 16 kHz grid
                        ax_twin.plot(t_wave, sal_up, color="magenta", lw=1.5,
                                     label="DDC salience")
                        # draw the STEPeakPick threshold
                        thr = model.trigger_temp
                        ax_twin.axhline(thr, color="orange", ls="--", lw=1, alpha=0.8)
                        ax_twin.set_ylim(-1, 1)

                        # plot trigger dots only on FIRST half (original clip)
                        if trig_s is not None:
                            trig_mask = (trig_s >= 0.0) & (trig_s <= t_wave[-1])
                            if trig_mask.any():
                                sal_vals = np.interp(trig_s[trig_mask], t_wave, sal_up)
                                ax_twin.scatter(trig_s[trig_mask], sal_vals,
                                                color="red", marker="o", s=50,
                                                label="triggers", zorder=6)

                    # collect legend entries from both y‑axes
                    h0, l0 = axes[0].get_legend_handles_labels()
                    h1, l1 = ax_twin.get_legend_handles_labels()
                    handles_all = h0 + h1
                    labels_all  = l0 + l1

                    keep_labels = ["DDC salience", "triggers"]   # desired entries
                    filtered = [(h, l) for h, l in zip(handles_all, labels_all)
                                if l in keep_labels]
                    if filtered:
                        h_keep, l_keep = zip(*filtered)
                        axes[0].legend(h_keep, l_keep, loc="upper right",
                                       fontsize="x-small")

                    # 2) Reconstruction
                    axes[1].plot(t_wave, wave_rec, label="recon")
                    axes[1].set_ylim(-1, 1)
                    axes[1].set_ylabel("Amplitude")
                    axes[1].legend(fontsize="x-small", loc="upper right")

                    # 3) Inverse-filtered signal
                    # resample from internal_sr back to 16 kHz for alignment
                    inv_sig_internal = model.decoder.get_inverse_filtered_signal()[idx].cpu()
                    inv_sig_16k = TAF.resample(
                        inv_sig_internal.unsqueeze(0),
                        orig_freq=model.internal_sr,
                        new_freq=config["sample_rate"]
                    ).squeeze(0)
                    inv_sig = inv_sig_16k.numpy()
                    t_inv = np.arange(inv_sig.shape[0]) / float(config["sample_rate"])
                    axes[2].plot(t_inv, inv_sig, label="inverse")
                    axes[2].set_ylim(-1, 1)
                    axes[2].set_ylabel("Amplitude")
                    axes[2].legend(fontsize="x-small", loc="upper right")

                    # --- Loop coefficients (resampled to 16kHz) ---
                    loop_np_k = loop_traj_np[idx]  # [N_int, L+1]
                    loop_np_k_16k = np.stack([
                        _interp_to_t_wave(loop_np_k[:, j], model.internal_sr, t_wave)
                        for j in range(loop_np_k.shape[1])
                    ], axis=1)  # [T_wave, L+1]
                    for j in range(loop_np_k_16k.shape[1]):
                        axes[3].plot(t_wave, loop_np_k_16k[:, j], label=f"loop[{j}]")
                    axes[3].set_ylabel("loop coeffs")
                    if loop_np_k_16k.shape[1] <= 8:
                        axes[3].legend(fontsize="x-small", ncol=2, loc="upper right")

                    # --- Excitation coefficients (resampled to 16kHz) ---
                    exc_np_k = exc_traj_np[idx]  # [N_int, E+1]
                    exc_np_k_16k = np.stack([
                        _interp_to_t_wave(exc_np_k[:, j], model.internal_sr, t_wave)
                        for j in range(exc_np_k.shape[1])
                    ], axis=1)  # [T_wave, E+1]
                    for j in range(exc_np_k_16k.shape[1]):
                        axes[4].plot(t_wave, exc_np_k_16k[:, j], label=f"exc[{j}]")
                    axes[4].set_ylabel("exc coeffs")
                    axes[4].set_xlabel("Time (s)")
                    if exc_np_k_16k.shape[1] <= 8:
                        axes[4].legend(fontsize="x-small", ncol=2, loc="upper right")

                    # 6) Raw loop‑coefficient frames (pre‑trigger)
                    raw_loop = model.last_loop_coeff_frames[idx].cpu().numpy()    # [T_enc, L+1]
                    t_frames = np.linspace(0, t_wave[-1], raw_loop.shape[0])      # align with clip duration
                    for j in range(raw_loop.shape[1]):
                        axes[5].plot(t_frames, raw_loop[:, j], label=f"loop_raw[{j}]")
                    raw_loop_gain = model.last_loop_gain_frames[idx].cpu().numpy()  # [T_enc, 1]
                    axes[5].plot(t_frames, raw_loop_gain[:, 0], linestyle='--', linewidth=2.0, label='loop_gain')
                    axes[5].set_ylabel("raw loop")
                    if raw_loop.shape[1] <= 8:
                        axes[5].legend(fontsize="x-small", ncol=2, loc="upper right")

                    # 7) Raw excitation‑coefficient frames (pre‑trigger)
                    raw_exc = model.last_exc_coeff_frames[idx].cpu().numpy()      # [T_enc, E+1]
                    for j in range(raw_exc.shape[1]):
                        if j == 0:
                            axes[6].plot(t_frames, raw_exc[:, j], linestyle='--', linewidth=2.0, label='exc_gain')
                        else:
                            axes[6].plot(t_frames, raw_exc[:, j], label=f"exc_raw[{j}]")
                    axes[6].set_ylabel("raw exc")
                    axes[6].set_xlabel("Time (s)")
                    if raw_exc.shape[1] <= 8:
                        axes[6].legend(fontsize="x-small", ncol=2, loc="upper right")

                    fig.tight_layout()

                    if wandb.run is not None:
                        # log the composite figure and the normalized audio comparison only
                        media_log[f"composite_plot_{k}"] = wandb.Image(fig)
                        media_log[f"audio_compare_{k}"] = wandb.Audio(
                            sample_play,
                            sample_rate=config["sample_rate"],
                            caption=f"epoch {epoch} | sample {k} | orig+recon (norm)",
                        )

                    plt.close(fig)

                    # Separate plot: resonator matrix taps
                    R = model.decoder.resonator_matrix[idx].cpu().numpy()  # [N_int, D]
                    t_R = np.arange(R.shape[0]) / float(MODEL_INTERNAL_SR)
                    fig_res, ax_res = plt.subplots(figsize=(10, 4))
                    color_cycle = plt.cm.tab10(np.linspace(0, 1, R.shape[1]))
                    non_zero_labels = []
                    for d in range(R.shape[1]):
                        if np.allclose(R[:, d], 0.0):
                            continue  # skip all‑zero taps
                        lbl = f"tap {d}"
                        ax_res.plot(t_R, R[:, d],
                                    lw=1.2,
                                    color=color_cycle[d % 10],
                                    label=lbl)
                        non_zero_labels.append(lbl)

                    if non_zero_labels:
                        ax_res.legend(fontsize="x-small", ncol=4, loc="upper right")
                    ax_res.set_title(f"Resonator matrix taps sample {idx}")
                    ax_res.set_xlabel("Time (s)")
                    ax_res.set_ylabel("Coefficient value")
                    fig_res.tight_layout()
                    if wandb.run is not None:
                        media_log[f"resonator_matrix_{idx}"] = wandb.Image(fig_res)
                    plt.close(fig_res)

                # --- log mean trigger count (deduplicated) -----------------------
                if wandb.run is not None and trig_times_s.ndim == 2:
                    # Only count triggers within the original clip duration
                    dur = fixed_audio.shape[1] / float(config["sample_rate"])
                    counts_unique = [
                        len(np.unique(row[(row >= 0.0) & (row <= dur)]))
                        for row in trig_times_s
                    ]
                    media_log["trigger_count_mean"] = float(np.mean(counts_unique))

                    # log all items together
                    if wandb.run is not None and len(media_log) > 0:
                        wandb.log(media_log, commit=True)

        print(f"[E{epoch}] train={t_loss:.4f} val={v_loss:.4f} best={best_val_loss:.4f} (no-improve {epochs_since_improve}/{config['patience']})")

    wandb.finish()


if __name__ == '__main__':
    mp.freeze_support()
    main()
