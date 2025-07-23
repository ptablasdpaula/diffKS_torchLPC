from tqdm import tqdm
import numpy as np
import soundfile as sf
import torch, torch.optim as optim, wandb
from third_party.auraloss.auraloss.freq import MultiResolutionSTFTLoss
from torch.utils.data import DataLoader
from .model import AE_KarplusModel, MfccTimeDistributedRnnEncoder
import argparse, os
import multiprocessing as mp
import psutil
from ddc_onset.constants import FRAME_RATE
import matplotlib.pyplot as plt

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
    p.add_argument("--learning_rate", type=float, default=float(env("LEARNING_RATE", 1e-4)))
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

    return p.parse_args()

def plot_wave_with_triggers(wave_np, sr, trig_s, title=None):
    """
    wave_np: 1D numpy array audio samples
    sr: sample rate (int)
    trig_s: 1D numpy array of trigger times in seconds
    """
    import numpy as np
    import matplotlib.pyplot as plt

    t = np.arange(len(wave_np)) / float(sr)
    # NOTE: we no longer draw vertical lines; red X markers are easier to see on WandB dark/light themes.
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(t, wave_np)
    if trig_s is not None and len(trig_s) > 0:
        # plot red X markers at the waveform value nearest each trigger time
        trig_s = np.asarray(trig_s, dtype=float)
        # clip in-range
        trig_s = trig_s[(trig_s >= 0.0) & (trig_s <= t[-1])]
        # drop duplicate trigger times so plots stay uncluttered
        trig_s = np.unique(trig_s)  # ascending order, duplicates removed
        if trig_s.size > 0:
            trig_idx = np.clip((trig_s * sr).astype(int), 0, len(wave_np) - 1)
            trig_y = wave_np[trig_idx]
            ax.plot(trig_s, trig_y, 'rx', markersize=5, mew=1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amp")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


# Helper to plot loop and excitation coefficient trajectories, with optional trigger markers
def plot_coeffs_with_triggers(loop_traj, exc_traj, sr_vis, trig_s=None, title=None):
    """
    loop_traj: [N, Lc] constrained loop coefficients at internal SR
    exc_traj: [N, Ec] constrained excitation coefficients at internal SR
    sr_vis: target sample-rate for x-axis (e.g. outer audio SR, 16 kHz)
    trig_s: optional 1D array of trigger times in seconds (outer SR domain)
    title: optional string
    """
    import numpy as np
    import matplotlib.pyplot as plt
    n = loop_traj.shape[0]
    # Determine internal sample rate for seconds axis
    sr_int = MODEL_INTERNAL_SR if MODEL_INTERNAL_SR is not None else sr_vis * (44100.0 / 16000.0)
    t = np.arange(n) / float(sr_int)
    fig, axs = plt.subplots(2, 1, figsize=(8, 3.5), sharex=True)
    # Plot loop coefficients
    lc = loop_traj.shape[1]
    for i in range(lc):
        axs[0].plot(t, loop_traj[:, i], label=f"loop[{i}]")
    axs[0].set_ylabel("coeff")
    axs[0].set_title("Loop coefficients")
    # Plot excitation coefficients
    ec = exc_traj.shape[1]
    for i in range(ec):
        axs[1].plot(t, exc_traj[:, i], label=f"exc[{i}]")
    axs[1].set_ylabel("coeff")
    axs[1].set_title("Excitation coefficients")
    axs[1].set_xlabel("Time (s)")
    # Add legends if number of traces is small
    if lc <= 8:
        axs[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    if ec <= 8:
        axs[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    # Plot trigger markers if provided
    if trig_s is not None and len(trig_s) > 0:
        trig_s = np.asarray(trig_s, dtype=float)
        trig_s = trig_s[(trig_s >= 0.0) & (trig_s <= t[-1])]
        if trig_s.size > 0:
            # For each trigger, place a red 'x' at y=0 on both axes
            axs[0].plot(trig_s, np.zeros_like(trig_s), 'rx', markersize=5, mew=1.0)
            axs[1].plot(trig_s, np.zeros_like(trig_s), 'rx', markersize=5, mew=1.0)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig

def main():
    args = parse_args()
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

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

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

        # ─── Training step ───────────────────────────────────────────────
        for batch_idx, (audio, pitch, loud) in enumerate(tqdm(train_loader, desc=f"[E{epoch:03d} train]")):
            if batch_idx >= config["batches_per_epoch"]:
                break
            audio, pitch, loud = audio.to(device), pitch.to(device), loud.to(device)
            recon = model(
                pitch=pitch,
                loudness=loud,
                audio=audio,
                audio_sr=config["sample_rate"],
                unfreeze_onset_after=config["unfreeze_onset_after"],
            )
            loss = mr_stft(recon.unsqueeze(1), audio.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
                        unfreeze_onset_after=config["unfreeze_onset_after"],
                    )
                    batch_v = mr_stft(recon.unsqueeze(1), audio.unsqueeze(1)).item()
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
                # forward pass (freeze schedule still respected)
                rec = model(
                    pitch=p,
                    loudness=l,
                    audio=a,
                    audio_sr=config["sample_rate"],
                    unfreeze_onset_after=config["unfreeze_onset_after"],
                )

                # grab trigger times (seconds)
                trig_times_s = model.last_trigger_times_s.detach().cpu().numpy()  # [B,K]

                # ---- Diagnostic: parameter-only forward to get constrained coeff frames and upsample to per-sample trajectories ----
                with torch.no_grad():
                    l_b_frames, exc_b_frames = model(
                        pitch=p,
                        loudness=l,
                        audio=a,
                        audio_sr=config["sample_rate"],
                        unfreeze_onset_after=config["unfreeze_onset_after"],
                        return_parameters=True,
                    )
                trig_n = (model.last_trigger_times_s * model.internal_sr).long()  # [B, Fmax]
                n_internal = int(round(a.size(1) * model.internal_sr / config["sample_rate"]))
                loop_traj = model.decoder._upsample_by_triggers(
                    l_b_frames.to(model.decoder.device), trig_n.to(model.decoder.device), n_internal, mode=model.decoder.upsample_mode)
                exc_traj = model.decoder._upsample_by_triggers(
                    exc_b_frames.to(model.decoder.device), trig_n.to(model.decoder.device), n_internal, mode=model.decoder.upsample_mode)
                loop_traj_np = loop_traj.detach().cpu().numpy()
                exc_traj_np = exc_traj.detach().cpu().numpy()

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
                        trig_concat_s = np.concatenate([trig_s, trig_s + seg_dur_s])
                    else:
                        trig_concat_s = None

                    # ---- Composite figure with waveform, reconstruction, inverse signal, and coefficients ----
                    fig, axes = plt.subplots(5, 1, figsize=(10, 12))

                    # 1) Target waveform with salience & triggers
                    t_wave = np.arange(wave_orig.shape[0]) / float(config["sample_rate"])
                    axes[0].plot(t_wave, wave_orig, color="blue", linewidth=1.0, label="target waveform")
                    if trig_concat_s is not None:
                        trig_mask = (trig_concat_s >= 0) & (trig_concat_s <= t_wave[-1])
                        axes[0].plot(trig_concat_s[trig_mask], np.zeros_like(trig_concat_s[trig_mask]),
                                     'rx', markersize=4, label="triggers")

                    # overlay salience + TriggerMLP output on twin axis
                    ratio = model.internal_sr // FRAME_RATE
                    pad_left_int = int(round(model.pad_left * model.internal_sr / config["sample_rate"]))
                    sal = model.last_trigger_probs[idx].cpu().numpy()
                    mlp_out = model.last_trigger_mlp[idx].cpu().numpy()
                    t_sal = (np.arange(sal.shape[0]) * ratio - pad_left_int) / float(config["sample_rate"])
                    mask = (t_sal >= 0) & (t_sal <= t_wave[-1])
                    ax_twin = axes[0].twinx()
                    if sal.max() > sal.min():
                        sal_norm = (sal - sal.min()) / (sal.max() - sal.min())
                    else:
                        sal_norm = sal
                    if mlp_out.max() > mlp_out.min():
                        mlp_norm = (mlp_out - mlp_out.min()) / (mlp_out.max() - mlp_out.min())
                    else:
                        mlp_norm = mlp_out
                    ax_twin.plot(t_sal[mask], sal_norm[mask], color="green", alpha=1.0, linewidth=2.0, label="salience")
                    ax_twin.plot(t_sal[mask], mlp_norm[mask], color="pink", alpha=1.0, linewidth=2.0, label="trigger MLP")
                    ax_twin.set_ylim(0, 1)

                    # Combine legends from both y-axes
                    lines_0, labels_0 = axes[0].get_legend_handles_labels()
                    lines_1, labels_1 = ax_twin.get_legend_handles_labels()
                    axes[0].legend(lines_0 + lines_1, labels_0 + labels_1, loc="upper right", fontsize="x-small")

                    # 2) Reconstruction
                    axes[1].plot(t_wave, wave_rec, label="recon")
                    axes[1].set_ylabel("Amplitude")
                    axes[1].legend(fontsize="x-small", loc="upper right")

                    # 3) Inverse-filtered signal
                    inv_sig_full = model.decoder.get_inverse_filtered_signal()[idx].cpu().numpy()
                    inv_sig = inv_sig_full[:wave_orig.shape[0]]  # match length for plotting
                    axes[2].plot(t_wave, inv_sig, label="inverse")
                    axes[2].set_ylabel("Amplitude")
                    axes[2].legend(fontsize="x-small", loc="upper right")

                    # 4) Loop coefficients
                    loop_np_k = loop_traj_np[idx]
                    t_loop = np.arange(loop_np_k.shape[0]) / float(model.internal_sr)
                    for j in range(loop_np_k.shape[1]):
                        axes[3].plot(t_loop, loop_np_k[:, j], label=f"loop[{j}]")
                    axes[3].set_ylabel("loop coeffs")
                    if loop_np_k.shape[1] <= 8:
                        axes[3].legend(fontsize="x-small", ncol=2, loc="upper right")

                    # 5) Excitation coefficients
                    exc_np_k = exc_traj_np[idx]
                    t_exc = np.arange(exc_np_k.shape[0]) / float(model.internal_sr)
                    for j in range(exc_np_k.shape[1]):
                        axes[4].plot(t_exc, exc_np_k[:, j], label=f"exc[{j}]")
                    axes[4].set_ylabel("exc coeffs")
                    axes[4].set_xlabel("Time (s)")
                    if exc_np_k.shape[1] <= 8:
                        axes[4].legend(fontsize="x-small", ncol=2, loc="upper right")

                    fig.tight_layout()

                    if wandb.run is not None:
                        # log both the composite figure and the audio
                        media_log[f"composite_plot_{k}"] = wandb.Image(fig)
                        media_log[f"audio_compare_{k}"] = wandb.Audio(
                            sample_play,
                            sample_rate=config["sample_rate"],
                            caption=f"epoch {epoch} | sample {k} | orig+recon (norm)",
                        )
                        media_log[f"audio_file_{k}"] = wandb.Audio(
                            wav_path,
                            sample_rate=config["sample_rate"],
                            caption=f"epoch {epoch} | sample {k} | raw file",
                        )

                    plt.close(fig)

                # --- log mean trigger count (deduplicated) -----------------------
                if wandb.run is not None and trig_times_s.ndim == 2:
                    # remove duplicates per item (head clamp & tail pad)
                    counts_unique = [len(np.unique(row)) for row in trig_times_s]
                    media_log["trigger_count_mean"] = float(np.mean(counts_unique))

                # log all items together
                if wandb.run is not None and len(media_log) > 0:
                    wandb.log(media_log, commit=True)

        print(f"[E{epoch}] train={t_loss:.4f} val={v_loss:.4f} best={best_val_loss:.4f} (no-improve {epochs_since_improve}/{config['patience']})")

    wandb.finish()


if __name__ == '__main__':
    mp.freeze_support()
    main()
