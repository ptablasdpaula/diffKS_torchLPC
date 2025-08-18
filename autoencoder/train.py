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

from collections import defaultdict


def plot_composite_four(fig_path: str,
                       target: np.ndarray,
                       reconstructed: np.ndarray,
                       loop_coeffs_c: np.ndarray,
                       eq_gains: np.ndarray,
                       sr: int) -> None:
    """
    Create a 4-panel composite:
      1) Target waveform
      2) Reconstructed waveform
      3) Loop filter coefficients
      4) EQ Band Gains (per-band dB)
    Saves to `fig_path`.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n_rows, n_cols = 2, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 7))
    axes = axes.ravel()

    # 1) Target waveform
    ax = axes[0]
    t = np.arange(len(target)) / sr
    ax.plot(t, target)
    ax.set_title("Target")
    ax.set_xlabel("Time (s)")

    # 2) Reconstructed waveform
    ax = axes[1]
    t_rec = np.arange(len(reconstructed)) / sr
    ax.plot(t_rec, reconstructed)
    ax.set_title("Reconstructed")
    ax.set_xlabel("Time (s)")

    # 3) Loop filter coefficients
    ax = axes[2]
    if loop_coeffs_c is not None:
        for k in range(loop_coeffs_c.shape[1]):
            ax.plot(np.arange(loop_coeffs_c.shape[0]), loop_coeffs_c[:, k], label=f"b{k}")
        ax.set_title("Loop Coefficients")
        ax.set_xlabel("Tap Index")
        ax.set_ylabel("Coefficient")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "(loop coeffs unavailable)", ha="center", va="center")
        ax.set_title("Loop Coefficients")

    # 4) EQ Band Gains
    ax = axes[3]
    ax.set_title("EQ Band Gains")
    if eq_gains is not None and len(eq_gains) > 0:
        band_indices = np.arange(1, len(eq_gains)+1)
        ax.plot(band_indices, eq_gains, marker='o', linestyle='-')
        ax.set_xlabel("Band Index")
        ax.set_ylabel("Gain (dB)")
        ax.set_ylim([-12, 12])
        ax.set_xticks(band_indices)
        ax.grid(True, which="both", alpha=0.2)
    else:
        ax.text(0.5, 0.5, "(no EQ gains)", ha="center", va="center")
        ax.set_xlabel("Band Index")
        ax.set_ylabel("Gain (dB)")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


from paths import NSYNTH_PREPROCESSED_DIR
from data.preprocess import NsynthDataset
from utils.misc import get_device, str2bool

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
    p.add_argument("--batches_per_epoch", type=int, default=int(env("BATCHES_PER_EPOCH", 10000)))
    p.add_argument("--max_epochs", type=int, default=int(env("MAX_EPOCHS", 10000)))
    p.add_argument("--patience", type=int, default=int(env("PATIENCE", 2000)))
    p.add_argument("--min_delta", type=float, default=float(env("MIN_DELTA", 0.001)),
                   help="Minimum relative (fractional) validation-loss improvement to reset patience. 0.001 = 0.1 %")

    # ─── Data-loading ──────────────────────────────────────────────────────
    p.add_argument("--batch_size", type=int, default=int(env("BATCH_SIZE", 1)))
    p.add_argument("--num_workers", type=int, default=int(env("NUM_WORKERS", 2)))

    # ─── Model size ────────────────────────────────────────────────────────
    p.add_argument("--hidden_size", type=int, default=int(env("HIDDEN_SIZE", 512)))

    # ─── DiffKS filter configuration ───────────────────────────────────────
    p.add_argument("--l_order", type=int, default=int(env("L_ORDER", 2)))

    # ─── Dataset filters ────────────────────────────────────────────────────
    p.add_argument("--families", type=str, default=env("FAMILIES", "guitar"))
    p.add_argument("--sources", type=str, default=env("SOURCES", "acoustic"))

    # ─── DiffKS decoder settings ───────────────────────────────────────────
    p.add_argument("--interpolation_type", type=str, default=env("INTERPOLATION_TYPE", "linear"))
    p.add_argument("--pitch_mode", type=str, default=env("PITCH_MODE", "meta"))

    # ─── Losses weights ────────────────────────────────────────────────────
    p.add_argument("--stft_weight", type=float, default=float(env("STFT_WEIGHT", 1.0)))

    # ─── Testing mode ────────────────────────────────────────────────────
    p.add_argument("--test", action="store_true",
                   default=str2bool(env("TEST", "false")),
                   help="If set, load the NSynth 'test' split for both training and validation")
    return p.parse_args()

# -----------------------------------------------------------------
def build_optimizer(model, lr):
    """
    Build Adam optimizer over all trainable params (decoder params excluded).
    """
    params = []
    for name, p in model.named_parameters():
        # Always freeze the differentiable decoder
        if name.startswith("decoder."):
            p.requires_grad = False
            continue
        if p.requires_grad:
            params.append(p)
    return optim.Adam(params, lr=lr)
# -----------------------------------------------------------------

# ---- Debug print helpers -------------------------------------------------

def print_trainable_summary(model, optimizer):
    """Print trainable component counts and optimizer param-group lrs."""
    comp_counts = {}
    comp_paramnums = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            key = name.split('.')[0] if '.' in name else name
            comp_counts[key] = comp_counts.get(key, 0) + 1
            comp_paramnums[key] = comp_paramnums.get(key, 0) + p.numel()
    total = sum(comp_paramnums.values())
    print("[TRAINABLE] components & parameter counts:")
    for k in sorted(comp_counts.keys()):
        print(f"  - {k:24s}: {comp_counts[k]:4d} tensors | {comp_paramnums[k]:8d} params")
    print(f"  Total trainable params: {total}")
    # Optimizer groups summary
    try:
        import torch.optim as _optim
        if isinstance(optimizer, _optim.Optimizer):
            print("[OPTIMIZER] param groups:")
            for i, g in enumerate(optimizer.param_groups):
                lr = g.get('lr', None)
                n = sum(p.numel() for p in g['params'])
                print(f"  group {i}: lr={lr} | params={n}")
    except Exception:
        pass


def print_grad_snapshot(model, components=None):
    """Print mean |grad| per component for the current batch.
    If `components` is provided, only those tops will be shown (and missing ones reported as no grads).
    """
    comp2vals = {}
    for name, p in model.named_parameters():
        key = name.split('.')[0] if '.' in name else name
        if (components is None or key in components) and p.requires_grad and (p.grad is not None):
            comp2vals.setdefault(key, []).append(p.grad.detach().abs().mean().item())
    if components:
        for k in components:
            comp2vals.setdefault(k, [])
    print("[GRAD SNAPSHOT] mean|grad| per component (current batch):")
    ordered_keys = components or sorted(comp2vals.keys())
    for k in ordered_keys:
        vals = comp2vals.get(k, [])
        if len(vals) == 0:
            print(f"  - {k:24s}: (no grads)")
        else:
            print(f"  - {k:24s}: {sum(vals)/len(vals):.6e} (n={len(vals)})")

def main():
    args = parse_args()
    # If --test is enabled, override splits to use the NSynth 'test' subset
    split_train = "test" if args.test else "train"
    split_val   = "test" if args.test else "valid"
    config = {
        "hidden_size": args.hidden_size,
        "loop_order": args.l_order,
        "sample_rate": 16000,
        "ks_sample_rate": 16000,
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
        "batches_per_epoch": args.batches_per_epoch,
        "stft_weight": args.stft_weight,
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
                            split=split_train,
                            pitch_mode=config["pitch_mode"],
                            families=config["families"],
                            sources=config["sources"], )

    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                              drop_last=True, pin_memory=True if device.type != "mps" else False,
                              num_workers=config["num_workers"])

    val_dataset = NsynthDataset(root=NSYNTH_PREPROCESSED_DIR,
                                split=split_val,
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
        internal_sr=config["ks_sample_rate"],
        interpolation_type=config["interpolation_type"],
        z_encoder=MfccTimeDistributedRnnEncoder(),
    ).to(device)
    optimizer = build_optimizer(model, lr=config["learning_rate"])
    print_trainable_summary(model, optimizer)
    global_step = 0
    # STFT loss
    mr_stft = MultiResolutionSTFTLoss(
        perceptual_weighting=True,
        scale_invariance=False,
        sample_rate=config["sample_rate"],
    )
    mr_stft = mr_stft.to(device)
    mr_stft = mr_stft.float()

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

    # Map a parameter name like "z_encoder.rnn.weight_ih_l0" to a component key "z_encoder"
    def _component_of(name: str) -> str:
        return name.split('.')[0] if '.' in name else name

    # ───────────────────────── training epochs ───────────────────────────
    for epoch in range(start_epoch, config["max_epochs"]):
        model.train()
        t_loss = 0
        batches_processed = 0

        # Accumulators for per-component gradient means across the epoch
        comp_grad_sums = defaultdict(float)
        comp_grad_counts = defaultdict(int)

        #torch.autograd.set_detect_anomaly(True)

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
            )
            assert recon.shape[1] == audio.shape[1], (
                f"Decoder returned {recon.shape[1]} samples, "
                f"but target has {audio.shape[1]}.")

            stft_loss = mr_stft(recon.unsqueeze(1), audio.unsqueeze(1)) * args.stft_weight
            loss = stft_loss
            optimizer.zero_grad()
            loss.backward()
            if batch_idx == 0:
                print_grad_snapshot(
                    model,
                    components=[
                        "z_encoder",
                        "coefficients_head",
                        "geq_head",
                        "gain_head",
                    ],
                )

            # Accumulate per-component mean absolute gradient (this batch)
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    key = _component_of(name)
                    g = p.grad.detach()
                    comp_grad_sums[key] += g.abs().mean().item()
                    comp_grad_counts[key] += 1

            optimizer.step()

            global_step += 1
            log_train_batch(loss.item())

            t_loss += loss.item()
            batches_processed += 1

        if batches_processed > 0:
            t_loss /= batches_processed

        # --- Per-epoch gradient means (per learnable component) ---
        if comp_grad_sums:
            grad_means_epoch = {k: comp_grad_sums[k] / max(1, comp_grad_counts[k]) for k in comp_grad_sums}
            # Print to stdout only (no wandb logging)
            print("[GRAD MEAN PER EPOCH]", {k: round(v, 6) for k, v in grad_means_epoch.items()})

        # ─── VALID ───────────────────────────────────────────────────────
        if epoch % config["eval_interval"] == 0:
            model.eval()
            v_losses_std = []
            with torch.no_grad():
                for audio, pitch, loud in val_loader:
                    audio, pitch, loud = audio.to(device), pitch.to(device), loud.to(device)
                    recon_std = model(
                        pitch=pitch,
                        loudness=loud,
                        audio=audio,
                        audio_sr=config["sample_rate"],
                    )
                    assert recon_std.shape[1] == audio.shape[1]
                    loss_std = mr_stft(recon_std.unsqueeze(1), audio.unsqueeze(1))
                    v_losses_std.append(loss_std.item())
            v_loss_std = float(np.mean(v_losses_std))

        # ─── LOGGING ───────────────────────────────────────────────────────
        if epoch % config["eval_interval"] == 0:
            log_epoch(t_loss, v_loss_std)
        else:
            # no validation this epoch
            pass

        # ─── CHKPTS  ───────────────────────────────────────────────────────
        improved = False
        if epoch % config["eval_interval"] == 0 and not np.isnan(v_loss_std):
            # Relative improvement wrt best
            if best_val_loss == float('inf') or (best_val_loss - v_loss_std) / best_val_loss >= config["min_delta"]:
                improved = True
                best_val_loss = v_loss_std
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
        elif epoch % config["eval_interval"] == 0:
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


                a, p, l = fixed_audio.to(device), fixed_pitch.to(device), fixed_loud.to(device)

                rec = model(
                    pitch=p,
                    loudness=l,
                    audio=a,
                    audio_sr=config["sample_rate"],
                )
                assert rec.shape[1] == a.shape[1]

                # Fetch parameter info only
                _lc, _geq = model(
                    pitch=p,
                    loudness=l,
                    audio=a,
                    audio_sr=config["sample_rate"],
                    return_parameters=True,
                )
                gains_db_batch = _geq["gains_db"]  # [B, K]
                if gains_db_batch is not None:
                    try:
                        gcpu = gains_db_batch.detach().cpu()
                        same = bool(torch.allclose(gcpu, gcpu[0:1].expand_as(gcpu), atol=1e-6)) if gcpu.size(0) > 1 else True
                        per_band_std = float(gcpu.std(dim=0).mean().item()) if gcpu.ndim == 2 else float("nan")
                        print(
                            f"[GEQ DBG] batch_size={gcpu.size(0)} bands={gcpu.size(1) if gcpu.ndim==2 else 'NA'} "
                            f"mean(per-band std)={per_band_std:.6f} dB | identical_across_batch={same} | "
                            f"min={gcpu.min().item():.3f} dB max={gcpu.max().item():.3f} dB"
                        )
                    except Exception:
                        pass

                # Save & log a few examples
                media_log = {}
                for idx in range(n_plot):
                    wave_orig = a[idx].cpu().numpy()
                    wave_rec  = rec[idx].cpu().numpy()
                    # Concatenate target || recon for quick listening
                    sample_cat = np.concatenate([wave_orig, wave_rec], axis=0)
                    peak = float(np.max(np.abs(sample_cat))) if sample_cat.size > 0 else 0.0
                    sample_cat = sample_cat / peak * 0.99 if peak > 0 else sample_cat

                    wav_name = f"sample_e{epoch}_{idx}.wav"
                    wav_path = os.path.join(config["save_dir"], wav_name)
                    sf.write(wav_path, sample_cat, config["sample_rate"])
                    print(f"[AUDIO] wrote: {wav_path}")

                    if wandb.run is not None:
                        media_log[f"audio_{idx}"] = wandb.Audio(
                            sample_cat,
                            sample_rate=config["sample_rate"],
                            caption=f"epoch {epoch} | sample {idx} | target+recon",
                        )

                # ---- Plotting of inversion debug info removed ----

                # ---- Composite (4‑panel) per example -----------------------------
                for idx in range(n_plot):
                    target_np = a[idx].cpu().numpy()
                    recon_np  = rec[idx].cpu().numpy()
                    lc_np = _lc[idx].detach().cpu().numpy()  # [T_int, K]
                    eq_gains_np = gains_db_batch[idx].detach().cpu().numpy() if gains_db_batch is not None else None
                    comp_path = os.path.join(config["save_dir"], f"composite_e{epoch}_{idx}.png")
                    plot_composite_four(
                        comp_path,
                        target=target_np,
                        reconstructed=recon_np,
                        loop_coeffs_c=lc_np,
                        eq_gains=eq_gains_np,
                        sr=config["sample_rate"],
                    )
                    if wandb.run is not None:
                        media_log[f"composite_{idx}"] = wandb.Image(comp_path, caption=f"Composite | e{epoch} i{idx}")

                if wandb.run is not None and len(media_log) > 0:
                    wandb.log(media_log, commit=True)

        if epoch % config["eval_interval"] == 0:
            print(f"[E{epoch}] train={t_loss:.4f} val_std={v_loss_std:.4f} best={best_val_loss:.4f} (no-improve {epochs_since_improve}/{config['patience']})")
        else:
            print(f"[E{epoch}] train={t_loss:.4f} best={best_val_loss:.4f} (no-improve {epochs_since_improve}/{config['patience']})")

    wandb.finish()


if __name__ == '__main__':
    mp.freeze_support()
    main()
