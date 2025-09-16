from tqdm import tqdm
import numpy as np
import soundfile as sf
import torch, torch.optim as optim, wandb
from torch.utils.data import DataLoader
from .model import nnKarplusStrong
import argparse, os
import multiprocessing as mp
import psutil

from collections import defaultdict

from paths import NSYNTH_PREPROCESSED_DIR
from data.preprocess import NsynthDataset
from utils.misc import get_device, str2bool


from losses import build_smooth_mrstft

import matplotlib
matplotlib.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    env = os.environ.get


    # ─── Run identification ────────────────────────────────────────────────
    p.add_argument("--name", type=str, default=env("NAME", "exp"),
                   help="Unique experiment name (used for checkpoints & wandb run)")
    p.add_argument("--continue_from_checkpoint", action="store_true",
                   default=str2bool(env("CONTINUE_FROM_CHECKPOINT") or "false"),
                   help="Resume training from latest checkpoint for this --name")

    # ─── Optimisation hyper-parameters ─────────────────────────────────────
    p.add_argument("--learning_rate", type=float, default=float(env("LEARNING_RATE") or 1e-4))
    p.add_argument("--batches_per_epoch", type=int, default=int(env("BATCHES_PER_EPOCH") or 10000))
    p.add_argument("--max_epochs", type=int, default=int(env("MAX_EPOCHS") or 10000))
    p.add_argument("--patience", type=int, default=int(env("PATIENCE") or 2000))
    p.add_argument("--min_delta", type=float, default=float(env("MIN_DELTA") or 0.001),
                   help="Minimum relative (fractional) validation-loss improvement to reset patience. 0.001 = 0.1 %")

    # ─── Data-loading ──────────────────────────────────────────────────────
    p.add_argument("--batch_size", type=int, default=int(env("BATCH_SIZE") or 16))
    p.add_argument("--num_workers", type=int, default=int(env("NUM_WORKERS") or 2))

    # ─── DiffKS filter configuration ───────────────────────────────────────
    p.add_argument("--l_order", type=int, default=int(env("L_ORDER") or 1))
    p.add_argument("--filter_type", type=str, default=(env("FILTER_TYPE", "iir")))

    # ─── Dataset filters ────────────────────────────────────────────────────
    p.add_argument("--families", type=str, default=env("FAMILIES", "guitar"))
    p.add_argument("--sources", type=str, default=env("SOURCES", "acoustic"))

    # ─── DiffKS decoder settings ───────────────────────────────────────────
    p.add_argument("--interpolation_type", type=str, default=env("INTERPOLATION_TYPE", "lagrange"))
    p.add_argument("--pitch_mode", type=str, default=env("PITCH_MODE", "meta"))

    # ─── Losses weights ────────────────────────────────────────────────────
    p.add_argument("--stft_weight", type=float, default=float(env("STFT_WEIGHT") or 1.0))

    # ─── DiffKS timesteps and noise bands ─────────────────────────────────
    p.add_argument("--n_noise_bands", type=int, default=int(env("N_NOISE_BANDS") or 16))

    # ─── Testing mode ──────────────────────────────────────────────────────
    p.add_argument("--test", action="store_true",
                   default=str2bool(env("TEST") or "false"),
                   help="If set, load the NSynth 'test' split for both training and validation")
    return p.parse_args()

# -----------------------------------------------------------------
def build_optimizer(model, lr_main, *_):
    params = []
    for name, p in model.named_parameters():
        if name.startswith("decoder."):
            p.requires_grad = False
            continue
        if p.requires_grad:
            params.append(p)
    return optim.Adam(params, lr=lr_main)
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
    # (Optimizer groups and z_encoder breakdown printout removed)


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


# -----------------------------------------------------------------
# Composite plotting helpers (dynamic: adapts to returned params)
# -----------------------------------------------------------------

def _plot_param_panel(ax, name, tensor_np, sample_rate: int, target_len: int):
    """Plot a returned parameter on a given axis.
    Rules:
    - 1D: line vs time (if matches target_len) else vs index.
    - 2D: if channels <= 8, draw lines; else heatmap (imshow).
    - 3D: treat as (T, A, B) with x=time, y=A, and color intensity = RMS over B.
    - >3D: raises ValueError.
    """
    import numpy as _np
    import torch

    arr = tensor_np
    # Convert to numpy array without altering rank, avoiding DeprecationWarning
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    else:
        arr = _np.asarray(arr)

    # 1D ---------------------------------------------------------
    if arr.ndim == 1:
        t = _np.arange(arr.shape[0], dtype=float)
        if arr.shape[0] == target_len and sample_rate > 0:
            t = t / float(sample_rate)
            ax.set_xlabel("time [s]")
        else:
            ax.set_xlabel("index")
        ax.plot(t, arr)
        ax.set_title(f"{name} {list(arr.shape)}")
        return

    # 2D ---------------------------------------------------------
    if arr.ndim == 2:
        T, C = arr.shape
        x = _np.arange(T, dtype=float)
        if T == target_len and sample_rate > 0:
            x = x / float(sample_rate)
            ax.set_xlabel("time [s]")
        else:
            ax.set_xlabel("frame")
        if C <= 8:
            for c in range(C):
                ax.plot(x, arr[:, c], linewidth=0.8, alpha=0.9, label=f"ch{c}")
            if C > 1:
                ax.legend(loc="best", fontsize=8)
            ax.set_title(f"{name} {list(arr.shape)}")
        else:
            ax.imshow(arr.T, aspect="auto", origin="lower")
            ax.set_ylabel("channel")
            ax.set_title(f"{name} heatmap {list(arr.shape)}")
        return

    # 3D ---------------------------------------------------------
    if arr.ndim == 3:
        dims = list(arr.shape)
        # Choose time axis: prefer dimension equal to target_len, else the largest
        if target_len in dims:
            t_idx = dims.index(target_len)
        else:
            t_idx = int(_np.argmax(dims))
        arr = _np.moveaxis(arr, t_idx, 0)  # now (T, A, B)
        T, A, B = arr.shape
        # Collapse the third dimension via RMS so intensity represents magnitude across B
        arr2d = _np.sqrt(_np.mean(arr ** 2, axis=2))  # (T, A)
        # Time axis
        x = _np.arange(T, dtype=float)
        if T == target_len and sample_rate > 0:
            x = x / float(sample_rate)
            ax.set_xlabel("time [s]")
        else:
            ax.set_xlabel("frame")
        # Heatmap: x=time, y=A, intensity=RMS over B
        ax.imshow(arr2d.T, aspect="auto", origin="lower")
        ax.set_ylabel("channel")
        ax.set_title(f"{name} heatmap (RMS over axis=2) {dims}")
        return

    # Unsupported dims ----------------------------------------------------
    raise ValueError(f"Unsupported tensor ndim={arr.ndim} for plotting param '{name}'")


def _make_and_save_composite(sample_idx: int,
                             wave_orig_np,
                             wave_rec_np,
                             params_dict,
                             sample_rate: int,
                             save_path: str):
    """Make a composite image with waveforms and one panel per param.
    - params_dict: mapping name -> tensor [B, ...]; we index sample_idx
    - save_path: where to write the PNG
    """
    import numpy as _np
    import torch

    panels = [("waveforms", None)]  # first panel: original vs recon

    for k, v in params_dict.items():
        # Prefer indexing the batch dimension when present
        if hasattr(v, "shape") and len(v.shape) >= 1 and v.shape[0] > sample_idx:
            arr = v[sample_idx]
        else:
            arr = v
        # Convert to numpy array without altering rank, avoiding DeprecationWarning
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        else:
            arr = _np.asarray(arr)
        panels.append((k, arr))

    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    # Panel 0: waveforms
    ax0 = axes[0]
    N = wave_orig_np.shape[0]
    t = _np.arange(N, dtype=float) / float(sample_rate)
    ax0.plot(t, wave_orig_np, label="target", linewidth=0.8)
    ax0.plot(t, wave_rec_np, label="recon", linewidth=0.8)
    ax0.set_title("Original vs Reconstructed waveform")
    ax0.set_xlabel("time [s]")
    ax0.legend(loc="best", fontsize=8)

    # Remaining panels: dynamic params
    for ax, (name, arr) in zip(axes[1:], panels[1:]):
        _plot_param_panel(ax, name, arr, sample_rate, target_len=N)

    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    # If --test is enabled, override splits to use the NSynth 'test' subset
    split_train = "test" if args.test else "train"
    split_val   = "test" if args.test else "valid"
    config = {
        "sample_rate": 16000,
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
        "filter_type": args.filter_type,
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
    def log_epoch(train_loss: float, val_loss: float, epoch: int):
        if wandb.run is not None:
            wandb.log({
                "epoch": int(epoch),
                "train loss per epoch": float(train_loss),
                "val loss per epoch":   float(val_loss),
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
    model = nnKarplusStrong(
        interpolation_type=config["interpolation_type"],
        filter_type=config["filter_type"],
        n_noise_bands=args.n_noise_bands,
        tcn_ch=64,
    ).to(device)

    optimizer = build_optimizer(model, lr_main=config["learning_rate"])
    # Log current stage to wandb (removed as per instruction)

    print_trainable_summary(model, optimizer)
    # STFT loss (scale-variant only)
    mr_stft = build_smooth_mrstft(scale_invariance=False).to(device).float()

    # ─── Resume from checkpoint if requested ──────────────────────────────
    start_epoch, best_val_loss = 0, float('inf')
    if args.continue_from_checkpoint and os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device)

        # strip out the two analysis buffers so their old shapes don't conflict
        sd = ckpt["model_state_dict"]
        for key in list(sd):
            if "ks_inverse_signal" in key or "excitation_filter_out" in key:
                sd.pop(key)

        model.load_state_dict(sd)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)

        print(f"[RESUME] Starting at epoch {start_epoch} (best so far {best_val_loss:.4f})")

    # Derive starting global step if resuming, to keep stage schedule consistent
    bpe = min(len(train_loader), config["batches_per_epoch"])
    global_step = start_epoch * bpe

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
        nan_or_inf_detected = False

        for batch_idx, (audio, pitch, loud) in enumerate(tqdm(train_loader, desc=f"[E{epoch:03d} train]")):
            if batch_idx >= config["batches_per_epoch"]:
                break
            audio, pitch, loud = audio.to(device), pitch.to(device), loud.to(device)

            recon = model(
                pitch, loud, audio, config["sample_rate"],
            )
            assert recon.shape[1] == audio.shape[1], (
                f"Decoder returned {recon.shape[1]} samples, "
                f"but target has {audio.shape[1]}."
            )

            stft_sv = mr_stft(recon.unsqueeze(1), audio.unsqueeze(1))
            loss = (
                args.stft_weight * stft_sv
            )

            # Abort early if loss is NaN or Inf
            if not torch.isfinite(loss):
                print(f"[ABORT] Non-finite loss detected at epoch {epoch}, batch {batch_idx}: {loss.item()}")
                if wandb.run is not None:
                    wandb.log({"nonfinite_loss_detected": True, "epoch": epoch, "batch": batch_idx})
                    wandb.run.summary["nonfinite_loss"] = True
                nan_or_inf_detected = True
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if wandb.run is not None:
                wandb.log({
                    "train loss per batch": float(loss.item()),
                    "train/loss_stft": float(stft_sv.item()),
                })

            t_loss += loss.item()
            batches_processed += 1

        if nan_or_inf_detected:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "wandb_id": wandb_id,
            }
            abort_ckpt = os.path.join(config["save_dir"], f"nan_abort_{args.name}.pth")
            torch.save(ckpt, abort_ckpt)
            print(f"[ABORT] Saved checkpoint to {abort_ckpt}")
            if wandb.run is not None:
                wandb.log({"nonfinite_loss_detected": True, "epoch": int(epoch), "batch": int(batch_idx)})
                wandb.run.summary["nonfinite_loss"] = True
                wandb.finish()
            return

        if batches_processed > 0:
            t_loss /= batches_processed

        # --- Per-epoch average |grad| per component (print after training loop, before validation/loss logging) ---
        # Reuse print_grad_snapshot logic, but do per-epoch mean
        comp2vals = {}
        for name, p in model.named_parameters():
            key = name.split('.')[0] if '.' in name else name
            if p.requires_grad and (p.grad is not None):
                comp2vals.setdefault(key, []).append(p.grad.detach().abs().mean().item())
        print("[GRAD MEAN PER EPOCH] mean|grad| per component:")
        for k in sorted(comp2vals.keys()):
            vals = comp2vals[k]
            if len(vals) == 0:
                print(f"  - {k:24s}: (no grads)")
            else:
                print(f"  - {k:24s}: {sum(vals)/len(vals):.6e} (n={len(vals)})")

        # ─── VALID ───────────────────────────────────────────────────────
        if epoch % config["eval_interval"] == 0:
            model.eval()
            v_losses_std = []
            with torch.no_grad():
                for audio, pitch, loud in val_loader:
                    audio, pitch, loud = audio.to(device), pitch.to(device), loud.to(device)
                    recon_std = model(
                        pitch, loud, audio, config["sample_rate"],
                    )
                    assert recon_std.shape[1] == audio.shape[1]

                    '''
                    # Amplitude normalization only if invariant mode
                    if args.invariant:
                        eps = 1e-9
                        rms_target = audio.pow(2).mean(dim=1, keepdim=True).sqrt() + eps
                        rms_recon  = recon_std.pow(2).mean(dim=1, keepdim=True).sqrt() + eps
                        gain = rms_target / rms_recon
                        recon_std = recon_std * gain  # rescaled recon
                    '''

                    loss_std = mr_stft(recon_std.unsqueeze(1), audio.unsqueeze(1))
                    v_losses_std.append(loss_std.item())
            v_loss_std = float(np.mean(v_losses_std))

        # ─── LOGGING ───────────────────────────────────────────────────────
        if epoch % config["eval_interval"] == 0:
            log_epoch(t_loss, v_loss_std, epoch)

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

        # ─── AUDIO + DIAGNOSTIC COMPOSITES ───────────────────────────────────
        if epoch % config["eval_interval"] == 0:
            with torch.no_grad():
                a, p, l = fixed_audio.to(device), fixed_pitch.to(device), fixed_loud.to(device)
                rec = model(
                    p, l, a, config["sample_rate"],
                )
                assert rec.shape[1] == a.shape[1]

                # --- Collect returned parameters for diagnostics (same batch as audio) ---
                params_ret = model(
                    p, l, a, config["sample_rate"], return_parameters=True,
                )

                # --- Audio logging ---
                media_log = {}
                for idx in range(n_plot):
                    wave_orig = a[idx].cpu().numpy()
                    wave_rec  = rec[idx].cpu().numpy()
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
                if wandb.run is not None and len(media_log) > 0:
                    wandb.log(media_log, commit=True)

                # --- Composites: waveforms + dynamic params from return_parameters ---
                img_log = {}
                params_cpu = {k: (v.detach().cpu() if hasattr(v, "detach") else v) for k, v in params_ret.items()}
                for idx in range(n_plot):
                    wave_orig = a[idx].cpu().numpy()
                    wave_rec  = rec[idx].cpu().numpy()
                    comp_name = f"composite_e{epoch}_{idx}.png"
                    comp_path = os.path.join(config["save_dir"], comp_name)
                    _make_and_save_composite(
                        sample_idx=idx,
                        wave_orig_np=wave_orig,
                        wave_rec_np=wave_rec,
                        params_dict=params_cpu,
                        sample_rate=int(config["sample_rate"]),
                        save_path=comp_path,
                    )
                    if wandb.run is not None:
                        img_log[f"composite_{idx}"] = wandb.Image(comp_path, caption=f"epoch {epoch} | sample {idx} | composite")
                if wandb.run is not None and len(img_log) > 0:
                    wandb.log(img_log, commit=True)

        if epoch % config["eval_interval"] == 0:
            print(f"[E{epoch}] train={t_loss:.4f} val_std={v_loss_std:.4f} best={best_val_loss:.4f} (no-improve {epochs_since_improve}/{config['patience']})")
        else:
            print(f"[E{epoch}] train={t_loss:.4f} best={best_val_loss:.4f} (no-improve {epochs_since_improve}/{config['patience']})")

    wandb.finish()


if __name__ == '__main__' or __name__.endswith("autoencoder.train"):
    mp.freeze_support()
    main()
