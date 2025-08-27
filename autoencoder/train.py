from tqdm import tqdm
import numpy as np
import soundfile as sf
import torch, torch.optim as optim, wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .model import nnKarplusStrong
import argparse, os
import multiprocessing as mp
import psutil

from collections import defaultdict

from paths import NSYNTH_PREPROCESSED_DIR
from data.preprocess import NsynthDataset, a_weighted_loudness
from utils.misc import get_device, str2bool

from .losses import _frame_env, build_smooth_mrstft
from .plotters import plot_composite_four, _resample_to_len, plot_excitation_composite_zoomed

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
    p.add_argument("--loud_weight", type=float, default=float(env("LOUD_WEIGHT", 0.0)),
                   help="Weight for the A-weighted loudness loss (L2/MSE on per-frame log-power).")
    p.add_argument("--env_weight", type=float, default=float(env("ENV_WEIGHT", 0.0)),
                   help="Weight for the envelope loss (L1 on framewise |x|).")

    # ─── Stages ────────────────────────────────────────────────────────────
    p.add_argument("--stage0_steps", type=int, default=int(env("STAGE0_STEPS", 5000)),
                   help="Number of steps with backbone frozen (stage 0). Then unfreeze all (stage 1). 0 disables stage 0.")

    # ─── Testing mode ──────────────────────────────────────────────────────
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
    if isinstance(optimizer, optim.Optimizer):
        print("[OPTIMIZER] param groups:")
        for i, g in enumerate(optimizer.param_groups):
            lr = g.get('lr', None)
            n = sum(p.numel() for p in g['params'])
            print(f"  group {i}: lr={lr} | params={n}")
    # Optional: breakdown of z_encoder heads if present
    if hasattr(model, "z_encoder"):
        z = model.z_encoder
        def _count(mod):
            if mod is None:
                return 0, 0
            t = sum(1 for p in mod.parameters() if p.requires_grad)
            n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            return t, n
        print("[TRAINABLE] z_encoder breakdown:")
        for label, mod in (
            ("z_encoder.gain_head", getattr(z, "gain_head", None)),
            ("z_encoder.loop_head", getattr(z, "loop_head", None)),
            ("z_encoder.geq_head",  getattr(z, "geq_head",  None)),
        ):
            t, n = _count(mod)
            print(f"  - {label:24s}: {t:4d} tensors | {n:8d} params")


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

# --- NEW: Print gradients for z_encoder heads (gain_head, loop_head, geq_head) if present
def print_ast_head_grads(model):
    """Print mean |grad| for z_encoder heads (gain_head, loop_head, geq_head) if present."""
    if not hasattr(model, "z_encoder"):
        return
    z = model.z_encoder
    lines = ["[Z_ENCODER HEAD GRADS]"]
    for label, mod in (
        ("z_encoder.gain_head", getattr(z, "gain_head", None)),
        ("z_encoder.loop_head", getattr(z, "loop_head", None)),
        ("z_encoder.geq_head",  getattr(z, "geq_head",  None)),
    ):
        if mod is None:
            lines.append(f"  - {label:24s}: (missing)")
            continue
        vals = []
        for p in mod.parameters():
            if p.requires_grad and p.grad is not None:
                vals.append(p.grad.detach().abs().mean().item())
        if len(vals) == 0:
            lines.append(f"  - {label:24s}: (no grads)")
        else:
            lines.append(f"  - {label:24s}: {sum(vals)/len(vals):.6e}")
    print("\n".join(lines))

# -----------------------------------------------------------------
# Staged training helpers


def set_stage0_backbone_frozen(model) -> None:
    """Stage 0: freeze AST backbone, train mel-attn + heads."""
    assert hasattr(model, "freeze_backbone"), (
        "Model is expected to expose freeze_backbone(); update the model or this call site."
    )
    model.freeze_backbone()
    # Always keep decoder frozen
    for name, p in model.named_parameters():
        if name.startswith("decoder."):
            p.requires_grad = False

def set_stage1_unfreeze_all(model) -> None:
    """Stage 1: unfreeze all learnable parts except the differentiable decoder."""
    assert hasattr(model, "unfreeze_all"), (
        "Model is expected to expose unfreeze_all(); update the model or this call site."
    )
    model.unfreeze_all()
    # Ensure decoder remains frozen
    for name, p in model.named_parameters():
        if name.startswith("decoder."):
            p.requires_grad = False

def rebuild_optimizer(optimizer, model, lr):
    """Recreate optimizer after changing requires_grad flags."""
    del optimizer
    optimizer = build_optimizer(model, lr)
    print("[STAGE] Rebuilt optimizer for new trainable set")
    print_trainable_summary(model, optimizer)
    return optimizer
# -----------------------------------------------------------------

def main():
    args = parse_args()
    # If --test is enabled, override splits to use the NSynth 'test' subset
    split_train = "test" if args.test else "train"
    split_val   = "test" if args.test else "valid"
    config = {
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
        "loud_weight": args.loud_weight,
        "env_weight": args.env_weight,
        "stage0_steps": args.stage0_steps,
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
    model = nnKarplusStrong(
        batch_size=config["batch_size"],
        loop_order=config["loop_order"],
        internal_sr=config["ks_sample_rate"],
        interpolation_type=config["interpolation_type"],
    ).to(device)

    # ---- Stage scheduling (stage 0 → stage 1) ----------------------------
    current_stage = 0 if config["stage0_steps"] > 0 else 1
    if current_stage == 0:
        print("[STAGE 0] Freezing AST backbone; training mel-attn + heads only")
        set_stage0_backbone_frozen(model)
    else:
        print("[STAGE 1] Training all (except decoder)")
        set_stage1_unfreeze_all(model)

    optimizer = build_optimizer(model, lr=config["learning_rate"])
    if wandb.run is not None:
        wandb.log({"stage": int(current_stage)})

    print_trainable_summary(model, optimizer)
    # STFT loss
    mr_stft = build_smooth_mrstft()
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

    # Derive starting global step if resuming, to keep stage schedule consistent
    bpe = min(len(train_loader), config["batches_per_epoch"])
    global_step = start_epoch * bpe

    # If we've already passed stage0_steps when resuming, advance to stage 1
    if current_stage == 0 and global_step >= config["stage0_steps"]:
        print(f"[STAGE TRANSITION @ startup] global_step={global_step} ≥ stage0_steps={config['stage0_steps']} → unfreezing all")
        set_stage1_unfreeze_all(model)
        optimizer = rebuild_optimizer(optimizer, model, lr=config["learning_rate"])
        current_stage = 1

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

        # Accumulators for per-component gradient means across the epoch
        comp_grad_sums = defaultdict(float)
        comp_grad_counts = defaultdict(int)

        #torch.autograd.set_detect_anomaly(True)

        # ─── Training step ───────────────────────────────────────────────
        for batch_idx, (audio, pitch, loud) in enumerate(tqdm(train_loader, desc=f"[E{epoch:03d} train]")):
            if batch_idx >= config["batches_per_epoch"]:
                break
            audio, pitch, loud = audio.to(device), pitch.to(device), loud.to(device)
            recon = model(pitch, loud, audio, config["sample_rate"])
            assert recon.shape[1] == audio.shape[1], (
                f"Decoder returned {recon.shape[1]} samples, "
                f"but target has {audio.shape[1]}.")

            stft_loss = mr_stft(recon.unsqueeze(1), audio.unsqueeze(1)) * args.stft_weight

            # A-weighted loudness loss (per-frame, log-power). No alignment, no try/except.
            # recon/audio are [B,N]; target loudness `loud` is [B,T] or [B,T,1].
            pred_loud = a_weighted_loudness(recon)        # [B, F_pred]
            tgt_loud  = loud
            if tgt_loud.dim() == 3 and tgt_loud.size(-1) == 1:
                tgt_loud = tgt_loud.squeeze(-1)           # [B, F_tgt]

            # Enforce exact shape match to avoid broadcasting
            pred_loud = pred_loud.reshape(pred_loud.size(0), -1)
            tgt_loud  = tgt_loud.reshape(tgt_loud.size(0), -1)
            if pred_loud.size(1) != tgt_loud.size(1):
                raise RuntimeError(
                    f"Loudness frames mismatch: pred {pred_loud.size(1)} vs target {tgt_loud.size(1)}. "
                    f"Ensure HOP_SIZE and framing match in a_weighted_loudness() and dataset 'loud'.")

            # L2 (MSE) on log-power loudness frames
            loudness_loss = F.mse_loss(pred_loud, tgt_loud)

            # Envelope loss (phase-agnostic, time-local). Use same frame count as loudness.
            num_frames = pred_loud.size(1)
            env_pred = _frame_env(recon, num_frames)  # [B,F]
            env_tgt = _frame_env(audio, num_frames)  # [B,F]
            env_loss = F.l1_loss(env_pred, env_tgt)

            loss = stft_loss + args.loud_weight * loudness_loss + args.env_weight * env_loss
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
            if batch_idx == 0:
                print_grad_snapshot(
                    model,
                    components=[
                        "z_encoder",
                    ],
                )
            if batch_idx == 0:
                print_ast_head_grads(model)

            # Accumulate per-component mean absolute gradient (this batch)
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    key = _component_of(name)
                    g = p.grad.detach()
                    comp_grad_sums[key] += g.abs().mean().item()
                    comp_grad_counts[key] += 1

            optimizer.step()

            global_step += 1

            # Stage transition: when reaching stage0_steps, unfreeze all
            if current_stage == 0 and global_step >= config["stage0_steps"]:
                print(f"[STAGE TRANSITION] Hit global_step={global_step} → switching to Stage 1 (unfreeze all)")
                set_stage1_unfreeze_all(model)
                optimizer = rebuild_optimizer(optimizer, model, lr=config["learning_rate"])
                current_stage = 1
                if wandb.run is not None:
                    wandb.log({"stage": 1, "global_step": int(global_step)})

            log_train_batch(loss.item())
            if wandb.run is not None:
                wandb.log({
                    "train/loss_stft": float(stft_loss.item()),
                    "train/loss_loud": float((args.loud_weight * loudness_loss).item()),
                    "train/loss_env": float((args.env_weight * env_loss).item()),
                })

            t_loss += loss.item()
            batches_processed += 1

        # If a non-finite loss was detected, save a checkpoint and stop training.
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
                wandb.finish()
            return

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
                    recon_std = model(pitch, loud, audio, config["sample_rate"])
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

                rec = model(p, l, a, config["sample_rate"])
                assert rec.shape[1] == a.shape[1]

                # Fetch parameter info and internal signals
                _lc, _geq, gain_frames_b, gain_up_b, excitation_pregain_b, excitation_postgain_b, excitation_b = model(
                    p, l, a, config["sample_rate"], return_parameters=True
                )
                gains_db_batch = _geq["gains_db"]  # [B, K]
                if gains_db_batch is not None:
                    gcpu = gains_db_batch.detach().cpu()
                    same = bool(torch.allclose(gcpu, gcpu[0:1].expand_as(gcpu), atol=1e-6)) if gcpu.size(0) > 1 else True
                    per_band_std = float(gcpu.std(dim=0).mean().item()) if gcpu.ndim == 2 else float("nan")
                    print(
                        f"[GEQ DBG] batch_size={gcpu.size(0)} bands={gcpu.size(1) if gcpu.ndim==2 else 'NA'} "
                        f"mean(per-band std)={per_band_std:.6f} dB | identical_across_batch={same} | "
                        f"min={gcpu.min().item():.3f} dB max={gcpu.max().item():.3f} dB"
                    )

                # Save & log a few examples
                media_log = {}
                for idx in range(n_plot):
                    wave_orig = a[idx].cpu().numpy()
                    wave_rec  = rec[idx].cpu().numpy()
                    wave_exc  = excitation_b[idx].detach().cpu().numpy() if isinstance(excitation_b, torch.Tensor) else np.asarray(excitation_b[idx])

                    # Ensure excitation is same length as audio for clean concatenation
                    if len(wave_exc) != len(wave_orig):
                        wave_exc = _resample_to_len(wave_exc, len(wave_orig))

                    # Concatenate target || excitation || recon for quick listening
                    sample_cat = np.concatenate([wave_orig, wave_exc, wave_rec], axis=0)
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
                            caption=f"epoch {epoch} | sample {idx} | target+excitation+recon",
                        )

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

                    # --- Excitation composite (3 stacked panels, zoomed on first trigger) ---
                    pre_np  = excitation_pregain_b[idx].detach().cpu().numpy()
                    post_np = excitation_postgain_b[idx].detach().cpu().numpy()
                    exc_np  = excitation_b[idx].detach().cpu().numpy()
                    gf_np   = gain_frames_b[idx].detach().cpu().numpy()
                    gu_np   = gain_up_b[idx].detach().cpu().numpy()

                    exc_comp_path = os.path.join(config["save_dir"], f"excitation_composite_zoom_e{epoch}_{idx}.png")
                    plot_excitation_composite_zoomed(
                        exc_comp_path,
                        sr=config["sample_rate"],
                        excitation_pregain=pre_np,
                        excitation_postgain=post_np,
                        excitation=exc_np,
                        gain_frames=gf_np,
                        gain_up=gu_np,
                        pre_ms=10.0,
                        post_ms=10.0,
                    )

                    if wandb.run is not None:
                        media_log[f"excitation_composite_{idx}"] = wandb.Image(
                            exc_comp_path, caption=f"Excitation composite (zoom) | e{epoch} i{idx}")

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
