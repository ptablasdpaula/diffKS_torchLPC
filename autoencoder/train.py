from tqdm import tqdm
import numpy as np
import soundfile as sf
import torch, torch.optim as optim, wandb
from torch.utils.data import DataLoader
from .model import nnKarplusStrong
import argparse, os
import multiprocessing as mp
import psutil
import math

from collections import defaultdict

from paths import NSYNTH_PREPROCESSED_DIR
from data.preprocess import NsynthDataset
from utils.misc import get_device, str2bool

from .losses import build_smooth_mrstft
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
    p.add_argument("--batch_size", type=int, default=int(env("BATCH_SIZE", 16)))
    p.add_argument("--num_workers", type=int, default=int(env("NUM_WORKERS", 2)))



    # ─── DiffKS filter configuration ───────────────────────────────────────
    p.add_argument("--l_order", type=int, default=int(env("L_ORDER", 1)))
    p.add_argument("--filter_type", type=str, default=(env("FILTER_TYPE", "fir")))

    # ─── Dataset filters ────────────────────────────────────────────────────
    p.add_argument("--families", type=str, default=env("FAMILIES", "guitar"))
    p.add_argument("--sources", type=str, default=env("SOURCES", "acoustic"))

    # ─── DiffKS decoder settings ───────────────────────────────────────────
    p.add_argument("--interpolation_type", type=str, default=env("INTERPOLATION_TYPE", "lagrange"))
    p.add_argument("--pitch_mode", type=str, default=env("PITCH_MODE", "meta"))

    # ─── Losses weights ────────────────────────────────────────────────────
    p.add_argument("--stft_weight", type=float, default=float(env("STFT_WEIGHT", 1.0)))
    p.add_argument("--pg_weight", type=float, default=float(env("PG_WEIGHT", 0.1)),
                   help="Weight for the (p,g) prior loss estimated per onset from two-harmonic decay fits. Set to 0 to disable.")

    # ─── Backbone fine-tuning flags ───────────────────────────────────────
    p.add_argument("--train_backbone_steps", type=int, default=int(env("TRAIN_BACKBONE_STEPS", 8000)),
                   help="Number of steps to keep backbone frozen before unfreezing (independent of DDSP stages). 0 disables backbone freeze.")
    p.add_argument("--backbone_unfreeze_layers", type=int, default=int(env("BACKBONE_UNFREEZE_LAYERS", 2)),
                   help="How many of the last AST/ViT layers to unfreeze at Stage 1.")
    p.add_argument("--backbone_lr", type=float, default=float(env("BACKBONE_LR", 1e-5)),
                   help="Learning rate for the (partially) unfrozen backbone param group.")
    p.add_argument("--unfreeze_layernorm", type=str2bool, default=str2bool(env("UNFREEZE_LAYERNORM", "true")),
                   help="Also unfreeze LayerNorm parameters across the backbone.")
    p.add_argument("--unfreeze_pos_embed", type=str2bool, default=str2bool(env("UNFREEZE_POS_EMBED", "false")),
                   help="Also unfreeze positional embeddings in the backbone.")

    # ─── Testing mode ──────────────────────────────────────────────────────
    p.add_argument("--test", action="store_true",
                   default=str2bool(env("TEST", "false")),
                   help="If set, load the NSynth 'test' split for both training and validation")
    return p.parse_args()

# -----------------------------------------------------------------
def build_optimizer(model, lr_main, lr_backbone):
    """Build Adam with two param groups: main (heads/etc) and backbone. Decoder is frozen."""
    def _is_backbone(name: str) -> bool:
        return (".backbone." in name) or name.startswith("backbone.") or name.startswith("z_encoder.backbone")

    main_params, bb_params = [], []
    for name, p in model.named_parameters():
        if name.startswith("decoder."):
            p.requires_grad = False
            continue
        if not p.requires_grad:
            continue
        (bb_params if _is_backbone(name) else main_params).append(p)

    groups = []
    if main_params:
        groups.append({"params": main_params, "lr": lr_main})
    if bb_params:
        groups.append({"params": bb_params, "lr": lr_backbone})
    return optim.Adam(groups)
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
# --- Print gradients for z_encoder heads (gain_head, loop_head, geq_head) if present
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

# === Two‑harmonic decay → (p,g) helpers ===========================================
@torch.no_grad()
def _stft_mag(x: torch.Tensor, fs: int, n_fft: int = 1024, hop: int = 256) -> torch.Tensor:
    """Return magnitude STFT with frames-major shape [frames, freq_bins]. x: [T] (1D)."""
    x = x.detach()
    win = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                   window=win, center=True, return_complex=True)
    mag = X.abs().transpose(0, 1)  # [frames, bins]
    return mag

@torch.no_grad()
def _sample_mag_at_freqs(mag: torch.Tensor, fs: int, n_fft: int, f_target_hz: torch.Tensor) -> torch.Tensor:
    """Linear-interpolate magnitude at target frequencies per frame.
    mag: [frames, bins], f_target_hz: [frames]
    """
    df = fs / float(n_fft)
    b = f_target_hz / df  # [frames]
    frames, bins = mag.shape
    b0 = torch.clamp(b.floor().long(), 0, bins - 2)
    w  = (b - b0.float()).clamp(0, 1)
    rows = torch.arange(frames, device=mag.device)
    m0 = mag[rows, b0]
    m1 = mag[rows, b0 + 1]
    return (1.0 - w) * m0 + w * m1  # [frames]

@torch.no_grad()
def _fit_log_decay(ampl: torch.Tensor, hop: int, fs: int, t_skip_sec: float = 0.040, eps: float = 1e-12) -> float:
    """OLS slope m (s^-1) of ln amplitude vs time for a single harmonic envelope."""
    frames = ampl.numel()
    if frames <= 3:
        return 0.0
    t = torch.arange(frames, device=ampl.device, dtype=ampl.dtype) * (hop / float(fs))
    y = torch.log(ampl + eps)
    start = int(round(t_skip_sec * fs / hop))
    start = min(max(start, 0), frames - 2)
    t = t[start:]; y = y[start:]
    if t.numel() <= 1:
        return 0.0
    t_mean = t.mean()
    y_mean = y.mean()
    num = ((t - t_mean) * (y - y_mean)).sum()
    den = ((t - t_mean) ** 2).sum().clamp_min(1e-12)
    m = (num / den).item()
    return m

@torch.no_grad()
def _estimate_pg_two_harmonics(x_seg: torch.Tensor, fs: int, f0_med: float,
                               n_fft: int = 1024, hop: int = 256) -> tuple[float, float, dict]:
    """Estimate (p,g) from fundamental & 3rd harmonic in one onset window.
    Returns (p_hat, g_hat, diagnostics_dict). Robust, closed-form.
    """
    T = x_seg.numel()
    if T < max(512, 4 * hop):
        return 0.0, 0.95, {"ok": False, "reason": "window too short"}
    # Guard f0 range
    if not (1.0 < f0_med < 0.45 * fs):
        return 0.0, 0.95, {"ok": False, "reason": "f0 out of range"}

    mag = _stft_mag(x_seg, fs, n_fft=n_fft, hop=hop)  # [frames, bins]
    frames = mag.shape[0]
    if frames <= 2:
        return 0.0, 0.95, {"ok": False, "reason": "too few frames"}

    f1 = torch.full((frames,), float(f0_med), dtype=mag.dtype, device=mag.device)
    f3 = 3.0 * f1

    # Keep 3rd below Nyquist margin
    if (f3.max().item() >= 0.45 * fs):
        return 0.0, 0.95, {"ok": False, "reason": "3rd near Nyquist"}

    A1 = _sample_mag_at_freqs(mag, fs, n_fft, f1)
    A3 = _sample_mag_at_freqs(mag, fs, n_fft, f3)

    m1 = _fit_log_decay(A1, hop, fs)  # s^-1
    m3 = _fit_log_decay(A3, hop, fs)

    # Per-period multipliers
    T_p = 1.0 / max(f0_med, 1e-9)
    M1 = math.exp(m1 * T_p)
    M3 = math.exp(m3 * T_p)

    # Solve quadratic for p
    R2 = (M3 / max(M1, 1e-12)) ** 2
    omega1 = 2.0 * math.pi * (f0_med / fs)
    omega3 = 3.0 * omega1
    c1, c3 = math.cos(omega1), math.cos(omega3)

    a = R2 - 1.0
    b = 2.0 * (c1 - R2 * c3)
    c = R2 - 1.0
    disc = b * b - 4.0 * a * c

    p_candidates = []
    if abs(a) > 1e-12 and disc >= 0.0:
        sqrt_disc = math.sqrt(disc)
        p_candidates = [(-b + sqrt_disc) / (2.0 * a), (-b - sqrt_disc) / (2.0 * a)]

    p = None
    for p_try in p_candidates:
        if 1e-6 < p_try < 1.0 - 1e-6:
            p = p_try
            break
    if p is None:
        p = 0.0  # flat loss fallback

    denom = (1.0 - p)
    num = math.sqrt(max(1e-18, 1.0 + p * p - 2.0 * p * c1))
    g = M1 * num / max(1e-9, denom)
    # Clamp to model domain
    p = float(min(max(p, 1e-6), 1.0 - 1e-6))
    g = float(min(max(g, 0.900001), 0.999999))

    return p, g, {"ok": True, "M1": M1, "M3": M3, "f0_med": f0_med}

@torch.no_grad()
def _samples_to_int_frame(n_samples: int, T_int: int, s_idx: int) -> int:
    """Map sample index to internal frame index [0, T_int-1] with linear scaling."""
    if n_samples <= 1:
        return 0
    pos = (float(s_idx) / float(max(1, n_samples - 1))) * float(max(1, T_int - 1))
    j = int(math.floor(pos + 1e-8))
    return max(0, min(T_int - 1, j))

# --- Onset normalization helper ------------------------------------------
@torch.no_grad()
def _normalize_onsets_to_samples(onsets, T: int, T_int: int, fs: int):
    """
    Normalize onset list to sample indices in [0, T-1] (int).
    Accepts onsets as list/array/tensor; supports 3 conventions:
      - If max_val <= T_int-1: treat as internal frame indices [0, T_int-1]
      - Else if max_val <= (T/fs)*1.05: treat as seconds
      - Else: treat as sample indices (clamp to [0, T-1])
    Returns list of ints (may be empty).
    """
    # Defensive: handle non-list or empty
    if not isinstance(onsets, (list, tuple, np.ndarray, torch.Tensor)) or len(onsets) == 0:
        return []
    # Convert to float list
    if isinstance(onsets, torch.Tensor):
        onsets = onsets.detach().cpu().tolist()
    vals = []
    try:
        vals = [float(v) for v in onsets]
    except Exception:
        return []
    if len(vals) == 0:
        return []
    max_val = max(vals)
    # Internal frame indices
    if max_val <= T_int - 1:
        # Map to sample indices
        return [int(round(v / max(1, T_int - 1) * (T - 1))) for v in vals]
    # Seconds
    elif max_val <= (T / fs) * 1.05:
        return [int(round(v * fs)) for v in vals]
    # Sample indices (already in samples)
    else:
        return [min(max(0, int(round(v))), T - 1) for v in vals]


@torch.no_grad()
def _build_pg_targets_for_batch(audio_b: torch.Tensor,
                                pitch_b: torch.Tensor,
                                onset_lists: list,
                                fs: int,
                                T_int: int,
                                n_fft: int = 1024,
                                hop: int = 256) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-frame piece‑wise constant targets p_hat, g_hat for each batch item.
    Returns (p_tgt, g_tgt, mask) with shape [B, T_int] on the same device as audio_b.
    """
    B, T = audio_b.shape
    device = audio_b.device
    dtype = audio_b.dtype
    p_tgt = torch.zeros((B, T_int), device=device, dtype=dtype)
    g_tgt = torch.zeros((B, T_int), device=device, dtype=dtype)
    mask  = torch.zeros((B, T_int), device=device, dtype=dtype)

    for b in range(B):
        x = audio_b[b]
        f0 = pitch_b[b]
        # Accept either python lists or tensors for onsets
        onsets = onset_lists[b]
        # Normalize onsets to sample indices
        onsets_samples = _normalize_onsets_to_samples(onsets, T=T, T_int=T_int, fs=fs)
        # Ensure boundaries
        edges = sorted(set([0] + [int(max(0, min(int(T - 1), int(s)))) for s in onsets_samples] + [int(T)]))
        # Make windows
        for i in range(len(edges) - 1):
            s = edges[i]
            e = edges[i + 1]
            if e - s < max(256, 2 * hop):
                continue
            x_seg = x[s:e]
            f0_seg = f0[s:e]
            # Robust median f0 in window (ignore zeros)
            f0_valid = f0_seg[f0_seg > 1.0]
            if f0_valid.numel() == 0:
                continue
            f0_med = float(torch.median(f0_valid).item())
            p_hat, g_hat, _diag = _estimate_pg_two_harmonics(x_seg, fs, f0_med, n_fft=n_fft, hop=hop)

            s_int = _samples_to_int_frame(T, T_int, s)
            e_int = _samples_to_int_frame(T, T_int, e - 1) + 1
            e_int = max(s_int + 1, min(T_int, e_int))

            p_tgt[b, s_int:e_int] = p_hat
            g_tgt[b, s_int:e_int] = g_hat
            mask[b, s_int:e_int]  = 1.0

    return p_tgt, g_tgt, mask
# ======================================================================

# --- DDSP stage trainability helpers -------------------------------------

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

def rebuild_optimizer(optimizer, model, lr_main, lr_backbone):
    """Recreate optimizer after changing requires_grad flags."""
    del optimizer
    optimizer = build_optimizer(model, lr_main, lr_backbone)
    print("[STAGE] Rebuilt optimizer for new trainable set")
    print_trainable_summary(model, optimizer)
    return optimizer
# -----------------------------------------------------------------

def set_stage1_unfreeze_partial(model, n_layers: int, also_unfreeze_layernorm: bool, train_pos_embed: bool) -> None:
    """Stage 1: partially unfreeze backbone (last N layers), keep decoder frozen."""
    assert hasattr(model, "unfreeze_backbone_last"), (
        "Model must implement unfreeze_backbone_last(); update the model or this call site."
    )
    model.unfreeze_backbone_last(
        n_layers=n_layers,
        also_unfreeze_layernorm=also_unfreeze_layernorm,
        train_pos_embed=train_pos_embed,
    )
    for name, p in model.named_parameters():
        if name.startswith("decoder."):
            p.requires_grad = False

def main():
    args = parse_args()
    # If --test is enabled, override splits to use the NSynth 'test' subset
    split_train = "test" if args.test else "train"
    split_val   = "test" if args.test else "valid"
    config = {
        "loop_order": args.l_order,
        "filter_type": args.filter_type,
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
        "backbone_unfreeze_layers": args.backbone_unfreeze_layers,
        "backbone_lr": args.backbone_lr,
        "unfreeze_layernorm": args.unfreeze_layernorm,
        "unfreeze_pos_embed": args.unfreeze_pos_embed,
        "train_backbone_steps": args.train_backbone_steps,
        "pg_weight": args.pg_weight,
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
    def log_epoch(train_loss: float, val_loss: float, step: int):
        if wandb.run is not None:
            wandb.log({
                "train loss per epoch": train_loss,
                "val loss per epoch":   val_loss,
            }, step=int(step))

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
        filter_type=config["filter_type"],
    ).to(device)

    # ---- Backbone freeze / partial unfreeze based on train_backbone_steps ----
    current_stage = 0 if config["train_backbone_steps"] > 0 else 1
    if current_stage == 0:
        print("[BACKBONE] Freezing backbone initially; heads/mel-attn trainable")
        set_stage0_backbone_frozen(model)
    else:
        print(f"[BACKBONE] Partially unfreezing backbone: last {args.backbone_unfreeze_layers} layers "
              f"(LayerNorm={'on' if args.unfreeze_layernorm else 'off'}, "
              f"pos_embed={'on' if args.unfreeze_pos_embed else 'off'})")
        set_stage1_unfreeze_partial(
            model,
            n_layers=args.backbone_unfreeze_layers,
            also_unfreeze_layernorm=args.unfreeze_layernorm,
            train_pos_embed=args.unfreeze_pos_embed,
        )

    optimizer = build_optimizer(model, lr_main=config["learning_rate"], lr_backbone=config["backbone_lr"])
    # Log current stage to wandb (removed as per instruction)

    print_trainable_summary(model, optimizer)
    # STFT loss (scale-variant)
    mr_stft_sv = build_smooth_mrstft(scale_invariance=False).to(device).float()

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

        # Accumulators for per-component gradient means across the epoch
        comp_grad_sums = defaultdict(float)
        comp_grad_counts = defaultdict(int)

        #torch.autograd.set_detect_anomaly(True)

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

            stft_sv = mr_stft_sv(recon.unsqueeze(1), audio.unsqueeze(1))
            loss = args.stft_weight * stft_sv

            # ---- Two-harmonic (p,g) prior loss from onset windows -----------------
            if config.get("pg_weight", 0.0) > 0.0:
                # Get loop coeffs (b0, a1, g) and onset lists from the model
                _lc, _geq, _gf, _gu, _pre, _post, _exc, onset_lists = model(
                    pitch, loud, audio, config["sample_rate"], return_parameters=True,
                )
                # _lc: [B, T_int, K]; with return_gain=True, K=3 and _lc[...,1] is a1, _lc[...,2] is g
                B, T_int, K = _lc.shape
                # Build per-frame targets
                p_tgt, g_tgt, mask = _build_pg_targets_for_batch(
                    audio_b=audio, pitch_b=pitch.squeeze(2), onset_lists=onset_lists,
                    fs=config["sample_rate"], T_int=T_int, n_fft=1024, hop=256,
                )
                # Derive predicted p,g per frame from loop coeffs
                if config["filter_type"].lower() == "iir":
                    p_pred = _lc[..., 1]                      # a1 == p
                    g_pred = _lc[..., 2]                      # g
                else:
                    # FIR: a1 = p * g, b0 = (1-p) * g
                    a1 = _lc[..., 1]
                    g_pred = _lc[..., 2]
                    eps = 1e-8
                    p_pred = a1 / (g_pred + eps)
                    p_pred = torch.clamp(p_pred, 1e-6, 1.0 - 1e-6)
                # Compute masked L2 prior
                mask = mask.to(p_pred.dtype)
                denom = mask.sum().clamp_min(1.0)
                pg_mse = (((p_pred - p_tgt) ** 2 + (g_pred - g_tgt) ** 2) * mask).sum() / denom
                loss_pg = pg_mse
                loss = loss + config["pg_weight"] * loss_pg
                if wandb.run is not None:
                    wandb.log({
                        "train/loss_pg": float(loss_pg.detach().cpu().item()),
                        "train/pg_mask_count": float(denom.detach().cpu().item()),
                    }, step=int(global_step))
            # -----------------------------------------------------------------------

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

            # Backbone transition: when reaching train_backbone_steps, partially unfreeze backbone
            if current_stage == 0 and global_step >= config["train_backbone_steps"]:
                print(f"[BACKBONE] Reached global_step={global_step} → partially unfreezing last {config['backbone_unfreeze_layers']} layers")
                set_stage1_unfreeze_partial(
                    model,
                    n_layers=config["backbone_unfreeze_layers"],
                    also_unfreeze_layernorm=config["unfreeze_layernorm"],
                    train_pos_embed=config["unfreeze_pos_embed"],
                )
                optimizer = rebuild_optimizer(optimizer, model, lr_main=config["learning_rate"], lr_backbone=config["backbone_lr"])
                current_stage = 1

            if wandb.run is not None:
                wandb.log({
                    "train loss per batch": float(loss.item()),
                    "train/loss_stft": float(stft_sv.item()),
                }, step=int(global_step))

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
                wandb.log({"nonfinite_loss_detected": True, "epoch": epoch, "batch": batch_idx}, step=int(global_step))
                wandb.run.summary["nonfinite_loss"] = True
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
                    recon_std = model(
                        pitch, loud, audio, config["sample_rate"],
                    )
                    assert recon_std.shape[1] == audio.shape[1]
                    loss_std = mr_stft_sv(recon_std.unsqueeze(1), audio.unsqueeze(1))
                    v_losses_std.append(loss_std.item())
            v_loss_std = float(np.mean(v_losses_std))

        # ─── LOGGING ───────────────────────────────────────────────────────
        if epoch % config["eval_interval"] == 0:
            log_epoch(t_loss, v_loss_std, step=int(global_step))
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
                    p, l, a, config["sample_rate"],
                )
                assert rec.shape[1] == a.shape[1]

                # Fetch parameter info and internal signals
                _lc, _geq, gain_frames_b, gain_up_b, excitation_pregain_b, excitation_postgain_b, excitation_b, fixed_onsets = model(
                    p, l, a, config["sample_rate"],
                    return_parameters=True,
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
                # --- Compute piece-wise constant (p,g) targets for composite plots ---
                # Use same logic as in training: build targets from fixed batch
                # Use fixed_audio, fixed_pitch, and onsets from model call
                # Build per-frame (p,g) targets for the fixed batch
                p_tgt, g_tgt, mask = _build_pg_targets_for_batch(
                    audio_b=a, pitch_b=p.squeeze(2), onset_lists=fixed_onsets,
                    fs=config["sample_rate"], T_int=_lc.shape[1], n_fft=1024, hop=256,
                )

                for idx in range(n_plot):
                    target_np = a[idx].cpu().numpy()
                    recon_np  = rec[idx].cpu().numpy()
                    lc_np = _lc[idx].detach().cpu().numpy()  # [T_int, K]
                    eq_gains_np = gains_db_batch[idx].detach().cpu().numpy() if gains_db_batch is not None else None

                    # Build piece-wise constant target loop coeffs for plotting
                    # p_tgt, g_tgt: [B, T_int]
                    p_tgt_np = p_tgt[idx].detach().cpu().numpy()
                    g_tgt_np = g_tgt[idx].detach().cpu().numpy()
                    if config["filter_type"].lower() == "iir":
                        a1 = p_tgt_np
                        g = g_tgt_np
                        b0 = (1.0 - a1) * g
                    else:
                        # FIR: a1 = p * g, b0 = (1-p) * g
                        a1 = p_tgt_np * g_tgt_np
                        g = g_tgt_np
                        b0 = (1.0 - p_tgt_np) * g_tgt_np
                    # Stack as [T_int, 3]: [b0, a1, g]
                    lc_targets_np = np.stack([b0, a1, g], axis=-1)

                    comp_path = os.path.join(config["save_dir"], f"composite_e{epoch}_{idx}.png")
                    plot_composite_four(
                        comp_path,
                        target=target_np,
                        reconstructed=recon_np,
                        loop_coeffs_c=lc_np,
                        eq_gains=eq_gains_np,
                        sr=config["sample_rate"],
                        loop_coeffs_target=lc_targets_np,
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
                    wandb.log(media_log, commit=True, step=int(global_step))

        if epoch % config["eval_interval"] == 0:
            print(f"[E{epoch}] train={t_loss:.4f} val_std={v_loss_std:.4f} best={best_val_loss:.4f} (no-improve {epochs_since_improve}/{config['patience']})")
        else:
            print(f"[E{epoch}] train={t_loss:.4f} best={best_val_loss:.4f} (no-improve {epochs_since_improve}/{config['patience']})")

    wandb.finish()


if __name__ == '__main__':
    mp.freeze_support()
    main()
