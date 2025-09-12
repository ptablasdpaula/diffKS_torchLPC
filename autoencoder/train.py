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
import matplotlib.pyplot as plt

from collections import defaultdict

from paths import NSYNTH_PREPROCESSED_DIR
from data.preprocess import NsynthDataset
from utils.misc import get_device, str2bool


from .losses import build_smooth_mrstft, build_jtfst, build_a_loudness_loss
from core import detect_onsets_librosa


def parse_args():
    p = argparse.ArgumentParser()
    env = os.environ.get

    # ─── Invariance flag ──────────────────────────────────────────────────
    p.add_argument("--invariant", type=str2bool, default=str2bool(env("INVARIANT") or "false"),
                   help="If set, use scale-invariant STFT loss and RMS normalization in validation/logging")

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
    p.add_argument("--filter_type", type=str, default=(env("FILTER_TYPE", "fir")))

    # ─── Dataset filters ────────────────────────────────────────────────────
    p.add_argument("--families", type=str, default=env("FAMILIES", "guitar"))
    p.add_argument("--sources", type=str, default=env("SOURCES", "acoustic"))

    # ─── DiffKS decoder settings ───────────────────────────────────────────
    p.add_argument("--interpolation_type", type=str, default=env("INTERPOLATION_TYPE", "lagrange"))
    p.add_argument("--pitch_mode", type=str, default=env("PITCH_MODE", "meta"))

    # ─── Losses weights ────────────────────────────────────────────────────
    p.add_argument("--stft_weight", type=float, default=float(env("STFT_WEIGHT") or 1.0))
    p.add_argument("--sf_weight", type=float, default=float(env("SF_WEIGHT") or 0.0),
                   help="Weight for spectral-flux onset loss (L1 between novelty curves)")
    p.add_argument("--sf_min_freq", type=float, default=float(env("SF_MIN_FREQ") or 0.0),
                   help="Optional high-pass in Hz for spectral-flux; 0 disables")
    p.add_argument("--pg_weight", type=float, default=float(env("PG_WEIGHT") or 0.0),
                   help="Weight for onset-to-onset (p,g) supervision loss")

    # ─── DiffKS timesteps and noise bands ─────────────────────────────────
    p.add_argument("--n_noise_bands", type=int, default=int(env("N_NOISE_BANDS") or 64))

    # ─── Testing mode ──────────────────────────────────────────────────────
    p.add_argument("--test", action="store_true",
                   default=str2bool(env("TEST") or "false"),
                   help="If set, load the NSynth 'test' split for both training and validation")
    # ─── JTFST loss weight ────────────────────────────────────────────────
    p.add_argument("--jtfst_weight", type=float, default=float(env("JTFST_WEIGHT") or 0.0),
                   help="Weight for joint time-frequency scattering loss")
    # ─── A-weighted loudness loss weight ──────────────────────────────────
    p.add_argument(
        "--a_loudness_weight",
        type=float,
        default=float(env("A_LOUDNESS_WEIGHT") or 0.0),
        help="Weight for A-weighted loudness difference loss"
    )
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

# --- NEW: Print gradients for z_encoder heads (gain_head, loop_head, geq_head) if present
# --- Print gradients for z_encoder heads (gain_head, loop_head, geq_head) if present
def _hz_to_bin(n_fft: int, fs: int, hz: float) -> int:
    if hz <= 0:
        return 0
    return int(min(max(0, round(hz / (fs / float(n_fft)))), n_fft // 2))

def spectral_flux(x: torch.Tensor, fs: int, n_fft: int = 1024, hop: int = 256,
                  min_hz: float = 0.0) -> torch.Tensor:
    """
    Differentiable spectral-flux novelty curve.
    x: [B, T] -> [B, frames]
    SF(n) = sum_w ReLU(|X_w(n)| - |X_w(n-1)|)^2
    """
    assert x.dim() == 2, "spectral_flux expects [B, T]"
    B, T = x.shape
    win = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                   window=win, center=True, return_complex=True)  # [B, bins, frames]
    mag = X.abs()
    if min_hz > 0.0:
        b0 = _hz_to_bin(n_fft, fs, min_hz)
        mag = mag[:, b0:, :]
    if mag.shape[-1] <= 1:
        return torch.zeros((B, mag.shape[-1]), device=x.device, dtype=x.dtype)
    diff = torch.relu(mag[:, :, 1:] - mag[:, :, :-1])
    sf = (diff ** 2).sum(dim=1)          # [B, frames-1]
    sf = torch.nn.functional.pad(sf, (1, 0))
    return sf

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
                # Fallback to global median f0 for this item
                f0_global_valid = f0[f0 > 1.0]
                if f0_global_valid.numel() == 0:
                    continue
                f0_med = float(torch.median(f0_global_valid).item())
            else:
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
        "sf_weight": args.sf_weight,
        "sf_min_freq": args.sf_min_freq,
        "n_noise_bands": args.n_noise_bands,
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
        batch_size=config["batch_size"],
        loop_order=config["loop_order"],
        internal_sr=config["ks_sample_rate"],
        interpolation_type=config["interpolation_type"],
        filter_type=config["filter_type"],
        n_noise_bands=config["n_noise_bands"],
    ).to(device)

    optimizer = build_optimizer(model, lr_main=config["learning_rate"])
    # Log current stage to wandb (removed as per instruction)

    print_trainable_summary(model, optimizer)
    # STFT loss (scale-invariant if --invariant is set)
    mr_stft_sv = build_smooth_mrstft(scale_invariance=args.invariant).to(device).float()

    # JTFST loss (optional)
    jtfst_loss = None
    if args.jtfst_weight > 0:
        jtfst_loss = build_jtfst(shape=64000).to(device).float()

    # A-weighted loudness loss (optional)
    a_loudness_loss = None
    if args.a_loudness_weight > 0:
        a_loudness_loss = build_a_loudness_loss(p=1, reduction="mean").to(device).float()

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
            # Spectral-flux MAE (Shier et al.)
            sf_recon = spectral_flux(recon, fs=config["sample_rate"], n_fft=1024, hop=256,
                                     min_hz=config["sf_min_freq"]).float()
            sf_target = spectral_flux(audio, fs=config["sample_rate"], n_fft=1024, hop=256,
                                      min_hz=config["sf_min_freq"]).float()

            # log-scale to stabilize (log1p handles zeros safely)
            sf_recon = torch.log1p(sf_recon)
            sf_target = torch.log1p(sf_target)

            loss_sf = torch.nn.functional.l1_loss(sf_recon, sf_target)

            # (p,g) supervision from onset-to-onset decays
            onset_lists = [detect_onsets_librosa(audio[b], sr=config["sample_rate"]) for b in range(audio.size(0))]
            p_tgt, g_tgt, m_tgt = _build_pg_targets_for_batch(audio, pitch.squeeze(-1), onset_lists,
                                                              fs=config["sample_rate"], T_int=250,
                                                              n_fft=1024, hop=256)
            # Get model parameters for (p,g)
            params = model(
                pitch, loud, audio, config["sample_rate"], return_parameters=True
            )
            loop_logits = params["loop_logits"]
            #excitation  = params["excitation"]
            loop_pg = model.decoder.design_loop(loop_logits, return_gain=True)  # [B, T, 3]
            p_pred = loop_pg[..., 1]
            g_pred = loop_pg[..., 2]
            # Masked L2 over frames
            denom = m_tgt.sum().clamp_min(1.0)
            loss_pg = ((p_pred - p_tgt).abs() + (g_pred - g_tgt).abs()) * m_tgt
            loss_pg = loss_pg.sum() / denom

            # JTFST loss (if enabled)
            if jtfst_loss is not None:
                loss_jtfst = jtfst_loss(recon.unsqueeze(1), audio.unsqueeze(1))
            else:
                loss_jtfst = torch.tensor(0.0, device=audio.device, dtype=stft_sv.dtype)

            # A-weighted loudness loss (if enabled)
            if a_loudness_loss is not None:
                loss_a_loudness = a_loudness_loss(recon.unsqueeze(1), audio.unsqueeze(1))
            else:
                loss_a_loudness = torch.zeros(1, device=audio.device, dtype=stft_sv.dtype)

            loss = (
                args.stft_weight * stft_sv
                + config["sf_weight"] * loss_sf
                + config["pg_weight"] * loss_pg
                + args.jtfst_weight * loss_jtfst
                + args.a_loudness_weight * loss_a_loudness
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
            # (Removed print_grad_snapshot and print_ast_head_grads)

            # Accumulate per-component mean absolute gradient (this batch)
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    key = _component_of(name)
                    g = p.grad.detach()
                    comp_grad_sums[key] += g.abs().mean().item()
                    comp_grad_counts[key] += 1

            optimizer.step()

            global_step += 1

            # (Removed backbone transition logic)

            if wandb.run is not None:
                wandb.log({
                    "train loss per batch": float(loss.item()),
                    "train/loss_stft": float(stft_sv.item()),
                    "train/loss_sf": float(loss_sf.item()),
                    "train/loss_pg": float(loss_pg.item()),
                    "train/loss_jtfst": float(loss_jtfst.item()) if jtfst_loss is not None else 0.0,
                    "train/loss_a_loudness": float(loss_a_loudness.item()) if a_loudness_loss is not None else 0.0,
                    "epoch": int(epoch),
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
                wandb.log({"nonfinite_loss_detected": True, "epoch": int(epoch), "batch": int(batch_idx)})
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

                    '''
                    # Amplitude normalization only if invariant mode
                    if args.invariant:
                        eps = 1e-9
                        rms_target = audio.pow(2).mean(dim=1, keepdim=True).sqrt() + eps
                        rms_recon  = recon_std.pow(2).mean(dim=1, keepdim=True).sqrt() + eps
                        gain = rms_target / rms_recon
                        recon_std = recon_std * gain  # rescaled recon
                    '''

                    loss_std = mr_stft_sv(recon_std.unsqueeze(1), audio.unsqueeze(1))
                    v_losses_std.append(loss_std.item())
            v_loss_std = float(np.mean(v_losses_std))

        # ─── LOGGING ───────────────────────────────────────────────────────
        if epoch % config["eval_interval"] == 0:
            log_epoch(t_loss, v_loss_std, epoch)
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

                # --- RMS amplitude matching for rec, as in validation loss, only if invariant ---
                '''
                if args.invariant:
                    eps = 1e-9
                    rms_target = a.pow(2).mean(dim=1, keepdim=True).sqrt() + eps
                    rms_recon  = rec.pow(2).mean(dim=1, keepdim=True).sqrt() + eps
                    gain = rms_target / rms_recon
                    rec = rec * gain
                '''

                # --- Simple audio logging ---
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

                # --- Diagnostics plots (waveforms, logits/coeffs/p&g, triggers, spectral flux) ---
                # Fetch parameters for the fixed batch
                params_b = model(
                    p, l, a, config["sample_rate"], return_parameters=True
                )
                loop_logits_b = params_b["loop_logits"]
                #excitation_b  = params_b["excitation"]
                loop_pg_b = model.decoder.design_loop(loop_logits_b, return_gain=True)  # [B, T, 3]

                # Build (p,g) targets for the fixed batch using the same onset detector
                onset_lists_fix = [detect_onsets_librosa(a[b], sr=config["sample_rate"]) for b in range(a.size(0))]
                p_tgt_b, g_tgt_b, m_tgt_b = _build_pg_targets_for_batch(
                    a, p.squeeze(-1), onset_lists_fix, T_int=250,
                    fs=config["sample_rate"], n_fft=1024, hop=256
                )

                # Convenience
                fs = config["sample_rate"]
                N = a.shape[1]
                T_int = 250
                t = np.arange(N) / float(fs)
                t_frames = np.linspace(0.0, N / float(fs), T_int)

                # Limit to n_plot examples
                for idx in range(n_plot):
                    # ---- (1) Waveforms: target vs reconstruction ----
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(t, a[idx].cpu().numpy(), label="target", linewidth=0.8)
                    ax.plot(t, rec[idx].cpu().numpy(), label="recon", linewidth=0.8, alpha=0.8)
                    ax.set_title(f"Waveforms (epoch {epoch}, sample {idx})")
                    ax.set_xlabel("Time [s]")
                    ax.set_ylabel("Amplitude")
                    ax.legend(loc="upper right")
                    wave_png = os.path.join(config["save_dir"], f"wave_e{epoch}_{idx}.png")
                    fig.tight_layout()
                    fig.savefig(wave_png, dpi=150)
                    print(f"[IMG] wrote: {wave_png}")
                    plt.close(fig)

                    # ---- (2) Loop logits, p/g (pred vs tgt) ----
                    l_logits = loop_logits_b[idx].detach().cpu().numpy()  # [T, L+1]
                    pg_vals = loop_pg_b[idx].detach().cpu().numpy()        # [T, 3]
                    p_pred = pg_vals[:, 1]
                    g_pred = pg_vals[:, 2]
                    p_tgt = p_tgt_b[idx].detach().cpu().numpy()
                    g_tgt = g_tgt_b[idx].detach().cpu().numpy()

                    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
                    # plot all raw logits
                    for k in range(l_logits.shape[1]):
                        ax[0].plot(t_frames, l_logits[:, k], label=f"logit[{k}]")
                    ax[0].set_ylabel("loop logits")
                    ax[0].legend(ncol=min(4, l_logits.shape[1]), fontsize=8)
                    # p & g predictions
                    ax[1].plot(t_frames, p_pred, label="p_pred")
                    ax[1].plot(t_frames, p_tgt, label="p_tgt", linestyle=":")
                    ax[1].set_ylabel("p")
                    ax[1].legend(loc="upper right", ncol=2, fontsize=8)
                    ax[2].plot(t_frames, g_pred, label="g_pred")
                    ax[2].plot(t_frames, g_tgt, label="g_tgt", linestyle=":")
                    ax[2].set_ylabel("g")
                    ax[2].set_xlabel("Time [s]")
                    ax[2].legend(loc="upper right", ncol=2, fontsize=8)
                    coeff_png = os.path.join(config["save_dir"], f"coeffs_pg_e{epoch}_{idx}.png")
                    fig.tight_layout()
                    fig.savefig(coeff_png, dpi=150)
                    print(f"[IMG] wrote: {coeff_png}")
                    plt.close(fig)

                    # ---- (3) Onsets visualization ----
                    onsets_samples = onset_lists_fix[idx]
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(t, a[idx].cpu().numpy(), label="target", linewidth=0.5, alpha=0.5)
                    for s in onsets_samples:
                        ax.axvline(x=s / float(fs), color="r", linestyle="--", linewidth=0.8, alpha=0.7)
                    ax.set_title(f"Onsets (epoch {epoch}, sample {idx})")
                    ax.set_xlabel("Time [s]")
                    ax.set_ylabel("amplitude / onsets")
                    ax.legend(loc="upper right")
                    trig_png = os.path.join(config["save_dir"], f"triggers_e{epoch}_{idx}.png")
                    fig.tight_layout()
                    fig.savefig(trig_png, dpi=150)
                    print(f"[IMG] wrote: {trig_png}")
                    plt.close(fig)

                    # ---- (4) Spectral flux: target vs recon (+ threshold used for onsets) ----
                    x_t = a[idx:idx+1]
                    x_r = rec[idx:idx+1]
                    sf_t = spectral_flux(x_t, fs=fs, n_fft=1024, hop=256, min_hz=config["sf_min_freq"])[0].detach().cpu().numpy()
                    sf_r = spectral_flux(x_r, fs=fs, n_fft=1024, hop=256, min_hz=config["sf_min_freq"])[0].detach().cpu().numpy()
                    # smooth & threshold like detector
                    if sf_t.shape[0] >= 5:
                        sf_ts = torch.nn.functional.avg_pool1d(torch.from_numpy(sf_t).view(1,1,-1), kernel_size=5, stride=1, padding=2).view(-1).numpy()
                    else:
                        sf_ts = sf_t
                    med = np.median(sf_ts)
                    mad = np.median(np.abs(sf_ts - med)) + 1e-8
                    thr = med + 1.0 * mad
                    f = np.arange(sf_t.shape[0]) * (256.0 / float(fs))  # frame times in seconds
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(f, sf_t, label="SF target")
                    ax.plot(f, sf_r, label="SF recon")
                    ax.plot(f, np.full_like(f, thr), label="threshold", linestyle="--")
                    ax.set_title(f"Spectral Flux (epoch {epoch}, sample {idx})")
                    ax.set_xlabel("Time [s]")
                    ax.set_ylabel("Flux")
                    ax.legend(loc="upper right")
                    flux_png = os.path.join(config["save_dir"], f"flux_e{epoch}_{idx}.png")
                    fig.tight_layout()
                    fig.savefig(flux_png, dpi=150)
                    print(f"[IMG] wrote: {flux_png}")
                    plt.close(fig)

                    # Log images to W&B
                    if wandb.run is not None:
                        wandb.log({
                            "epoch": int(epoch),
                            f"img/wave_{idx}": wandb.Image(wave_png),
                            f"img/coeffs_pg_{idx}": wandb.Image(coeff_png),
                            f"img/triggers_{idx}": wandb.Image(trig_png),
                            f"img/flux_{idx}": wandb.Image(flux_png),
                        })

        if epoch % config["eval_interval"] == 0:
            print(f"[E{epoch}] train={t_loss:.4f} val_std={v_loss_std:.4f} best={best_val_loss:.4f} (no-improve {epochs_since_improve}/{config['patience']})")
        else:
            print(f"[E{epoch}] train={t_loss:.4f} best={best_val_loss:.4f} (no-improve {epochs_since_improve}/{config['patience']})")

    wandb.finish()


if __name__ == '__main__' or __name__.endswith("autoencoder.train"):
    mp.freeze_support()
    main()
