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

import json

#
# ─── Composite plotting helpers ──────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np

def _fetch_excitation_after_shelves(model: torch.nn.Module) -> torch.Tensor | None:
    """Return the per‑batch excitation saved by the decoder after shelf filters.
    Expected to be stored as a buffer `decoder.excitation` or `excitation_filter_out`.
    Returns a tensor [B, N] or None if unavailable.
    """
    x = None
    # Preferred place we've been saving it
    if hasattr(model, "decoder") and hasattr(model.decoder, "excitation") and model.decoder.excitation is not None:
        x = model.decoder.excitation
    elif hasattr(model, "excitation_filter_out") and model.excitation_filter_out is not None:
        x = model.excitation_filter_out
    if x is None:
        return None
    # Ensure shape [B, N]
    if x.dim() == 3 and x.size(1) == 1:
        x = x[:, 0, :]
    return x.detach().cpu()

def _as_param_triplet(p, idx: int | None = None):
    """Return (fc, Q, gain_db) as floats from various shapes/types.
    Accepts: dict with common keys, tuple/list of 3, tensor/ndarray of shape (3,)
    or batched shape (B,3) in which case `idx` selects the row.
    """
    if p is None:
        return None

    # Dict variants
    if isinstance(p, dict):
        for keys in (("fc", "Q", "gain_db"), ("fc_hz", "Q", "gain_db"), ("fc", "q", "gain_db")):
            if all(k in p for k in keys):
                import torch as _t
                fc  = float(_t.as_tensor(p[keys[0]]).detach().cpu().reshape(-1)[0].item())
                Q   = float(_t.as_tensor(p[keys[1]]).detach().cpu().reshape(-1)[0].item())
                gdb = float(_t.as_tensor(p[keys[2]]).detach().cpu().reshape(-1)[0].item())
                return (fc, Q, gdb)
        # If dict has batch entries, try to index into them via idx
        if idx is not None:
            if "fc" in p and hasattr(p["fc"], "__getitem__"):
                import torch as _t
                fc  = float(_t.as_tensor(p["fc"][idx]).detach().cpu().reshape(-1)[0].item())
                Q   = float(_t.as_tensor(p.get("Q", p.get("q"))[idx]).detach().cpu().reshape(-1)[0].item())
                gdb = float(_t.as_tensor(p["gain_db"][idx]).detach().cpu().reshape(-1)[0].item())
                return (fc, Q, gdb)
        raise KeyError("Unrecognized dict structure for shelf params")

    # Tuple/list of length 3
    if isinstance(p, (tuple, list)):
        if len(p) == 3:
            import torch as _t
            v = [_t.as_tensor(x).detach().cpu().reshape(-1)[0].item() for x in p]
            return (float(v[0]), float(v[1]), float(v[2]))
        raise ValueError("List/tuple shelf params must have length 3")

    # Tensor / ndarray
    try:
        import torch as _t, numpy as _np
        t = _t.as_tensor(p)
        if t.ndim == 1 and t.numel() == 3:
            t = t.flatten()
            return (float(t[0].item()), float(t[1].item()), float(t[2].item()))
        if t.ndim == 2 and t.size(-1) == 3:
            i = 0 if idx is None else int(idx)
            row = t[i]
            return (float(row[0].item()), float(row[1].item()), float(row[2].item()))
    except Exception:
        pass

    raise TypeError(f"Unsupported shelf param type/shape: {type(p)}")


# === RBJ shelf frequency-response helpers (approximation) =====================
import numpy as _np

def _rbj_shelf_coeffs_np(fc_hz: float, Q: float, gain_db: float, sr: int, which: str):
    """Return normalized biquad (b0,b1,b2,a1,a2) for RBJ low/high shelf.
    Coefficients follow Robert Bristow‑Johnson's Audio EQ Cookbook formulas.
    `which` in {"low", "high"}.
    """
    fc = float(max(1e-6, fc_hz))
    Q  = float(max(1e-6, Q))
    A  = 10.0 ** (float(gain_db) / 40.0)
    w0 = 2.0 * _np.pi * fc / float(sr)
    c, s = _np.cos(w0), _np.sin(w0)
    alpha = s / (2.0 * Q)
    beta  = 2.0 * _np.sqrt(A) * alpha
    m = (A + 1.0)
    n = (A - 1.0)

    if which == "high":
        b0 = A * ( m + n * c + beta)
        b1 = -2.0 * A * ( n + m * c)
        b2 = A * ( m + n * c - beta)
        a0 =      ( m - n * c + beta)
        a1 =  2.0 *      ( n - m * c)
        a2 =      ( m - n * c - beta)
    else:  # "low"
        b0 = A * ( m - n * c + beta)
        b1 =  2.0 * A * ( n - m * c)
        b2 = A * ( m - n * c - beta)
        a0 =      ( m + n * c + beta)
        a1 = -2.0 *      ( n + m * c)
        a2 =      ( m + n * c - beta)

    # normalize so a0 = 1
    inv_a0 = 1.0 / (a0 + 1e-30)
    b0, b1, b2 = b0 * inv_a0, b1 * inv_a0, b2 * inv_a0
    a1, a2     = a1 * inv_a0, a2 * inv_a0
    return b0, b1, b2, a1, a2


def _biquad_mag_db_np(b: tuple, a: tuple, freqs: _np.ndarray, sr: int) -> _np.ndarray:
    """Magnitude response (dB) for biquad given normalized (b0,b1,b2) and (a1,a2).
    Evaluates H(e^{jω}) on a frequency grid (cf. earlevel.com frequency-response derivation).
    """
    w = 2.0 * _np.pi * (freqs.astype(_np.float64) / float(sr))
    z1 = _np.exp(-1j * w)
    z2 = _np.exp(-2j * w)
    b0, b1, b2 = b
    a1, a2     = a
    num = b0 + b1 * z1 + b2 * z2
    den = 1.0 + a1 * z1 + a2 * z2
    H = num / (den + 1e-30)
    mag = _np.abs(H)
    # avoid -inf
    return 20.0 * _np.log10(_np.maximum(mag, 1e-12))

def plot_composite_four(fig_path: str,
                       target: np.ndarray,
                       reconstructed: np.ndarray,
                       loop_coeffs_c: np.ndarray,
                       low_params: tuple[float, float, float] | None,
                       high_params: tuple[float, float, float] | None,
                       sr: int) -> None:
    """
    Create a 4-panel composite:
      1) Target waveform
      2) Reconstructed waveform
      3) Loop filter coefficients
      4) Shelf frequency response (combined low/high)
    Saves to `fig_path`.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import sosfreqz

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

    # 4) Shelf frequency response (combined low/high)
    ax = axes[3]
    ax.set_title("Shelf Frequency Response")
    f_lo = 20.0
    f_hi = max(22.0, sr / 2.0 - 1.0)
    freqs = np.logspace(np.log10(f_lo), np.log10(f_hi), 1024)
    combined_db = np.zeros_like(freqs)
    # Use RBJ formulas as before
    def _get_sos(fc, Q, gain_db, sr, which):
        # RBJ formulas, output as [b0, b1, b2, a0, a1, a2]
        b0, b1, b2, a1, a2 = _rbj_shelf_coeffs_np(fc, Q, gain_db, sr, which)
        # a0 is always 1 after normalization
        sos = np.array([[b0, b1, b2, 1.0, a1, a2]])
        return sos
    if (low_params is not None) or (high_params is not None):
        sos_list = []
        if low_params is not None:
            l_fc, l_Q, l_g = low_params
            sos_list.append(_get_sos(l_fc, l_Q, l_g, sr, "low"))
        if high_params is not None:
            h_fc, h_Q, h_g = high_params
            sos_list.append(_get_sos(h_fc, h_Q, h_g, sr, "high"))
        if sos_list:
            # Stack cascaded SOS
            sos = np.vstack(sos_list)
            w, h = sosfreqz(sos, worN=freqs, fs=sr)
            mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
            ax.semilogx(freqs, mag_db, label="Combined")
            if low_params is not None:
                # Plot low shelf alone
                sos_low = sos_list[0:1]
                w_l, h_l = sosfreqz(np.vstack(sos_low), worN=freqs, fs=sr)
                mag_db_l = 20 * np.log10(np.maximum(np.abs(h_l), 1e-12))
                ax.semilogx(freqs, mag_db_l, linestyle=":", label="Low Shelf")
                ax.axvline(max(1.0, low_params[0]), linestyle=":", alpha=0.3)
            if high_params is not None:
                sos_high = sos_list[-1:]
                w_h, h_h = sosfreqz(np.vstack(sos_high), worN=freqs, fs=sr)
                mag_db_h = 20 * np.log10(np.maximum(np.abs(h_h), 1e-12))
                ax.semilogx(freqs, mag_db_h, linestyle="--", label="High Shelf")
                ax.axvline(max(1.0, high_params[0]), linestyle=":", alpha=0.3)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude (dB)")
            ax.grid(True, which="both", alpha=0.2)
            ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "(no shelf params)", ha="center", va="center")
        ax.set_xlabel("Frequency (Hz)")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

from data.preprocess import a_weighted_loudness

import matplotlib.pyplot as plt
# Try both known import paths for JTFS; fall back with a helpful error
try:
    from kymatio.torch import TimeFrequencyScattering  # v0.4-dev torch frontend
except Exception:
    try:
        # Some builds export JTFS at top-level as TimeFrequencyScattering1D
        from kymatio import TimeFrequencyScattering1D as TimeFrequencyScattering  # type: ignore[attr-defined]
    except Exception as e:
        raise ImportError(
            "TimeFrequencyScattering is not available in your installed kymatio.\n"
            "Install Kymatio from source (v0.4-dev) to get JTFS, e.g.:\n"
            "  pip install --upgrade 'git+https://github.com/kymatio/kymatio.git#egg=kymatio'\n"
        ) from e
import torch.nn as nn  # for JTFSTLoss subclass

# ─── JTFSTLoss: Joint Time-Frequency Scattering Loss Wrapper ─────────────
class JTFSTLoss(nn.Module):
    def __init__(
        self,
        shape,
        J=12,
        Q=(8, 2),
        J_fr=3,
        Q_fr=2,
        T="none",
        F="none",
        format="joint",
        p=2,
        device=None,
    ):
        super().__init__()
        # Q can be int or tuple
        Q1, Q2 = Q if isinstance(Q, (tuple, list)) else (Q, Q)
        # Normalize T/F from CLI (strings like "none") to what Kymatio expects
        T_arg = None if (T is None or str(T).lower() == "none") else int(T)
        F_arg = None if (F is None or str(F).lower() == "none") else int(F)
        shape_1d = shape if (isinstance(shape, tuple) and len(shape) == 1) else (shape,)
        self.jtfs = TimeFrequencyScattering(
            J=J,
            shape=shape_1d,
            Q=(Q1, Q2),
            J_fr=J_fr,
            Q_fr=Q_fr,
            T=T_arg,
            F=F_arg,
            format=format,
        )
        if device is not None:
            self.jtfs = self.jtfs.to(device)
        self.p = p

    def forward(self, x, y):
        # x, y: [B, 1, T]
        Sx = self.jtfs(x)
        Sy = self.jtfs(y)
        # Flatten all scattering coeffs into a single tensor for Lp comparison
        # If output is a dict (format="dict"), flatten all tensors; else, flatten directly
        if isinstance(Sx, dict):
            Sx_flat = torch.cat([v.flatten(1) for v in Sx.values()], dim=1)
            Sy_flat = torch.cat([v.flatten(1) for v in Sy.values()], dim=1)
        else:
            Sx_flat = Sx.flatten(1)
            Sy_flat = Sy.flatten(1)
        return torch.nn.functional.l1_loss(Sx_flat, Sy_flat) if self.p == 1 else torch.norm(Sx_flat - Sy_flat, p=self.p, dim=1).mean()

# ─── Training phase schedule ──────────────────────────────────────────────
LR_MAIN      = 1e-4   # base LR (overridden by CLI)

# Global variable for internal sample rate used by plotting helpers
MODEL_INTERNAL_SR = None  # set in main() after model construction; used by plotting helpers

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
    p.add_argument("--learning_rate", type=float, default=float(env("LEARNING_RATE", 1e-2)))
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

    # ─── Training stages ──────────────────────────────────────────────────

    # ─── Losses weights ────────────────────────────────────────────────────
    p.add_argument("--stft_weight", type=float, default=float(env("STFT_WEIGHT", 1.0)))
    p.add_argument("--smooth_weight", type=float, default=float(env("SMOOTH_WEIGHT", 0.0)))
    p.add_argument("--loud_weight", type=float, default=float(env("LOUD_WEIGHT", 0.0)))
    p.add_argument("--loud_deriv_weight", type=float,
                   default=float(env("LOUD_DERIV_WEIGHT", 0.0)),
                   help="Weight for derivative‑of‑loudness loss term")

    # ─── Testing mode ────────────────────────────────────────────────────
    p.add_argument("--test", action="store_true",
                   default=str2bool(env("TEST", "false")),
                   help="If set, load the NSynth 'test' split for both training and validation")
    # ─── JTFS Loss (Joint Time-Frequency Scattering) ──────────────────────
    p.add_argument("--jtfst_weight", type=float, default=float(env("JTFST_WEIGHT", 0.0)), help="Weight for JTFS loss term")
    p.add_argument("--jtfst_J", type=int, default=int(env("JTFST_J", 12)), help="JTFS J (time scale)")
    p.add_argument("--jtfst_Q1", type=int, default=int(env("JTFST_Q1", 8)), help="JTFS Q1 (number of wavelets per octave, time)")
    p.add_argument("--jtfst_Q2", type=int, default=int(env("JTFST_Q2", 2)), help="JTFS Q2 (number of wavelets per octave, frequency)")
    p.add_argument("--jtfst_J_fr", type=int, default=int(env("JTFST_J_FR", 3)), help="JTFS J_fr (frequency scale)")
    p.add_argument("--jtfst_Q_fr", type=int, default=int(env("JTFST_Q_FR", 2)), help="JTFS Q_fr (number of freq wavelets per octave)")
    p.add_argument("--jtfst_T", type=str, default=env("JTFST_T", "none"), help="JTFS T (time averaging)")
    p.add_argument("--jtfst_F", type=str, default=env("JTFST_F", "none"), help="JTFS F (freq averaging)")
    p.add_argument("--jtfst_format", type=str, default=env("JTFST_FORMAT", "joint"), help="JTFS output format")
    p.add_argument("--jtfst_p", type=int, default=int(env("JTFST_P", 2)), help="JTFS Lp norm (1 or 2)")
    return p.parse_args()

# -----------------------------------------------------------------
def build_optimizer(model):
    """
    Freeze decoder/spec front‑end parameters and add only parameters that
    currently require gradients. This respects layer‑level freezing done
    inside the model (e.g., coefficient projection during Stage 1).
    """
    trainable_params = []
    for name, p in model.named_parameters():
        # Always freeze the differentiable decoder
        if name.startswith("decoder."):
            p.requires_grad = False
            continue
        # Respect the current requires_grad flag
        if p.requires_grad:
            trainable_params.append(p)
    return optim.Adam([{"params": trainable_params, "lr": LR_MAIN}])
# -----------------------------------------------------------------

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
        "smooth_weight": args.smooth_weight,
        "loud_weight": args.loud_weight,
        "loud_deriv_weight": args.loud_deriv_weight,
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

    stats_path = os.path.join(NSYNTH_PREPROCESSED_DIR, split_train, args.pitch_mode, f"{split_train}_stats.json")
    with open(stats_path, "r") as f:
        stats = json.load(f)
    mu, std = float(stats["mean"]), float(stats["std"])

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
        loudness_mu=mu, loudness_std=std
    ).to(device)
    # Set global for plotting helpers
    global MODEL_INTERNAL_SR
    MODEL_INTERNAL_SR = model.internal_sr

    optimizer = build_optimizer(model)
    global_step = 0
    # Stage bookkeeping no longer needed
    #mr_stft = MultiResolutionSTFTLoss(scale_invariance=False, perceptual_weighting=True,
                                      #sample_rate=config["sample_rate"], device=device, )

    mr_stft = MultiResolutionSTFTLoss(
        fft_sizes=[257, 509, 1019, 2039, 4093],
        hop_sizes=[128, 254, 509, 1019, 2046],
        win_lengths=[257, 509, 1019, 2039, 4093],
        window="flattop",          # WF: Flat‑top window with low sidelobes
        mag_distance="L2",         # D2: squared‑L2 distance
        log_eps=1.0,               # C2: log‑compression with ε=1 to keep values ≥0
        w_sc=1.0,                  # default weighting for spectral‑convergence term
        w_log_mag=1.0,             # log‑magnitude term
        w_lin_mag=1.0,             # (disabled) linear‑magnitude term
        perceptual_weighting=True,
        scale_invariance=True,
        sample_rate=config["sample_rate"],
    )
    mr_stft = mr_stft.to(device)
    mr_stft = mr_stft.float()

    # ─── JTFS Loss instantiation ─────────────────────────────────────────
    jtfst_loss_fn = None
    if args.jtfst_weight > 0.0:
        # shape: (audio_len,) for JTFS
        jtfs_shape = (fixed_audio.size(1),)
        jtfst_loss_fn = JTFSTLoss(
            shape=jtfs_shape,
            J=args.jtfst_J,
            Q=(args.jtfst_Q1, args.jtfst_Q2),
            J_fr=args.jtfst_J_fr,
            Q_fr=args.jtfst_Q_fr,
            T=args.jtfst_T,
            F=args.jtfst_F,
            format=args.jtfst_format,
            p=args.jtfst_p,
            device=device,
        )

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
    for epoch in range(start_epoch, config["max_epochs"]):
        model.train()
        t_loss = 0
        batches_processed = 0

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
            # Sanity‑check: recon must come back at the original sample‑rate
            assert recon.shape[1] == audio.shape[1], (
                f"Decoder returned {recon.shape[1]} samples, "
                f"but target has {audio.shape[1]}. "
                "This usually means an incorrect `audio_sr` was passed to the model."
            )

            stft_loss = mr_stft(recon.unsqueeze(1), audio.unsqueeze(1)) * args.stft_weight
            smooth_loss = torch.tensor(0.0, device=device)

            # Loudness‑based losses
            loud_recon       = a_weighted_loudness(recon).transpose(1, 2)                # [B, T_sal, 1]
            loud_loss        = torch.abs(loud - loud_recon).sum() * args.loud_weight

            # ─── NEW: derivative‑of‑loudness loss ────────────────────────
            loud_deriv_target = torch.diff(loud,       dim=1)            # [B, T_sal‑1, 1]
            loud_deriv_recon  = torch.diff(loud_recon, dim=1)            # [B, T_sal‑1, 1]

            loud_deriv_loss   = torch.abs(loud_deriv_target - loud_deriv_recon).sum() * args.loud_deriv_weight

            # ─── JTFS loss (if enabled) ────────────────────────────────
            jtfs_loss = None
            if jtfst_loss_fn is not None:
                jtfs_loss = jtfst_loss_fn(recon.unsqueeze(1), audio.unsqueeze(1)) * args.jtfst_weight
                print(f"stft: {stft_loss}; loud: {loud_loss}; loud_deriv: {loud_deriv_loss}; jtfs: {jtfs_loss}")
            else:
                print(f"stft: {stft_loss}; loud: {loud_loss}; loud_deriv: {loud_deriv_loss}")

            loss = stft_loss + smooth_loss + loud_loss + loud_deriv_loss
            if jtfs_loss is not None:
                loss = loss + jtfs_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            log_train_batch(loss.item())
            t_loss += loss.item()
            batches_processed += 1

        if batches_processed > 0:
            t_loss /= batches_processed

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
                    if jtfst_loss_fn is not None:
                        loss_std = loss_std + (jtfst_loss_fn(recon_std.unsqueeze(1), audio.unsqueeze(1)) * args.jtfst_weight)
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

                print (f"CULPRIT: {p.shape}")


                rec = model(
                    pitch=p,
                    loudness=l,
                    audio=a,
                    audio_sr=config["sample_rate"],
                )
                assert rec.shape[1] == a.shape[1]



                # Fetch parameter info only
                _lc, _lp, _hp = model(
                    pitch=p,
                    loudness=l,
                    audio=a,
                    audio_sr=config["sample_rate"],
                    return_parameters=True,
                )

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
                    low_vals  = _as_param_triplet(_lp, idx=idx) if _lp is not None else None
                    high_vals = _as_param_triplet(_hp, idx=idx) if _hp is not None else None
                    comp_path = os.path.join(config["save_dir"], f"composite_e{epoch}_{idx}.png")
                    plot_composite_four(
                        comp_path,
                        target=target_np,
                        reconstructed=recon_np,
                        loop_coeffs_c=lc_np,
                        low_params=low_vals,
                        high_params=high_vals,
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
