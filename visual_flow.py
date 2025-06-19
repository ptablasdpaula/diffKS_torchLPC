# visual_flow.py
# ----------------------------------------------------------------------
# • One shuffled NSynth-test batch (B = 8)                      (16 kHz)
# • Inference with pretrained DDSP-MetaF0 DiffKS auto-encoder
# • Extract inverse-filtered signal (resampled to 16 kHz)
# • Save audio as WAV and/or MP4 depending on DO_SAVE_* flags
# • Produce batch grid + per-sample spectrogram composites
# ----------------------------------------------------------------------
from __future__ import annotations
import random, warnings, subprocess, tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch, torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ── project imports ──────────────────────────────────────────────────
from data.preprocess            import NsynthDataset
from experiments.engines        import AutoencoderInference
from diffKS                     import DiffKS
from paths                      import NSYNTH_PREPROCESSED_DIR, DDSP_METAF0
from utils                      import get_device

# ── global run-time switches ─────────────────────────────────────────
DO_SAVE_WAV = False    # set True to keep 16-kHz WAVs
DO_SAVE_MP4 = True     # set False to skip MP4 export (needs ffmpeg)

# ── reproducible + paths ─────────────────────────────────────────────
SR, BS, SEED = 16_000, 8, 2025
DEV          = torch.device(get_device())

ROOT   = Path("autoencoder_test")
AUDIO  = ROOT / "audio";  AUDIO.mkdir(parents=True, exist_ok=True)
VIS    = ROOT / "visual"; VIS.mkdir(parents=True,  exist_ok=True)

# --------------------------------------------------------------------
# helpers                                                             #
# --------------------------------------------------------------------
def SAVE_WAV(fname: str, wav: torch.Tensor, sr: int = SR) -> None:
    if not DO_SAVE_WAV:           # honour global flag
        return
    p = AUDIO / fname
    w = wav.detach().cpu()
    if w.dim() == 1:
        w = w.unsqueeze(0)
    torchaudio.save(str(p), w, sr)

def SAVE_MP4(fname: str, wav: torch.Tensor, sr: int = SR) -> None:
    if not DO_SAVE_MP4:           # honour global flag
        return
    p = AUDIO / fname
    w = wav.detach().cpu()
    if w.dim() == 1:
        w = w.unsqueeze(0)

    # encode via ffmpeg
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        torchaudio.save(tmp.name, w, sr)
        tmp.flush()
        subprocess.check_call([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", tmp.name, "-c:a", "aac", "-b:a", "192k", str(p)
        ])

def spec(ax, sig: torch.Tensor, title: str):
    sig = sig.squeeze().cpu().numpy()
    Pxx, freqs, bins, im = ax.specgram(
        sig, NFFT=1024, Fs=SR, noverlap=768, cmap="magma", scale="dB"
    )
    im.cmap.set_bad(color="black"); im.cmap.set_under(color="black")
    im.set_clim(vmin=-100, vmax=0)
    ax.set_title(title, fontsize=8)
    ax.set_ylabel("Hz"); ax.set_xlabel("s")
    ax.set_ylim(0, SR//2); ax.set_xlim(0, len(sig)/SR)

def save_individual_spec(idx: int,
                         tgt: torch.Tensor,
                         inv: torch.Tensor,
                         pred: torch.Tensor) -> None:
    fig, axs = plt.subplots(3, 1, figsize=(4, 6), constrained_layout=True)
    spec(axs[0], tgt,  f"Tgt #{idx}")
    spec(axs[1], inv,  f"Inv #{idx}")
    spec(axs[2], pred, f"Pred #{idx}")
    fig.savefig(VIS / f"sample_{idx}.png", dpi=150)
    plt.close(fig)

def find_diffks(root: torch.nn.Module) -> Optional[DiffKS]:
    for m in root.modules():
        if isinstance(m, DiffKS):
            return m
    return None

# --------------------------------------------------------------------
# main                                                                #
# --------------------------------------------------------------------
def main() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    # 1. one shuffled batch -------------------------------------------
    loader = DataLoader(NsynthDataset(NSYNTH_PREPROCESSED_DIR, pitch_mode="meta"),
                        batch_size=BS, shuffle=True, drop_last=True)
    audio, pitch, loud = next(iter(loader))
    audio, pitch, loud = [t.to(DEV) for t in (audio, pitch, loud)]

    # 2. AE inference -------------------------------------------------
    ae  = AutoencoderInference({"checkpoint": str(DDSP_METAF0)}, DEV)
    net = ae.net.eval()
    with torch.no_grad():
        pred = net(pitch=pitch, loudness=loud, audio=audio, audio_sr=SR)

    # 3. inverse-filter path -----------------------------------------
    diffks = find_diffks(net)
    assert diffks, "DiffKS not found in checkpoint"

    f0_frames = pitch[:, :1].squeeze(-1)        # (8,1)
    with torch.no_grad():
        diffks(f0_frames=f0_frames, input=audio, input_sr=SR, direct=False)
        inv_41k = diffks.get_inverse_filtered_signal()

    inv_16k = torchaudio.functional.resample(inv_41k, 41_000, SR)

    # 4. save audio + per-sample plots -------------------------------
    for i in range(BS):
        SAVE_WAV(f"{i}-truth.wav",  audio[i])
        SAVE_WAV(f"{i}-pred.wav",   pred[i])
        SAVE_WAV(f"{i}-inv.wav",    inv_16k[i])

        SAVE_MP4(f"{i}-truth.mp4",  audio[i])
        SAVE_MP4(f"{i}-pred.mp4",   pred[i])
        SAVE_MP4(f"{i}-inv.mp4",    inv_16k[i])

        save_individual_spec(i, audio[i], inv_16k[i], pred[i])

    # 5. batch spectrogram grid --------------------------------------
    fig, axes = plt.subplots(3, BS, figsize=(2.6*BS, 7), constrained_layout=True)
    for i in range(BS):
        spec(axes[0, i], audio[i],   f"Tgt #{i}")
        spec(axes[1, i], inv_16k[i], f"Inv #{i}")
        spec(axes[2, i], pred[i],    f"Pred #{i}")
    fig.suptitle("DDSP-MetaF0 • NSynth-test batch (B = 8)", fontsize=14)
    fig.savefig(VIS / "ae_metaf0_batch.png", dpi=150)
    plt.close(fig)

    print("✓ done — audio:", AUDIO, "  •  visuals:", VIS)
    if not DO_SAVE_WAV:
        print("  (WAV saving disabled)")
    if not DO_SAVE_MP4:
        print("  (MP4 saving disabled)")

# --------------------------------------------------------------------
if __name__ == "__main__":
    main()
