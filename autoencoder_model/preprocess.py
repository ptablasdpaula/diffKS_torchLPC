"""
Optimised NSynth preprocessing with:
   - PESTO
   - A‑weighting (with Auraloss' FIR filter)
We aim to keep everything in pytorch, with no use of numpy and tensorflow and vectorization where possible.
"""
from __future__ import annotations
import os, json
from typing import List, Dict, Any

import torch
import torchaudio
from tqdm import tqdm
from third_party.auraloss.auraloss.perceptual import FIRFilter
import penn
from utils.helpers import get_device
from paths import NSYNTH_DIR, NSYNTH_PREPROCESSED_DIR
import argparse

# --------------------------
# Globals
# --------------------------
SAMPLE_RATE = 16_000
HOP_SIZE = 256                      # 250 frames per 4‑s clip
SEGMENT_LENGTH = SAMPLE_RATE * 4    # 64000 samples
MIN_F0_HZ = 27.5                    # A0 – longest KS delay we allow
DEVICE = get_device()

# --------------------------
# Filters & models (loaded once)
# --------------------------
a_weight = FIRFilter(filter_type="aw", fs=SAMPLE_RATE).to(DEVICE).eval()
for p in a_weight.parameters():
    p.requires_grad_(False)

# --------------------------
# Helper functions
# --------------------------
@torch.no_grad()
def a_weighted_loudness(x: torch.Tensor) -> torch.Tensor:
    """Return log‑power loudness per frame (B, F). x is (B, N)."""
    y = a_weight.fir(x.unsqueeze(1)).pow(2)                               # (B,1,N)
    frames = y.unfold(-1, HOP_SIZE, HOP_SIZE).mean(-1)                    # (B,F)
    return torch.log(frames + 1e-8)

@torch.no_grad()
def fcnf0pp_pitch(batch: torch.Tensor,
                  sr: int = SAMPLE_RATE,
                  hop_s: float = HOP_SIZE / SAMPLE_RATE,
                  fmin: float = MIN_F0_HZ,
                  fmax: float = SAMPLE_RATE / 2,
                  batch_frames: int = HOP_SIZE):
    """
    Args
    ----
    batch : (B, N) mono waveform, normalised to [-1,1]
    Returns
    -------
    pitch : (B, F) frequency in Hz (float32)
    """
    # penn expects (B,1,N) and centre-zero audio
    x = batch                  # (B,1,N)
    x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True)+1e-9)

    # Feed the whole batch through FCNF0++
    pitches = []
    for clip in x:  # iterate over batch dimension
        p, _ = penn.from_audio(
            clip.unsqueeze(0), sr,  # shape (1,1,N)
            hopsize=hop_s,
            fmin=fmin, fmax=fmax,
            batch_size=batch_frames,
            decoder='viterbi',
            center='half-hop',
            gpu = get_device().index
        )
        pitches.append(p)  # (1, F)

    return torch.cat(pitches, dim=0)  # (B, F)

def centre_clip(x, clip_ratio=0.3):
    thr = clip_ratio * x.abs().max(dim=-1, keepdim=True).values
    return torch.where(x >  thr, x - thr,
           torch.where(x < -thr, x + thr, torch.zeros_like(x)))

@torch.no_grad()
def autocorrelation_pitch(batch: torch.Tensor,
              sr: int = SAMPLE_RATE,
              hop: int = HOP_SIZE,
              fmin: float = MIN_F0_HZ,
              fmax: float = SAMPLE_RATE / 2) -> torch.Tensor:
    """
    Fast, vectorised autocorrelation-based F0 tracker.

    batch: (B, N) float-tensor in -1…1
    returns: (B, ⌊N/hop⌋) frequency in Hz
    """
    B, N = batch.shape
    frames = batch.unfold(-1, hop*4, hop)                    # (B,F,4H)
    F, L = frames.shape[1], frames.shape[-1]

    frames = centre_clip(frames - frames.mean(-1, keepdim=True))

    pad = 8*L
    spec = torch.fft.rfft(frames, n=pad)                           # (B,F, pad//2+1)
    acf  = torch.fft.irfft(spec * spec.conj(), n=pad)              # power spectrum trick
    acf  = acf[..., :L]                                      # keep first L lags

    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    valid = acf[..., min_lag:max_lag]                        # (B,F,lag)
    peaks = valid.argmax(dim=-1) + min_lag                  # (B,F)
    f0 = sr / peaks.float()                                 # Hz

    f0[~torch.isfinite(f0)] = fmin
    f0 = torch.clamp(f0, fmin, fmax)

    return f0.to(batch)                                     # keep dtype / device

def midi_to_hz(midi : torch.Tensor) -> torch.Tensor:
    return 440 * 2 ** ((midi - 69) / 12)

# --------------------------
# Preprocessing routine
# --------------------------
@torch.no_grad()
def preprocess_nsynth(nsynth_root: str,
                      out_dir: str,
                      splits: List[str] = ("train","valid","test"),
                      families: List[str] | None = None,
                      sources: List[str] | None = None,
                      batch_size: int = 4,
                      pitch_mode: str = "fcnf0",
                      max_files: int | None = None):
    os.makedirs(out_dir, exist_ok=True)

    for split in splits:
        print(f"\n▶ Processing {split}…")
        split_in = os.path.join(nsynth_root, f"nsynth-{split}")
        with open(os.path.join(split_in, "examples.json")) as f:
            meta_json: Dict[str, Any] = json.load(f)

        keys = [k for k,m in meta_json.items()
                if (not families or m["instrument_family_str"] in families)
                and (not sources or m["instrument_source_str"] in sources)]
        if max_files: keys = keys[:max_files]

        split_out = os.path.join(out_dir, split)
        os.makedirs(split_out, exist_ok=True)
        meta_out: Dict[str,Any] = {}
        loud_all: List[torch.Tensor] = []

        # ---- batched pass ----
        for i in tqdm(range(0, len(keys), batch_size)):
            batch_keys = keys[i:i+batch_size]
            audio_list = []
            for k in batch_keys:
                wav = os.path.join(split_in, "audio", f"{k}.wav")
                wav_t, sr = torchaudio.load(wav)
                assert sr == SAMPLE_RATE and wav_t.shape[0] == 1 and wav_t.numel()==SEGMENT_LENGTH
                f"Bad file {k}"
                audio_list.append(wav_t.squeeze(0))
            audio_batch = torch.stack(audio_list).to(DEVICE)

            # features
            loud = a_weighted_loudness(audio_batch)
            loud_all.append(loud.cpu())

            if pitch_mode == "fcnf0":
                pitch = fcnf0pp_pitch(audio_batch)
            elif pitch_mode == "autocorrelation":
                #pitch = autocorrelation_pitch(audio_batch)
                raise ValueError(f"{pitch_mode} not yet supported")
            elif pitch_mode == "meta":
                pitch = midi_to_hz(torch.tensor([[meta_json[k]["pitch"]] * (SEGMENT_LENGTH // HOP_SIZE)
                                     for k in batch_keys],
                                     dtype = torch.float32, device = DEVICE))
            else:
                raise ValueError(f"Unknown pitch_mode {pitch_mode}")


            # save per‑file
            for j,k in enumerate(batch_keys):
                torch.save({
                    "audio": audio_batch[j].cpu(),
                    "pitch": pitch[j].cpu(),
                    "loudness": loud[j].cpu()
                }, os.path.join(split_out, f"{k}.pt"))
                meta_out[k] = {
                    "path": os.path.join(split, f"{k}.pt"),
                    "num_samples": SEGMENT_LENGTH,
                    "num_frames": loud.size(1),
                    **{p: meta_json[k][p] for p in ("instrument_family_str","instrument_source_str")}
                }

        # ---- loudness stats & normalisation ----
        loud_cat = torch.cat(loud_all, dim=0)   # (N,F)
        mu, std = loud_cat.mean().item(), loud_cat.std().item()
        # overwrite loudness with z‑scored version
        for k in tqdm(meta_out.keys(), desc="normalise loudness"):
            item = torch.load(os.path.join(split_out, f"{k}.pt"))
            item["loudness"] = (item["loudness"] - mu) / std
            torch.save(item, os.path.join(split_out, f"{k}.pt"))

        json.dump({"mean":mu,"std":std}, open(os.path.join(split_out, f"{split}_stats.json"),"w"), indent=2)
        json.dump(meta_out, open(os.path.join(split_out, "metadata.json"),"w"), indent=2)
        print(f"{split}: {len(meta_out)} files, μ={mu:.4f}, σ={std:.4f}")

# --------------------------
# Dataset loader
# --------------------------
class NsynthDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root:str,
            split:str="train",
            families:List[str]|None=None,
            sources:List[str]|None=None
            ):
        self.base = os.path.join(root, split)
        self.meta = json.load(open(os.path.join(self.base, "metadata.json")))
        stats = json.load(open(os.path.join(self.base, f"{split}_stats.json")))
        self.mu, self.std = stats["mean"], stats["std"]
        self.keys = [k for k,m in self.meta.items()
                     if (not families or m["instrument_family_str"] in families) and
                        (not sources or m["instrument_source_str"] in sources)]
    def __len__(self): return len(self.keys)
    def __getitem__(self, idx):
        k = self.keys[idx]
        itm = torch.load(os.path.join(self.base, f"{k}.pt"), weights_only=True)

        pitch = itm["pitch"].view(-1)  # (F_p,)
        loud = itm["loudness"].view(-1)  # (F_l,)

        F = min(len(pitch), len(loud))
        pitch = pitch[:F]
        loud = loud[:F]

        loud = (loud - self.mu) / self.std

        pitch = pitch.unsqueeze(-1)  # (F,1)
        loud = loud.unsqueeze(-1)  # (F,1)

        return itm["audio"], pitch, loud

# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    env = os.environ.get

    parser.add_argument("--batch_size", type=int, default=int(env("BATCH_SIZE", 8)))
    parser.add_argument("--split", type=str, default=env("SPLIT", "test"))

    parser.add_argument("--families", type=str, default=env("FAMILIES", "guitar"),
                        help="comma-separated list, e.g. guitar,piano")
    parser.add_argument("--sources", type=str, default=env("SOURCES", "acoustic"),
                        help="comma-separated list, e.g. acoustic,electric")

    parser.add_argument("--pitch_mode", type=str, default=env("PITCH_MODE", "fcnf0"),
                        choices=["fcnf0", "autocorrelation", "meta"])

    cli_args, _ = parser.parse_known_args()

    preprocess_nsynth(
        nsynth_root=NSYNTH_DIR,
        out_dir=NSYNTH_PREPROCESSED_DIR,
        families=[f.strip() for f in cli_args.families.split(",")],
        sources=[s.strip() for s in cli_args.sources.split(",")],
        splits=[s.strip() for s in cli_args.split.split(",")],
        batch_size=cli_args.batch_size,
        pitch_mode=cli_args.pitch_mode,
    )