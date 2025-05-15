from __future__ import annotations
import os, json
from typing import List, Dict, Any

import torch
from torch import nn
import torchaudio
from tqdm import tqdm
from third_party.auraloss.auraloss.perceptual import FIRFilter
import penn
from paths import NSYNTH_DIR, NSYNTH_PREPROCESSED_DIR
import argparse

from utils import get_device, midi_to_hz

# --------------------------
# Globals
# --------------------------
SAMPLE_RATE = 16_000
HOP_SIZE = 256                      # 250 frames per 4‑s clip
SEGMENT_LENGTH = SAMPLE_RATE * 4    # 64000 samples

E2_MIDI, E6_MIDI = 40, 88
E2_HZ, E6_HZ = 82.41, 1318.51

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
def a_weighted_loudness(x: torch.Tensor) -> torch.Tensor:
    """Return log‑power loudness per frame (B, F). x is (B, N)."""
    y = a_weight.fir(x.unsqueeze(1)).pow(2)                               # (B,1,N)
    frames = y.unfold(-1, HOP_SIZE, HOP_SIZE).mean(-1)                    # (B,F)
    return torch.log(frames + 1e-8)

@torch.no_grad()
def fcnf0pp_pitch(batch: torch.Tensor,
                  sr: int = SAMPLE_RATE,
                  hop_s: float = HOP_SIZE / SAMPLE_RATE,
                  fmin: float = E2_HZ,
                  fmax: float = E6_HZ,
                  interpolation_unvoiced: float = 0.065):
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

    device = x.device

    if device.type == "cpu":
        batch_size = 1024
        gpu = None
    elif device.type == "mps":
        batch_size = 2048
        gpu = None
    else:
        batch_size = 8192
        gpu = device.index

    # Feed the whole batch through FCNF0++
    pitches = []
    for clip in x:  # iterate over batch dimension
        p, _ = penn.from_audio(
            clip.unsqueeze(0), sr,  # shape (1,1,N)
            hopsize=hop_s,
            fmin=fmin, fmax=fmax,
            batch_size=batch_size,
            decoder='argmax',
            center='half-hop',
            gpu = gpu,
            interp_unvoiced_at = interpolation_unvoiced,
        )
        pitches.append(p)  # (1, F)

    return torch.cat(pitches, dim=0)  # (B, F)

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
                      pitch_mode: str = "meta",
                      interpolation_unvoiced: float = 0.065,
                      max_files: int | None = None):
    os.makedirs(out_dir, exist_ok=True)

    for split in splits:
        print(f"\n▶ Processing {split}…")
        split_in = os.path.join(nsynth_root, f"nsynth-{split}")
        with open(os.path.join(split_in, "examples.json")) as f:
            meta_json: Dict[str, Any] = json.load(f)

        keys = [k for k,m in meta_json.items()
                if E2_MIDI <= m["pitch"] <= E6_MIDI
                and (not families or m["instrument_family_str"] in families)
                and (not sources or m["instrument_source_str"] in sources)
                and m.get("qualities", [0]*10)[8] == 0  # Remove tempo-synced notes
                and m.get("qualities", [0]*10)[9] == 0] # Remove notes with reverberation
        if max_files: keys = keys[:max_files]

        split_out = os.path.join(out_dir, split, pitch_mode)
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
                pitch = fcnf0pp_pitch(audio_batch, interpolation_unvoiced=interpolation_unvoiced)
            elif pitch_mode == "autocorrelation":
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
                    "path": os.path.join(split, pitch_mode, f"{k}.pt"),
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
            split:str="test",
            pitch_mode:str="meta",
            families:List[str]|None="guitar",
            sources:List[str]|None="acoustic"
            ):
        self.base = os.path.join(root, split, pitch_mode)
        self.meta = json.load(open(os.path.join(self.base, "metadata.json")))
        stats = json.load(open(os.path.join(self.base, f"{split}_stats.json")))
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

        pitch = pitch.unsqueeze(-1)  # (F,1)
        loud = loud.unsqueeze(-1)  # (F,1)

        return itm["audio"], pitch, loud

    def get_filename(self, idx):
        return self.keys[idx]

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

    parser.add_argument("--pitch_mode", type=str, default=env("PITCH_MODE", "meta"),
                        choices=["fcnf0", "meta"])
    parser.add_argument("--interpolation_unvoiced", type=float, default=env("INTERPOLATION_UNVOICED", 0.065),)

    parser.add_argument("--max_files", type=int, default=env("MAX_FILES", None))

    cli_args, _ = parser.parse_known_args()

    print("\n▶Running with config:")
    for k, v in vars(cli_args).items():
        print(f"   {k:12}: {v}")

    preprocess_nsynth(
        nsynth_root=NSYNTH_DIR,
        out_dir=NSYNTH_PREPROCESSED_DIR,
        families=[f.strip() for f in cli_args.families.split(",")],
        sources=[s.strip() for s in cli_args.sources.split(",")],
        splits=[s.strip() for s in cli_args.split.split(",")],
        batch_size=cli_args.batch_size,
        pitch_mode=cli_args.pitch_mode,
        interpolation_unvoiced=cli_args.interpolation_unvoiced,
        max_files=cli_args.max_files,
    )