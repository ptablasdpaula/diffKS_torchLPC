from __future__ import annotations
"""DiffKS auto‑encoder inference runner

Evaluates a pretrained DiffKS auto‑encoder on
• the **entire acoustic‑guitar test split** of NSynth, and/or
• on‑the‑fly synthetic notes with known Karplus–Strong coefficients.

For the NSynth path we simply iterate over the full preprocessed
dataset—no hand‑picked subset.  The script aggregates STFT loss (and
parameter L1 loss for synthetic) into a JSON/CSV summary;

Environment overrides (uppercase in brackets):
  • DEVICE    – cuda|cpu|mps
  • METHODS   – ae_meta,ae_fcn,ae_sup
  • SEED      – RNG seed
  • DATASET   – nsynth|synthetic|both
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.preprocess import NsynthDataset, E2_HZ
from data.synthetic_generate import OnTheFlySynth
from diffKS import DiffKS
from experiments.engines import AutoencoderInference
from paths import NSYNTH_PREPROCESSED_DIR, DDSP_METAF0, DDSP_FCNF0, SUPERVISED
from utils import get_device

# ───────────────────────── configuration ─────────────────────────
SR = 16_000
BS = 8
CFG_DIFFKS: Dict = dict(
    batch_size       = BS,
    internal_sr      = 41_000,
    min_f0_hz        = E2_HZ,
    loop_order       = 2,
    loop_n_frames    = 16,
    exc_order        = 5,
    exc_n_frames     = 25,
    exc_length_s     = 0.025,
    interp_type      = "linear",
)

CKPTS: Dict[str, str] = {
    "ae_meta": str(DDSP_METAF0),
    "ae_fcn":  str(DDSP_FCNF0),
    "ae_sup":  str(SUPERVISED),
}

# ─────────────────────────── main ────────────────────────────────
def main() -> None:
    env = os.environ.get

    p = argparse.ArgumentParser(description="DiffKS AE inference benchmark")
    p.add_argument("--device", default=env("DEVICE", get_device()), choices=["cuda", "cpu", "mps"])
    p.add_argument("--methods", nargs="+", default=env("METHODS", "ae_meta,ae_fcn,ae_sup").split(","),
                   choices=list(CKPTS.keys()))
    p.add_argument("--seed", type=int, default=int(env("SEED", "42")))
    p.add_argument("--dataset", choices=["nsynth", "synthetic", "both"],
                   default=env("DATASET", "both"))
    args = p.parse_args()

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    dev = torch.device(args.device)

    nsynth_ds = NsynthDataset(root=NSYNTH_PREPROCESSED_DIR, pitch_mode="meta")  # full test split
    ns_loader = DataLoader(nsynth_ds, batch_size=BS, shuffle=False, drop_last=True)

    num_batches = len(nsynth_ds) // BS

    synth_agent = DiffKS(**CFG_DIFFKS).to(dev)
    synth_loader = DataLoader(OnTheFlySynth(synth_agent, num_batches=num_batches), batch_size=None, shuffle=False)

    out_root = Path("experiments/results")

    for method in args.methods:
        print(f"\n=== {method.upper()} ===")
        inferencer = AutoencoderInference({"checkpoint": CKPTS[method]}, dev)
        run_dir = out_root / method
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "nsynth" / "target").mkdir(parents=True, exist_ok=True)
        (run_dir / "nsynth" / "pred").mkdir(parents=True, exist_ok=True)
        (run_dir / "synth" / "target").mkdir(parents=True, exist_ok=True)
        (run_dir / "synth" / "pred").mkdir(parents=True, exist_ok=True)

        bucket: Dict[str, List[float]] = defaultdict(list)

        # ── NSynth evaluation ────────────────────────────────────
        if args.dataset in ("nsynth", "both"):
            for idx, (audio, pitch, loud) in enumerate(tqdm(ns_loader, desc=method+"-nsynth"), 1):
                audio, pitch, loud = audio.to(dev), pitch.to(dev), loud.to(dev)
                res = inferencer.infer((audio, pitch, loud))
                bucket["stft"].append(float(res["stft"]))

                for b in range(audio.size(0)):  # B = 8
                    torchaudio.save((run_dir / "nsynth" / "target" / f"{idx:04d}_{b}.wav").as_posix(),
                        audio[b].unsqueeze(0).cpu(), SR)
                    torchaudio.save((run_dir / "nsynth" / "pred" / f"{idx:04d}_{b}.wav").as_posix(),
                        res["pred"][b].unsqueeze(0).cpu(),  SR)

        # ── Synthetic evaluation ─────────────────────────────────
        if args.dataset in ("synthetic", "both"):
            for idx, (audio, pitch, loud, true_loop, true_exc) in enumerate(tqdm(synth_loader, desc=method + "-synth"), 1):
                audio, pitch, loud = audio.to(dev), pitch.to(dev), loud.to(dev)
                true_loop, true_exc = true_loop.to(dev), true_exc.to(dev)

                net = inferencer.net

                pred_loop, pred_exc = net(
                    pitch=pitch, loudness=loud,
                    audio=audio, audio_sr=SR,
                    return_parameters=True
                )

                pred_audio = net(
                    pitch=pitch, loudness=loud,
                    audio=audio, audio_sr=SR,
                ).detach()

                param_loss = torch.nn.functional.l1_loss(pred_loop, true_loop) + \
                             torch.nn.functional.l1_loss(pred_exc,  true_exc)
                bucket["param"].append(float(param_loss))

                for b in range(audio.size(0)):
                    torchaudio.save((run_dir / "synth" / "target" / f"{idx:04d}_{b}.wav").as_posix(),
                        audio[b].unsqueeze(0).cpu(), SR)
                    torchaudio.save((run_dir / "synth" / "pred" / f"{idx:04d}_{b}.wav").as_posix(),
                        pred_audio[b].unsqueeze(0).cpu(), SR)

        # ── summary ───────────────────────────────────────────────
        summary = {
            "stft_mean":  float(np.mean(bucket["stft"])) if bucket.get("stft") else None,
            "stft_std":   float(np.std(bucket["stft"]))  if bucket.get("stft") else None,
            "param_mean": float(np.mean(bucket["param"])) if bucket.get("param") else None,
            "param_std":  float(np.std(bucket["param"]))  if bucket.get("param") else None,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        pd.DataFrame.from_dict(summary, orient="index", columns=["value"]).to_csv(run_dir / "summary.csv")
        print("✔ saved summary to", run_dir)


if __name__ == "__main__":
    main()
