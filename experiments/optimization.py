from __future__ import annotations
"""DiffKS optimisation runner

Benchmarks two search strategies—gradient descent and a genetic algorithm—on
NSynth acoustic‑guitar targets and/or randomly generated synthetic notes with
ground‑truth coefficients.

Environment variables can override CLI defaults (uppercase names in brackets):
  • METHODS   – comma‑separated list (gradient,genetic)
  • DEVICE    – cuda|cpu|mps
  • SEED      – global RNG seed
  • DATASET   – nsynth|synthetic|both

Examples
--------
Run only genetic on synthetic set via env‑vars::

    METHODS=genetic DATASET=synthetic python experiments/runner.py

Same with explicit flags::

    python experiments/runner.py --methods genetic --dataset synthetic
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.preprocess import NsynthDataset, E2_HZ
from data.synthetic_generate import OnTheFlySynth
from diffKS import DiffKS
from experiments.engines import ENGINE_REGISTRY
from utils import get_device
from paths import NSYNTH_PREPROCESSED_DIR

# ───────────────────────── configuration ─────────────────────────
SR = 16_000
CFG_DIFFKS: Dict = dict(
    batch_size       = 1,
    internal_sr      = 41_000,
    min_f0_hz        = E2_HZ,
    loop_order       = 2,
    loop_n_frames    = 16,
    exc_order        = 5,
    exc_n_frames     = 25,
    exc_length_s     = 0.025,
    interp_type      = "linear",
)

OPT_CFG: Dict[str, Dict] = {
    "gradient": {"lr": 0.1, "max_steps": 250},
    "genetic":  {"population": 20, "parents": 10, "max_steps": 250, "seed": 42},
}

TARGET_FILES: List[str] = [
    "guitar_acoustic_010-047-100",
    "guitar_acoustic_010-055-075",
    "guitar_acoustic_010-063-025",
    "guitar_acoustic_010-070-127",
    "guitar_acoustic_021-087-100",
    "guitar_acoustic_021-067-025",
]
SYNTH_BATCHES = 6  # exactly six synthetic notes

# ───────────────────── helper utilities ──────────────────────────
def locate_indices(ds: NsynthDataset, filenames: List[str]) -> List[int]:
    lookup = {ds.get_filename(i): i for i in range(len(ds))}
    return [lookup[f] for f in filenames if f in lookup]


# ─────────────────────────── main ────────────────────────────────

def main() -> None:
    env = os.environ.get  # convenience

    parser = argparse.ArgumentParser(description="DiffKS optimisation benchmark")
    parser.add_argument("--device", default=env("DEVICE", get_device()), choices=["cuda", "cpu", "mps"],
                        help="computation device [DEVICE]")
    parser.add_argument("--methods", nargs="+",
                        default=env("METHODS", "gradient,genetic").split(","),
                        choices=list(OPT_CFG.keys()), help="optimisation strategies [METHODS]")
    parser.add_argument("--seed", type=int, default=int(env("SEED", "42")), help="RNG seed [SEED]")
    parser.add_argument("--dataset", choices=["nsynth", "synthetic", "both"],
                        default=env("DATASET", "both"), help="which dataset(s) to run [DATASET]")
    args = parser.parse_args()

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    dev = torch.device(args.device)

    # datasets– build only what we need
    if args.dataset in ("nsynth", "both"):
        nsynth = NsynthDataset(root=NSYNTH_PREPROCESSED_DIR, pitch_mode="meta")
        real_loader = DataLoader(
            Subset(nsynth, locate_indices(nsynth, TARGET_FILES)), batch_size=1, shuffle=False
        )
    else:
        real_loader = None

    if args.dataset in ("synthetic", "both"):
        synth_agent = DiffKS(**CFG_DIFFKS).to(dev)
        synth_loader = DataLoader(
            OnTheFlySynth(synth_agent, num_batches=SYNTH_BATCHES),
            batch_size=None
        )
    else:
        synth_loader = None

    out_root = Path("experiments/results")

    for method in args.methods:
        print(f"\n=== {method.upper()} ===")
        optimiser = ENGINE_REGISTRY[method](OPT_CFG[method], dev)
        run_dir = out_root / method
        (run_dir / "real" / "target").mkdir(parents=True, exist_ok=True)
        (run_dir / "real" / "pred").mkdir(exist_ok=True)
        (run_dir / "synth" / "target").mkdir(parents=True, exist_ok=True)
        (run_dir / "synth" / "pred").mkdir(exist_ok=True)

        bucket: Dict[str, List[float]] = defaultdict(list)
        all_iter_times: List[float] = []
        total_iters = 0
        start_time = time.time()

        # ── NSynth evaluation ────────────────────────────────────
        if real_loader is not None:
            for idx, (audio, pitch, _loud) in enumerate(tqdm(real_loader, desc=method+"-real"), 1):
                audio, pitch = audio.to(dev), pitch.to(dev)
                agent = DiffKS(**CFG_DIFFKS).to(dev); agent.reinit()
                res = optimiser.optimise(agent, (audio, pitch.squeeze(-1)))

                bucket["stft"].append(float(res["stft"]))
                all_iter_times.extend(res.get("iteration_times", []))
                total_iters += res.get("total_iterations", 0)

                torchaudio.save(run_dir / "real" / "target" / f"{idx:03d}.wav", audio.cpu(), SR)
                torchaudio.save(run_dir / "real" / "pred"   / f"{idx:03d}.wav", res["pred"].cpu(), SR)

        # ── Synthetic evaluation ─────────────────────────────────
        if synth_loader is not None:
            for idx, batch in enumerate(tqdm(synth_loader, desc=method+"-synth"), 1):
                audio, pitch, _loud, true_loop, true_exc = batch
                audio, pitch = audio.to(dev), pitch.to(dev)
                true_loop, true_exc = true_loop.to(dev), true_exc.to(dev)

                agent = DiffKS(**CFG_DIFFKS).to(dev); agent.reinit()
                res = optimiser.optimise(agent, (audio, pitch.squeeze(-1)))

                pred_loop = agent.get_constrained_l_coefficients(agent.loop_coefficients, agent.loop_gain)
                pred_exc  = agent.get_constrained_exc_coefficients(agent.exc_coefficients)
                param_loss = torch.nn.functional.l1_loss(pred_loop, true_loop) + \
                             torch.nn.functional.l1_loss(pred_exc,  true_exc)
                bucket["param"].append(float(param_loss))

                all_iter_times.extend(res.get("iteration_times", []))
                total_iters += res.get("total_iterations", 0)

                torchaudio.save(run_dir / "synth" / "target" / f"{idx:03d}.wav", audio.cpu(), SR)
                torchaudio.save(run_dir / "synth" / "pred"   / f"{idx:03d}.wav", res["pred"].cpu(), SR)

        # ── summary ───────────────────────────────────────────────
        total_time = time.time() - start_time
        summary = {
            "stft_mean":  float(np.mean(bucket["stft"])) if bucket.get("stft") else None,
            "stft_std":   float(np.std(bucket["stft"]))  if bucket.get("stft") else None,
            "param_mean": float(np.mean(bucket["param"])) if bucket.get("param") else None,
            "param_std":  float(np.std(bucket["param"]))  if bucket.get("param") else None,
            "method_total_time": total_time,
            "avg_iteration_time": float(np.mean(all_iter_times)) if all_iter_times else 0.0,
            "total_iterations": total_iters,
        }

        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        pd.DataFrame.from_dict(summary, orient="index", columns=["value"]).to_csv(run_dir / "summary.csv")
        print("✔ saved results to", run_dir)


if __name__ == "__main__":
    main()
