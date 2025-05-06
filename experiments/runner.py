# experiments/runner.py
from __future__ import annotations
import argparse, json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch, torchaudio
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.preprocess import NsynthDataset, E2_HZ
from data.synthetic_generate import SyntheticDataset
from diffKS          import DiffKS
from experiments.losses     import STFTLoss, parameter_loss_from_meta
from experiments.optimisers import OPTIMISER_REGISTRY, NeuralInference
from paths           import NSYNTH_PREPROCESSED_DIR, SYNTHETIC_DIR, AUTOENCODER_MODEL
from utils           import get_device

# ───────────────────────────── config ────────────────────────────
CFG_DIFFKS = dict(
    batch_size     = 1,
    internal_sr    = 41_000,
    min_f0_hz      = E2_HZ,
    loop_order     = 2,
    loop_n_frames  = 16,
    exc_order      = 5,
    exc_n_frames   = 25,
    exc_length_s   = 0.025,
    interp_type    = "linear",
)
SR = 16_000

OPT_CFG: Dict[str, Dict] = {
    "gradient": { "lr": 0.5, "max_steps": 600 },
    "genetic" : { "population": 32, "parents": 16, "max_steps": 600, "seed": 42 },
    "autoencoder": { "checkpoint": AUTOENCODER_MODEL },
}

# ─── helper to convert list‑ified coeffs to tensors ──────────────
def coeffs_to_agent(loop_b, loop_g, exc_b, device) -> DiffKS:
    est = DiffKS(**CFG_DIFFKS).to(device)
    est.loop_coefficients.data = torch.tensor(loop_b, device=device).unsqueeze(0)
    est.loop_gain.data         = torch.tensor(loop_g, device=device).unsqueeze(0)
    est.exc_coefficients.data  = torch.tensor(exc_b, device=device).unsqueeze(0)
    return est

# ───────────────────────────── main ──────────────────────────────
def main() -> None:
    cli = argparse.ArgumentParser()

    cli.add_argument("--device",  default=get_device(), choices=["cuda","cpu","mps"])
    cli.add_argument("--methods", nargs="+", default=["gradient", "genetic", "autoencoder"]) # Work out how CNN will look
    cli.add_argument("--seed", type=int, default=42, help="global RNG seed")
    cli.add_argument("--demo", action="store_true")

    args = cli.parse_args()

    # ── Init globals ---------------------------------------------------
    dev      = torch.device(args.device)
    methods  = tuple(args.methods)
    out_root = Path("experiments/results"); out_root.mkdir(parents=True, exist_ok=True)

    # ── Init Random Seeds ----------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    # ── metrics --------------------------------------------------------
    stft = STFTLoss(SR).to(dev)
    metrics = {"stft": lambda p,t: stft(p,t)}

    # ----- DataLoaders -------------------------------------------------
    bs_infer = 8  # for AE / CNN
    bs_opt = 1  # for GD / GA

    ns_full  = NsynthDataset(root=NSYNTH_PREPROCESSED_DIR)
    ns_ds    = ns_full if args.demo is False else Subset(ns_full, list(range(8)))
    ns_infer = DataLoader(ns_ds, batch_size=bs_infer, shuffle=False)
    ns_opt   = DataLoader(ns_ds, batch_size=bs_opt, shuffle=False)

    syn_full = SyntheticDataset(root=SYNTHETIC_DIR,)
    syn_ds   = syn_full if args.demo is False else Subset(syn_full, list(range(8)))
    syn_infer = DataLoader(syn_ds, batch_size=bs_infer, shuffle=False)
    syn_opt  = DataLoader(syn_ds, batch_size=bs_opt, shuffle=False)

    # ╭──────────────── iterate over methods ─────────────────╮
    for method in methods:
        print(f"\n=== {method.upper()} ===")
        optimiser = OPTIMISER_REGISTRY[method](OPT_CFG[method], dev, metrics)
        run_dir   = out_root / method; run_dir.mkdir(parents=True, exist_ok=True)

        is_neural = isinstance(optimiser, NeuralInference)
        ns_loader = ns_infer if is_neural else ns_opt
        syn_loader = syn_infer if is_neural else syn_opt

        bucket: Dict[str, List[float]] = defaultdict(list)

        # ───── 1. Nsynth reconstruction ─────────────────────
        tgt_dir = run_dir / "nsynth" / "target"; tgt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir = run_dir / "nsynth" / "pred"; pred_dir.mkdir(parents=True, exist_ok=True)

        for idx, (audio, pitch, loud) in enumerate(tqdm(ns_loader, desc=f"{method}-NSynth", position=0, leave=True), 1):
            audio, pitch, loud = audio.to(dev), pitch.to(dev), loud.to(dev)

            if is_neural:
                res = optimiser.infer((audio, pitch, loud))
            else:
                agent = DiffKS(**CFG_DIFFKS).to(dev)
                agent.reinit()
                res = optimiser.optimise(agent, (audio, pitch.squeeze(-1)))

            bucket["stft"].append(res["stft"])

            for b in range(audio.size(0)):
                torchaudio.save(
                    (tgt_dir / f"{idx:05d}_{b}.wav").as_posix(),
                    audio[b].unsqueeze(0).cpu(),  # (1, N)
                    SR
                )
                torchaudio.save(
                    (pred_dir / f"{idx:05d}_{b}.wav").as_posix(),
                    res["pred"][b].unsqueeze(0),  # (1, N)
                    SR
                )

        # ───── 2. parameter‑loss benchmark ──────────────────
        tgt_p = run_dir / "param" / "target"; tgt_p.mkdir(parents=True, exist_ok=True)
        pred_p = run_dir / "param" / "pred"; pred_p.mkdir(parents=True, exist_ok=True)

        for idx, (audio, pitch, loud) in enumerate(tqdm(syn_loader, desc=method + "-PARAM"), 1):
            audio, pitch, loud = audio.to(dev), pitch.to(dev), loud.to(dev)

            audio = audio.squeeze(1)
            subset_idx = syn_ds.indices[idx - 1]
            meta = syn_full.meta[subset_idx]

            if is_neural:
                res = optimiser.infer((audio, pitch, loud))
                # ◼ Compute parameter‑loss for **every** b in batch
                for b in range(audio.size(0)):
                    est_agent = coeffs_to_agent(
                        res["loop_coefficients"][b],
                        res["loop_gain"][b],
                        res["exc_coefficients"][b],
                        dev
                    )
                    meta = syn_full.meta[syn_ds.indices[idx - 1] * bs_infer + b] \
                        if args.demo else syn_full.meta[(idx - 1) * bs_infer + b]
                    pl = float(parameter_loss_from_meta(est_agent, meta).cpu())
                    bucket["param"].append(pl)

                    torchaudio.save((tgt_p / f"{idx:05d}_{b}.wav").as_posix(),
                                    audio[b].unsqueeze(0).cpu(), SR)
                    torchaudio.save((pred_p / f"{idx:05d}_{b}.wav").as_posix(),
                                    res["pred"][b].unsqueeze(0), SR)
            else:
                est_agent = DiffKS(**CFG_DIFFKS).to(dev);
                est_agent.reinit()
                res = optimiser.optimise(est_agent, (audio, pitch.squeeze(-1)))

            pl = float(parameter_loss_from_meta(est_agent, meta).cpu())
            bucket["param"].append(pl)

            torchaudio.save((tgt_p / f"{idx:05d}.wav").as_posix(), audio[0].cpu().unsqueeze(0), SR)
            torchaudio.save((pred_p / f"{idx:05d}.wav").as_posix(), res["pred"][0].unsqueeze(0), SR)

        # ───── 3. summary CSV ─────────────────────────────
        df = pd.DataFrame([{"metric": m, "mean": np.mean(v), "std": np.std(v)}
                           for m,v in bucket.items()])

        df.to_csv(run_dir / "summary.csv", index=False)
        print("✔ summary:", run_dir / "summary.csv")
    # ╰─────────────────────────────────────────────────────╯

if __name__ == "__main__":
    main()