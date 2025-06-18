from __future__ import annotations
"""
DiffKS optimisation runner
--------------------------
For every *method* (`gradient`, `genetic`, …) and requested
*dataset* (`nsynth`, `synthetic`, or `both`) it

1.  Loads `n_files` audio examples in mini-batches
    (`CFG_DIFFKS["batch_size"]`).
2.  Optimises DiffKS with the selected engine, capturing
      • the predicted audio per file  
      • the **multi-resolution STFT loss (MSL)** returned by the engine  
      • iteration-time statistics.
3.  Writes results to disk

       experiments/results/<method>/<dataset>/<target|pred>/<idx>.wav  
       experiments/results/<method>/<dataset>/per_file_metrics.csv  
       experiments/results/<method>/<dataset>/summary.{csv,json}

4.  Logs metrics and audio previews to Weights & Biases.

CLI flags
~~~~~~~~~
--device   {"cuda","cpu","mps"}     Hardware target (default: auto-detect)  
--methods  list[str]                Subset of engines to run; choose from  
                                     {gradient, genetic}.  Default: both.  
--dataset  {"nsynth","synthetic","both"}  
                                    Which dataset(s) to benchmark.  
--n_files  int                      Number of files per *real* dataset.  
--seed     int                      Global RNG seed.  

Example
~~~~~~~
$ python optimisation.py --methods gradient genetic --dataset nsynth --n_files 32
"""

# ───────────────────────── imports ─────────────────────────
import argparse, json, math, os, random, time
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Dict, List
from pprint import pprint

import numpy as np
import pandas as pd
import torch, torchaudio, wandb
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.preprocess import NsynthDataset, E2_HZ
from data.synthetic_generate import OnTheFlySynth
from diffKS import DiffKS
from experiments.engines import ENGINE_REGISTRY
from utils import get_device
from paths import NSYNTH_PREPROCESSED_DIR

# ───────────────────────── configuration ─────────────────────────
CFG_DIFFKS: Dict = dict(
    batch_size   = 4,
    internal_sr  = 41_000,
    min_f0_hz    = E2_HZ,
    loop_order   = 2,
    loop_n_frames= 4,
    exc_order    = 5,
    exc_n_frames = 25,
    exc_length_s = 0.025,
    interp_type  = "linear",
)

SR = 16_000
BATCH_SIZE = CFG_DIFFKS["batch_size"]

OPT_CFG: Dict[str, Dict] = {
    "gradient": {"lr": 0.15, "max_steps": 250},
    "genetic":  {"population": 20, "parents": 10, "max_steps": 250, "seed": 42},
}

# ───────────────────── helper utilities ──────────────────────────
def sample_nsynth_indices(ds: NsynthDataset, n: int, rng: np.random.Generator) -> List[int]:
    if n > len(ds):
        raise ValueError(f"Requested {n} files but dataset only contains {len(ds)} items.")
    return rng.choice(len(ds), size=n, replace=False).tolist()

def log_example_audio(group:str,audio_1d:torch.Tensor,sr:int,idx:int,method:str)->None:
    wandb.log({f"{method}/{group}/{idx}": wandb.Audio(audio_1d.cpu().numpy(),
                                                      sample_rate=sr,
                                                      caption=f"{group}_{idx}")}, commit=False)

# ─────────────────────────── main ────────────────────────────────
def main() -> None:
    # ─── CLI arguments ───────────────────────────────────────────
    env = os.environ.get
    p = argparse.ArgumentParser("DiffKS optimisation benchmark")
    p.add_argument("--device",  default=env("DEVICE", get_device()), choices=["cuda","cpu","mps"])
    p.add_argument("--methods", nargs="+", default=env("METHODS","gradient,genetic").split(","),
                   choices=list(OPT_CFG.keys()))
    p.add_argument("--seed",    type=int, default=int(env("SEED", "42")))
    p.add_argument("--dataset", choices=["nsynth","synthetic","both"], default=env("DATASET","both"))
    p.add_argument("--n_files", type=int, default=int(env("N_FILES","100")))
    args = p.parse_args(); pprint(vars(args))

    # ─── Reproducibility ─────────────────────────────────────────
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    rng, dev = np.random.default_rng(args.seed), torch.device(args.device)

    # ─── W&B ───────────────────────────────────────────────
    wandb.init(project="diffks-optimisation",
               name=f"opt-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
               config={**vars(args), **CFG_DIFFKS},
               dir=str(Path("experiments/wandb").resolve()),
               resume="allow")

    # ─── datasets ──────────────────────────────────────────
    real_loader = synth_loader = None
    if args.dataset in ("nsynth","both"):
        nsynth = NsynthDataset(root=NSYNTH_PREPROCESSED_DIR, pitch_mode="meta")
        sel = sample_nsynth_indices(nsynth, args.n_files, rng)
        real_loader = DataLoader(Subset(nsynth, sel),
                                 batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    if args.dataset in ("synthetic","both"):
        synth_agent = DiffKS(**CFG_DIFFKS).to(dev)
        n_batches = len(real_loader) if args.dataset=="both" and real_loader is not None else \
                    math.floor(args.n_files / BATCH_SIZE)
        synth_loader = DataLoader(OnTheFlySynth(synth_agent, num_batches=n_batches),
                                  batch_size=None)

    if real_loader is None and synth_loader is None:
        raise ValueError("Nothing to run – choose at least one dataset.")

    out_root = Path("experiments/results")

    # ─── per-method loop ─────────────────────────────────────
    for method in args.methods:
        print(f"\n=== {method.upper()} ===")
        optimiser = ENGINE_REGISTRY[method](OPT_CFG[method], dev)

        run_dir = out_root / method
        for ds in ["nsynth","synth"]:
            for sub in ["target","pred"]:
                (run_dir/ds/sub).mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str,float]] = []
        all_times: List[float] = []
        total_iters, t0 = 0, time.time()
        real_idx = synth_idx = 0

        # ── NSynth evaluation ─────────────────────────────
        if real_loader is not None:
            for audio, pitch, _ in tqdm(real_loader, desc=method+"-nsynth", unit="batch"):
                audio, pitch = audio.to(dev), pitch.to(dev)
                agent = DiffKS(**CFG_DIFFKS).to(dev); agent.reinit()
                res = optimiser.optimise(agent, (audio, pitch.squeeze(-1)))

                for k in range(audio.size(0)):
                    tgt, pred, msl = audio[k], res["pred"][k], float(res["msl"][k])
                    rows.append(dict(dataset="nsynth", file_id=real_idx, msl=msl))

                    torchaudio.save(run_dir/"nsynth/target"/f"{real_idx:03d}.wav",
                                    tgt.unsqueeze(0).cpu(), SR)
                    torchaudio.save(run_dir/"nsynth/pred"/f"{real_idx:03d}.wav",
                                    pred.unsqueeze(0).cpu(), SR)

                    log_example_audio("nsynth/target", tgt, SR, real_idx, method)
                    log_example_audio("nsynth/pred",   pred, SR, real_idx, method)
                    wandb.log({f"{method}/nsynth/msl": msl}, commit=True)
                    real_idx += 1

                all_times.extend(res["iteration_times"])
                total_iters += res["total_iterations"]
        # ── Synthetic evaluation ─────────────────────────
        if synth_loader is not None:
            for batch in tqdm(synth_loader, desc=method+"-synth", unit="batch"):
                audio, pitch, _loud, true_loop, true_exc = batch
                audio, pitch = audio.to(dev), pitch.to(dev)

                agent = DiffKS(**CFG_DIFFKS).to(dev); agent.reinit()
                res = optimiser.optimise(agent, (audio, pitch.squeeze(-1)))

                # obtain optimised coefficients
                pred_loop = agent.get_constrained_l_coefficients(agent.loop_coefficients,
                                                                 agent.loop_gain)
                pred_exc  = agent.get_constrained_exc_coefficients(agent.exc_coefficients)

                # per-item L1 parameter loss
                loop_err = torch.abs(pred_loop - true_loop.to(dev)).mean(dim=(1,2))
                exc_err  = torch.abs(pred_exc  - true_exc.to(dev) ).mean(dim=(1,2))
                param_vec = loop_err + exc_err    # (B,)

                for k in range(audio.size(0)):
                    tgt, pred = audio[k], res["pred"][k]
                    msl_val   = float(res["msl"][k])
                    p_loss    = float(param_vec[k])

                    rows.append(dict(dataset="synth", file_id=synth_idx,
                                     msl=msl_val, param_loss=p_loss))

                    torchaudio.save(run_dir/"synth/target"/f"{synth_idx:03d}.wav",
                                    tgt.unsqueeze(0).cpu(), SR)
                    torchaudio.save(run_dir/"synth/pred"/f"{synth_idx:03d}.wav",
                                    pred.unsqueeze(0).cpu(), SR)

                    log_example_audio("synth/target", tgt, SR, synth_idx, method)
                    log_example_audio("synth/pred",   pred, SR, synth_idx, method)
                    wandb.log({f"{method}/synth/param_loss": p_loss,
                               f"{method}/synth/msl":        msl_val},
                              commit=True)
                    synth_idx += 1

                all_times.extend(res["iteration_times"])
                total_iters += res["total_iterations"]

        # ── summaries & persistence ─────────────────────
        wandb.log({}, commit=True)                             # flush
        total_time = time.time() - t0
        df = pd.DataFrame(rows)
        iter_mu = float(np.mean(all_times)) if all_times else 0.0

        for ds in df["dataset"].unique():
            ds_dir = run_dir / ds
            df_ds  = df[df["dataset"]==ds]
            df_ds.to_csv(ds_dir/"per_file_metrics.csv", index=False)

            summary = {
                "msl_mean":   float(df_ds["msl"].mean()),
                "msl_std":    float(df_ds["msl"].std(ddof=0)),
                "n_files":    len(df_ds),
                "avg_iteration_time": iter_mu,
                "total_iterations":   total_iters,
                "method_total_time":  total_time,
            }
            # only synthetic has param_loss
            if ds == "synth" and not df_ds["param_loss"].isna().all():
                summary.update({
                    "param_mean": float(df_ds["param_loss"].mean()),
                    "param_std":  float(df_ds["param_loss"].std(ddof=0)),
                })
            Path(ds_dir/"summary.json").write_text(json.dumps(summary, indent=2))
            pd.DataFrame.from_dict(summary, orient="index",
                                   columns=["value"]).to_csv(ds_dir/"summary.csv")

            wandb.log({f"{method}/{ds}/metrics_table": wandb.Table(dataframe=df_ds)})
            wandb.log({f"{method}/{ds}/summary/{k}": v for k,v in summary.items()})

        print(f"✔  results saved to {run_dir}")

    wandb.finish()

if __name__ == "__main__":
    main()
