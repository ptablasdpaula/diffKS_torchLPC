# experiments/runner.py
from __future__ import annotations
import argparse, json
import random, time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch, torchaudio
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.synthetic_generate import random_param_batch
from data.preprocess import NsynthDataset, E2_HZ, fcnf0pp_pitch
from diffKS import DiffKS
from experiments.losses import STFTLoss
from experiments.optimisers import OPTIMISER_REGISTRY, NeuralInference
from paths import NSYNTH_PREPROCESSED_DIR, DDSP_METAF0, DDSP_FCNF0, SUPERVISED
from utils import get_device

# ───────────────────────────── config ────────────────────────────
CFG_DIFFKS = dict(
    batch_size=1,
    internal_sr=41_000,
    min_f0_hz=E2_HZ,
    loop_order=2,
    loop_n_frames=16,
    exc_order=5,
    exc_n_frames=25,
    exc_length_s=0.025,
    interp_type="linear",
)
SR = 16_000

OPT_CFG: Dict[str, Dict] = {
    "gradient": {"lr": 0.5, "max_steps": 600},
    "genetic": {"population": 32, "parents": 16, "max_steps": 600, "seed": 42},
    "ae_meta": {"checkpoint": DDSP_METAF0},
    "ae_fcn": {"checkpoint": DDSP_FCNF0},
    "ae_sup": {"checkpoint": SUPERVISED},
}


# ───────────────────────────── main ──────────────────────────────
def main() -> None:
    cli = argparse.ArgumentParser()

    cli.add_argument("--device", default=get_device(), choices=["cuda", "cpu", "mps"])
    cli.add_argument("--methods", nargs="+", default=["gradient", "genetic", "ae_meta", "ae_fcn", "ae_sup"])
    cli.add_argument("--seed", type=int, default=42, help="global RNG seed")
    cli.add_argument("--demo", action="store_true")

    args = cli.parse_args()

    # ── Init globals ---------------------------------------------------
    dev = torch.device(args.device)
    methods = tuple(args.methods)
    out_root = Path("experiments/results")
    out_root.mkdir(parents=True, exist_ok=True)

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
    metrics = {"stft": lambda p, t: stft(p, t)}

    # ----- DataLoaders -------------------------------------------------
    bs_infer = 8  # for AE
    bs_opt = 1  # for GD / GA

    ns_meta_full = NsynthDataset(root=NSYNTH_PREPROCESSED_DIR, pitch_mode="meta")
    ns_fcn_full = NsynthDataset(root=NSYNTH_PREPROCESSED_DIR, pitch_mode="fcnf0")

    if args.demo:
        idxs = list(range(8))
        ns_meta_ds = Subset(ns_meta_full, idxs)
        ns_fcn_ds = Subset(ns_fcn_full, idxs)
    else:
        ns_meta_ds = ns_meta_full
        ns_fcn_ds = ns_fcn_full

    meta_inf = DataLoader(ns_meta_ds, batch_size=bs_infer, shuffle=False)
    fcn_inf = DataLoader(ns_fcn_ds, batch_size=bs_infer, shuffle=False)
    meta_opt = DataLoader(ns_meta_ds, batch_size=bs_opt, shuffle=False)

    class OnTheFlySynth(torch.utils.data.IterableDataset):
        def __init__(self,
                     n_items: int,
                     batch_size: int,
                     diffks: DiffKS,
                     requires_fcnf0: bool = False):
            self.n_items = n_items
            self.batch_size = batch_size
            self.diffks = diffks
            self.requires_fcnf0 = requires_fcnf0
            self.rand_gen = torch.Generator(device=get_device()).manual_seed(42)

        def __iter__(self):
            for _ in range(self.n_items):
                audio, pitch, loud, true_loop, true_exc = random_param_batch(
                    self.diffks,
                    self.batch_size,
                    generator=self.rand_gen,
                )

                if self.requires_fcnf0:
                    pitch = fcnf0pp_pitch(audio, sr=SR).unsqueeze(-1)

                yield audio, pitch, loud, true_loop, true_exc

    N_SYN_ITEMS = len(ns_meta_ds)  # same count as Nsynth subset

    # instead of one synth_agent with batch_size=1, make two:
    synth_agent_inf = DiffKS(**{**CFG_DIFFKS, "batch_size": bs_infer}).to(dev)
    synth_agent_opt = DiffKS(**{**CFG_DIFFKS, "batch_size": bs_opt}).to(dev)

    raw_syn_inf = lambda fcn: DataLoader(
        OnTheFlySynth(N_SYN_ITEMS, bs_infer, synth_agent_inf, requires_fcnf0=fcn),
        batch_size=None
    )
    raw_syn_opt = lambda fcn: DataLoader(
        OnTheFlySynth(N_SYN_ITEMS, bs_opt, synth_agent_opt, requires_fcnf0=fcn),
        batch_size=None
    )

    # wrap them in a dict keyed by pitch_mode
    syn_inf = {"meta": raw_syn_inf(False),
               "fcnf0": raw_syn_inf(True)}
    syn_opt = {"meta": raw_syn_opt(False),
               "fcnf0": raw_syn_opt(True)}

    # ╭──────────────── iterate over methods ─────────────────╮
    for method in methods:
        print(f"\n=== {method.upper()} ===")
        optimiser = OPTIMISER_REGISTRY[method](OPT_CFG[method], dev, metrics)
        run_dir = out_root / method
        run_dir.mkdir(parents=True, exist_ok=True)

        is_neural = isinstance(optimiser, NeuralInference)
        requires_fcn = ("fcn" in method.lower())

        if is_neural:
            ns_loader = fcn_inf if requires_fcn else meta_inf
            syn_loader = syn_inf["fcnf0"] if requires_fcn else syn_inf["meta"]
        else:
            ns_loader = meta_opt  # search‑based methods always stick to meta
            syn_loader = syn_opt["meta"]  # synthetic set with meta‑pitch

        bucket: Dict[str, List[float]] = defaultdict(list)
        all_iteration_times = []
        total_iterations = 0

        method_start_time = time.time()

        # ───── 1. Nsynth reconstruction ─────────────────────
        tgt_dir = run_dir / "nsynth" / "target"
        tgt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir = run_dir / "nsynth" / "pred"
        pred_dir.mkdir(parents=True, exist_ok=True)

        for idx, (audio, pitch, loud) in enumerate(tqdm(ns_loader, desc=f"{method}-NSynth", position=0, leave=True), 1):
            audio, pitch, loud = audio.to(dev), pitch.to(dev), loud.to(dev)

            if is_neural:
                res = optimiser.infer((audio, pitch, loud))
            else:
                agent = DiffKS(**CFG_DIFFKS).to(dev)
                agent.reinit()
                res = optimiser.optimise(agent, (audio, pitch.squeeze(-1)))

                # Collect timing statistics from non-neural optimizers
                if "total_time" in res:
                    bucket["total_time"].append(res["total_time"])
                if "avg_iteration_time" in res:
                    bucket["avg_iteration_time"].append(res["avg_iteration_time"])
                if "total_iterations" in res:
                    total_iterations += res.get("total_iterations", 0)
                if "iteration_times" in res:
                    all_iteration_times.extend(res["iteration_times"])

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

        # ───── 2. parameter-loss benchmark ──────────────────
        tgt_p = run_dir / "param" / "target"
        tgt_p.mkdir(parents=True, exist_ok=True)
        pred_p = run_dir / "param" / "pred"
        pred_p.mkdir(parents=True, exist_ok=True)

        for idx, batch in enumerate(tqdm(syn_loader, desc=method + "-PARAM"), 1):
            # OnTheFlySynth yields five items:
            audio, pitch, loud, true_loop, true_exc = batch
            audio, pitch, loud = audio.to(dev), pitch.to(dev), loud.to(dev)
            true_loop = true_loop.to(dev)  # [B, loop_n_frames, order+1]
            true_exc = true_exc.to(dev)  # [B, exc_n_frames, order+1]

            if is_neural:
                net = optimiser.net

                pred_loop, pred_exc = net(
                    pitch=pitch, loudness=loud,
                    audio=audio, audio_sr=SR,
                    return_parameters=True
                )

                pred_audio = net(
                    pitch=pitch, loudness=loud,
                    audio=audio, audio_sr=SR,
                )

                res = {"pred": pred_audio.cpu()}
            else:
                # optimise from scratch for this one batch
                agent = DiffKS(**CFG_DIFFKS).to(dev)
                agent.reinit()
                out = optimiser.optimise(agent, (audio, pitch.squeeze(-1)))

                # Collect timing statistics from non-neural optimizers
                if "total_time" in out:
                    bucket["total_time"].append(out["total_time"])
                if "avg_iteration_time" in out:
                    bucket["avg_iteration_time"].append(out["avg_iteration_time"])
                if "total_iterations" in out:
                    total_iterations += out.get("total_iterations", 0)
                if "iteration_times" in out:
                    all_iteration_times.extend(out["iteration_times"])

                pred_loop = agent.get_constrained_l_coefficients(
                    agent.loop_coefficients,
                    agent.loop_gain
                ).detach()
                pred_exc = agent.get_constrained_exc_coefficients(
                    agent.exc_coefficients
                ).detach()
                res = {"pred": out["pred"]}

            # compute simple L1 parameter loss
            pl_loop = torch.nn.functional.l1_loss(pred_loop, true_loop)
            pl_exc = torch.nn.functional.l1_loss(pred_exc, true_exc)
            bucket["param"].append((pl_loop + pl_exc).item())

            # save target + predicted audio
            for b in range(audio.size(0)):
                torchaudio.save((tgt_p / f"{idx:05d}_{b}.wav").as_posix(),
                                audio[b].unsqueeze(0).cpu(), SR)
                torchaudio.save((pred_p / f"{idx:05d}_{b}.wav").as_posix(),
                                res["pred"][b].detach().unsqueeze(0).cpu(), SR)

        # ───── 3. summary CSV ───────────────────────────── (FIXED INDENTATION)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Calculate total method time
        method_total_time = time.time() - method_start_time
        bucket["method_total_time"] = [method_total_time]

        # For non-neural methods, store and print timing statistics
        if not is_neural:
            # Store average iteration time in the bucket
            avg_iteration_time = np.mean(all_iteration_times) if all_iteration_times else 0
            bucket["avg_iteration_time_overall"] = [avg_iteration_time]
            bucket["total_iterations"] = [total_iterations]

            # Calculate average optimization time
            avg_total_time = np.mean(bucket["total_time"]) if bucket["total_time"] else 0

            print(f"\n--- {method.upper()} Timing Stats ---")
            print(f"Method total time: {method_total_time:.4f} seconds")
            print(f"Average optimization time: {avg_total_time:.4f} seconds")
            print(f"Average time per iteration: {avg_iteration_time:.6f} seconds")
            print(f"Total iterations: {total_iterations}")

        df = pd.DataFrame([{"metric": k,
                            "mean": float(np.mean(v)),
                            "std": float(np.std(v))}
                           for k, v in bucket.items()])
        df.to_csv(run_dir / "summary.csv", index=False)

        # Add timing data to JSON summary
        summary_data = {k: {"mean": np.mean(v), "std": np.std(v)}
                        for k, v in bucket.items()}

        # Add method total time and detailed timing for non-neural methods
        summary_data["method_execution"] = {
            "total_time": method_total_time
        }

        if not is_neural:
            summary_data["method_execution"].update({
                "avg_iteration_time": avg_iteration_time,
                "total_iterations": total_iterations,
                "avg_optimization_time": avg_total_time
            })

        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary_data, f, indent=2)

        print("✔ summary saved to", run_dir / "summary.csv")
    # ╰─────────────────────────────────────────────────────╯


if __name__ == "__main__":
    main()