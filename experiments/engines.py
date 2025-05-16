from __future__ import annotations
"""
Optimisation / inference layer for DiffKS experiments.
  • `GradientOptimiser`  – Adam-based parameter search
  • `GeneticOptimiser`   – evolutionary search with PyGAD
  • `AutoencoderInference` – frozen DiffKS auto‑encoder
"""

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pygad
import torch
from torch import nn
from tqdm import tqdm
from third_party.auraloss.auraloss.freq import MultiResolutionSTFTLoss

# ─────────────────────────── loss helper ───────────────────────────
class STFTLoss(nn.Module):
    """Multi‑resolution STFT loss wrapper (mono tensors)."""

    def __init__(self, sample_rate: int, *, scale_invariant: bool = True, perceptual: bool = True):
        super().__init__()
        self._loss = MultiResolutionSTFTLoss(
            sample_rate=sample_rate,
            scale_invariance=scale_invariant,
            perceptual_weighting=perceptual,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # (B,1,N)
        return self._loss(pred, target)


# ╭──────────────────── 1. Gradient Descent ─────────────────────╮
class GradientOptimiser:
    """Adam + early stopping search for DiffKS parameters."""

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.verbose = cfg.get("verbose", True)
        self.sample_rate = cfg.get("sample_rate", 16_000)
        self.lr = cfg.get("lr", 0.1)
        self.max_steps = cfg.get("max_steps", 250)
        self.direct = cfg.get("direct", False)
        self.early_thr = cfg.get("early_stop_threshold", 1e-4)
        self.early_pat = cfg.get("early_stop_patience", 20)
        self.loss_fn = STFTLoss(self.sample_rate).to(device)

    # main entry -----------------------------------------------------
    def optimise(self, agent: nn.Module, batch: Tuple[torch.Tensor, ...]):
        target, pitch = (t.to(self.device) for t in batch)
        opt = torch.optim.Adam(agent.parameters(), lr=self.lr)
        best_loss, best_pred = math.inf, None
        patience = 0
        iter_times: List[float] = []

        it = tqdm(range(self.max_steps), desc="GD", leave=False) if self.verbose else range(self.max_steps)
        for i in it:
            t0 = time.time()
            pred = agent(f0_frames=pitch, input=target, input_sr=self.sample_rate, direct=self.direct)
            loss = self.loss_fn(pred.unsqueeze(1), target.unsqueeze(1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            iter_times.append(time.time() - t0)

            if best_loss - loss.item() > self.early_thr:
                best_loss = loss.item()
                best_pred = pred.detach()
                patience = 0
            else:
                patience += 1

            if self.verbose:
                if isinstance(it, tqdm):
                    it.set_postfix(loss=f"{best_loss:.4f}")
            if patience >= self.early_pat:
                if self.verbose and isinstance(it, tqdm):
                    it.set_description(f"GD‑early @ {i+1}")
                break
        final_pred = best_pred if best_pred is not None else pred.detach()
        return {
            "pred": final_pred.cpu(),
            "stft": best_loss,
            "iteration_times": iter_times,
            "total_iterations": len(iter_times),
        }


# ╭──────────────────── 2. Genetic Algorithm ────────────────────╮
class GeneticOptimiser:
    """PyGAD evolutionary search for DiffKS parameters."""

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.verbose = cfg.get("verbose", True)
        self.sample_rate = cfg.get("sample_rate", 16_000)
        self.direct = cfg.get("direct", False)
        self.loss_fn = STFTLoss(self.sample_rate).to(device)
        self.seed = cfg.get("seed", 42)
        self.max_steps = cfg.get("max_steps", 250)
        self.pop = cfg.get("population", 20)
        self.parents = cfg.get("parents", 10)
        self.early_thr = cfg.get("early_stop_threshold", 1e-4)
        self.early_pat = cfg.get("early_stop_patience", 20)

    # helper to flatten / unflatten coefficients --------------------
    @staticmethod
    def _split(sol: np.ndarray, shapes, device):
        out, idx = [], 0
        for sh in shapes:
            size = int(np.prod(sh))
            out.append(torch.tensor(sol[idx:idx+size].reshape(sh), device=device))
            idx += size
        return tuple(out)

    # main entry ----------------------------------------------------
    def optimise(self, agent: nn.Module, batch: Tuple[torch.Tensor, ...]):
        target, pitch = (t.to(self.device) for t in batch)

        shapes = (
            agent.loop_coefficients.shape,
            agent.loop_gain.shape,
            agent.exc_coefficients.shape,
        )
        num_genes = sum(int(np.prod(s)) for s in shapes)
        iter_times: List[float] = []
        best_fitness = -math.inf
        patience = 0

        def fitness(_, sol, __):
            loop_b, loop_g, exc_b = self._split(sol, shapes, self.device)
            with torch.no_grad():
                pred = agent(
                    f0_frames=pitch,
                    input=target,
                    input_sr=self.sample_rate,
                    direct=self.direct,
                    loop_coefficients=loop_b,
                    loop_gain=loop_g,
                    exc_coefficients=exc_b,
                )
                return -float(self.loss_fn(pred.unsqueeze(1), target.unsqueeze(1)))

        def on_gen(ga_inst):
            end = time.time()
            if hasattr(self, "_last"):
                iter_times.append(end - self._last)
            self._last = end
            nonlocal best_fitness, patience
            cur = ga_inst.best_solution()[1]
            if cur - best_fitness > self.early_thr:
                best_fitness = cur
                patience = 0
            else:
                patience += 1
            if self.verbose and isinstance(pbar, tqdm):
                pbar.set_postfix(fitness=f"{-cur:.4f}")
            if patience >= self.early_pat:
                return "stop"

        pbar = tqdm(total=self.max_steps, desc="GA", leave=False) if self.verbose else None
        self._last = time.time()
        ga = pygad.GA(
            num_generations=self.max_steps,
            sol_per_pop=self.pop,
            num_parents_mating=self.parents,
            num_genes=num_genes,
            fitness_func=fitness,
            gene_type=np.float32,
            init_range_low=-1,
            init_range_high=1,
            on_generation=on_gen,
            random_seed=self.seed,
            keep_elitism=0,
        )
        ga.run()
        if pbar: pbar.close()

        best_sol = ga.best_solution()[0]
        loop_b, loop_g, exc_b = self._split(best_sol, shapes, self.device)
        with torch.no_grad():
            final_pred = agent(
                f0_frames=pitch,
                input=target,
                input_sr=self.sample_rate,
                direct=self.direct,
                loop_coefficients=loop_b,
                loop_gain=loop_g,
                exc_coefficients=exc_b,
            )
        return {
            "pred": final_pred.cpu(),
            "stft": -ga.best_solution()[1],
            "iteration_times": iter_times,
            "total_iterations": len(iter_times),
        }


# ╭──────────────────── 3. Auto‑encoder inference ───────────────╮
class AutoencoderInference:
    """Loads a pretrained AE_KarplusModel checkpoint and returns reconstructions."""

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.sample_rate = cfg.get("sample_rate", 16_000)
        self.loss_fn = STFTLoss(self.sample_rate).to(device)

        ckpt_path = Path(cfg["checkpoint"]).expanduser()
        raw = torch.load(ckpt_path, map_location="cpu")
        state = raw.get("model_state_dict", raw)

        from autoencoder.model import AE_KarplusModel, MfccTimeDistributedRnnEncoder
        self.net = AE_KarplusModel(
            hidden_size=512,
            batch_size=cfg.get("batch_size", 8),
            loop_order=cfg.get("loop_order", 2),
            loop_n_frames=cfg.get("loop_n_frames", 16),
            exc_order=cfg.get("exc_order", 5),
            exc_n_frames=cfg.get("exc_n_frames", 25),
            internal_sr=cfg.get("internal_sr", 41_000),
            interpolation_type=cfg.get("interpolation_type", "linear"),
            z_encoder=MfccTimeDistributedRnnEncoder(rnn_channels=512, z_dims=16, z_time_steps=250, sample_rate=self.sample_rate),
        ).to(device)
        self.net.load_state_dict(state, strict=False)
        self.net.eval()
        if not hasattr(self.net, "forward_with_coeffs"):
            self._patch_forward()

    def _patch_forward(self):
        """Expose coeffs alongside audio via `forward_with_coeffs`."""
        def fwd(*, pitch, loudness, audio, audio_sr):
            z = self.net.z_encoder(audio, f0_scaled=pitch)
            hidden = torch.cat([
                self.net.in_mlps[0](pitch),
                self.net.in_mlps[1](loudness),
            ], -1)
            hidden = torch.cat([self.net.gru(hidden)[0], pitch, loudness, z], -1)
            hidden = self.net.out_mlp(hidden).mean(1, keepdim=True)
            B = hidden.size(0)
            lc = self.net.loop_coeff_proj(hidden).reshape(B, self.net.loop_n_frames, self.net.loop_order+1)
            ec = self.net.exc_coeff_proj(hidden).reshape(B, self.net.exc_n_frames, self.net.exc_order+1)
            lg = self.net.loop_gain_proj(hidden).reshape(B, self.net.loop_n_frames, 1)
            pred = self.net.decoder(f0_frames=pitch.squeeze(-1), input=audio, input_sr=audio_sr,
                                     loop_coefficients=lc, loop_gain=lg, exc_coefficients=ec)
            return pred, lc, lg, ec
        self.net.forward_with_coeffs = fwd  # type: ignore

    @torch.no_grad()
    def infer(self, batch: Tuple[torch.Tensor, ...]):
        audio, pitch, loud = (t.to(self.device) for t in batch)
        pred, lc, lg, ec = self.net.forward_with_coeffs(pitch=pitch, loudness=loud, audio=audio, audio_sr=self.sample_rate)
        stft_val = float(self.loss_fn(pred.unsqueeze(1), audio.unsqueeze(1)))
        return {
            "pred": pred.cpu(),
            "loop_coefficients": lc.cpu().tolist(),
            "loop_gain": lg.cpu().tolist(),
            "exc_coefficients": ec.cpu().tolist(),
            "stft": stft_val,
        }


# ────────────────────────── registry ────────────────────────────
ENGINE_REGISTRY = {
    "gradient": GradientOptimiser,
    "genetic": GeneticOptimiser,
    "ae_meta": AutoencoderInference,
    "ae_fcn": AutoencoderInference,
    "ae_sup": AutoencoderInference,
}
