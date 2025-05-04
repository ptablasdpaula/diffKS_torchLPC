"""
Optimisation / inference layer with two clear families:

  ▸ Optimiser       – actively searches for parameters (GD, GA …)
  ▸ NeuralInference – frozen networks evaluated in one forward pass
"""
from __future__ import annotations
import math, time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple, Callable, List

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import pygad

# ───────────────────────── loss utility ─────────────────────────
def _compute_losses(metrics: Dict[str, Callable],
                    pred: torch.Tensor, target: torch.Tensor,
                    agent: nn.Module | None = None):
    out = {}
    for name, fn in metrics.items():
        if name == "param":
            out[name] = float(fn(agent).cpu())
        else:
            out[name] = float(fn(pred, target).mean().cpu())
    return out


# ╭──────────────── base classes – different signatures ──────────╮
class Optimiser(ABC):
    """Search‑type strategies (gradient, genetic…)."""

    def __init__(self, cfg, device, metrics):
        self.cfg, self.device, self.metrics = cfg, device, metrics
        self.verbose = cfg.get("verbose", True)

    @abstractmethod
    def optimise(self, agent: nn.Module,
                 batch: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
        ...

class NeuralInference(ABC):
    """Frozen neural networks (AE, CNN…)."""

    def __init__(self, cfg, device, metrics):
        self.cfg, self.device, self.metrics = cfg, device, metrics

    @abstractmethod
    def infer(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
        ...
# ╰───────────────────────────────────────────────────────────────╯


# ╭───────────────── 1. Gradient Descent ─────────────────────────╮
class GradientDescentOptimiser(Optimiser):
    def __init__(self, cfg, device, metrics):
        super().__init__(cfg, device, metrics)
        self.sample_rate = cfg.get("sample_rate", 16_000)
        self.direct      = cfg.get("direct", False)
        self.loss_fn     = metrics["stft"]

    def optimise(self, agent, batch):
        target, pitch = (t.to(self.device) for t in batch)

        opt = torch.optim.Adam(agent.parameters(), lr=self.cfg.get("lr", 0.035))
        best_loss, best_pred = math.inf, None

        iterator = (tq := tqdm(range(self.cfg.get("max_steps", 500)),
                               desc="GD", leave=False)) if self.verbose else range(self.cfg.get("max_steps", 500))

        for _ in iterator:
            pred = agent(f0_frames=pitch, input=target,
                         input_sr=self.sample_rate, direct=self.direct)
            loss = self.loss_fn(pred.unsqueeze(1), target.unsqueeze(1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if loss.item() < best_loss:
                best_loss, best_pred = loss.item(), pred.detach()
            if self.verbose:
                tq.set_postfix(loss=f"{best_loss:.4f}")

        final_pred = best_pred if best_pred is not None else pred.detach()
        return {
            "pred": final_pred.cpu(),
            **_compute_losses(self.metrics,
                              final_pred.unsqueeze(1),
                              target.unsqueeze(1),
                              agent)
        }


# ╭───────────────── 2. Genetic Algorithm ────────────────────────╮
class GeneticAlgorithmOptimiser(Optimiser):
    def __init__(self, cfg, device, metrics):
        super().__init__(cfg, device, metrics)
        self.sample_rate = cfg.get("sample_rate", 16_000)
        self.direct      = cfg.get("direct", False)
        self.loss_fn     = metrics["stft"]

    @staticmethod
    def _split(sol: np.ndarray, shapes: List[Tuple[int, ...]], device) -> Tuple[torch.Tensor, ...]:
        idx, parts = 0, []
        for sh in shapes:
            size = int(np.prod(sh))
            arr  = torch.tensor(sol[idx:idx+size].reshape(sh), device=device)
            parts.append(arr); idx += size
        return tuple(parts)

    def optimise(self, agent, batch):
        target, pitch = (t.to(self.device) for t in batch)

        shapes = (agent.loop_coefficients.shape,
                  agent.loop_gain.shape,
                  agent.exc_coefficients.shape)
        num_genes = sum(int(np.prod(s)) for s in shapes)

        def fitness(_, sol, __):
            loop_b, loop_g, exc_b = self._split(sol, shapes, self.device)
            with torch.no_grad():
                pred = agent(f0_frames=pitch, input=target,
                             input_sr=self.sample_rate, direct=self.direct,
                             loop_coefficients=loop_b, loop_gain=loop_g,
                             exc_coefficients=exc_b)
                return -float(self.loss_fn(pred.unsqueeze(1), target.unsqueeze(1)))

        pbar = tqdm(total=self.cfg.get("max_steps", 200),
                    desc="GA", leave=False) if self.verbose else None
        ga = pygad.GA(num_generations=self.cfg.get("max_steps", 200),
                      sol_per_pop=self.cfg.get("population", 32),
                      num_parents_mating=self.cfg.get("parents", 20),
                      num_genes=num_genes, fitness_func=fitness,
                      gene_type=np.float32,
                      on_generation=lambda g: pbar.update(1) if pbar else None)
        ga.run(); pbar.close() if pbar else None

        best_sol = ga.best_solution()[0]
        loop_b, loop_g, exc_b = self._split(best_sol, shapes, self.device)

        with torch.no_grad():
            final_pred = agent(f0_frames=pitch, input=target,
                               input_sr=self.sample_rate, direct=self.direct,
                               loop_coefficients=loop_b, loop_gain=loop_g,
                               exc_coefficients=exc_b)

        losses = _compute_losses(self.metrics,
                                 final_pred.unsqueeze(1),
                                 target.unsqueeze(1))
        losses["stft"] = -ga.best_solution()[1]
        return {"pred": final_pred.cpu(), **losses}

def inspect_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    print(f"Top-level type: {type(ckpt)}")

    if isinstance(ckpt, dict):
        print("\nTop-level keys:")
        for k in ckpt.keys():
            print(f"  • {k}: {type(ckpt[k])}")

        # If it contains a nested state_dict:
        model_sd = ckpt.get("model_state_dict", ckpt)
    else:
        model_sd = ckpt

    if isinstance(model_sd, dict):
        print(f"\nFound {len(model_sd)} tensors in state_dict:")
        for name, tensor in model_sd.items():
            # only print a few to avoid huge dumps
            print(f"  • {name:<50} → shape {tuple(tensor.shape)}")
    else:
        print("Checkpoint isn’t a dict of tensors – found:", type(model_sd))

# ╭───────────────── 3. Auto‑encoder Inference ────────────────────╮
class AutoencoderInference(NeuralInference):
    """
    Runs the pretrained AE_KarplusModel, returning both audio and the
    network-predicted KS parameters for parameter-loss benchmarking.

    Output dict keys:
      • "pred"              – reconstructed audio (Tensor [B, N])
      • "loop_coefficients" – list of lists [[…]] shape [B, loop_n_frames, loop_order+1]
      • "loop_gain"         – list of lists [[…]] shape [B, loop_n_frames, 1]
      • "exc_coefficients"  – list of lists [[…]] shape [B, exc_n_frames, exc_order+1]
      • plus any loss metrics (e.g. "stft")
    """
    def __init__(self,
                 cfg: Dict[str, Any],
                 device: torch.device,
                 metrics: Dict[str, Callable]):
        super().__init__(cfg, device, metrics)
        self.sample_rate = cfg.get("sample_rate", 16_000)

        # 1) load checkpoint on CPU
        ckpt_path = Path(cfg["checkpoint"]).expanduser()
        raw = torch.load(ckpt_path, map_location="cpu")
        state = raw.get("model_state_dict", raw)

        # 3) build model with the correct batch_size
        from experiments.autoencoder.model import AE_KarplusModel, MfccTimeDistributedRnnEncoder
        defaults = dict(
            hidden_size        = 512,
            batch_size         = 8,
            loop_order         = cfg.get("loop_order", 2),
            loop_n_frames      = cfg.get("loop_n_frames", 16),
            exc_order          = cfg.get("exc_order", 5),
            exc_n_frames       = cfg.get("exc_n_frames", 25),
            internal_sr        = cfg.get("internal_sr", 41_000),
            interpolation_type = cfg.get("interpolation_type", "linear"),
            z_encoder          = MfccTimeDistributedRnnEncoder(
                                     rnn_channels=512,
                                     z_dims      = 16,
                                     z_time_steps=250,
                                     sample_rate = self.sample_rate
                                 )
        )
        defaults.update(cfg.get("model_kwargs", {}))
        self.net = AE_KarplusModel(**defaults).to(device)

        # 4) load weights strictly (now shapes match)
        self.net.load_state_dict(state, strict=False)
        self.net.eval()

        # 5) ensure we can grab the raw network-predicted coeffs
        if not hasattr(self.net, "forward_with_coeffs"):
            self._wrap_forward_with_coeffs()

    def _wrap_forward_with_coeffs(self):
        """
        Monkey-patches AE_KarplusModel to expose its three projection outputs
        (loop_coefficients, loop_gain, exc_coefficients) alongside the audio.
        """
        orig_forward = self.net.forward

        def forward_with_coeffs(*args, **kwargs):
            # extract inputs
            pitch    = kwargs["pitch"]     # [B, F, 1]
            loudness = kwargs["loudness"]  # [B, F, 1]
            audio    = kwargs["audio"]     # [B, N]
            audio_sr = kwargs["audio_sr"]

            # replicate the network-only portion to grab coeff tensors
            z = self.net.z_encoder(audio, f0_scaled=pitch)

            hidden = torch.cat([
                self.net.in_mlps[0](pitch),
                self.net.in_mlps[1](loudness),
            ], dim=-1)

            hidden = torch.cat([self.net.gru(hidden)[0], pitch, loudness, z], dim=-1)
            hidden = self.net.out_mlp(hidden).mean(dim=1, keepdim=True)

            B = hidden.shape[0]
            lc_flat = self.net.loop_coeff_proj(hidden)   # [B, loop_n_frames*(order+1)]
            ec_flat = self.net.exc_coeff_proj(hidden)    # [B, exc_n_frames*(order+1)]
            lg_flat = self.net.loop_gain_proj(hidden)    # [B, loop_n_frames]

            loop_coefficients = lc_flat.reshape(
                B, self.net.loop_n_frames, self.net.loop_order + 1
            )
            exc_coefficients = ec_flat.reshape(
                B, self.net.exc_n_frames, self.net.exc_order + 1
            )
            loop_gain = lg_flat.reshape(
                B, self.net.loop_n_frames, 1
            )

            # now run the frozen decoder
            pred = self.net.decoder(
                f0_frames         = pitch.squeeze(-1),
                input             = audio,
                input_sr          = audio_sr,
                loop_coefficients = loop_coefficients,
                loop_gain         = loop_gain,
                exc_coefficients  = exc_coefficients,
            )
            return pred, loop_coefficients, loop_gain, exc_coefficients

        self.net.forward_with_coeffs = forward_with_coeffs  # type: ignore

    def infer(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:

        target, pitch, loud = (t.to(self.device) for t in batch)

        with torch.no_grad():
            pred, lc, lg, ec = self.net.forward_with_coeffs(
                pitch=pitch, loudness=loud,
                audio=target, audio_sr=self.sample_rate
            )

        return {
            "pred": pred.cpu(),
            "loop_coefficients": lc.squeeze(0).cpu().tolist(),
            "loop_gain":         lg.squeeze(0).cpu().tolist(),
            "exc_coefficients":  ec.squeeze(0).cpu().tolist(),
            **_compute_losses(self.metrics,
                              pred.unsqueeze(1),
                              target.unsqueeze(1))
        }



# ╭───────────────── 4. CNN Inference (placeholder) ───────────────╮
class CNNInference(NeuralInference):
    def __init__(self, cfg, device, metrics):
        super().__init__(cfg, device, metrics)
        self.net = cfg["model"]                      # supply from runner
        self.net.to(device).eval()
        self.device = device

    def infer(self, batch):
        target, pitch = (t.to(self.device) for t in batch)
        with torch.no_grad():
            pred = self.net(audio=target, pitch=pitch)
        return {
            "pred": pred.cpu(),
            **_compute_losses(self.metrics,
                              pred.unsqueeze(1),
                              target.unsqueeze(1))
        }

# ───────────────────────── registry ──────────────────────────────
OPTIMISER_REGISTRY = {
    "gradient"    : GradientDescentOptimiser,
    "genetic"     : GeneticAlgorithmOptimiser,
    "autoencoder" : AutoencoderInference,
    "cnn"         : CNNInference,
}
