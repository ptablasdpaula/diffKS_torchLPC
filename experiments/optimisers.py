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
            out[name] = fn(agent)
        else:
            out[name] = fn(pred, target)
    return out


# ╭──────────────── base classes – different signatures ──────────╮
class Optimiser(ABC):
    """Search‑type strategies (gradient, genetic…)."""

    def __init__(self, cfg, device, metrics):
        self.cfg, self.device, self.metrics = cfg, device, metrics
        self.verbose = cfg.get("verbose", True)
        self.iteration_times = []

    def optimise(self, agent: nn.Module,
                 batch: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
        """
        Wrapper that adds timing measurements around the actual optimization.
        The actual algorithm implementation should be in _optimise.
        """
        # Record total optimization time
        total_start_time = time.time()

        # Clear previous timing data
        self.iteration_times = []

        # Call the concrete implementation
        result = self._optimise(agent, batch)

        # Calculate total time
        total_time = time.time() - total_start_time

        # Add timing information to results
        result["total_time"] = total_time
        result["avg_iteration_time"] = np.mean(self.iteration_times) if self.iteration_times else 0
        result["total_iterations"] = len(self.iteration_times)
        result["iteration_times"] = self.iteration_times

        return result

    @abstractmethod
    def _optimise(self, agent: nn.Module,
                 batch: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
        """
        Concrete implementations should override this method.
        This is where the actual optimization algorithm goes.
        """
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
        self.direct = cfg.get("direct", False)
        self.loss_fn = metrics["stft"]
        self.early_stop_threshold = cfg.get("early_stop_threshold", 0.0001)
        self.early_stop_patience = cfg.get("early_stop_patience", 20)

    def _optimise(self, agent, batch):
        target, pitch = (t.to(self.device) for t in batch)

        opt = torch.optim.Adam(agent.parameters(), lr=self.cfg.get("lr", 0.035))
        best_loss, best_pred = math.inf, None

        # Early stopping variables
        non_improving_iterations = 0

        max_steps = self.cfg.get("max_steps", 500)
        iterator = (tq := tqdm(range(max_steps), desc="GD", leave=False)) if self.verbose else range(max_steps)

        for i in iterator:
            iter_start_time = time.time()

            pred = agent(f0_frames=pitch, input=target,
                         input_sr=self.sample_rate, direct=self.direct)
            loss = self.loss_fn(pred.unsqueeze(1), target.unsqueeze(1))
            opt.zero_grad(set_to_none=True);
            loss.backward();
            opt.step()

            self.iteration_times.append(time.time() - iter_start_time)

            current_loss = loss.item()

            # Check if loss improved by at least threshold
            if best_loss - current_loss > self.early_stop_threshold:
                best_loss = current_loss
                best_pred = pred.detach()
                non_improving_iterations = 0
            else:
                non_improving_iterations += 1

            # Early stopping check
            if non_improving_iterations >= self.early_stop_patience:
                if self.verbose:
                    tq.set_description(f"GD - Early stopped at iteration {i + 1}/{max_steps}")
                break

            if self.verbose:
                tq.set_postfix(loss=f"{best_loss:.4f}",
                               patience=f"{non_improving_iterations}/{self.early_stop_patience}")

        final_pred = best_pred if best_pred is not None else pred.detach()
        return {
            "pred": final_pred.cpu(),
            **_compute_losses(self.metrics,
                              final_pred.unsqueeze(1),
                              target.unsqueeze(1),
                              agent)
        }


class GeneticAlgorithmOptimiser(Optimiser):
    def __init__(self, cfg, device, metrics):
        super().__init__(cfg, device, metrics)
        self.sample_rate = cfg.get("sample_rate", 16_000)
        self.direct = cfg.get("direct", False)
        self.loss_fn = metrics["stft"]
        self.seed = cfg.get("seed", 42)
        self.early_stop_threshold = cfg.get("early_stop_threshold", 0.0001)
        self.early_stop_patience = cfg.get("early_stop_patience", 20)

        # Early stopping variables to be used by callback
        self.best_fitness = -math.inf
        self.non_improving_generations = 0
        self.early_stopped = False

    @staticmethod
    def _split(sol: np.ndarray, shapes: List[Tuple[int, ...]], device) -> Tuple[torch.Tensor, ...]:
        idx, parts = 0, []
        for sh in shapes:
            size = int(np.prod(sh))
            arr = torch.tensor(sol[idx:idx + size].reshape(sh), device=device)
            parts.append(arr);
            idx += size
        return tuple(parts)

    def _optimise(self, agent, batch):
        target, pitch = (t.to(self.device) for t in batch)

        shapes = (agent.loop_coefficients.shape,
                  agent.loop_gain.shape,
                  agent.exc_coefficients.shape)
        num_genes = sum(int(np.prod(s)) for s in shapes)

        # Reset early stopping variables for this optimization run
        self.best_fitness = -math.inf
        self.non_improving_generations = 0
        self.early_stopped = False

        def fitness(_, sol, __):
            loop_b, loop_g, exc_b = self._split(sol, shapes, self.device)
            with torch.no_grad():
                pred = agent(f0_frames=pitch, input=target,
                             input_sr=self.sample_rate, direct=self.direct,
                             loop_coefficients=loop_b, loop_gain=loop_g,
                             exc_coefficients=exc_b)
                return -float(self.loss_fn(pred.unsqueeze(1), target.unsqueeze(1)))

        # Create a callback to track generation timing and implement early stopping
        def on_generation_callback(ga_instance):
            # Time each generation/iteration
            iter_end_time = time.time()

            # Only record the time for generations after the first one
            if hasattr(self, '_last_gen_time'):
                self.iteration_times.append(iter_end_time - self._last_gen_time)

            # Update the start time for the next generation
            self._last_gen_time = iter_end_time

            # Get current best fitness
            current_fitness = ga_instance.best_solution()[1]

            # Check if fitness improved by at least threshold (negative because we're maximizing)
            if current_fitness - self.best_fitness > self.early_stop_threshold:
                self.best_fitness = current_fitness
                self.non_improving_generations = 0
            else:
                self.non_improving_generations += 1

            # Update progress bar if verbose
            if pbar:
                pbar.update(1)
                pbar.set_postfix(
                    fitness=f"{current_fitness:.4f}",
                    patience=f"{self.non_improving_generations}/{self.early_stop_patience}"
                )

            # Early stopping check
            if self.non_improving_generations >= self.early_stop_patience:
                self.early_stopped = True
                if pbar:
                    pbar.set_description(
                        f"GA - Early stopped at generation {ga_instance.generations_completed}/{ga_instance.num_generations}")
                return "stop"

        # Set initial generation time
        self._last_gen_time = time.time()

        max_steps = self.cfg.get("max_steps", 200)
        pbar = tqdm(total=max_steps, desc="GA", leave=False) if self.verbose else None

        ga = pygad.GA(
            num_generations=max_steps,
            sol_per_pop=self.cfg.get("population", 32),
            num_parents_mating=self.cfg.get("parents", 18),
            num_genes=num_genes,
            fitness_func=fitness,
            gene_type=np.float32,
            init_range_low=-1,
            init_range_high=1,
            keep_elitism=0,
            random_mutation_min_val=-0.1,
            random_mutation_max_val=0.1,
            mutation_probability=0.5,
            on_generation=on_generation_callback,
            random_seed=self.seed,
            stop_criteria=["reach_0", "saturate_20"]  # Add stop criteria for PyGAD
        )

        ga.run()
        if pbar:
            pbar.close()

        best_sol = ga.best_solution()[0]
        loop_b, loop_g, exc_b = self._split(best_sol, shapes, self.device)

        with torch.no_grad():
            final_pred = agent(f0_frames=pitch, input=target,
                               input_sr=self.sample_rate, direct=self.direct,
                               loop_coefficients=loop_b, loop_gain=loop_g,
                               exc_coefficients=exc_b)

            agent.loop_coefficients.data = loop_b
            agent.loop_gain.data = loop_g
            agent.exc_coefficients.data = exc_b

        losses = _compute_losses(self.metrics,
                                 final_pred.unsqueeze(1),
                                 target.unsqueeze(1))
        losses["stft"] = -ga.best_solution()[1]

        # Add early stopping information to results
        losses["early_stopped"] = self.early_stopped
        losses["stopped_at_generation"] = ga.generations_completed

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

        ckpt_path = Path(cfg["checkpoint"]).expanduser()
        raw = torch.load(ckpt_path, map_location="cpu")
        state = raw.get("model_state_dict", raw)

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

# ───────────────────────── registry ──────────────────────────────
OPTIMISER_REGISTRY = {
    "gradient"    : GradientDescentOptimiser,
    "genetic"     : GeneticAlgorithmOptimiser,
    "ae_meta"     : AutoencoderInference,
    "ae_fcn"      : AutoencoderInference,
    "ae_sup"      : AutoencoderInference,
}
