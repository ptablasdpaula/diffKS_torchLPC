from __future__ import annotations

import argparse, json, math, random
from functools import partial
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchaudio

from diffKS import DiffKS
from data.preprocess import E2_HZ
from utils import get_device, noise_burst, midi_to_hz

from torch.utils.data import Dataset

from paths import SYNTHETIC_DIR

# ───────────────────────── configuration ──────────────────────────────────
KS_CFG = dict(
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

SR_OUT      = 16_000              # export sample‑rate
DUR_SEC     = 4.0                 # full clip length
BURST_LEN_S = KS_CFG["exc_length_s"]  # 25ms
START_2ND   = 3.0                 # second burst at 3s
SAMPLE_RATE = 16000

MIDI_E2, MIDI_E6 = 40, 88
N_NOTES          = 88 - 40

# ─── loudness helpers ───────────────────────────────────────────
from third_party.auraloss.auraloss.perceptual import FIRFilter

HOP        = 256                      # match Nsynth preprocessing
a_weight   = FIRFilter(filter_type="aw", fs=SR_OUT).to(get_device()).eval()
for p in a_weight.parameters():  p.requires_grad_(False)

def a_weighted_loudness(x: torch.Tensor) -> torch.Tensor: # TODO: we should find a way to not repeat this
    """Log‑power loudness per frame –(B,F).  x is (B,N)."""
    y = a_weight.fir(x.unsqueeze(1)).pow(2)          # (B,1,N)
    frames = y.unfold(-1, HOP, HOP).mean(-1)         # (B,F)
    return torch.log(frames + 1e-8)

# ───────────────────────── helpers ────────────────────────────────────────
def gen_burst(batch_size=1, generator=None, sr=SR_OUT) -> torch.Tensor:
    """Return [1,SR_OUT·DUR_SEC] excitation with bursts at 0s and 3s."""
    total_n  = int(sr * DUR_SEC)
    burst_n  = int(sr * BURST_LEN_S)
    offset2  = int(sr * START_2ND)

    x = torch.zeros(batch_size, total_n, device=get_device())

    burst = noise_burst(sample_rate=sr, length_s=BURST_LEN_S, burst_width_s=BURST_LEN_S, normalize=True, batch_size=batch_size, generator=generator)

    x[:, :burst_n] = burst
    x[:, offset2:offset2 + burst_n] = burst * 0.1
    return x


def make_random_params(agent: DiffKS) -> None:
    """Randomise learnables with sinusoidal patterns using random frequencies."""
    with torch.no_grad():
        # Handle loop_gain (shape [1, 16, 1])
        gains_main = agent.loop_gain[:, :12]
        gains_tail = agent.loop_gain[:, 12:]
        gains_main.uniform_(1e9, 1e10)
        gains_tail.uniform_(1e2, 1e5)

        # Get shapes for coefficient tensors
        batch_size, loop_n_frames, loop_order = agent.loop_coefficients.shape
        _, exc_n_frames, exc_order = agent.exc_coefficients.shape

        # Generate sinusoidal patterns for loop coefficients
        x_loop = torch.linspace(0, 2 * math.pi, loop_n_frames, device=agent.device)
        for o in range(loop_order):
            # Random frequency and phase for this order
            freq = torch.rand(1, device=agent.device).item() * 10  # Random freq between 0-10
            phase = torch.rand(1, device=agent.device).item() * 2 * math.pi

            # Generate sinusoid with these parameters and reshape correctly
            sinusoid = torch.sin(freq * x_loop + phase)
            sinusoid = (sinusoid * 0.5 - 0.25)

            # Reshape to match the slice we're assigning to
            agent.loop_coefficients[:, :, o] = sinusoid.reshape(1, loop_n_frames)

        # Generate sinusoidal patterns for excitation coefficients
        x_exc = torch.linspace(0, 2 * math.pi, exc_n_frames, device=agent.device)
        for o in range(exc_order):
            # Different random freq and phase for each excitation order
            freq = torch.rand(1, device=agent.device).item() * 10
            phase = torch.rand(1, device=agent.device).item() * 2 * math.pi

            # Generate sinusoid and reshape correctly
            sinusoid = torch.sin(freq * x_exc + phase)
            sinusoid = (sinusoid * 0.5 - 0.25)

            # Reshape to match the slice we're assigning to
            agent.exc_coefficients[:, :, o] = sinusoid.reshape(1, exc_n_frames)

def synth_one(agent: DiffKS, f0_hz: float) -> torch.Tensor:
    excitation = gen_burst()
    pitch = torch.tensor([[f0_hz]], device=agent.device)
    with torch.no_grad():
        audio = agent(f0_frames=pitch, input=excitation, input_sr=SR_OUT, direct=True)
    return audio

def params_to_dict(agent: DiffKS) -> dict:
    return {
        "loop_coefficients": agent.loop_coefficients.squeeze(0).tolist(),
        "loop_gain"       : agent.loop_gain.squeeze(0).tolist(),
        "exc_coefficients" : agent.exc_coefficients.squeeze(0).tolist(),
    }


def random_param_batch(agent: DiffKS, batch_size: int, generator: torch.Generator):
    rand_kwargs = {
        "dtype": agent.loop_coefficients.dtype,
        "device": agent.device,
        "generator": generator
    }

    rand = partial(torch.rand, **rand_kwargs)
    batch_size, loop_n_frames, loop_order = agent.loop_coefficients.shape
    _, exc_n_frames, exc_order = agent.exc_coefficients.shape
    exc_order = exc_order - 1

    def piecewise_linear(start, end, n_points):
        n_coeffs = start.shape[1]

        start = start[:, None, :]
        end = end[:, None, :]

        mid_x = n_points * (rand(*start.shape) * 0.5 + 0.25)
        mid_y = (start + end) / 2

        xs = torch.arange(n_points, device=agent.device)[None, :, None]

        curves = torch.zeros((batch_size, n_points, n_coeffs), device=agent.device)
        left_idxs = xs < mid_x
        right_idxs = xs >= mid_x

        # linear interpolation from the start to mid_x
        curves[left_idxs] = (xs * (mid_y - start) / mid_x + start)[left_idxs]
        # linear interpolation from mid_x to end
        curves[right_idxs] = ((xs - mid_x) * (end - mid_y) / (n_points - mid_x) + mid_y)[right_idxs]

        return curves

    pitch = rand(batch_size, 1) * (MIDI_E6 - MIDI_E2) + MIDI_E2
    pitch = midi_to_hz(pitch)

    gain_start = rand(batch_size, 1) * 5 + 5
    gain_end = rand(batch_size, 1) * 5 + 5

    gain_start = torch.pow(1e-3, 1 / (440 * gain_start))
    gain_end = torch.pow(1e-3, 1 / (440 * gain_end))

    loop_gain = piecewise_linear(gain_start, gain_end, loop_n_frames)

    loop_coeffs_start = rand(batch_size, loop_order) * 0.9 + 0.0999
    loop_coeffs_end = rand(batch_size, loop_order) * 0.9 + 0.0999
    loop_coeffs = piecewise_linear(loop_coeffs_start, loop_coeffs_end, loop_n_frames)

    loop_coeffs = loop_coeffs / torch.sum(loop_coeffs, dim=-1, keepdim=True)

    exc_coeffs_start = rand(batch_size, exc_order) * 0.9 + 0.0999
    exc_coeffs_end = rand(batch_size, exc_order) * 0.9 + 0.0999

    exc_coeffs = piecewise_linear(exc_coeffs_start, exc_coeffs_end, exc_n_frames)

    exc_gain_start = rand(batch_size, 1) * 5 + 5
    exc_gain_start = torch.pow(1e-3, 1 / (440 * exc_gain_start))

    exc_gain_end = rand(batch_size, 1) * 5 + 5
    exc_gain_end = torch.pow(1e-3, 1 / (440 * exc_gain_end))

    exc_gain = piecewise_linear(exc_gain_start, exc_gain_end, 1)

    exc_coeffs = exc_coeffs / torch.sum(exc_coeffs, dim=-1, keepdim=True) * exc_gain

    with torch.no_grad():
        audio = gen_burst(sr=SR_OUT, batch_size=batch_size, generator=generator)
        audio = agent(
            f0_frames=pitch,
            input=audio,
            input_sr=SR_OUT,
            direct=True,
            loop_coefficients=loop_coeffs,
            loop_gain=loop_gain,
            exc_coefficients=exc_coeffs,
            constrain_coefficients=False,
        )
        # audio = torchaudio.functional.resample(audio, KS_CFG['internal_sr'], SR_OUT)
        loud = a_weighted_loudness(audio).squeeze()

    pitch = pitch.expand(-1, loud.shape[1])

    return audio, pitch.unsqueeze(-1), loud.unsqueeze(-1), loop_coeffs * loop_gain, exc_coeffs

class SyntheticDataset(Dataset):
    """
    Loads the clips generated by make_param_dataset.py.
    Returns audio, pitch‑track (F0), loudness‑track.
    """
    def __init__(self, root: str):
        self.root = Path(root)
        self.meta = json.loads((self.root / "index.json").read_text())
        self.wav_dir = self.root / "wav"

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        m   = self.meta[idx]
        wav = self.wav_dir / f"{m['midi']:02d}.wav"
        audio, _ = torchaudio.load(wav)

        audio = torch.as_tensor(audio).squeeze()  # (N,)
        loud = torch.tensor(m["loud"]).view(-1, 1)  # (F, 1)
        pitch = torch.full((loud.shape[0], 1), m["freq"]) # (F, 1)

        return audio, pitch, loud

# ───────────────────────── main script ────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",  default=SYNTHETIC_DIR, type=str,
                    help="output folder")
    ap.add_argument("--seed", default=2025, type=int)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_root = Path(args.out)
    (out_root / "wav").mkdir(parents=True, exist_ok=True)

    midi_vals = np.linspace(MIDI_E2, MIDI_E6, N_NOTES, dtype=int)
    midi_vals = sorted(set(midi_vals))

    dev   = torch.device(get_device())
    agent = DiffKS(**KS_CFG, device=dev,).to(dev).eval()

    index: List[dict] = []

    from tqdm import tqdm
    for midi in tqdm(midi_vals, desc="Generating audio samples"):
        freq = float(midi_to_hz(torch.tensor([midi])))

        make_random_params(agent)
        audio = synth_one(agent, freq)

        stem = f"{midi:02d}"
        wav_path  = out_root / "wav" / f"{stem}.wav"
        json_path = out_root / f"{stem}.json"

        torchaudio.save(wav_path.as_posix(), audio.cpu(), SR_OUT)
        loud = a_weighted_loudness(audio).squeeze(0)

        meta = {"midi" : int(midi),
                "freq" : float(freq),
                "loud" : loud.tolist(),
                **params_to_dict(agent)}

        json_path.write_text(json.dumps(meta, indent=2))
        index.append(meta)

    loud_all = torch.cat([torch.tensor(m["loud"]) for m in index])
    mu, std = loud_all.mean().item(), loud_all.std().item()

    for m in index:
        loud = (torch.tensor(m["loud"]) - mu) / std
        m["loud"] = loud.tolist()
        (out_root / f"{m['midi']:02d}.json").write_text(json.dumps(m, indent=2))

    json.dump({"mean": mu, "std": std},
              open(out_root / "loudness_stats.json", "w"), indent=2)

    (out_root / "index.json").write_text(json.dumps(index, indent=2))
    print(f"✔  {len(index)} files written in {out_root}  (μ={mu:.4f}, σ={std:.4f})")

if __name__ == "__main__":
    main()
