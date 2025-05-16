from __future__ import annotations
from functools import partial
import torch
from diffKS import DiffKS
from data.preprocess import E2_HZ
from utils import get_device, noise_burst, midi_to_hz

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

        curves = torch.zeros(
            (batch_size, n_points, n_coeffs),
            device=agent.device,
        dtype=start.dtype,
        )

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
        loud = a_weighted_loudness(audio).squeeze(1)

    pitch = pitch.expand(-1, loud.shape[1])

    return audio, pitch.unsqueeze(-1), loud.unsqueeze(-1), loop_coeffs * loop_gain, exc_coeffs

class OnTheFlySynth(torch.utils.data.IterableDataset):
    """
    Generates `num_batches` random DiffKS examples.

    If `batch_size` is omitted, we use `diffks.batch_size`
    (so you don’t have to pass it twice).
    """
    def __init__(
        self,
        diffks: DiffKS,
        *,
        num_batches: int,
        batch_size: int | None = None,
        seed: int = 42,
    ):
        self.diffks = diffks
        self.batch_size = batch_size if batch_size is not None else diffks.batch_size
        self.num_batches = num_batches
        self.gen = torch.Generator(device=get_device()).manual_seed(seed)

    def __iter__(self):
        for _ in range(self.num_batches):
            yield random_param_batch(self.diffks, self.batch_size, generator=self.gen)
