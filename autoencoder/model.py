import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

from utils import get_device
from diffKS import DiffKS

from data.preprocess import E2_HZ

from ddc_onset.spectral import SpectrogramExtractor
from ddc_onset.cnn      import SpectrogramNormalizer, PlacementCNN
from ddc_onset.constants import FRAME_RATE, Difficulty
from ddc_onset import find_peaks, threshold_peaks
import torch.nn.functional as F
import numpy as np

def mlp(in_size, hidden_size, n_layers):
    channels = [in_size] + (n_layers) * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)

def gru(n_input, hidden_size):
    return nn.GRU(n_input * hidden_size, hidden_size, batch_first=True)

class ZEncoder(nn.Module):
    def __init__(self, input_keys=None):
        super().__init__()

    def forward(self, audio, f0_scaled=None):
        """Forward pass computing the z embedding."""
        z = self.compute_z(audio)
        if f0_scaled is not None:
            time_steps = f0_scaled.shape[1]
            z = self.expand_z(z, time_steps)

        return z

    def expand_z(self, z, time_steps):
        """Ensure z has same temporal resolution as other conditioning."""
        if len(z.shape) == 2:
            z = z.unsqueeze(1)

        z_time_steps = z.shape[1]

        if z_time_steps != time_steps:
            z = z.transpose(1, 2)  # [batch, channels, time]
            z = torch.nn.functional.interpolate(
                z,
                size=time_steps,
                mode='linear',
                align_corners=False
            )
            z = z.transpose(1, 2)  # [batch, time, channels]

        return z

    def compute_z(self, audio):
        """Takes audio tensor and returns latent tensor z."""
        raise NotImplementedError

class MfccTimeDistributedRnnEncoder(ZEncoder):
    """MFCC-based encoder with RNN processing."""
    def __init__(self,
                 rnn_channels=512,
                 rnn_type='gru',
                 z_dims=16,
                 z_time_steps=250,
                 sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.z_dims = z_dims
        self.z_time_steps = z_time_steps

        # Configure based on z_time_steps as in the original implementation
        if z_time_steps == 63:
            self.fft_size = 2048
            self.overlap = 0.5
        elif z_time_steps == 125:
            self.fft_size = 1024
            self.overlap = 0.5
        elif z_time_steps == 250:
            self.fft_size = 1024
            self.overlap = 0.75
        elif z_time_steps == 500:
            self.fft_size = 512
            self.overlap = 0.75
        elif z_time_steps == 1000:
            self.fft_size = 256
            self.overlap = 0.75
        else:
            raise ValueError(
                '`z_time_steps` currently limited to 63, 125, 250, 500 and 1000')

        self.hop_length = int(self.fft_size * (1.0 - self.overlap))

        # MFCC extraction
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=30,
            melkwargs={
                'n_mels': 128,
                'f_min': 20.0,
                'f_max': 8000.0,
                'n_fft': self.fft_size,
                'hop_length': self.hop_length,
                'pad_mode': 'reflect'
            }
        )

        # Normalization layer
        self.z_norm = nn.InstanceNorm1d(30)

        # RNN and output layers
        if rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(30, rnn_channels, batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

        self.dense_out = nn.Linear(rnn_channels, z_dims)

    def compute_z(self, audio):
        """Compute z embedding from audio."""
        # Extract MFCCs
        mfccs = self.mfcc_transform(audio).transpose(1, 2)  # [batch, time, n_mfcc]

        # Normalize
        mfccs = mfccs.transpose(1, 2)  # [batch, n_mfcc, time]
        mfccs = self.z_norm(mfccs)
        mfccs = mfccs.transpose(1, 2)  # [batch, time, n_mfcc]

        # Run RNN
        rnn_out, _ = self.rnn(mfccs)

        # Dense projection
        z = self.dense_out(rnn_out)

        return z

class AE_KarplusModel(nn.Module):
    def __init__(self,
                 hidden_size,
                 batch_size,
                 loop_order,
                 exc_order,
                 internal_sr,
                 interpolation_type,
                 z_encoder,
                 ):
        super().__init__()
        self.internal_sr = internal_sr
        self.loop_order = loop_order
        self.exc_order = exc_order

        self.z_encoder = z_encoder

        # Neural network components
        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3)] * 2)
        self.gru = gru(2, hidden_size)

        self.out_mlp = mlp(hidden_size + 2 + z_encoder.z_dims, hidden_size, 3)

        # Output projections
        self.loop_coeff_proj = nn.Linear(hidden_size, loop_order + 1)  # B, T, *
        self.exc_coeff_proj = nn.Linear(hidden_size, exc_order + 1)
        self.loop_gain_proj = nn.Linear(hidden_size, 1)  # scalar per frame

        # Create a buffer for GRU state
        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))

        # ONSET DETECTOR CONFIG
        self.onset_extract = SpectrogramExtractor()
        self.onset_norm = SpectrogramNormalizer()
        self.onset_cnn = PlacementCNN()

        for m in (self.onset_extract, self.onset_norm, self.onset_cnn):
            m.eval()                    # Start in Inference mode
            for p in m.parameters():
                p.requires_grad_(False) # Initialize Frozen!

        self.pad_left   = 512
        self.onset_unfrozen = False
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

        # ----------  differentiable KS decoder  ----------
        self.decoder = DiffKS(
            batch_size = batch_size,
            internal_sr = internal_sr,
            loop_order = loop_order,
            exc_order = exc_order,
            interp_type = interpolation_type, # Only linear remains stable for NNs
            use_double_precision = True if get_device() != torch.device('mps') else False,
            min_f0_hz= E2_HZ - 10,
        )

        for p in self.decoder.parameters():
            p.requires_grad = False

    def get_hidden(self, pitch, loudness, audio):
        z = self.z_encoder(audio, f0_scaled=pitch)

        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)
        hidden = torch.cat([self.gru(hidden)[0], pitch, loudness, z], -1)
        hidden = self.out_mlp(hidden)
        return hidden.mean(dim=1, keepdim=True)  # Assuming you're using mean pooling

    # -------------- helper: compute triggers --------------------------------
    def _make_triggers(self, audio: torch.Tensor, audio_sr: int,
                       thresh: float = 0.15) -> torch.Tensor:
        B, _ = audio.shape
        ratio = int(self.internal_sr // FRAME_RATE)  # 441

        audio_pad = F.pad(audio, (self.pad_left, 0))
        audio_44k = torchaudio.functional.resample(audio_pad, audio_sr,
                                                   self.internal_sr)

        out = []
        for b in range(B):
            spec = self.onset_extract(audio_44k[b:b + 1])
            sal = self.onset_cnn(
                self.onset_norm(spec),
                torch.tensor([Difficulty.CHALLENGE.value],
                             device=audio.device, dtype=torch.int64)
            )[0]
            peaks = threshold_peaks(sal.cpu().numpy(),
                                    find_peaks(sal.cpu().numpy()), thresh)
            if len(peaks) == 0:
                peaks = [0]

            trig = torch.tensor(peaks, device=audio.device) * ratio
            trig = trig - self.pad_left
            trig = trig.clamp(min=0)
            out.append(trig.long())

        return self._pad_list(out, pad_value=0)  # [B, Fmax]

    @staticmethod
    def _pad_list(list_of_tensors: list[torch.Tensor],
                  pad_value: float = 0.0) -> torch.Tensor:
        """
        Turns a Python list of [Fᵢ, …] tensors into one padded
        tensor [B, Fmax, …] using right‑padding with *pad_value*.
        """
        Fmax = max(t.size(0) for t in list_of_tensors)
        padded = []
        for t in list_of_tensors:
            if t.size(0) < Fmax:
                pad = t.new_full((Fmax - t.size(0), *t.shape[1:]), pad_value)
                t = torch.cat([t, pad], dim=0)
            padded.append(t)
        return torch.stack(padded, dim=0)

    def forward(
            self,
            pitch,
            loudness,
            audio,
            audio_sr,
            unfreeze_onset_after: int = 0,
            return_parameters=False):
        """
        Forward pass of the neural Karplus-Strong model.

        Args:
            pitch: Tensor of shape [batch_size, frames, 1] - pitch values (f0)
            loudness: Tensor of shape [batch_size, frames, 1] - Loudness values

        Returns:
            Tensor of shape [batch_size, n_samples] - Synthesized audio
        """
        if (not self.onset_unfrozen) and (self.step.item() >= unfreeze_onset_after):
            for m in (self.onset_extract, self.onset_norm, self.onset_cnn):
                m.train()
                for p in m.parameters():
                    p.requires_grad_(True)
            self.onset_unfrozen = True

        # ─── 1.  build the full‑resolution hidden sequence ───────────────────────
        z = self.z_encoder(audio, f0_scaled=pitch)  # [B, T, z_dim]
        hidden = torch.cat([self.in_mlps[0](pitch),
                            self.in_mlps[1](loudness)], -1)  # [B, T, 2×H]
        hidden = torch.cat([self.gru(hidden)[0], pitch, loudness, z], -1)
        hidden = self.out_mlp(hidden)  # [B, T, H]

        # ─── 2.  predict a coefficient frame *per* encoder step ──────────────────
        loop_coeff_all = self.loop_coeff_proj(hidden)  # [B, T, loop_order+1]
        exc_coeff_all = self.exc_coeff_proj(hidden)  # [B, T, exc_order +1]
        loop_gain_all = self.loop_gain_proj(hidden)  # [B, T, 1]

        # ─── 3.  detect triggers (padded) and map to frame indices ───────────────
        triggers = self._make_triggers(audio, audio_sr)  # [B, Fmax]
        B, Fmax = triggers.shape
        T = pitch.size(1)  # encoder frames per clip
        n_samples = audio.size(1)

        # sample_idx → frame_idx  (integer rounding)
        frame_idx = (triggers.float() / n_samples * T).long().clamp(max=T - 1)

        # ─── 4.  gather the corresponding coefficient frames ─────────────────────
        def batch_gather(src, idx):  # src: [B, T, D]  idx: [B, F]
            B, F = idx.shape
            D = src.size(-1)
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)  # [B, F, D]
            return src.gather(1, idx_exp)  # [B, F, D]

        loop_coeff_sel = batch_gather(loop_coeff_all, frame_idx)
        exc_coeff_sel = batch_gather(exc_coeff_all, frame_idx)
        loop_gain_sel = batch_gather(loop_gain_all, frame_idx)

        if return_parameters:
            return self.decoder.get_constrained_l_coefficients(loop_coeff_sel, loop_gain_sel), self.decoder.get_constrained_exc_coefficients(exc_coeff_sel)

        # ─── 5.  call DiffKS with the *selected* frames + triggers ───────────────
        out = self.decoder(
            f0_frames=pitch.squeeze(2),
            input=audio,
            input_sr=audio_sr,
            loop_coefficients=loop_coeff_sel,
            loop_gain=loop_gain_sel,
            exc_coefficients=exc_coeff_sel,
            triggers=triggers,
        )


        with torch.no_grad():
            self.step += 1

        return out