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
from typing import List

class TriggerMLP(nn.Module):
    """
    Minimal learned refinement module that maps an onset salience timeline
    [B, T] -> trigger logits [B, T]. Pure ML; no hand‑crafted features.
    Implemented as a tiny 1x1 Conv "MLP" with ReLU, optional fixed moving‑average
    smoothing for local context.
    """
    def __init__(self, hidden: int = 64, context: int = 5):
        super().__init__()
        self.context = context
        self.fc1 = nn.Conv1d(1, hidden, kernel_size=1)
        self.act = nn.ReLU()
        self.fc2 = nn.Conv1d(hidden, 1, kernel_size=1)
        if context > 1:
            self.smooth = nn.Conv1d(
                1, 1, kernel_size=context, padding=context // 2,
                bias=False, groups=1
            )
            with torch.no_grad():
                self.smooth.weight.fill_(1.0 / context)
            for p in self.smooth.parameters():
                p.requires_grad_(False)  # fixed averaging kernel
        else:
            self.smooth = None

    def forward(self, salience: torch.Tensor) -> torch.Tensor:
        if salience.dim() != 2:
            raise ValueError(f"TriggerMLP expected [B,T], got {salience.shape}")
        # per‑item z‑score
        mean = salience.mean(dim=1, keepdim=True)
        std = salience.std(dim=1, keepdim=True).clamp_min_(1e-6)
        x = (salience - mean) / std
        x = x.unsqueeze(1)  # [B,1,T]
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)     # [B,1,T]
        x = x.squeeze(1)    # [B,T]
        if self.smooth is not None:
            x = self.smooth(x.unsqueeze(1)).squeeze(1)
        return x

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
                 trigger_topk: int = 16,
                 trigger_temp: float = 0.25,
                 trigger_hidden: int = 64,
                 trigger_context: int = 5,
                 trigger_rel_thresh: float = 0.5,
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

        # ---- learned trigger refinement ------------------------------------
        self.trigger_topk = trigger_topk
        self.trigger_temp = trigger_temp
        self.trigger_rel_thresh = trigger_rel_thresh
        self.trigger_mlp = TriggerMLP(hidden=trigger_hidden, context=trigger_context)
        self.register_buffer("last_trigger_probs", torch.zeros(1, 1), persistent=False)
        self.register_buffer("last_trigger_times_s", torch.zeros(1, 1), persistent=False)
        self.register_buffer("last_trigger_counts", torch.zeros(1, dtype=torch.long), persistent=False)

        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

        # ----------  differentiable KS decoder  ----------
        self.decoder = DiffKS(
            batch_size = batch_size,
            internal_sr = internal_sr,
            loop_order = loop_order,
            exc_order = exc_order,
            interp_type = interpolation_type,  # Only linear remains stable for NNs
            use_double_precision = True if get_device() != torch.device('mps') else False,
            min_f0_hz = E2_HZ - 10,
            upsample_mode = "soft",       # <<< differentiable instead of hard ZOH
            soft_zoh_tau = 5.0,           # edge softness in *samples* @ internal_sr
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

    def _make_triggers(self,
                       audio: torch.Tensor,
                       audio_sr: int) -> torch.Tensor:
        """
        Learned trigger path ONLY (legacy peak-pick removed).
        Runs onset CNN -> TriggerMLP -> per-item probability mask -> capped top-K.
        We DO NOT force every item to emit exactly K events: instead we keep all
        frames whose probability exceeds (max_prob * self.trigger_rel_thresh) per item,
        fall back to the single best frame if none pass, and cap at self.trigger_topk.
        Returned tensor is right-padded (repeat last) to [B, Fmax] for downstream code.
        """
        B, _ = audio.shape
        ratio = int(self.internal_sr // FRAME_RATE)  # samples per frame @ internal_sr

        # left-pad then resample (keep in graph)
        audio_pad = F.pad(audio, (self.pad_left, 0))
        audio_44k = torchaudio.functional.resample(audio_pad, audio_sr, self.internal_sr)

        pad_left_internal = int(round(self.pad_left * self.internal_sr / audio_sr))

        sal_batch = []
        for b in range(B):
            spec = self.onset_extract(audio_44k[b:b + 1])  # [1,C,F,T]
            spec_n = self.onset_norm(spec[0])              # [C,F,T]
            sal = self.onset_cnn(
                spec_n,
                torch.tensor([Difficulty.CHALLENGE.value],
                             device=audio.device, dtype=torch.int64)
            )[0]  # [T]
            sal_batch.append(sal)

        # pad sequences in-batch by repeating last element
        maxT = max(s.shape[0] for s in sal_batch)
        if maxT == 0:
            raise RuntimeError("onset_cnn produced empty salience.")
        sal_stack = []
        for s in sal_batch:
            if s.shape[0] < maxT:
                s = torch.cat([s, s[-1:].expand(maxT - s.shape[0])], dim=0)
            sal_stack.append(s)
        sal = torch.stack(sal_stack, dim=0)  # [B,T]

        # cache raw salience for logging
        self.last_trigger_probs = sal.detach()

        # learned refinement MLP
        logits = self.trigger_mlp(sal)  # [B,T]
        probs = torch.sigmoid(logits / self.trigger_temp)  # [B,T]

        # ----- per-item threshold & cap -----------------------------------
        trig_list = []
        for b in range(B):
            pb = probs[b]
            # relative mask wrt per-item max
            max_pb = pb.max()
            if max_pb <= 0:
                keep_mask = torch.zeros_like(pb, dtype=torch.bool)
            else:
                keep_mask = pb >= (max_pb * self.trigger_rel_thresh)

            idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)

            # fallback: at least 1 trigger (argmax)
            if idx.numel() == 0:
                idx = pb.argmax(dim=0, keepdim=True)

            # cap at topk if too many pass threshold
            if idx.numel() > self.trigger_topk:
                top_idx = torch.topk(pb, k=self.trigger_topk, dim=0).indices
                idx = torch.sort(top_idx).values
            else:
                idx = torch.sort(idx).values

            # map frame idx -> internal_sr samples
            ts = idx.to(audio.device, dtype=torch.long) * ratio
            ts = ts - pad_left_internal
            ts = ts.clamp(min=0)
            trig_list.append(ts)

        # record lengths BEFORE padding
        trig_lengths = torch.tensor([t.numel() for t in trig_list],
                                    device=audio.device, dtype=torch.long)
        # right-pad to common length by repeating last (uses helper)
        trig_padded = self._pad_list(trig_list)  # [B, Fmax]
        # cache lengths for logging
        self.last_trigger_counts = trig_lengths
        return trig_padded.long()

    @staticmethod
    def _pad_list(list_of_tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Right-pad per-item tensors by **repeating the last element**.
        This preserves non-decreasing trigger timelines and prevents
        negative segment lengths downstream in DiffKS._upsample_by_triggers().
        Supports 1-D [F] or 2-D [F, D] tensors.
        """
        if len(list_of_tensors) == 0:
            raise ValueError("_pad_list received empty list")
        Fmax = max(t.size(0) for t in list_of_tensors)
        out = []
        for t in list_of_tensors:
            pad = Fmax - t.size(0)
            if pad > 0:
                last = t[-1:].expand(pad, *t.shape[1:])
                t = torch.cat([t, last], dim=0)
            out.append(t)
        return torch.stack(out, dim=0)

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

        # NOTE: use learned trigger path unless explicitly disabled.
        triggers = self._make_triggers(audio, audio_sr)
        # cache last trigger times in seconds for logging
        self.last_trigger_times_s = triggers.float() / self.internal_sr

        # ─── 2.  predict a coefficient frame *per* encoder step ──────────────────
        loop_coeff_all = self.loop_coeff_proj(hidden)  # [B, T, loop_order+1]
        exc_coeff_all = self.exc_coeff_proj(hidden)  # [B, T, exc_order +1]
        loop_gain_all = self.loop_gain_proj(hidden)  # [B, T, 1]

        B, Fmax = triggers.shape
        T = pitch.size(1)  # encoder frames per clip

        # duration in seconds at input sample rate
        dur_s = audio.size(1) / audio_sr
        # triggers are in internal_sr samples -> seconds
        trig_sec = triggers.float() / self.internal_sr  # [B, F]
        # proportion of clip -> encoder frame index
        frame_idx = (trig_sec / dur_s * T).long().clamp(min=0, max=T - 1)

        # NOTE: frame_idx is clamped to [0, T-1] so gather is safe.
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