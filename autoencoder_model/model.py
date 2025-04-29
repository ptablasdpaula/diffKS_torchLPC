import torch
import torch.nn as nn
import torchaudio.transforms as T

from utils.helpers import get_device
from third_party.ddsp_pytorch.ddsp.core import mlp, gru
from diffKS import DiffKS

from autoencoder_model.preprocess import E2_HZ

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
                 loop_n_frames,
                 exc_order,
                 exc_n_frames,
                 internal_sr,
                 interpolation_type,
                 z_encoder,
                 ):
        super().__init__()
        self.internal_sr = internal_sr
        self.loop_order = loop_order
        self.loop_n_frames = loop_n_frames
        self.exc_order = exc_order
        self.exc_n_frames = exc_n_frames

        self.z_encoder = z_encoder

        # Neural network components
        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3)] * 2)
        self.gru = gru(2, hidden_size)

        self.out_mlp = mlp(hidden_size + 2 + z_encoder.z_dims, hidden_size, 3)

        # Output projections
        self.loop_coeff_proj = nn.Linear(hidden_size, loop_n_frames * (loop_order + 1))
        self.exc_coeff_proj = nn.Linear(hidden_size, exc_n_frames * (exc_order + 1))
        self.loop_gain_proj = nn.Linear(hidden_size, loop_n_frames)

        # Create a buffer for GRU state
        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))

        # ----------  differentiable KS decoder  ----------
        self.decoder = DiffKS(
            batch_size = batch_size,
            internal_sr = internal_sr,
            loop_order = loop_order,
            loop_n_frames = loop_n_frames,
            exc_order = exc_order,
            exc_n_frames = exc_n_frames,
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

    def forward(self, pitch, loudness, audio, audio_sr):
        """
        Forward pass of the neural Karplus-Strong model.

        Args:
            pitch: Tensor of shape [batch_size, frames, 1] - MIDI pitch values
            loudness: Tensor of shape [batch_size, frames, 1] - Loudness values

        Returns:
            Tensor of shape [batch_size, n_samples] - Synthesized audio
        """
        z = self.z_encoder(audio, f0_scaled=pitch)

        # Process through network
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)

        hidden = torch.cat([self.gru(hidden)[0], pitch, loudness, z], -1)
        hidden = self.out_mlp(hidden)

        hidden_avg = hidden.mean(dim=1, keepdim=True)

        # Get raw outputs
        batch_size = hidden.shape[0]
        loop_coeff_flat = self.loop_coeff_proj(hidden_avg)
        exc_coeff_flat = self.exc_coeff_proj(hidden_avg)
        loop_gain_flat = self.loop_gain_proj(hidden_avg)

        # Reshape outputs to match expected dimensions
        loop_coefficients = loop_coeff_flat.reshape(
            batch_size,
            self.loop_n_frames,
            self.loop_order + 1
        )

        exc_coefficients = exc_coeff_flat.reshape(
            batch_size,
            self.exc_n_frames,
            self.exc_order + 1
        )

        loop_gain = loop_gain_flat.reshape(
            batch_size,
            self.loop_n_frames,
            1
        )

        # Handle batch dimension for DiffKS
        outputs = []

        # Run DiffKS
        out = self.decoder(f0_frames= self.internal_sr / pitch.squeeze(2),
                           input=audio,
                           input_sr=audio_sr,
                           loop_coefficients=loop_coefficients,
                           loop_gain=loop_gain,
                           exc_coefficients=exc_coefficients,)

        return out