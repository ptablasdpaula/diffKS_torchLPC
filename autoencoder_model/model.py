import torch
import torch.nn as nn

from third_party.ddsp_pytorch.ddsp.core import mlp, gru
from diffKS import DiffKS

class AE_KarplusModel(nn.Module):
    def __init__(self,
                 hidden_size,
                 batch_size,
                 loop_order,
                 loop_n_frames,
                 exc_order,
                 exc_n_frames,
                 sample_rate,
                 ):
        super().__init__()
        self.sample_rate = sample_rate
        self.loop_order = loop_order
        self.loop_n_frames = loop_n_frames
        self.exc_order = exc_order
        self.exc_n_frames = exc_n_frames

        # Neural network components
        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3)] * 2)
        self.gru = gru(2, hidden_size)
        self.out_mlp = mlp(hidden_size + 2, hidden_size, 3)

        # Output projections
        self.loop_coeff_proj = nn.Linear(hidden_size, loop_n_frames * (loop_order + 1))
        self.exc_coeff_proj = nn.Linear(hidden_size, exc_n_frames * (exc_order + 1))
        self.loop_gain_proj = nn.Linear(hidden_size, loop_n_frames)

        # Create a buffer for GRU state
        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))

        # ----------  differentiable KS decoder  ----------
        self.decoder = DiffKS(
            batch_size = batch_size,  # will be expanded inside forward
            sample_rate = sample_rate,
            loop_order = loop_order,
            loop_n_frames = loop_n_frames,
            exc_order = exc_order,
            exc_n_frames = exc_n_frames,
            interp_type = 'lagrange',
            use_double_precision = False,
        )

        for p in self.decoder.parameters():
            p.requires_grad = False

    def get_hidden(self, pitch, loudness):
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)
        hidden = torch.cat([self.gru(hidden)[0], pitch, loudness], -1)
        hidden = self.out_mlp(hidden)
        return hidden.mean(dim=1, keepdim=True)  # Assuming you're using mean pooling

    def forward(self, pitch, loudness, input):
        """
        Forward pass of the neural Karplus-Strong model.

        Args:
            pitch: Tensor of shape [batch_size, frames, 1] - MIDI pitch values
            loudness: Tensor of shape [batch_size, frames, 1] - Loudness values

        Returns:
            Tensor of shape [batch_size, n_samples] - Synthesized audio
        """
        # Process through network
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)
        hidden = torch.cat([self.gru(hidden)[0], pitch, loudness], -1)
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
        out = self.decoder(f0_frames= self.sample_rate / pitch.squeeze(2),
                           input=input,
                           loop_coefficients=loop_coefficients,
                           loop_gain=loop_gain,
                           exc_coefficients=exc_coefficients,)

        return out