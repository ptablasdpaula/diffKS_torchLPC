import torch
import torch.nn as nn

from ddsp_pytorch.ddsp.core import mlp, gru
from diffKS import DiffKS, noise_burst


class NN_KarplusModel(nn.Module):
    def __init__(self,
                 hidden_size,
                 loop_order,
                 loop_n_frames,
                 exc_order,
                 exc_n_frames,
                 target_for_inverse=None,
                 sampling_rate=16000,
                 n_samples=16000 * 4,
                 min_f0_hz=20.0,
                 ):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))

        # Store these for later use
        self.loop_order = loop_order
        self.loop_n_frames = loop_n_frames
        self.exc_order = exc_order
        self.exc_n_frames = exc_n_frames
        self.n_samples = n_samples
        self.target_for_inverse = target_for_inverse
        self.min_f0_hz = min_f0_hz

        # Neural network components
        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3)] * 2)
        self.gru = gru(2, hidden_size)
        self.out_mlp = mlp(hidden_size + 2, hidden_size, 3)

        # Output projections
        self.loop_coeff_proj = nn.Linear(hidden_size, loop_n_frames * (loop_order + 1))
        self.exc_coeff_proj = nn.Linear(hidden_size, exc_n_frames * exc_order)
        self.loop_gain_proj = nn.Linear(hidden_size, loop_n_frames)

        # Create a buffer for GRU state
        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))

        # Create the excitation burst
        burst = noise_burst(sample_rate=sampling_rate,
                            length_s=4,
                            burst_width_s=0.02,
                            return_1D=True,
                            normalize=True)
        self.register_buffer("burst", burst)

    def forward(self, pitch, loudness):
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

        # Get raw outputs
        batch_size = hidden.shape[0]
        loop_coeff_flat = self.loop_coeff_proj(hidden)
        exc_coeff_flat = self.exc_coeff_proj(hidden)
        loop_gain_flat = self.loop_gain_proj(hidden)

        # Reshape outputs to match expected dimensions
        loop_coefficients = loop_coeff_flat.reshape(
            batch_size,
            self.loop_n_frames,
            self.loop_order + 1
        )

        exc_coefficients = exc_coeff_flat.reshape(
            batch_size,
            self.exc_n_frames,
            self.exc_order
        )

        loop_gain = loop_gain_flat.reshape(batch_size, self.loop_n_frames, 1)

        # Handle batch dimension for DiffKS
        outputs = []

        # Convert pitch to frequency (Hz)
        f0 = (torch.Tensor(self.sampling_rate / 159.6))

        # Create DiffKS model for this example
        diffKS = DiffKS(
            burst=self.burst,
            loop_n_frames=self.loop_n_frames,
            sample_rate=self.sampling_rate,
            min_f0_hz=self.min_f0_hz,
            loop_order=self.loop_order,
            exc_n_frames=self.exc_n_frames,
            exc_order=self.exc_order,
            interp_type='linear',
        )

        # Run DiffKS
        out = diffKS(
            f0_frames=f0,
            n_samples=self.n_samples,
            target=self.target_for_inverset,
            loop_coefficients=loop_coefficients,
            loop_gain=loop_gain,
            exc_coefficients=exc_coefficients,
        )

        return out