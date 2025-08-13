import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np

from utils.ml import mlp, gru
from diffKS import DiffKS
from data.preprocess import E2_HZ
from core import make_onset_noise, detect_onsets_librosa, StaticShelf, DualShelfController

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
                 internal_sr,
                 interpolation_type,
                 z_encoder,
                 loudness_mu=None, loudness_std=None
                 ):
        super().__init__()
        self.internal_sr = internal_sr
        self.loop_order = loop_order
        self.z_encoder = z_encoder

        # Three separate 3‑layer MLPs as in the original DDSP decoder
        self.mlp_f = mlp(1, hidden_size, 3)          # f(t)
        self.mlp_l = mlp(1, hidden_size, 3)          # l(t)
        self.mlp_z = mlp(z_encoder.z_dims, hidden_size, 3)  # z(t)
        self.gru = gru(3, hidden_size)

        # Concatenate GRU output with f‑ and l‑MLP outputs (z is **not** appended, per paper)
        self.out_mlp = mlp(hidden_size * 3, hidden_size, 3)

        # Heads
        self.shelf_head = nn.Linear(hidden_size, 6) # shelves: [low_fc, low_Q, low_g, high_fc, high_Q, high_g]
        self.coefficients_head = nn.Linear(hidden_size, loop_order + 1)  # loop filter taps + 1 (gain)
        # Initialize coefficients_head bias and weight so that gain=1, others=0
        #with torch.no_grad():
        #    self.coefficients_head.weight.zero_()
        #    self.coefficients_head.bias.zero_()
        #    self.coefficients_head.bias[0] = 2.197

        # Store loudness z-score stats
        mu = 0.0 if loudness_mu is None else float(loudness_mu)
        sd = 1.0 if loudness_std is None else float(loudness_std)
        self.register_buffer("loudness_mu", torch.tensor(mu))
        self.register_buffer("loudness_std", torch.tensor(sd))

        # Excitation shaping: learnable shelves
        self.sample_rate = 16000  # set to your dataset SR
        self.low_shelf = StaticShelf(
            which="low",
            sample_rate=self.sample_rate,
            init_fc_hz=120.0,
            fmin_hz=20.0,
            fmax_hz=self.sample_rate / 2 - 200.0,
            init_Q=0.707,
            init_gain_db=-3.0,
        )
        self.high_shelf = StaticShelf(
            which="high",
            sample_rate=self.sample_rate,
            init_fc_hz=3000.0,
            fmin_hz=30.0,
            fmax_hz=self.sample_rate / 2 - 200.0,
            init_Q=0.707,
            init_gain_db=-1.5,
        )
        # Controller that maps AE shelf logits → physically valid shelf params
        self.shelf_ctrl = DualShelfController(
            fs=self.sample_rate,
            fmin=20.0,
            fmax=self.sample_rate / 2 - 200.0,
            max_gain_db=12.0,
        )

        # Create a buffer for GRU state
        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))

        # ----------  differentiable KS decoder  ----------
        self.decoder = DiffKS(
            batch_size = batch_size,
            internal_sr = internal_sr,
            loop_order = loop_order,
            loop_n_frames = z_encoder.z_time_steps if hasattr(z_encoder, 'z_time_steps') else 250,
            interp_type = interpolation_type,
            use_double_precision = True,
            min_f0_hz = E2_HZ - 10,
        )

        # Decoder parameters are frozen, since it obtains its values from the autoencoder
        for p in self.decoder.parameters():
            p.requires_grad = False

    def forward(
            self,
            pitch,
            loudness,
            audio,
            audio_sr,
            return_parameters=False):
        """
        Forward pass of the neural Karplus-Strong model.

        Args:
            pitch: Tensor of shape [batch_size, frames, 1] - pitch values (f0)
            loudness: Tensor of shape [batch_size, frames, 1] - Loudness values

        Returns:
            Tensor of shape [batch_size, n_samples] - Synthesized audio
        """
        # ─── build the full‑resolution hidden sequence ───────────────────────
        z = self.z_encoder(audio, f0_scaled=pitch)  # [B, T, z_dim]
        x_f = self.mlp_f(pitch)                      # [B, T, H]
        x_l = self.mlp_l(loudness)                   # [B, T, H]
        x_z = self.mlp_z(z)                          # [B, T, H]

        gru_in  = torch.cat([x_f, x_l, x_z], -1)     # [B, T, 3H]
        gru_out = self.gru(gru_in)[0]               # [B, T, H]

        hidden  = torch.cat([gru_out, x_f, x_l], -1) # [B, T, 3H]
        hidden  = self.out_mlp(hidden)               # [B, T, H]

        # --- Predict shelf parameters from hidden; pool across time to make them static per example
        h_pool = hidden.mean(dim=1)               # [B, H]
        shelf_raw = self.shelf_head(h_pool)       # [B, 6] = [l_fc_r, l_Q_r, l_g_r, h_fc_r, h_Q_r, h_g_r]
        # Map to ordered & constrained params (f_high > f_low, Q>0, gains bounded)
        (l_fc, l_Q, l_gdb), (h_fc, h_Q, h_gdb) = self.shelf_ctrl(shelf_raw)

        # ─── 2.  predict a coefficient frame *per* encoder step ──────────────────
        coefficients_raw = self.coefficients_head(hidden)              # [B, T, L+1]

        # ─── 3. Onset-based excitation (DiffKS) ────────────────────────────────
        B, N = audio.shape
        T = pitch.size(1)  # number of pitch frames

        # De-normalize incoming A-weighted log-power loudness (z-scored) back to log-power
        logpow_all = loudness.squeeze(-1) * self.loudness_std + self.loudness_mu  # [B, T]

        # (No inversion) — detect onsets directly on the target audio
        # We intentionally skip any inverse-filtering path and let KS handle amplitude
        # shaping. Shelves are applied later to the excitation only.
        onset_list = []
        for b in range(B):
            on_b = detect_onsets_librosa(
                audio[b], sr=audio_sr, pad_ms=50.0, hop_length=512, backtrack=True
            )
            if on_b.size == 0:
                on_b = np.array([0], dtype=int)
            onset_list.append(on_b)

        # Helper: map sample index → pitch frame index
        def sample_to_frame_idx(s: int) -> int:
            # linear mapping with clamp
            idx = int(round(s * T / max(1, N)))
            return max(0, min(T - 1, idx))

        # Build bursts per item with per-onset length derived from local pitch,
        # apply per-onset RMS scaling to match the target audio for each burst
        raw_noise_rows = []
        exc_rows = []
        for b in range(B):
            raw_b = torch.zeros(1, N, device=audio.device, dtype=audio.dtype)
            exc_b = torch.zeros(1, N, device=audio.device, dtype=audio.dtype)
            for s in onset_list[b]:
                s = int(s)
                f_idx = sample_to_frame_idx(s)
                f0_loc = float(torch.clamp(pitch[b, f_idx, 0], min=E2_HZ).item())
                L_loc = int(round(float(audio_sr) / max(f0_loc, 1e-6)))

                # Generate a single-burst noise at this onset (unscaled reference)
                nb = make_onset_noise(
                    onset_samples=np.array([s], dtype=int),
                    num_samples=N,
                    sample_rate=audio_sr,
                    batch_size=1,
                    device=audio.device,
                    dtype=audio.dtype,
                    burst_len_samples=L_loc,
                )  # [1, N]
                raw_b = raw_b + nb

                # Use de-normalized A-weighted log-power loudness at this pitch frame
                logpow = logpow_all[b, f_idx]
                A_rms = torch.exp(logpow * 0.5).view(1, 1) * 10 # [1,1]

                burst_start = s
                burst_end   = min(burst_start + L_loc, N)
                seg_noi     = nb[:, burst_start:burst_end]  # [1, L]

                # Current burst RMS (avoid divide-by-zero)
                seg_rms = torch.sqrt(torch.clamp((seg_noi ** 2).mean(dim=-1, keepdim=True), min=1e-12))  # [1,1]
                gain = A_rms / (seg_rms + 1e-12)

                nb_scaled = nb.clone()
                nb_scaled[:, burst_start:burst_end] = seg_noi * gain
                exc_b = exc_b + nb_scaled
            raw_noise_rows.append(raw_b)
            exc_rows.append(exc_b)

        raw_noise = torch.cat(raw_noise_rows, dim=0)  # [B, N]
        excitation = torch.cat(exc_rows, dim=0)       # [B, N]

        # Shelf shaping (learnable); AE can override via raw heads elsewhere
        excitation = self.low_shelf(
            excitation,
            fc_hz=l_fc, Q=l_Q, gain_db=l_gdb,
            from_raw=False,
        )
        excitation = self.high_shelf(
            excitation,
            fc_hz=h_fc, Q=h_Q, gain_db=h_gdb,
            from_raw=False,
        )

        if return_parameters:
            loop_coeffs_c = self.decoder.get_constrained_l_coefficients(f0=pitch.squeeze(-1), l_b=coefficients_raw)
            low_params  = {"fc_hz": l_fc,  "Q": l_Q,  "gain_db": l_gdb}
            high_params = {"fc_hz": h_fc, "Q": h_Q, "gain_db": h_gdb}
            return loop_coeffs_c, low_params, high_params

        out = self.decoder(
            f0_frames=pitch.squeeze(2),
            input=excitation,
            input_sr=audio_sr,
            loop_coefficients=coefficients_raw,
        )

        return out