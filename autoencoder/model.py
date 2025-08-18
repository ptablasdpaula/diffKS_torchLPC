import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import numpy as np

from utils.ml import mlp, gru
from diffKS import DiffKS
from data.preprocess import E2_HZ
from core import make_onset_noise, detect_onsets_librosa
from flamo.auxiliary.eq import eq_freqs, geq as geq_sos

import math

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
                 z_encoder):
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
        # GEQ (graphic EQ) gains head – size depends on chosen band layout
        # Build GEQ band layout (one‑octave bands by default)
        self.sample_rate = 16000  # set to your dataset SR
        cf, sh = eq_freqs(interval=1)
        # Clamp bands to be valid for our SR (avoid > Nyquist)
        nyq = self.sample_rate * 0.5
        cf = torch.as_tensor(cf, dtype=torch.float32)
        cf = cf[cf <= (nyq * 0.98)]
        self.geq_centers = cf  # tensor of center frequencies (Hz)
        # shelving crossovers as tensor of length 2 (low & high)
        sh = torch.as_tensor(sh, dtype=torch.float32)
        sh = torch.stack([torch.clamp(sh[0], min=20.0), torch.clamp(sh[1], max=nyq * 0.98)])
        self.geq_shelves = sh
        # Two shelves + one peak per center frequency
        self.n_geq = int(self.geq_centers.numel() + 3)
        self.max_gain_db = 12.0

        # MLP head to predict per‑band dB gains
        self.geq_head = nn.Linear(hidden_size, self.n_geq)

        # Per‑frame loop design parameters (g, b1, b2) – raw logits, linear head
        self.coefficients_head = nn.Linear(hidden_size, 3)  # [g_logit, b1_logit, b2_logit]
        with torch.no_grad():
            if isinstance(self.coefficients_head, nn.Linear):
                self.coefficients_head.weight.zero_()
                self.coefficients_head.bias.zero_()

        # Time‑varying gain head (applied to excitation before GEQ)
        self.gain_head = nn.Linear(hidden_size, 1)
        # Initialize to 0 so g_db(0) = 0 dB (neutral)
        with torch.no_grad():
            self.gain_head.weight.zero_()
            self.gain_head.bias.zero_()

        with torch.no_grad():
            # GEQ: start at 0 dB on every band
            self.geq_head.weight.zero_()
            self.geq_head.bias.zero_()

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
            return_parameters=False
    ):
        """
        Forward pass of the neural Karplus-Strong model.

        Args:
            pitch: Tensor of shape [batch_size, frames, 1] - pitch values (f0)
            loudness: Tensor of shape [batch_size, frames, 1] - Loudness values

        Returns:
            If `return_parameters` is False:
                Tensor of shape [batch_size, n_samples] — synthesized audio.
            If `return_parameters` is True:
                Tuple (loop_coeffs_c, geq_info) where:
                    loop_coeffs_c: Tensor [B, T, 3] of raw loop coefficients logits [g, b1, b2].
                    geq_info: dict with keys "centers_hz", "shelves_hz", and "gains_db".
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

        # --- Predict GEQ gains from hidden; pool across time to make them static per example
        h_pool = hidden.mean(dim=1)               # [B, H]
        geq_logits = self.geq_head(h_pool)        # [B, K] – K = self.n_geq
        gains_db   = self.max_gain_db * torch.tanh(geq_logits)  # clamp to ±max_gain_db

        # ─── 3. Onset-based excitation (DiffKS) ────────────────────────────────
        B, N = audio.shape
        T = pitch.size(1)  # number of pitch frames
        # (No inversion) — detect onsets directly on the target audio (moved earlier for segment attention)
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

        # Per-frame head (simple MLP projection)
        l_b_frames = self.coefficients_head(hidden)  # [B, T, 3]

        # Build bursts per item with per-onset length derived from local pitch,
        exc_rows = []
        for b in range(B):
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

                burst_start = s
                burst_end   = min(burst_start + L_loc, N)
                seg_noi     = nb[:, burst_start:burst_end]  # [1, L]

                nb_scaled = nb.clone()
                nb_scaled[:, burst_start:burst_end] = seg_noi #* gain
                exc_b = exc_b + nb_scaled
            exc_rows.append(exc_b)

        excitation = torch.cat(exc_rows, dim=0)       # [B, N]

        # --- Time‑varying gain applied to bursts before GEQ ---
        # Asymmetric dB mapping: [-inf, +12] dB (no cap on attenuation, +12 dB max boost)
        # g_db(x) = 12 - (12/ln 2) * softplus(x); g_db(0)=0 dB, g_db→12 dB as x→-inf, g_db→-inf as x→+inf
        g_logits = self.gain_head(hidden)  # [B, T, 1]
        g_db = 12.0 - (12.0 / math.log(2.0)) * F.softplus(g_logits)

        gain_frames = torch.pow(10.0, g_db / 20.0)  # dB → linear; (0, 10^(12/20)]
        # Numerical floor to avoid exact zeros that can kill gradients
        gain_frames = torch.clamp(gain_frames, min=1e-6)

        gain_up = F.interpolate(gain_frames.transpose(1, 2), size=N, mode="linear", align_corners=False).squeeze(1)
        excitation = excitation * gain_up

        # Graphic‑EQ shaping (learnable dB gains) in frequency domain
        excitation = self._apply_geq_fd(
            excitation,   # [B, N]
            gains_db=gains_db,  # [B, K]
            sr=audio_sr,
        )

        if return_parameters:
            loop_coeffs_c = self.decoder.design_loop(f0= pitch.squeeze(2), l_b=l_b_frames)  # [B, T, 3] = [g_logit, b1_logit, b2_logit]
            geq_info = {
                "centers_hz": self.geq_centers.detach().cpu(),
                "shelves_hz": self.geq_shelves.detach().cpu(),
                "gains_db":    gains_db.detach().cpu(),
            }
            return loop_coeffs_c, geq_info

        out = self.decoder(
            f0_frames=pitch.squeeze(2),
            input=excitation,
            input_sr=audio_sr,
            loop_coefficients=l_b_frames,  # feed raw logits [g, b1, b2] directly (no upsampling)
        )

        return out

    def _apply_geq_fd(self, x: torch.Tensor, gains_db: torch.Tensor, sr: int) -> torch.Tensor:
        """Apply FLAMO GEQ as a cascade of SOS in the frequency domain.
        x: [B, N] time‑domain mono
        gains_db: [B, K] where K = 2 + len(self.geq_centers) – ordering assumed
                  [low_shelf, centers..., high_shelf].
        Returns: [B, N]
        """
        B, N = x.shape
        # FFT size: power‑of‑two >= N for decent speed
        nfft = 1
        while nfft < N:
            nfft <<= 1
        # Precompute rfft frequency grid once
        # Shape [nfft//2+1]
        w = 2.0 * math.pi * torch.arange(0, nfft // 2 + 1, device=x.device, dtype=x.dtype) / float(nfft)
        z1 = torch.exp(-1j * w)
        z2 = torch.exp(-2j * w)

        y_list = []
        # Build tensors for flamo.geq()
        cf = self.geq_centers.to(device=x.device, dtype=x.dtype)
        sh = self.geq_shelves.to(device=x.device, dtype=x.dtype)
        R  = torch.tensor(2.7, device=x.device, dtype=x.dtype)  # resonance used by FLAMO GEQ

        for b in range(B):
            gdb = gains_db[b]  # [K]
            # Sanity: if user supplied head dimension matches K exactly, use as is
            # If K differs by one (due to library layout), truncate/pad gracefully
            K_needed = int(cf.numel() + 3)
            if gdb.numel() != K_needed:
                if gdb.numel() > K_needed:
                    gdb = gdb[:K_needed]
                else:
                    pad = torch.zeros(K_needed - gdb.numel(), device=gdb.device, dtype=gdb.dtype)
                    gdb = torch.cat([gdb, pad], dim=0)

            b_sos, a_sos = geq_sos(
                center_freq=cf,
                shelving_freq=sh,
                R=R,
                gain_db=gdb,  # length = len(cf) + 3
                fs=float(sr),
                device=x.device
            )
            # b_sos, a_sos expected shapes [3, S] where S = number of SOS
            # Compute section responses and multiply
            b0, b1, b2 = b_sos[0], b_sos[1], b_sos[2]
            a0, a1, a2 = a_sos[0], a_sos[1], a_sos[2]
            # Normalize to a0 = 1 if needed
            a0_safe = a0 + 1e-12
            b0, b1, b2 = b0 / a0_safe, b1 / a0_safe, b2 / a0_safe
            a1, a2     = a1 / a0_safe, a2 / a0_safe

            # Shape broadcasting: each section against frequency grid
            # Section response: (b0 + b1 z^{-1} + b2 z^{-2}) / (1 + a1 z^{-1} + a2 z^{-2})
            num = b0.view(-1, 1) + b1.view(-1, 1) * z1 + b2.view(-1, 1) * z2
            den = 1.0 + a1.view(-1, 1) * z1 + a2.view(-1, 1) * z2
            H_sections = num / (den + 1e-30)
            H = torch.prod(H_sections, dim=0)  # [nfft//2+1]

            X = torch.fft.rfft(x[b], n=nfft)
            H = H.to(dtype=X.dtype)
            Y = X * H
            # Time-domain signal after EQ
            y = torch.fft.irfft(Y, n=nfft).real[:N]

            # --- Make-up gain: preserve overall RMS loudness of excitation ---
            pre_rms  = torch.sqrt(torch.clamp((x[b] ** 2).mean(), min=1e-12))
            post_rms = torch.sqrt(torch.clamp((y    ** 2).mean(), min=1e-12))
            makeup   = pre_rms / (post_rms + 1e-12)
            y = y * makeup

            y_list.append(y.to(dtype=x.dtype))

        return torch.stack(y_list, dim=0)