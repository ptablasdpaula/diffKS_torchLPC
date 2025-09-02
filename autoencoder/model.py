import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import numpy as np
from diffKS import DiffKS
from data.preprocess import E2_HZ
from utils.ml import mlp, gru
from core import make_onset_noise, detect_onsets_librosa
import math
from torchlpc import sample_wise_lpc
from flamo.auxiliary.eq import eq_freqs, geq as geq_sos

# --- Generic autoencoder z-encoder classes ---
class ZEncoder(nn.Module):
    def __init__(self, input_keys=None):
        super().__init__()

    def forward(self, audio, f0_scaled=None):
        z = self.compute_z(audio)
        if f0_scaled is not None:
            time_steps = f0_scaled.shape[1]
            z = self.expand_z(z, time_steps)
        return z

    def expand_z(self, z, time_steps):
        if len(z.shape) == 2:
            z = z.unsqueeze(1)
        z_time_steps = z.shape[1]
        if z_time_steps != time_steps:
            z = z.transpose(1, 2)
            z = torch.nn.functional.interpolate(
                z,
                size=time_steps,
                mode='linear',
                align_corners=False
            )
            z = z.transpose(1, 2)
        return z

    def compute_z(self, audio):
        raise NotImplementedError

class MfccTimeDistributedRnnEncoder(ZEncoder):
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
        elif z_time_steps == 2000:
            self.fft_size = 128
            self.overlap = 0.75
        else:
            raise ValueError('`z_time_steps` currently limited to 63, 125, 250, 500, 1000, and 2000')
        self.hop_length = int(self.fft_size * (1.0 - self.overlap))
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
        self.z_norm = nn.InstanceNorm1d(30)
        if rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(30, rnn_channels, batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        self.dense_out = nn.Linear(rnn_channels, z_dims)

    def compute_z(self, audio):
        # MFCC: [B, n_mfcc, T]
        mfccs = self.mfcc_transform(audio)
        # InstanceNorm1d expects [B, C, T]
        mfccs = self.z_norm(mfccs)
        # GRU expects [B, T, C]
        mfccs = mfccs.transpose(1, 2)
        rnn_out, _ = self.rnn(mfccs)
        z = self.dense_out(rnn_out)
        return z



# --- nnKarplusStrong with learned noisebank excitation (no GEQ, no AST) ---
class nnKarplusStrong(nn.Module):
    def __init__(self,
                 batch_size,
                 loop_order,
                 internal_sr,
                 interpolation_type,
                 filter_type,
                 timesteps: int = 250,
                 n_noise_bands: int = 64,
                 hidden_size: int = 256):
        super().__init__()
        self.internal_sr = internal_sr
        self.loop_order = loop_order
        self.timesteps = timesteps
        self.n_noise_bands = n_noise_bands

        # --- z-encoder (MFCC+RNN) ---
        self.z_encoder = MfccTimeDistributedRnnEncoder(
            rnn_channels=512,
            z_dims=16,
            z_time_steps=timesteps,
            sample_rate=internal_sr,
        )

        # --- small control network (restored AE-style wiring) ---
        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3), mlp(1, hidden_size, 3)])
        self.mlp_z = mlp(self.z_encoder.z_dims, hidden_size, 3)
        self.gru = nn.GRU(3 * hidden_size, hidden_size, batch_first=True)
        self.out_mlp = mlp(3 * hidden_size, hidden_size, 3)

        # Ensure both heads exist
        self.loop_head = nn.Linear(hidden_size, loop_order + 1)   # loop filter coeffs per frame
        self.burst_gain_head = nn.Linear(hidden_size, 1)          # g_db[t] for burst scaling (attenuation)
        # --- LPC head for excitation colouring ---
        self.lpc_order = 6  # choose 5–8; 6 is a good default for excitation colour
        self.lpc_head  = nn.Linear(hidden_size, self.lpc_order)  # predicts RCs per frame
        nn.init.zeros_(self.lpc_head.weight)
        nn.init.zeros_(self.lpc_head.bias)
        # With rc = 0 -> a = 0 -> identity all-pole filter (no filtering)

        # --- GEQ layout (one‑third‑octave bands), applied POST-KS ---
        self.sample_rate = internal_sr
        cf, sh = eq_freqs(interval=3)
        nyq = self.sample_rate * 0.5
        cf = torch.as_tensor(cf, dtype=torch.float32)
        cf = cf[cf <= (nyq * 0.98)]
        self.geq_centers = cf
        sh = torch.as_tensor(sh, dtype=torch.float32)
        sh = torch.stack([torch.clamp(sh[0], min=20.0), torch.clamp(sh[1], max=nyq * 0.98)])
        self.geq_shelves = sh
        self.n_geq = int(self.geq_centers.numel() + 3)
        self.max_gain_db = 12.0
        self.geq_head = nn.Linear(hidden_size, self.n_geq)  # clip-global body EQ

        # Differentiable KS decoder
        self.decoder = DiffKS(
            batch_size=batch_size,
            internal_sr=internal_sr,
            loop_order=loop_order,
            loop_n_frames=timesteps,
            interp_type=interpolation_type,
            use_double_precision=False,
            min_f0_hz=E2_HZ - 10,
            loop_filter_kind=filter_type,
        )
        for p in self.decoder.parameters():
            p.requires_grad = False

    def forward(self, pitch, _loudness, audio, audio_sr, return_parameters=False,
                triggers: bool = False, trigger_width_frames: int = 1):
        B, N = audio.shape
        T_frames = pitch.size(1)
        def _resize_time(x, T):
            # x: [B, T_old, C]
            if x.size(1) == T:
                return x
            x = x.transpose(1, 2)  # [B, C, T_old]
            x = nn.functional.interpolate(x, size=T, mode='linear', align_corners=False)
            return x.transpose(1, 2)  # [B, T, C]
        if T_frames != self.timesteps:
            pitch = _resize_time(pitch, self.timesteps)
            _loudness = _resize_time(_loudness, self.timesteps)
            T_frames = self.timesteps
        # --- Encode to z ---
        z = self.z_encoder(audio, f0_scaled=pitch)  # [B, T, z_dim]

        # --- Build hidden sequence (AE-style) ---
        x_f = self.in_mlps[0](pitch)           # [B, T, H]
        x_l = self.in_mlps[1](_loudness)       # [B, T, H]
        x_z = self.mlp_z(z)                    # [B, T, H]
        # GRU sees [x_f, x_l, x_z]
        h_in = torch.cat([x_f, x_l, x_z], dim=-1)  # [B, T, 3H]
        h_gru, _ = self.gru(h_in)                  # [B, T, H]
        # out_mlp sees [gru_out, x_f, x_l] (z not appended post-GRU)
        hidden = torch.cat([h_gru, x_f, x_l], dim=-1)  # [B, T, 3H]
        hidden = self.out_mlp(hidden)                  # [B, T, H]
        loop_logits = self.loop_head(hidden)  # [B, T, loop_order+1]

        # --- Onset-triggered noise burst excitation (recovered) ---
        # Map sample index -> frame index helper
        def _sample_to_frame_idx(s: int, N: int, T: int) -> int:
            idx = int(round(s * T / max(1, N)))
            return max(0, min(T - 1, idx))

        # Detect onsets per batch
        onset_list = []
        for b in range(B):
            on_b = detect_onsets_librosa(audio[b], sr=int(audio_sr))
            if on_b.size == 0:
                on_b = np.array([0], dtype=int)
            onset_list.append(on_b)

        # Build bursts-only excitation per batch (no impulses)
        exc_burst_rows = []
        for b in range(B):
            burst_b = torch.zeros(1, N, device=audio.device, dtype=audio.dtype)
            for s in onset_list[b].tolist():
                f_idx = _sample_to_frame_idx(s, N, T_frames)
                f0_loc = float(torch.clamp(pitch[b, f_idx, 0], min=E2_HZ).item())
                L_loc = int(round(float(audio_sr) / max(f0_loc, 1e-6)))
                nb_burst = make_onset_noise(
                    onset_samples=np.array([s], dtype=int),
                    num_samples=N,
                    sample_rate=int(audio_sr),
                    batch_size=1,
                    device=audio.device,
                    dtype=audio.dtype,
                    burst_len_samples=L_loc,
                    impulse_instead=False,
                )  # [1, N]
                burst_b = burst_b + nb_burst
            exc_burst_rows.append(burst_b)
        exc_burst = torch.cat(exc_burst_rows, dim=0)  # [B, N]

        # --- Framewise burst gain (attenuation in dB) from head ---
        g_db_frames = self.burst_gain_head(hidden)              # [B, T, 1]
        gain_frames = torch.pow(10.0, (-F.softplus(g_db_frames)) / 20.0)  # [B, T, 1], <= 1
        gain_up = F.interpolate(
            gain_frames.transpose(1, 2),  # [B, 1, T]
            size=N, mode="linear", align_corners=False
        ).squeeze(1)  # [B, N]
        excitation = exc_burst * gain_up                        # [B, N]

        # --- Time-varying LPC (all-pole) shaping of the burst (pre-KS) ---
        # 1) Predict reflection coefficients per frame and squash with tanh to keep |k|<1 (stable lattice)
        rc_frames = torch.tanh(self.lpc_head(hidden))  # [B, T, P]

        # 2) Convert RC (PARCOR) -> LPC a-coeffs per frame via step-up recursion (vectorised over batch/time)
        # Returns A_frames with shape [B, T, P], where the synthesis filter is 1 / (1 - sum a_i z^{-i})
        def rc_to_lpc(rc: torch.Tensor) -> torch.Tensor:
            Bf, Tf, P = rc.shape
            a = torch.zeros(Bf, Tf, P, device=rc.device, dtype=rc.dtype)
            for m in range(P):
                km = rc[..., m]  # [Bf, Tf]
                if m == 0:
                    a[..., 0] = km
                else:
                    # Use the previous stage coefficients a[..., :m] from the last iteration
                    prev = a[..., :m].clone()          # clone only the needed slice (a_{m-1})
                    rev  = torch.flip(prev, dims=[-1])
                    a[..., :m] = prev + km.unsqueeze(-1) * rev
                    a[..., m]  = km
            return a

        A_frames = rc_to_lpc(rc_frames)  # [B, T, P]

        # 3) Upsample LPC a-coeffs from frame-rate to sample-rate
        A_up = F.interpolate(A_frames.transpose(1, 2),  # [B, P, T]
                             size=N,
                             mode="linear",
                             align_corners=False).transpose(1, 2)  # [B, N, P]

        # 4) Apply differentiable sample-wise LPC synthesis filter (torchLPC)
        excitation = sample_wise_lpc(excitation, A_up)  # [B, N]

        # 5) Optional: small per-burst local DC removal (safety)
        B2, N2 = excitation.shape
        cleaned = []
        for b in range(B2):
            y = excitation[b]
            abs_y = y.abs()
            thr = torch.maximum(abs_y.max() * y.new_tensor(1e-3), y.new_tensor(1e-8))
            mask = abs_y >= thr
            if bool(mask.any()):
                m = mask.to(torch.int8)
                dm = m[1:] - m[:-1]
                starts = torch.nonzero(dm == 1, as_tuple=False).squeeze(-1) + 1
                ends   = torch.nonzero(dm == -1, as_tuple=False).squeeze(-1) + 1
                if bool(m[0]):
                    starts = torch.cat([torch.tensor([0], device=y.device, dtype=starts.dtype), starts], dim=0) if starts.numel() else torch.tensor([0], device=y.device, dtype=torch.long)
                if bool(m[-1]):
                    ends = torch.cat([ends, torch.tensor([N2], device=y.device, dtype=ends.dtype)], dim=0) if ends.numel() else torch.tensor([N2], device=y.device, dtype=torch.long)
                if starts.numel() and ends.numel():
                    count = min(starts.numel(), ends.numel())
                    starts = starts[:count]
                    ends   = ends[:count]
                    for s_idx, e_idx in zip(starts.tolist(), ends.tolist()):
                        if e_idx > s_idx:
                            seg = y[s_idx:e_idx]
                            new_seg = seg - seg.mean()
                            y = torch.cat([y[:s_idx], new_seg, y[e_idx:]], dim=0)
            cleaned.append(y)
        excitation = torch.stack(cleaned, dim=0)

        # --- Synthesize with DiffKS ---
        out = self.decoder(
            f0_frames=pitch.squeeze(2),
            input=excitation,
            input_sr=audio_sr,
            loop_coefficients=loop_logits,
        )  # [B, N]

        # --- Clip-global GEQ (body) applied POST-KS ---
        h_pool = hidden.mean(dim=1)                          # [B, H]
        geq_logits = self.geq_head(h_pool)                   # [B, K]
        gains_db = self.max_gain_db * torch.tanh(geq_logits) # [-max,+max] dB per band
        out_eq = self._apply_geq_fd(out, gains_db=gains_db, sr=int(audio_sr))

        if return_parameters:
            return {
                "loop_logits": loop_logits,
                "excitation_burst_raw": exc_burst.detach(),
                "gain_frames_db": g_db_frames.detach(),
                "lpc_rc_frames": rc_frames.detach(),
                "lpc_a_frames": A_frames.detach(),
                "excitation_burst_lpc": excitation.detach(),
                "geq_info": {
                    "centers_hz": self.geq_centers.detach().cpu(),
                    "shelves_hz": self.geq_shelves.detach().cpu(),
                    "gains_db": gains_db.detach().cpu(),
                },
                "decoder_out_pre_geq": out.detach(),
                "decoder_out_post_geq": out_eq.detach(),
                "onsets": onset_list,
            }

        return out_eq

    def _apply_geq_fd(self, x: torch.Tensor, gains_db: torch.Tensor, sr: int) -> torch.Tensor:
        """Apply FLAMO GEQ as a cascade of SOS in the frequency domain.
        x: [B, N] time‑domain mono
        gains_db: [B, K] where K = 2 + len(self.geq_centers) – ordering assumed
                  [low_shelf, centers..., high_shelf].
        Returns: [B, N]
        """
        B, N = x.shape
        # FFT size: next power-of-two >= N
        nfft = 1
        while nfft < N:
            nfft <<= 1

        # Frequency grid for IIR frequency response (complex z^{-1})
        w = 2.0 * math.pi * torch.arange(0, nfft // 2 + 1, device=x.device, dtype=x.dtype) / float(nfft)
        z1 = torch.exp(-1j * w)
        z2 = torch.exp(-2j * w)

        cf = self.geq_centers.to(device=x.device, dtype=x.dtype)
        sh = self.geq_shelves.to(device=x.device, dtype=x.dtype)
        R  = torch.tensor(2.7, device=x.device, dtype=x.dtype)  # resonance used by FLAMO GEQ

        y_list = []
        K_needed = int(cf.numel() + 3)

        for b in range(B):
            gdb = gains_db[b]
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
                gain_db=gdb,
                fs=float(sr),
                device=x.device,
            )

            b0, b1, b2 = b_sos[0], b_sos[1], b_sos[2]
            a0, a1, a2 = a_sos[0], a_sos[1], a_sos[2]
            a0_safe = a0 + 1e-12
            b0, b1, b2 = b0 / a0_safe, b1 / a0_safe, b2 / a0_safe
            a1, a2     = a1 / a0_safe, a2 / a0_safe

            num = b0.view(-1, 1) + b1.view(-1, 1) * z1 + b2.view(-1, 1) * z2
            den = 1.0 + a1.view(-1, 1) * z1 + a2.view(-1, 1) * z2
            H_sections = num / (den + 1e-30)
            H = torch.prod(H_sections, dim=0)  # [nfft//2+1]

            X = torch.fft.rfft(x[b], n=nfft)
            H = H.to(dtype=X.dtype)
            Y = X * H
            y_time = torch.fft.irfft(Y, n=nfft).real[:N]
            y_list.append(y_time.to(dtype=x.dtype))

        return torch.stack(y_list, dim=0)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]