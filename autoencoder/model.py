import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import numpy as np
from transformers import ASTModel
from diffKS import DiffKS
from data.preprocess import E2_HZ
from core import make_onset_noise, detect_onsets_librosa
from flamo.auxiliary.eq import eq_freqs, geq as geq_sos

import math
import re

# ========================================================================
# ASTConditioner: Audio-Spectrogram-Transformer based conditioner
# ========================================================================

class ASTConditioner(nn.Module):
    """Audio-Spectrogram-Transformer style conditioner that predicts
    per-frame parameters for the DDSP decoder (gain + loop coeffs)
    and clip-global GEQ band gains.

    Differences vs. canonical AST:
    - Matches AST: 16x16 patches, stride 10×10, 128-mel, 25 ms / 10 ms frontend.
    - Log-mel normalization with configurable mean/std (AudioSet defaults).
    """
    def __init__(self,
                   n_mels: int = 128,
                   input_tdim: int = 401,
                   sample_rate: int = 16000,
                   embed_dim: int = 768,
                   loop_order: int = 2,
                   n_geq: int = 32,
                   do_normalize: bool = True,
                   norm_mean: float = -4.2677393,
                   norm_std: float = 4.5689974):
        super().__init__()
        self.sample_rate = sample_rate
        self.input_tdim = input_tdim
        self.n_mels = n_mels
        self.embed_dim = embed_dim
        self.loop_order = loop_order
        self.n_geq = n_geq
        self.do_normalize = do_normalize

        # Spectrogram frontend (AST defaults)
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=int(round(self.sample_rate * 0.025)),  # 25 ms window
            hop_length=int(round(self.sample_rate * 0.01)),   # 10 ms hop
            f_min=20.0,
            f_max=8000.0,
            n_mels=n_mels,
            window_fn=torch.hamming_window,
            power=2.0,
            pad_mode='reflect'
        )
        self.ampl_to_db = T.AmplitudeToDB(stype='power')

        # Normalization buffers (AudioSet defaults)
        self.register_buffer('norm_mean', torch.tensor(norm_mean))
        self.register_buffer('norm_std', torch.tensor(norm_std))

        # Hugging Face AST backbone (pretrained). This includes the conv patch embed,
        # learned 1D positional embeddings (with interpolation), and ViT encoder.
        self.backbone = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            ignore_mismatched_sizes=True,
        )

        # Mel-axis attention pooling (to collapse mel patches per time step)
        self.mel_attn = nn.Linear(embed_dim, 1)

        # Heads
        self.gain_head = nn.Linear(embed_dim, 1)                 # g_db[t]
        self.loop_head = nn.Linear(embed_dim, loop_order + 1)    # raw logits per frame
        self.geq_head  = nn.Linear(embed_dim, n_geq)             # clip-global from pooled token

    def set_heads(self, loop_order: int, n_geq: int):
        self.loop_order = loop_order
        self.n_geq = n_geq
        self.loop_head = nn.Linear(self.embed_dim, loop_order + 1)
        self.geq_head  = nn.Linear(self.embed_dim, n_geq)

    def _set_requires_grad(self, module: nn.Module, flag: bool) -> None:
        for p in module.parameters():
            p.requires_grad = flag


    def freeze_backbone(self) -> None:
        """Freeze the AST backbone; leave mel-attn and heads trainable."""
        self._set_requires_grad(self.backbone, False)
        self.backbone.eval()
        # Ensure heads remain trainable
        self._set_requires_grad(self.mel_attn, True)
        self._set_requires_grad(self.gain_head, True)
        self._set_requires_grad(self.loop_head, True)
        self._set_requires_grad(self.geq_head, True)


    def unfreeze_backbone_last(self,
                               n_layers: int = 2,
                               also_unfreeze_layernorm: bool = True,
                               train_pos_embed: bool = False) -> None:
        """Partially unfreeze the AST backbone.
        Unfreezes the last `n_layers` Transformer blocks and (optionally) all
        LayerNorm parameters and/or positional embeddings.
        This is safer than unfreezing the whole backbone and often stabilizes
        fine‑tuning.
        """
        # 1) Start from a fully frozen backbone
        self._set_requires_grad(self.backbone, False)
        # Put backbone in train mode so dropout etc. behave correctly; only the
        # parameters we re‑enable below will receive grads.
        self.backbone.train()

        # 2) Determine total number of hidden layers, if available
        total_layers = int(getattr(self.backbone.config, "num_hidden_layers", -1))

        # 3) Re‑enable gradients for the last N encoder blocks by name pattern
        # Works for HF models that expose modules like: *.encoder.layer.{i}.*
        for name, p in self.backbone.named_parameters():
            unfreeze = False

            m = re.search(r"\.layer\.(\d+)\.", name)
            if m is not None and total_layers > 0:
                idx = int(m.group(1))
                if idx >= max(0, total_layers - n_layers):
                    unfreeze = True

            # Optionally always let LayerNorms update (helps stability)
            if also_unfreeze_layernorm:
                lname = name.lower()
                if ("layernorm" in lname) or (".ln" in lname) or ("norm" in lname):
                    unfreeze = True or unfreeze

            # Optionally allow positional embeddings to update
            if train_pos_embed and ("pos_embed" in name or
                                    "position_embeddings" in name or
                                    "position_embedding" in name):
                unfreeze = True or unfreeze

            if unfreeze:
                p.requires_grad = True

        # 4) Ensure our heads remain trainable regardless
        self._set_requires_grad(self.mel_attn, True)
        self._set_requires_grad(self.gain_head, True)
        self._set_requires_grad(self.loop_head, True)
        self._set_requires_grad(self.geq_head, True)

    def _logmel(self, audio: torch.Tensor) -> torch.Tensor:
        """Return normalized log-mel [B, M, Tm]."""
        mel = self.mel(audio)
        logmel = self.ampl_to_db(mel + 1e-10)
        if self.do_normalize:
            logmel = (logmel - self.norm_mean) / (self.norm_std + 1e-6)
        return logmel

    def predict_parameters(self,
                             audio: torch.Tensor,
                             ):
        """Predict (g_db[t], loop_logits[t], geq_logits) from audio."""

        # 1) Log-mel
        logmel = self._logmel(audio)        # [B, M, Tm]
        Tm = logmel.shape[-1]

        # 2) Pad/trim time axis to backbone config.max_length (AST expects fixed T)
        T_target = int(getattr(self.backbone.config, "max_length", Tm))
        if Tm < T_target:
            pad_right = T_target - Tm
            # pad last dimension (time) with zeros AFTER normalization (ASTFeatureExtractor uses 0.0)
            logmel = F.pad(logmel, (0, pad_right), value=0.0)
        elif Tm > T_target:
            logmel = logmel[..., :T_target]

        # 3) AST backbone: feed normalized log-mel as [B, T, M]
        Bsz = logmel.size(0)
        ast_out = self.backbone(input_values=logmel.transpose(1, 2))  # [B, special + patches, E]
        mem_all = ast_out.last_hidden_state                           # includes CLS (+ distill, depending on backbone)

        # Derive grid sizes from the backbone config to avoid hard-coding
        cfg = self.backbone.config
        pm = (int(cfg.num_mel_bins) - int(cfg.patch_size)) // int(cfg.frequency_stride) + 1
        pt = (int(cfg.max_length)   - int(cfg.patch_size)) // int(cfg.time_stride)     + 1
        expected_patches = pm * pt

        # Keep only the patch tokens (drop 1 or 2 special tokens safely)
        # Token order is [CLS][(distill?)][patches...], so we slice the last `expected_patches` tokens.
        mem = mem_all[:, -expected_patches:, :]                       # [B, pm*pt, E]
        L = mem.size(1)
        assert L == expected_patches, (
            f"Unexpected number of patch tokens: got {L}, expected {expected_patches} (pm={pm}, pt={pt})."
        )
        mem2d = mem.view(Bsz, pm, pt, self.embed_dim)                 # [B, pm, pt, E]
        # --- Keep only the time tokens that correspond to the *actual* clip (pre-pad) ---
        # Tm is the original mel-frame length before zero-padding to T_target.
        # Compute effective time-patch count for content: pt_eff = floor((Tm - K) / s) + 1
        Kt = int(cfg.patch_size)
        st = int(cfg.time_stride)
        T_valid = int(min(Tm, T_target))
        pt_eff = max(1, min(pt, (T_valid - Kt) // st + 1))
        mem2d = mem2d[:, :, :pt_eff, :]                                 # [B, pm, pt_eff, E]

        # Pool over mel axis to get one token per *content* time step (no padded tail)
        scores = self.mel_attn(mem2d).squeeze(-1)                        # [B, pm, pt_eff]
        weights = torch.softmax(scores, dim=1)                           # softmax over mel axis
        h_time = (weights.unsqueeze(-1) * mem2d).sum(dim=1)              # [B, pt_eff, E]

        # 6) Heads directly on the native time base (no up/downsampling)
        h_ds = h_time                                             # [B, pt, E]
        g_db_frames = self.gain_head(h_ds)                        # [B, pt, 1]
        loop_logits = self.loop_head(h_ds)                        # [B, pt, loop_order+1]
        h_pool = h_ds.mean(dim=1)                                 # [B, E]
        geq_logits = self.geq_head(h_pool)                        # [B, n_geq]
        return g_db_frames, loop_logits, geq_logits

class nnKarplusStrong(nn.Module):
    def __init__(self,
                   batch_size,
                   loop_order,
                   internal_sr,
                   interpolation_type,
                   filter_type):
        super().__init__()
        self.internal_sr = internal_sr
        self.loop_order = loop_order

        # GEQ layout (one‑third‑octave bands)
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

        # AST conditioner
        self.z_encoder = ASTConditioner(
            n_mels=128,
            input_tdim=401,            # 4 s @ 10 ms hop -> ~401 mel frames
            sample_rate=internal_sr,
            embed_dim=768,             # ViT-Base width
            loop_order=loop_order,
            n_geq=self.n_geq,
            do_normalize=True,
        )

        cfg = self.z_encoder.backbone.config
        # Derive the *content* time token count from the conditioner input_tdim (e.g., 401 → 39)
        self.control_frames = ((int(self.z_encoder.input_tdim) - int(cfg.patch_size)) // int(cfg.time_stride)) + 1

        # Differentiable KS decoder
        self.decoder = DiffKS(
            batch_size=batch_size,
            internal_sr=internal_sr,
            loop_order=loop_order,
            loop_n_frames=self.control_frames,
            interp_type=interpolation_type,
            use_double_precision=False,
            min_f0_hz=E2_HZ - 10,
            loop_filter_kind=filter_type,
        )
        for p in self.decoder.parameters():
            p.requires_grad = False

    # --- training helpers -------------------------------------------------
    def freeze_backbone(self) -> None:
        """Freeze AST backbone; keep heads trainable. Decoder stays frozen."""
        if hasattr(self, "z_encoder") and hasattr(self.z_encoder, "freeze_backbone"):
            self.z_encoder.freeze_backbone()


    def unfreeze_backbone_last(self,
                               n_layers: int = 2,
                               also_unfreeze_layernorm: bool = True,
                               train_pos_embed: bool = False) -> None:
        """Partially unfreeze the conditioner backbone (last N blocks)."""
        if hasattr(self, "z_encoder") and hasattr(self.z_encoder, "unfreeze_backbone_last"):
            self.z_encoder.unfreeze_backbone_last(
                n_layers=n_layers,
                also_unfreeze_layernorm=also_unfreeze_layernorm,
                train_pos_embed=train_pos_embed,
            )

    def forward(self,
                pitch,
                _loudness,
                audio,
                audio_sr,
                return_parameters=False,):
        B, N = audio.shape
        T_frames = pitch.size(1)

        # 1) Predict frame-wise parameters and global EQ gains
        g_db_frames, l_b_frames, geq_logits = self.z_encoder.predict_parameters(
            audio=audio,
        )

        gains_db = self.max_gain_db * torch.tanh(geq_logits)

        # 2) Onset-based excitation
        onset_list = []
        for b in range(B):
            on_b = detect_onsets_librosa(
                audio[b], sr=audio_sr, pad_ms=50.0, hop_length=512, backtrack=True
            )
            if on_b.size == 0:
                on_b = np.array([0], dtype=int)
            onset_list.append(on_b)

        def sample_to_frame_idx(s: int) -> int:
            idx = int(round(s * T_frames / max(1, N)))
            return max(0, min(T_frames - 1, idx))

        # Build bursts-only excitation per batch (no impulses)
        burst_rows = []
        for b in range(B):
            burst_b = torch.zeros(1, N, device=audio.device, dtype=audio.dtype)
            for s in onset_list[b]:
                s = int(s)
                f_idx = sample_to_frame_idx(s)
                f0_loc = float(torch.clamp(pitch[b, f_idx, 0], min=E2_HZ).item())
                L_loc = int(round(float(audio_sr) / max(f0_loc, 1e-6)))

                # Noise-burst branch only
                nb_burst = make_onset_noise(
                    onset_samples=np.array([s], dtype=int),
                    num_samples=N,
                    sample_rate=audio_sr,
                    batch_size=1,
                    device=audio.device,
                    dtype=audio.dtype,
                    burst_len_samples=L_loc,
                    impulse_instead=False,
                )  # [1, N]

                burst_b = burst_b + nb_burst
            burst_rows.append(burst_b)

        exc_burst = torch.cat(burst_rows, dim=0)  # [B, N]

        # 3) Time‑varying gain (framewise), apply only to the burst branch in stages that learn gain
        gain_frames = torch.pow(10.0, (-F.softplus(g_db_frames)) / 20.0)  # [B, T_f, 1]
        gain_up = F.interpolate(
            gain_frames.transpose(1, 2),  # -> [B, 1, T_f]
            size=N,
            mode="linear",
            align_corners=False
        ).squeeze(1)  # -> [B, N]

        # Apply framewise gain to bursts (always)
        exc_burst_scaled = exc_burst * gain_up

        # Bursts-only excitation
        excitation_pregain = exc_burst
        excitation_postgain = exc_burst_scaled
        excitation = excitation_postgain

        # 4) Graphic‑EQ shaping (always on)
        excitation = self._apply_geq_fd(
            excitation,
            gains_db=gains_db,
            sr=audio_sr,
        )

        if return_parameters:
            # Optional: external smoothness/regularization can use these parameters.
            f0_for_params = F.interpolate(
                pitch.squeeze(2).unsqueeze(1),
                size=self.decoder.loop_n_frames,
                mode='linear',
                align_corners=False,
            ).squeeze(1)
            loop_coeffs_c = self.decoder.design_loop(f0=f0_for_params, l_b=l_b_frames, return_gain=True)
            geq_info = {
                "centers_hz": self.geq_centers.detach().cpu(),
                "shelves_hz": self.geq_shelves.detach().cpu(),
                "gains_db":    gains_db.detach().cpu(),
            }
            return loop_coeffs_c, geq_info, gain_frames, gain_up, excitation_pregain, excitation_postgain, excitation

        # 5) Synthesize with DiffKS
        out = self.decoder(
            f0_frames=pitch.squeeze(2),
            input=excitation,
            input_sr=audio_sr,
            loop_coefficients=l_b_frames,
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

            # --- Per-burst DC removal (POST-GEQ): detect active regions on y and subtract local mean ---
            # Build a robust activity mask from the post-EQ signal. Use a relative threshold
            # to include the ring-down so we don't introduce clicks at the burst edges.
            abs_y = y.abs()
            # Threshold at ~-60 dB of the peak, with a small floor for numerical stability
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
                    ends = torch.cat([ends, torch.tensor([N], device=y.device, dtype=ends.dtype)], dim=0) if ends.numel() else torch.tensor([N], device=y.device, dtype=torch.long)
                if starts.numel() and ends.numel():
                    count = min(starts.numel(), ends.numel())
                    starts = starts[:count]
                    ends   = ends[:count]
                    for s_idx, e_idx in zip(starts.tolist(), ends.tolist()):
                        if e_idx > s_idx:
                            seg = y[s_idx:e_idx]
                            seg_mean = seg.mean()
                            y[s_idx:e_idx] = seg - seg_mean

            y_list.append(y.to(dtype=x.dtype))


        return torch.stack(y_list, dim=0)
    def trainable_parameters(self):
        """Return only parameters that require gradients (useful if freezing legacy path)."""
        return [p for p in self.parameters() if p.requires_grad]