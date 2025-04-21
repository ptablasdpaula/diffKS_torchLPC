import json, os, librosa, torch
import numpy as np
from torch.utils.data import Dataset
from ddsp_pytorch.ddsp.core import extract_pitch

def _extract_loudness(signal, sampling_rate, block_size, n_fft=2048):
    S = librosa.stft(
        signal,
        n_fft=n_fft,
        hop_length=block_size,
        win_length=n_fft,
        center=True,
    )
    S = np.log(abs(S) + 1e-7)
    f = librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
    a_weight = librosa.A_weighting(f)

    S = S + a_weight.reshape(-1, 1)

    S = np.mean(S, 0)[..., :-1]

    return S

class SingleAudioDataset(Dataset):
    def __init__(self, audio_path, segment_length=16000 * 4, hop_size=256):
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Make sure audio is at least segment_length long
        if len(audio) < segment_length:
            audio = np.pad(audio, (0, segment_length - len(audio)))

        # Convert to torch tensor
        self.audio = torch.from_numpy(audio.astype(np.float32))
        self.segment_length = segment_length
        self.hop_size = hop_size

        # Calculate number of possible segments
        self.num_segments = (len(audio) - segment_length) // hop_size + 1

        print(f"Loaded audio length: {len(audio)} samples")
        print(f"Segment length: {segment_length}")
        print(f"Hop size: {hop_size}")
        print(f"Number of segments: {self.num_segments}")

        # Extract features for whole file
        self._extract_features()

    def _extract_features(self):
        # Extract pitch and loudness for the entire audio file
        audio_np = self.audio.numpy()

        # Extract pitch (f0)
        self.pitch = extract_pitch(audio_np, 16000, self.hop_size)

        # Extract loudness
        self.loudness = _extract_loudness(audio_np, 16000, self.hop_size)

        # Convert to torch tensors
        self.pitch = torch.from_numpy(self.pitch.astype(np.float32))
        self.loudness = torch.from_numpy(self.loudness.astype(np.float32))

    def __len__(self):
        return self.num_segments

    def __getitem__(self, idx):
        # Get audio segment
        start_idx = idx * self.hop_size
        end_idx = start_idx + self.segment_length
        audio_segment = self.audio[start_idx:end_idx]

        # Calculate corresponding frame indices for features
        frames_per_segment = (self.segment_length // self.hop_size)
        start_frame = idx
        end_frame = start_frame + frames_per_segment

        # Get pitch and loudness for this segment
        pitch_segment = self.pitch[start_frame:end_frame]
        loudness_segment = self.loudness[start_frame:end_frame]

        return (audio_segment,
                pitch_segment.unsqueeze(-1),    # [F, 1]
                loudness_segment.unsqueeze(-1)) # [F, 1]

class NsynthDataset(Dataset):
    """
    Each item  ->  (audio [T],  pitch [F,1],  loudness [F,1])
    All audio in NSynth is 4s @ 16kHz  (64000 samples, 1000 frames if hop=64)
    """
    _FAMILY_STR = {"bass",
                   "brass",
                   "flute",
                   "guitar",
                   "keyboard",
                   "mallet",
                   "organ",
                   "reeds",
                   "string",
                   "synth_lead",
                   "vocal"}

    _SRC_STR = {"acoustic",
                "electronic",
                "synthetic"}

    def __init__(
            self,
            root_dir,  # …/data/nsynth
            split="test",  # 'train' | 'valid' | 'test'
            families=("guitar",),  # keep anything in this set
            sources=("acoustic",),  # ditto for instrument_source_str
            sample_rate=16000,
            hop_size=256,
            segment_length=None  # None → entire 4‑s clip
    ):
        self.sr = sample_rate
        self.hop_size = hop_size
        self.seg_len = segment_length

        # ─── Load metadata ─────────────────────────────────── #
        split_dir = os.path.join(root_dir, f"nsynth-{split}")
        meta_path = os.path.join(split_dir, "examples.json")
        with open(meta_path) as fp:
            meta = json.load(fp)

        # ─── Keep only wanted instruments ─────────────────── #
        self.items = []
        for key, m in meta.items():
            if m["instrument_family_str"] not in families:      continue
            if m["instrument_source_str"] not in sources:       continue
            wav = os.path.join(split_dir, "audio", f"{key}.wav")
            self.items.append(wav)

        if not self.items:
            raise RuntimeError("No files matched the given filters!")

        print(f"[NSynthDataset] split={split}, "
              f"{len(self.items)} files, "
              f"families={families}, sources={sources}")

    # ───────────────────────────────────────────────────────── #
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path = self.items[idx]
        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        if self.seg_len:  # optional slicing
            if len(audio) < self.seg_len:
                audio = np.pad(audio, (0, self.seg_len - len(audio)))
            else:
                audio = audio[: self.seg_len]

        # ── Feature extraction (frame hop = hop_size) ───────── #
        pitch = extract_pitch(audio, self.sr, self.hop_size)  # [F]
        loud = _extract_loudness(audio, self.sr, self.hop_size)  # [F]

        # Torchify & add channel dim expected by your model
        return (torch.from_numpy(audio.astype(np.float32)),
                torch.from_numpy(pitch.astype(np.float32)).unsqueeze(-1),
                torch.from_numpy(loud.astype(np.float32)).unsqueeze(-1))