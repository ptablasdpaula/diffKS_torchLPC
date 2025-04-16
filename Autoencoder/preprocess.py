import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import librosa
from ddsp_pytorch.ddsp.core import extract_pitch

def extract_loudness(signal, sampling_rate, block_size, n_fft=2048):
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
        self.loudness = extract_loudness(audio_np, 16000, self.hop_size)

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