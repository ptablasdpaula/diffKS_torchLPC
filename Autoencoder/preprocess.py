import json, os, librosa, torch
import numpy as np
from torch.utils.data import Dataset
from ddsp_pytorch.ddsp.core import extract_pitch
from tqdm import tqdm
import pesto
import torchaudio
import librosa

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
    f = np.maximum(f, 1e-7)
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
    NSynth dataset implementation using PESTO for pitch extraction
    with batch dimension handling
    """
    _FAMILY_STR = {"bass", "brass", "flute", "guitar", "keyboard", "mallet",
                   "organ", "reeds", "string", "synth_lead", "vocal"}

    _SRC_STR = {"acoustic", "electronic", "synthetic"}

    def __init__(
            self,
            root_dir,
            split="test",
            families=("guitar",),
            sources=("acoustic",),
            sample_rate=16000,
            hop_size=256,
            segment_length=16000 * 4,
            max_size=None
    ):
        self.sr = sample_rate
        self.hop_size = hop_size
        self.segment_length = segment_length

        split_dir = os.path.join(root_dir, f"nsynth-{split}")
        meta_path = os.path.join(split_dir, "examples.json")
        with open(meta_path) as fp:
            meta = json.load(fp)

        matched_files = []
        for key, m in meta.items():
            if m["instrument_family_str"] not in families: continue
            if m["instrument_source_str"] not in sources: continue
            wav_path = os.path.join(split_dir, "audio", f"{key}.wav")
            matched_files.append(wav_path)

        if max_size is not None and max_size > 0:
            matched_files = matched_files[:max_size]

        if not matched_files:
            raise RuntimeError("No files matched the given filters!")

        # Initialize lists to store data
        self.signals = []
        self.pitches = []
        self.loudness = []
        self.file_indices = []  # To track where each file starts

        print(f"Preprocessing {len(matched_files)} files...")
        file_start_idx = 0

        for file_path in tqdm(matched_files):
            # Use librosa to load audio
            audio_np, sr = librosa.load(file_path, sr=sample_rate, mono=True)

            # Load with torchaudio for PESTO
            waveform, sr_torch = torchaudio.load(file_path)
            if waveform.shape[0] > 1:  # Convert stereo to mono
                waveform = waveform.mean(dim=0)

            # Resample if needed for PESTO
            if sr_torch != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr_torch, new_freq=sample_rate)
                waveform = resampler(waveform)

            print(f"Audio loaded: {len(audio_np)} samples")

            # Pad to multiple of segment_length
            pad_size = (segment_length - len(audio_np) % segment_length) % segment_length
            audio_np = np.pad(audio_np, (0, pad_size), 'constant')
            print(f"Audio after padding: {len(audio_np)} samples")

            # Extract loudness
            print("Extracting loudness features...")
            loudness = _extract_loudness(audio_np, sample_rate, hop_size)
            print(f"Loudness shape: {loudness.shape}")

            # Extract pitch using PESTO
            print("Extracting pitch with PESTO...")
            # Ensure waveform has a batch dimension for PESTO
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)

            timesteps, pitch, confidence, _ = pesto.predict(waveform, sample_rate)
            print(f"PESTO output - timesteps: {len(timesteps)}, pitch shape: {pitch.shape}")

            # PESTO returns [batch, time] shaped tensors
            pitch_np = pitch[0].numpy()  # Get first batch

            # Calculate expected number of frames
            expected_frames = len(audio_np) // hop_size

            # Create interpolation points
            orig_times = timesteps.numpy()
            target_times = np.arange(expected_frames) * hop_size / sample_rate

            # Resample pitch to match our frame rate
            resampled_pitch = np.interp(
                target_times,
                orig_times,
                pitch_np,
                left=pitch_np[0],
                right=pitch_np[-1]
            )

            # Convert to tensors
            audio_tensor = torch.from_numpy(audio_np.astype(np.float32))
            pitch_tensor = torch.from_numpy(resampled_pitch.astype(np.float32))
            loudness_tensor = torch.from_numpy(loudness.astype(np.float32))

            # Divide into segments and add to dataset
            num_segments = len(audio_np) // segment_length
            for i in range(num_segments):
                start_sample = i * segment_length
                end_sample = start_sample + segment_length

                # Audio segment
                audio_segment = audio_tensor[start_sample:end_sample]

                # Feature segments
                start_frame = start_sample // hop_size
                end_frame = start_frame + (segment_length // hop_size)

                pitch_segment = pitch_tensor[start_frame:end_frame]
                loudness_segment = loudness_tensor[start_frame:end_frame]

                self.signals.append(audio_segment)
                self.pitches.append(pitch_segment.unsqueeze(-1))  # [F, 1]
                self.loudness.append(loudness_segment.unsqueeze(-1))  # [F, 1]

            # Track file boundaries
            file_end_idx = len(self.signals)
            self.file_indices.append((file_start_idx, file_end_idx))
            file_start_idx = file_end_idx

        print(f"Dataset created with {len(self.signals)} segments from {len(matched_files)} files")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return (
            self.signals[idx],
            self.pitches[idx],
            self.loudness[idx]
        )