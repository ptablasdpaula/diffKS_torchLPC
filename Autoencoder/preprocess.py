import json, os, librosa, torch, pickle
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import pesto
import torchaudio

from paths import NSYNTH_DIR, NSYNTH_PREPROCESSED_DIR

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


def preprocess_nsynth(
        nsynth_dir,
        output_dir,
        splits=["train", "valid", "test"],
        families=["guitar"],
        sources=["acoustic"],
        sample_rate=16000,
        hop_size=256,
        segment_length=16000 * 4,
        max_files_per_split=None,
        normalize_loudness=True  # Option to normalize loudness during preprocessing
):
    """
    Preprocess NSynth dataset and save extracted features with loudness statistics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each split
    all_metadata = {}
    loudness_stats = {}

    for split in splits:
        print(f"\nProcessing {split} split...")
        split_dir = os.path.join(nsynth_dir, f"nsynth-{split}")

        # Load metadata
        meta_path = os.path.join(split_dir, "examples.json")
        with open(meta_path) as fp:
            meta = json.load(fp)

        # Find matching files
        matched_files = []
        matched_keys = []
        for key, m in meta.items():
            if m["instrument_family_str"] not in families: continue
            if m["instrument_source_str"] not in sources: continue
            wav_path = os.path.join(split_dir, "audio", f"{key}.wav")
            matched_files.append(wav_path)
            matched_keys.append(key)

        if max_files_per_split is not None and max_files_per_split > 0:
            matched_files = matched_files[:max_files_per_split]
            matched_keys = matched_keys[:max_files_per_split]

        if not matched_files:
            print(f"No files matched the filters for {split} split!")
            continue

        # Create split directory
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)
        os.makedirs(os.path.join(split_output_dir, "audio"), exist_ok=True)
        os.makedirs(os.path.join(split_output_dir, "pitch"), exist_ok=True)
        os.makedirs(os.path.join(split_output_dir, "confidence"), exist_ok=True)  # Store PESTO confidence
        os.makedirs(os.path.join(split_output_dir, "loudness"), exist_ok=True)

        # Process files
        split_metadata = {}

        # First pass: Extract loudness and calculate statistics
        print(f"First pass: Calculating loudness statistics for {len(matched_files)} files...")
        all_loudness_values = []

        for file_path in tqdm(matched_files):
            try:
                # Load audio
                audio_np, sr = librosa.load(file_path, sr=sample_rate, mono=True)

                # Extract loudness
                loudness = _extract_loudness(audio_np, sample_rate, hop_size)

                # Gather all loudness values for statistics
                all_loudness_values.append(loudness)

            except Exception as e:
                print(f"Error processing file {file_path} for loudness stats: {e}")

        # Calculate loudness statistics for this split
        all_loudness = np.concatenate(all_loudness_values)
        loudness_mean = float(np.mean(all_loudness))
        loudness_std = float(np.std(all_loudness))

        print(f"Loudness statistics for {split} split:")
        print(f"  Mean: {loudness_mean:.4f}")
        print(f"  Std:  {loudness_std:.4f}")

        loudness_stats[split] = {
            "mean": loudness_mean,
            "std": loudness_std
        }

        # Second pass: Process and save all features
        print(f"Second pass: Processing and saving {len(matched_files)} files...")
        for i, (file_path, key) in enumerate(tqdm(zip(matched_files, matched_keys))):
            try:
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

                # Pad to multiple of segment_length
                pad_size = (segment_length - len(audio_np) % segment_length) % segment_length
                audio_np = np.pad(audio_np, (0, pad_size), 'constant')

                # Extract loudness
                loudness = _extract_loudness(audio_np, sample_rate, hop_size)

                # Normalize loudness if requested
                if normalize_loudness:
                    loudness_normalized = (loudness - loudness_mean) / loudness_std
                else:
                    loudness_normalized = loudness

                # Extract pitch using PESTO with matched step size
                if len(waveform.shape) == 1:
                    waveform = waveform.unsqueeze(0)

                # Calculate step_size in milliseconds to match our hop_size
                step_size_ms = (hop_size / sample_rate) * 1000  # Convert to milliseconds

                # Get pitch predictions directly from PESTO with matching step size
                timesteps, pitch, confidence, activations = pesto.predict(
                    waveform,
                    sample_rate,
                    step_size=step_size_ms,  # This aligns PESTO frames with our desired frames
                    reduction="alwa",  # Use Argmax-Local Weighted Averaging (recommended)
                    convert_to_freq=True  # Get frequency in Hz
                )

                # PESTO returns [batch, time] shaped tensors
                pitch_np = pitch[0].numpy()  # Get first batch
                confidence_np = confidence[0].numpy()  # Get confidence values

                # Verify the frame count matches our expected frames
                expected_frames = len(audio_np) // hop_size
                if len(pitch_np) != expected_frames:
                    # If needed, perform minimal adjustment to match expected frame count
                    if len(pitch_np) > expected_frames:
                        # Trim extra frames
                        pitch_np = pitch_np[:expected_frames]
                        confidence_np = confidence_np[:expected_frames]
                    else:
                        # Pad with last frame if we have too few
                        padding = expected_frames - len(pitch_np)
                        pitch_np = np.pad(pitch_np, (0, padding), mode='edge')
                        confidence_np = np.pad(confidence_np, (0, padding), mode='edge')

                # Save processed data
                audio_path = os.path.join(split_output_dir, "audio", f"{key}.npy")
                pitch_path = os.path.join(split_output_dir, "pitch", f"{key}.npy")
                confidence_path = os.path.join(split_output_dir, "confidence", f"{key}.npy")
                loudness_path = os.path.join(split_output_dir, "loudness", f"{key}.npy")

                np.save(audio_path, audio_np)
                np.save(pitch_path, pitch_np)
                np.save(confidence_path, confidence_np)
                np.save(loudness_path, loudness_normalized if normalize_loudness else loudness)

                # Calculate segment information
                num_segments = len(audio_np) // segment_length
                segments = []

                for j in range(num_segments):
                    start_sample = j * segment_length
                    end_sample = start_sample + segment_length

                    start_frame = start_sample // hop_size
                    end_frame = start_frame + (segment_length // hop_size)

                    segments.append({
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                        "start_frame": start_frame,
                        "end_frame": end_frame
                    })

                # Store metadata
                split_metadata[key] = {
                    "audio_path": os.path.relpath(audio_path, output_dir),
                    "pitch_path": os.path.relpath(pitch_path, output_dir),
                    "confidence_path": os.path.relpath(confidence_path, output_dir),
                    "loudness_path": os.path.relpath(loudness_path, output_dir),
                    "num_samples": len(audio_np),
                    "num_frames": len(pitch_np),
                    "segments": segments,
                    "metadata": meta[key]  # Keep original metadata
                }

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

        # Save split metadata
        all_metadata[split] = split_metadata
        print(f"Processed {len(split_metadata)} files in {split} split")

    # Save global metadata
    metadata_path = os.path.join(output_dir, "metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(all_metadata, f)

    # Save config with loudness statistics
    config = {
        "sample_rate": sample_rate,
        "hop_size": hop_size,
        "segment_length": segment_length,
        "families": families,
        "sources": sources,
        "splits": splits,
        "loudness_stats": loudness_stats,
        "loudness_normalized": normalize_loudness
    }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nPreprocessing complete! Data saved to {output_dir}")
    return metadata_path


class NsynthDataset(Dataset):
    """
    NSynth dataset implementation that loads preprocessed data
    """

    def __init__(
            self,
            preprocessed_dir,
            split="test",
            families=None,  # Optional filtering
            sources=None,  # Optional filtering
            segment_length=None,  # Use None to use the original config
            max_items=None
    ):
        # Load metadata
        metadata_path = os.path.join(preprocessed_dir, "metadata.pkl")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        # Load config
        config_path = os.path.join(preprocessed_dir, "config.json")
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Use config values if not specified
        self.sr = self.config["sample_rate"]
        self.hop_size = self.config["hop_size"]
        self.segment_length = segment_length or self.config["segment_length"]

        # Check if split exists
        if split not in self.metadata:
            raise ValueError(f"Split '{split}' not found in preprocessed data")

        # Filter by family and source if specified
        filtered_keys = []

        for key, item in self.metadata[split].items():
            item_metadata = item["metadata"]

            if families and item_metadata["instrument_family_str"] not in families:
                continue

            if sources and item_metadata["instrument_source_str"] not in sources:
                continue

            filtered_keys.append(key)

        if max_items:
            filtered_keys = filtered_keys[:max_items]

        self.split = split
        self.base_dir = preprocessed_dir

        # Build segment index
        self.segments = []

        for key in filtered_keys:
            item = self.metadata[split][key]

            for i, segment in enumerate(item["segments"]):
                self.segments.append({
                    "key": key,
                    "segment_idx": i,
                    "start_sample": segment["start_sample"],
                    "end_sample": segment["end_sample"],
                    "start_frame": segment["start_frame"],
                    "end_frame": segment["end_frame"]
                })

        print(f"Dataset loaded with {len(self.segments)} segments from {len(filtered_keys)} files")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        key = segment["key"]

        # Get paths
        item = self.metadata[self.split][key]
        audio_path = os.path.join(self.base_dir, item["audio_path"])
        pitch_path = os.path.join(self.base_dir, item["pitch_path"])
        loudness_path = os.path.join(self.base_dir, item["loudness_path"])

        # Load data
        audio = np.load(audio_path, mmap_mode='r')
        pitch = np.load(pitch_path, mmap_mode='r')
        loudness = np.load(loudness_path, mmap_mode='r')

        # Extract segment
        audio_segment = audio[segment["start_sample"]:segment["end_sample"]]
        pitch_segment = pitch[segment["start_frame"]:segment["end_frame"]]
        loudness_segment = loudness[segment["start_frame"]:segment["end_frame"]]

        # Convert to tensors
        audio_tensor = torch.from_numpy(audio_segment.astype(np.float32))
        pitch_tensor = torch.from_numpy(pitch_segment.astype(np.float32))
        loudness_tensor = torch.from_numpy(loudness_segment.astype(np.float32))

        return (
            audio_tensor,
            pitch_tensor.unsqueeze(-1),  # [F, 1]
            loudness_tensor.unsqueeze(-1)  # [F, 1]
        )


# Example usage:
if __name__ == "__main__":
    # First preprocess the data
    preprocess_nsynth(
        nsynth_dir=NSYNTH_DIR,
        output_dir=NSYNTH_PREPROCESSED_DIR,
        splits=["test"],
        families=["guitar"],
        sources=["acoustic"],
        #max_files_per_split=10
    )

    # Then load the dataset
    dataset = NsynthDataset(
        preprocessed_dir=NSYNTH_PREPROCESSED_DIR,
        split="test"
    )

    # Get a sample
    audio, pitch, loudness = dataset[0]
    print(f"Audio shape: {audio.shape}")
    print(f"Pitch shape: {pitch.shape}")
    print(f"Loudness shape: {loudness.shape}")