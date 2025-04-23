import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import os
from preprocess import NsynthDataset
from paths import NSYNTH_DIR, NSYNTH_PREPROCESSED_DIR


def visualize_preprocessed_nsynth(preprocessed_dir, split="test", random_sample=True):
    """
    Visualize data from a preprocessed NSynth dataset

    Args:
        preprocessed_dir: Directory containing preprocessed data
        split: Split to visualize (train, valid, test)
        random_sample: If True, pick a random sample, otherwise use the first one
    """
    # Create dataset
    print(f"Loading NsynthDataset from {preprocessed_dir}...")
    dataset = NsynthDataset(
        preprocessed_dir=preprocessed_dir,
        split=split
    )

    # Print dataset properties
    print(f"\nDataset Properties:")
    print(f"Number of segments: {len(dataset)}")

    # Select a sample
    if random_sample and len(dataset) > 1:
        idx = random.randint(0, len(dataset) - 1)
        print(f"Visualizing random sample at index {idx}")
    else:
        idx = 0
        print("Visualizing first sample")

    # Get the sample
    audio, pitch, loudness = dataset[idx]

    # Get metadata to show in the title
    segment_info = dataset.segments[idx]
    file_key = segment_info["key"]
    segment_idx = segment_info["segment_idx"]

    # Get original metadata if available
    try:
        instrument_name = dataset.metadata[split][file_key]["metadata"]["instrument_str"]
        family_name = dataset.metadata[split][file_key]["metadata"]["instrument_family_str"]
        source_type = dataset.metadata[split][file_key]["metadata"]["instrument_source_str"]
        title_info = f"Instrument: {instrument_name} ({family_name}, {source_type})"
    except (KeyError, TypeError):
        title_info = f"File: {file_key}, Segment: {segment_idx}"

    # Get sample rate and hop size from dataset
    sample_rate = 16000
    hop_size = dataset.hop_size

    # Convert to numpy for plotting
    audio = audio.numpy()
    pitch = pitch.squeeze(-1).numpy()
    loudness = loudness.squeeze(-1).numpy()

    # Calculate time axes
    time_audio = np.arange(len(audio)) / sample_rate
    time_features = np.arange(len(pitch)) * hop_size / sample_rate

    # Create figure with aligned subplots
    plt.figure(figsize=(15, 10))

    # Main title with metadata
    plt.suptitle(title_info, fontsize=16)

    # 1. Plot audio waveform
    plt.subplot(3, 1, 1)
    plt.plot(time_audio, audio)
    plt.title(f"Audio Waveform (length: {len(audio)} samples, {len(audio) / sample_rate:.2f}s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # 2. Plot pitch
    plt.subplot(3, 1, 2)
    plt.plot(time_features, pitch)
    plt.title(
        f"Pitch (F0) (length: {len(pitch)} frames, {len(pitch) * hop_size / sample_rate:.2f}s)")
    plt.ylabel("Frequency (Hz)")
    plt.grid(True)

    # 3. Plot loudness
    plt.subplot(3, 1, 3)
    plt.plot(time_features, loudness)
    plt.title(f"Loudness (length: {len(loudness)} frames)")
    plt.ylabel("dB")
    plt.xlabel("Time (s)")
    plt.grid(True)

    # Add vertical alignment lines every 0.5 seconds
    duration = len(audio) / sample_rate
    for ax in plt.gcf().axes:
        for t in np.arange(0, duration, 0.5):
            ax.axvline(x=t, color='r', linestyle='--', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle

    # Print diagnostics
    print("\nData Diagnostics:")
    print(f"Audio shape: {audio.shape}")
    print(f"Pitch shape: {pitch.shape}")
    print(f"Loudness shape: {loudness.shape}")

    print(f"Audio min/max/mean: {audio.min():.4f}/{audio.max():.4f}/{audio.mean():.4f}")
    print(f"Pitch min/max/mean: {pitch.min():.4f}/{pitch.max():.4f}/{pitch.mean():.4f}")
    print(f"Loudness min/max/mean: {loudness.min():.4f}/{loudness.max():.4f}/{loudness.mean():.4f}")

    # Check expected vs. actual frame count
    segment_length = len(audio)
    frames_per_segment = segment_length // hop_size
    print(f"Expected frames for {segment_length / sample_rate:.2f}s: {frames_per_segment}")
    print(f"Actual frames in dataset: {len(pitch)}")

    plt.savefig("nsynth_visualization.png")
    plt.show()

    # Create a zoomed view of a small time window
    plt.figure(figsize=(15, 8))
    window_start = duration * 0.125  # 1/8 through the sample
    window_end = duration * 0.25  # 1/4 through the sample

    # Convert to indices
    sample_start = int(window_start * sample_rate)
    sample_end = int(window_end * sample_rate)
    frame_start = int(window_start * sample_rate / hop_size)
    frame_end = int(window_end * sample_rate / hop_size)

    # Plot zoomed audio
    plt.subplot(3, 1, 1)
    plt.plot(time_audio[sample_start:sample_end], audio[sample_start:sample_end])
    plt.title(f"Zoomed Audio [{window_start:.2f}-{window_end:.2f}s]")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Plot zoomed pitch and loudness
    plt.subplot(3, 1, 2)
    plt.plot(time_features[frame_start:frame_end], pitch[frame_start:frame_end])
    plt.title(f"Zoomed Pitch [{window_start:.2f}-{window_end:.2f}s]")
    plt.ylabel("Frequency (Hz)")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time_features[frame_start:frame_end], loudness[frame_start:frame_end])
    plt.title(f"Zoomed Loudness [{window_start:.2f}-{window_end:.2f}s]")
    plt.ylabel("dB")
    plt.xlabel("Time (s)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("nsynth_zoomed_visualization.png")
    plt.show()


if __name__ == "__main__":
    # Path to your preprocessed directory
    PREPROCESSED_DIR = NSYNTH_PREPROCESSED_DIR

    # Visualize a random sample from the test split
    visualize_preprocessed_nsynth(
        preprocessed_dir=PREPROCESSED_DIR,
        split="test",
        random_sample=True
    )