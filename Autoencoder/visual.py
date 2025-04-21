import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import json
import librosa
from preprocess import NsynthDataset, _extract_loudness, extract_pitch
from paths import NSYNTH_DIR


def visualize_nsynth_dataset():
    # Configuration - single sample, single batch
    config = {
        "sample_rate": 16000,
        "hop_size": 256,
        "segment_length": 16000 * 4,  # 4 seconds
        "batch_size": 1,
        "num_samples": 1,
        "split": "test",
        "families": ["guitar"],
        "sources": ["acoustic"],
    }

    # Create dataset with only one sample
    print("Creating NsynthDataset...")
    dataset = NsynthDataset(
        root_dir=NSYNTH_DIR,
        split=config["split"],
        families=config["families"],
        sources=config["sources"],
        sample_rate=config["sample_rate"],
        hop_size=config["hop_size"],
        segment_length=config["segment_length"],
        max_size=config["num_samples"]
    )

    # Print dataset properties
    print(f"\nDataset Properties:")
    print(f"Number of segments: {len(dataset)}")
    print(f"First audio shape: {dataset.signals[0].shape}")
    print(f"First pitch shape: {dataset.pitches[0].shape}")
    print(f"First loudness shape: {dataset.loudness[0].shape}")

    # Get the first sample directly
    audio, pitch, loudness = dataset[0]

    # Convert to numpy for plotting
    audio = audio.numpy()
    pitch = pitch.squeeze(-1).numpy()
    loudness = loudness.squeeze(-1).numpy()

    # Calculate time axes
    time_audio = np.arange(len(audio)) / config["sample_rate"]
    time_features = np.arange(len(pitch)) * config["hop_size"] / config["sample_rate"]

    # Create figure with aligned subplots
    plt.figure(figsize=(15, 10))

    # 1. Plot audio waveform
    plt.subplot(3, 1, 1)
    plt.plot(time_audio, audio)
    plt.title(f"Audio Waveform (length: {len(audio)} samples, {len(audio) / config['sample_rate']:.2f}s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # 2. Plot pitch
    plt.subplot(3, 1, 2)
    plt.plot(time_features, pitch)
    plt.title(
        f"Pitch (F0) (length: {len(pitch)} frames, {len(pitch) * config['hop_size'] / config['sample_rate']:.2f}s)")
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
    for ax in plt.gcf().axes:
        for t in np.arange(0, 4, 0.5):
            ax.axvline(x=t, color='r', linestyle='--', alpha=0.3)

    plt.tight_layout()

    # Print diagnostics
    print("\nData Diagnostics:")
    print(f"Audio shape: {audio.shape}")
    print(f"Pitch shape: {pitch.shape}")
    print(f"Loudness shape: {loudness.shape}")

    print(f"Audio min/max/mean: {audio.min():.4f}/{audio.max():.4f}/{audio.mean():.4f}")
    print(f"Pitch min/max/mean: {pitch.min():.4f}/{pitch.max():.4f}/{pitch.mean():.4f}")
    print(f"Loudness min/max/mean: {loudness.min():.4f}/{loudness.max():.4f}/{loudness.mean():.4f}")

    # Check expected vs. actual frame count
    frames_per_segment = config["segment_length"] // config["hop_size"]
    print(f"Expected frames for {config['segment_length'] / config['sample_rate']:.2f}s: {frames_per_segment}")
    print(f"Actual frames in dataset: {len(pitch)}")

    plt.savefig("nsynth_visualization.png")
    plt.show()

    # Create a zoomed view of a small time window
    plt.figure(figsize=(15, 8))
    window_start = 0.5  # seconds
    window_end = 1.0  # seconds

    # Convert to indices
    sample_start = int(window_start * config["sample_rate"])
    sample_end = int(window_end * config["sample_rate"])
    frame_start = int(window_start * config["sample_rate"] / config["hop_size"])
    frame_end = int(window_end * config["sample_rate"] / config["hop_size"])

    # Plot zoomed audio
    plt.subplot(3, 1, 1)
    plt.plot(time_audio[sample_start:sample_end], audio[sample_start:sample_end])
    plt.title(f"Zoomed Audio [{window_start}-{window_end}s]")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Plot zoomed pitch and loudness
    plt.subplot(3, 1, 2)
    plt.plot(time_features[frame_start:frame_end], pitch[frame_start:frame_end])
    plt.title(f"Zoomed Pitch [{window_start}-{window_end}s]")
    plt.ylabel("Frequency (Hz)")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time_features[frame_start:frame_end], loudness[frame_start:frame_end])
    plt.title(f"Zoomed Loudness [{window_start}-{window_end}s]")
    plt.ylabel("dB")
    plt.xlabel("Time (s)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("nsynth_zoomed_visualization.png")
    plt.show()


if __name__ == "__main__":
    visualize_nsynth_dataset()
