import random
import numpy as np
import matplotlib.pyplot as plt
from preprocess import NsynthDataset
from paths import NSYNTH_PREPROCESSED_DIR

SAMPLE_RATE = 16_000
HOP_SIZE = 256  # must match preprocess


def visualize(
        pre_dir: str,
        split: str = "test",
        families=None,
        sources=None,
        random_sample: bool = True
):
    # Load dataset with both pitch modes
    ds_meta = NsynthDataset(pre_dir,
                            split=split,
                            pitch_mode="meta",
                            families=families,
                            sources=sources)

    ds_fcnf0 = NsynthDataset(pre_dir,
                             split=split,
                             pitch_mode="fcnf0",
                             families=families,
                             sources=sources)

    print(f"Dataset has {len(ds_fcnf0)} items")

    # Make sure we use the same random index for both datasets
    idx = random.randint(0, len(ds_fcnf0) - 1) if random_sample else 0

    # Get data from both datasets
    audio_meta, pitch_meta, loud_meta = ds_meta[idx]
    audio_fcnf0, pitch_fcnf0, loud_fcnf0 = ds_fcnf0[idx]

    # Extract numpy arrays
    pitch_meta = pitch_meta.squeeze().numpy()  # (F_p,)
    pitch_fcnf0 = pitch_fcnf0.squeeze().numpy()  # (F_p,)
    loud = loud_meta.squeeze().numpy()  # (F_l,)
    audio = audio_meta.numpy()

    # Determine common length for time alignment
    Fp_meta = len(pitch_meta)
    Fp_fcnf0 = len(pitch_fcnf0)
    Fl = len(loud)

    F = min(Fp_meta, Fp_fcnf0, Fl)
    pitch_meta = pitch_meta[:F]
    pitch_fcnf0 = pitch_fcnf0[:F]
    loud = loud[:F]

    t_audio = np.arange(len(audio)) / SAMPLE_RATE
    t_feature = np.arange(F) * HOP_SIZE / SAMPLE_RATE

    plt.figure(figsize=(13, 9))
    plt.subplot(3, 1, 1)
    plt.plot(t_audio, audio)
    plt.title("Waveform")
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(t_feature, pitch_meta, label="meta", color="blue")
    plt.plot(t_feature, pitch_fcnf0, label="fcnf0", color="red")
    plt.title("Pitch Comparison (Hz)")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(t_feature, loud)
    plt.title("Loudness (norm)")
    plt.grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize(NSYNTH_PREPROCESSED_DIR,
              split="test",
              families=["guitar"],
              sources=["acoustic"],
              random_sample=True)