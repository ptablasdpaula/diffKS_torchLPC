import random
import numpy as np
import matplotlib.pyplot as plt
from preprocess import NsynthDataset
from paths import NSYNTH_PREPROCESSED_DIR

SAMPLE_RATE = 16_000
HOP_SIZE = 256          # must match preprocess

def visualize(
        pre_dir: str,
        split: str = "test",
        families=None,
        sources=None,
        random_sample: bool = True
):
    ds = NsynthDataset(pre_dir, split=split,
                       families=families, sources=sources)
    print(f"Dataset has {len(ds)} items")

    idx = random.randint(0, len(ds)-1) if random_sample else 0
    audio, pitch, loud = ds[idx]

    pitch = pitch.squeeze().numpy()       # (F_p,)
    loud  = loud.squeeze().numpy()        # (F_l,)

    Fp, Fl = len(pitch), len(loud)
    F  = min(Fp, Fl)
    pitch, loud = pitch[:F], loud[:F]

    audio = audio.numpy()

    t_audio = np.arange(len(audio)) / SAMPLE_RATE
    t_feature = np.arange(F) * HOP_SIZE / SAMPLE_RATE

    plt.figure(figsize=(13,9))
    plt.subplot(3,1,1); plt.plot(t_audio, audio);  plt.title("Waveform");  plt.grid()
    plt.subplot(3,1,2); plt.plot(t_feature, pitch); plt.title("Pitch (Hz)"); plt.grid()
    plt.subplot(3,1,3); plt.plot(t_feature, loud);  plt.title("Loudness (norm)"); plt.grid()
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    visualize(NSYNTH_PREPROCESSED_DIR,
              split="test",
              families=["guitar"],
              sources=["acoustic"],
              random_sample=True)
