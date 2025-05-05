from pathlib import Path

ROOT_DIR   = Path(__file__).resolve().parent
DATA_DIR   = ROOT_DIR / "data"

NSYNTH_DIR = DATA_DIR  / "nsynth"
NSYNTH_PREPROCESSED_DIR = DATA_DIR / "nsynth-preprocessed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

AUTOENCODER_MODEL = ROOT_DIR / "experiments" / "autoencoder" / "best_model.pth"