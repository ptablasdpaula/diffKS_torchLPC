from pathlib import Path

ROOT_DIR   = Path(__file__).resolve().parent
DATA_DIR   = ROOT_DIR / "data"
NSYNTH_DIR = DATA_DIR  / "nsynth"

# existing constants stay as‑is if you want quick aliases
NSYNTH_TEST_DIR  = NSYNTH_DIR / "nsynth-test"
NSYNTH_TRAIN_DIR = NSYNTH_DIR / "nsynth-train"
NSYNTH_VALID_DIR = NSYNTH_DIR / "nsynth-valid"
NSYNTH_PREPROCESSED_DIR = DATA_DIR / "nsynth-preprocessed"