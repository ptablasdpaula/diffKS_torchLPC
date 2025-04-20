import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = os.path.join(ROOT_DIR, "data")
NSYNTH_DIR = os.path.join(DATA_DIR, "nsynth")