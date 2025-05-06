from pathlib import Path

ROOT_DIR   = Path(__file__).resolve().parent
DATA_DIR   = ROOT_DIR / "data"

NSYNTH_DIR = DATA_DIR  / "nsynth"
NSYNTH_PREPROCESSED_DIR = DATA_DIR / "nsynth-preprocessed"

MODELS_CKP_DIR = ROOT_DIR / "experiments" / "autoencoder" / "models"
DDSP_METAF0 = MODELS_CKP_DIR / "ddsp_metaf0.pth"
DDSP_FCNF0 = MODELS_CKP_DIR / "ddsp_fcnf0.pth"
SUPERVISED = MODELS_CKP_DIR / "supervised.pth"