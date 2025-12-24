import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "smart_credit_model.joblib"
META_PATH = BASE_DIR / "smart_credit_metadata.joblib"

model = joblib.load(MODEL_PATH)
metadata = joblib.load(META_PATH)

if __name__ == "__main__":
    print(BASE_DIR)