from pathlib import Path
import joblib
import pandas as pd
from features import prepare_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "return_risk_pipeline.joblib"

def predict_return_risk(order: dict) -> dict:
    """Predict return probability for one order dictionary."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the model first: python src/train.py")
    model = joblib.load(MODEL_PATH)
    frame = pd.DataFrame([order])
    X = prepare_features(frame)
    probability = float(model.predict_proba(X)[0, 1])
    return {
        "return_probability": probability,
        "risk_level": "High" if probability >= 0.65 else "Medium" if probability >= 0.35 else "Low",
    }
