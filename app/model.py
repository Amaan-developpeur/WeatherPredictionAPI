import pandas as pd
from pathlib import Path              
import joblib                         
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, Any

# ----- FeatureBuilder-------
class FeatureBuilder(BaseEstimator, TransformerMixin):
    base_cols  = ["timestamp", "temperature", "humidity",
                  "pressure", "wind_speed"]
    final_cols = ["temperature", "humidity", "pressure",
                  "wind_speed", "hour", "weekday", "month"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=self.base_cols).copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"]      = df["timestamp"].dt.hour
        df["weekday"]   = df["timestamp"].dt.weekday
        df["month"]     = df["timestamp"].dt.month
        return df[self.final_cols]


ROOT_DIR   = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "model" / "weather_pipeline.pkl"

_pipeline = None  #

def load_pipeline():
    """Load and cache the persisted scikit‑learn pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline

def predict_row(row: Dict[str, Any]) -> float:
    """
    Accept a raw JSON‑style dict and return the next‑hour temperature prediction.
    """
    pipe = load_pipeline()
    df = pd.DataFrame([row])
    return float(pipe.predict(df)[0])
