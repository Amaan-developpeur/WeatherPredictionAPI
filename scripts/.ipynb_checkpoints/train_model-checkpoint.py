# scripts/train_model.py

import sys
import logging
import joblib
import pandas as pd
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# --- Setup project root in path (for app.model import) ---
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app.model import FeatureBuilder  

# --- Paths ---
DATA_DIR   = ROOT_DIR / "data"
LOG_DIR    = ROOT_DIR / "logs"
MODEL_DIR  = ROOT_DIR / "model"
LOG_FILE   = LOG_DIR / "train_model.log"
DATA_FILE  = DATA_DIR / "DATA_FRAME.csv"
MODEL_PATH = MODEL_DIR / "weather_pipeline.pkl"

# --- Ensure required directories exist ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    filename=str(LOG_FILE),
    filemode="a",
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Starting train_model.py")

# --- Load Data ---
try:
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    logger.info("Data loaded successfully with shape %s", df.shape)
except Exception as e:
    logger.exception("Failed to load data")
    raise

# --- Prepare Feature Matrix and Target ---
raw_cols   = ["timestamp", "temperature", "humidity", "pressure", "wind_speed"]
target_col = "temp_next_hour"
split_idx  = int(len(df) * 0.8)

X_train_raw = df[raw_cols].iloc[:split_idx]
X_test_raw  = df[raw_cols].iloc[split_idx:]
y_train     = df[target_col].iloc[:split_idx]
y_test      = df[target_col].iloc[split_idx:]

# --- Pipeline ---
pipeline = Pipeline([
    ("feat", FeatureBuilder()),
    ("model", GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])

# --- Train ---
pipeline.fit(X_train_raw, y_train)
logger.info("Model pipeline trained successfully")

# --- Predict ---
y_train_pred = pipeline.predict(X_train_raw)
y_test_pred  = pipeline.predict(X_test_raw)

# --- Metrics ---
def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)

train_metrics = {
    "MAE":  mean_absolute_error(y_train, y_train_pred),
    "RMSE": rmse(y_train, y_train_pred),
    "R2":   r2_score(y_train, y_train_pred)
}
test_metrics = {
    "MAE":  mean_absolute_error(y_test, y_test_pred),
    "RMSE": rmse(y_test, y_test_pred),
    "R2":   r2_score(y_test, y_test_pred)
}
cv_score = cross_val_score(pipeline, X_train_raw, y_train, cv=5, scoring="r2").mean()

# --- Output ---
print("Train:", train_metrics)
print("Test :", test_metrics)
print("CV R²:", round(cv_score, 5))

logger.info("Train Metrics: %s", train_metrics)
logger.info("Test Metrics: %s", test_metrics)
logger.info("Cross-validated R²: %.5f", cv_score)

# --- Save Model ---
joblib.dump(pipeline, MODEL_PATH)
logger.info("Pipeline saved to %s", MODEL_PATH)
