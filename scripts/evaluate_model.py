
import sys
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score



ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))            # for app.model.FeatureBuilder


DATA_FILE  = ROOT_DIR / "data"  / "DATA_FRAME.csv"
MODEL_PATH = ROOT_DIR / "model" / "weather_pipeline.pkl"
LOG_DIR    = ROOT_DIR / "logs"
LOG_FILE   = LOG_DIR  / "evaluate_model.log"
LOG_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    filename=str(LOG_FILE),
    filemode="a",
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
logger.info("Starting evaluate_model.py")


try:
    model = joblib.load(MODEL_PATH)
    df    = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    logger.info("Model and data loaded successfully")
except Exception as e:
    logger.exception("Failed to load model or data")
    raise


raw_cols = ["timestamp", "temperature", "humidity", "pressure", "wind_speed"]
X = df[raw_cols]
y_true = df["temp_next_hour"]


y_pred = model.predict(X)

mae  = mean_absolute_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)

logger.info("Full‑dataset evaluation:")
logger.info("MAE  = %.4f", mae)
logger.info("RMSE = %.4f", rmse)
logger.info("R²   = %.4f", r2)


print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ---------- CI/CD guardrail --------------------------
if mae > 0.6:
    logger.error("MAE too high (%.4f) — model rejected", mae)
    raise SystemExit("Model performance unacceptable")
else:
    logger.info("Model evaluation passed")
