#---- Importing necessary libraries ----
import pandas as pd
import logging
from pathlib import Path

# --- Paths ------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"
DATA_FILE = DATA_DIR / "historical_weather_hyd.csv"
OUTPUT_FILE = DATA_DIR / "DATA_FRAME.csv"
LOG_FILE = LOG_DIR / "feature_engineer.log"

LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---- Logging Setup --------------
logging.basicConfig(
    filename=str(LOG_FILE),
    filemode="a",
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---- Loading the Data ----------
logger.info("Starting feature_engineer.py")
try:
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    logger.info("Loaded DataFrame shape: %s", df.shape)
except Exception as e:
    logger.exception("Failed to load CSV: %s", e)
    raise

# ------ Feature Engineering --------

logger.info("Creating time-based features...")
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["weekday"] = df["timestamp"].dt.weekday

logger.info("Created time-based features: %s", df.columns.tolist())

# ---- Target Varible -----------

df["temp_next_hour"] = df["temperature"].shift(-1)
initial_len = len(df)
df.dropna(subset=["temp_next_hour"], inplace=True)
logger.info("Dropped %d rows due to target shift (final shape: %s)", initial_len - len(df), df.shape)

# ---- Inspection of Duplicates -----

duplicates = df.duplicated().sum()
if duplicates > 0:
    logger.warning("DataFrame contains %d duplicate rows", duplicates)
else:
    logger.info("No duplicate rows detected")

# --- Saving the output DataFrame

df.to_csv(OUTPUT_FILE, index=False)
logger.info("Engineered DataFrame saved to %s", OUTPUT_FILE)
