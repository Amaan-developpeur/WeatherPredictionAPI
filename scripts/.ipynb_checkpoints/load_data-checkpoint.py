#  Importing the necessary  libraries
import pandas as pd
import logging
from pathlib import Path


#---  Naming Necessary Project paths ---------

ROOT_DIR = Path(__file__).resolve().parents[1]      # weather-predictor/
LOG_DIR  = ROOT_DIR / "logs"
DATA_DIR = ROOT_DIR / "data"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE  = LOG_DIR / "load_data.log"
DATA_FILE = DATA_DIR / "historical_weather_hyd.csv"


#------  Logging setup-----------

logging.basicConfig(
    filename=str(LOG_FILE),
    filemode="a",
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


#----------  Load CSV created by fetch_and_save.py into DataFrame---------------

if not DATA_FILE.exists():
    logger.error("Data file %s not found — run fetch_and_save.py first", DATA_FILE)
    raise FileNotFoundError(DATA_FILE)

df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
logger.info("CSV loaded — shape: %s", df.shape)


#--------  Performing Necessary  validation-----------------

null_counts = df.isna().sum()
if null_counts.any():
    logger.warning("Null values detected:\n%s", null_counts[null_counts > 0])
else:
    logger.info("No null values detected")

# This is for quicK inspection on COnsole
if __name__ == "__main__":
    print("DataFrame shape :", df.shape)
    print("Null values per column:\n", null_counts)
