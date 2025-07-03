
"""Fetch the latest hourly weather data from Open‑Meteo and append
it to data/live_weather.csv. No ML inference happens here.

This script is idempotent and safe to schedule (cron / apscheduler).
The FastAPI service will later consume live_weather.csv ⇢ predict.
"""

import sys
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
LOG_DIR  = ROOT_DIR / "logs"
CSV_FILE = DATA_DIR / "live_weather.csv"
LOG_FILE = LOG_DIR  / "live_collector.log"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    filename=str(LOG_FILE),
    filemode="a",
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("⏳ live_collector.py start")


LATITUDE  = 17.3850
LONGITUDE = 78.4867
API_URL   = "https://api.open-meteo.com/v1/forecast"
API_PARAMS = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m",
    "timezone": "auto",
    "forecast_days": 1
}


MAX_RETRIES = 3
BACKOFF     = 3   # seconds

for attempt in range(1, MAX_RETRIES + 1):
    try:
        r = requests.get(API_URL, params=API_PARAMS, timeout=8)
        r.raise_for_status()
        payload = r.json()
        break
    except Exception as exc:
        logger.warning("Attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
        if attempt == MAX_RETRIES:
            logger.error("All retries failed; aborting run.")
            sys.exit(1)
        time.sleep(BACKOFF ** attempt)


row = {
    "timestamp":  payload["hourly"]["time"][0],                 # first hour row
    "temperature": payload["hourly"]["temperature_2m"][0],
    "humidity":    payload["hourly"]["relative_humidity_2m"][0],
    "pressure":    payload["hourly"]["pressure_msl"][0],
    "wind_speed":  payload["hourly"]["wind_speed_10m"][0]
}
df_new = pd.DataFrame([row])
df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], utc=True)


header_needed = not CSV_FILE.exists()
df_new.to_csv(CSV_FILE, mode="a", header=header_needed, index=False)


df_all = pd.read_csv(CSV_FILE, parse_dates=["timestamp"])
before = len(df_all)
df_all.drop_duplicates(subset="timestamp", inplace=True)
after = len(df_all)
if after < before:
    df_all.to_csv(CSV_FILE, index=False)
    logger.info("Removed %d duplicate rows", before - after)

logger.info("Collected row for %s", df_new["timestamp"].iloc[0])
print(f"Collected row ----> {CSV_FILE}")
