#  Importing the necessary libraries
import pandas as pd
import requests
import logging
from pathlib import Path
from datetime import datetime


#  Config / constants

latitude  = 17.3850 # Latitude of the city Hyderabad
longitude = 78.4867 # Longitude of the city Hyderabad
start_date = "2023-05-01"
end_date   = "2025-06-28" #--- Almost morethan 2 years of data will be fetched

ROOT_DIR = Path(__file__).resolve().parents[1]      # weather-predictor/
LOG_DIR  = ROOT_DIR / "logs"
DATA_DIR = ROOT_DIR / "data"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE   = LOG_DIR / "fetch_and_save.log"
OUTPUT_CSV = DATA_DIR / "historical_weather_hyd.csv"


#  Logging setup

logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)
logger.info("Starting fetch_and_save.py")
logger.info("Logger is working. This should appear at the top.")


#  API calling

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": start_date,
    "end_date":   end_date,
    "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m",
    "timezone": "auto",
}

try:
    logger.info(
        "Fetching weather data for Hyderabad (%.4f, %.4f) from %s to %s",
        latitude, longitude, start_date, end_date
    )

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json() 

    df = pd.DataFrame({
        "timestamp":   data["hourly"]["time"],
        "temperature": data["hourly"]["temperature_2m"],
        "humidity":    data["hourly"]["relative_humidity_2m"],
        "pressure":    data["hourly"]["pressure_msl"],
        "wind_speed":  data["hourly"]["wind_speed_10m"],
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    
    #  Saving to the CSV file
    
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info("Saved %d rows to %s", len(df), OUTPUT_CSV)
    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")

except Exception as e:
    logger.exception("Failed to fetch or save weather data")
    print("Error occurred:", str(e))
