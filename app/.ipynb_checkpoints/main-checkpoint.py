#---- Importing Necessary Libararies---------------
from fastapi import FastAPI, HTTPException
from pathlib import Path
import requests
import pandas as pd
import logging
from app.model import predict_row

app = FastAPI(title="Weather Prediction API", version="1.0")

# --- Logging Setup -----------
ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "api.log",
    level=logging.INFO,
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --- Defining the  Constants ----
# --- As I've trained the model to predict about one city [Hyderrabad] ---
# --- This Project can be scaled by introducing more Cities ---
LAT = 17.3850
LON = 78.4867
URL = "https://api.open-meteo.com/v1/forecast"
PARAMS = {
    "latitude": LAT,
    "longitude": LON,
    "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m",
    "timezone": "auto",
    "forecast_days": 1
}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/predict-live")
def predict_live():
    try:
        response = requests.get(URL, params=PARAMS, timeout=10)
        response.raise_for_status()
        data = response.json()["hourly"]

        df = pd.DataFrame({
            "timestamp": data["time"],
            "temperature": data["temperature_2m"],
            "humidity": data["relative_humidity_2m"],
            "pressure": data["pressure_msl"],
            "wind_speed": data["wind_speed_10m"]
        })

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        next_row = df.iloc[0:1]  # just the first/recent hour

        prediction = predict_row(next_row.iloc[0].to_dict())
        logger.info("Predicted %.2f from live weather input", prediction)

        return {
            "predicted_temp": round(prediction, 2),
            "timestamp_used": str(next_row.iloc[0]['timestamp'])
        }

    except Exception as e:
        logger.exception("Live prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
