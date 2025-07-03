import pytest
import joblib
import pandas as pd
from pathlib import Path

# Path to the serialized model pipeline
MODEL_PATH = Path("model/weather_pipeline.pkl")

@pytest.mark.order(1)
def test_model_file_exists():
    """Ensure the trained model file exists."""
    assert MODEL_PATH.exists(), f"Model file not found at {MODEL_PATH}"

@pytest.mark.order(2)
def test_model_can_predict():
    """Check if the model can make a prediction on valid input."""
    model = joblib.load(MODEL_PATH)
    test_input = pd.DataFrame([{
        "timestamp": "2025-06-03T00:00:00",  # Valid ISO 8601 format
        "temperature": 25,
        "humidity": 80,
        "pressure": 1012,
        "wind_speed": 12
    }])
    prediction = model.predict(test_input)
    assert prediction.shape == (1,), "Model should return a single prediction"

@pytest.mark.order(3)
def test_prediction_reasonable_range():
    """Ensure the predicted value is within a realistic temperature range."""
    model = joblib.load(MODEL_PATH)
    
    test_input = pd.DataFrame([{
        "timestamp": "2025-07-03T00:00:00",  # Ensure no non-breaking hyphen here
        "temperature": 30,
        "humidity": 60,
        "pressure": 1008,
        "wind_speed": 10
    }])
    
    # Optionally cleaning timestamp string
    test_input["timestamp"] = test_input["timestamp"].astype(str).str.replace(r"[^\x00-\x7F]", "-", regex=True)

    prediction = model.predict(test_input)[0]
    
    # Temperature in Celsius
    assert -50 <= prediction <= 60, f"Prediction {prediction} outside expected range"
