# tests/test_data_validation.py

import pandas as pd
from pathlib import Path

def test_data_file_exists():
    data_path = Path("data/DATA_FRAME.csv")
    assert data_path.exists(), "Engineered data file does not exist"

def test_data_has_required_columns():
    df = pd.read_csv("data/DATA_FRAME.csv")
    required_cols = ["timestamp", "temperature", "hour", "day", "month", "weekday", "temp_next_hour"]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

def test_no_null_values():
    df = pd.read_csv("data/DATA_FRAME.csv")
    assert df.isnull().sum().sum() == 0, "Data contains null values"

def test_temperature_ranges():
    df = pd.read_csv("data/DATA_FRAME.csv")
    assert df["temperature"].between(-50, 60).all(), "Temperature values out of range"
    assert df["temp_next_hour"].between(-50, 60).all(), "Target temperature out of range"
