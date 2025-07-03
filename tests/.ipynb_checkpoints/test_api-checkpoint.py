# tests/test_api.py
import json
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

# Import the FastAPI app defined in app/main.py
from app.main import app, URL, PARAMS

client = TestClient(app)

# ---------- health check --------------------------------------------------
def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

# ---------- predict‑live --------------------------------------------------
def fake_open_meteo_ok(*args, **kwargs):
    """
    Dummy replacement for requests.get.
    Returns one hourly record so predict‑live can run end‑to‑end without
    hitting the real Open‑Meteo API.
    """
    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {
                "hourly": {
                    "time":               ["2025‑07‑03T00:00:00"],
                    "temperature_2m":     [28],
                    "relative_humidity_2m":[70],
                    "pressure_msl":       [1010],
                    "wind_speed_10m":     [12],
                }
            }
    return _Resp()

@pytest.mark.patch
def test_predict_live(monkeypatch):
    # patch requests.get only inside this test
    import app.main as main_mod
    monkeypatch.setattr(main_mod.requests, "get", fake_open_meteo_ok)

    resp = client.get("/predict-live")
    assert resp.status_code == 200

    body = resp.json()
    # ensure keys exist and numeric prediction returned
    assert "predicted_temp" in body and "timestamp_used" in body
    assert isinstance(body["predicted_temp"], (int, float))
