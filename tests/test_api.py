from fastapi.testclient import TestClient
import pytest
from app.main import app

client = TestClient(app)

#---- health check ----
def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

# --- predict‑live ----
def fake_open_meteo_ok(*_args, **_kwargs):
    """Mock Open‑Meteo response with one hourly row."""
    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {
                "hourly": {
                    "time":               ["2025-07-03T00:00:00"],  # ASCII hyphens
                    "temperature_2m":     [28],
                    "relative_humidity_2m":[70],
                    "pressure_msl":       [1010],
                    "wind_speed_10m":     [12],
                }
            }
    return _Resp()

def test_predict_live(monkeypatch):
    # Patch requests.get only for this test
    import app.main as main_mod
    monkeypatch.setattr(main_mod.requests, "get", fake_open_meteo_ok)

    resp = client.get("/predict-live")
    assert resp.status_code == 200

    body = resp.json()
    assert set(body) == {"predicted_temp", "timestamp_used"}
    assert isinstance(body["predicted_temp"], (int, float))
