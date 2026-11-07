import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.monitor import psi
from src.models import DeltaModel, make_model


def test_psi_same_distribution_is_small():
    rng = np.random.default_rng(1)
    a, b = pd.Series(rng.normal(size=5000)), pd.Series(rng.normal(size=5000))
    assert psi(a, b) < 0.02


def test_psi_shifted_distribution_is_large():
    rng = np.random.default_rng(2)
    a, b = pd.Series(rng.normal(size=5000)), pd.Series(rng.normal(1.5, 1, size=5000))
    assert psi(a, b) > 0.25


def test_delta_model_adds_today_back_and_clips():
    X = pd.DataFrame({"aqi_today": [100.0, 490.0, 10.0], "x": [1.0, 2.0, 3.0]})
    y = X["aqi_today"] + 5
    m = make_model("ridge").fit(X, y)
    assert isinstance(m, DeltaModel)
    pred = m.predict(X)
    assert pred == pytest.approx([105, 495, 15], abs=1)
    assert (pred <= 500).all() and (pred >= 0).all()


# ---- API ------------------------------------------------------------------- #
FAKE = {
    "city": "delhi", "city_name": "Delhi",
    "last_observed": {"date": "2026-11-06", "aqi": 300, "category": "Very Poor", "pm2_5": 150, "pm10": 220},
    "forecast": [{"date": "2026-11-07", "days_ahead": 1, "aqi": 290, "low": 250, "high": 330,
                  "category": "Poor"}],
    "drivers_tomorrow": [{"feature": "aqi_today", "label": "Today's AQI", "value": 300.0, "effect": -5.0}],
    "history": [{"date": "2026-11-06", "aqi": 300}],
    "model": {"name": "xgboost", "trained_at": "2026-11-07"},
}


@pytest.fixture
def client(monkeypatch):
    import api.main as api_main
    monkeypatch.setattr(api_main, "forecast_city", lambda city: {**FAKE, "city": city})
    monkeypatch.setattr(api_main, "load_bundle", lambda: {"model_name": "xgboost", "trained_at": "x"})
    api_main._cache.clear()
    return TestClient(api_main.app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_cities(client):
    ids = [c["id"] for c in client.get("/cities").json()]
    assert "belagavi" in ids and "delhi" in ids


def test_forecast_ok(client):
    r = client.get("/forecast/Delhi")
    assert r.status_code == 200
    assert r.json()["forecast"][0]["aqi"] == 290


def test_unknown_city_is_404(client):
    assert client.get("/forecast/atlantis").status_code == 404
