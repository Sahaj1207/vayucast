"""
VayuCast REST API.

Run:   uvicorn api.main:app --reload
Docs:  http://127.0.0.1:8000/docs   (FastAPI builds interactive docs for free)

Endpoints
  GET /health              is the service up, which model is loaded
  GET /cities              list of supported cities
  GET /forecast/{city}     1–3 day AQI forecast with intervals and drivers
  GET /metrics             how accurate the model was on the test year
"""
import json
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import CITIES, REPORTS_DIR
from src.predict import forecast_city, load_bundle

app = FastAPI(
    title="VayuCast API",
    description="Air-quality (Indian AQI) forecasts for Indian cities, 1–3 days ahead.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])

# Forecasts change once a day, so keep results for 30 minutes instead of
# calling the weather APIs on every request.
CACHE_SECONDS = 30 * 60
_cache: dict[str, tuple[float, dict]] = {}


# Response models: FastAPI checks our output matches and documents it in /docs
class DayForecast(BaseModel):
    date: str
    days_ahead: int
    aqi: int
    low: int
    high: int
    category: str


class Driver(BaseModel):
    feature: str
    label: str
    value: float | None
    effect: float


class Forecast(BaseModel):
    city: str
    city_name: str
    last_observed: dict
    forecast: list[DayForecast]
    drivers_tomorrow: list[Driver]
    history: list[dict]
    model: dict


@app.get("/health")
def health() -> dict:
    bundle = load_bundle()
    return {"status": "ok", "model": bundle["model_name"], "trained_at": bundle["trained_at"]}


@app.get("/cities")
def cities() -> list[dict]:
    return [{"id": cid, **info} for cid, info in CITIES.items()]


@app.get("/forecast/{city}", response_model=Forecast)
def forecast(city: str) -> dict:
    city = city.lower()
    if city not in CITIES:
        raise HTTPException(404, f"Unknown city '{city}'. Try one of: {', '.join(CITIES)}")
    cached = _cache.get(city)
    if cached and time.time() - cached[0] < CACHE_SECONDS:
        return cached[1]
    try:
        result = forecast_city(city)
    except Exception as error:   # weather API down, etc.
        raise HTTPException(503, f"Could not build forecast right now: {error}") from error
    _cache[city] = (time.time(), result)
    return result


@app.get("/metrics")
def metrics() -> dict:
    path = REPORTS_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "No metrics yet. Run: python -m src.train")
    return json.loads(path.read_text())
