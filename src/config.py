"""
Project settings in one place.

Beginner tip: keeping settings in one file means you change them once,
instead of hunting through every script.
"""
from pathlib import Path

# Folder paths (Path works the same on Windows, Mac and Linux)
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PREDICTIONS_DIR = ROOT / "data" / "predictions"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_BUNDLE = MODELS_DIR / "vayucast_models.joblib"

# Forecast 1, 2 and 3 days ahead. One model per horizon ("direct" strategy).
HORIZONS = [1, 2, 3]

# The most recent 365 days are held back as the test set — the model never
# sees them while training or choosing settings.
TEST_DAYS = 365

# Cities we forecast. Latitude/longitude come from Google Maps.
# Delhi is included because its pollution swings a lot — great for learning.
CITIES = {
    "belagavi":  {"name": "Belagavi",  "lat": 15.8497, "lon": 74.4977},
    "kolhapur":  {"name": "Kolhapur",  "lat": 16.7050, "lon": 74.2433},
    "pune":      {"name": "Pune",      "lat": 18.5204, "lon": 73.8567},
    "mumbai":    {"name": "Mumbai",    "lat": 19.0760, "lon": 72.8777},
    "bengaluru": {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946},
    "hyderabad": {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
    "kurnool":   {"name": "Kurnool",   "lat": 15.8281, "lon": 78.0373},
    "lucknow":   {"name": "Lucknow",   "lat": 26.8467, "lon": 80.9462},
    "delhi":     {"name": "Delhi",     "lat": 28.6139, "lon": 77.2090},
}

# Historical air-quality data on Open-Meteo starts in August 2022.
START_DATE = "2022-08-01"

# Pollutants (from the Air Quality API) and weather (from the Archive API)
AIR_VARIABLES = ["pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide",
                 "ozone", "carbon_monoxide"]
WEATHER_VARIABLES = ["temperature_2m", "relative_humidity_2m", "precipitation",
                     "wind_speed_10m", "wind_direction_10m", "surface_pressure"]

TIMEZONE = "Asia/Kolkata"

# Inputs we watch for data drift (see src/monitor.py)
DRIFT_FEATURES = ["aqi_today", "pm2_5_today", "pm10_today", "temperature_today",
                  "humidity_today", "wind_speed_today", "rain_today", "pressure_today"]
