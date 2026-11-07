"""
Step 4: make LIVE forecasts with the saved model.

Run:  python -m src.predict            (all cities)
      python -m src.predict delhi      (one city)

Both the API (api/main.py) and the dashboard (app/dashboard.py) call
`forecast_city()` from here, so there is only one place where the
forecasting logic lives.
"""
import sys
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.aqi import category
from src.config import CITIES, HORIZONS, MODEL_BUNDLE
from src.features import build_features, clean
from src.fetch_data import fetch_live_daily

# Plain-English names for the features most likely to show up as "drivers"
FRIENDLY = {
    "aqi_today": "Today's AQI",
    "aqi_1d_ago": "Yesterday's AQI",
    "aqi_2d_ago": "AQI 2 days ago",
    "aqi_3d_ago": "AQI 3 days ago",
    "aqi_6d_ago": "AQI a week ago",
    "aqi_13d_ago": "AQI 2 weeks ago",
    "aqi_change_1d": "Change since yesterday",
    "humidity_today": "Today's humidity",
    "wind_speed_today": "Today's wind speed",
    "temperature_today": "Today's temperature",
    "ozone_today": "Today's ozone",
    "nitrogen_dioxide_today": "Today's NO₂",
    "carbon_monoxide_today": "Today's CO",
    "aqi_mean_3d": "3-day average AQI",
    "aqi_mean_7d": "7-day average AQI",
    "aqi_mean_14d": "14-day average AQI",
    "aqi_mean_30d": "30-day average AQI",
    "aqi_max_7d": "Worst AQI this week",
    "aqi_std_7d": "How jumpy AQI was this week",
    "pm2_5_today": "Today's PM2.5",
    "pm10_today": "Today's PM10",
    "pm_ratio_today": "Fine-vs-coarse dust ratio",
    "rain_target_day": "Forecast rain",
    "rain_change": "More/less rain than today",
    "rain_sum_7d": "Rain over the last week",
    "wind_speed_target_day": "Forecast wind speed",
    "wind_change": "Wind picking up / dropping",
    "humidity_target_day": "Forecast humidity",
    "temperature_target_day": "Forecast temperature",
    "pressure_target_day": "Forecast air pressure",
    "days_from_diwali": "Days from Diwali",
    "is_diwali_window": "Diwali (±2 days)",
    "target_month": "Month",
    "target_doy_sin": "Time of year",
    "target_doy_cos": "Time of year",
}


def friendly(feature: str) -> str:
    return FRIENDLY.get(feature, feature.replace("_", " ").capitalize())


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    """Load the trained models once and keep them in memory."""
    if not MODEL_BUNDLE.exists():
        raise FileNotFoundError("No trained model found. Run: python -m src.train")
    return joblib.load(MODEL_BUNDLE)


@lru_cache(maxsize=1)
def _explainer():
    import shap
    point = load_bundle()["horizons"][1]["point"]
    return shap.TreeExplainer(point.model)


def explain(X: pd.DataFrame, top: int = 5) -> list[dict]:
    """Which features pushed tomorrow's forecast up or down, and by how much."""
    try:
        values = _explainer().shap_values(X)[0]
    except Exception:          # e.g. a linear model was chosen — skip explanations
        return []
    order = np.argsort(-np.abs(values))[:top]
    return [{"feature": X.columns[i], "label": friendly(X.columns[i]),
             "value": None if pd.isna(X.iloc[0, i]) else round(float(X.iloc[0, i]), 2),
             "effect": round(float(values[i]), 1)} for i in order]


def forecast_city(city_id: str, live_daily: pd.DataFrame | None = None) -> dict:
    """Forecast the next 1–3 days for one city.

    `live_daily` can be passed in (tests do this); otherwise it is downloaded.
    """
    if city_id not in CITIES:
        raise KeyError(city_id)
    bundle = load_bundle()
    daily = live_daily if live_daily is not None else fetch_live_daily(city_id)
    daily = clean(daily, detect_glitches=False)

    observed = daily.dropna(subset=["aqi"])
    origin = observed["date"].max()                # last complete day with data
    last = observed[observed["date"] == origin].iloc[0]

    forecasts, drivers = [], []
    for h in HORIZONS:
        features = build_features(daily, h)
        X = features.loc[features["date"] == origin, bundle["features"]]
        models = bundle["horizons"][h]
        point = float(models["point"].predict(X)[0])
        low, high = sorted([float(models["q10"].predict(X)[0]), float(models["q90"].predict(X)[0])])
        low, high = min(low, point), max(high, point)   # the interval always contains the forecast
        forecasts.append({
            "date": f"{origin + pd.Timedelta(days=h):%Y-%m-%d}",
            "days_ahead": h,
            "aqi": round(point),
            "low": round(low),
            "high": round(high),
            "category": category(point),
        })
        if h == 1:
            drivers = explain(X)

    history = observed.tail(30)
    return {
        "city": city_id,
        "city_name": CITIES[city_id]["name"],
        "last_observed": {"date": f"{origin:%Y-%m-%d}", "aqi": int(last["aqi"]),
                          "category": last["aqi_category"],
                          "pm2_5": round(float(last["pm2_5"]), 1), "pm10": round(float(last["pm10"]), 1)},
        "forecast": forecasts,
        "drivers_tomorrow": drivers,
        "history": [{"date": f"{d:%Y-%m-%d}", "aqi": int(a)}
                    for d, a in zip(history["date"], history["aqi"])],
        "model": {"name": bundle["model_name"], "trained_at": bundle["trained_at"]},
    }


def main() -> None:
    cities = sys.argv[1:] or list(CITIES)
    for city in cities:
        result = forecast_city(city)
        lo = result["last_observed"]
        print(f"\n{result['city_name']}: last observed {lo['date']} AQI {lo['aqi']} ({lo['category']})")
        for f in result["forecast"]:
            print(f"   {f['date']}  AQI {f['aqi']:>3}  ({f['low']}–{f['high']})  {f['category']}")
        if result["drivers_tomorrow"]:
            top = result["drivers_tomorrow"][0]
            print(f"   biggest driver: {top['label']} ({top['effect']:+.1f} AQI)")


if __name__ == "__main__":
    main()
