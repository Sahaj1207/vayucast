"""
Step 1 of the pipeline: download air-quality + weather history for each city.

Run it from the project folder:
    python -m src.fetch_data

What it does:
  1. Calls two free Open-Meteo APIs (no API key needed):
       - Air Quality API  -> PM2.5, PM10, NO2, SO2, O3, CO (hourly)
       - Weather Archive  -> temperature, humidity, rain, wind, pressure (hourly)
  2. Joins them on the timestamp.
  3. Saves one hourly CSV per city in data/raw/.
  4. Builds data/processed/daily.csv: one row per city per day, with the
     Indian AQI calculated from the daily averages.
"""
import time
from datetime import date, timedelta

import pandas as pd
import requests

from src.aqi import add_aqi_columns
from src.config import (AIR_VARIABLES, CITIES, PROCESSED_DIR, RAW_DIR, START_DATE,
                    TIMEZONE, WEATHER_VARIABLES)

AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
# The archive lags a few days behind. For the most recent days we use the
# forecast API, which also returns the past few months of observed weather.
RECENT_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


def get_json(url: str, params: dict, retries: int = 3) -> dict:
    """Call an API and return its JSON. Retries a few times if the network blips."""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()   # raises an error for 4xx/5xx replies
            return response.json()
        except requests.RequestException as error:
            if attempt == retries:
                raise
            wait = 5 * attempt
            print(f"   request failed ({error}); retrying in {wait}s...")
            time.sleep(wait)


def fetch_hourly(url: str, lat: float, lon: float, variables: list[str],
                 start: str | None = None, end: str | None = None,
                 past_days: int | None = None, forecast_days: int = 1) -> pd.DataFrame:
    """Download hourly data and turn the JSON into a pandas table.

    Give either a date range (start + end) or `past_days` for "the last N days"
    (plus `forecast_days` of future, counting today).
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(variables),
        "timezone": TIMEZONE,
    }
    if past_days is not None:
        params.update(past_days=past_days, forecast_days=forecast_days)
    else:
        params.update(start_date=start, end_date=end)
    data = get_json(url, params)["hourly"]
    df = pd.DataFrame(data)                 # each key becomes a column
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_city(city_id: str, info: dict, end: str) -> pd.DataFrame:
    print(f"-> {info['name']}")
    air = fetch_hourly(AIR_URL, info["lat"], info["lon"], AIR_VARIABLES, START_DATE, end)
    weather = fetch_hourly(WEATHER_URL, info["lat"], info["lon"], WEATHER_VARIABLES, START_DATE, end)

    # "inner" join keeps only hours that exist in both tables
    hourly = air.merge(weather, on="time", how="inner")
    hourly.insert(0, "city", city_id)

    path = RAW_DIR / f"{city_id}_hourly.csv"
    hourly.to_csv(path, index=False)
    print(f"   saved {len(hourly):,} hourly rows -> {path.relative_to(RAW_DIR.parent.parent)}")
    return hourly


def fetch_live_daily(city_id: str, past_days: int = 60, future_days: int = 4) -> pd.DataFrame:
    """Recent history + upcoming weather forecast for one city (for live forecasts).

    - Pollution: only COMPLETE past days. Today is dropped because it isn't
      over yet (part of it would be the data provider's own forecast).
    - Weather: past days AND the next few days of weather FORECAST — the
      model uses the target day's forecast weather as a feature.
    Future rows therefore have weather but empty pollution/AQI columns.
    """
    info = CITIES[city_id]
    today_in_india = pd.Timestamp.now(tz=TIMEZONE).date()   # correct even on a UTC server

    air = fetch_hourly(AIR_URL, info["lat"], info["lon"], AIR_VARIABLES, past_days=past_days)
    air = air[air["time"].dt.date < today_in_india].copy()
    air["city"] = city_id
    air_daily = to_daily(air, variables=AIR_VARIABLES)

    weather = fetch_hourly(RECENT_WEATHER_URL, info["lat"], info["lon"], WEATHER_VARIABLES,
                           past_days=past_days, forecast_days=future_days)
    weather["date"] = pd.to_datetime(weather["time"].dt.date)
    rules = {col: "mean" for col in WEATHER_VARIABLES}
    rules["precipitation"] = "sum"
    weather_daily = weather.groupby("date").agg(rules).round(2).reset_index()

    daily = weather_daily.merge(air_daily.drop(columns="city"), on="date", how="left")
    daily.insert(0, "city", city_id)
    daily = daily[daily["date"] >= air_daily["date"].min()]
    return add_aqi_columns(daily)


def to_daily(hourly: pd.DataFrame, variables: list[str] | None = None) -> pd.DataFrame:
    """Turn 24 hourly rows into 1 daily row (official AQI uses 24-hour averages)."""
    hourly = hourly.copy()
    hourly["date"] = hourly["time"].dt.date

    # How to combine each column: most are averaged, rain is added up.
    variables = variables or AIR_VARIABLES + WEATHER_VARIABLES
    rules = {col: "mean" for col in variables}
    if "precipitation" in rules:
        rules["precipitation"] = "sum"

    # Count how many hours actually had a PM2.5 reading (True counts as 1)
    hourly["hours_recorded"] = hourly["pm2_5"].notna()
    rules["hours_recorded"] = "sum"

    daily = hourly.groupby(["city", "date"]).agg(rules).reset_index()

    # Only trust days with at least 18 of 24 hours of PM data (CPCB uses a similar rule)
    daily = daily[daily["hours_recorded"] >= 18].drop(columns="hours_recorded")
    daily = daily.round(2)
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # The weather archive lags a few days behind today, so stop a week back.
    end = (date.today() - timedelta(days=7)).isoformat()
    print(f"Downloading {START_DATE} -> {end} for {len(CITIES)} cities\n")

    all_daily = []
    for city_id, info in CITIES.items():
        hourly = fetch_city(city_id, info, end)
        all_daily.append(to_daily(hourly))
        time.sleep(1)   # be polite to the free API

    daily = pd.concat(all_daily, ignore_index=True)
    daily = add_aqi_columns(daily)

    path = PROCESSED_DIR / "daily.csv"
    daily.to_csv(path, index=False)
    print(f"\nDone. {len(daily):,} daily rows saved -> data/processed/daily.csv")
    print(daily.groupby("city")["aqi"].describe()[["count", "mean", "min", "max"]].round(1))


if __name__ == "__main__":
    main()
