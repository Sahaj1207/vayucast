"""
Step 2 of the pipeline: clean the daily data and build features.

THE GOLDEN RULE OF FORECASTING
------------------------------
A row dated `t` may only use information known at the END of day t.
Its target is the AQI on day t + h (h = 1, 2 or 3 days later).
Using anything from the future is called *data leakage*: it makes the
model look great in testing and fail in real life.
`tests/test_features.py` checks this rule automatically.
"""
import numpy as np
import pandas as pd

from src.aqi import add_aqi_columns
from src.config import AIR_VARIABLES, CITIES

# Diwali is a known pollution spike (fireworks). The dates are public in
# advance, so using them for the target day is NOT leakage.
DIWALI_DATES = pd.to_datetime([
    "2022-10-24", "2023-11-12", "2024-10-31", "2025-10-20",
    "2026-11-08", "2027-10-29", "2028-10-17",
])

POLLUTANT_COLUMNS = AIR_VARIABLES  # pm2_5, pm10, no2, so2, ozone, co

# Weather we take from the forecast for the target day
TARGET_WEATHER = {
    "temperature": "temperature_2m",
    "humidity": "relative_humidity_2m",
    "rain": "precipitation",
    "wind_speed": "wind_speed_10m",
    "pressure": "surface_pressure",
}


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def find_glitch_dates(daily: pd.DataFrame, min_cities: int = 5,
                      pm25_threshold: float = 2.0) -> list:
    """Dates where (almost) every city reports near-zero PM2.5 at once.

    One clean city after heavy rain is real. Five cities across India all
    reading ~0 on the same day (including Delhi in winter) is a data-source
    glitch. Found during exploration (Lesson 1, Dec 14–15 2022).
    """
    near_zero = daily[daily["pm2_5"] < pm25_threshold]
    counts = near_zero.groupby("date")["city"].nunique()
    return sorted(counts[counts >= min_cities].index)


def clean(daily: pd.DataFrame, detect_glitches: bool = True) -> pd.DataFrame:
    """Fill gaps, remove glitches, recompute AQI.

    1. Give every city a row for every day (missing days become NaN).
    2. Blank out glitch days.
    3. Fill short gaps (up to 3 days) by straight-line interpolation.
       Longer gaps stay empty — guessing a whole week would be made-up data.
    """
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])

    glitches = find_glitch_dates(df) if detect_glitches else []
    df.loc[df["date"].isin(glitches), POLLUTANT_COLUMNS] = np.nan

    cleaned = []
    numeric = [c for c in df.columns if c not in ("city", "date", "aqi_category")]
    for city, group in df.groupby("city"):
        full_range = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        g = group.set_index("date").reindex(full_range)
        g[numeric] = g[numeric].interpolate(method="time", limit=3, limit_area="inside")
        g["city"] = city
        cleaned.append(g.rename_axis("date").reset_index())

    out = pd.concat(cleaned, ignore_index=True)
    out = out.drop(columns=["pm25_index", "pm10_index", "aqi", "aqi_category"], errors="ignore")
    return add_aqi_columns(out)


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def _origin_features(g: pd.DataFrame) -> pd.DataFrame:
    """Features describing one city's situation at the end of each day.

    `shift(k)` = the value k days earlier. `rolling(n)` = the last n days
    INCLUDING today, which is fine because today is already over.
    """
    f = pd.DataFrame(index=g.index)
    aqi = g["aqi"]

    # Recent AQI ("lags"): the strongest predictor of tomorrow is today
    f["aqi_today"] = aqi
    for k in (1, 2, 3, 6, 13):
        f[f"aqi_{k}d_ago"] = aqi.shift(k)

    # Recent trend and level
    f["aqi_change_1d"] = aqi - aqi.shift(1)
    for n in (3, 7, 14, 30):
        f[f"aqi_mean_{n}d"] = aqi.rolling(n, min_periods=max(2, n // 2)).mean()
    f["aqi_std_7d"] = aqi.rolling(7, min_periods=4).std()
    f["aqi_max_7d"] = aqi.rolling(7, min_periods=4).max()

    # Today's individual pollutants
    for col in POLLUTANT_COLUMNS:
        f[f"{col}_today"] = g[col]
    f["pm_ratio_today"] = g["pm2_5"] / (g["pm10"] + 1)   # fine vs coarse dust

    # Today's weather: rain washes particles out, wind blows them away,
    # humid and cold still air traps them
    f["temperature_today"] = g["temperature_2m"]
    f["humidity_today"] = g["relative_humidity_2m"]
    f["pressure_today"] = g["surface_pressure"]
    f["wind_speed_today"] = g["wind_speed_10m"]
    f["wind_speed_mean_3d"] = g["wind_speed_10m"].rolling(3, min_periods=2).mean()
    f["rain_today"] = g["precipitation"]
    f["rain_sum_7d"] = g["precipitation"].rolling(7, min_periods=4).sum()
    # Wind direction is an angle (359° is next to 1°), so use sin/cos
    radians = np.deg2rad(g["wind_direction_10m"])
    f["wind_dir_sin"] = np.sin(radians)
    f["wind_dir_cos"] = np.cos(radians)
    return f


def _calendar_features(dates: pd.Series) -> pd.DataFrame:
    """Calendar facts about the TARGET day — known in advance, so allowed."""
    dates = pd.to_datetime(dates)
    day_of_year = dates.dt.day_of_year
    c = pd.DataFrame(index=dates.index)
    c["target_month"] = dates.dt.month
    c["target_weekday"] = dates.dt.weekday          # 0 = Monday
    c["target_is_weekend"] = (c["target_weekday"] >= 5).astype(int)
    # Cyclical encoding: Dec 31 and Jan 1 should be "close"
    c["target_doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    c["target_doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    # Days from the nearest Diwali (negative = before), clipped to ±30
    diffs = (dates.values[:, None] - DIWALI_DATES.values[None, :]).astype("timedelta64[D]").astype(int)
    nearest = diffs[np.arange(len(diffs)), np.abs(diffs).argmin(axis=1)]
    c["days_from_diwali"] = np.clip(nearest, -30, 30)
    c["is_diwali_window"] = (np.abs(nearest) <= 2).astype(int)
    return c


def build_features(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """One row per (city, origin date) with features, plus target if known.

    Columns: city, date (origin), target_date, <features...>, target
    `target` is NaN for the last `horizon` days (the future isn't known yet).
    """
    frames = []
    for city, g in daily.sort_values("date").groupby("city"):
        g = g.set_index("date")
        f = _origin_features(g)

        # Weather ON THE TARGET DAY. In real life this comes from a weather
        # forecast (free from Open-Meteo), so it IS known when we predict.
        # For history we use the observed weather as a stand-in for the
        # forecast — slightly optimistic, and written in the README.
        for name, col in TARGET_WEATHER.items():
            f[f"{name}_target_day"] = g[col].shift(-horizon)
        target_radians = np.deg2rad(g["wind_direction_10m"].shift(-horizon))
        f["wind_dir_sin_target_day"] = np.sin(target_radians)
        f["wind_dir_cos_target_day"] = np.cos(target_radians)
        f["rain_change"] = f["rain_target_day"] - f["rain_today"]
        f["wind_change"] = f["wind_speed_target_day"] - f["wind_speed_today"]

        f["target"] = g["aqi"].shift(-horizon)          # AQI h days later
        f = f.reset_index()
        f.insert(0, "city", city)
        frames.append(f)

    df = pd.concat(frames, ignore_index=True)
    df["target_date"] = df["date"] + pd.Timedelta(days=horizon)
    df = pd.concat([df, _calendar_features(df["target_date"])], axis=1)

    # One-hot encode the city (the model can learn "Delhi is different")
    for city in CITIES:
        df[f"city_{city}"] = (df["city"] == city).astype(int)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Everything except identifiers and the answer."""
    return [c for c in df.columns if c not in ("city", "date", "target_date", "target")]
