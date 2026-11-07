"""
Indian National Air Quality Index (AQI), as defined by CPCB.

How it works (simplified):
  1. Each pollutant gets a "sub-index" from 0 to 500 using a lookup table.
  2. Inside each band, we scale linearly (straight-line interpolation).
  3. The overall AQI is the WORST (highest) sub-index.

Official CPCB AQI needs at least 3 pollutants including PM2.5 or PM10.
Here we use PM2.5 and PM10 only — they drive the AQI in most Indian cities.
This simplification is written in the README so readers know.
"""
import numpy as np
import pandas as pd

# (concentration_low, concentration_high, aqi_low, aqi_high) — 24-hour averages in µg/m³
PM25_BANDS = [(0, 30, 0, 50), (30, 60, 50, 100), (60, 90, 100, 200),
              (90, 120, 200, 300), (120, 250, 300, 400), (250, 380, 400, 500)]
PM10_BANDS = [(0, 50, 0, 50), (50, 100, 50, 100), (100, 250, 100, 200),
              (250, 350, 200, 300), (350, 430, 300, 400), (430, 510, 400, 500)]

CATEGORIES = [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"),
              (300, "Poor"), (400, "Very Poor"), (500, "Severe")]


def sub_index(concentration: float, bands: list[tuple]) -> float:
    """Convert one concentration into a 0–500 sub-index."""
    if pd.isna(concentration):
        return np.nan
    for c_lo, c_hi, i_lo, i_hi in bands:
        if concentration <= c_hi:
            # straight line between (c_lo, i_lo) and (c_hi, i_hi)
            return i_lo + (concentration - c_lo) * (i_hi - i_lo) / (c_hi - c_lo)
    return 500.0   # above the top band: cap at the maximum


def category(aqi: float) -> str:
    if pd.isna(aqi):
        return "Unknown"
    for upper, name in CATEGORIES:
        if aqi <= upper:
            return name
    return "Severe"


def add_aqi_columns(daily: pd.DataFrame) -> pd.DataFrame:
    """Add pm25_index, pm10_index, aqi and aqi_category columns to a daily table."""
    df = daily.copy()
    df["pm25_index"] = df["pm2_5"].apply(sub_index, bands=PM25_BANDS)
    df["pm10_index"] = df["pm10"].apply(sub_index, bands=PM10_BANDS)
    df["aqi"] = df[["pm25_index", "pm10_index"]].max(axis=1).round()
    df["aqi_category"] = df["aqi"].apply(category)
    return df


if __name__ == "__main__":
    # Quick self-check: run `python src/aqi.py`
    assert sub_index(0, PM25_BANDS) == 0
    assert sub_index(30, PM25_BANDS) == 50
    assert sub_index(45, PM25_BANDS) == 75        # halfway through 30–60 -> 75
    assert sub_index(500, PM25_BANDS) == 500       # capped
    assert category(42) == "Good" and category(250) == "Poor"
    print("All AQI checks passed.")
