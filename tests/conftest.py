import numpy as np
import pandas as pd
import pytest

from src.aqi import add_aqi_columns
from src.config import AIR_VARIABLES, WEATHER_VARIABLES


def make_daily(cities=("belagavi", "delhi"), days=120, seed=0) -> pd.DataFrame:
    """Small fake dataset with the same columns as data/processed/daily.csv."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    frames = []
    for i, city in enumerate(cities):
        df = pd.DataFrame({"city": city, "date": dates})
        for col in AIR_VARIABLES + WEATHER_VARIABLES:
            df[col] = rng.uniform(5, 100, size=days) * (1 + i)
        df["wind_direction_10m"] = rng.uniform(0, 360, size=days)
        frames.append(df)
    return add_aqi_columns(pd.concat(frames, ignore_index=True))


@pytest.fixture
def daily():
    return make_daily()
