"""
The most important tests in the project: they prove the features never peek
into the future (no data leakage).
"""
import numpy as np
import pandas as pd
import pytest

from src.features import TARGET_WEATHER, build_features, clean, feature_columns, find_glitch_dates

# Features that are ALLOWED to describe the target day, because they are known
# in advance: the calendar and the weather forecast.
KNOWN_IN_ADVANCE = ("target_", "_target_day", "days_from_diwali", "is_diwali_window",
                    "rain_change", "wind_change")


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_target_is_aqi_h_days_later(daily, horizon):
    f = build_features(daily, horizon)
    row = f[(f["city"] == "delhi")].iloc[10]
    later = daily[(daily["city"] == "delhi") & (daily["date"] == row["date"] + pd.Timedelta(days=horizon))]
    assert row["target"] == later["aqi"].iloc[0]


@pytest.mark.parametrize("horizon", [1, 3])
def test_no_leakage_from_the_future(daily, horizon):
    """Scramble everything AFTER day t. Features for day t (other than the
    allowed forecast/calendar ones) must not change at all."""
    cut = daily["date"].min() + pd.Timedelta(days=60)
    before = build_features(daily, horizon)

    scrambled = daily.copy()
    future = scrambled["date"] > cut
    numeric = scrambled.select_dtypes("number").columns
    scrambled.loc[future, numeric] = scrambled.loc[future, numeric] * 7 + 3
    after = build_features(scrambled, horizon)

    past_only = [c for c in feature_columns(before) if not any(k in c for k in KNOWN_IN_ADVANCE)]
    mask = before["date"] <= cut
    pd.testing.assert_frame_equal(before.loc[mask, past_only], after.loc[mask, past_only])


def test_target_day_weather_is_shifted(daily):
    f = build_features(daily, 2)
    row = f[f["city"] == "belagavi"].iloc[5]
    target_day = daily[(daily["city"] == "belagavi") & (daily["date"] == row["target_date"])].iloc[0]
    for name, col in TARGET_WEATHER.items():
        assert row[f"{name}_target_day"] == pytest.approx(target_day[col])


def test_glitch_detection_and_cleaning():
    from tests.conftest import make_daily
    d = make_daily(cities=("belagavi", "kolhapur", "pune", "mumbai", "bengaluru", "delhi"))
    glitch_day = d["date"].min() + pd.Timedelta(days=30)
    d.loc[d["date"] == glitch_day, ["pm2_5", "pm10"]] = 0.5
    assert find_glitch_dates(d) == [glitch_day]
    cleaned = clean(d)
    assert (cleaned.loc[cleaned["date"] == glitch_day, "pm2_5"] > 2).all()   # filled from neighbours


def test_missing_days_are_added_and_short_gaps_filled(daily):
    d = daily[~daily["date"].isin(pd.date_range("2025-02-01", periods=2))]
    cleaned = clean(d, detect_glitches=False)
    assert len(cleaned) == len(daily)
    assert not cleaned["aqi"].isna().any()


def test_city_one_hot(daily):
    f = build_features(daily, 1)
    assert (f.loc[f["city"] == "delhi", "city_delhi"] == 1).all()
    assert (f.loc[f["city"] == "delhi", "city_belagavi"] == 0).all()
    assert np.isfinite(f["target_doy_sin"]).all()
