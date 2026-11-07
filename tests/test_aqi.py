import math

import pytest

from src.aqi import PM10_BANDS, PM25_BANDS, category, sub_index


@pytest.mark.parametrize("pm25, expected", [(0, 0), (30, 50), (45, 75), (60, 100), (90, 200),
                                            (120, 300), (250, 400), (1000, 500)])
def test_pm25_breakpoints(pm25, expected):
    assert sub_index(pm25, PM25_BANDS) == pytest.approx(expected)


def test_pm10_breakpoints():
    assert sub_index(100, PM10_BANDS) == pytest.approx(100)
    assert sub_index(250, PM10_BANDS) == pytest.approx(200)


def test_missing_value_stays_missing():
    assert math.isnan(sub_index(float("nan"), PM25_BANDS))


@pytest.mark.parametrize("aqi, name", [(0, "Good"), (50, "Good"), (51, "Satisfactory"),
                                       (150, "Moderate"), (250, "Poor"), (350, "Very Poor"),
                                       (480, "Severe")])
def test_categories(aqi, name):
    assert category(aqi) == name
