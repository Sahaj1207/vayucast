"""
Step 5: daily monitoring — is the model still working in the real world?

Run:  python -m src.monitor      (GitHub Actions runs this every morning)

It does three things:
  1. LOG    today's forecasts for every city -> data/predictions/forecast_log.csv
  2. SCORE  old forecasts whose day has now passed, against what really happened
            (this is the only error number that truly matters)
  3. DRIFT  compare the last 30 days of inputs with the same season in past
            years. If the world changes (new sensors, a new pollution source,
            a strange season), the inputs move before the model's error does.
Writes reports/monitoring.json, which the dashboard shows.
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import CITIES, MODELS_DIR, PREDICTIONS_DIR, REPORTS_DIR
from src.features import build_features, clean
from src.fetch_data import fetch_live_daily
from src.predict import forecast_city
from src.config import DRIFT_FEATURES

LOG_PATH = PREDICTIONS_DIR / "forecast_log.csv"
SHIFT_WARN, SHIFT_ALERT = 1.0, 2.0   # standard deviations from the seasonal normal


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index.

    Split the reference data into 10 equal-sized buckets. If the current data
    falls into those buckets in the same proportions, PSI ≈ 0. The more the
    proportions differ, the bigger PSI gets.
        PSI = Σ (current% − reference%) × ln(current% / reference%)
    """
    reference, current = reference.dropna(), current.dropna()
    if len(reference) < 50 or len(current) < 10:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    ref_pct, cur_pct = np.clip(ref_pct, 1e-4, None), np.clip(cur_pct, 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def standardized_by_city_season(reference: pd.DataFrame, current: pd.DataFrame,
                                 features: list[str], margin_days: int = 15):
    """Put every city on the same scale, using that city's own history for
    the same time of year.

    Two lessons learned building this:
    - SEASON: comparing November with the whole year always looks like drift,
      because November is simply a polluted month. So each city is compared
      with its own previous years at the same time of year.
    - SAMPLE SIZE: 30 days for one city is too few for PSI (pure chance gives
      ~0.3). So each city's values are turned into z-scores ("how unusual for
      this city and season") and the cities are pooled: 6 × 30 = 180 values.
    """
    reference = reference.assign(date=pd.to_datetime(reference["date"]))
    ref_parts, cur_parts = [], []
    for city, cur in current.groupby("city"):
        days = cur["date"].dt.day_of_year
        lo, hi = days.min() - margin_days, days.max() + margin_days
        ref_doy = reference["date"].dt.day_of_year
        in_season = ((ref_doy >= lo) & (ref_doy <= hi)) | (ref_doy >= lo + 365) | (ref_doy <= hi - 365)
        ref = reference[(reference["city"] == city) & in_season]
        mean, std = ref[features].mean(), ref[features].std().replace(0, 1)
        ref_parts.append((ref[features] - mean) / std)
        cur_parts.append((cur[features] - mean) / std)
    return pd.concat(ref_parts), pd.concat(cur_parts)


def _psi_for_window(reference: pd.DataFrame, window: pd.DataFrame, features: list[str]):
    """PSI of `window` rows vs the same season in OTHER years.

    The whole same-year season (window ± 60 days) is left out of the reference:
    days right next to the window look a lot like it, and keeping them would
    make some windows seem more "normal" than others.
    """
    lo = window["date"].min() - pd.Timedelta(days=60)
    hi = window["date"].max() + pd.Timedelta(days=60)
    same_year = reference["date"].between(lo, hi)
    ref_z, cur_z = standardized_by_city_season(reference[~same_year], window, features)
    return {f: psi(ref_z[f], cur_z[f]) for f in features}, cur_z


def drift_check(reference: pd.DataFrame, current: pd.DataFrame, features: list[str],
                past_years: int = 4) -> list[dict]:
    """How unusual were the last 30 days, compared with the same season in other years?

    MAIN SIGNAL — seasonal shift (in standard deviations):
        |shift| < 1   normal
        1 – 2         unusual month (worth a look)
        > 2           extreme (alert)

    EXTRA INFO — PSI, with a "normal" range taken from previous years.
    Lesson learned: daily data is autocorrelated (30 days in a row behave like
    far fewer independent samples), so PSI on a 30-day window is very noisy.
    In ordinary past years it reached 1–3 for some features, way above the
    textbook 0.1 / 0.25 thresholds. So PSI alone is NOT used to raise alerts.
    """
    now, cur_z = _psi_for_window(reference, current, features)
    history = {f: [] for f in features}
    for k in range(1, past_years + 1):
        shifted = current[["city", "date"]].assign(date=current["date"] - pd.DateOffset(years=k))
        past_window = reference.merge(shifted, on=["city", "date"])
        if len(past_window) < 0.8 * len(current):
            continue                                   # not enough history for that year
        past, _ = _psi_for_window(reference, past_window, features)
        for f in features:
            history[f].append(past[f])

    rows = []
    for f in features:
        # +1.0 = one standard deviation above normal for this city and season
        shift = float(cur_z[f].mean())
        status = ("n/a" if np.isnan(shift) else "extreme" if abs(shift) > SHIFT_ALERT
                  else "unusual" if abs(shift) > SHIFT_WARN else "normal")
        rows.append({"feature": f,
                     "shift_vs_normal_season": round(shift, 2),
                     "status": status,
                     "psi": round(now[f], 3),
                     "psi_normal_max": round(max(history[f]), 3) if history[f] else None})
    return rows


def log_forecasts(results: list[dict]) -> pd.DataFrame:
    rows = [{"city": r["city"], "made_from": r["last_observed"]["date"], "target_date": f["date"],
             "days_ahead": f["days_ahead"], "aqi_forecast": f["aqi"], "low": f["low"], "high": f["high"],
             "model_trained_at": r["model"]["trained_at"]}
            for r in results for f in r["forecast"]]
    new = pd.DataFrame(rows)
    log = pd.concat([pd.read_csv(LOG_PATH), new]) if LOG_PATH.exists() else new
    log = log.drop_duplicates(subset=["city", "made_from", "days_ahead"], keep="last")
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    log.sort_values(["target_date", "city", "days_ahead"]).to_csv(LOG_PATH, index=False)
    return log


def score_past_forecasts(log: pd.DataFrame, actuals: pd.DataFrame) -> dict:
    actuals = actuals.assign(target_date=actuals["date"].dt.strftime("%Y-%m-%d"))[["city", "target_date", "aqi"]]
    scored = log.merge(actuals, on=["city", "target_date"], how="inner")
    if scored.empty:
        return {"n_scored": 0, "note": "No forecast has reached its day yet — check back tomorrow."}
    scored["abs_error"] = (scored["aqi_forecast"] - scored["aqi"]).abs()
    scored["inside_range"] = scored["aqi"].between(scored["low"], scored["high"])
    by_h = scored.groupby("days_ahead")["abs_error"].mean()
    return {
        "n_scored": int(len(scored)),
        "mae_1day": round(float(by_h.get(1, np.nan)), 2),
        "mae_by_days_ahead": {int(k): round(float(v), 2) for k, v in by_h.items()},
        "range_coverage": round(float(scored["inside_range"].mean()), 3),
    }


def main() -> None:
    results, live_frames = [], []
    for city in CITIES:
        live = fetch_live_daily(city)
        live_frames.append(live)
        results.append(forecast_city(city, live_daily=live))
        print(f"forecast logged: {CITIES[city]['name']}")
    live_all = pd.concat(live_frames, ignore_index=True)
    observed = live_all.dropna(subset=["aqi"])

    # 1 + 2: log and score
    log = log_forecasts(results)
    live_perf = score_past_forecasts(log, observed)

    # 3: drift on the last 30 observed days, against the same season in training
    current = build_features(clean(live_all, detect_glitches=False), horizon=1)
    current = current[current["date"] > observed["date"].max() - pd.Timedelta(days=30)]
    current = current.dropna(subset=["aqi_today"])
    reference = pd.read_csv(MODELS_DIR / "reference_features.csv", parse_dates=["date"])
    drift = drift_check(reference, current[["city", "date"] + DRIFT_FEATURES], DRIFT_FEATURES)

    metrics_path = REPORTS_DIR / "metrics.json"
    test_mae = json.loads(metrics_path.read_text())["test"]["1"]["mae"] if metrics_path.exists() else None
    problems = [f"{d['feature']} is extreme for the season" for d in drift if d["status"] == "extreme"]
    if live_perf.get("n_scored", 0) >= 30 and test_mae and live_perf["mae_1day"] > 1.5 * test_mae:
        problems.append("live error 50% worse than test error")
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "retrain recommended" if problems else "ok",
        "problems": problems,
        "test_mae_1day": test_mae,
        "live_performance": live_perf,
        "drift": drift,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "monitoring.json").write_text(json.dumps(report, indent=2))

    print(f"\nStatus: {report['status'].upper()}  {problems or ''}")
    print(f"Forecasts scored so far: {live_perf.get('n_scored', 0)}")
    for d in drift:
        print(f"   {d['feature']:20s} {d['shift_vs_normal_season']:+5.2f} SD vs normal season  {d['status']}")


if __name__ == "__main__":
    main()
