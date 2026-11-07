"""
Step 3 of the pipeline: train, compare and save the forecasting models.

Run:  python -m src.train

What happens, in order:
  1. Load daily data -> clean -> build features (one table per horizon).
  2. Split by TIME: the last 365 days are the test set, locked away.
  3. Time-series cross-validation on the training years picks the best model
     type. The test set is NOT used for choosing — that would be cheating.
  4. Every model is scored once on the test set (MAE, RMSE, R², category accuracy).
  5. Leakage demo: shows how a random (shuffled) split lies about performance.
  6. Quantile models give an 80% prediction interval ("between X and Y").
  7. SHAP explains which features drive predictions.
  8. The chosen model is retrained on ALL data and saved for the API/dashboard.
  9. Everything is logged to MLflow (if installed) and to reports/.
"""
import json
import os
import time
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")  # draw charts to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from src.aqi import category
from src.config import (CITIES, DRIFT_FEATURES, HORIZONS, MODEL_BUNDLE, MODELS_DIR, PROCESSED_DIR,
                        REPORTS_DIR, ROOT, TEST_DAYS)
from src.features import build_features, clean, feature_columns
from src.models import ML_MODELS, RANDOM_STATE, make_model, make_quantile_model

FIG_DIR = REPORTS_DIR / "figures"



# --------------------------------------------------------------------------- #
# Baselines (the models live in src/models.py)
# --------------------------------------------------------------------------- #
BASELINES = {
    "baseline_persistence": "aqi_today",   # "tomorrow will be like today"
    "baseline_7day_mean": "aqi_mean_7d",   # "tomorrow will be like this week"
}


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def score(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    same_category = np.mean([category(a) == category(p) for a, p in zip(y_true, y_pred)])
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "category_accuracy": float(same_category),
    }


def time_folds(df: pd.DataFrame, n_folds: int = 4, val_days: int = 90):
    """Expanding-window cross-validation.

    Fold 1: train on the oldest data, validate on the next 90 days.
    Fold 2: train on more data, validate on the 90 days after that. And so on.
    Training data is always BEFORE validation data, just like real life.
    """
    end = df["target_date"].max()
    for i in range(n_folds, 0, -1):
        val_start = end - pd.Timedelta(days=val_days * i - 1)
        val_end = val_start + pd.Timedelta(days=val_days - 1)
        train_idx = df.index[df["target_date"] < val_start]
        val_idx = df.index[(df["target_date"] >= val_start) & (df["target_date"] <= val_end)]
        yield train_idx, val_idx


def predict_any(name: str, model, X: pd.DataFrame) -> np.ndarray:
    if name in BASELINES:
        return X[BASELINES[name]].to_numpy()
    return model.predict(X)


def cross_validate(df: pd.DataFrame, features: list[str]) -> dict:
    results = {}
    for name in list(BASELINES) + ML_MODELS:
        maes = []
        for train_idx, val_idx in time_folds(df):
            model = None
            if name in ML_MODELS:
                model = make_model(name).fit(df.loc[train_idx, features], df.loc[train_idx, "target"])
            pred = predict_any(name, model, df.loc[val_idx, features])
            maes.append(mean_absolute_error(df.loc[val_idx, "target"], pred))
        results[name] = float(np.mean(maes))
    return results


def leakage_demo(df: pd.DataFrame, features: list[str]) -> dict:
    """Same model, two ways of testing it. Shuffled K-fold lets the model see
    days from the 'future' and from right next to the test day, so it looks
    better than it really is."""
    shuffled, honest = [], []
    for tr, va in KFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE).split(df):
        m = make_model("xgboost").fit(df.iloc[tr][features], df.iloc[tr]["target"])
        shuffled.append(mean_absolute_error(df.iloc[va]["target"], m.predict(df.iloc[va][features])))
    for tr, va in time_folds(df):
        m = make_model("xgboost").fit(df.loc[tr, features], df.loc[tr, "target"])
        honest.append(mean_absolute_error(df.loc[va, "target"], m.predict(df.loc[va, features])))
    return {"shuffled_kfold_mae": float(np.mean(shuffled)), "time_based_mae": float(np.mean(honest))}


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
def explain_with_shap(model, X: pd.DataFrame, cities: pd.Series) -> dict:
    """SHAP: how much did each feature push each prediction up or down?"""
    import shap

    sample = X.sample(min(2000, len(X)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model.model)   # the inner tree model (explains the change)
    values = explainer.shap_values(sample)

    plt.figure()
    shap.summary_plot(values, sample, max_display=15, show=False)
    plt.title("What drives tomorrow's AQI forecast (SHAP)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close("all")

    impact = pd.DataFrame(np.abs(values), columns=sample.columns, index=sample.index)
    impact["city"] = cities.loc[sample.index].values
    overall = impact.drop(columns="city").mean().sort_values(ascending=False)
    by_city = {
        city: impact[impact["city"] == city].drop(columns="city").mean()
                     .sort_values(ascending=False).head(6).round(2).to_dict()
        for city in CITIES
    }
    return {"overall_top": overall.head(10).round(2).to_dict(), "by_city": by_city}


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def plot_model_comparison(table: pd.DataFrame) -> None:
    pivot = table.pivot(index="model", columns="horizon", values="mae").sort_values(1)
    ax = pivot.plot(kind="barh", figsize=(9, 4.5))
    ax.set_xlabel("Mean Absolute Error on test year (AQI points, lower is better)")
    ax.set_ylabel("")
    ax.set_title("Model comparison — test set (last 365 days, never seen in training)")
    ax.legend(title="days ahead")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_comparison.png", dpi=130)
    plt.close()


def plot_forecasts(test: pd.DataFrame, cities=("delhi", "mumbai", "belagavi")) -> None:
    fig, axes = plt.subplots(len(cities), 1, figsize=(12, 3.2 * len(cities)), sharex=True)
    for ax, city in zip(axes, cities):
        t = test[test["city"] == city]
        ax.fill_between(t["target_date"], t["q10"], t["q90"], alpha=0.25, label="80% interval")
        ax.plot(t["target_date"], t["target"], color="black", lw=1.2, label="actual")
        ax.plot(t["target_date"], t["prediction"], lw=1.2, label="forecast (1 day ahead)")
        ax.set_title(CITIES[city]["name"])
        ax.set_ylabel("AQI")
    axes[0].legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "forecast_vs_actual.png", dpi=130)
    plt.close()


def plot_error_by_city(per_city: dict) -> None:
    s = pd.Series(per_city).sort_values()
    ax = s.plot(kind="barh", figsize=(7, 3.5), color="#4c72b0")
    ax.set_xlabel("MAE, 1 day ahead (AQI points)")
    ax.set_title("Forecast error by city — dirtier, more variable air is harder")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "error_by_city.png", dpi=130)
    plt.close()


# --------------------------------------------------------------------------- #
# MLflow (optional)
# --------------------------------------------------------------------------- #
def start_mlflow():
    try:
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import mlflow
    except ImportError:
        print("(MLflow not installed — skipping experiment tracking)")
        return None
    # Preferred: a small SQLite database. Fallback: plain files in mlruns/.
    for uri in (f"sqlite:///{(ROOT / 'mlruns.db').as_posix()}", (ROOT / "mlruns").as_uri()):
        try:
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment("vayucast")
            print(f"MLflow tracking -> {uri.split('/')[-1]}")
            return mlflow
        except Exception as error:   # e.g. a missing database driver
            print(f"(MLflow store {uri.split(':')[0]} unavailable: {type(error).__name__})")
    print("(MLflow unavailable — skipping experiment tracking)")
    return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    started = time.time()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    daily = clean(pd.read_csv(PROCESSED_DIR / "daily.csv", parse_dates=["date"]))
    print(f"Loaded {len(daily):,} cleaned daily rows "
          f"({daily['date'].min():%Y-%m-%d} -> {daily['date'].max():%Y-%m-%d})")

    datasets = {}
    for h in HORIZONS:
        df = build_features(daily, h)
        df = df.dropna(subset=["target", "aqi_today"]).reset_index(drop=True)
        datasets[h] = df
    features = feature_columns(datasets[1])

    test_start = datasets[1]["target_date"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    print(f"Test set: {test_start:%Y-%m-%d} onwards ({TEST_DAYS} days) — locked away\n")

    mlflow = start_mlflow()
    parent = mlflow.start_run(run_name=f"train-{datetime.now():%Y%m%d-%H%M}") if mlflow else None

    # ---- 1. Cross-validation on training years only -> choose model -------- #
    cv = {}
    for h in HORIZONS:
        train = datasets[h][datasets[h]["target_date"] < test_start].reset_index(drop=True)
        print(f"[h={h}] time-series cross-validation...")
        cv[h] = cross_validate(train, features)
        for name, mae in sorted(cv[h].items(), key=lambda kv: kv[1]):
            print(f"      {name:22s} CV MAE {mae:6.2f}")
    avg_cv = {m: np.mean([cv[h][m] for h in HORIZONS]) for m in ML_MODELS}
    chosen = min(avg_cv, key=avg_cv.get)
    print(f"\nChosen model (lowest average CV error): {chosen}\n")

    # ---- 2. Score every model once on the locked test year ------------------ #
    rows, test_frames, per_city = [], {}, {}
    for h in HORIZONS:
        df = datasets[h]
        train, test = df[df["target_date"] < test_start], df[df["target_date"] >= test_start].copy()
        for name in list(BASELINES) + ML_MODELS:
            model = make_model(name).fit(train[features], train["target"]) if name in ML_MODELS else None
            pred = predict_any(name, model, test[features])
            s = score(test["target"], pred)
            rows.append({"model": name, "horizon": h, "cv_mae": cv[h][name], **s})
            if name == chosen:
                test["prediction"] = pred
            if mlflow:
                with mlflow.start_run(run_name=f"{name}-h{h}", nested=True):
                    mlflow.log_params({"model": name, "horizon": h, "n_features": len(features),
                                       "train_rows": len(train), "test_rows": len(test)})
                    if model is not None:
                        params = model.get_params()
                        mlflow.log_params({k: v for k, v in params.items()
                                           if isinstance(v, (int, float, str)) and len(str(v)) < 200})
                    mlflow.log_metrics({"cv_mae": cv[h][name], **{f"test_{k}": v for k, v in s.items()}})

        # Prediction interval on test
        q10 = make_quantile_model(0.1).fit(train[features], train["target"]).predict(test[features])
        q90 = make_quantile_model(0.9).fit(train[features], train["target"]).predict(test[features])
        test["q10"], test["q90"] = np.minimum(q10, q90), np.maximum(q10, q90)
        coverage = float(np.mean((test["target"] >= test["q10"]) & (test["target"] <= test["q90"])))
        for r in rows:
            if r["model"] == chosen and r["horizon"] == h:
                r["interval_coverage"] = coverage
        test_frames[h] = test
        if h == 1:
            per_city = {CITIES[c]["name"]: float(mean_absolute_error(g["target"], g["prediction"]))
                        for c, g in test.groupby("city")}
        print(f"[h={h}] test scored. 80% interval actually covered {coverage:.0%} of days")

    table = pd.DataFrame(rows)
    table.round(3).to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print("\nTest-set results (MAE = average miss in AQI points):")
    print(table.pivot(index="model", columns="horizon", values="mae").round(1).sort_values(1))

    # ---- 3. Leakage demo ------------------------------------------------------ #
    print("\nLeakage demo (shuffled vs time-based validation)...")
    train1 = datasets[1][datasets[1]["target_date"] < test_start].reset_index(drop=True)
    leakage = leakage_demo(train1, features)
    print(f"   shuffled K-fold MAE {leakage['shuffled_kfold_mae']:.2f}  vs  "
          f"time-based MAE {leakage['time_based_mae']:.2f}")

    # ---- 4. Explain (chosen model, 1 day ahead) ------------------------------ #
    shap_summary = None
    if chosen in ("xgboost", "random_forest"):
        print("Explaining with SHAP...")
        train_mask = datasets[1]["target_date"] < test_start
        explain_model = make_model(chosen).fit(datasets[1].loc[train_mask, features],
                                               datasets[1].loc[train_mask, "target"])
        test1 = datasets[1][~train_mask]
        shap_summary = explain_with_shap(explain_model, test1[features], test1["city"])

    # ---- 5. Charts ------------------------------------------------------------ #
    plot_model_comparison(table)
    plot_forecasts(test_frames[1])
    plot_error_by_city(per_city)

    # ---- 6. Retrain on ALL data and save ------------------------------------- #
    print("Retraining the chosen model on all data for live use...")
    bundle = {
        "model_name": chosen,
        "features": features,
        "horizons": {},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_end": f"{daily['date'].max():%Y-%m-%d}",
    }
    for h in HORIZONS:
        df = datasets[h]
        bundle["horizons"][h] = {
            "point": make_model(chosen).fit(df[features], df["target"]),
            "q10": make_quantile_model(0.1).fit(df[features], df["target"]),
            "q90": make_quantile_model(0.9).fit(df[features], df["target"]),
        }
    joblib.dump(bundle, MODEL_BUNDLE, compress=3)

    # Reference data for drift monitoring
    ref = datasets[1][["city", "date"] + DRIFT_FEATURES]
    ref.to_csv(MODELS_DIR / "reference_features.csv", index=False)

    best = table[table["model"] == chosen].set_index("horizon")
    base = table[table["model"] == "baseline_persistence"].set_index("horizon")
    metrics = {
        "chosen_model": chosen,
        "trained_at": bundle["trained_at"],
        "data_range": [f"{daily['date'].min():%Y-%m-%d}", bundle["data_end"]],
        "test_period_start": f"{test_start:%Y-%m-%d}",
        "n_features": len(features),
        "cv_mae_by_model": {m: round(float(v), 2) for m, v in avg_cv.items()},
        "test": {int(h): {k: round(float(best.loc[h, k]), 3) for k in
                          ("mae", "rmse", "r2", "category_accuracy", "interval_coverage")}
                 for h in HORIZONS},
        "improvement_vs_persistence_pct": {
            int(h): round(100 * (1 - best.loc[h, "mae"] / base.loc[h, "mae"]), 1) for h in HORIZONS},
        "mae_by_city_1day": {k: round(v, 2) for k, v in per_city.items()},
        "leakage_demo": {k: round(v, 2) for k, v in leakage.items()},
        "shap": shap_summary,
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if mlflow:
        mlflow.log_params({"chosen_model": chosen, "test_start": metrics["test_period_start"]})
        for h in HORIZONS:
            mlflow.log_metrics({f"chosen_test_mae_h{h}": metrics["test"][h]["mae"]})
        mlflow.log_artifacts(str(FIG_DIR), artifact_path="figures")
        mlflow.log_artifact(str(REPORTS_DIR / "metrics.json"))
        mlflow.end_run()

    print(f"\nSaved model -> {MODEL_BUNDLE.relative_to(ROOT)}")
    print(f"Saved reports -> reports/   (took {time.time() - started:.0f}s)")
    for h in HORIZONS:
        t = metrics["test"][h]
        print(f"   {h} day(s) ahead: MAE {t['mae']:.1f}, category right {t['category_accuracy']:.0%}, "
              f"{metrics['improvement_vs_persistence_pct'][h]:+.1f}% vs 'same as today'")


if __name__ == "__main__":
    main()
