"""
Model definitions, in one place so training and the live API use the same ones.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42
ML_MODELS = ["ridge", "random_forest", "xgboost"]


class DeltaModel:
    """Predict the CHANGE from today, then add today's AQI back.

    Why: "Delhi at 300" and "Belagavi at 40" are very different levels, but
    "how much will it change by tomorrow?" is a similar question everywhere.
    Tree models can't extrapolate beyond levels seen in training; predicting
    the change avoids that problem.
    """

    def __init__(self, model, base_column: str = "aqi_today"):
        self.model = model
        self.base_column = base_column

    def fit(self, X, y):
        self.model.fit(X, np.asarray(y) - X[self.base_column].to_numpy())
        return self

    def predict(self, X):
        prediction = self.model.predict(X) + X[self.base_column].to_numpy()
        return np.clip(prediction, 0, 500)          # AQI lives between 0 and 500


def make_model(name: str) -> DeltaModel:
    """A fresh, untrained model, from simple to powerful."""
    if name == "ridge":
        # Linear model: needs no missing values and features on the same scale
        inner = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    elif name == "random_forest":
        # Many decision trees, each on a random sample of rows and columns, averaged
        inner = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                      n_jobs=-1, random_state=RANDOM_STATE)
    elif name == "xgboost":
        # Trees built one after another, each one fixing the previous trees' mistakes
        inner = XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                             reg_lambda=2.0, n_jobs=-1, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"unknown model {name!r}")
    return DeltaModel(inner)


def make_quantile_model(alpha: float) -> DeltaModel:
    """Predicts a percentile instead of the average.

    alpha=0.9 means "the AQI should be below this on 90% of days".
    The 0.1 and 0.9 models together give an 80% prediction interval.
    """
    inner = XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha,
                         n_estimators=400, learning_rate=0.03, max_depth=5,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                         random_state=RANDOM_STATE)
    return DeltaModel(inner)
