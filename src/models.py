"""Model factory: builds an unfitted estimator from an experiment config's `model` block."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_X(df: pd.DataFrame, numeric: list[str], categorical: list[str],
              categories: dict[str, list[str]]) -> pd.DataFrame:
    """Feature frame with categoricals on a fixed vocabulary (unseen values -> missing)."""
    X = df[numeric].astype(float).copy()
    for col in categorical:
        X[col] = pd.Categorical(df[col], categories=categories[col])
    return X


def fit_categories(train: pd.DataFrame, categorical: list[str]) -> dict[str, list[str]]:
    return {col: sorted(train[col].dropna().unique()) for col in categorical}


def _logistic(params: dict, numeric: list[str], categorical: list[str], seed: int) -> Pipeline:
    # Missing numeric values: training median plus an explicit missingness indicator.
    numeric_steps = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    pre = ColumnTransformer([
        ("num", numeric_steps, numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    clf = LogisticRegression(random_state=seed, **params)
    return Pipeline([("pre", pre), ("clf", clf)])


def _lightgbm(params: dict, seed: int) -> LGBMClassifier:
    fixed = {"random_state": seed, "deterministic": True, "force_row_wise": True, "verbose": -1}
    return LGBMClassifier(**{**params, **fixed})


def build_model(spec: dict, numeric: list[str], categorical: list[str], seed: int):
    kind, params = spec["kind"], dict(spec.get("params") or {})
    if kind == "logistic":
        return _logistic(params, numeric, categorical, seed)
    if kind == "lightgbm":
        return _lightgbm(params, seed)
    raise ValueError(f"unknown model kind: {kind}")


def predict_risk(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
