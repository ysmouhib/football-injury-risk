"""Post-hoc probability calibration maps (Stage 4)."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .metrics import _logit, _newton_logistic


class PlattCalibrator:
    """p -> sigmoid(a + b * logit(p)), fitted by unpenalised logistic regression."""

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        X = np.column_stack([np.ones(len(p)), _logit(p)])
        self.a, self.b = _newton_logistic(X, np.asarray(y, dtype=float), np.ones(len(p)), np.zeros(len(p)))
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.a + self.b * _logit(p))))

    def describe(self) -> dict:
        return {"method": "platt", "a": float(self.a), "b": float(self.b)}


class IsotonicCalibrator:
    """Monotone increasing step map, clipped to the fitted range."""

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        self.iso.fit(np.asarray(p, dtype=float), np.asarray(y, dtype=float))
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self.iso.predict(np.asarray(p, dtype=float))

    def describe(self) -> dict:
        return {"method": "isotonic", "steps": int(len(self.iso.X_thresholds_))}


def build_calibrator(method: str):
    if method == "platt":
        return PlattCalibrator()
    if method == "isotonic":
        return IsotonicCalibrator()
    raise ValueError(f"unknown calibration method: {method}")
