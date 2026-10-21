"""Conformal sets against MAPIE and coverage on exchangeable data; net benefit against the harness."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from src import conformal, decision_curves, metrics


class _Prefit(ClassifierMixin, BaseEstimator):
    """Classifier whose P(y = 1) is the single input column."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.asarray(X)[:, 0]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (np.asarray(X)[:, 0] >= 0.5).astype(int)


def _sample(n, seed):
    rng = np.random.default_rng(seed)
    p = rng.beta(1, 12, n)
    return (rng.random(n) < p).astype(int), p


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_marginal_lac_matches_mapie(alpha):
    from mapie.classification import SplitConformalClassifier

    y_cal, p_cal = _sample(3000, 0)
    _, p_test = _sample(2000, 1)
    est = _Prefit().fit(p_cal[:, None], y_cal)
    mapie = SplitConformalClassifier(estimator=est, confidence_level=1 - alpha, conformity_score="lac", prefit=True)
    mapie.conformalize(p_cal[:, None], y_cal)
    _, sets = mapie.predict_set(p_test[:, None])
    q0, q1 = conformal.thresholds(y_cal, p_cal, alpha, mondrian=False)
    has0, has1 = conformal.prediction_sets(p_test, q0, q1)
    np.testing.assert_array_equal(sets[:, 0, 0], has0)
    np.testing.assert_array_equal(sets[:, 1, 0], has1)


def test_mondrian_sets_cover_each_class_on_exchangeable_data():
    y_cal, p_cal = _sample(20_000, 2)
    y, p = _sample(50_000, 3)
    q0, q1 = conformal.thresholds(y_cal, p_cal, 0.10, mondrian=True)
    has0, has1 = conformal.prediction_sets(p, q0, q1)
    assert has1[y == 1].mean() >= 0.88 and has0[y == 0].mean() >= 0.88


def test_net_benefit_matches_harness():
    y, p = _sample(5000, 4)
    pts = metrics.DECISION_THRESHOLDS
    expected = metrics.decision_curve(y, p, pts)["net_benefit_model"].to_numpy()
    np.testing.assert_allclose(decision_curves.net_benefit_parts(y.astype(float), p, pts).mean(axis=0), expected)
