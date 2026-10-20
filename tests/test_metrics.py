"""Metric implementations against sklearn and hand-computed values."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, matthews_corrcoef

from src import metrics


def _data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    p = np.round(np.clip(0.03 + 0.1 * y + rng.normal(0, 0.05, n), 0.001, 0.999), 2)  # rounding -> ties
    return y, p


def test_vectorised_ap_matches_sklearn_with_ties_and_weights():
    y, p = _data()
    rng = np.random.default_rng(1)
    weights = rng.integers(0, 3, size=(5, len(y))).astype(float)
    got = metrics._SortedAP(y, p)(weights)
    expected = [average_precision_score(y, p, sample_weight=w) for w in weights]
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    np.testing.assert_allclose(metrics._SortedAP(y, p)(np.ones((1, len(y))))[0], metrics.pr_auc(y, p))


def test_operating_point_and_mcc():
    y = np.array([0] * 10 + [1] * 4)
    p = np.array([0.1] * 8 + [0.5, 0.9] + [0.2, 0.6, 0.8, 0.95])
    sens, thr = metrics.operating_point(y, p, specificity=0.90)
    # threshold 0.6: FPR 1/10 (the 0.9 negative), TPR 3/4
    assert (sens, thr) == (0.75, 0.6)
    pred = (p >= thr).astype(int)
    assert metrics.mcc_at(y, p, thr) == pytest.approx(matthews_corrcoef(y, pred))


def test_calibration_slope_and_intercept():
    rng = np.random.default_rng(0)
    true = rng.uniform(0.01, 0.4, 200_000)
    y = (rng.random(len(true)) < true).astype(int)
    slope, intercept = metrics.calibration(y, true)
    assert slope == pytest.approx(1.0, abs=0.03) and intercept == pytest.approx(0.0, abs=0.02)
    logit = np.log(true / (1 - true))
    overconfident = 1 / (1 + np.exp(-2 * logit))
    assert metrics.calibration(y, overconfident)[0] == pytest.approx(0.5, abs=0.03)


def test_decision_curve_reference_lines():
    y, p = _data()
    dc = metrics.decision_curve(y, p, thresholds=np.array([0.01, 0.5]))
    prevalence = y.mean()
    assert dc.loc[0, "net_benefit_treat_all"] == pytest.approx(prevalence - (1 - prevalence) * 0.01 / 0.99)
    assert (dc["net_benefit_treat_none"] == 0).all()


def test_bootstrap_is_paired_and_reproducible():
    y, p = _data()
    players = np.repeat(np.arange(400), 10)
    origins = [metrics.OriginScores(origin=o, player_id=players, y=y, scores={"a": p, "b": p})
               for o in (2021, 2022)]
    first = metrics.bootstrap_pr_auc(origins, 50, seed=3)
    second = metrics.bootstrap_pr_auc(origins, 50, seed=3)
    np.testing.assert_array_equal(first["a"], second["a"])
    np.testing.assert_array_equal(first["a"], first["b"])  # same resample for both models
    np.testing.assert_array_equal(first["a"][:, 0], first["a"][:, 1])  # same players in both origins
    assert first["a"].std() > 0
