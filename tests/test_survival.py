"""Survival-arm metrics and models against reference implementations."""

import numpy as np
import pandas as pd
import pytest
from sksurv.metrics import brier_score
from sksurv.util import Surv

from src import survival

H = 30


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    time = rng.integers(1, H + 1, n).astype(float)
    event = (rng.random(n) < 0.4).astype(float)
    time[event == 0] = np.where(rng.random((event == 0).sum()) < 0.5, H, time[event == 0])
    # crossing, rounded curves -> ties and risk orderings that change over time
    a, b = rng.random(n)[:, None], rng.random(n)[:, None]
    days = survival.day_grid(H)[None, :]
    S = np.round(np.exp(-(a * days / 20) ** (0.5 + b)), 2)
    return time, event, S


def _naive_antolini(time, event, S, weights=None):
    w = np.ones(len(time)) if weights is None else weights
    num = den = 0.0
    for i in range(len(time)):
        if event[i] != 1 or time[i] > H:
            continue
        ti = int(time[i])
        for j in range(len(time)):
            if time[j] > time[i] or (time[j] == time[i] and event[j] == 0):
                pair = w[i] * w[j]
                den += pair
                si, sj = S[i, ti - 1], S[j, ti - 1]
                num += pair * (1.0 if sj > si else 0.5 if sj == si else 0.0)
    return num / den


def test_concordance_matches_naive_antolini():
    time, event, S = _data()
    assert survival.concordance(time, event, S, H) == pytest.approx(_naive_antolini(time, event, S))


def test_cluster_resample_quadratic_form_matches_row_duplication():
    time, event, S = _data(n=200, seed=1)
    players = np.repeat(np.arange(40), 5)
    parts = survival.ConcordanceParts(time, event, S, players, 40, H)
    counts = np.random.default_rng(2).integers(0, 3, size=(3, 40)).astype(float)
    got = parts.value(counts)
    for k in range(3):
        assert got[k] == pytest.approx(_naive_antolini(time, event, S, counts[k][players]))
    assert parts.value() == pytest.approx(_naive_antolini(time, event, S))


def test_brier_matches_sksurv_per_day():
    time, event, S = _data(n=400, seed=3)
    train_time, train_event, _ = _data(n=1000, seed=4)
    G = survival.censoring_survival(train_time, train_event, survival.day_grid(H))
    y_train, y_test = Surv.from_arrays(train_event.astype(bool), train_time), Surv.from_arrays(event.astype(bool), time)
    days = np.array([3, 10, 25])
    _, expected = brier_score(y_train, y_test, S[:, days - 1], days)
    for d, e in zip(days, expected):
        g_at_time = G[np.clip(time.astype(int), 1, H) - 1]
        s = S[:, d - 1]
        got = np.mean(np.where((time <= d) & (event == 1), s ** 2 / g_at_time, 0)
                      + np.where(time > d, (1 - s) ** 2 / G[d - 1], 0))
        assert got == pytest.approx(e)
    contrib = survival.brier_contributions(time, event, S, G, H)
    assert contrib.shape == (len(time),) and np.all(contrib >= 0)


@pytest.mark.parametrize("kind, params", [
    ("rsf", {"n_estimators": 10, "min_samples_leaf": 10, "max_features": "sqrt", "n_jobs": 1}),
    ("xgb_aft", {"n_estimators": 20, "learning_rate": 0.1, "max_depth": 3, "aft_loss_distribution": "normal",
                 "aft_loss_distribution_scale": 1.0, "tree_method": "hist", "nthread": 1}),
])
def test_models_return_valid_survival_curves(kind, params):
    rng = np.random.default_rng(5)
    X = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500),
                      "c": pd.Categorical(rng.choice(["x", "y"], 500))})
    X.loc[::17, "a"] = np.nan
    time = np.clip(np.round(rng.exponential(30 * np.exp(-X["b"].to_numpy()))), 1, H).astype(float)
    event = (time < H).astype(float)
    model = survival.build_survival_model({"kind": kind, "params": params}, ["a", "b"], ["c"], seed=0)
    S = model.fit(X, time, event).survival(X, survival.day_grid(H))
    assert S.shape == (500, H)
    assert np.all((S >= 0) & (S <= 1))
    assert np.all(np.diff(S, axis=1) <= 1e-12)
    # the informative covariate must be picked up
    assert survival.concordance(time, event, S, H) > 0.6
