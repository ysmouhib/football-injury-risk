"""Evaluation metrics and the paired player-cluster bootstrap.

Definitions follow the evaluation protocol section 2 and 3a. Every function takes
optional sample weights so the same code serves point estimates (weights 1) and
bootstrap resamples (weight = number of times a player was drawn).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_curve

DECISION_THRESHOLDS = np.round(np.arange(0.01, 0.3001, 0.01), 2)
_CLIP = 1e-6


def _w(y: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    return np.ones(len(y)) if w is None else np.asarray(w, dtype=float)


def pr_auc(y, p, w=None) -> float:
    return float(average_precision_score(y, p, sample_weight=w))


def operating_point(y, p, specificity: float = 0.90, w=None) -> tuple[float, float]:
    """Largest sensitivity with false-positive rate <= 1 - specificity, and its threshold."""
    fpr, tpr, thresholds = roc_curve(y, p, sample_weight=w, drop_intermediate=False)
    i = np.flatnonzero(fpr <= 1 - specificity + 1e-12)[-1]
    return float(tpr[i]), float(thresholds[i])


def mcc_at(y, p, threshold: float, w=None) -> float:
    y, w = np.asarray(y), _w(y, w)
    pred = np.asarray(p) >= threshold
    tp, fp = w[pred & (y == 1)].sum(), w[pred & (y == 0)].sum()
    fn, tn = w[~pred & (y == 1)].sum(), w[~pred & (y == 0)].sum()
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _CLIP, 1 - _CLIP)
    return np.log(p / (1 - p))


def _newton_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray, offset: np.ndarray,
                     iters: int = 50) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(X @ beta + offset)))
        grad = X.T @ (w * (y - mu))
        hess = (X * (w * mu * (1 - mu))[:, None]).T @ X
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def calibration(y, p, w=None) -> tuple[float, float]:
    """(slope, calibration-in-the-large intercept) by logistic recalibration."""
    y, w, x = np.asarray(y, dtype=float), _w(y, w), _logit(p)
    ones = np.ones((len(y), 1))
    slope = _newton_logistic(np.column_stack([ones, x]), y, w, np.zeros(len(y)))[1]
    intercept = _newton_logistic(ones, y, w, x)[0]
    return float(slope), float(intercept)


def reliability_table(y, p, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    return df.groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"), observed=("y", "mean")).reset_index()


def decision_curve(y, p, thresholds: np.ndarray = DECISION_THRESHOLDS) -> pd.DataFrame:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    n, prevalence = len(y), y.mean()
    rows = []
    for pt in thresholds:
        pred = p >= pt
        tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
        odds = pt / (1 - pt)
        rows.append({
            "threshold": pt,
            "net_benefit_model": tp / n - fp / n * odds,
            "net_benefit_treat_all": prevalence - (1 - prevalence) * odds,
            "net_benefit_treat_none": 0.0,
        })
    return pd.DataFrame(rows)


def dossier(y, p) -> dict:
    """All scalar metrics of the evaluation protocol section 2 for one evaluation set."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    sens, thr = operating_point(y, p)
    slope, intercept = calibration(y, p)
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "pr_auc": pr_auc(y, p),
        "sens_at_90spec": sens,
        "threshold_at_90spec": thr,
        "mcc_at_90spec": mcc_at(y, p, thr),
        "brier": float(brier_score_loss(y, p)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "mean_predicted": float(p.mean()),
    }


# --- bootstrap ---------------------------------------------------------------

class _SortedAP:
    """Weighted average precision for many weight vectors over fixed scores.

    Matches sklearn.metrics.average_precision_score: tied scores form one
    threshold; AP = sum over thresholds of (recall step) * precision.
    """

    def __init__(self, y: np.ndarray, p: np.ndarray):
        order = np.argsort(-p, kind="mergesort")
        self.order = order
        self.y = y[order].astype(float)
        sp = p[order]
        # last index of each group of tied scores
        self.ends = np.flatnonzero(np.r_[sp[1:] != sp[:-1], True])

    def __call__(self, weights: np.ndarray) -> np.ndarray:
        """weights: (k, n) in original row order -> (k,) AP values."""
        w = weights[:, self.order]
        tp = np.cumsum(w * self.y, axis=1)[:, self.ends]
        fp = np.cumsum(w * (1 - self.y), axis=1)[:, self.ends]
        total = tp[:, -1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            recall = tp / total
        steps = np.diff(np.concatenate([np.zeros((len(w), 1)), recall], axis=1), axis=1)
        ap = (steps * precision).sum(axis=1)
        return np.where(total[:, 0] > 0, ap, np.nan)


@dataclass
class OriginScores:
    """Evaluation rows of one origin: player ids, labels and one score column per model."""
    origin: int
    player_id: np.ndarray
    y: np.ndarray
    scores: dict[str, np.ndarray]


def resample_counts(n_players: int, n_resamples: int, seed: int, chunk: int = 100):
    """Yield (start, counts) with counts (k, n_players): player multiplicities of k resamples.

    The stream depends only on (n_players, n_resamples, seed, chunk), so every
    statistic of one comparison can be computed on the same resamples.
    """
    rng = np.random.default_rng(seed)
    for start in range(0, n_resamples, chunk):
        k = min(chunk, n_resamples - start)
        draws = rng.integers(0, n_players, size=(k, n_players))
        yield start, np.stack([np.bincount(d, minlength=n_players) for d in draws]).astype(float)


def _player_index(origins: list[OriginScores]) -> tuple[np.ndarray, list[np.ndarray]]:
    players = np.unique(np.concatenate([o.player_id for o in origins]))
    return players, [np.searchsorted(players, o.player_id) for o in origins]


def bootstrap_pr_auc(origins: list[OriginScores], n_resamples: int, seed: int,
                     chunk: int = 100) -> dict[str, np.ndarray]:
    """PR-AUC of every model on paired player-cluster resamples.

    Returns {model: array (n_resamples, n_origins)}. All models and origins share
    each resample: a drawn player contributes all of their rows in every origin.
    """
    players, index = _player_index(origins)
    scorers = [{m: _SortedAP(o.y, s) for m, s in o.scores.items()} for o in origins]
    models = list(origins[0].scores)
    out = {m: np.empty((n_resamples, len(origins))) for m in models}
    for start, counts in resample_counts(len(players), n_resamples, seed, chunk):
        for j, idx in enumerate(index):
            weights = counts[:, idx]
            for m in models:
                out[m][start:start + len(counts), j] = scorers[j][m](weights)
    return out


def bootstrap_calibration_slope(origins: list[OriginScores], n_resamples: int, seed: int,
                                chunk: int = 100) -> dict[str, np.ndarray]:
    """Calibration slope of every model on the same resamples as bootstrap_pr_auc."""
    players, index = _player_index(origins)
    models = list(origins[0].scores)
    out = {m: np.empty((n_resamples, len(origins))) for m in models}
    for start, counts in resample_counts(len(players), n_resamples, seed, chunk):
        for j, (o, idx) in enumerate(zip(origins, index)):
            for r, w in enumerate(counts[:, idx]):
                keep = w > 0
                for m in models:
                    out[m][start + r, j] = calibration(o.y[keep], o.scores[m][keep], w[keep])[0]
    return out


def percentile_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    alpha = (1 - level) / 2
    lo, hi = np.nanpercentile(values, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)
