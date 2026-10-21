"""Survival arm (Stage 3): models, survival curves on a daily grid, and metrics.

Every model predicts S(d | x) for d = 1..H days after the prediction time.
Metrics (PLAN amendment A9):
- Antolini's time-dependent concordance C^td over events with T <= H;
- integrated Brier score over days 1..H-1 (Graf, IPCW with a censoring
  Kaplan-Meier fitted on the training rows);
- h-day risk 1 - S(h) scored against the binary labels with the Arm-A metrics.
Bootstrap resamples are player-cluster resamples shared with metrics.py, so all
statistics of one comparison use the same draws.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv

from . import metrics


def day_grid(horizon: int) -> np.ndarray:
    return np.arange(1, horizon + 1)


# --- models ------------------------------------------------------------------

class RSFModel:
    """Random Survival Forest with median imputation + missingness indicators and one-hot categoricals."""

    def __init__(self, params: dict, numeric: list[str], categorical: list[str], seed: int):
        pre = ColumnTransformer([
            ("num", SimpleImputer(strategy="median", add_indicator=True), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ])
        self.pipeline = Pipeline([("pre", pre), ("rsf", RandomSurvivalForest(random_state=seed, **params))])

    def fit(self, X: pd.DataFrame, time: np.ndarray, event: np.ndarray) -> "RSFModel":
        self.pipeline.fit(X, Surv.from_arrays(event.astype(bool), time))
        return self

    def survival(self, X: pd.DataFrame, days: np.ndarray, chunk: int = 4000) -> np.ndarray:
        rsf = self.pipeline.named_steps["rsf"]
        Z = self.pipeline.named_steps["pre"].transform(X)
        idx = np.searchsorted(rsf.unique_times_, days, side="right") - 1
        out = np.empty((len(Z), len(days)))
        for start in range(0, len(Z), chunk):
            curves = rsf.predict_survival_function(Z[start:start + chunk], return_array=True)
            out[start:start + chunk] = np.where(idx >= 0, curves[:, np.maximum(idx, 0)], 1.0)
        return out


class XGBAFTModel:
    """XGBoost accelerated-failure-time model; censored rows get an open upper bound."""

    def __init__(self, params: dict, seed: int):
        params = dict(params)
        self.rounds = params.pop("n_estimators")
        self.distribution = params.get("aft_loss_distribution", "normal")
        self.scale = float(params.get("aft_loss_distribution_scale", 1.0))
        self.params = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "seed": seed,
                       "verbosity": 0, **params}

    def fit(self, X: pd.DataFrame, time: np.ndarray, event: np.ndarray) -> "XGBAFTModel":
        upper = np.where(event.astype(bool), time, np.inf)
        dtrain = xgb.DMatrix(X, enable_categorical=True)
        dtrain.set_float_info("label_lower_bound", time)
        dtrain.set_float_info("label_upper_bound", upper)
        self.booster = xgb.train(self.params, dtrain, num_boost_round=self.rounds)
        return self

    def survival(self, X: pd.DataFrame, days: np.ndarray) -> np.ndarray:
        mu = self.booster.predict(xgb.DMatrix(X, enable_categorical=True), output_margin=True)
        z = (np.log(days)[None, :] - mu[:, None]) / self.scale
        if self.distribution == "normal":
            return norm.sf(z)
        if self.distribution == "logistic":
            return 1 / (1 + np.exp(z))
        if self.distribution == "extreme":
            return np.exp(-np.exp(z))
        raise ValueError(f"unknown AFT distribution {self.distribution}")


def build_survival_model(spec: dict, numeric: list[str], categorical: list[str], seed: int):
    kind, params = spec["kind"], dict(spec.get("params") or {})
    if kind == "rsf":
        return RSFModel(params, numeric, categorical, seed)
    if kind == "xgb_aft":
        return XGBAFTModel(params, seed)
    raise ValueError(f"unknown survival model kind: {kind}")


def censoring_survival(time: np.ndarray, event: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Kaplan-Meier estimate of the censoring survival G(d) at each day (from training rows)."""
    cens = CensoringDistributionEstimator().fit(Surv.from_arrays(event.astype(bool), time))
    return cens.predict_proba(days.astype(float))


# --- metrics -----------------------------------------------------------------

class ConcordanceParts:
    """Antolini C^td as a ratio of player-by-player pair sums.

    R[a, b] / D[a, b] = concordance-weighted / comparable pairs whose event row
    belongs to player a and comparison row to player b. With player multiplicities
    c from a cluster resample, C^td = c'Rc / c'Dc; with c = 1 it is the point value.
    """

    def __init__(self, time: np.ndarray, event: np.ndarray, S: np.ndarray, player_index: np.ndarray,
                 n_players: int, horizon: int, chunk: int = 256):
        n = len(time)
        to_player_t = sparse.csr_array((np.ones(n), (player_index, np.arange(n))), shape=(n_players, n))
        cases = np.flatnonzero((event == 1) & (time <= horizon))
        self.R = np.zeros((n_players, n_players))
        self.D = np.zeros((n_players, n_players))
        censored = event == 0
        for start in range(0, len(cases), chunk):
            i = cases[start:start + chunk]
            col = time[i].astype(int) - 1
            s_case = S[i, col]
            s_other = S[:, col].T
            ti = time[i][:, None]
            comparable = (time[None, :] > ti) | ((time[None, :] == ti) & censored[None, :])
            score = comparable * ((s_other > s_case[:, None]) + 0.5 * (s_other == s_case[:, None]))
            np.add.at(self.R, player_index[i], (to_player_t @ score.T).T)
            np.add.at(self.D, player_index[i], (to_player_t @ comparable.T.astype(float)).T)

    def value(self, counts: np.ndarray | None = None) -> np.ndarray | float:
        """counts: (k, n_players) resample multiplicities -> (k,) values; None -> point value."""
        if counts is None:
            return float(self.R.sum() / self.D.sum())
        num = np.einsum("kp,pq,kq->k", counts, self.R, counts)
        den = np.einsum("kp,pq,kq->k", counts, self.D, counts)
        return num / den


def brier_contributions(time: np.ndarray, event: np.ndarray, S: np.ndarray, G: np.ndarray,
                        horizon: int) -> np.ndarray:
    """Per-row integrated Brier contribution over days 1..horizon-1 (IBS = weighted mean of these)."""
    days = day_grid(horizon - 1)
    g_at_time = G[np.clip(time.astype(int), 1, horizon) - 1]
    total = np.zeros(len(time))
    # G can reach 0 at the administrative censoring time; those rows are never cases on the grid
    with np.errstate(divide="ignore", invalid="ignore"):
        for d in days:
            s = S[:, d - 1]
            case = (time <= d) & (event == 1)
            alive = time > d
            total += np.where(case, s ** 2 / g_at_time, 0.0) + np.where(alive, (1 - s) ** 2 / G[d - 1], 0.0)
    return total / len(days)


def concordance(time, event, S, horizon: int) -> float:
    players = np.zeros(len(time), dtype=int)
    return ConcordanceParts(np.asarray(time), np.asarray(event), S, players, 1, horizon).value()


def survival_dossier(time, event, S, G, horizon: int, y_by_horizon: dict[int, np.ndarray]) -> dict:
    """Point metrics for one evaluation set; y_by_horizon maps h -> binary labels (NaN = masked)."""
    time, event = np.asarray(time, dtype=float), np.asarray(event, dtype=float)
    out = {
        "n": int(len(time)), "events": int(event.sum()),
        "event_rate": float(event.mean()),
        "c_td": concordance(time, event, S, horizon),
        "ibs": float(brier_contributions(time, event, S, G, horizon).mean()),
    }
    for h, y in y_by_horizon.items():
        keep = ~np.isnan(y)
        risk = 1 - S[keep, h - 1]
        slope, intercept = metrics.calibration(y[keep], risk)
        out.update({
            f"risk{h}_pr_auc": metrics.pr_auc(y[keep], risk),
            f"risk{h}_calibration_slope": slope,
            f"risk{h}_calibration_intercept": intercept,
            f"risk{h}_brier": float(np.mean((risk - y[keep]) ** 2)),
            f"risk{h}_mean": float(risk.mean()),
            f"y{h}_prevalence": float(y[keep].mean()),
        })
    return out
