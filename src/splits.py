"""Player-grouped rolling-origin splits with outcome-window purging and embargo.

Players are partitioned once, by a seeded hash of player_id, into train /
validation / test groups. Each split draws training rows only from the train
group, validation rows only from the validation group and holdout rows only
from the test group, so player sets are disjoint in every split.

Training rows are purged so that every training outcome window (t, t + H]
closes at least embargo_days before the first evaluation timestamp.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import season_window, week_end
from .labels import label_horizon

ROLES = ("train", "validation", "test")


def player_roles(player_ids: pd.Series, fractions: dict[str, float], seed: int) -> pd.Series:
    """Role per row, from a hash of (seed, player_id): stable across panel versions."""
    uniq = pd.unique(player_ids)
    u = np.array([
        int(hashlib.sha256(f"{seed}:{int(pid)}".encode()).hexdigest()[:15], 16) / 16**15 for pid in uniq
    ])
    edges = np.cumsum([fractions[r] for r in ROLES])
    idx = np.minimum(np.searchsorted(edges, u, side="right"), len(ROLES) - 1)
    role_of = dict(zip(uniq, np.array(ROLES)[idx]))
    return player_ids.map(role_of)


def first_sunday_of_season(season: int) -> pd.Timestamp:
    return week_end(pd.Series([season_window(season)[0]])).iloc[0]


@dataclass(frozen=True)
class Split:
    name: str
    label: str | None
    horizon_days: int
    eval_season: int
    eval_role: str
    first_eval_t: pd.Timestamp
    train_cutoff_t: pd.Timestamp
    train_idx: np.ndarray
    eval_idx: np.ndarray


def _split(panel: pd.DataFrame, cfg: dict, eval_season: int, eval_role: str,
           train_roles: tuple[str, ...], label: str | None, horizon_days: int, name: str) -> Split:
    s = cfg["splits"]
    roles = player_roles(panel["player_id"], s["group_fractions"], cfg["seed"])
    first_eval_t = first_sunday_of_season(eval_season)
    cutoff = first_eval_t - pd.Timedelta(days=horizon_days + s["embargo_days"])
    usable = panel[label].notna() if label else pd.Series(True, index=panel.index)
    train = (
        roles.isin(train_roles)
        & (panel["season"] >= cfg["seasons"]["first"])
        & (panel["t"] <= cutoff)
        & usable
    )
    evaluate = roles.eq(eval_role) & panel["season"].eq(eval_season) & usable
    return Split(
        name=name, label=label, horizon_days=horizon_days, eval_season=eval_season,
        eval_role=eval_role, first_eval_t=first_eval_t, train_cutoff_t=cutoff,
        train_idx=np.flatnonzero(train.to_numpy()), eval_idx=np.flatnonzero(evaluate.to_numpy()),
    )


def origin_split(panel: pd.DataFrame, cfg: dict, val_season: int, label: str) -> Split:
    """Train on train-group seasons before val_season, validate on validation-group rows of val_season."""
    if val_season >= cfg["splits"]["holdout_season"]:
        raise ValueError(f"season {val_season} is not a validation season (holdout is one-shot)")
    return _split(panel, cfg, val_season, "validation", ("train",), label,
                  label_horizon(label), f"origin-{val_season}-{label}")


def rolling_origins(panel: pd.DataFrame, cfg: dict, label: str) -> list[Split]:
    return [origin_split(panel, cfg, v, label) for v in cfg["splits"]["validation_seasons"]]


def final_split(panel: pd.DataFrame, cfg: dict, label: str | None, horizon_days: int,
                allow_holdout: bool = False) -> Split:
    """Stage 6 only: train on train+validation groups, evaluate on the test group's holdout season.

    label=None builds the index sets without reading any label (used by the leakage tests).
    """
    if not allow_holdout:
        raise PermissionError("the final holdout split is reserved for the pre-registered Stage 6 run")
    season = cfg["splits"]["holdout_season"]
    return _split(panel, cfg, season, "test", ("train", "validation"), label, horizon_days, f"final-{season}")


def calibration_halves(player_ids: pd.Series, seed: int) -> pd.Series:
    """'A' (fits the calibrator) or 'B' (conformal scores) per row, by a salted hash of player_id."""
    uniq = pd.unique(player_ids)
    half = {pid: "A" if int(hashlib.sha256(f"{seed}:calibration:{int(pid)}".encode()).hexdigest()[:15], 16) % 2 == 0
            else "B" for pid in uniq}
    return player_ids.map(half)


@dataclass(frozen=True)
class CalibrationSplit:
    name: str
    label: str
    horizon_days: int
    val_season: int
    calibration_season: int
    inner_cutoff_t: pd.Timestamp
    calibration_cutoff_t: pd.Timestamp
    inner_idx: np.ndarray
    calib_a_idx: np.ndarray
    calib_b_idx: np.ndarray


def calibration_split(panel: pd.DataFrame, cfg: dict, val_season: int, label: str,
                      train_roles: tuple[str, ...] = ("train",), allow_holdout: bool = False) -> CalibrationSplit:
    """Rows for post-hoc calibration at an evaluation origin, all from the training player group(s).

    inner: seasons <= val_season - 2 with outcome windows closing >= embargo before season
    val_season - 1; calibration: season val_season - 1 with outcome windows closing >= embargo
    before val_season (the training cutoff), split by player into halves A and B.
    """
    s = cfg["splits"]
    if val_season >= s["holdout_season"] and not allow_holdout:
        raise ValueError(f"season {val_season} is not a validation season (holdout is one-shot)")
    horizon = label_horizon(label)
    gap = pd.Timedelta(days=horizon + s["embargo_days"])
    calibration_season = val_season - 1
    inner_cutoff = first_sunday_of_season(calibration_season) - gap
    calibration_cutoff = first_sunday_of_season(val_season) - gap
    roles = player_roles(panel["player_id"], s["group_fractions"], cfg["seed"])
    train = roles.isin(train_roles) & panel[label].notna()
    inner = train & (panel["season"] >= cfg["seasons"]["first"]) & (panel["t"] <= inner_cutoff)
    calib = train & panel["season"].eq(calibration_season) & (panel["t"] <= calibration_cutoff)
    halves = calibration_halves(panel["player_id"], cfg["seed"])
    return CalibrationSplit(
        name=f"calibration-{val_season}-{label}", label=label, horizon_days=horizon, val_season=val_season,
        calibration_season=calibration_season, inner_cutoff_t=inner_cutoff, calibration_cutoff_t=calibration_cutoff,
        inner_idx=np.flatnonzero(inner.to_numpy()),
        calib_a_idx=np.flatnonzero((calib & halves.eq("A")).to_numpy()),
        calib_b_idx=np.flatnonzero((calib & halves.eq("B")).to_numpy()),
    )
