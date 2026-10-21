"""Acute:chronic workload ratio from dated minutes (current-practice baseline, Stage 7).

acute = minutes in (t - 7 d, t]; chronic = minutes in (t - 28 d, t] / 4, the mean
weekly minutes of the last 28 days; ratio = acute / chronic, undefined when chronic
is 0; flag = ratio > 1.5. Only minutes dated on or before t enter: a match on
Sunday t is over by the end-of-day prediction time, anything later is ignored.

The ratio needs minutes per match date. The Transfermarkt datalake holds season
totals only, so the panel cannot be scored with it (PLAN decision 20); exp-0028
scores the season-scale analogues of `src/load.py` instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACUTE_DAYS = 7
CHRONIC_DAYS = 28
FLAG_ABOVE = 1.5


def acwr(minutes: pd.DataFrame, origins: pd.DataFrame, acute_days: int = ACUTE_DAYS,
         chronic_days: int = CHRONIC_DAYS, flag_above: float = FLAG_ABOVE) -> pd.DataFrame:
    """minutes: player_id, date, minutes (one row per appearance); origins: player_id, t.

    Returns origins with acute, chronic (mean weekly minutes), acwr and flag columns.
    """
    keys = origins[["player_id", "t"]].reset_index(drop=True)
    pairs = keys.assign(_row=np.arange(len(keys))).merge(
        minutes[["player_id", "date", "minutes"]], on="player_id", how="inner")
    age = (pairs["t"].dt.normalize() - pairs["date"].dt.normalize()).dt.days
    played = pairs["minutes"].astype(float)
    acute = played.where((age >= 0) & (age < acute_days), 0.0).groupby(pairs["_row"]).sum()
    chronic = played.where((age >= 0) & (age < chronic_days), 0.0).groupby(pairs["_row"]).sum()
    out = keys.copy()
    out["acute"] = acute.reindex(range(len(keys)), fill_value=0.0).to_numpy()
    out["chronic"] = chronic.reindex(range(len(keys)), fill_value=0.0).to_numpy() / (chronic_days / 7)
    out["acwr"] = (out["acute"] / out["chronic"].where(out["chronic"] > 0)).astype(float)
    out["flag"] = out["acwr"].gt(flag_above)
    return out
