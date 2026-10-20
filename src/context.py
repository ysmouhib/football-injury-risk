"""Context features (feature family 2.3): league, season phase, European club load.

National-team load is not included: the datalake's national-team table holds
career totals without dates, so it cannot be restricted to information before t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import last_completed_season

UEFA_CLUB_COMPETITIONS = ["CL", "CLQ", "EL", "ELQ", "UCOL", "UEFA", "UEQ", "USC"]
CONTEXT_NUMERIC = ["weeks_into_season", "prev1_euro_minutes"]
CONTEXT_CATEGORICAL = ["league", "season_month"]


def context_features(panel: pd.DataFrame, euro_exp: pd.DataFrame, overrides: dict[int, str]) -> pd.DataFrame:
    """euro_exp: per (player_id, season) exposure restricted to UEFA club competitions."""
    out = pd.DataFrame(index=panel.index)
    out["league"] = panel["competition_id"].astype(str)
    out["season_month"] = "m" + panel["t"].dt.month.map("{:02d}".format)
    season_start = pd.to_datetime(panel["season"].astype(str) + "-07-01")
    out["weeks_into_season"] = ((panel["t"] - season_start).dt.days // 7).astype(float)
    key = last_completed_season(panel["t"], overrides)
    lagged = euro_exp[["player_id", "season", "minutes_est"]].rename(
        columns={"season": "_key", "minutes_est": "prev1_euro_minutes"})
    merged = (panel[["player_id"]].assign(_key=key.to_numpy(), _row=np.arange(len(panel)))
              .merge(lagged, on=["player_id", "_key"], how="left").sort_values("_row"))
    out["prev1_euro_minutes"] = merged["prev1_euro_minutes"].fillna(0).to_numpy()
    return out[CONTEXT_NUMERIC + CONTEXT_CATEGORICAL]
