"""Season-load derivative features (feature family 2.2).

Weekly load ratios (ACWR, EWMA, monotony, strain) need match-level minutes,
which the datalake does not have. These are their season-level analogues over
the last three completed seasons, plus within-season availability up to t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import last_completed_season, season_start_years

# A load ratio over less than one full match of chronic minutes is undefined.
MIN_CHRONIC_MINUTES = 90.0
LOAD_FEATURES = [
    "prev3_minutes_est",
    "season_acwr_uncoupled",
    "season_acwr_coupled",
    "minutes_delta",
    "prev1_starts_share",
    "weeks_absent_season",
]


def _ratio(num: pd.Series, den: pd.Series, min_den: float = 0.0) -> pd.Series:
    return (num / den.where(den > min_den)).astype(float)


def weeks_absent_this_season(panel: pd.DataFrame, absences: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Sundays s < t of the same season on which the player was absent, and the last such s."""
    a = absences.assign(season=season_start_years(absences["t"])).sort_values("t", kind="mergesort")
    a["n"] = a.groupby(["player_id", "season"]).cumcount() + 1
    a = a.rename(columns={"t": "s"})
    left = panel[["player_id", "season", "t"]].assign(_row=np.arange(len(panel))).sort_values("t", kind="mergesort")
    merged = pd.merge_asof(left, a, left_on="t", right_on="s", by=["player_id", "season"],
                           direction="backward", allow_exact_matches=False)
    merged = merged.sort_values("_row").set_index(panel.index)
    return merged["n"].fillna(0).astype(float), merged["s"]


def load_features(panel: pd.DataFrame, season_exp: pd.DataFrame, absences: pd.DataFrame,
                  overrides: dict[int, str]) -> pd.DataFrame:
    """Load-derivative columns plus `load_ts` (latest information date used)."""
    prev3 = last_completed_season(panel["t"], overrides) - 2
    lagged = season_exp[["player_id", "season", "minutes_est"]].rename(
        columns={"season": "_key", "minutes_est": "prev3_minutes_est"})
    p3 = (panel[["player_id"]].assign(_key=prev3.to_numpy(), _row=np.arange(len(panel)))
          .merge(lagged, on=["player_id", "_key"], how="left").sort_values("_row"))
    out = pd.DataFrame(index=panel.index)
    out["prev3_minutes_est"] = p3["prev3_minutes_est"].fillna(0).to_numpy()
    m1, m2, m3 = panel["prev1_minutes_est"], panel["prev2_minutes_est"], out["prev3_minutes_est"]
    out["season_acwr_uncoupled"] = _ratio(m1, (m2 + m3) / 2, MIN_CHRONIC_MINUTES)
    out["season_acwr_coupled"] = _ratio(m1, (m1 + m2 + m3) / 3, MIN_CHRONIC_MINUTES)
    out["minutes_delta"] = m1 - m2
    out["prev1_starts_share"] = _ratio(panel["prev1_starts"], panel["prev1_apps"])
    weeks, last_absent = weeks_absent_this_season(panel, absences)
    out["weeks_absent_season"] = weeks
    # absence on Sunday s is confirmed once the player has not returned by s + 1 day
    out["load_ts"] = last_absent + pd.Timedelta(days=1)
    return out
