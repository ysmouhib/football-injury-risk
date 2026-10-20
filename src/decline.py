"""Performance-decline proxies (feature family 2.4): market value and role trends.

Valuations are public on their date, so the last valuation dated <= t is known
at t. Zero valuations are placeholders in the datalake and treated as missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import last_completed_season

DECLINE_FEATURES = ["log_market_value", "mv_change_12m", "starts_share_delta"]


def _valuation_at(panel: pd.DataFrame, mv: pd.DataFrame, at: pd.Series) -> pd.DataFrame:
    left = pd.DataFrame({"player_id": panel["player_id"], "at": at, "_row": np.arange(len(panel))})
    merged = pd.merge_asof(left.sort_values("at", kind="mergesort"), mv, left_on="at", right_on="date",
                           by="player_id", direction="backward")
    return merged.sort_values("_row").set_index(panel.index)


def decline_features(panel: pd.DataFrame, market_values: pd.DataFrame, season_exp: pd.DataFrame,
                     overrides: dict[int, str]) -> pd.DataFrame:
    """Decline columns plus `decline_ts` (date of the valuation used)."""
    mv = market_values[market_values["value"] > 0][["player_id", "date", "value"]]
    mv = mv.assign(log_value=np.log(mv["value"].astype(float))).sort_values("date", kind="mergesort")
    now = _valuation_at(panel, mv, panel["t"])
    year_ago = _valuation_at(panel, mv, panel["t"] - pd.Timedelta(days=365))
    out = pd.DataFrame(index=panel.index)
    out["log_market_value"] = now["log_value"]
    out["mv_change_12m"] = now["log_value"] - year_ago["log_value"]

    prev2 = last_completed_season(panel["t"], overrides) - 1
    lagged = season_exp[["player_id", "season", "starts", "apps"]].rename(columns={"season": "_key"})
    p2 = (panel[["player_id"]].assign(_key=prev2.to_numpy(), _row=np.arange(len(panel)))
          .merge(lagged, on=["player_id", "_key"], how="left").sort_values("_row"))
    share1 = panel["prev1_starts"] / panel["prev1_apps"].where(panel["prev1_apps"] > 0)
    share2 = pd.Series(p2["starts"].to_numpy() / np.where(p2["apps"].to_numpy() > 0, p2["apps"].to_numpy(), np.nan),
                       index=panel.index)
    out["starts_share_delta"] = (share1 - share2).astype(float)
    out["decline_ts"] = now["date"]
    return out
