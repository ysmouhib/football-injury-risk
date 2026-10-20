"""Injury-history features (feature family 2.1).

Built from injury and unknown-reason events with onset <= t. For panel rows
every such event has already ended (a player with an absence spanning t is not
in the panel), so durations are known at t; this is asserted, not assumed.
Window counts use onsets in (t - W, t].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BODY_PARTS = ["hamstring", "groin_hip", "thigh", "knee", "ankle", "lower_leg"]
WINDOWS = (365, 730)
RECENT_RETURN_DAYS = 28
HISTORY_FEATURES = (
    ["days_since_return", "no_prior_injury", "returned_within_28d", "career_injuries", "career_days_lost"]
    + [f"injuries_{w}d" for w in WINDOWS]
    + [f"days_lost_{w}d" for w in WINDOWS]
    + [f"{bp}_injuries_730d" for bp in BODY_PARTS]
    + ["muscle_injuries_730d"]
)


def _counters(events: pd.DataFrame) -> pd.DataFrame:
    """Per-event cumulative counters in onset order, one row per (player, onset date)."""
    ev = events.sort_values(["player_id", "from_date", "end_date"], kind="mergesort").copy()
    ev["one"] = 1.0
    ev["muscle"] = ev["tissue"].eq("muscle").astype(float)
    for bp in BODY_PARTS:
        ev[bp] = ev["body_part"].eq(bp).astype(float)
    counted = ["one", "days_missed", "muscle"] + BODY_PARTS
    g = ev.groupby("player_id")
    cum = g[counted].cumsum()
    cum["last_end"] = g["end_date"].cummax()
    cum[["player_id", "from_date"]] = ev[["player_id", "from_date"]]
    # several onsets on one day: keep the state after the last of them
    cum = cum.drop_duplicates(subset=["player_id", "from_date"], keep="last")
    return cum.sort_values("from_date", kind="mergesort")


def _state_at(keys: pd.DataFrame, counters: pd.DataFrame, at: pd.Series) -> pd.DataFrame:
    """Counter values after all onsets dated <= `at`, aligned to keys' index."""
    left = pd.DataFrame({"player_id": keys["player_id"], "at": at, "_row": np.arange(len(keys))})
    merged = pd.merge_asof(left.sort_values("at"), counters, left_on="at", right_on="from_date",
                           by="player_id", direction="backward")
    return merged.sort_values("_row").set_index(keys.index)


def history_features(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """History feature columns for panel rows, plus `history_ts` (latest information date used)."""
    injuries = events[events["category"].isin(["injury", "unknown"])]
    counters = _counters(injuries)
    now = _state_at(panel, counters, panel["t"])
    if (now["last_end"] > panel["t"]).any():
        raise ValueError("an injury that had not ended by t would enter the history features")

    out = pd.DataFrame(index=panel.index)
    out["days_since_return"] = (panel["t"] - now["last_end"]).dt.days.astype(float)
    out["no_prior_injury"] = now["last_end"].isna().astype(int)
    out["returned_within_28d"] = (out["days_since_return"] <= RECENT_RETURN_DAYS).astype(int)
    out["career_injuries"] = now["one"].fillna(0)
    out["career_days_lost"] = now["days_missed"].fillna(0)
    for w in WINDOWS:
        before = _state_at(panel, counters, panel["t"] - pd.Timedelta(days=w))
        out[f"injuries_{w}d"] = now["one"].fillna(0) - before["one"].fillna(0)
        out[f"days_lost_{w}d"] = now["days_missed"].fillna(0) - before["days_missed"].fillna(0)
        if w == 730:
            for bp in BODY_PARTS:
                out[f"{bp}_injuries_730d"] = now[bp].fillna(0) - before[bp].fillna(0)
            out["muscle_injuries_730d"] = now["muscle"].fillna(0) - before["muscle"].fillna(0)
    out["history_ts"] = now["last_end"]
    return out[HISTORY_FEATURES + ["history_ts"]]
