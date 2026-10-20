"""Season-level exposure from appearance counts.

The datalake's minutes_played column holds minutes *per goal* for outfield
players (missing whenever goals == 0) and total minutes only for goalkeepers
(whose goals column is empty). Appearance counts are complete, so minutes are
estimated from them for every row:

    minutes ~ b_full * full games + b_off * starts subbed off + b_on * sub appearances

with b fitted on pre-panel seasons where totals can be reconstructed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import season_start_year

COMPONENTS = ["full_games", "subbed_off_starts", "sub_apps"]


def appearance_components(perf: pd.DataFrame) -> pd.DataFrame:
    """Split appearances into full games, starts subbed off and substitute appearances.

    A player subbed on and later off in one game is counted in both subed_in and
    subed_out; clipping keeps the components non-negative and summing to apps.
    """
    apps = perf["nb_on_pitch"].astype(int)
    sub_apps = perf["subed_in"].clip(upper=apps)
    starts = apps - sub_apps
    off = perf["subed_out"].clip(upper=starts)
    return pd.DataFrame(
        {"full_games": starts - off, "subbed_off_starts": off, "sub_apps": sub_apps, "starts": starts},
        index=perf.index,
    )


def reconstruct_total_minutes(perf: pd.DataFrame) -> pd.Series:
    """True season-competition minutes where recoverable, else NaN."""
    goalkeeper = perf["goals"].isna()
    minutes_per_goal = perf["minutes_played"].astype(float)
    return minutes_per_goal.where(goalkeeper, minutes_per_goal * perf["goals"])


def add_season(perf: pd.DataFrame) -> pd.DataFrame:
    return perf.assign(season=perf["season_name"].map(season_start_year))


def fit_minutes_model(perf: pd.DataFrame, last_season: int) -> np.ndarray:
    """Least-squares coefficients (no intercept) on rows of seasons <= last_season."""
    perf = perf if "season" in perf else add_season(perf)
    total = reconstruct_total_minutes(perf)
    comps = appearance_components(perf)
    fit = total.notna() & (perf["season"] <= last_season) & (perf["nb_on_pitch"] > 0)
    coefs, *_ = np.linalg.lstsq(
        comps.loc[fit, COMPONENTS].to_numpy(float), total[fit].to_numpy(float), rcond=None
    )
    return coefs


def estimate_minutes(perf: pd.DataFrame, coefs: np.ndarray) -> pd.Series:
    return pd.Series(appearance_components(perf)[COMPONENTS].to_numpy(float) @ coefs, index=perf.index)


def season_exposure(perf: pd.DataFrame, coefs: np.ndarray) -> pd.DataFrame:
    """Per (player_id, season) totals over all club competitions."""
    perf = perf if "season" in perf else add_season(perf)
    comps = appearance_components(perf)
    rows = pd.DataFrame({
        "player_id": perf["player_id"],
        "season": perf["season"],
        "competition_id": perf["competition_id"],
        "apps": perf["nb_on_pitch"],
        "starts": comps["starts"],
        "sub_apps": comps["sub_apps"],
        "squad": perf["nb_in_group"],
        "minutes_est": estimate_minutes(perf, coefs),
    })
    played = rows[rows["apps"] > 0]
    n_comps = played.groupby(["player_id", "season"])["competition_id"].nunique().rename("n_comps")
    totals = rows.groupby(["player_id", "season"])[["apps", "starts", "sub_apps", "squad", "minutes_est"]].sum()
    return totals.join(n_comps).fillna({"n_comps": 0}).reset_index()
