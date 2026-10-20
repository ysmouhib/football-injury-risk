"""Time-t club registration from transfer history.

A player's club at date d is the destination of the last transfer dated <= d.
Transfers are public on their date, so registration at t is knowable at t,
unlike in-season appearance records, which depend on the whole season.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import season_start_years, week_end

# On a shared date a loan return is applied before the next move (e.g. return
# on Jul 1 then a new loan on Jul 1), so the later move defines the club.
_TYPE_ORDER = {"Return from loan": 0}


def club_spells(transfers: pd.DataFrame) -> pd.DataFrame:
    """Registration spells [start, end) per player; end is NaT for the open spell."""
    t = transfers.dropna(subset=["transfer_date", "to_team_id"]).copy()
    t["_order"] = t["transfer_type"].map(_TYPE_ORDER).fillna(1)
    t = t.sort_values(["player_id", "transfer_date", "_order"], kind="mergesort")
    t["end"] = t.groupby("player_id")["transfer_date"].shift(-1)
    spells = t.rename(columns={"to_team_id": "club_id", "transfer_date": "start"})
    spells = spells[spells["end"].isna() | (spells["end"] > spells["start"])]
    return spells[["player_id", "club_id", "start", "end"]].reset_index(drop=True)


def top_league_clubs(tcs: pd.DataFrame, competitions: list[str]) -> pd.DataFrame:
    """(club_id, season, competition_id) for clubs playing in the given leagues."""
    top = tcs[tcs["competition_id"].isin(competitions)]
    top = top.rename(columns={"season_id": "season"})[["club_id", "season", "competition_id"]]
    return top.drop_duplicates(subset=["club_id", "season"]).astype({"season": int})


def club_on(spells: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """Club per player on a given date (players without a spell covering it are absent)."""
    live = spells[(spells["start"] <= date) & (spells["end"].isna() | (spells["end"] > date))]
    return live.set_index("player_id")["club_id"]


def weekly_membership(
    spells: pd.DataFrame, top: pd.DataFrame, first_t: pd.Timestamp, last_t: pd.Timestamp
) -> pd.DataFrame:
    """(player_id, t, season, club_id, competition_id) for every Sunday t in
    [first_t, last_t] on which the player is registered with a top-league club."""
    s = spells[spells["club_id"].isin(top["club_id"].unique())]
    end = s["end"].fillna(last_t + pd.Timedelta(days=1))
    s = s[(s["start"] <= last_t) & (end > first_t)]
    end = end.loc[s.index]
    first = week_end(s["start"]).clip(lower=first_t)
    # last Sunday strictly before the spell end
    last = (week_end(end) - pd.Timedelta(days=7)).clip(upper=last_t)
    n = ((last - first).dt.days // 7 + 1).clip(lower=0).to_numpy()
    week_index = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    weeks = pd.DataFrame({
        "player_id": np.repeat(s["player_id"].to_numpy(), n),
        "club_id": np.repeat(s["club_id"].to_numpy(), n),
        "t": pd.Series(np.repeat(first.to_numpy(), n)) + pd.to_timedelta(week_index * 7, unit="D"),
    })
    weeks["season"] = season_start_years(weeks["t"])
    return weeks.merge(top, on=["club_id", "season"], how="inner")
