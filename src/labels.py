"""Injury-reason mapping, absence intervals and outcome-window labels.

A panel row is a player at prediction time t (a Sunday, end of day). Its
outcome window is (t, t + H]: an onset on day t itself belongs to the previous
row. Because t is a Sunday and H is a multiple of 7, an onset belongs to the
window of t exactly when its week-closing Sunday is one of t+7, ..., t+H.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LABEL_MAPPING_CONFIG, load_yaml
from .constants import sunday_on_or_before, week_end

# label column -> (horizon days, min days missed, event kind)
LABEL_SPECS: dict[str, tuple[int, int, str]] = {
    "y7": (7, 7, "any"),
    "y28": (28, 7, "any"),
    "y7_ge3": (7, 3, "any"),
    "y28_ge3": (28, 3, "any"),
    "y28_soft": (28, 7, "soft"),
}
PRIMARY_LABELS = ("y7", "y28")


def label_horizon(label: str) -> int:
    return LABEL_SPECS[label][0]


def _compile(pattern: str) -> re.Pattern:
    # Folded YAML scalars join lines with spaces; strip them around alternations.
    return re.compile(re.sub(r"\s*\|\s*", "|", pattern.strip()), re.IGNORECASE)


def load_label_mapping(path: Path = LABEL_MAPPING_CONFIG) -> dict:
    return load_yaml(path)


def _first_match(reason: str, rules: list[dict], key: str, default: str) -> str:
    for rule in rules:
        if _compile(rule["pattern"]).search(reason):
            return rule[key]
    return default


def map_reasons(reasons: pd.Series, mapping: dict) -> pd.DataFrame:
    """Map raw injury_reason strings to category, body part, tissue and soft-tissue flag."""
    rows = []
    for reason in pd.unique(reasons.dropna()):
        category = _first_match(reason, mapping["categories"], "category", mapping["default_category"])
        if category == "injury":
            body_part = _first_match(reason, mapping["body_parts"], "body_part", mapping["default_body_part"])
            tissue = _first_match(reason, mapping["tissues"], "tissue", mapping["default_tissue"])
        else:
            body_part = tissue = "n/a"
        rows.append({"injury_reason": reason, "category": category, "body_part": body_part, "tissue": tissue})
    table = pd.DataFrame(rows, columns=["injury_reason", "category", "body_part", "tissue"])
    table["soft_tissue"] = table["tissue"].isin(mapping["soft_tissue"])
    return table


def prepare_events(injuries: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Clean injury records into absence events with mapped categories.

    end_date is the last day missed (days_missed == end_date - from_date + 1 on
    every row with both dates); missing end dates are filled from days_missed.
    """
    ev = injuries.drop_duplicates().dropna(subset=["from_date", "days_missed"]).copy()
    filled = ev["from_date"] + pd.to_timedelta(ev["days_missed"].clip(lower=1) - 1, unit="D")
    ev["end_date"] = ev["end_date"].fillna(filled)
    ev = ev.merge(map_reasons(ev["injury_reason"], mapping), on="injury_reason", how="left")
    ev["category"] = ev["category"].fillna("unknown")
    return ev[["player_id", "from_date", "end_date", "days_missed", "injury_reason",
               "category", "body_part", "tissue", "soft_tissue"]].reset_index(drop=True)


def absence_weeks(events: pd.DataFrame, first_t: pd.Timestamp, last_t: pd.Timestamp) -> pd.DataFrame:
    """(player_id, t) pairs where a recorded absence of any kind spans Sunday t.

    A player is absent at t when from_date <= t < end_date: the absence started
    by t and the player is not yet back for the next day. This is observable at t.
    """
    ev = events[(events["from_date"] <= last_t) & (events["end_date"] > first_t)]
    first = week_end(ev["from_date"]).clip(lower=first_t)
    last = sunday_on_or_before(ev["end_date"] - pd.Timedelta(days=1)).clip(upper=last_t)
    n = ((last - first).dt.days // 7 + 1).clip(lower=0).to_numpy()
    starts_at = np.repeat(np.cumsum(n) - n, n)
    week_index = np.arange(n.sum()) - starts_at
    out = pd.DataFrame({
        "player_id": np.repeat(ev["player_id"].to_numpy(), n),
        "t": pd.Series(np.repeat(first.to_numpy(), n)) + pd.to_timedelta(week_index * 7, unit="D"),
    })
    return out.drop_duplicates(ignore_index=True)


def _event_flags(events: pd.DataFrame) -> pd.DataFrame:
    """Per (player_id, week-closing Sunday) positive/ambiguous flags for every label spec."""
    ev = events.assign(week=week_end(events["from_date"]))
    injury = ev["category"].eq("injury")
    unknown = ev["category"].eq("unknown")
    flags = pd.DataFrame({"player_id": ev["player_id"], "week": ev["week"]})
    for name, (_, min_days, kind) in LABEL_SPECS.items():
        long_enough = ev["days_missed"] >= min_days
        if kind == "soft":
            pos = injury & ev["soft_tissue"] & long_enough
            # unknown reasons or injuries of unspecified tissue may be soft-tissue
            amb = long_enough & (unknown | (injury & ev["tissue"].eq("unspecified")))
        else:
            pos = injury & long_enough
            amb = unknown & long_enough
        flags[f"{name}_pos"] = pos
        flags[f"{name}_amb"] = amb
        flags[f"{name}_onset"] = ev["from_date"].where(pos)
    agg = {c: ("max" if not c.endswith("_onset") else "min") for c in flags.columns[2:]}
    return flags.groupby(["player_id", "week"], as_index=False).agg(agg)


def attach_labels(
    panel: pd.DataFrame, events: pd.DataFrame, observed_until: pd.Timestamp
) -> pd.DataFrame:
    """Add label columns (1.0 / 0.0 / NaN when ambiguous or unobservable) and onset dates.

    onset_<label> is the first qualifying onset in the window; it is outcome
    information and must never be used as a feature.
    """
    flags = _event_flags(events)
    out = panel.copy()
    keys = out[["player_id", "t"]]
    max_weeks = max(h for h, _, _ in LABEL_SPECS.values()) // 7
    per_week = []
    for k in range(1, max_weeks + 1):
        shifted = keys.assign(week=keys["t"] + pd.Timedelta(days=7 * k))
        per_week.append(shifted.merge(flags, on=["player_id", "week"], how="left"))
    for name, (horizon, _, _) in LABEL_SPECS.items():
        weeks = per_week[: horizon // 7]
        pos = np.logical_or.reduce([w[f"{name}_pos"].eq(True).to_numpy() for w in weeks])
        amb = np.logical_or.reduce([w[f"{name}_amb"].eq(True).to_numpy() for w in weeks])
        onset = pd.concat([w[f"{name}_onset"] for w in weeks], axis=1).min(axis=1)
        y = np.where(pos, 1.0, np.where(amb, np.nan, 0.0))
        unobservable = (out["t"] + pd.Timedelta(days=horizon) > observed_until).to_numpy()
        y[unobservable] = np.nan
        out[name] = y
        out[f"onset_{name}"] = onset.to_numpy()
    return out
