"""Feature store v0: player x week panel with exposure, demographics and labels.

Run as: python -m src.panel
Writes data/processed/panel_v0.parquet and panel_v0.meta.json.

Row = (player, t) with t a Sunday (end of day), for players registered with a
top-league senior team at t, not absent at t, and with a matchday-squad record
in one of the last two completed seasons. Every time-varying feature is built
from information dated <= feat_ts < t + 1 day (the first possible onset).
"""

from __future__ import annotations

import hashlib
import json
import logging

import pandas as pd

from . import datasets, exposure, history, labels, load, membership
from .config import DATA_CONFIG, LABEL_MAPPING_CONFIG, load_data_config, load_yaml, stable_hash
from .constants import (
    PROCESSED_DIR,
    last_completed_season,
    season_effective_end,
    season_window,
    week_end,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ID_COLUMNS = ["player_id", "t", "season", "club_id", "competition_id", "feat_ts"]
EXPOSURE_FIELDS = ["apps", "starts", "sub_apps", "squad", "minutes_est", "n_comps"]
NUMERIC_FEATURES_V0 = (
    ["age", "height_cm"]
    + [f"prev1_{f}" for f in EXPOSURE_FIELDS]
    + ["prev1_no_record", "prev2_minutes_est", "prev2_apps", "prev2_no_record"]
)
CATEGORICAL_FEATURES_V0 = ["position_group", "foot"]
FEATURES_V0 = NUMERIC_FEATURES_V0 + CATEGORICAL_FEATURES_V0
# feature-set name used in experiment configs -> (numeric, categorical)
FEATURE_SETS = {
    "v0": (NUMERIC_FEATURES_V0, CATEGORICAL_FEATURES_V0),
    "v0+history": (NUMERIC_FEATURES_V0 + history.HISTORY_FEATURES, CATEGORICAL_FEATURES_V0),
    "v0+history+load": (NUMERIC_FEATURES_V0 + history.HISTORY_FEATURES + load.LOAD_FEATURES,
                        CATEGORICAL_FEATURES_V0),
}


def panel_sundays(cfg: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    first_day = season_window(cfg["seasons"]["first"])[0]
    last_day = season_window(cfg["seasons"]["last"])[1]
    first_t = week_end(pd.Series([first_day])).iloc[0]
    last_t = last_day - pd.Timedelta(days=(last_day.weekday() + 1) % 7)
    return first_t, last_t


def demographics(profiles: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Static profile attributes (2025 snapshot, treated as time-invariant)."""
    p = cfg["panel"]
    height = profiles["height"].astype(float)
    height = height.where(height.between(p["min_height_cm"], p["max_height_cm"]))
    return pd.DataFrame({
        "player_id": profiles["player_id"],
        "date_of_birth": profiles["date_of_birth"],
        "height_cm": height,
        "position_group": profiles["main_position"].fillna("unknown").replace("", "unknown"),
        "foot": profiles["foot"].fillna("unknown"),
    })


def _add_exposure(panel: pd.DataFrame, season_exp: pd.DataFrame, overrides: dict) -> pd.DataFrame:
    panel["prev1_season"] = last_completed_season(panel["t"], overrides)
    panel["feat_ts"] = panel["prev1_season"].map(lambda y: season_effective_end(int(y), overrides))
    for lag in (1, 2):
        key = panel["prev1_season"] - (lag - 1)
        lagged = season_exp.rename(columns={c: f"prev{lag}_{c}" for c in EXPOSURE_FIELDS})
        lagged = lagged.rename(columns={"season": "_key"})
        cols = ["player_id", "_key"] + [f"prev{lag}_{c}" for c in EXPOSURE_FIELDS]
        panel = panel.assign(_key=key).merge(lagged[cols], on=["player_id", "_key"], how="left")
        panel[f"prev{lag}_no_record"] = panel[f"prev{lag}_apps"].isna().astype(int)
        panel = panel.drop(columns="_key")
    fill = {f"prev{lag}_{c}": 0.0 for lag in (1, 2) for c in EXPOSURE_FIELDS}
    panel = panel.fillna(fill)
    keep = {"prev2_minutes_est", "prev2_apps", "prev2_no_record"}
    drop = [c for c in panel.columns if c.startswith("prev2_") and c not in keep]
    return panel.drop(columns=drop + ["prev1_season"])


def _latest(current: pd.Series, candidate: pd.Series) -> pd.Series:
    """Element-wise max of two timestamp columns, ignoring missing candidates."""
    return current.where(candidate.isna() | (candidate <= current), candidate)


def build_panel(cfg: dict, mapping: dict) -> tuple[pd.DataFrame, dict]:
    first_t, last_t = panel_sundays(cfg)
    overrides = cfg["season_end_overrides"]

    log.info("membership")
    spells = membership.club_spells(datasets.load_transfer_history())
    top = membership.top_league_clubs(datasets.load_team_competitions_seasons(), cfg["competitions"])
    panel = membership.weekly_membership(spells, top, first_t, last_t)
    n_registered = len(panel)

    log.info("absences")
    events = labels.prepare_events(datasets.load_injuries(), mapping)
    absences = labels.absence_weeks(events, first_t, last_t)
    panel = panel.merge(absences.assign(_absent=True), on=["player_id", "t"], how="left")
    panel = panel[panel["_absent"].isna()].drop(columns="_absent")
    n_available = len(panel)

    log.info("demographics")
    demo = demographics(datasets.load_profiles(), cfg)
    panel = panel.merge(demo, on="player_id", how="left")
    panel["age"] = (panel["t"] - panel["date_of_birth"]).dt.days / 365.25
    age_ok = panel["age"].isna() | panel["age"].between(cfg["panel"]["min_age"], cfg["panel"]["max_age"])
    panel = panel[age_ok].drop(columns="date_of_birth")
    panel["position_group"] = panel["position_group"].fillna("unknown")
    panel["foot"] = panel["foot"].fillna("unknown")

    log.info("exposure")
    perf = exposure.add_season(datasets.load_performances())
    coefs = exposure.fit_minutes_model(perf, cfg["exposure"]["minutes_model_last_season"])
    season_exp = exposure.season_exposure(perf, coefs)
    panel = _add_exposure(panel, season_exp, overrides)
    n_age_ok = len(panel)
    # Registration spells without a recorded exit never close (retired or deceased
    # players stay "registered"); require a matchday-squad record in one of the
    # last two completed seasons.
    panel = panel[panel["prev1_no_record"].eq(0) | panel["prev2_no_record"].eq(0)]

    families = cfg.get("feature_families") or []
    if "history" in families:
        log.info("history features")
        hist = history.history_features(panel, events)
        panel = pd.concat([panel, hist.drop(columns="history_ts")], axis=1)
        panel["feat_ts"] = _latest(panel["feat_ts"], hist["history_ts"])
    if "load" in families:
        log.info("load-derivative features")
        lf = load.load_features(panel, season_exp, absences, overrides)
        panel = pd.concat([panel, lf.drop(columns="load_ts")], axis=1)
        panel["feat_ts"] = _latest(panel["feat_ts"], lf["load_ts"])

    log.info("labels")
    panel = labels.attach_labels(panel, events, pd.Timestamp(cfg["labels"]["observed_until"]))

    feature_columns = [c for num, cat in FEATURE_SETS.values() for c in num + cat]
    ordered = ID_COLUMNS + [c for c in dict.fromkeys(feature_columns) if c in panel.columns]
    rest = [c for c in panel.columns if c not in ordered]
    panel = panel[ordered + rest].sort_values(["player_id", "t"], ignore_index=True)

    meta = {
        "first_t": str(first_t.date()),
        "last_t": str(last_t.date()),
        "minutes_model_coefs": dict(zip(exposure.COMPONENTS, [round(float(c), 4) for c in coefs])),
        "rows_registered": n_registered,
        "rows_after_absence_filter": n_available,
        "rows_after_age_filter": n_age_ok,
        "rows": len(panel),
        "players": int(panel["player_id"].nunique()),
    }
    return panel, meta


def content_hash(df: pd.DataFrame) -> str:
    """Order-independent of file format: hashes column names, dtypes and row values."""
    digest = hashlib.sha256()
    digest.update(json.dumps([(c, str(t)) for c, t in df.dtypes.items()]).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    cfg = load_data_config()
    mapping = labels.load_label_mapping()
    panel, meta = build_panel(cfg, mapping)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    name = cfg["version"]
    panel.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)

    holdout = cfg["splits"]["holdout_season"]
    dev = panel[panel["season"] < holdout]
    meta.update({
        "version": name,
        "panel_sha256": content_hash(panel),
        "data_config_sha256": stable_hash(load_yaml(DATA_CONFIG)),
        "label_mapping_sha256": stable_hash(load_yaml(LABEL_MAPPING_CONFIG)),
        "raw_sha256": {k: datasets.sha256_file(datasets.raw_path(k)) for k in datasets.RAW_FILES},
        "seed": cfg["seed"],
        # development seasons only: nothing about holdout-season labels is summarised
        "dev_label_prevalence": {
            y: {"rate": round(float(dev[y].mean()), 5), "masked_share": round(float(dev[y].isna().mean()), 5)}
            for y in labels.LABEL_SPECS
        },
        "rows_per_season": {int(k): int(v) for k, v in panel.groupby("season").size().items()},
    })
    (PROCESSED_DIR / f"{name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("panel %s: %s rows, sha256 %s", name, f"{len(panel):,}", meta["panel_sha256"][:12])
    print(json.dumps({k: meta[k] for k in ("rows", "players", "panel_sha256", "minutes_model_coefs")}, indent=2))


if __name__ == "__main__":
    main()
