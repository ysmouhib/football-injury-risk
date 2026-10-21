"""Prospective logging of the frozen champion's weekly risk scores (Stage 7, PLAN decision 24).

Run as: python -m src.cli prospective --date YYYY-MM-DD
Prediction time t is the Sunday on or before the date. The feature rows of every
active player at t (registered with a top-5 senior team, not absent, active in one
of the last two seasons) are built by the unchanged panel code over t's season,
scored by the frozen champion (never refitted) and appended to
data/prospective/<date>.csv. Rows are only ever appended; each carries the model
and code version that produced it.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import pandas as pd

from . import champion, datasets
from .config import load_data_config
from .constants import DATA_DIR, season_start_year_of_date, sunday_on_or_before
from .experiment import git_state
from .labels import load_label_mapping
from .panel import build_panel, content_hash

log = logging.getLogger(__name__)

PROSPECTIVE_DIR = DATA_DIR / "prospective"
LABELS = ("y7", "y28")
COLUMNS = ["player_id", "week", "p7", "p28", "model_hash", "git_hash", "scored_at", "features_sha256"]


def prediction_sunday(date: str | pd.Timestamp) -> pd.Timestamp:
    return sunday_on_or_before(pd.Series([pd.Timestamp(date).normalize()])).iloc[0]


def check_coverage(t: pd.Timestamp, cfg: dict, team_competitions: pd.DataFrame) -> None:
    """Refuse prediction times the local data snapshot cannot describe."""
    observed = pd.Timestamp(cfg["labels"]["observed_until"])
    if t > observed:
        raise SystemExit(f"{t.date()} is after the data snapshot ({observed.date()}): refresh the datalake first")
    season = season_start_year_of_date(t)
    present = team_competitions.loc[team_competitions["season_id"].eq(season), "competition_id"]
    missing = sorted(set(cfg["competitions"]) - set(present))
    if missing:
        raise SystemExit(f"the snapshot has no {season}/{(season + 1) % 100:02d} club table for {missing}")


def features_at(t: pd.Timestamp, cfg: dict | None = None) -> pd.DataFrame:
    """Panel rows at t, built by build_panel over t's season (labels dropped)."""
    cfg = copy.deepcopy(cfg or load_data_config())
    season = season_start_year_of_date(t)
    cfg["seasons"] = {**cfg["seasons"], "first": season, "last": season}
    panel, _ = build_panel(cfg, load_label_mapping())
    rows = panel[panel["t"].eq(t)].reset_index(drop=True)
    outcome = [c for c in rows.columns
               if c.startswith(("y7", "y28", "onset_", "surv"))]
    return rows.drop(columns=outcome)


def score(rows: pd.DataFrame, models: dict[str, champion.Frozen]) -> pd.DataFrame:
    out = rows[["player_id", "t"]].rename(columns={"t": "week"}).reset_index(drop=True)
    for label, model in models.items():
        out[f"p{label[1:]}"] = model.predict(rows)[1]
    return out


def append(frame: pd.DataFrame, path: Path) -> None:
    """Append rows; the header is written once and existing lines are never rewritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    if not new:
        with open(path, encoding="utf-8") as fh:
            header = fh.readline().rstrip("\r\n").split(",")
        if header != COLUMNS:
            raise SystemExit(f"{path} has an unexpected header {header}; refusing to append")
    with open(path, "a", encoding="utf-8", newline="") as fh:
        frame[COLUMNS].to_csv(fh, header=new, index=False, date_format="%Y-%m-%d", float_format="%.17g",
                              lineterminator="\n")


def _models() -> dict[str, champion.Frozen]:
    try:
        return {label: champion.load(label) for label in LABELS}
    except FileNotFoundError:
        log.info("frozen champion not stored yet: rebuilding it from the final-run code path")
        champion.freeze()
        return {label: champion.load(label) for label in LABELS}


def main(date: str) -> Path:
    t = prediction_sunday(date)
    cfg = load_data_config()
    check_coverage(t, cfg, datasets.load_team_competitions_seasons())
    models = _models()
    rows = features_at(t, cfg)
    if rows.empty:
        raise SystemExit(f"no active players at {t.date()}")
    used = list(dict.fromkeys(c for m in models.values() for c in m.numeric + m.categorical))
    git = git_state()
    scored = score(rows, models).assign(
        model_hash=champion.model_hash(models),
        git_hash=git["commit"] + ("-dirty" if git["dirty"] else ""),
        scored_at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        features_sha256=content_hash(rows[["player_id", "t", *used]]))
    path = PROSPECTIVE_DIR / f"{pd.Timestamp(date).date()}.csv"
    append(scored, path)
    log.info("scored %d players at %s -> %s", len(scored), t.date(), path)
    return path
