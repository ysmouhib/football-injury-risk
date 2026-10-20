"""Paths and season-window conventions shared across the pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATALAKE_DIR = REPO_ROOT / "football-datasets" / "datalake" / "transfermarkt"
DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = DATA_DIR / "audit"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = REPO_ROOT / "reports"
CONFIGS_DIR = REPO_ROOT / "configs"

SEASON_START_MONTH = 7  # July
SEASON_END_MONTH = 6  # June


def season_start_year(season_name: str) -> int:
    """Map a season label ('25/26', '97/98', '2009/10') to its start year."""
    first = int(str(season_name).split("/")[0])
    if first >= 1000:  # full four-digit year, e.g. '2009/10'
        return first
    return 2000 + first if first < 50 else 1900 + first


def season_window(start_year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Season with start year Y runs Jul 1 of Y through Jun 30 of Y+1, inclusive."""
    return (
        pd.Timestamp(start_year, SEASON_START_MONTH, 1),
        pd.Timestamp(start_year + 1, SEASON_END_MONTH, 30),
    )


def season_start_year_of_date(ts: pd.Timestamp) -> int:
    """Start year of the season containing a calendar date."""
    return ts.year if ts.month >= SEASON_START_MONTH else ts.year - 1
