"""Loaders for the raw Transfermarkt datalake tables, plus content hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .constants import DATALAKE_DIR, DATA_DIR

RAW_FILES = {
    "player_injuries": Path("player_injuries/player_injuries_updated_20260929.csv"),
    "player_performances": Path("player_performances/player_performances.csv"),
    "player_profiles": Path("player_profiles/player_profiles.csv"),
    "transfer_history": Path("transfer_history/transfer_history.csv"),
    "team_competitions_seasons": Path("team_competitions_seasons/team_competitions_seasons.csv"),
    "player_market_value": Path("player_market_value/player_market_value.csv"),
}


def raw_path(name: str, base_dir: Path | None = None) -> Path:
    return (base_dir or DATALAKE_DIR) / RAW_FILES[name]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_hashes(
    out_path: Path = DATA_DIR / "HASHES.txt", base_dir: Path | None = None
) -> dict[str, str]:
    hashes = {name: sha256_file(raw_path(name, base_dir)) for name in RAW_FILES}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{digest}  {RAW_FILES[name].as_posix()}" for name, digest in hashes.items()]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return hashes


def load_injuries(base_dir: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(
        raw_path("player_injuries", base_dir),
        dtype={"player_id": "int64", "season_name": "str", "injury_reason": "str"},
        parse_dates=["from_date", "end_date"],
    )
    return df


def load_performances(base_dir: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(
        raw_path("player_performances", base_dir),
        dtype={
            "player_id": "int64",
            "season_name": "str",
            "competition_id": "str",
            "competition_name": "str",
            "team_id": "Int64",
            "team_name": "str",
            "minutes_played": "Float64",
        },
    )
    return df


def load_profiles(base_dir: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(
        raw_path("player_profiles", base_dir),
        dtype={
            "player_id": "int64",
            "player_name": "str",
            "height": "Float64",
            "main_position": "str",
            "foot": "str",
        },
        parse_dates=["date_of_birth"],
    )
    return df


def load_transfer_history(base_dir: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(
        raw_path("transfer_history", base_dir),
        dtype={"player_id": "int64", "season_name": "str"},
        parse_dates=["transfer_date"],
    )
    return df


def load_team_competitions_seasons(base_dir: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(
        raw_path("team_competitions_seasons", base_dir),
        dtype={
            "competition_id": "str",
            "competition_name": "str",
            "season_id": "Int64",
            "season_league_level_level_number": "Float64",
            "season_total_matches": "Float64",
        },
    )
    return df


def load_market_values(base_dir: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(
        raw_path("player_market_value", base_dir),
        dtype={"player_id": "int64", "value": "Float64"},
        parse_dates=["date_unix"],
    )
    df = df.rename(columns={"date_unix": "date"})
    return df
