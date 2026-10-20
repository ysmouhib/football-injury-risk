"""Experiment configs, output locations and run provenance."""

from __future__ import annotations

import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from .config import load_data_config, load_yaml, stable_hash
from .constants import CONFIGS_DIR, DATA_DIR, PROCESSED_DIR, REPO_ROOT, REPORTS_DIR
from .panel import FEATURE_SETS

EXPERIMENTS_DATA_DIR = DATA_DIR / "experiments"
EXPERIMENTS_REPORTS_DIR = REPORTS_DIR / "experiments"
EVALUATION_CONFIG = CONFIGS_DIR / "evaluation.yaml"
_PACKAGES = ["numpy", "pandas", "scikit-learn", "lightgbm", "pyarrow"]


def load_experiment(path: Path) -> dict:
    exp = load_yaml(path)
    exp["config_path"] = Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    exp["config_sha256"] = stable_hash(load_yaml(path))
    exp["data"] = load_data_config(REPO_ROOT / exp["data_config"])
    exp["numeric"], exp["categorical"] = FEATURE_SETS[exp["features"]]
    return exp


def run_dir(exp_id: str) -> Path:
    return EXPERIMENTS_DATA_DIR / exp_id


def predictions_path(exp_id: str) -> Path:
    return run_dir(exp_id) / "predictions.parquet"


def panel_path(data_cfg: dict) -> Path:
    return PROCESSED_DIR / f"{data_cfg['version']}.parquet"


def panel_meta(data_cfg: dict) -> dict:
    return json.loads((PROCESSED_DIR / f"{data_cfg['version']}.meta.json").read_text(encoding="utf-8"))


def load_predictions(exp_id: str) -> pd.DataFrame:
    return pd.read_parquet(predictions_path(exp_id))


def load_run(exp_id: str) -> dict:
    return json.loads((run_dir(exp_id) / "run.json").read_text(encoding="utf-8"))


def git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def environment() -> dict:
    return {"python": platform.python_version(), **{p: version(p) for p in _PACKAGES}}
