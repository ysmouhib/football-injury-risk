"""Train an experiment at every validation origin and store validation predictions.

Run as: python -m src.train configs/exp-0001.yaml
The leakage tests must pass on the current feature store before any model is
fitted; the result is recorded with the run.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import splits, survival
from .constants import REPO_ROOT
from .labels import SURVIVAL_SPECS
from .experiment import environment, git_state, load_experiment, panel_meta, panel_path, predictions_path, run_dir
from .models import build_model, fit_categories, predict_risk, prepare_X
from .panel import content_hash

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

KEEP_COLUMNS = ["player_id", "t", "season", "competition_id", "position_group"]


def leakage_tests_pass() -> bool:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_leakage.py"],
                            cwd=REPO_ROOT, capture_output=True, text=True)
    log.info("leakage tests: %s", result.stdout.strip().splitlines()[-1] if result.stdout else "no output")
    return result.returncode == 0


def train(config_path: Path) -> None:
    exp = load_experiment(config_path)
    data_cfg = exp["data"]
    if not leakage_tests_pass():
        raise SystemExit("leakage tests failing: refusing to train")

    panel = pd.read_parquet(panel_path(data_cfg))
    meta = panel_meta(data_cfg)
    digest = content_hash(panel)
    if digest != meta["panel_sha256"]:
        raise SystemExit("feature store content does not match its recorded hash")

    if exp.get("arm") == "survival":
        predictions, split_info, extra = _train_survival(exp, panel)
    else:
        predictions, split_info, extra = _train_classification(exp, panel)

    out = run_dir(exp["id"])
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(predictions, ignore_index=True).to_parquet(predictions_path(exp["id"]), index=False)
    run = {
        "id": exp["id"], "config_path": exp["config_path"], "config_sha256": exp["config_sha256"],
        "data_version": data_cfg["version"], "panel_sha256": digest, "seed": exp["seed"],
        "leakage_tests": "pass", "git": git_state(), "environment": environment(),
        "splits": split_info, **extra,
    }
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    log.info("wrote %s", out)


def _split_record(split, label: str, tr: pd.DataFrame, ev: pd.DataFrame, train_first: int,
                  positives: int, seconds: float) -> dict:
    return {
        "split": split.name, "label": label, "origin": split.eval_season,
        "train_cutoff_t": str(split.train_cutoff_t.date()), "train_first_season": int(train_first),
        "first_eval_t": str(split.first_eval_t.date()),
        "train_rows": len(tr), "train_players": int(tr["player_id"].nunique()),
        "train_positives": positives,
        "eval_rows": len(ev), "eval_players": int(ev["player_id"].nunique()),
        "fit_seconds": round(seconds, 2),
    }


def _train_survival(exp: dict, panel: pd.DataFrame) -> tuple[list, list, dict]:
    data_cfg, numeric, categorical = exp["data"], exp["numeric"], exp["categorical"]
    train_first = exp.get("train_first_season", data_cfg["seasons"]["first"])
    name = exp["outcome"]
    days = survival.day_grid(SURVIVAL_SPECS[name])
    time_col, event_col = f"{name}_time", f"{name}_event"
    predictions, split_info, censoring = [], [], {}
    for split in splits.rolling_origins(panel, data_cfg, event_col):
        tr, ev = panel.iloc[split.train_idx], panel.iloc[split.eval_idx]
        tr = tr[tr["season"] >= train_first]
        categories = fit_categories(tr, categorical)
        model = survival.build_survival_model(exp["model"], numeric, categorical, exp["seed"])
        start = time.perf_counter()
        model.fit(prepare_X(tr, numeric, categorical, categories),
                  tr[time_col].to_numpy(float), tr[event_col].to_numpy(float))
        seconds = time.perf_counter() - start
        S = model.survival(prepare_X(ev, numeric, categorical, categories), days).astype(np.float32)
        censoring[str(split.eval_season)] = survival.censoring_survival(
            tr[time_col].to_numpy(float), tr[event_col].to_numpy(float), days).tolist()
        frame = ev[KEEP_COLUMNS + ["y7", "y28"]].assign(
            label=name, origin=split.eval_season,
            time=ev[time_col].to_numpy(), event=ev[event_col].to_numpy()).reset_index(drop=True)
        curves = pd.DataFrame(S, columns=[f"S{d}" for d in days])
        predictions.append(pd.concat([frame, curves], axis=1))
        split_info.append(_split_record(split, event_col, tr, ev, train_first, int(tr[event_col].sum()), seconds))
        log.info("%s: fit %.1fs on %s rows", split.name, seconds, f"{len(tr):,}")
    return predictions, split_info, {"censoring_km": censoring}


def _train_classification(exp: dict, panel: pd.DataFrame) -> tuple[list, list, dict]:
    data_cfg, numeric, categorical = exp["data"], exp["numeric"], exp["categorical"]
    # optional data-scope change: train on a later-starting window (never earlier than the panel)
    train_first = exp.get("train_first_season", data_cfg["seasons"]["first"])
    predictions, split_info = [], []
    for label in exp["labels"]:
        for split in splits.rolling_origins(panel, data_cfg, label):
            tr, ev = panel.iloc[split.train_idx], panel.iloc[split.eval_idx]
            tr = tr[tr["season"] >= train_first]
            categories = fit_categories(tr, categorical)
            model = build_model(exp["model"], numeric, categorical, exp["seed"])
            start = time.perf_counter()
            model.fit(prepare_X(tr, numeric, categorical, categories), tr[label].astype(int))
            seconds = time.perf_counter() - start
            p = predict_risk(model, prepare_X(ev, numeric, categorical, categories))
            predictions.append(ev[KEEP_COLUMNS].assign(label=label, origin=split.eval_season,
                                                       y=ev[label].astype(int).to_numpy(), p=p))
            split_info.append(_split_record(split, label, tr, ev, train_first, int(tr[label].sum()), seconds))
            log.info("%s: fit %.1fs on %s rows", split.name, seconds, f"{len(tr):,}")
    return predictions, split_info, {}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.train")
    parser.add_argument("config", type=Path)
    train(parser.parse_args().config)


if __name__ == "__main__":
    main()
