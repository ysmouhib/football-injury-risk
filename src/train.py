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
from .calibration import build_calibrator
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

    out = run_dir(exp["id"])
    out.mkdir(parents=True, exist_ok=True)
    if exp.get("arm") == "survival":
        predictions, split_info, extra = _train_survival(exp, panel)
    else:
        predictions, split_info, extra = _train_classification(exp, panel)

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


def _fit_predict(exp: dict, train: pd.DataFrame, label: str, *scored: pd.DataFrame) -> list[np.ndarray]:
    numeric, categorical = exp["numeric"], exp["categorical"]
    categories = fit_categories(train, categorical)
    model = build_model(exp["model"], numeric, categorical, exp["seed"])
    model.fit(prepare_X(train, numeric, categorical, categories), train[label].astype(int))
    return [predict_risk(model, prepare_X(df, numeric, categorical, categories)) for df in scored]


def _calibrate(exp: dict, panel: pd.DataFrame, label: str, origin: int, p_raw: np.ndarray,
               train_first: int, deployed, train_roles: tuple[str, ...] = ("train",),
               allow_holdout: bool = False) -> tuple[np.ndarray, dict, pd.DataFrame]:
    """Calibrated validation probabilities, a record of the calibration fit, and the calibrated
    out-of-sample scores of calibration half B (for conformal prediction).

    The calibrator learns from the inner model's (seasons <= origin - 2) out-of-time scores on half
    A of the calibration season and is applied to the deployed model's raw validation scores. With
    `fit_on: deployed` it learns from the deployed model's own (in-sample) scores of the same rows.
    """
    cs = splits.calibration_split(panel, exp["data"], origin, label, train_roles, allow_holdout)
    inner = panel.iloc[cs.inner_idx]
    inner = inner[inner["season"] >= train_first]
    half_a, half_b = panel.iloc[cs.calib_a_idx], panel.iloc[cs.calib_b_idx]
    if exp["calibration"].get("fit_on", "inner") == "deployed":
        model, categories = deployed
        score_a, score_b = (predict_risk(model, prepare_X(df, exp["numeric"], exp["categorical"], categories))
                            for df in (half_a, half_b))
    else:
        score_a, score_b = _fit_predict(exp, inner, label, half_a, half_b)
    calibrator = build_calibrator(exp["calibration"]["method"]).fit(score_a, half_a[label].to_numpy())
    record = {
        "split": cs.name, "calibration_season": cs.calibration_season,
        "inner_cutoff_t": str(cs.inner_cutoff_t.date()), "calibration_cutoff_t": str(cs.calibration_cutoff_t.date()),
        "inner_rows": len(inner), "half_a_rows": len(half_a), "half_b_rows": len(half_b),
        "half_a_positives": int(half_a[label].sum()), "fit_on": exp["calibration"].get("fit_on", "inner"),
        **calibrator.describe(),
    }
    conformal = half_b[KEEP_COLUMNS].assign(label=label, origin=origin, y=half_b[label].astype(int).to_numpy(),
                                            p=calibrator.transform(score_b))
    return calibrator.transform(p_raw), record, conformal


def _train_classification(exp: dict, panel: pd.DataFrame) -> tuple[list, list, dict]:
    data_cfg, numeric, categorical = exp["data"], exp["numeric"], exp["categorical"]
    # optional data-scope change: train on a later-starting window (never earlier than the panel)
    train_first = exp.get("train_first_season", data_cfg["seasons"]["first"])
    predictions, split_info, calibration, conformal = [], [], {}, []
    for label in exp["labels"]:
        for split in splits.rolling_origins(panel, data_cfg, label):
            tr, ev = panel.iloc[split.train_idx], panel.iloc[split.eval_idx]
            tr = tr[tr["season"] >= train_first]
            categories = fit_categories(tr, categorical)
            model = build_model(exp["model"], numeric, categorical, exp["seed"])
            start = time.perf_counter()
            model.fit(prepare_X(tr, numeric, categorical, categories), tr[label].astype(int))
            seconds = time.perf_counter() - start
            p_raw = predict_risk(model, prepare_X(ev, numeric, categorical, categories))
            p = p_raw
            if exp.get("calibration"):
                p, record, half_b = _calibrate(exp, panel, label, split.eval_season, p_raw, train_first,
                                               (model, categories))
                calibration[split.name] = record
                conformal.append(half_b)
            predictions.append(ev[KEEP_COLUMNS].assign(label=label, origin=split.eval_season,
                                                       y=ev[label].astype(int).to_numpy(), p=p, p_raw=p_raw))
            split_info.append(_split_record(split, label, tr, ev, train_first, int(tr[label].sum()), seconds))
            log.info("%s: fit %.1fs on %s rows", split.name, seconds, f"{len(tr):,}")
    if conformal:
        pd.concat(conformal, ignore_index=True).to_parquet(run_dir(exp["id"]) / "conformal_calibration.parquet",
                                                            index=False)
    return predictions, split_info, ({"calibration": calibration} if calibration else {})


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.train")
    parser.add_argument("config", type=Path)
    train(parser.parse_args().config)


if __name__ == "__main__":
    main()
