"""The frozen deployed champion (Stage 7): the final-run model, re-materialised and verified.

Run as: python -m src.cli freeze
Refits the champion exactly as the one-shot final run did (train + validation
player groups; Platt map learned from the inner model's scores of calibration
half A of the season before the holdout) and stores nothing unless every
holdout prediction and both Platt maps equal the final run's output bit for
bit. Stored: the fitted objects per horizon, a fingerprint of their parameters,
and the calibration-half scores (calibrator_A o inner model) that the final run
used for its conformal quantiles. The model itself is never changed here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from . import splits
from .calibration import build_calibrator
from .config import load_yaml
from .constants import CONFIGS_DIR
from .experiment import git_state, load_experiment, panel_path
from .final_run import FINAL_CONFIG, FINAL_DIR, KEYS
from .labels import label_horizon
from .models import build_model, fit_categories, predict_risk, prepare_X
from .panel import content_hash
from .train import KEEP_COLUMNS, _fit_predict, leakage_tests_pass

log = logging.getLogger(__name__)

CHAMPION_DIR = FINAL_DIR / "champion"
MANIFEST = CHAMPION_DIR / "manifest.json"


@dataclass
class Frozen:
    """Deployed pipeline of one horizon: fitted model plus its calibration map."""

    label: str
    numeric: list[str]
    categorical: list[str]
    categories: dict[str, list[str]]
    model: object
    calibrator: object

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(raw, calibrated) risk for panel-shaped rows."""
        p_raw = predict_risk(self.model, prepare_X(df, self.numeric, self.categorical, self.categories))
        return p_raw, self.calibrator.transform(p_raw)

    def fingerprint(self) -> str:
        """sha256 over the fitted parameters; identical across processes for the same fit."""
        pre, clf = self.model.named_steps["pre"], self.model.named_steps["clf"]
        imputer, scaler = (step for _, step in pre.named_transformers_["num"].steps)
        encoder = pre.named_transformers_["cat"]
        digest = hashlib.sha256()
        digest.update(json.dumps([
            self.label, self.numeric, self.categorical,
            {k: [str(v) for v in vals] for k, vals in self.categories.items()},
            [[str(v) for v in c] for c in encoder.categories_],
            [int(i) for i in imputer.indicator_.features_],
        ]).encode("utf-8"))
        for arr in (imputer.statistics_, scaler.mean_, scaler.scale_, clf.coef_, clf.intercept_,
                    [self.calibrator.a, self.calibrator.b]):
            digest.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        return digest.hexdigest()


def build(panel: pd.DataFrame, label: str) -> tuple[Frozen, pd.DataFrame, pd.DataFrame, dict]:
    """Frozen model, its holdout predictions and calibration-half scores, as `final_run.classifier`
    computes them for the champion in holdout mode."""
    spec = load_yaml(FINAL_CONFIG)
    exp = load_experiment(CONFIGS_DIR / f"{spec['champion']}.yaml")
    cfg, numeric, categorical = exp["data"], exp["numeric"], exp["categorical"]
    train_first = exp.get("train_first_season", cfg["seasons"]["first"])
    roles = tuple(spec["training_roles"])
    split = splits.final_split(panel, cfg, label, label_horizon(label), allow_holdout=True)
    tr = panel.iloc[split.train_idx]
    tr = tr[tr["season"] >= train_first]
    ev = panel.iloc[split.eval_idx]
    categories = fit_categories(tr, categorical)
    model = build_model(exp["model"], numeric, categorical, exp["seed"])
    model.fit(prepare_X(tr, numeric, categorical, categories), tr[label].astype(int))

    # train._calibrate with fit_on: inner, on the final run's training roles
    cs = splits.calibration_split(panel, cfg, split.eval_season, label, roles, allow_holdout=True)
    inner = panel.iloc[cs.inner_idx]
    inner = inner[inner["season"] >= train_first]
    half_a, half_b = panel.iloc[cs.calib_a_idx], panel.iloc[cs.calib_b_idx]
    score_a, score_b = _fit_predict(exp, inner, label, half_a, half_b)
    calibrator = build_calibrator(exp["calibration"]["method"]).fit(score_a, half_a[label].to_numpy())

    frozen = Frozen(label, numeric, categorical, categories, model, calibrator)
    p_raw, p = frozen.predict(ev)
    holdout = ev[KEYS].assign(p=p, p_raw=p_raw, y=ev[label].astype(int).to_numpy())
    calibration = pd.concat([
        half[KEEP_COLUMNS].assign(half=name, y=half[label].astype(int).to_numpy(),
                                  p=calibrator.transform(score), p_raw=score)
        for name, half, score in (("A", half_a, score_a), ("B", half_b, score_b))
    ], ignore_index=True)
    record = {"train_rows": len(tr), "inner_rows": len(inner), "half_a_rows": len(half_a),
              "half_b_rows": len(half_b), **calibrator.describe()}
    return frozen, holdout, calibration, record


def _file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    """Rebuild, verify against the one-shot holdout output, then store."""
    spec = load_yaml(FINAL_CONFIG)
    exp = load_experiment(CONFIGS_DIR / f"{spec['champion']}.yaml")
    panel = pd.read_parquet(panel_path(exp["data"]))
    digest = content_hash(panel)
    if digest != spec["panel_sha256"]:
        raise SystemExit("feature store does not match the pre-registered hash")
    if not leakage_tests_pass():
        raise SystemExit("leakage tests failing: refusing to fit")
    final = json.loads((FINAL_DIR / "holdout" / "results.json").read_text(encoding="utf-8"))
    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "champion": spec["champion"],
                "panel_sha256": digest, "git": git_state(), "final_run_commit": final["git"]["commit"],
                "labels": {}}
    built = {}
    for label in spec["labels"]:
        frozen, holdout, calibration, record = build(panel, label)
        stored = pd.read_parquet(FINAL_DIR / "holdout" / f"predictions_{spec['champion']}_{label}.parquet")
        final_record = final["records"][f"{spec['champion']}-{label}"]
        checks = {
            "rows": len(holdout) == len(stored),
            "keys": holdout[KEYS].reset_index(drop=True).equals(stored[KEYS]),
            "p": np.array_equal(holdout["p"].to_numpy(), stored["p"].to_numpy()),
            "p_raw": np.array_equal(holdout["p_raw"].to_numpy(), stored["p_raw"].to_numpy()),
            "y": np.array_equal(holdout["y"].to_numpy(), stored["y"].to_numpy()),
            "platt_a": record["a"] == final_record["a"],
            "platt_b": record["b"] == final_record["b"],
        }
        if not all(checks.values()):
            raise SystemExit(f"{label}: the rebuilt champion does not reproduce the final run: {checks}")
        built[label] = (frozen, calibration, record, checks)
        log.info("%s: final-run output reproduced bit for bit", label)

    for label, (frozen, calibration, record, checks) in built.items():
        path = CHAMPION_DIR / f"{spec['champion']}_{label}.joblib"
        joblib.dump(frozen, path)
        calibration.to_parquet(CHAMPION_DIR / f"calibration_{label}.parquet", index=False)
        manifest["labels"][label] = {"file": path.name, "file_sha256": _file_sha256(path),
                                     "fingerprint": frozen.fingerprint(), "record": record,
                                     "reproduces_final_run": checks}
    manifest["model_hash"] = model_hash({label: b[0] for label, b in built.items()})
    MANIFEST.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    log.info("frozen champion stored in %s (model hash %s)", CHAMPION_DIR, manifest["model_hash"][:12])
    return manifest


def model_hash(models: dict[str, Frozen]) -> str:
    """One hash for the deployed pipeline across horizons."""
    return hashlib.sha256("".join(f"{k}:{models[k].fingerprint()}" for k in sorted(models)).encode()).hexdigest()


def load(label: str) -> Frozen:
    """A stored horizon model, checked against the manifest's parameter fingerprint."""
    if not MANIFEST.exists():
        raise FileNotFoundError(f"{MANIFEST} missing: run `python -m src.cli freeze` first")
    entry = json.loads(MANIFEST.read_text(encoding="utf-8"))["labels"][label]
    frozen = joblib.load(CHAMPION_DIR / entry["file"])
    if frozen.fingerprint() != entry["fingerprint"]:
        raise SystemExit(f"{entry['file']} does not match its recorded fingerprint")
    return frozen


def calibration_scores(label: str) -> pd.DataFrame:
    return pd.read_parquet(CHAMPION_DIR / f"calibration_{label}.parquet")
