"""Split-conformal prediction sets for binary injury risk (PLAN 4.2, amendment A10).

Run as: python -m src.conformal <experiment>
Conformity score (LAC): 1 - p_hat(y | x) with p_hat(1 | x) = p, p_hat(0 | x) = 1 - p.
Quantiles come from calibration half B of the experiment's calibration split
(scored by the calibrator fitted on half A and the inner model, so out of
sample for both); coverage is checked on the validation rows. Descriptive only.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import metrics
from .config import load_yaml
from .constants import REPORTS_DIR
from .experiment import EVALUATION_CONFIG, load_predictions, load_run, run_dir

ALPHAS = (0.05, 0.10, 0.20)
TIERS = {(True, False): "low ({0})", (False, True): "high ({1})", (True, True): "uncertain ({0,1})",
         (False, False): "empty"}


def _scores(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.where(y == 1, 1 - p, p)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest score (inf if that exceeds n)."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[k - 1]) if k <= n else np.inf


def thresholds(y_cal: np.ndarray, p_cal: np.ndarray, alpha: float, mondrian: bool) -> tuple[float, float]:
    """(q for class 0, q for class 1); marginal LAC uses one quantile for both."""
    s = _scores(y_cal, p_cal)
    if not mondrian:
        q = conformal_quantile(s, alpha)
        return q, q
    return conformal_quantile(s[y_cal == 0], alpha), conformal_quantile(s[y_cal == 1], alpha)


def prediction_sets(p: np.ndarray, q0: float, q1: float) -> tuple[np.ndarray, np.ndarray]:
    """Boolean membership of class 0 and class 1 in each prediction set."""
    return p <= q0, (1 - p) <= q1


def _coverage_draws(covered: np.ndarray, player_index: np.ndarray, n_players: int, n: int, seed: int) -> np.ndarray:
    hits = np.bincount(player_index, weights=covered.astype(float), minlength=n_players)
    rows = np.bincount(player_index, minlength=n_players).astype(float)
    return np.concatenate([(c @ hits) / (c @ rows) for _, c in metrics.resample_counts(n_players, n, seed)])


def analyse(exp_id: str) -> dict:
    ev_cfg = load_yaml(EVALUATION_CONFIG)
    seed, n_res = load_run(exp_id)["seed"], ev_cfg["n_resamples"]
    preds = load_predictions(exp_id)
    cal = pd.read_parquet(run_dir(exp_id) / "conformal_calibration.parquet")
    out = []
    for (label, origin), val in preds.groupby(["label", "origin"]):
        c = cal[(cal["label"] == label) & (cal["origin"] == origin)]
        y_cal, p_cal = c["y"].to_numpy(), c["p"].to_numpy()
        y, p = val["y"].to_numpy(), val["p"].to_numpy()
        players, index = np.unique(val["player_id"].to_numpy(), return_inverse=True)
        for mondrian in (False, True):
            for alpha in ALPHAS:
                q0, q1 = thresholds(y_cal, p_cal, alpha, mondrian)
                has0, has1 = prediction_sets(p, q0, q1)
                covered = np.where(y == 1, has1, has0)
                draws = _coverage_draws(covered, index, len(players), n_res, seed)
                tier = pd.Series([TIERS[(a, b)] for a, b in zip(has0, has1)])
                tiers = {name: {"share": float((tier == name).mean()),
                                "injury_rate": float(y[(tier == name).to_numpy()].mean()) if (tier == name).any() else None}
                         for name in TIERS.values()}
                out.append({
                    "label": label, "origin": int(origin), "method": "mondrian" if mondrian else "marginal",
                    "alpha": alpha, "nominal": 1 - alpha, "q0": q0, "q1": q1,
                    "calibration_rows": len(c), "rows": len(val),
                    "coverage": float(covered.mean()), "coverage_ci": metrics.percentile_ci(draws),
                    "coverage_injured": float(covered[y == 1].mean()),
                    "coverage_uninjured": float(covered[y == 0].mean()),
                    "calibration_coverage_injured": float((1 - p_cal[y_cal == 1] <= q1).mean()),
                    "tiers": tiers,
                })
    return {"experiment": exp_id, "results": out}


def _report(result: dict) -> str:
    lines = [f"# Conformal prediction sets — {result['experiment']}", "",
             "LAC score 1 − p̂(y|x); quantiles from calibration half B (out of sample for the inner model "
             "and the calibrator); coverage on validation rows with player-cluster bootstrap CI.", "",
             "| label | origin | method | α | coverage [95% CI] | injured | uninjured | low {0} share / rate "
             "| uncertain {0,1} share / rate | high {1} share / rate | empty |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["results"]:
        t = r["tiers"]

        def cell(name: str) -> str:
            v = t[name]
            return f"{v['share']:.3f} / {v['injury_rate']:.3f}" if v["injury_rate"] is not None else f"{v['share']:.3f} / —"
        lines.append(
            f"| {r['label']} | {r['origin']} | {r['method']} | {r['alpha']:.2f} | {r['coverage']:.3f} "
            f"[{r['coverage_ci'][0]:.3f}, {r['coverage_ci'][1]:.3f}] | {r['coverage_injured']:.3f} | "
            f"{r['coverage_uninjured']:.3f} | {cell('low ({0})')} | {cell('uncertain ({0,1})')} | "
            f"{cell('high ({1})')} | {t['empty']['share']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.conformal")
    parser.add_argument("experiment")
    exp_id = parser.parse_args().experiment
    result = analyse(exp_id)
    (run_dir(exp_id) / "conformal.json").write_text(json.dumps(result, indent=2, default=lambda v: v.item()),
                                                     encoding="utf-8")
    (REPORTS_DIR / f"conformal_{exp_id}.md").write_text(_report(result), encoding="utf-8")
    print(_report(result))


if __name__ == "__main__":
    main()
