"""Arm A (classifier) vs Arm B (survival) on identical main-origin rows (PLAN task 3.4).

Run as: python -m src.arm_comparison <arm-A experiment> <arm-B experiment>
Descriptive only: no promotion decision is taken here.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import metrics
from .config import load_yaml
from .constants import REPORTS_DIR
from .experiment import EVALUATION_CONFIG, load_predictions, load_run, run_dir

KEYS = ["player_id", "t"]
HORIZONS = (7, 28)
TOP_SHARE = 0.10


def compare(arm_a: str, arm_b: str) -> dict:
    ev_cfg = load_yaml(EVALUATION_CONFIG)
    main, seed, n_res = ev_cfg["main_origin"], load_run(arm_b)["seed"], ev_cfg["n_resamples"]
    a_all, b_all = load_predictions(arm_a), load_predictions(arm_b)
    b = b_all[b_all["origin"] == main]
    out = {"arm_a": arm_a, "arm_b": arm_b, "origin": main, "horizons": {}}
    for h in HORIZONS:
        a = a_all[(a_all["label"] == f"y{h}") & (a_all["origin"] == main)]
        rows = a[KEYS + ["y", "p"]].merge(b[KEYS + [f"S{h}"]], on=KEYS, how="inner", validate="one_to_one")
        if len(rows) != len(a):
            raise SystemExit(f"y{h}: arm B is missing rows scored by arm A")
        rows = rows.sort_values(KEYS).reset_index(drop=True)
        y, p_a, p_b = rows["y"].to_numpy(float), rows["p"].to_numpy(float), 1 - rows[f"S{h}"].to_numpy(float)
        origin = metrics.OriginScores(main, rows["player_id"].to_numpy(), y, {"arm_a": p_a, "arm_b": p_b})
        boot = metrics.bootstrap_pr_auc([origin], n_res, seed)
        k = int(np.ceil(TOP_SHARE * len(rows)))
        top_a, top_b = set(np.argsort(-p_a)[:k]), set(np.argsort(-p_b)[:k])
        res = {"n": len(rows), "positives": int(y.sum())}
        for name, p in (("arm_a", p_a), ("arm_b", p_b)):
            slope, intercept = metrics.calibration(y, p)
            res[name] = {"pr_auc": metrics.pr_auc(y, p), "pr_auc_ci": metrics.percentile_ci(boot[name][:, 0]),
                         "calibration_slope": slope, "calibration_intercept": intercept,
                         "mean_predicted": float(p.mean()), "brier": float(np.mean((p - y) ** 2))}
        res["delta_pr_auc_b_minus_a"] = res["arm_b"]["pr_auc"] - res["arm_a"]["pr_auc"]
        res["delta_pr_auc_ci"] = metrics.percentile_ci(boot["arm_b"][:, 0] - boot["arm_a"][:, 0])
        res["spearman"] = float(spearmanr(p_a, p_b).statistic)
        res["top_decile_overlap"] = len(top_a & top_b) / k
        res["prevalence"] = float(y.mean())
        out["horizons"][h] = res
    return out


def _report(result: dict) -> str:
    lines = [f"# Arm A ({result['arm_a']}) vs Arm B ({result['arm_b']}) — origin {result['origin']}", "",
             "Arm A = class-weighted classifier probability for the horizon; Arm B = 1 − S(h) from the "
             "survival model. Identical rows; paired player-cluster bootstrap.", "",
             "| horizon | prevalence | PR-AUC A | PR-AUC B | Δ (B − A) [95% CI] | slope A / B | intercept A / B "
             "| mean predicted A / B | Spearman | top-decile overlap |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for h, r in result["horizons"].items():
        a, b = r["arm_a"], r["arm_b"]
        ci = r["delta_pr_auc_ci"]
        lines.append(f"| {h} d | {r['prevalence']:.4f} | {a['pr_auc']:.4f} | {b['pr_auc']:.4f} | "
                     f"{r['delta_pr_auc_b_minus_a']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                     f"{a['calibration_slope']:.3f} / {b['calibration_slope']:.3f} | "
                     f"{a['calibration_intercept']:+.3f} / {b['calibration_intercept']:+.3f} | "
                     f"{a['mean_predicted']:.3f} / {b['mean_predicted']:.3f} | {r['spearman']:.3f} | "
                     f"{r['top_decile_overlap']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.arm_comparison")
    parser.add_argument("arm_a")
    parser.add_argument("arm_b")
    args = parser.parse_args()
    result = compare(args.arm_a, args.arm_b)
    name = f"arm_comparison_{args.arm_a}_{args.arm_b}"
    (REPORTS_DIR / f"{name}.md").write_text(_report(result), encoding="utf-8")
    (run_dir(args.arm_b) / f"{name}.json").write_text(
        json.dumps(result, indent=2, default=lambda v: v.item()), encoding="utf-8")
    print(_report(result))


if __name__ == "__main__":
    main()
