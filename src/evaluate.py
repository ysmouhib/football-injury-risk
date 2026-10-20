"""Metric dossier, champion comparison and verdict for one trained experiment.

Run as: python -m src.evaluate configs/exp-0002.yaml
Reads data/experiments/<id>/predictions.parquet (written by src.train) and
writes data/experiments/<id>/metrics.json and reports/experiments/<id>.md.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .config import load_yaml
from .experiment import (
    EVALUATION_CONFIG,
    EXPERIMENTS_REPORTS_DIR,
    load_experiment,
    load_predictions,
    load_run,
    run_dir,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DOSSIER_COLUMNS = ["n", "positives", "prevalence", "pr_auc", "sens_at_90spec", "mcc_at_90spec",
                   "brier", "calibration_slope", "calibration_intercept", "mean_predicted"]


def _origin_scores(rows: pd.DataFrame, score_columns: dict[str, str]) -> list[metrics.OriginScores]:
    return [
        metrics.OriginScores(origin=int(o), player_id=g["player_id"].to_numpy(), y=g["y"].to_numpy(),
                             scores={m: g[c].to_numpy() for m, c in score_columns.items()})
        for o, g in rows.sort_values(["origin", "player_id", "t"]).groupby("origin")
    ]


def _aggregate(per_origin: pd.Series, how: str, main: int) -> float:
    return float(per_origin.loc[main] if how == "main" else per_origin.mean())


def _aggregate_boot(boot: np.ndarray, origins: list[int], how: str, main: int) -> np.ndarray:
    return boot[:, origins.index(main)] if how == "main" else boot.mean(axis=1)


def _strata(rows: pd.DataFrame, column: str) -> pd.DataFrame:
    per = [
        {"stratum": s, "origin": o, "n": len(g), "positives": int(g["y"].sum()),
         "pr_auc": metrics.pr_auc(g["y"], g["p"]) if 0 < g["y"].sum() < len(g) else np.nan}
        for (o, s), g in rows.groupby(["origin", column])
    ]
    per = pd.DataFrame(per)
    return per.groupby("stratum").agg(n=("n", "sum"), positives=("positives", "sum"),
                                      pr_auc_mean_origins=("pr_auc", "mean")).reset_index()


def _label_dossier(rows: pd.DataFrame, ev_cfg: dict, seed: int, out_dir: Path, label: str) -> dict:
    main = ev_cfg["main_origin"]
    per_origin = pd.DataFrame({o: metrics.dossier(g["y"], g["p"]) for o, g in rows.groupby("origin")}).T
    main_rows = rows[rows["origin"] == main]
    metrics.reliability_table(main_rows["y"], main_rows["p"]).to_csv(
        out_dir / f"reliability_{label}_{main}.csv", index=False)
    metrics.decision_curve(main_rows["y"], main_rows["p"]).to_csv(
        out_dir / f"decision_curve_{label}_{main}.csv", index=False)

    origins = _origin_scores(rows, {"model": "p"})
    order = [o.origin for o in origins]
    boot = metrics.bootstrap_pr_auc(origins, ev_cfg["n_resamples"], seed)["model"]
    agg = {}
    for how in ("main", "mean_origins"):
        dist = _aggregate_boot(boot, order, how, main)
        agg[how] = {
            "pr_auc": _aggregate(per_origin["pr_auc"], how, main),
            "pr_auc_ci": metrics.percentile_ci(dist, ev_cfg["ci_level"]),
            "pr_auc_se": float(np.nanstd(dist, ddof=1)),
            "calibration_slope": _aggregate(per_origin["calibration_slope"], how, main),
        }
    return {
        "per_origin": {int(o): {k: float(v) for k, v in r.items()} for o, r in per_origin.iterrows()},
        "aggregate": agg,
        "strata_league": _strata(rows, "competition_id").to_dict("records"),
        "strata_position": _strata(rows, "position_group").to_dict("records"),
    }


def _compare(preds: pd.DataFrame, champ: pd.DataFrame, own: dict, champ_metrics: dict,
             ev_cfg: dict, seed: int) -> dict:
    how, main = ev_cfg["aggregation"], ev_cfg["main_origin"]
    keys = ["label", "origin", "player_id", "t"]
    result = {}
    for label in ev_cfg["decision_labels"]:
        a = preds[preds["label"] == label]
        b = champ[champ["label"] == label]
        both = a.merge(b[keys + ["y", "p"]], on=keys, suffixes=("", "_champion"), validate="one_to_one")
        if not (len(both) == len(a) == len(b)) or not both["y"].equals(both["y_champion"]):
            raise SystemExit(f"{label}: challenger and champion were not scored on identical rows")
        origins = _origin_scores(both, {"champion": "p_champion", "challenger": "p"})
        order = [o.origin for o in origins]
        boot = metrics.bootstrap_pr_auc(origins, ev_cfg["n_resamples"], seed)
        delta_dist = (_aggregate_boot(boot["challenger"], order, how, main)
                      - _aggregate_boot(boot["champion"], order, how, main))
        mine, theirs = own[label]["aggregate"][how], champ_metrics["labels"][label]["aggregate"][how]
        delta = mine["pr_auc"] - theirs["pr_auc"]
        ci = metrics.percentile_ci(delta_dist, ev_cfg["ci_level"])
        regression = abs(mine["calibration_slope"] - 1) - abs(theirs["calibration_slope"] - 1)
        result[label] = {
            "champion_pr_auc": theirs["pr_auc"], "challenger_pr_auc": mine["pr_auc"],
            "delta": delta, "delta_ci": ci,
            "champion_slope": theirs["calibration_slope"], "challenger_slope": mine["calibration_slope"],
            "slope_regression": regression,
            "cond1_delta_positive": delta > 0,
            "cond2_ci_excludes_zero": ci[0] > 0,
            "cond3_no_slope_regression": regression <= ev_cfg["max_slope_regression"],
        }
    conds = pd.DataFrame(result).T
    if conds[["cond1_delta_positive", "cond2_ci_excludes_zero", "cond3_no_slope_regression"]].all().all():
        verdict = "PROMOTE"
    elif conds["cond1_delta_positive"].all() and not conds["cond2_ci_excludes_zero"].all():
        verdict = "INCONCLUSIVE"
    else:
        verdict = "REJECT"
    return {"aggregation": how, "labels": result, "verdict": verdict}


def _fmt_ci(ci) -> str:
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def _report(exp: dict, run: dict, result: dict, ev_cfg: dict) -> str:
    main = ev_cfg["main_origin"]
    lines = [
        f"# {exp['id']} — {exp['title']}", "",
        f"- Stage {exp['stage']}; hypothesis: {exp['hypothesis']}",
        f"- Champion compared against: {exp.get('champion') or 'none (first champion of record)'}",
        f"- Config: `{run['config_path']}` (sha256 {run['config_sha256'][:12]}); seed {run['seed']}",
        f"- Data: `{run['data_version']}` sha256 {run['panel_sha256'][:12]}; "
        f"origins {sorted({s['origin'] for s in run['splits']})} (main {main})",
        f"- Code: commit {run['git']['commit'][:10]}{' (dirty)' if run['git']['dirty'] else ''}; "
        f"leakage tests at run time: {run['leakage_tests']}",
        f"- Environment: {', '.join(f'{k} {v}' for k, v in run['environment'].items())}",
        "",
    ]
    for label, d in result["labels"].items():
        per = pd.DataFrame(d["per_origin"]).T[DOSSIER_COLUMNS]
        lines += [f"## {label}", "", "| origin | " + " | ".join(DOSSIER_COLUMNS) + " |",
                  "|" + "---|" * (len(DOSSIER_COLUMNS) + 1)]
        for o, r in per.iterrows():
            cells = [f"{int(r[c]):,}" if c in ("n", "positives") else f"{r[c]:.4f}" for c in DOSSIER_COLUMNS]
            lines.append(f"| {o} | " + " | ".join(cells) + " |")
        lines.append("")
        for how, a in d["aggregate"].items():
            lines.append(f"- PR-AUC ({how}): {a['pr_auc']:.4f} {_fmt_ci(a['pr_auc_ci'])}, "
                         f"bootstrap SE {a['pr_auc_se']:.4f}; calibration slope {a['calibration_slope']:.3f}")
        lines += ["", "| league | n | positives | PR-AUC (mean of origins) |", "|---|---|---|---|"]
        lines += [f"| {s['stratum']} | {s['n']:,} | {s['positives']:,} | {s['pr_auc_mean_origins']:.4f} |"
                  for s in d["strata_league"]]
        lines += ["", "| position | n | positives | PR-AUC (mean of origins) |", "|---|---|---|---|"]
        lines += [f"| {s['stratum']} | {s['n']:,} | {s['positives']:,} | {s['pr_auc_mean_origins']:.4f} |"
                  for s in d["strata_position"]]
        lines += ["", f"Reliability and decision-curve tables (origin {main}): "
                  f"`reports/experiments/{exp['id']}/`", ""]
    if "power" in result:
        lines += ["## Open decision 11 — power rule", "", "| label | SE main | SE mean of origins | ratio |",
                  "|---|---|---|---|"]
        lines += [f"| {lab} | {p['se_main']:.4f} | {p['se_mean_origins']:.4f} | {p['ratio']:.3f} |"
                  for lab, p in result["power"]["labels"].items()]
        lines += ["", f"Rule: choose mean_origins if ratio ≤ 0.8 for every decision label → "
                  f"**{result['power']['choice']}**", ""]
    if "comparison" in result:
        c = result["comparison"]
        lines += [f"## Comparison with {exp['champion']} (aggregation: {c['aggregation']})", "",
                  "| label | champion | challenger | Δ | 95% CI | slope champ → chall | conditions 1/2/3 |",
                  "|---|---|---|---|---|---|---|"]
        for lab, r in c["labels"].items():
            flags = "/".join("✓" if r[k] else "✗" for k in
                             ("cond1_delta_positive", "cond2_ci_excludes_zero", "cond3_no_slope_regression"))
            lines.append(f"| {lab} | {r['champion_pr_auc']:.4f} | {r['challenger_pr_auc']:.4f} | "
                         f"{r['delta']:+.4f} | {_fmt_ci(r['delta_ci'])} | "
                         f"{r['champion_slope']:.3f} → {r['challenger_slope']:.3f} | {flags} |")
        lines += ["", f"**Verdict: {c['verdict']}**", ""]
    return "\n".join(lines)


def evaluate(config_path: Path) -> dict:
    exp = load_experiment(config_path)
    ev_cfg = load_yaml(EVALUATION_CONFIG)
    run = load_run(exp["id"])
    if run["config_sha256"] != exp["config_sha256"]:
        raise SystemExit("config changed since training: retrain before evaluating")
    preds = load_predictions(exp["id"])
    out_dir = EXPERIMENTS_REPORTS_DIR / exp["id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {"id": exp["id"], "labels": {}}
    for label in exp["labels"]:
        rows = preds[preds["label"] == label]
        result["labels"][label] = _label_dossier(rows, ev_cfg, exp["seed"], out_dir, label)

    if exp.get("champion"):
        if ev_cfg["aggregation"] is None:
            raise SystemExit("configs/evaluation.yaml aggregation is unset (open decision 11)")
        champ_id = exp["champion"]
        champ_metrics = json.loads((run_dir(champ_id) / "metrics.json").read_text(encoding="utf-8"))
        result["comparison"] = _compare(preds, load_predictions(champ_id), result["labels"],
                                        champ_metrics, ev_cfg, exp["seed"])
    else:
        power = {}
        for label in ev_cfg["decision_labels"]:
            a = result["labels"][label]["aggregate"]
            power[label] = {"se_main": a["main"]["pr_auc_se"], "se_mean_origins": a["mean_origins"]["pr_auc_se"]}
            power[label]["ratio"] = power[label]["se_mean_origins"] / power[label]["se_main"]
        choice = "mean_origins" if all(p["ratio"] <= 0.8 for p in power.values()) else "main"
        result["power"] = {"labels": power, "choice": choice}

    (run_dir(exp["id"]) / "metrics.json").write_text(
        json.dumps(result, indent=2, default=lambda v: v.item()), encoding="utf-8")
    report = _report(exp, run, result, ev_cfg)
    (EXPERIMENTS_REPORTS_DIR / f"{exp['id']}.md").write_text(report, encoding="utf-8")
    log.info("wrote %s", EXPERIMENTS_REPORTS_DIR / f"{exp['id']}.md")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.evaluate")
    parser.add_argument("config", type=Path)
    evaluate(parser.parse_args().config)


if __name__ == "__main__":
    main()
