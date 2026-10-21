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


def calibration_rule_v2(exp_id: str, ev_cfg: dict) -> bool:
    """Owner amendment of 2026-10-21 applies from the configured experiment on (forward only)."""
    start = ev_cfg.get("calibration_rule_v2_from")
    return bool(start) and int(exp_id.split("-")[1]) >= int(start.split("-")[1])


def calibration_condition(champion_slope: float, challenger_slope: float, champ_draws: np.ndarray | None,
                          chall_draws: np.ndarray | None, ev_cfg: dict) -> dict:
    """Condition 3. v1: point-estimate regression <= tolerance. v2 (draws given): fail only if
    delta_cal < -tol and the bootstrap CI of delta_cal lies entirely below -tol."""
    tol = ev_cfg["max_slope_regression"]
    delta_cal = abs(champion_slope - 1) - abs(challenger_slope - 1)
    if champ_draws is None:
        return {"rule": "v1", "delta_cal": delta_cal, "cond3_no_slope_regression": -delta_cal <= tol}
    ci = metrics.percentile_ci(np.abs(champ_draws - 1) - np.abs(chall_draws - 1), ev_cfg["ci_level"])
    return {"rule": "v2", "delta_cal": delta_cal, "delta_cal_ci": ci,
            "cond3_no_slope_regression": not (delta_cal < -tol and ci[1] < -tol)}


def _compare(preds: pd.DataFrame, champ: pd.DataFrame, own: dict, champ_metrics: dict,
             ev_cfg: dict, seed: int, boot_path: Path, rule_v2: bool, decision_metric: str = "pr_auc") -> dict:
    how, main = ev_cfg["aggregation"], ev_cfg["main_origin"]
    keys = ["label", "origin", "player_id", "t"]
    result, deltas = {}, {}
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
        deltas[label] = delta_dist
        mine, theirs = own[label]["aggregate"][how], champ_metrics["labels"][label]["aggregate"][how]
        delta = mine["pr_auc"] - theirs["pr_auc"]
        ci = metrics.percentile_ci(delta_dist, ev_cfg["ci_level"])
        draws = (None, None)
        if rule_v2:
            slopes = metrics.bootstrap_calibration_slope(origins, ev_cfg["n_resamples"], seed)
            draws = tuple(_aggregate_boot(slopes[m], order, how, main) for m in ("champion", "challenger"))
        if decision_metric == "slope" and draws[0] is None:
            raise SystemExit("slope-decided loops need the bootstrap calibration rule (v2)")
        cal = calibration_condition(theirs["calibration_slope"], mine["calibration_slope"], *draws, ev_cfg)
        result[label] = {
            "champion_pr_auc": theirs["pr_auc"], "challenger_pr_auc": mine["pr_auc"],
            "delta": delta, "delta_ci": ci,
            "champion_slope": theirs["calibration_slope"], "challenger_slope": mine["calibration_slope"],
            "slope_regression": -cal["delta_cal"],
            **{k: v for k, v in cal.items() if k != "cond3_no_slope_regression"},
            "cond1_delta_positive": delta > 0,
            "cond2_ci_excludes_zero": ci[0] > 0,
            "cond3_no_slope_regression": cal["cond3_no_slope_regression"],
        }
        if decision_metric in ("brier", "slope"):
            brier = metrics.bootstrap_brier(origins, ev_cfg["n_resamples"], seed)
            brier_dist = (_aggregate_boot(brier["challenger"], order, how, main)
                          - _aggregate_boot(brier["champion"], order, how, main))
            b_mine = _aggregate(pd.Series({int(k): v["brier"] for k, v in own[label]["per_origin"].items()}), how, main)
            b_theirs = _aggregate(pd.Series({int(k): v["brier"] for k, v in
                                             champ_metrics["labels"][label]["per_origin"].items()}), how, main)
            b_ci = metrics.percentile_ci(brier_dist, ev_cfg["ci_level"])
            result[label].update({
                "decision_metric": decision_metric, "champion_brier": b_theirs, "challenger_brier": b_mine,
                "delta_brier": b_mine - b_theirs, "delta_brier_ci": b_ci,
                "guard_pr_auc_not_worse": ci[1] >= 0,
            })
        if decision_metric == "brier":
            # calibration-layer loop under PLAN decision 15 (exp-0016..exp-0019): Brier decides
            deltas[label] = brier_dist
            result[label].update({"cond1_delta_positive": b_mine - b_theirs < 0,
                                  "cond2_ci_excludes_zero": b_ci[1] < 0})
        elif decision_metric == "slope":
            # calibration-layer loop under PLAN decision 16 (exp-0020 on): change in |slope - 1| decides
            deltas[label] = np.abs(draws[0] - 1) - np.abs(draws[1] - 1)
            result[label].update({
                "cond1_delta_positive": cal["delta_cal"] > 0,
                "cond2_ci_excludes_zero": cal["delta_cal_ci"][0] > 0,
                "guard_brier_not_worse": b_ci[0] <= 0,
            })
    # kept so a deterministic seed repeat can be decided on the pooled draws
    np.savez(boot_path, **deltas)
    return {"aggregation": how, "labels": result, "verdict": verdict_from(result)}


def verdict_from(conditions: dict[str, dict]) -> str:
    """Promotion rule over one or more decision metrics (intersection)."""
    c1 = all(c["cond1_delta_positive"] for c in conditions.values())
    c2 = all(c["cond2_ci_excludes_zero"] for c in conditions.values())
    c3 = all(c["cond3_no_slope_regression"] for c in conditions.values())
    guard = all(v for c in conditions.values() for k, v in c.items() if k.startswith("guard_"))
    if c1 and c2 and c3 and guard:
        return "PROMOTE"
    if c1 and not c2:
        return "INCONCLUSIVE"
    return "REJECT"


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
        slope_rows = [(lab, r) for lab, r in c["labels"].items() if r.get("decision_metric") == "slope"]
        if slope_rows:
            lines += ["", "Calibration-layer loop: decision metric delta_cal = |b_champion - 1| - "
                      "|b_challenger - 1| (higher is better); guards PR-AUC and Brier (PLAN open decision 16).", "",
                      "| label | slope champion -> challenger | delta_cal | 95% CI | Brier delta [95% CI] "
                      "| guards PR-AUC / Brier |", "|---|---|---|---|---|---|"]
            for lab, r in slope_rows:
                guards = ("ok" if r["guard_pr_auc_not_worse"] else "FAIL") + " / " + (
                    "ok" if r["guard_brier_not_worse"] else "FAIL")
                lines.append(
                    f"| {lab} | {r['champion_slope']:.3f} -> {r['challenger_slope']:.3f} | {r['delta_cal']:+.4f} | "
                    f"[{r['delta_cal_ci'][0]:+.4f}, {r['delta_cal_ci'][1]:+.4f}] | {r['delta_brier']:+.6f} "
                    f"[{r['delta_brier_ci'][0]:+.6f}, {r['delta_brier_ci'][1]:+.6f}] | {guards} |")
        brier_rows = [(lab, r) for lab, r in c["labels"].items() if r.get("decision_metric") == "brier"]
        if brier_rows:
            lines += ["", "Calibration-layer loop: decision metric Brier (lower is better), PR-AUC guard "
                      "(PLAN open decision 15).", "",
                      "| label | Brier champion | Brier challenger | Δ | 95% CI | PR-AUC guard |", "|---|---|---|---|---|---|"]
            lines += [f"| {lab} | {r['champion_brier']:.5f} | {r['challenger_brier']:.5f} | {r['delta_brier']:+.5f} | "
                      f"[{r['delta_brier_ci'][0]:+.5f}, {r['delta_brier_ci'][1]:+.5f}] | "
                      f"{'✓' if r['guard_pr_auc_not_worse'] else '✗'} |" for lab, r in brier_rows]
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
    if exp.get("arm") == "survival":
        from .evaluate_survival import evaluate_survival

        return evaluate_survival(exp, ev_cfg, run, preds)

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
                                        champ_metrics, ev_cfg, exp["seed"],
                                        run_dir(exp["id"]) / "delta_bootstrap.npz",
                                        calibration_rule_v2(exp["id"], ev_cfg),
                                        exp.get("decision_metric", "pr_auc"))
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
