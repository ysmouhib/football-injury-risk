"""Roll-up of evaluated experiments: reports/experiments_summary.md.

The champion lineage is replayed from the verdicts: the first experiment
without a champion is the baseline champion, and each PROMOTE replaces it. A
later run whose config differs from an INCONCLUSIVE one only in id, title and
seed is its protocol repeat, and the loop is decided at the repeat:
- stochastic learner (predictions differ): the repeat's own verdict;
- deterministic learner (bit-identical predictions): condition 2 is judged on
  the pooled bootstrap draws of both runs, conditions 1 and 3 as in the repeat.
"""

from __future__ import annotations

import json

import numpy as np

from . import metrics
from .config import load_yaml
from .constants import CONFIGS_DIR, REPORTS_DIR
from .experiment import EVALUATION_CONFIG, EXPERIMENTS_DATA_DIR, load_predictions, run_dir

_RUN_ONLY_KEYS = {"id", "title", "seed"}


def _core(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k not in _RUN_ONLY_KEYS}


def _loop_decision(original: dict, repeat: dict, ev_cfg: dict) -> tuple[str, str]:
    """(decision, how it was reached) for an INCONCLUSIVE run and its repeat."""
    a, b = load_predictions(original["id"]), load_predictions(repeat["id"])
    if not a.equals(b):
        return ("PROMOTE" if repeat["verdict"] == "PROMOTE" else "REJECT"), "repeat verdict"
    draws = [np.load(run_dir(r["id"]) / "delta_bootstrap.npz") for r in (original, repeat)]
    labels = repeat["metrics"]["comparison"]["labels"]
    passed, cis = True, []
    for label, conds in labels.items():
        lo, hi = metrics.percentile_ci(np.concatenate([d[label] for d in draws]), ev_cfg["ci_level"])
        cis.append(f"{label} [{lo:.4f}, {hi:.4f}]")
        passed &= conds["cond1_delta_positive"] and conds["cond3_no_slope_regression"] and lo > 0
    return ("PROMOTE" if passed else "REJECT"), "pooled bootstrap " + ", ".join(cis)


def _arm_section(runs: list[dict], metric_names: list[str], ev_cfg: dict, title: str) -> list[str]:
    decisions, notes, repeat_of, has_repeat = {}, {}, {}, set()
    for i, run in enumerate(runs):
        if run["verdict"] != "INCONCLUSIVE" or run["id"] in repeat_of:
            continue
        repeat = next((r for r in runs[i + 1:] if _core(r["cfg"]) == _core(run["cfg"])), None)
        if repeat:
            repeat_of[repeat["id"]] = run["id"]
            has_repeat.add(run["id"])
            decision, note = _loop_decision(run, repeat, ev_cfg)
            decisions[run["id"]] = decisions[repeat["id"]] = decision
            notes[repeat["id"]] = note

    champion, baseline = None, None
    for run in runs:
        if run["verdict"] == "BASELINE":
            champion = champion or run["id"]
            baseline = baseline or run
        elif decisions.get(run["id"], run["verdict"]) == "PROMOTE" and run["id"] not in has_repeat:
            champion = run["id"]

    lines = [f"## {title}", "",
             "| id | title | vs | " + " | ".join(metric_names) + " | run verdict | loop decision |",
             "|---|---|---|" + "---|" * len(metric_names) + "---|---|"]
    for run in runs:
        note = f" (repeat of {repeat_of[run['id']]})" if run["id"] in repeat_of else ""
        pending = run["verdict"] if run["verdict"] != "INCONCLUSIVE" else "pending repeat"
        decision = decisions.get(run["id"], pending)
        if run["id"] in notes:
            decision += f" — {notes[run['id']]}"
        lines.append(f"| {run['id']} | {run['cfg']['title']} | {run['cfg'].get('champion') or '—'} | "
                     + " | ".join(f"{run['scores'][k]:.4f}" for k in metric_names)
                     + f" | {run['verdict']}{note} | {decision} |")
    if champion and baseline:
        champ = next(r for r in runs if r["id"] == champion)
        lines += ["", f"**Current champion: {champion}.** Cumulative Δ vs baseline {baseline['id']}: "
                  + ", ".join(f"{k} {champ['scores'][k] - baseline['scores'][k]:+.4f}" for k in metric_names)]
    return lines + [""]


def main() -> None:
    ev_cfg = load_yaml(EVALUATION_CONFIG)
    how = ev_cfg["aggregation"] or "main"
    labels = ev_cfg["decision_labels"]
    arms = {"classification": [], "survival": []}
    for path in sorted(EXPERIMENTS_DATA_DIR.glob("exp-*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        cfg = load_yaml(CONFIGS_DIR / f"{m['id']}.yaml")
        arm = m.get("arm", "classification")
        if arm == "survival":
            scores = {"C^td": m["aggregate"]["c_td"], "IBS": m["aggregate"]["ibs"]}
        else:
            scores = {f"PR-AUC {lab}": m["labels"][lab]["aggregate"][how]["pr_auc"] for lab in labels}
        arms[arm].append({"id": m["id"], "cfg": cfg, "metrics": m, "scores": scores,
                          "verdict": m["comparison"]["verdict"] if "comparison" in m else "BASELINE"})

    lines = ["# Experiment summary", "",
             f"Aggregation: {how}; classification decision labels: {', '.join(labels)}; "
             "survival decision metric: C^td (main origin).", ""]
    lines += _arm_section(arms["classification"], [f"PR-AUC {lab}" for lab in labels], ev_cfg,
                          "Arm A — classification")
    if arms["survival"]:
        lines += _arm_section(arms["survival"], ["C^td", "IBS"], ev_cfg, "Arm B — survival")
    out = REPORTS_DIR / "experiments_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
