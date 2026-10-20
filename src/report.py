"""Roll-up of evaluated experiments: reports/experiments_summary.md.

The champion lineage is replayed from the verdicts: the first experiment
without a champion is the baseline champion, and each PROMOTE replaces it. A
later run whose config differs from an INCONCLUSIVE one only in id, title and
seed is its protocol repeat; the loop is then decided by the repeat (PROMOTE
if the repeat promotes, otherwise REJECT).
"""

from __future__ import annotations

import json

from .config import load_yaml
from .constants import CONFIGS_DIR, REPORTS_DIR
from .experiment import EVALUATION_CONFIG, EXPERIMENTS_DATA_DIR

_RUN_ONLY_KEYS = {"id", "title", "seed"}


def _core(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k not in _RUN_ONLY_KEYS}


def main() -> None:
    ev_cfg = load_yaml(EVALUATION_CONFIG)
    how = ev_cfg["aggregation"] or "main"
    labels = ev_cfg["decision_labels"]
    runs = []
    for path in sorted(EXPERIMENTS_DATA_DIR.glob("exp-*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        cfg = load_yaml(CONFIGS_DIR / f"{m['id']}.yaml")
        runs.append({
            "id": m["id"], "cfg": cfg,
            "scores": {lab: m["labels"][lab]["aggregate"][how]["pr_auc"] for lab in labels},
            "verdict": m["comparison"]["verdict"] if "comparison" in m else "BASELINE",
        })

    decisions, repeat_of = {}, {}
    for i, run in enumerate(runs):
        if run["verdict"] != "INCONCLUSIVE" or run["id"] in repeat_of:
            continue
        repeat = next((r for r in runs[i + 1:] if _core(r["cfg"]) == _core(run["cfg"])), None)
        if repeat:
            repeat_of[repeat["id"]] = run["id"]
            decisions[run["id"]] = decisions[repeat["id"]] = (
                "PROMOTE" if repeat["verdict"] == "PROMOTE" else "REJECT")

    champion, baseline = None, None
    for run in runs:
        decision = decisions.get(run["id"], run["verdict"])
        if run["verdict"] == "BASELINE":
            champion = champion or run["id"]
            baseline = baseline or run
        elif decision == "PROMOTE" and run["verdict"] == "PROMOTE":
            champion = run["id"]

    lines = ["# Experiment summary", "",
             f"Aggregation: {how}; decision labels: {', '.join(labels)}.", "",
             "| id | title | vs | " + " | ".join(f"PR-AUC {lab}" for lab in labels)
             + " | run verdict | loop decision |",
             "|---|---|---|" + "---|" * len(labels) + "---|---|"]
    for run in runs:
        note = f" (repeat of {repeat_of[run['id']]})" if run["id"] in repeat_of else ""
        decision = decisions.get(run["id"], run["verdict"] if run["verdict"] != "INCONCLUSIVE" else "pending repeat")
        lines.append(f"| {run['id']} | {run['cfg']['title']} | {run['cfg'].get('champion') or '—'} | "
                     + " | ".join(f"{run['scores'][lab]:.4f}" for lab in labels)
                     + f" | {run['verdict']}{note} | {decision} |")
    if champion and baseline:
        champ = next(r for r in runs if r["id"] == champion)
        lines += ["", f"**Current champion: {champion}.** Cumulative Δ PR-AUC vs baseline {baseline['id']}: "
                  + ", ".join(f"{lab} {champ['scores'][lab] - baseline['scores'][lab]:+.4f}" for lab in labels)]
    out = REPORTS_DIR / "experiments_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
