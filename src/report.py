"""Roll-up of evaluated experiments: reports/experiments_summary.md.

The champion lineage is replayed from the verdicts: the first experiment
without a champion is the baseline champion, and each PROMOTE replaces it.
"""

from __future__ import annotations

import json

from .config import load_yaml
from .constants import CONFIGS_DIR, REPORTS_DIR
from .experiment import EVALUATION_CONFIG, EXPERIMENTS_DATA_DIR


def main() -> None:
    ev_cfg = load_yaml(EVALUATION_CONFIG)
    how = ev_cfg["aggregation"] or "main"
    labels = ev_cfg["decision_labels"]
    rows, champion, baseline = [], None, None
    for path in sorted(EXPERIMENTS_DATA_DIR.glob("exp-*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        cfg = load_yaml(CONFIGS_DIR / f"{m['id']}.yaml")
        scores = {lab: m["labels"][lab]["aggregate"][how]["pr_auc"] for lab in labels}
        if "comparison" in m:
            verdict = m["comparison"]["verdict"]
            if verdict == "PROMOTE":
                champion = m["id"]
        else:
            verdict = "BASELINE"
            champion = champion or m["id"]
            baseline = baseline or (m["id"], scores)
        rows.append((m["id"], cfg["title"], cfg.get("champion") or "—", scores, verdict))

    lines = ["# Experiment summary", "",
             f"Aggregation: {how}; decision labels: {', '.join(labels)}.", "",
             "| id | title | vs | " + " | ".join(f"PR-AUC {lab}" for lab in labels) + " | verdict |",
             "|---|---|---|" + "---|" * len(labels) + "---|"]
    for exp_id, title, vs, scores, verdict in rows:
        lines.append(f"| {exp_id} | {title} | {vs} | " + " | ".join(f"{scores[lab]:.4f}" for lab in labels)
                     + f" | {verdict} |")
    if champion and baseline:
        champ_scores = next(s for i, _, _, s, _ in rows if i == champion)
        lines += ["", f"**Current champion: {champion}.** Cumulative Δ PR-AUC vs baseline {baseline[0]}: "
                  + ", ".join(f"{lab} {champ_scores[lab] - baseline[1][lab]:+.4f}" for lab in labels)]
    out = REPORTS_DIR / "experiments_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
