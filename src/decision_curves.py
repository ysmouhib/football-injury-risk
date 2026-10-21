"""Decision-curve analysis on the main validation origin (PLAN 4.3, amendment A10).

Run as: python -m src.decision_curves <champion> [comparator ...]
Comparators may be classification experiments (probability p) or survival
experiments (risk 1 - S(h)). Net benefit NB(pt) = TP/n - FP/n * pt / (1 - pt);
the champion's NB and its margin over max(treat-all, treat-none) get
player-cluster bootstrap CIs. Descriptive only.
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

KEYS = ["player_id", "t"]


def _risk(exp_id: str, label: str, origin: int) -> pd.DataFrame:
    preds = load_predictions(exp_id)
    if "label" in preds and label in set(preds["label"]):
        rows = preds[(preds["label"] == label) & (preds["origin"] == origin)]
        return rows[KEYS].assign(risk=rows["p"].to_numpy())
    horizon = int(label[1:])
    rows = preds[preds["origin"] == origin]
    return rows[KEYS].assign(risk=1 - rows[f"S{horizon}"].to_numpy())


def net_benefit_parts(y: np.ndarray, risk: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Per-row NB contributions (n, len(pts)); NB = column means."""
    treat = risk[:, None] >= pts[None, :]
    odds = pts / (1 - pts)
    return treat * (y[:, None] - (1 - y[:, None]) * odds[None, :])


def analyse(champion: str, comparators: list[str]) -> dict:
    ev_cfg = load_yaml(EVALUATION_CONFIG)
    main, n_res, seed = ev_cfg["main_origin"], ev_cfg["n_resamples"], load_run(champion)["seed"]
    pts = metrics.DECISION_THRESHOLDS
    out = {"champion": champion, "comparators": comparators, "origin": main, "thresholds": pts.tolist(), "labels": {}}
    for label in ev_cfg["decision_labels"]:
        base = load_predictions(champion)
        base = base[(base["label"] == label) & (base["origin"] == main)].sort_values(KEYS).reset_index(drop=True)
        y = base["y"].to_numpy(float)
        risks = {champion: base["p"].to_numpy()}
        for comp in comparators:
            r = base[KEYS].merge(_risk(comp, label, main), on=KEYS, how="left", validate="one_to_one")
            if r["risk"].isna().any():
                raise SystemExit(f"{comp} does not score every {label} row of {champion}")
            risks[comp] = r["risk"].to_numpy()
        curves = {m: net_benefit_parts(y, r, pts).mean(axis=0) for m, r in risks.items()}
        treat_all = net_benefit_parts(y, np.ones(len(y)), pts).mean(axis=0)

        players, index = np.unique(base["player_id"].to_numpy(), return_inverse=True)
        P = len(players)
        champ_sum = np.stack([np.bincount(index, weights=col, minlength=P)
                              for col in net_benefit_parts(y, risks[champion], pts).T], axis=1)
        all_sum = np.stack([np.bincount(index, weights=col, minlength=P)
                            for col in net_benefit_parts(y, np.ones(len(y)), pts).T], axis=1)
        rows = np.bincount(index, minlength=P).astype(float)
        nb_draws, margin_draws = [], []
        for _, c in metrics.resample_counts(P, n_res, seed):
            n = (c @ rows)[:, None]
            nb, nb_all = (c @ champ_sum) / n, (c @ all_sum) / n
            nb_draws.append(nb)
            margin_draws.append(nb - np.maximum(nb_all, 0.0))
        nb_draws, margin_draws = np.vstack(nb_draws), np.vstack(margin_draws)
        nb_ci = np.percentile(nb_draws, [2.5, 97.5], axis=0)
        margin_ci = np.percentile(margin_draws, [2.5, 97.5], axis=0)
        useful = pts[margin_ci[0] > 0]
        out["labels"][label] = {
            "prevalence": float(y.mean()), "rows": len(y),
            "net_benefit": {m: v.tolist() for m, v in curves.items()},
            "treat_all": treat_all.tolist(),
            "champion_nb_ci": nb_ci.T.tolist(), "champion_margin_ci": margin_ci.T.tolist(),
            "useful_range": [float(useful.min()), float(useful.max())] if len(useful) else None,
            "useful_thresholds": useful.tolist(),
        }
    return out


def _report(result: dict) -> str:
    champ, comps = result["champion"], result["comparators"]
    lines = [f"# Decision curves — champion {champ}, origin {result['origin']}", "",
             "Net benefit per 1 player-week; margin = champion NB − max(treat-all, treat-none), "
             "player-cluster bootstrap 95% CI.", ""]
    for label, r in result["labels"].items():
        rng = r["useful_range"]
        lines += [f"## {label} (prevalence {r['prevalence']:.4f}, {r['rows']:,} rows)", "",
                  f"Useful range (margin CI > 0): {f'{rng[0]:.2f}–{rng[1]:.2f}' if rng else 'none'}"
                  + (f" — thresholds {', '.join(f'{x:.2f}' for x in r['useful_thresholds'])}" if rng else ""), "",
                  "| pt | " + " | ".join([champ] + comps) + " | treat-all | champion margin [95% CI] |",
                  "|---|" + "---|" * (len(comps) + 3)]
        for i, pt in enumerate(result["thresholds"]):
            cells = [f"{r['net_benefit'][m][i]:+.5f}" for m in [champ] + comps]
            lo, hi = r["champion_margin_ci"][i]
            margin = r["net_benefit"][champ][i] - max(r["treat_all"][i], 0.0)
            lines.append(f"| {pt:.2f} | " + " | ".join(cells) + f" | {r['treat_all'][i]:+.5f} | "
                         f"{margin:+.5f} [{lo:+.5f}, {hi:+.5f}] |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.decision_curves")
    parser.add_argument("champion")
    parser.add_argument("comparators", nargs="*")
    args = parser.parse_args()
    result = analyse(args.champion, args.comparators)
    (run_dir(args.champion) / "decision_curves.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (REPORTS_DIR / f"decision_curves_{args.champion}.md").write_text(_report(result), encoding="utf-8")
    print(f"wrote {REPORTS_DIR / f'decision_curves_{args.champion}.md'}")


if __name__ == "__main__":
    main()
