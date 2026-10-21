"""Survival-arm dossier, champion comparison and verdict (PLAN amendment A9).

Primary metric: Antolini C^td on the main origin. Condition 3 uses the slope of
the 28-day risk against y28. All bootstrap statistics of one evaluation (C^td,
IBS, calibration slope, for champion and challenger) use the same player
resamples.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import metrics, survival
from .evaluate import calibration_condition, calibration_rule_v2, verdict_from
from .experiment import EXPERIMENTS_REPORTS_DIR, load_predictions, run_dir
from .labels import SURVIVAL_SPECS

log = logging.getLogger(__name__)

RISK_HORIZONS = (7, 28)
CALIBRATION_HORIZON = 28
KEYS = ["origin", "player_id", "t"]


def _curves(rows: pd.DataFrame, horizon: int) -> np.ndarray:
    return rows[[f"S{d}" for d in survival.day_grid(horizon)]].to_numpy(float)


class _MainOrigin:
    """Arrays of the main-origin rows plus per-player aggregates for fast resampling."""

    def __init__(self, rows: pd.DataFrame, horizon: int):
        rows = rows.sort_values(["player_id", "t"]).reset_index(drop=True)
        self.rows = rows
        self.players, self.index = np.unique(rows["player_id"].to_numpy(), return_inverse=True)
        self.time, self.event = rows["time"].to_numpy(float), rows["event"].to_numpy(float)
        self.y28 = rows["y28"].to_numpy(float)
        self.horizon = horizon

    def statistics(self, S: np.ndarray, G: np.ndarray) -> dict:
        P = len(self.players)
        parts = survival.ConcordanceParts(self.time, self.event, S, self.index, P, self.horizon)
        brier = survival.brier_contributions(self.time, self.event, S, G, self.horizon)
        keep = ~np.isnan(self.y28)
        return {
            "parts": parts,
            "brier_sum": np.bincount(self.index, weights=brier, minlength=P),
            "rows": np.bincount(self.index, minlength=P).astype(float),
            "cal_y": self.y28[keep], "cal_p": 1 - S[keep, CALIBRATION_HORIZON - 1], "cal_index": self.index[keep],
        }


def _bootstrap(stats_by_model: dict[str, dict], n_players: int, n_resamples: int, seed: int,
               with_calibration: bool) -> dict[str, dict[str, np.ndarray]]:
    out = {m: {"c_td": np.empty(n_resamples), "ibs": np.empty(n_resamples), "slope": np.full(n_resamples, np.nan)}
           for m in stats_by_model}
    for start, counts in metrics.resample_counts(n_players, n_resamples, seed):
        sl = slice(start, start + len(counts))
        for m, st in stats_by_model.items():
            out[m]["c_td"][sl] = st["parts"].value(counts)
            out[m]["ibs"][sl] = (counts @ st["brier_sum"]) / (counts @ st["rows"])
            if with_calibration:
                for r, c in enumerate(counts):
                    w = c[st["cal_index"]]
                    keep = w > 0
                    out[m]["slope"][start + r] = metrics.calibration(st["cal_y"][keep], st["cal_p"][keep], w[keep])[0]
    return out


def _dossier(preds: pd.DataFrame, run: dict, horizon: int) -> dict:
    per_origin = {}
    for origin, g in preds.groupby("origin"):
        G = np.asarray(run["censoring_km"][str(origin)])
        per_origin[int(origin)] = survival.survival_dossier(
            g["time"], g["event"], _curves(g, horizon), G, horizon,
            {h: g[f"y{h}"].to_numpy(float) for h in RISK_HORIZONS})
    return per_origin


def _strata(rows: pd.DataFrame, horizon: int, column: str) -> list[dict]:
    out = []
    for stratum, g in rows.groupby(column):
        out.append({"stratum": stratum, "n": len(g), "events": int(g["event"].sum()),
                    "c_td": survival.concordance(g["time"].to_numpy(float), g["event"].to_numpy(float),
                                                 _curves(g, horizon), horizon)})
    return out


def evaluate_survival(exp: dict, ev_cfg: dict, run: dict, preds: pd.DataFrame) -> dict:
    name = exp["outcome"]
    horizon = SURVIVAL_SPECS[name]
    main = ev_cfg["main_origin"]
    level, n_res, seed = ev_cfg["ci_level"], ev_cfg["n_resamples"], exp["seed"]
    per_origin = _dossier(preds, run, horizon)

    main_rows = preds[preds["origin"] == main]
    G_main = np.asarray(run["censoring_km"][str(main)])
    mo = _MainOrigin(main_rows, horizon)
    own = mo.statistics(_curves(mo.rows, horizon), G_main)
    stats = {"challenger": own}
    comparison = None
    champ_id = exp.get("champion")
    if champ_id:
        champ_run = json.loads((run_dir(champ_id) / "run.json").read_text(encoding="utf-8"))
        champ = load_predictions(champ_id)
        both = mo.rows[KEYS + ["time", "event"]].merge(
            champ[champ["origin"] == main], on=KEYS, how="left", suffixes=("", "_champion"), validate="one_to_one")
        if both["time_champion"].isna().any() or not (both["time"].equals(both["time_champion"])
                                                      and both["event"].equals(both["event_champion"])):
            raise SystemExit("challenger and champion were not scored on identical rows and outcomes")
        stats["champion"] = mo.statistics(_curves(both, horizon), np.asarray(champ_run["censoring_km"][str(main)]))
    draws = _bootstrap(stats, len(mo.players), n_res, seed, with_calibration=bool(champ_id))

    main_point = per_origin[main]
    aggregate = {
        "c_td": main_point["c_td"], "c_td_ci": metrics.percentile_ci(draws["challenger"]["c_td"], level),
        "ibs": main_point["ibs"], "ibs_ci": metrics.percentile_ci(draws["challenger"]["ibs"], level),
        "calibration_slope": main_point[f"risk{CALIBRATION_HORIZON}_calibration_slope"],
    }
    if champ_id:
        champ_metrics = json.loads((run_dir(champ_id) / "metrics.json").read_text(encoding="utf-8"))
        theirs = champ_metrics["aggregate"]
        delta_dist = draws["challenger"]["c_td"] - draws["champion"]["c_td"]
        delta = aggregate["c_td"] - theirs["c_td"]
        ci = metrics.percentile_ci(delta_dist, level)
        slope_draws = (draws["champion"]["slope"], draws["challenger"]["slope"])
        if not calibration_rule_v2(exp["id"], ev_cfg):
            slope_draws = (None, None)
        cal = calibration_condition(theirs["calibration_slope"], aggregate["calibration_slope"], *slope_draws, ev_cfg)
        conditions = {"c_td": {
            "champion_c_td": theirs["c_td"], "challenger_c_td": aggregate["c_td"],
            "delta": delta, "delta_ci": ci,
            "delta_ibs": aggregate["ibs"] - theirs["ibs"],
            "delta_ibs_ci": metrics.percentile_ci(draws["challenger"]["ibs"] - draws["champion"]["ibs"], level),
            "champion_slope": theirs["calibration_slope"], "challenger_slope": aggregate["calibration_slope"],
            **{k: v for k, v in cal.items() if k != "cond3_no_slope_regression"},
            "cond1_delta_positive": delta > 0,
            "cond2_ci_excludes_zero": ci[0] > 0,
            "cond3_no_slope_regression": cal["cond3_no_slope_regression"],
        }}
        np.savez(run_dir(exp["id"]) / "delta_bootstrap.npz", c_td=delta_dist)
        comparison = {"aggregation": "main", "labels": conditions, "verdict": verdict_from(conditions)}

    result = {"id": exp["id"], "arm": "survival", "outcome": name, "per_origin": per_origin,
              "aggregate": aggregate,
              "strata_league": _strata(main_rows, horizon, "competition_id"),
              "strata_position": _strata(main_rows, horizon, "position_group")}
    if comparison:
        result["comparison"] = comparison
    (run_dir(exp["id"]) / "metrics.json").write_text(
        json.dumps(result, indent=2, default=lambda v: v.item()), encoding="utf-8")
    (EXPERIMENTS_REPORTS_DIR / f"{exp['id']}.md").write_text(_report(exp, run, result, main), encoding="utf-8")
    log.info("wrote %s", EXPERIMENTS_REPORTS_DIR / f"{exp['id']}.md")
    return result


_COLUMNS = ["n", "events", "event_rate", "c_td", "ibs", "risk7_pr_auc", "risk28_pr_auc",
            "risk28_calibration_slope", "risk28_calibration_intercept", "risk28_brier", "risk28_mean",
            "y28_prevalence"]


def _report(exp: dict, run: dict, result: dict, main: int) -> str:
    a = result["aggregate"]
    lines = [
        f"# {exp['id']} — {exp['title']}", "",
        f"- Stage {exp['stage']} (survival arm, outcome `{result['outcome']}`); hypothesis: {exp['hypothesis']}",
        f"- Champion compared against: {exp.get('champion') or 'none (first survival champion of record)'}",
        f"- Config: `{run['config_path']}` (sha256 {run['config_sha256'][:12]}); seed {run['seed']}",
        f"- Data: `{run['data_version']}` sha256 {run['panel_sha256'][:12]}; "
        f"origins {sorted({s['origin'] for s in run['splits']})} (main {main})",
        f"- Code: commit {run['git']['commit'][:10]}{' (dirty)' if run['git']['dirty'] else ''}; "
        f"leakage tests at run time: {run['leakage_tests']}",
        f"- Environment: {', '.join(f'{k} {v}' for k, v in run['environment'].items())}",
        "", "| origin | " + " | ".join(_COLUMNS) + " |", "|" + "---|" * (len(_COLUMNS) + 1),
    ]
    for o, r in sorted(result["per_origin"].items()):
        cells = [f"{int(r[c]):,}" if c in ("n", "events") else f"{r[c]:.4f}" for c in _COLUMNS]
        lines.append(f"| {o} | " + " | ".join(cells) + " |")
    lines += ["", f"- C^td (main origin): {a['c_td']:.4f} [{a['c_td_ci'][0]:.4f}, {a['c_td_ci'][1]:.4f}]",
              f"- IBS days 1–89 (main origin): {a['ibs']:.4f} [{a['ibs_ci'][0]:.4f}, {a['ibs_ci'][1]:.4f}]",
              f"- 28-day risk calibration slope (main origin): {a['calibration_slope']:.3f}", "",
              "| league | n | events | C^td |", "|---|---|---|---|"]
    lines += [f"| {s['stratum']} | {s['n']:,} | {s['events']:,} | {s['c_td']:.4f} |" for s in result["strata_league"]]
    lines += ["", "| position | n | events | C^td |", "|---|---|---|---|"]
    lines += [f"| {s['stratum']} | {s['n']:,} | {s['events']:,} | {s['c_td']:.4f} |" for s in result["strata_position"]]
    if "comparison" in result:
        c = result["comparison"]["labels"]["c_td"]
        cal = (f"Δ_cal {c['delta_cal']:+.3f}" + (f" [{c['delta_cal_ci'][0]:+.3f}, {c['delta_cal_ci'][1]:+.3f}]"
                                                  if "delta_cal_ci" in c else "") + f" (rule {c['rule']})")
        flags = "/".join("✓" if c[k] else "✗" for k in
                         ("cond1_delta_positive", "cond2_ci_excludes_zero", "cond3_no_slope_regression"))
        lines += ["", f"## Comparison with {exp['champion']} (main origin)", "",
                  f"- C^td: champion {c['champion_c_td']:.4f} → challenger {c['challenger_c_td']:.4f}, "
                  f"Δ {c['delta']:+.4f} [{c['delta_ci'][0]:+.4f}, {c['delta_ci'][1]:+.4f}]",
                  f"- IBS: Δ {c['delta_ibs']:+.4f} [{c['delta_ibs_ci'][0]:+.4f}, {c['delta_ibs_ci'][1]:+.4f}] "
                  "(negative = better; reported, not gating)",
                  f"- 28-day calibration slope: {c['champion_slope']:.3f} → {c['challenger_slope']:.3f}; {cal}",
                  f"- Conditions 1/2/3: {flags}", "", f"**Verdict: {result['comparison']['verdict']}**", ""]
    return "\n".join(lines) + "\n"
