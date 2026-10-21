"""Stage 6 final run: one evaluation of the pre-registered champion on the final holdout.

Rehearsal (no holdout access; the identical code path on the main validation
origin, where every prediction must equal the stored experiment predictions):
    python -m src.final_run --rehearsal
The one-shot holdout evaluation (pre-registered in FINAL_RUN.md):
    python -m src.final_run --expected-commit <sha>
The holdout run refuses to start unless HEAD is the pre-registered commit, the
tracked tree is clean, the feature store matches its registered hash and the
leakage tests pass; it writes its one-shot marker before any holdout row is
scored and never runs again.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import conformal, decision_curves, metrics, splits, survival
from .config import load_yaml, stable_hash
from .constants import CONFIGS_DIR, DATA_DIR, REPORTS_DIR
from .experiment import environment, git_state, load_experiment, load_predictions, panel_path
from .labels import label_horizon
from .models import build_model, fit_categories, predict_risk, prepare_X
from .panel import content_hash
from .train import _calibrate, leakage_tests_pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FINAL_CONFIG = CONFIGS_DIR / "final_run.yaml"
FINAL_DIR = DATA_DIR / "final"
MARKER = FINAL_DIR / "holdout_evaluated.json"
KEYS = ["player_id", "t"]
SURVIVAL_HORIZONS = (7, 28)


# --- predictions ------------------------------------------------------------------

def _split(panel: pd.DataFrame, cfg: dict, label: str, rehearsal: bool) -> splits.Split:
    if rehearsal:
        return splits.origin_split(panel, cfg, load_yaml(CONFIGS_DIR / "evaluation.yaml")["main_origin"], label)
    return splits.final_split(panel, cfg, label, label_horizon(label), allow_holdout=True)


def classifier(exp_id: str, panel: pd.DataFrame, label: str, rehearsal: bool, roles: tuple[str, ...]):
    """Evaluation rows with p / p_raw, calibration half-B frame (or None), calibration record."""
    exp = load_experiment(CONFIGS_DIR / f"{exp_id}.yaml")
    cfg, numeric, categorical = exp["data"], exp["numeric"], exp["categorical"]
    train_first = exp.get("train_first_season", cfg["seasons"]["first"])
    split = _split(panel, cfg, label, rehearsal)
    tr = panel.iloc[split.train_idx]
    tr = tr[tr["season"] >= train_first]
    ev = panel.iloc[split.eval_idx]
    categories = fit_categories(tr, categorical)
    model = build_model(exp["model"], numeric, categorical, exp["seed"])
    model.fit(prepare_X(tr, numeric, categorical, categories), tr[label].astype(int))
    p_raw = predict_risk(model, prepare_X(ev, numeric, categorical, categories))
    p, half_b, record = p_raw, None, {}
    if exp.get("calibration"):
        p, record, half_b = _calibrate(exp, panel, label, split.eval_season, p_raw, train_first, (model, categories),
                                       roles, allow_holdout=not rehearsal)
    record.update({"train_rows": len(tr), "train_players": int(tr["player_id"].nunique()),
                   "train_cutoff_t": str(split.train_cutoff_t.date()), "eval_rows": len(ev),
                   "eval_players": int(ev["player_id"].nunique())})
    return ev.assign(p=p, p_raw=p_raw, y=ev[label].astype(int).to_numpy()), half_b, record


def survival_model(exp_id: str, panel: pd.DataFrame, rehearsal: bool):
    exp = load_experiment(CONFIGS_DIR / f"{exp_id}.yaml")
    cfg, numeric, categorical = exp["data"], exp["numeric"], exp["categorical"]
    name = exp["outcome"]
    horizon = label_horizon(f"{name}_event")
    split = _split(panel, cfg, f"{name}_event", rehearsal)
    tr, ev = panel.iloc[split.train_idx], panel.iloc[split.eval_idx]
    categories = fit_categories(tr, categorical)
    model = survival.build_survival_model(exp["model"], numeric, categorical, exp["seed"])
    t_tr, e_tr = tr[f"{name}_time"].to_numpy(float), tr[f"{name}_event"].to_numpy(float)
    model.fit(prepare_X(tr, numeric, categorical, categories), t_tr, e_tr)
    days = survival.day_grid(horizon)
    S = model.survival(prepare_X(ev, numeric, categorical, categories), days).astype(np.float32)
    G = survival.censoring_survival(t_tr, e_tr, days)
    rows = ev.assign(time=ev[f"{name}_time"].to_numpy(), event=ev[f"{name}_event"].to_numpy())
    record = {"train_rows": len(tr), "eval_rows": len(ev), "train_cutoff_t": str(split.train_cutoff_t.date())}
    return rows.reset_index(drop=True), S, G, horizon, record


# --- metrics ------------------------------------------------------------------------

def _dossier(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> dict:
    sens, thr = metrics.operating_point(y, p, w=w)
    slope, intercept = metrics.calibration(y, p, w)
    ww = np.ones(len(y)) if w is None else w
    return {
        "pr_auc": metrics.pr_auc(y, p, w), "roc_auc": float(roc_auc_score(y, p, sample_weight=w)),
        "sens_at_90spec": sens, "mcc_at_90spec": metrics.mcc_at(y, p, thr, w),
        "brier": float(np.sum(ww * (p - y) ** 2) / ww.sum()),
        "calibration_slope": slope, "calibration_intercept": intercept,
        "mean_predicted": float(np.sum(ww * p) / ww.sum()), "prevalence": float(np.sum(ww * y) / ww.sum()),
    }


def _players(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return np.unique(rows["player_id"].to_numpy(), return_inverse=True)


def dossier_with_ci(rows: pd.DataFrame, n: int, seed: int) -> dict:
    y, p = rows["y"].to_numpy(float), rows["p"].to_numpy(float)
    players, index = _players(rows)
    point = _dossier(y, p)
    draws = {k: [] for k in point}
    for _, counts in metrics.resample_counts(len(players), n, seed):
        for c in counts:
            w = c[index]
            keep = w > 0
            for k, v in _dossier(y[keep], p[keep], w[keep]).items():
                draws[k].append(v)
    return {"n": len(rows), "positives": int(y.sum()),
            **{k: {"value": v, "ci": metrics.percentile_ci(np.asarray(draws[k]))} for k, v in point.items()},
            "reliability": metrics.reliability_table(y, p).to_dict("records")}


def paired_pr_auc(rows: dict[str, pd.DataFrame], champion: str, n: int, seed: int) -> dict:
    base = rows[champion]
    scores = {m: base[KEYS].merge(r[KEYS + ["p"]], on=KEYS, how="left")["p"].to_numpy() for m, r in rows.items()}
    origin = metrics.OriginScores(0, base["player_id"].to_numpy(), base["y"].to_numpy(), scores)
    boot = metrics.bootstrap_pr_auc([origin], n, seed)
    return {m: {"pr_auc": metrics.pr_auc(base["y"], scores[m]), "pr_auc_ci": metrics.percentile_ci(boot[m][:, 0]),
                "delta_vs_champion": metrics.pr_auc(base["y"], scores[champion]) - metrics.pr_auc(base["y"], scores[m]),
                "delta_ci": metrics.percentile_ci(boot[champion][:, 0] - boot[m][:, 0])}
            for m in rows if m != champion}


def decision_curve(rows: pd.DataFrame, risks: dict[str, np.ndarray], champion: str, n: int, seed: int) -> dict:
    y = rows["y"].to_numpy(float)
    pts = metrics.DECISION_THRESHOLDS
    curves = {m: decision_curves.net_benefit_parts(y, r, pts).mean(axis=0).tolist() for m, r in risks.items()}
    treat_all = decision_curves.net_benefit_parts(y, np.ones(len(y)), pts)
    champ = decision_curves.net_benefit_parts(y, risks[champion], pts)
    players, index = _players(rows)
    sums = lambda parts: np.stack([np.bincount(index, weights=col, minlength=len(players)) for col in parts.T], 1)  # noqa: E731
    cs, alls = sums(champ), sums(treat_all)
    per = np.bincount(index, minlength=len(players)).astype(float)
    nb, margin = [], []
    for _, c in metrics.resample_counts(len(players), n, seed):
        d = (c @ per)[:, None]
        nb.append((c @ cs) / d)
        margin.append((c @ cs) / d - np.maximum((c @ alls) / d, 0))
    nb_ci = np.percentile(np.vstack(nb), [2.5, 97.5], axis=0).T
    margin_ci = np.percentile(np.vstack(margin), [2.5, 97.5], axis=0).T
    useful = pts[margin_ci[:, 0] > 0]
    return {"thresholds": pts.tolist(), "net_benefit": curves, "treat_all": treat_all.mean(axis=0).tolist(),
            "champion_nb_ci": nb_ci.tolist(), "champion_margin_ci": margin_ci.tolist(),
            "useful_range": [float(useful.min()), float(useful.max())] if len(useful) else None}


def conformal_coverage(rows: pd.DataFrame, half_b: pd.DataFrame, alphas: list[float], n: int, seed: int) -> list[dict]:
    y, p = rows["y"].to_numpy(), rows["p"].to_numpy()
    players, index = _players(rows)
    out = []
    for alpha in alphas:
        q0, q1 = conformal.thresholds(half_b["y"].to_numpy(), half_b["p"].to_numpy(), alpha, mondrian=True)
        has0, has1 = conformal.prediction_sets(p, q0, q1)
        covered = np.where(y == 1, has1, has0)
        draws = conformal._coverage_draws(covered, index, len(players), n, seed)
        tier = np.select([has0 & ~has1, has0 & has1, ~has0 & has1], ["low", "uncertain", "high"], "empty")
        out.append({"alpha": alpha, "coverage": float(covered.mean()), "coverage_ci": metrics.percentile_ci(draws),
                    "coverage_injured": float(covered[y == 1].mean()) if (y == 1).any() else None,
                    "coverage_uninjured": float(covered[y == 0].mean()),
                    "tiers": {t: {"share": float((tier == t).mean()),
                                  "injury_rate": float(y[tier == t].mean()) if (tier == t).any() else None}
                              for t in ("low", "uncertain", "high", "empty")}})
    return out


def strata(rows: pd.DataFrame, column: str) -> list[dict]:
    out = []
    for value, g in rows.groupby(column):
        if 0 < g["y"].sum() < len(g):
            slope, intercept = metrics.calibration(g["y"], g["p"])
            out.append({"stratum": value, "n": len(g), "positives": int(g["y"].sum()),
                        "prevalence": float(g["y"].mean()), "pr_auc": metrics.pr_auc(g["y"], g["p"]),
                        "calibration_slope": slope, "calibration_intercept": intercept})
    return out


def survival_metrics(rows: pd.DataFrame, S: np.ndarray, G: np.ndarray, horizon: int, n: int, seed: int) -> dict:
    time_, event = rows["time"].to_numpy(float), rows["event"].to_numpy(float)
    point = survival.survival_dossier(time_, event, S, G, horizon,
                                      {h: rows[f"y{h}"].to_numpy(float) for h in SURVIVAL_HORIZONS})
    players, index = _players(rows)
    parts = survival.ConcordanceParts(time_, event, S, index, len(players), horizon)
    brier = survival.brier_contributions(time_, event, S, G, horizon)
    bsum = np.bincount(index, weights=brier, minlength=len(players))
    per = np.bincount(index, minlength=len(players)).astype(float)
    c_td, ibs = [], []
    for _, c in metrics.resample_counts(len(players), n, seed):
        c_td.append(parts.value(c))
        ibs.append((c @ bsum) / (c @ per))
    return {**point, "c_td_ci": metrics.percentile_ci(np.concatenate(c_td)),
            "ibs_ci": metrics.percentile_ci(np.concatenate(ibs))}


# --- run -------------------------------------------------------------------------------

def _guard(spec: dict, expected_commit: str | None, rehearsal: bool, panel: pd.DataFrame) -> dict:
    digest = content_hash(panel)
    if digest != spec["panel_sha256"]:
        raise SystemExit("feature store does not match the pre-registered hash")
    git = git_state()
    if not rehearsal:
        if MARKER.exists():
            raise SystemExit(f"the holdout has already been evaluated ({MARKER}); the final run is one-shot")
        if git["dirty"] or git["commit"] != expected_commit:
            raise SystemExit(f"HEAD {git['commit'][:10]} (dirty={git['dirty']}) is not the pre-registered commit")
    if not leakage_tests_pass():
        raise SystemExit("leakage tests failing: refusing to run")
    return {"panel_sha256": digest, "git": git}


def run(rehearsal: bool, expected_commit: str | None) -> dict:
    spec = load_yaml(FINAL_CONFIG)
    exp = load_experiment(CONFIGS_DIR / f"{spec['champion']}.yaml")
    panel = pd.read_parquet(panel_path(exp["data"]))
    provenance = _guard(spec, expected_commit, rehearsal, panel)
    roles = ("train",) if rehearsal else tuple(spec["training_roles"])
    out_dir = FINAL_DIR / ("rehearsal" if rehearsal else "holdout")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rehearsal:
        MARKER.write_text(json.dumps({"started": time.strftime("%Y-%m-%dT%H:%M:%S"), **provenance}, indent=2),
                          encoding="utf-8")
    n, seed = spec["n_resamples"], spec["seed"]
    models = [spec["champion"], *spec["comparators"]]
    result = {"mode": "rehearsal" if rehearsal else "holdout", "spec": spec,
              "spec_sha256": stable_hash(spec), "environment": environment(), **provenance,
              "records": {}, "labels": {}}

    surv_rows, S, G, horizon, record = survival_model(spec["survival_champion"], panel, rehearsal)
    result["records"][spec["survival_champion"]] = record
    for label in spec["labels"]:
        rows, half_b = {}, None
        for m in models:
            ev, hb, record = classifier(m, panel, label, rehearsal, roles)
            rows[m], result["records"][f"{m}-{label}"] = ev, record
            if m == spec["champion"]:
                half_b = hb
            ev[KEYS + ["p", "p_raw", "y"]].to_parquet(out_dir / f"predictions_{m}_{label}.parquet", index=False)
        h = int(label[1:])
        surv_risk = rows[spec["champion"]][KEYS].merge(
            surv_rows[KEYS].assign(r=1 - S[:, h - 1]), on=KEYS, how="left")["r"].to_numpy()
        result["labels"][label] = {}
        for pop, pspec in spec["populations"].items():
            mask = {m: r["t"].dt.month.isin(pspec["months"]).to_numpy() for m, r in rows.items()}
            sub = {m: r[mask[m]].reset_index(drop=True) for m, r in rows.items()}
            champ = sub[spec["champion"]]
            risks = {m: champ[KEYS].merge(r[KEYS + ["p"]], on=KEYS, how="left")["p"].to_numpy() for m, r in sub.items()}
            risks[spec["survival_champion"]] = surv_risk[mask[spec["champion"]]]
            result["labels"][label][pop] = {
                "champion": dossier_with_ci(champ, n, seed),
                "comparators": paired_pr_auc(sub, spec["champion"], n, seed),
                "comparator_dossiers": {m: _dossier(r["y"].to_numpy(float), r["p"].to_numpy(float))
                                        for m, r in sub.items() if m != spec["champion"]},
                "decision_curve": decision_curve(champ, risks, spec["champion"], n, seed),
                "conformal": conformal_coverage(champ, half_b, spec["conformal_alphas"], n, seed),
                "strata_league": strata(champ, "competition_id"),
                "strata_position": strata(champ, "position_group"),
            }
            log.info("%s %s: PR-AUC %.4f", label, pop, result["labels"][label][pop]["champion"]["pr_auc"]["value"])

    result["survival"] = {}
    for pop, pspec in spec["populations"].items():
        mask = surv_rows["t"].dt.month.isin(pspec["months"]).to_numpy()
        result["survival"][pop] = survival_metrics(surv_rows[mask].reset_index(drop=True), S[mask], G, horizon, n, seed)

    if rehearsal:
        result["rehearsal_checks"] = _rehearsal_checks(spec, out_dir, surv_rows, S)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, default=lambda v: v.item()), encoding="utf-8")
    if not rehearsal:
        marker = json.loads(MARKER.read_text(encoding="utf-8"))
        MARKER.write_text(json.dumps({**marker, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2),
                          encoding="utf-8")
    log.info("wrote %s", out_dir / "results.json")
    return result


def _rehearsal_checks(spec: dict, out_dir: Path, surv_rows: pd.DataFrame, S: np.ndarray) -> dict:
    """On the main validation origin every prediction must equal the stored experiment predictions."""
    main = load_yaml(CONFIGS_DIR / "evaluation.yaml")["main_origin"]
    checks = {}
    for m in [spec["champion"], *spec["comparators"]]:
        stored = load_predictions(m)
        for label in spec["labels"]:
            mine = pd.read_parquet(out_dir / f"predictions_{m}_{label}.parquet")
            ref = stored[(stored["label"] == label) & (stored["origin"] == main)]
            both = mine.merge(ref[KEYS + ["p"]], on=KEYS, suffixes=("", "_stored"))
            checks[f"{m}-{label}"] = bool(len(both) == len(mine) == len(ref) and np.array_equal(both["p"], both["p_stored"]))
    ref = load_predictions(spec["survival_champion"])
    ref = ref[ref["origin"] == main].reset_index(drop=True)
    checks[spec["survival_champion"]] = bool(np.array_equal(S[:, 27], ref["S28"].to_numpy())
                                             and surv_rows["player_id"].equals(ref["player_id"]))
    if not all(checks.values()):
        raise SystemExit(f"rehearsal does not reproduce the stored predictions: {checks}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.final_run")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rehearsal", action="store_true")
    mode.add_argument("--expected-commit")
    args = parser.parse_args()
    run(args.rehearsal, args.expected_commit)


if __name__ == "__main__":
    main()
