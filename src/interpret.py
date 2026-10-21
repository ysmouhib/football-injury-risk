"""Stage 5 interpretability (PLAN amendment A11): SHAP, dose-response, positions, factor classes.

Run as: python -m src.interpret <champion> <tree challenger> <survival champion>
Every explained model is refitted at the main origin through the training code
path and must reproduce its stored validation predictions exactly.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from . import datasets, labels, metrics, splits, survival
from .config import load_yaml
from .constants import CONFIGS_DIR, REPORTS_DIR
from .experiment import EVALUATION_CONFIG, load_experiment, load_predictions, panel_path, run_dir
from .models import build_model, fit_categories, predict_risk, prepare_X
from .train import _calibrate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PDP_FEATURES = ["prev1_minutes_est", "prev2_minutes_est", "prev1_apps", "age", "days_since_return"]
GRID_POINTS, PDP_ROWS, ICE_ROWS = 25, 1000, 200
SHAPE_MARGIN = 0.05  # log-odds
FACTOR_CLASSES = {
    "modifiable (exposure / load)": [
        "prev1_apps", "prev1_starts", "prev1_sub_apps", "prev1_squad", "prev1_minutes_est", "prev1_n_comps",
        "prev1_no_record", "prev2_minutes_est", "prev2_apps", "prev2_no_record", "prev1_euro_minutes"],
    "partially modifiable (return to play)": ["days_since_return", "returned_within_28d"],
    "non-modifiable": [
        "age", "height_cm", "position_group", "foot", "no_prior_injury", "career_injuries", "career_days_lost",
        "injuries_365d", "injuries_730d", "days_lost_365d", "days_lost_730d", "hamstring_injuries_730d",
        "groin_hip_injuries_730d", "thigh_injuries_730d", "knee_injuries_730d", "ankle_injuries_730d",
        "lower_leg_injuries_730d", "muscle_injuries_730d", "league", "season_month", "weeks_into_season"],
}


def _logit(p: np.ndarray) -> np.ndarray:
    return metrics._logit(p)


# --- refits ---------------------------------------------------------------------

class Refit:
    """A model refitted at one origin, with a log-odds risk function over raw panel rows."""

    def __init__(self, exp_id: str, label: str, origin: int, panel: pd.DataFrame):
        self.exp = load_experiment(CONFIGS_DIR / f"{exp_id}.yaml")
        self.id, self.label, self.origin = exp_id, label, origin
        cfg, numeric, categorical = self.exp["data"], self.exp["numeric"], self.exp["categorical"]
        train_first = self.exp.get("train_first_season", cfg["seasons"]["first"])
        self.survival = self.exp.get("arm") == "survival"
        target = f"{self.exp['outcome']}_event" if self.survival else label
        split = splits.origin_split(panel, cfg, origin, target)
        self.train = panel.iloc[split.train_idx]
        self.train = self.train[self.train["season"] >= train_first]
        self.val = panel.iloc[split.eval_idx]
        self.categories = fit_categories(self.train, categorical)
        X_train = prepare_X(self.train, numeric, categorical, self.categories)
        self.calibrator = None
        if self.survival:
            name = self.exp["outcome"]
            self.model = survival.build_survival_model(self.exp["model"], numeric, categorical, self.exp["seed"])
            self.model.fit(X_train, self.train[f"{name}_time"].to_numpy(float), self.train[f"{name}_event"].to_numpy(float))
            stored = load_predictions(exp_id)
            stored = stored[stored["origin"] == origin]
            check = 1 - self.model.survival(self.X(self.val), np.array([28]))[:, 0].astype(np.float32)
            reference = 1 - stored["S28"].to_numpy()
        else:
            self.model = build_model(self.exp["model"], numeric, categorical, self.exp["seed"])
            self.model.fit(X_train, self.train[label].astype(int))
            p_raw = predict_risk(self.model, self.X(self.val))
            check = p_raw
            if self.exp.get("calibration"):
                check, _, _ = _calibrate(self.exp, panel, label, origin, p_raw, train_first, (self.model, self.categories))
                self.calibrator = self._calibrator(panel, train_first)
            stored = load_predictions(exp_id)
            stored = stored[(stored["label"] == label) & (stored["origin"] == origin)]
            reference = stored["p"].to_numpy()
        if not np.array_equal(np.asarray(check), np.asarray(reference)):
            raise SystemExit(f"{exp_id} {label} {origin}: refit does not reproduce the stored predictions")
        log.info("%s %s %s: refit reproduces %s stored predictions", exp_id, label, origin, f"{len(reference):,}")

    def _calibrator(self, panel: pd.DataFrame, train_first: int):
        """The calibrator object of the deployed pipeline (refitted exactly as in training)."""
        from .calibration import build_calibrator
        from .train import _fit_predict

        cs = splits.calibration_split(panel, self.exp["data"], self.origin, self.label)
        inner = panel.iloc[cs.inner_idx]
        inner = inner[inner["season"] >= train_first]
        half_a = panel.iloc[cs.calib_a_idx]
        if self.exp["calibration"].get("fit_on", "inner") == "deployed":
            (score_a,) = [predict_risk(self.model, self.X(half_a))]
        else:
            (score_a,) = _fit_predict(self.exp, inner, self.label, half_a)
        return build_calibrator(self.exp["calibration"]["method"]).fit(score_a, half_a[self.label].to_numpy())

    def X(self, rows: pd.DataFrame) -> pd.DataFrame:
        return prepare_X(rows, self.exp["numeric"], self.exp["categorical"], self.categories)

    def log_odds(self, rows: pd.DataFrame) -> np.ndarray:
        if self.survival:
            return _logit(1 - self.model.survival(self.X(rows), np.array([28]))[:, 0])
        p = predict_risk(self.model, self.X(rows))
        if self.calibrator is not None:
            p = self.calibrator.transform(p)
        return _logit(p)


# --- SHAP -------------------------------------------------------------------------

def _original_feature(encoded: str, numeric: list[str], categorical: list[str]) -> str:
    name = encoded.split("__", 1)[1]
    if name.startswith("missingindicator_"):
        return name.removeprefix("missingindicator_")
    if name in numeric:
        return name
    return next(c for c in sorted(categorical, key=len, reverse=True) if name.startswith(c + "_"))


def linear_shap(refit: Refit, rows: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Exact interventional SHAP on the calibrated logit, grouped to original features."""
    pre, clf = refit.model.named_steps["pre"], refit.model.named_steps["clf"]
    dense = lambda Z: Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)  # noqa: E731
    Z_bg, Z = dense(pre.transform(refit.X(refit.train))), dense(pre.transform(refit.X(rows)))
    beta, mean_bg = clf.coef_[0], Z_bg.mean(axis=0)
    scale = getattr(refit.calibrator, "b", 1.0)
    shift = getattr(refit.calibrator, "a", 0.0)
    phi = scale * (Z - mean_bg) * beta
    base = shift + scale * (clf.intercept_[0] + mean_bg @ beta)
    names = [_original_feature(n, refit.exp["numeric"], refit.exp["categorical"]) for n in pre.get_feature_names_out()]
    grouped = pd.DataFrame(phi, columns=names, index=rows.index).T.groupby(level=0).sum().T
    return grouped, float(base)


def shap_check(refit: Refit, rows: pd.DataFrame, n: int = 500) -> float:
    """Max abs difference between the manual SHAP values and shap.LinearExplainer (raw logit)."""
    import shap

    pre, clf = refit.model.named_steps["pre"], refit.model.named_steps["clf"]
    dense = lambda Z: Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)  # noqa: E731
    background = refit.train.sample(20_000, random_state=0)  # verifies the formula, not the full background
    Z_bg = dense(pre.transform(refit.X(background)))
    Z = dense(pre.transform(refit.X(rows.iloc[:n])))
    explainer = shap.LinearExplainer((clf.coef_[0], clf.intercept_[0]), shap.maskers.Independent(Z_bg, max_samples=len(Z_bg)))
    reference = explainer.shap_values(Z)
    manual = (Z - Z_bg.mean(axis=0)) * clf.coef_[0]
    return float(np.abs(reference - manual).max())


# --- dose-response -------------------------------------------------------------------

def pdp_ice(refit: Refit, rows: pd.DataFrame, feature: str, grid: np.ndarray, seed: int, n_resamples: int) -> dict:
    sample = rows.sample(min(PDP_ROWS, len(rows)), random_state=seed)
    curves = np.empty((len(sample), len(grid)))
    for k, g in enumerate(grid):
        curves[:, k] = refit.log_odds(sample.assign(**{feature: g}))
    pdp = curves.mean(axis=0)
    players, index = np.unique(sample["player_id"].to_numpy(), return_inverse=True)
    sums = np.stack([np.bincount(index, weights=curves[:, k], minlength=len(players)) for k in range(len(grid))], 1)
    rows_per = np.bincount(index, minlength=len(players)).astype(float)
    draws = np.vstack([(c @ sums) / (c @ rows_per)[:, None] for _, c in metrics.resample_counts(len(players), n_resamples, seed)])
    band = np.percentile(draws - draws[:, :1], [2.5, 97.5], axis=0)
    ice = curves[:ICE_ROWS] - curves[:ICE_ROWS, :1]
    return {"grid": grid.tolist(), "pdp": (pdp - pdp[0]).tolist(), "band": band.T.tolist(), "ice": ice.tolist()}


def shape_verdict(rows: pd.DataFrame, feature: str, grid: np.ndarray, pdp: np.ndarray) -> dict:
    q05, q10, q90, q95 = np.nanpercentile(rows[feature].astype(float), [5, 10, 90, 95])
    i_min = int(np.argmin(pdp))
    at = lambda q: float(np.interp(q, grid, pdp))  # noqa: E731
    interior = q10 < grid[i_min] < q90
    u_shaped = interior and at(q05) - pdp[i_min] >= SHAPE_MARGIN and at(q95) - pdp[i_min] >= SHAPE_MARGIN
    return {"minimum_at": float(grid[i_min]), "p10": float(q10), "p90": float(q90),
            "rise_to_p05": at(q05) - float(pdp[i_min]), "rise_to_p95": at(q95) - float(pdp[i_min]),
            "shape": "U/J-shaped" if u_shaped else "monotone/flat",
            "end_to_end": float(pdp[-1] - pdp[0])}


# --- positions and factor classes ------------------------------------------------------

def position_breakdown(champion: str) -> list[dict]:
    preds = load_predictions(champion)
    out = []
    for (label, origin, position), g in preds.groupby(["label", "origin", "position_group"]):
        slope, intercept = metrics.calibration(g["y"], g["p"])
        out.append({"label": label, "origin": int(origin), "position": position, "n": len(g),
                    "prevalence": float(g["y"].mean()), "mean_predicted": float(g["p"].mean()),
                    "pr_auc": metrics.pr_auc(g["y"], g["p"]), "calibration_slope": slope,
                    "calibration_intercept": intercept})
    return out


def body_parts_by_position(rows: pd.DataFrame) -> pd.DataFrame:
    """Body part of the first qualifying injury behind each positive y28 row."""
    events = labels.prepare_events(datasets.load_injuries(), labels.load_label_mapping())
    events = events[events["category"].eq("injury") & (events["days_missed"] >= 7)]
    pos = rows[rows["y28"].eq(1)][["player_id", "position_group", "onset_y28"]]
    first = events.sort_values("from_date").drop_duplicates(["player_id", "from_date"])
    merged = pos.merge(first, left_on=["player_id", "onset_y28"], right_on=["player_id", "from_date"], how="left")
    table = pd.crosstab(merged["position_group"], merged["body_part"], normalize="index")
    return table


def factor_report(shap_values: pd.DataFrame, refit: Refit) -> dict:
    importance = shap_values.abs().mean()
    total = importance.sum()
    pre, clf = refit.model.named_steps["pre"], refit.model.named_steps["clf"]
    coefs = dict(zip(pre.get_feature_names_out(), clf.coef_[0]))
    out = {"classes": {}, "features": {}}
    for cls, feats in FACTOR_CLASSES.items():
        present = [f for f in feats if f in importance.index]
        out["classes"][cls] = float(importance[present].sum() / total)
        for f in present:
            out["features"][f] = {"class": cls, "mean_abs_shap": float(importance[f]),
                                  "share": float(importance[f] / total),
                                  "coefficient_scaled": float(coefs.get(f"num__{f}", np.nan))}
    unclassified = sorted(set(importance.index) - {f for v in FACTOR_CLASSES.values() for f in v})
    if unclassified:
        raise SystemExit(f"features without a factor class: {unclassified}")
    return out


# --- main ------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.interpret")
    parser.add_argument("champion")
    parser.add_argument("tree_challenger")
    parser.add_argument("survival_champion")
    args = parser.parse_args()
    ev_cfg = load_yaml(EVALUATION_CONFIG)
    main_origin, n_res = ev_cfg["main_origin"], ev_cfg["n_resamples"]
    exp = load_experiment(CONFIGS_DIR / f"{args.champion}.yaml")
    panel = pd.read_parquet(panel_path(exp["data"]))
    seed = exp["seed"]
    out_dir = run_dir(args.champion) / "interpret"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"champion": args.champion, "tree_challenger": args.tree_challenger,
              "survival_champion": args.survival_champion, "origin": main_origin, "labels": {}}

    for label in ev_cfg["decision_labels"]:
        champ = Refit(args.champion, label, main_origin, panel)
        val = champ.val
        phi, base = linear_shap(champ, val)
        phi.assign(player_id=val["player_id"].to_numpy(), t=val["t"].to_numpy(),
                   position_group=val["position_group"].to_numpy(), y=val[label].to_numpy()).to_parquet(
            out_dir / f"shap_{label}.parquet")
        logit_p = champ.log_odds(val)
        risk = 1 / (1 + np.exp(-logit_p))
        additivity = float(np.abs(phi.sum(axis=1).to_numpy() + base - logit_p).max())
        order = np.argsort(-risk)
        injured = np.flatnonzero(val[label].to_numpy() == 1)
        picks = {"highest predicted risk": int(order[0]),
                 "median predicted risk": int(order[len(order) // 2]),
                 "highest-risk week followed by an injury": int(injured[np.argmax(risk[injured])])}
        local = {}
        for i, (name, row) in enumerate(picks.items()):
            r = val.iloc[row]
            contrib = phi.iloc[row].sort_values(key=np.abs, ascending=False)
            local[name] = {"case": f"Player {'ABC'[i]}", "position": r["position_group"],
                           "age": round(float(r["age"]), 1), "risk": float(risk[row]), "base_logit": base,
                           "injured_within_window": int(r[label]),
                           "top_contributions": {k: float(v) for k, v in contrib.head(8).items()},
                           "other_contributions": float(contrib.iloc[8:].sum()),
                           "feature_values": {k: (None if pd.isna(r[k]) else (float(r[k]) if np.isscalar(r[k]) and not isinstance(r[k], str) else str(r[k])))
                                              for k in contrib.head(8).index}}
        importance = phi.abs().mean().sort_values(ascending=False)
        by_position = phi.abs().groupby(val["position_group"].to_numpy()).mean()
        result["labels"][label] = {
            "base_logit": base, "additivity_max_error": additivity,
            "shap_vs_linear_explainer_max_diff": shap_check(champ, val),
            "global_importance": importance.to_dict(),
            "importance_by_position": {pos: row.sort_values(ascending=False).head(10).to_dict()
                                       for pos, row in by_position.iterrows()},
            "local": local, "factors": factor_report(phi, champ),
        }

        dose = {}
        tree = Refit(args.tree_challenger, label, main_origin, panel)
        surv = Refit(args.survival_champion, label, main_origin, panel) if label == "y28" else None
        for feature in PDP_FEATURES:
            if feature not in champ.exp["numeric"] or feature not in tree.exp["numeric"]:
                continue
            values = val[feature].astype(float).dropna()
            grid = np.unique(np.nanpercentile(values, np.linspace(1, 99, GRID_POINTS)))
            curves = {args.champion: pdp_ice(champ, val, feature, grid, seed, n_res),
                      args.tree_challenger: pdp_ice(tree, val, feature, grid, seed, n_res)}
            if surv is not None:
                curves[args.survival_champion] = pdp_ice(surv, val, feature, grid, seed, n_res)
            dose[feature] = {m: {**c, "shape": shape_verdict(val, feature, grid, np.asarray(c["pdp"]))}
                             for m, c in curves.items()}
            log.info("%s %s: %s", label, feature, {m: c["shape"]["shape"] for m, c in dose[feature].items()})
        result["labels"][label]["dose_response"] = dose

    result["positions"] = position_breakdown(args.champion)
    main_rows = Refit(args.champion, "y28", main_origin, panel).val
    result["body_parts_y28"] = body_parts_by_position(main_rows).round(4).to_dict(orient="index")
    (out_dir / "interpret.json").write_text(json.dumps(result, indent=2, default=lambda v: v.item()), encoding="utf-8")
    log.info("wrote %s", out_dir / "interpret.json")


if __name__ == "__main__":
    main()
