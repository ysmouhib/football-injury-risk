"""Stage 7 practitioner evidence pack: descriptive analyses of the frozen champion.

Run as: python -m src.cli analyse configs/exp-00NN.yaml
Each config names one analysis: operational (alarm burden at fixed operating
thresholds), subgroups (calibration and discrimination by player subgroup or
league), seasons (per-season stability over the development period) or acwr
(workload-ratio baseline). Holdout analyses read only the final run's frozen
predictions; the per-season analysis rebuilds the champion pipeline on
development seasons and never reads the test player group. No result here
feeds a model-selection decision. Definitions: PLAN open decisions 19-23.
"""

from __future__ import annotations

import json
import logging
import subprocess

import numpy as np
import pandas as pd

from . import champion, conformal, datasets, metrics, splits
from .config import load_yaml
from .constants import CONFIGS_DIR, PROCESSED_DIR, REPO_ROOT, REPORTS_DIR, week_end
from .decision_curves import net_benefit_parts
from .experiment import (EXPERIMENTS_DATA_DIR, EXPERIMENTS_REPORTS_DIR, git_state, load_experiment,
                         load_predictions, panel_meta, panel_path)
from .final_run import FINAL_CONFIG, FINAL_DIR, KEYS
from .labels import LABEL_SPECS, label_horizon, load_label_mapping, prepare_events
from .models import build_model, fit_categories, predict_risk, prepare_X
from .panel import content_hash
from .train import _calibrate, leakage_tests_pass

log = logging.getLogger(__name__)

POSITIONS = {"Goalkeeper": "GK", "Defender": "DEF", "Midfield": "MID", "Attack": "FWD"}
AGE_BANDS = ["≤23", "24–28", "≥29"]


# --- shared ---------------------------------------------------------------------------

def _spec() -> dict:
    return load_yaml(FINAL_CONFIG)


def _months(population: str) -> list[int]:
    return _spec()["populations"][population]["months"]


def holdout_rows(label: str, columns: list[str] | tuple[str, ...] = ()) -> pd.DataFrame:
    """The final run's frozen champion predictions on the holdout, with panel columns attached."""
    spec = _spec()
    rows = pd.read_parquet(FINAL_DIR / "holdout" / f"predictions_{spec['champion']}_{label}.parquet")
    if columns:
        panel = pd.read_parquet(PROCESSED_DIR / f"{spec['data_version']}.parquet", columns=KEYS + list(columns))
        rows = rows.merge(panel, on=KEYS, how="left", validate="one_to_one")
    return rows


def _clusters(player_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.unique(player_id, return_inverse=True)


def pr_auc_ci(y: np.ndarray, p: np.ndarray, player_id: np.ndarray, n: int, seed: int) -> tuple[float, tuple]:
    players, index = _clusters(player_id)
    ap = metrics._SortedAP(np.asarray(y, dtype=float), np.asarray(p, dtype=float))
    draws = np.concatenate([ap(c[:, index]) for _, c in metrics.resample_counts(len(players), n, seed)])
    return metrics.pr_auc(y, p), metrics.percentile_ci(draws)


def slope_ci(y: np.ndarray, p: np.ndarray, player_id: np.ndarray, n: int, seed: int) -> tuple[float, float, tuple]:
    players, index = _clusters(player_id)
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    draws = []
    for _, counts in metrics.resample_counts(len(players), n, seed):
        for w in counts[:, index]:
            keep = w > 0
            draws.append(metrics.calibration(y[keep], p[keep], w[keep])[0])
    slope, intercept = metrics.calibration(y, p)
    return slope, intercept, metrics.percentile_ci(np.asarray(draws))


def _write(cfg: dict, result: dict, report: str) -> None:
    out = EXPERIMENTS_DATA_DIR / cfg["id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(result, indent=2, default=lambda v: v.item()), encoding="utf-8")
    EXPERIMENTS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENTS_REPORTS_DIR / f"{cfg['id']}.md").write_text(report, encoding="utf-8")
    print(report)


def _provenance(cfg: dict, path) -> dict:
    return {"id": cfg["id"], "config_path": path.as_posix(), "git": git_state(),
            "panel_sha256": panel_meta(load_yaml(CONFIGS_DIR / "data.yaml"))["panel_sha256"],
            "final_run_commit": json.loads((FINAL_DIR / "holdout_evaluated.json").read_text())["git"]["commit"],
            "seed": cfg["seed"], "n_resamples": cfg["n_resamples"]}


def _ci(v: tuple, fmt: str = ".4f") -> str:
    return f"[{v[0]:{fmt}}, {v[1]:{fmt}}]"


# --- exp-0024: operational characteristics --------------------------------------------

def select_thresholds(cfg: dict) -> dict:
    """Operating thresholds from calibration half B only (decision 21): for each target precision
    k x prevalence, the lowest cut-off whose precision reaches the target."""
    months = _months(cfg["population"])
    out = {}
    for label in cfg["labels"]:
        cal = champion.calibration_scores(label)
        cal = cal[cal["half"].eq("B") & cal["t"].dt.month.isin(months)]
        y, p = cal["y"].to_numpy(float), cal["p"].to_numpy(float)
        order = np.argsort(-p, kind="mergesort")
        ys, ps = y[order], p[order]
        ends = np.flatnonzero(np.r_[ps[1:] != ps[:-1], True])
        tp, flagged = np.cumsum(ys)[ends], ends + 1
        precision = tp / flagged
        res = {"rows": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()), "by_multiple": []}
        for k in cfg["precision_multiples"]:
            target = k * y.mean()
            ok = np.flatnonzero(precision >= target)
            if len(ok):
                i, achieved = ok[-1], True
            else:
                big = np.flatnonzero(flagged >= 0.01 * len(y))
                i, achieved = big[np.argmax(precision[big])], False
            res["by_multiple"].append({
                "multiple": k, "target_precision": float(target), "threshold": float(ps[ends[i]]),
                "precision": float(precision[i]), "recall": float(tp[i] / y.sum()),
                "flagged_share": float(flagged[i] / len(y)), "achieved": achieved})
        out[label] = res
    return out


def _committed(path) -> bool:
    rel = path.resolve().relative_to(REPO_ROOT).as_posix()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO_ROOT, capture_output=True)
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=REPO_ROOT)
    return tracked.returncode == 0 and clean.returncode == 0


def qualifying_onsets(min_days: int) -> pd.DataFrame:
    """Onset dates of known injuries with >= min_days missed: the positive events of the labels."""
    ev = prepare_events(datasets.load_injuries(), load_label_mapping())
    ev = ev[ev["category"].eq("injury") & (ev["days_missed"] >= min_days)]
    return ev[["player_id", "from_date"]].drop_duplicates().rename(columns={"from_date": "onset"})


def event_rows(onsets: pd.DataFrame, rows: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """(event, evaluated row) pairs: rows whose outcome window (t, t + H] contains the onset."""
    onsets = onsets[onsets["player_id"].isin(rows["player_id"].unique())]
    closing = week_end(onsets["onset"])
    pairs = pd.concat([onsets.assign(t=closing - pd.Timedelta(days=7 * k)) for k in range(1, horizon // 7 + 1)])
    return pairs.merge(rows[KEYS].assign(row=np.arange(len(rows))), on=KEYS, how="inner")


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    cum = np.cumsum(weights[order])
    if cum[-1] <= 0:
        return np.nan
    return float(values[order][np.searchsorted(cum, cum[-1] / 2)])


def operational(cfg: dict, path) -> dict:
    selected = select_thresholds(cfg)
    chosen = {label: [m["threshold"] for m in selected[label]["by_multiple"]] for label in cfg["labels"]}
    if any((cfg.get("thresholds") or {}).get(label) != chosen[label] for label in cfg["labels"]):
        out = EXPERIMENTS_DATA_DIR / cfg["id"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "thresholds.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
        print(json.dumps(selected, indent=2))
        raise SystemExit(f"record the selected thresholds {chosen} in {path} and commit it before the holdout is read")
    if not _committed(path):
        raise SystemExit(f"{path} must be committed unchanged before the holdout rows are read")
    n, seed = cfg["n_resamples"], cfg["seed"]
    months = _months(cfg["population"])
    result = {**_provenance(cfg, path), "selection": selected, "labels": {}}
    for label in cfg["labels"]:
        horizon = label_horizon(label)
        rows = holdout_rows(label)
        rows = rows[rows["t"].dt.month.isin(months)].reset_index(drop=True)
        y, p = rows["y"].to_numpy(), rows["p"].to_numpy()
        pairs = event_rows(qualifying_onsets(LABEL_SPECS[label][1]), rows, horizon)
        hit_rows = np.zeros(len(rows), bool)
        hit_rows[pairs["row"].to_numpy()] = True
        if not np.array_equal(hit_rows, y == 1):
            raise SystemExit(f"{label}: event matching disagrees with the labels")
        events = pairs[["player_id", "onset"]].drop_duplicates().reset_index(drop=True)
        pairs = pairs.merge(events.assign(event=np.arange(len(events))), on=["player_id", "onset"])
        players, index = _clusters(rows["player_id"].to_numpy())
        event_index = np.searchsorted(players, events["player_id"].to_numpy())
        n_sundays = rows["t"].nunique()
        per = lambda v, idx=index: np.bincount(idx, weights=np.asarray(v, float), minlength=len(players))  # noqa: E731
        rows_pp, pos_pp, events_pp = per(np.ones(len(rows))), per(y), per(np.ones(len(events)), event_index)
        squad_periods = 7 / (25 * 90)
        out = {"rows": len(rows), "players": len(players), "positives": int(y.sum()), "prevalence": float(y.mean()),
               "sundays": int(n_sundays), "rows_per_sunday": len(rows) / n_sundays, "events": len(events),
               "by_multiple": []}
        for spec in selected[label]["by_multiple"]:
            flag = p >= spec["threshold"]
            pairs["flag"] = flag[pairs["row"].to_numpy()]
            fl = pairs[pairs["flag"]]
            first = fl.groupby("event")["t"].min()
            caught = np.zeros(len(events), bool)
            caught[first.index.to_numpy()] = True
            lead = np.full(len(events), np.nan)
            lead[first.index.to_numpy()] = (events.loc[first.index, "onset"] - first).dt.days.to_numpy()
            flags_pp, tp_pp = per(flag), per(flag & (y == 1))
            caught_pp = per(caught, event_index)
            point = {
                "flags_per_sunday": flag.sum() / n_sundays,
                "flags_per_week_per_25": 25 * flag.mean(),
                "precision": (flag & (y == 1)).sum() / flag.sum(),
                "recall": (flag & (y == 1)).sum() / (y == 1).sum(),
                "events_caught": int(caught.sum()), "events_missed": int((~caught).sum()),
                "event_recall": caught.mean(),
                "caught_per_squad_90d": caught.sum() / (len(rows) * squad_periods),
                "missed_per_squad_90d": (~caught).sum() / (len(rows) * squad_periods),
                "median_lead_days": _weighted_median(lead[caught], np.ones(caught.sum())),
            }
            draws = {k: [] for k in point if k not in ("events_caught", "events_missed")}
            for _, c in metrics.resample_counts(len(players), n, seed):
                R, F, T, P = c @ rows_pp, c @ flags_pp, c @ tp_pp, c @ pos_pp
                E, C = c @ events_pp, c @ caught_pp
                draws["flags_per_sunday"].append(F / n_sundays)
                draws["flags_per_week_per_25"].append(25 * F / R)
                draws["precision"].append(T / F)
                draws["recall"].append(T / P)
                draws["event_recall"].append(C / E)
                draws["caught_per_squad_90d"].append(C / (R * squad_periods))
                draws["missed_per_squad_90d"].append((E - C) / (R * squad_periods))
                we = c[:, event_index][:, caught]
                draws["median_lead_days"].append([_weighted_median(lead[caught], w) for w in we])
            out["by_multiple"].append({
                **{k: spec[k] for k in ("multiple", "threshold", "target_precision")},
                "calibration_precision": spec["precision"],
                **{k: float(v) if not isinstance(v, int) else v for k, v in point.items()},
                **{f"{k}_ci": metrics.percentile_ci(np.concatenate([np.atleast_1d(d) for d in v]))
                   for k, v in draws.items()},
            })
        result["labels"][label] = out
    return result


def _operational_report(r: dict) -> str:
    lines = [f"# {r['id']} — operational characteristics of the frozen champion (holdout 2024/25, in-season)", "",
             "Thresholds chosen on calibration half B (2023/24, in-season) before the holdout was read "
             "(PLAN decision 21); 95% player-cluster bootstrap CIs.", ""]
    for label, o in r["labels"].items():
        lines += [f"## {label[1:]}-day horizon — {o['rows']:,} player-weeks, {o['players']} players, "
                  f"{o['sundays']} Sundays ({o['rows_per_sunday']:.0f} at-risk players per Sunday), "
                  f"prevalence {o['prevalence']:.4f}, {o['events']} qualifying injuries", "",
                  "| target precision | threshold | flags / Sunday | flags / week per 25 players | precision | recall "
                  "| injuries caught / missed | share of injuries caught | caught per 25-player squad per 90 days "
                  "| missed per squad per 90 days | median lead time (days) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for m in o["by_multiple"]:
            lines.append(
                f"| {m['multiple']}× prevalence ({m['target_precision']:.4f}; calibration {m['calibration_precision']:.4f}) "
                f"| {m['threshold']:.4f} | {m['flags_per_sunday']:.1f} {_ci(m['flags_per_sunday_ci'], '.1f')} "
                f"| {m['flags_per_week_per_25']:.2f} {_ci(m['flags_per_week_per_25_ci'], '.2f')} "
                f"| {m['precision']:.4f} {_ci(m['precision_ci'])} | {m['recall']:.3f} {_ci(m['recall_ci'], '.3f')} "
                f"| {m['events_caught']} / {m['events_missed']} | {m['event_recall']:.3f} {_ci(m['event_recall_ci'], '.3f')} "
                f"| {m['caught_per_squad_90d']:.2f} {_ci(m['caught_per_squad_90d_ci'], '.2f')} "
                f"| {m['missed_per_squad_90d']:.2f} {_ci(m['missed_per_squad_90d_ci'], '.2f')} "
                f"| {m['median_lead_days']:.0f} {_ci(m['median_lead_days_ci'], '.0f')} |")
        lines.append("")
    return "\n".join(lines)


# --- exp-0025 / exp-0027: subgroups -------------------------------------------------------

SUBGROUP_COLUMNS = ["season", "competition_id", "position_group", "age", "prev1_minutes_est"]


def _subgroups(rows: pd.DataFrame, usage_cut: float) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index)
    out["position"] = rows["position_group"].map(POSITIONS)
    start = pd.to_datetime(rows["season"].astype(str) + "-07-01")
    age0 = np.floor(rows["age"] - (rows["t"] - start).dt.days / 365.25)
    out["age_band"] = pd.Series(np.select([age0 <= 23, age0 <= 28, age0 >= 29], AGE_BANDS, ""),
                                index=rows.index).where(age0.notna()).replace("", np.nan)
    out["usage_band"] = np.where(rows["prev1_minutes_est"] > usage_cut, "top half", "bottom half")
    out["league"] = rows["competition_id"]
    return out


def _cell(g: pd.DataFrame, n: int, seed: int, full: bool, q: tuple | None, min_cell: int) -> dict:
    y, p, pid = g["y"].to_numpy(), g["p"].to_numpy(), g["player_id"].to_numpy()
    cell = {"n": len(g), "players": int(g["player_id"].nunique()), "positives": int(y.sum()),
            "prevalence": float(y.mean()) if len(g) else None}
    if len(g) < min_cell or y.sum() == 0 or y.sum() == len(y):
        return {**cell, "status": "insufficient"}
    pr, pr_ci = pr_auc_ci(y, p, pid, n, seed)
    cell.update(status="ok", pr_auc=pr, pr_auc_ci=pr_ci)
    if full:
        slope, intercept, s_ci = slope_ci(y, p, pid, n, seed)
        cell.update(calibration_slope=slope, calibration_slope_ci=s_ci, calibration_intercept=intercept,
                    mean_predicted=float(p.mean()))
        if q is not None:
            has0, has1 = conformal.prediction_sets(p, *q)
            covered = np.where(y == 1, has1, has0)
            players, index = _clusters(pid)
            draws = conformal._coverage_draws(covered, index, len(players), n, seed)
            cell.update(coverage=float(covered.mean()), coverage_ci=metrics.percentile_ci(draws))
    return cell


def subgroups(cfg: dict, path) -> dict:
    n, seed = cfg["n_resamples"], cfg["seed"]
    spec = _spec()
    final = json.loads((FINAL_DIR / "holdout" / "results.json").read_text(encoding="utf-8"))
    half_b = champion.calibration_scores(cfg["coverage"]["label"])
    half_b = half_b[half_b["half"].eq("B")]
    alpha = cfg["coverage"]["alpha"]
    q = conformal.thresholds(half_b["y"].to_numpy(), half_b["p"].to_numpy(), alpha, mondrian=True)
    months = {pop: _months(pop) for pop in cfg["populations"]}
    data = {label: holdout_rows(label, SUBGROUP_COLUMNS) for label in cfg["labels"]}
    keys = pd.concat([d[KEYS + ["prev1_minutes_est"]] for d in data.values()]).drop_duplicates(KEYS)
    primary = cfg["populations"][0]
    usage_cut = float(keys.loc[keys["t"].dt.month.isin(months[primary]), "prev1_minutes_est"].median())
    result = {**_provenance(cfg, path), "min_cell": cfg["min_cell_player_weeks"], "usage_cut_minutes": usage_cut, "conformal": {"alpha": alpha, "q0": q[0],
              "q1": q[1], "label": cfg["coverage"]["label"]}, "unknown": {}, "cells": []}
    for label, rows in data.items():
        groups = _subgroups(rows, usage_cut)
        for pop in cfg["populations"]:
            in_pop = rows["t"].dt.month.isin(months[pop])
            if pop == primary and label == cfg["coverage"]["label"]:
                # the quantiles reproduce the final run's overall coverage on this population
                allq = _cell(rows[in_pop], 1, seed, True, q, 0)["coverage"]
                ref = next(c for c in final["labels"][label][pop]["conformal"] if c["alpha"] == alpha)["coverage"]
                if allq != ref:
                    raise SystemExit(f"conformal quantiles do not reproduce the final run ({allq} vs {ref})")
            for dim in cfg["dimensions"]:
                known = groups[dim].notna()
                result["unknown"][f"{label}/{pop}/{dim}"] = int((in_pop & ~known).sum())
                order = cfg["order"].get(dim) or sorted(groups.loc[known, dim].unique())
                for value in order:
                    g = rows[in_pop & groups[dim].eq(value)]
                    full = pop == primary
                    cq = q if full and label == cfg["coverage"]["label"] else None
                    result["cells"].append({"label": label, "population": pop, "dimension": dim, "group": value,
                                            **_cell(g, n, seed, full, cq, cfg["min_cell_player_weeks"])})
    result["final_run_strata_check"] = {
        "champion": spec["champion"],
        "note": "point estimates of position/league strata also appear in the final-run entry"}
    return result


def _subgroup_report(r: dict, title: str) -> str:
    lines = [f"# {r['id']} — {title}", "",
             f"Holdout 2024/25, frozen champion; 95% player-cluster bootstrap CIs within each subgroup; cells < "
             f"{r['min_cell']} player-weeks or without injuries: insufficient. Usage band cut (decision 20a): "
             f"{r['usage_cut_minutes']:.0f} estimated minutes last season. Mondrian conformal α = "
             f"{r['conformal']['alpha']} ({r['conformal']['label']}), final-run quantiles.", "",
             "Unknown (dropped) player-weeks: " + ", ".join(f"{k} {v}" for k, v in r["unknown"].items()), ""]
    pops = list(dict.fromkeys(c["population"] for c in r["cells"]))
    primary = pops[0]
    lines += [f"## Primary population: {primary}", "",
              "| dimension | group | horizon | player-weeks | players | injuries | prevalence | PR-AUC [95% CI] "
              "| calibration slope [95% CI] | intercept | mean predicted | coverage 28-day α=0.10 [95% CI] |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in r["cells"]:
        if c["population"] != primary:
            continue
        head = (f"| {c['dimension']} | {c['group']} | {c['label']} | {c['n']:,} | {c['players']} | {c['positives']} "
                f"| {c['prevalence']:.4f} |" if c["n"] else f"| {c['dimension']} | {c['group']} | {c['label']} | 0 | 0 | 0 | — |")
        if c["status"] != "ok":
            lines.append(head + " insufficient | — | — | — | — |")
            continue
        cov = f"{c['coverage']:.3f} {_ci(c['coverage_ci'], '.3f')}" if "coverage" in c else "—"
        lines.append(head + f" {c['pr_auc']:.4f} {_ci(c['pr_auc_ci'])} | {c['calibration_slope']:.2f} "
                     f"{_ci(c['calibration_slope_ci'], '.2f')} | {c['calibration_intercept']:+.2f} "
                     f"| {c['mean_predicted']:.4f} | {cov} |")
    for pop in pops[1:]:
        lines += ["", f"## Secondary population: {pop}", "",
                  "| dimension | group | horizon | player-weeks | injuries | prevalence | PR-AUC [95% CI] |",
                  "|---|---|---|---|---|---|---|"]
        for c in r["cells"]:
            if c["population"] != pop:
                continue
            pr = f"{c['pr_auc']:.4f} {_ci(c['pr_auc_ci'])}" if c["status"] == "ok" else "insufficient"
            lines.append(f"| {c['dimension']} | {c['group']} | {c['label']} | {c['n']:,} | {c['positives']} "
                         f"| {c['prevalence']:.4f} | {pr} |")
    return "\n".join(lines) + "\n"


# --- exp-0026: per-season stability (development seasons only) ----------------------------

def seasons(cfg: dict, path) -> dict:
    exp = load_experiment(CONFIGS_DIR / f"{cfg['champion']}.yaml")
    data_cfg, numeric, categorical = exp["data"], exp["numeric"], exp["categorical"]
    panel = pd.read_parquet(panel_path(data_cfg))
    digest = content_hash(panel)
    if digest != panel_meta(data_cfg)["panel_sha256"]:
        raise SystemExit("feature store content does not match its recorded hash")
    if not leakage_tests_pass():
        raise SystemExit("leakage tests failing: refusing to fit")
    train_first = exp.get("train_first_season", data_cfg["seasons"]["first"])
    stored = load_predictions(cfg["champion"])
    n, seed = cfg["n_resamples"], cfg["seed"]
    months = {pop: _months(pop) for pop in cfg["populations"]}
    result = {**_provenance(cfg, path), "panel_sha256": digest, "leakage_tests": "pass", "fits": [], "cells": []}
    frames = []
    for label in cfg["labels"]:
        for season in cfg["seasons"]:
            split = splits.origin_split(panel, data_cfg, season, label)
            tr = panel.iloc[split.train_idx]
            tr = tr[tr["season"] >= train_first]
            ev = panel.iloc[split.eval_idx]
            categories = fit_categories(tr, categorical)
            model = build_model(exp["model"], numeric, categorical, exp["seed"])
            model.fit(prepare_X(tr, numeric, categorical, categories), tr[label].astype(int))
            p_raw = predict_risk(model, prepare_X(ev, numeric, categorical, categories))
            fit = {"label": label, "season": season, "train_rows": len(tr), "train_seasons": int(tr["season"].nunique()),
                   "train_cutoff_t": str(split.train_cutoff_t.date()), "eval_rows": len(ev),
                   "eval_players": int(ev["player_id"].nunique())}
            y = ev[label].astype(int).to_numpy()
            if season >= data_cfg["seasons"]["first"] + 2:
                p, record, _ = _calibrate(exp, panel, label, season, p_raw, train_first, (model, categories))
                gap = abs(metrics.pr_auc(y, p) - metrics.pr_auc(y, p_raw))
                fit.update(calibrated=True, platt_a=record["a"], platt_b=record["b"], pr_auc_raw_vs_calibrated=gap)
                if record["b"] <= 0 or gap > 1e-9:
                    raise SystemExit(f"{label} {season}: Platt map not increasing or ranking changed ({record})")
            else:
                p = p_raw
                fit.update(calibrated=False)
            ref = stored[(stored["label"] == label) & (stored["origin"] == season)]
            if len(ref):
                both = ev[KEYS].assign(p=p).merge(ref[KEYS + ["p"]], on=KEYS, suffixes=("", "_stored"))
                fit["reproduces_stored"] = bool(len(both) == len(ev) == len(ref)
                                                and np.array_equal(both["p"], both["p_stored"]))
                if not fit["reproduces_stored"]:
                    raise SystemExit(f"{label} {season}: does not reproduce the stored {cfg['champion']} predictions")
            result["fits"].append(fit)
            frames.append(ev[KEYS].assign(label=label, season=season, y=y, p=p, p_raw=p_raw))
            log.info("%s %s: fitted on %s rows", label, season, f"{len(tr):,}")
            for pop in cfg["populations"]:
                keep = ev["t"].dt.month.isin(months[pop]).to_numpy()
                yy, pp, pid = y[keep], p[keep], ev["player_id"].to_numpy()[keep]
                pr, pr_ci = pr_auc_ci(yy, pp, pid, n, seed)
                cell = {"label": label, "season": season, "population": pop, "n": int(keep.sum()),
                        "players": int(len(np.unique(pid))), "positives": int(yy.sum()), "prevalence": float(yy.mean()),
                        "pr_auc": pr, "pr_auc_ci": pr_ci}
                if fit["calibrated"]:
                    slope, intercept, s_ci = slope_ci(yy, pp, pid, n, seed)
                    cell.update(calibration_slope=slope, calibration_slope_ci=s_ci, calibration_intercept=intercept)
                result["cells"].append(cell)
    out = EXPERIMENTS_DATA_DIR / cfg["id"]
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(out / "predictions.parquet", index=False)
    return result


def _season_name(s: int) -> str:
    return f"{s}/{(s + 1) % 100:02d}"


def _seasons_report(r: dict) -> str:
    lines = [f"# {r['id']} — per-season stability of the champion pipeline (development seasons, validation group)",
             "", "Champion pipeline refitted per season on earlier train-group seasons (PLAN decision 23); "
             "95% player-cluster bootstrap CIs.", ""]
    for pop in dict.fromkeys(c["population"] for c in r["cells"]):
        lines += [f"## {pop}", "", "| season | horizon | train seasons | player-weeks | injuries | prevalence "
                  "| PR-AUC [95% CI] | PR-AUC / prevalence | calibration slope [95% CI] |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for c in r["cells"]:
            if c["population"] != pop:
                continue
            fit = next(f for f in r["fits"] if f["label"] == c["label"] and f["season"] == c["season"])
            slope = (f"{c['calibration_slope']:.2f} {_ci(c['calibration_slope_ci'], '.2f')}"
                     if "calibration_slope" in c else "— (no inner season)")
            lines.append(f"| {_season_name(c['season'])} | {c['label']} | {fit['train_seasons']} | {c['n']:,} "
                         f"| {c['positives']} | {c['prevalence']:.4f} | {c['pr_auc']:.4f} {_ci(c['pr_auc_ci'])} "
                         f"| {c['pr_auc'] / c['prevalence']:.2f} | {slope} |")
        lines.append("")
    return "\n".join(lines)


# --- exp-0028: workload-ratio baseline ------------------------------------------------------

def acwr(cfg: dict, path) -> dict:
    n, seed = cfg["n_resamples"], cfg["seed"]
    flag_at = cfg["flag_above"]
    forms = cfg["forms"]
    pts = metrics.DECISION_THRESHOLDS
    result = {**_provenance(cfg, path), "labels": {}}
    for label in cfg["labels"]:
        rows = holdout_rows(label, [forms[f] for f in forms])
        result["labels"][label] = {}
        for pop in cfg["populations"]:
            r = rows[rows["t"].dt.month.isin(_months(pop))].reset_index(drop=True)
            y, pid = r["y"].to_numpy(), r["player_id"].to_numpy()
            scores = {"champion": r["p"].to_numpy()}
            info = {}
            for form, column in forms.items():
                ratio = r[column].to_numpy(float)
                defined = ~np.isnan(ratio)
                scores[form] = np.where(defined, ratio, -1.0)
                flag = defined & (ratio > flag_at)
                scores[f"{form}_flag"] = flag.astype(float)
                info[form] = {"undefined_share": float((~defined).mean()), "flag_share": float(flag.mean()),
                              "flag_precision": float(y[flag].mean()) if flag.any() else None,
                              "flag_recall": float(flag[y == 1].mean())}
                d = defined
                sub = metrics.OriginScores(0, pid[d], y[d], {"champion": scores["champion"][d], form: ratio[d]})
                boot = metrics.bootstrap_pr_auc([sub], n, seed)
                info[form]["defined_rows"] = {
                    "n": int(d.sum()), "prevalence": float(y[d].mean()),
                    "champion_pr_auc": metrics.pr_auc(y[d], scores["champion"][d]),
                    "pr_auc": metrics.pr_auc(y[d], ratio[d]),
                    "pr_auc_ci": metrics.percentile_ci(boot[form][:, 0]),
                    "delta": metrics.pr_auc(y[d], scores["champion"][d]) - metrics.pr_auc(y[d], ratio[d]),
                    "delta_ci": metrics.percentile_ci(boot["champion"][:, 0] - boot[form][:, 0])}
            boot = metrics.bootstrap_pr_auc([metrics.OriginScores(0, pid, y, scores)], n, seed)
            summary = {}
            for m, s in scores.items():
                summary[m] = {"pr_auc": metrics.pr_auc(y, s), "pr_auc_ci": metrics.percentile_ci(boot[m][:, 0])}
                if m != "champion":
                    summary[m].update(delta=summary["champion"]["pr_auc"] - summary[m]["pr_auc"],
                                      delta_ci=metrics.percentile_ci(boot["champion"][:, 0] - boot[m][:, 0]))
            curves = {m: net_benefit_parts(y.astype(float), scores[m], pts).mean(axis=0).tolist()
                      for m in ["champion", *[f"{f}_flag" for f in forms]]}
            result["labels"][label][pop] = {"n": len(r), "prevalence": float(y.mean()), "scores": summary,
                                            "forms": info, "net_benefit": curves, "thresholds": pts.tolist()}
    return result


def _acwr_report(r: dict, cfg: dict) -> str:
    lines = [f"# {r['id']} — workload-ratio baseline vs the frozen champion (holdout 2024/25)", "",
             "**The specified 7:28-day ACWR cannot be computed: the datalake has no dated minutes (PLAN decision "
             "20).** Scored instead: season-scale analogues (last-season minutes over earlier seasons), "
             f"continuous and flagged above {cfg['flag_above']}. Baseline comparison only — no promotion decision. "
             "Undefined ratios rank lowest and are never flagged. Δ = champion − baseline, paired 95% "
             "player-cluster bootstrap CIs.", ""]
    for label, pops in r["labels"].items():
        for pop, o in pops.items():
            lines += [f"## {label[1:]}-day, {pop} — {o['n']:,} player-weeks, prevalence {o['prevalence']:.4f}", "",
                      "| score | PR-AUC [95% CI] | Δ champion − score [95% CI] |", "|---|---|---|"]
            for m, s in o["scores"].items():
                delta = f"{s['delta']:+.4f} {_ci(s['delta_ci'])}" if "delta" in s else "—"
                lines.append(f"| {m} | {s['pr_auc']:.4f} {_ci(s['pr_auc_ci'])} | {delta} |")
            lines += ["", "| form | undefined share | flag share | flag precision | flag recall "
                      "| defined rows: n / prevalence | defined rows: champion vs ratio PR-AUC | Δ [95% CI] |",
                      "|---|---|---|---|---|---|---|---|"]
            for form, i in o["forms"].items():
                d = i["defined_rows"]
                prec = f"{i['flag_precision']:.4f}" if i["flag_precision"] is not None else "—"
                lines.append(f"| {form} | {i['undefined_share']:.3f} | {i['flag_share']:.3f} | {prec} "
                             f"| {i['flag_recall']:.3f} | {d['n']:,} / {d['prevalence']:.4f} "
                             f"| {d['champion_pr_auc']:.4f} vs {d['pr_auc']:.4f} {_ci(d['pr_auc_ci'])} "
                             f"| {d['delta']:+.4f} {_ci(d['delta_ci'])} |")
            lines.append("")
    return "\n".join(lines)


# --- dispatch ---------------------------------------------------------------------------

def run(path) -> dict:
    from . import figures

    cfg = load_yaml(path)
    kind = cfg["analysis"]
    if kind == "operational":
        result = operational(cfg, path)
        _write(cfg, result, _operational_report(result))
        figures.operational(result, cfg["id"])
    elif kind == "subgroups":
        result = subgroups(cfg, path)
        _write(cfg, result, _subgroup_report(result, cfg["title"]))
        figures.subgroups(result, cfg)
    elif kind == "seasons":
        result = seasons(cfg, path)
        _write(cfg, result, _seasons_report(result))
        figures.seasons(result, cfg)
    elif kind == "acwr":
        result = acwr(cfg, path)
        _write(cfg, result, _acwr_report(result, cfg))
        overlay = {(label, pop): (cfg["overlay"]["name"], o["net_benefit"][f"{cfg['overlay']['form']}_flag"])
                   for label, pops in result["labels"].items() for pop, o in pops.items()}
        figures.holdout_figures(overlay=overlay)
    else:
        raise SystemExit(f"unknown analysis: {kind}")
    return result
