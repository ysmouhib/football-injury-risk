"""Report figures (Stage 5) as static print PNGs in reports/figures/.

Run as: python -m src.figures
Colours: the reference palette's validated categorical slots (1-3 for
multi-series charts), its blue/red diverging pair and blue ordinal ramp; text in
ink tokens only; solid hairline grids; one y-axis per panel.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import metrics  # noqa: E402
from .config import load_yaml  # noqa: E402
from .constants import CONFIGS_DIR, REPORTS_DIR  # noqa: E402
from .experiment import EXPERIMENTS_DATA_DIR, load_predictions, run_dir  # noqa: E402

OUT = REPORTS_DIR / "figures"
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DIVERGING = {"neg": "#2a78d6", "pos": "#e34948"}
ORDINAL = ["#86b6ef", "#2a78d6", "#104281"]
LINE, MARKER = 1.5, 6

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.labelcolor": INK_2, "text.color": INK,
})


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _metrics(exp_id: str) -> dict:
    return json.loads((run_dir(exp_id) / "metrics.json").read_text(encoding="utf-8"))


def trajectory() -> None:
    """Arm A: main-origin PR-AUC with 95% CI per experiment; promoted champions highlighted."""
    runs = []
    for path in sorted(EXPERIMENTS_DATA_DIR.glob("exp-*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        if m.get("arm") == "survival":
            continue
        cfg = load_yaml(CONFIGS_DIR / f"{m['id']}.yaml")
        verdict = m.get("comparison", {}).get("verdict", "BASELINE")
        runs.append((m["id"], cfg["title"], verdict, m))
    champions = {"exp-0001", "exp-0004", "exp-0016", "exp-0018"}
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    x = np.arange(len(runs))
    for ax, label in zip(axes, ("y28", "y7")):
        vals = np.array([r[3]["labels"][label]["aggregate"]["main"]["pr_auc"] for r in runs])
        ci = np.array([r[3]["labels"][label]["aggregate"]["main"]["pr_auc_ci"] for r in runs])
        is_champ = np.array([r[0] in champions for r in runs])
        best = np.maximum.accumulate(np.where(is_champ, vals, -np.inf))
        ax.step(x, best, where="post", color=SERIES[0], lw=LINE, label="champion of record")
        for mask, colour, name in ((~is_champ, MUTED, "challenger not promoted"), (is_champ, SERIES[0], "promoted")):
            ax.errorbar(x[mask], vals[mask], yerr=[vals[mask] - ci[mask, 0], ci[mask, 1] - vals[mask]], fmt="o",
                        ms=MARKER, color=colour, ecolor=colour, elinewidth=1, capsize=0, mec=SURFACE, mew=1.2,
                        label=name)
        prevalence = runs[0][3]["labels"][label]["per_origin"]["2023"]["prevalence"]
        ax.set_ylim(prevalence * 0.9, ci[:, 1].max() * 1.04)
        ax.axhline(prevalence, color=AXIS, lw=1)
        ax.annotate("prevalence (no-skill PR-AUC)", (x[-1], prevalence), xytext=(0, 4),
                    textcoords="offset points", ha="right", color=MUTED, fontsize=8)
        ax.set_ylabel(f"PR-AUC, {label} (2023/24)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8)
    axes[1].set_xticks(x, [r[0].replace("exp-", "") for r in runs], rotation=0, fontsize=7.5)
    axes[1].set_xlabel("experiment (classification arm, chronological)")
    fig.suptitle("Evidence-gated progression of the classification champion", x=0.01, ha="left", color=INK)
    _save(fig, "fig1_trajectory")


def reliability(champion: str, survival_champion: str) -> None:
    preds, surv = load_predictions(champion), load_predictions(survival_champion)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, h in zip(axes, (7, 28)):
        rows = preds[(preds["label"] == f"y{h}") & (preds["origin"] == 2023)]
        s = surv[surv["origin"] == 2023].merge(rows[["player_id", "t", "y"]], on=["player_id", "t"])
        top = 0
        for (name, y, p), colour in zip(((f"{champion} (Platt-calibrated classifier)", rows["y"], rows["p"]),
                                         (f"{survival_champion} (survival 1 - S({h}))", s["y"], 1 - s[f"S{h}"])), SERIES):
            t = metrics.reliability_table(y, p, bins=10)
            ax.plot(t["mean_predicted"], t["observed"], "-o", color=colour, lw=LINE, ms=MARKER, mec=SURFACE,
                    mew=1.2, label=name)
            top = max(top, t["mean_predicted"].max(), t["observed"].max())
        lim = top * 1.08
        ax.plot([0, lim], [0, lim], color=AXIS, lw=1)
        ax.set_xlim(0, lim), ax.set_ylim(0, lim)
        ax.set_title(f"{h}-day horizon", loc="left")
        ax.set_xlabel("mean predicted risk (decile)")
        ax.set_ylabel("observed injury rate")
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("Calibration on the 2023/24 validation season", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    _save(fig, "fig2_reliability")


def decision_curves(champion: str, survival_champion: str) -> None:
    dc = json.loads((run_dir(champion) / "decision_curves.json").read_text(encoding="utf-8"))
    pts = np.array(dc["thresholds"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, label, xmax in zip(axes, ("y28", "y7"), (0.20, 0.08)):
        r = dc["labels"][label]
        keep = pts <= xmax
        ci = np.array(r["champion_nb_ci"])
        ax.fill_between(pts[keep], ci[keep, 0], ci[keep, 1], color=SERIES[0], alpha=0.12, lw=0)
        ax.plot(pts[keep], np.array(r["net_benefit"][champion])[keep], color=SERIES[0], lw=LINE, label=f"{champion} (champion)")
        ax.plot(pts[keep], np.array(r["net_benefit"][survival_champion])[keep], color=SERIES[1], lw=LINE,
                label=f"{survival_champion} (survival)")
        ax.plot(pts[keep], np.array(r["treat_all"])[keep], color=SERIES[2], lw=LINE, label="treat all")
        ax.axhline(0, color=AXIS, lw=1)
        top = max(np.array(r["net_benefit"][champion])[keep].max(), r["prevalence"])
        ax.set_ylim(-0.25 * top, top * 1.15)
        ax.set_xlim(0, xmax)
        if r["useful_range"]:
            ax.axvspan(*r["useful_range"], color=GRID, alpha=0.5, lw=0)
            ax.text(np.mean(r["useful_range"]), top * 1.1, "useful\nrange", ha="center", va="top",
                    color=MUTED, fontsize=7, linespacing=0.9)
        ax.set_title(f"{label[1:]}-day horizon (prevalence {r['prevalence']:.1%})", loc="left")
        ax.set_xlabel("threshold probability")
        ax.set_ylabel("net benefit")
    axes[0].legend(loc="upper right", fontsize=7.5)
    fig.suptitle("Decision curves, 2023/24 validation season (band: champion 95% CI)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    _save(fig, "fig3_decision_curves")


def shap_global(champion: str) -> None:
    r = json.loads((run_dir(champion) / "interpret" / "interpret.json").read_text(encoding="utf-8"))
    L = r["labels"]["y28"]
    feats = L["factors"]["features"]
    classes = list(L["factors"]["classes"])
    top = list(L["global_importance"].items())[:15][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, (name, value) in enumerate(top):
        colour = SERIES[classes.index(feats[name]["class"])]
        ax.barh(i, value, height=0.62, color=colour)
        ax.text(value, i, f"  {feats[name]['share']:.0%}", va="center", color=INK_2, fontsize=7.5)
    ax.set_yticks(range(len(top)), [n for n, _ in top])
    ax.set_xlabel("mean |SHAP| on the calibrated log-odds, 28-day risk (2023/24)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[i]) for i in range(len(classes))]
    shares = L["factors"]["classes"]
    ax.legend(handles, [f"{c} — {shares[c]:.0%} of total" for c in classes], loc="lower right", fontsize=7.5)
    ax.set_title("What drives the champion's 28-day risk (top 15 features)", loc="left")
    _save(fig, "fig4_shap_global")


def shap_local(champion: str) -> None:
    r = json.loads((run_dir(champion) / "interpret" / "interpret.json").read_text(encoding="utf-8"))
    local = r["labels"]["y28"]["local"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.6), sharex=True)
    for ax, (title, c) in zip(axes, local.items()):
        items = list(c["top_contributions"].items())[:6][::-1]
        vals = [v for _, v in items]
        ax.barh(range(len(items)), vals, height=0.6,
                color=[DIVERGING["pos"] if v > 0 else DIVERGING["neg"] for v in vals])
        ax.set_yticks(range(len(items)), [k for k, _ in items], fontsize=7.5)
        ax.axvline(0, color=AXIS, lw=1)
        ax.grid(axis="y", visible=False)
        outcome = "injured within 28 d" if c["injured_within_window"] else "not injured within 28 d"
        ax.set_title(f"{c['case']}: {c['position']}, {c['age']:.0f} y\nrisk {c['risk']:.1%}, {outcome}",
                     loc="left", fontsize=8.5)
    axes[1].set_xlabel("SHAP contribution to log-odds (red raises, blue lowers risk)")
    fig.suptitle("Local explanations for three anonymised player-weeks (2023/24)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    _save(fig, "fig5_shap_local")


def dose_response(champion: str, tree: str, survival_champion: str) -> None:
    r = json.loads((run_dir(champion) / "interpret" / "interpret.json").read_text(encoding="utf-8"))
    dose = r["labels"]["y28"]["dose_response"]
    names = {champion: f"{champion} logistic champion", tree: f"{tree} LightGBM", survival_champion: f"{survival_champion} survival forest"}
    labels = {"prev1_minutes_est": "minutes last season\n(chronic load)", "prev2_minutes_est": "minutes two\nseasons ago",
              "prev1_apps": "appearances\nlast season", "age": "age\n(years)",
              "days_since_return": "days since return\nfrom injury"}
    feats = list(dose)
    fig, axes = plt.subplots(1, len(feats), figsize=(7.6, 2.8), sharey=True)
    for ax, f in zip(axes, feats):
        for colour, model in zip(SERIES, (champion, tree, survival_champion)):
            if model not in dose[f]:
                continue
            c = dose[f][model]
            g, band = np.array(c["grid"]), np.array(c["band"])
            ax.fill_between(g, band[:, 0], band[:, 1], color=colour, alpha=0.12, lw=0)
            ax.plot(g, c["pdp"], color=colour, lw=LINE, label=names[model])
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xlabel(labels[f], fontsize=7.5)
    axes[0].set_ylabel("Δ log-odds of 28-day injury\n(partial dependence, centred)")
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Dose-response: no U/J shape in chronic load under any model", x=0.01, y=1.11, ha="left", color=INK)
    fig.tight_layout()
    _save(fig, "fig6_dose_response")


def positions(champion: str) -> None:
    r = json.loads((run_dir(champion) / "interpret" / "interpret.json").read_text(encoding="utf-8"))
    pos = pd.DataFrame(r["positions"])
    pos = pos[(pos["label"] == "y28") & (pos["origin"] == 2023)].set_index("position")
    order = ["Goalkeeper", "Defender", "Midfield", "Attack"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    y = np.arange(len(order))
    ax.hlines(y, pos.loc[order, "prevalence"], pos.loc[order, "pr_auc"], color=AXIS, lw=1)
    ax.plot(pos.loc[order, "prevalence"], y, "o", color=MUTED, ms=MARKER, mec=SURFACE, mew=1.2, label="prevalence (no skill)")
    ax.plot(pos.loc[order, "pr_auc"], y, "o", color=SERIES[0], ms=MARKER, mec=SURFACE, mew=1.2, label="PR-AUC")
    ax.set_yticks(y, order)
    ax.set_xlabel("PR-AUC vs prevalence, 28-day (2023/24)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.5)
    ax.set_title("Discrimination by position", loc="left")
    parts = pd.DataFrame(r["body_parts_y28"]).T.reindex(order).fillna(0)
    keep = ["hamstring", "knee", "ankle", "groin_hip", "lower_leg"]
    parts = pd.concat([parts[keep], parts.drop(columns=keep).sum(axis=1).rename("other / unspecified")], axis=1)
    ax = axes[1]
    left = np.zeros(len(order))
    for colour, col in zip(SERIES[:5] + [GRID], parts.columns):
        ax.barh(y, parts[col], left=left, height=0.55, color=colour, edgecolor=SURFACE, linewidth=1.5,
                label={"groin_hip": "groin / hip", "lower_leg": "lower leg"}.get(col, col))
        left += parts[col].to_numpy()
    ax.set_yticks(y, [])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.set_title("Site of the 28-day injuries", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=7.5)
    fig.tight_layout()
    _save(fig, "fig7_positions")


def conformal_tiers(champion: str) -> None:
    res = json.loads((run_dir(champion) / "conformal.json").read_text(encoding="utf-8"))["results"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
    for ax, label in zip(axes, ("y28", "y7")):
        r = next(x for x in res if x["label"] == label and x["origin"] == 2023 and x["method"] == "mondrian"
                 and abs(x["alpha"] - 0.10) < 1e-9)
        names = ["low ({0})", "uncertain ({0,1})", "high ({1})"]
        rates = [r["tiers"][n]["injury_rate"] for n in names]
        shares = [r["tiers"][n]["share"] for n in names]
        x = np.arange(3)
        ax.bar(x, rates, width=0.5, color=ORDINAL)
        for i in x:
            ax.text(i, rates[i], f"{rates[i]:.1%}\n{shares[i]:.0%} of weeks", ha="center", va="bottom", color=INK_2, fontsize=7.5)
        ax.set_xticks(x, ["low", "uncertain", "high"])
        ax.set_ylim(0, max(rates) * 1.35)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_ylabel("observed injury rate")
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{label[1:]}-day horizon (coverage {r['coverage']:.1%})", loc="left")
    fig.suptitle("Class-conditional conformal risk tiers, α = 0.10, 2023/24", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    _save(fig, "fig8_conformal_tiers")


def in_season(estimates: dict) -> None:
    """Δ PR-AUC (context − history only) on all weeks vs in-season weeks, with 95% CI."""
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    rows = [(f"{lab}, {scope}", v) for lab in ("y28", "y7") for scope, v in estimates[lab].items()]
    y = np.arange(len(rows))[::-1]
    for yi, (name, (d, lo, hi)) in zip(y, rows):
        ax.hlines(yi, lo, hi, color=SERIES[0], lw=LINE)
        ax.plot(d, yi, "o", color=SERIES[0], ms=MARKER, mec=SURFACE, mew=1.2)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y, [n for n, _ in rows])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Δ PR-AUC, context family vs history only (2023/24, 95% CI)")
    ax.set_title("The context gain is a calendar effect: it vanishes within the playing season", loc="left")
    _save(fig, "fig9_context_in_season")


def _in_season_estimates(champion: str, previous: str) -> dict:
    a, b = load_predictions(champion), load_predictions(previous)
    keys = ["label", "origin", "player_id", "t"]
    m = a.merge(b[keys + ["p"]], on=keys, suffixes=("", "_prev"))
    m = m[m["origin"] == 2023]
    out = {}
    for lab in ("y28", "y7"):
        g = m[m["label"] == lab]
        out[lab] = {}
        for scope, rows in (("all weeks", g), ("in-season Sep–Apr", g[g["t"].dt.month.isin([9, 10, 11, 12, 1, 2, 3, 4])])):
            o = metrics.OriginScores(2023, rows["player_id"].to_numpy(), rows["y"].to_numpy(),
                                     {"a": rows["p"].to_numpy(), "b": rows["p_prev"].to_numpy()})
            boot = metrics.bootstrap_pr_auc([o], 1000, 20261020)
            d = boot["a"][:, 0] - boot["b"][:, 0]
            out[lab][scope] = (metrics.pr_auc(rows["y"], rows["p"]) - metrics.pr_auc(rows["y"], rows["p_prev"]),
                               float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    return out


def main() -> None:
    champion, tree, surv = "exp-0018", "exp-0005", "exp-0013"
    trajectory()
    reliability(champion, surv)
    decision_curves(champion, surv)
    shap_global(champion)
    shap_local(champion)
    dose_response(champion, tree, surv)
    positions(champion)
    conformal_tiers(champion)
    estimates = _in_season_estimates(champion, "exp-0016")
    (run_dir(champion) / "interpret" / "in_season.json").write_text(json.dumps(estimates, indent=2), encoding="utf-8")
    in_season(estimates)
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
