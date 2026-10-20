"""Stage 0 data audit: profiles the raw tables and derives the analysis scope.

Run as: python -m src.data_audit
Writes reports/stage0_audit.md and summary CSVs under data/audit/.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import datasets, exposure, labels, membership
from .config import load_data_config
from .constants import (
    AUDIT_DIR,
    REPORTS_DIR,
    season_effective_end,
    season_start_year,
    season_start_years,
    season_window,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# A season counts as complete when injury records reach within this many days of
# the season end (Jun 30) and the season ended at least this many days before the
# snapshot.
COVERAGE_TOLERANCE_DAYS = 45
# Top-league squads are fully scraped when some player was in the matchday squad
# for at least this many league games (34 = shortest top-5 season, Bundesliga).
MIN_TOP_SQUAD_GAMES = 34
# Label completeness: injury-reporting density must reach this share of the
# reference density (median of the REFERENCE_SEASONS complete seasons before the
# holdout season).
MIN_REL_REPORTING_DENSITY = 0.80
REFERENCE_SEASONS = 5
AUDIT_FIRST_SEASON = 2005


def _table_summary(name: str, df: pd.DataFrame, date_cols: list[str], key_cols: list[str]) -> dict:
    row = {"table": name, "rows": len(df)}
    for col in date_cols:
        row[f"{col}_min"] = df[col].min()
        row[f"{col}_max"] = df[col].max()
    for col in key_cols:
        row[f"null_{col}"] = float(df[col].isna().mean())
    return row


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{int(v)}" if float(v).is_integer() else f"{v:.4f}"
    if isinstance(v, pd.Timestamp):
        return str(v.date())
    return str(v)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _minutes_section(perf: pd.DataFrame, last_fit_season: int, top: list[str]) -> list[str]:
    goals_state = np.select([perf["goals"].isna(), perf["goals"].eq(0)], ["missing (goalkeepers)", "0"], ">= 1")
    cross = pd.crosstab(goals_state, perf["minutes_played"].isna().map({True: "minutes null", False: "minutes present"}))
    cross.to_csv(AUDIT_DIR / "minutes_vs_goals.csv")

    coefs = exposure.fit_minutes_model(perf, last_fit_season)
    total = exposure.reconstruct_total_minutes(perf)
    pred = exposure.estimate_minutes(perf, coefs)
    known = total.notna() & perf["nb_on_pitch"].gt(0)
    gk = perf["goals"].isna()
    groups = {
        f"outfield, seasons <= {last_fit_season} (fit)": known & ~gk & perf["season"].le(last_fit_season),
        f"outfield, seasons > {last_fit_season}": known & ~gk & perf["season"].gt(last_fit_season),
        f"goalkeepers, seasons > {last_fit_season}": known & gk & perf["season"].gt(last_fit_season),
        f"top-5 leagues, seasons > {last_fit_season}": known & perf["competition_id"].isin(top)
        & perf["season"].gt(last_fit_season),
    }
    rows = []
    for name, m in groups.items():
        y, p = total[m].to_numpy(), pred[m].to_numpy()
        rows.append({
            "rows": name, "n": int(m.sum()),
            "R2": 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
            "MAE_min": float(np.abs(y - p).mean()),
            "bias_min": float((p - y).mean()),
        })
    val = pd.DataFrame(rows)
    val.to_csv(AUDIT_DIR / "minutes_model_validation.csv", index=False)
    return [
        "## Minutes column defect and exposure estimate",
        "",
        "`minutes_played` is missing on every row with `goals == 0`:",
        "",
        _md_table(cross.reset_index().rename(columns={"row_0": "goals"})),
        "",
        "For outfield players the column holds **minutes per goal** (e.g. 48 goals × 65 = 3,120 min "
        "for a 35-game LaLiga season); for goalkeepers (`goals` empty) it holds total minutes. "
        "Total minutes are recoverable only for goalkeepers and goal scorers, so raw minutes would "
        "encode goal scoring. Appearance counts are complete; minutes are estimated from them for every row.",
        "",
        f"Estimator (least squares, no intercept, fitted on seasons <= {last_fit_season}): "
        f"minutes = {coefs[0]:.2f} × full games + {coefs[1]:.2f} × starts subbed off "
        f"+ {coefs[2]:.2f} × substitute appearances",
        "",
        _md_table(val),
        "",
    ]


def _labels_section(injuries: pd.DataFrame, mapping: dict) -> tuple[list[str], pd.DataFrame]:
    events = labels.prepare_events(injuries, mapping)
    table = labels.map_reasons(injuries["injury_reason"], mapping)
    counts = injuries["injury_reason"].value_counts().rename("count")
    table = table.merge(counts, left_on="injury_reason", right_index=True).sort_values("count", ascending=False)
    table.to_csv(AUDIT_DIR / "label_mapping_table.csv", index=False)

    def share(col: str, df: pd.DataFrame) -> pd.DataFrame:
        out = df[col].value_counts().rename_axis(col).to_frame("rows")
        out["share"] = out["rows"] / len(df)
        return out.reset_index()

    inj = events[events["category"].eq("injury")]
    ge7 = events["days_missed"].ge(7)
    unspecified = table[(table["category"] == "injury") & (table["tissue"] == "unspecified")].head(15)
    md = [
        "## Label mapping (`configs/label_mapping.yaml`)",
        "",
        f"- events after dropping exact duplicates / missing onset: {len(events):,}",
        "- full reason → category / body part / tissue table: `data/audit/label_mapping_table.csv`",
        "",
        "### Category (all events; ≥ 7 days missed)",
        "",
        _md_table(share("category", events).merge(
            share("category", events[ge7]), on="category", suffixes=("_all", "_ge7"))),
        "",
        "### Body part and tissue (injury events)",
        "",
        _md_table(share("body_part", inj)),
        "",
        _md_table(share("tissue", inj)),
        "",
        "### Most frequent injury reasons with unspecified tissue",
        "",
        _md_table(unspecified[["injury_reason", "body_part", "count"]]),
        "",
    ]
    return md, events


def _scope_section(
    injuries: pd.DataFrame, events: pd.DataFrame, perf: pd.DataFrame, snapshot_end: pd.Timestamp, cfg: dict
) -> tuple[list[str], int, int]:
    top = cfg["competitions"]
    overrides = cfg["season_end_overrides"]
    inj_season = injuries.assign(season=season_start_years(injuries["from_date"]))
    per_season = inj_season.groupby("season").agg(max_from=("from_date", "max"))

    top_perf = perf[perf["competition_id"].isin(top)]
    squads = top_perf.groupby("season").agg(
        top_player_seasons=("player_id", "nunique"), top_max_squad_games=("nb_in_group", "max"))
    # reporting density: >= 7-day injuries (any category) per top-5 player-season
    ps = top_perf[["player_id", "season"]].drop_duplicates()
    ev = events[events["days_missed"].ge(7)].assign(season=season_start_years(events["from_date"]))
    n_inj = ev.groupby(["player_id", "season"]).size().rename("n")
    density = ps.join(n_inj, on=["player_id", "season"]).fillna({"n": 0}).groupby("season")["n"].mean()

    ev_all = per_season.join(squads, how="outer").join(density.rename("injuries_ge7_per_player_season"))
    ev_all = ev_all.loc[AUDIT_FIRST_SEASON:].reset_index().rename(columns={"index": "season"})
    ends = ev_all["season"].map(lambda y: season_effective_end(int(y), overrides))
    # injury onsets are binned by calendar season, so compare with the calendar end
    calendar_ends = ev_all["season"].map(lambda y: season_window(int(y))[1])
    ev_all["ended_before_snapshot"] = ends <= snapshot_end - pd.Timedelta(days=COVERAGE_TOLERANCE_DAYS)
    ev_all["injuries_reach_june"] = (calendar_ends - ev_all["max_from"]).dt.days <= COVERAGE_TOLERANCE_DAYS
    ev_all["top_squads_complete"] = ev_all["top_max_squad_games"] >= MIN_TOP_SQUAD_GAMES
    ev_all["inputs_complete"] = (
        ev_all["ended_before_snapshot"] & ev_all["injuries_reach_june"] & ev_all["top_squads_complete"]
    )
    complete = ev_all.loc[ev_all["inputs_complete"], "season"].astype(int)
    holdout = int(complete.max())
    reference = ev_all[ev_all["season"].isin(complete[complete < holdout].tail(REFERENCE_SEASONS))]
    ref_density = float(reference["injuries_ge7_per_player_season"].median())
    ev_all["labels_complete"] = (
        ev_all["injuries_ge7_per_player_season"] >= MIN_REL_REPORTING_DENSITY * ref_density
    )
    ok = ev_all["inputs_complete"] & ev_all["labels_complete"]
    # first season of the contiguous run of usable seasons ending at the holdout
    first = holdout
    for season, good in zip(ev_all["season"][::-1], ok[::-1]):
        if season > holdout:
            continue
        if not good:
            break
        first = int(season)
    ev_all.to_csv(AUDIT_DIR / "season_completeness.csv", index=False)
    md = [
        "## Season scope evidence",
        "",
        f"- snapshot end (max injury onset): **{snapshot_end.date()}**",
        f"- inputs complete: season ended ≥ {COVERAGE_TOLERANCE_DAYS} days before the snapshot, injury "
        f"onsets reach within {COVERAGE_TOLERANCE_DAYS} days of the season end, and some top-5 player "
        f"was in ≥ {MIN_TOP_SQUAD_GAMES} matchday squads (performance scrape not cut mid-season)",
        f"- labels complete: ≥ 7-day injuries per top-5 player-season ≥ {MIN_REL_REPORTING_DENSITY:.0%} "
        f"of the median of the {REFERENCE_SEASONS} complete seasons before the last one "
        f"(reference {ref_density:.3f})",
        "",
        _md_table(ev_all.assign(max_from=ev_all["max_from"].dt.strftime("%Y-%m-%d"))),
        "",
        f"**Derived scope:** seasons {first}/{str(first + 1)[2:]} .. {holdout}/{str(holdout + 1)[2:]}; "
        f"last complete season {holdout}/{str(holdout + 1)[2:]} is the final holdout.",
        "",
    ]
    return md, first, holdout


def _membership_section(transfers: pd.DataFrame, tcs: pd.DataFrame, perf: pd.DataFrame, cfg: dict) -> list[str]:
    spells = membership.club_spells(transfers)
    top = membership.top_league_clubs(tcs, cfg["competitions"])
    rows = []
    for season in range(cfg["seasons"]["first"], cfg["seasons"]["last"] + 1):
        clubs = set(top.loc[top["season"] == season, "club_id"])
        perf_players = set(perf.loc[perf["competition_id"].isin(cfg["competitions"])
                                    & perf["season"].eq(season), "player_id"])
        registered = set()
        for d in (pd.Timestamp(season, 10, 15), pd.Timestamp(season + 1, 3, 15)):
            on = membership.club_on(spells, d)
            registered |= set(on[on.isin(clubs)].index)
        rows.append({
            "season": season,
            "with_top5_appearance_record": len(perf_players),
            "registered_oct15_or_mar15": len(registered),
            "records_covered_by_registration": len(perf_players & registered) / len(perf_players),
            "registered_without_record": len(registered - perf_players) / len(registered),
        })
    table = pd.DataFrame(rows)
    table.to_csv(AUDIT_DIR / "membership_agreement.csv", index=False)
    return [
        "## Panel membership: registration vs appearance records",
        "",
        "Registration (last transfer dated ≤ d) is knowable at d; appearance records depend on the whole "
        "season. Records not covered by senior-team registration are mostly youth/reserve registrations "
        "(B teams, U19/U23) with few first-team games; registered players without a record were never "
        "in a matchday squad (e.g. injured all season) and are kept.",
        "",
        _md_table(table),
        "",
    ]


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_data_config()
    mapping = labels.load_label_mapping()

    log.info("hashing raw csvs")
    datasets.write_hashes()

    injuries = datasets.load_injuries()
    perf = exposure.add_season(datasets.load_performances())
    profiles = datasets.load_profiles()
    tcs = datasets.load_team_competitions_seasons()
    transfers = datasets.load_transfer_history()
    mv = datasets.load_market_values()

    md: list[str] = ["# Stage 0 data audit", ""]

    # --- per-table summary -------------------------------------------------
    summaries = [
        _table_summary("player_injuries", injuries, ["from_date", "end_date"],
                       ["player_id", "season_name", "injury_reason", "from_date", "end_date", "days_missed"]),
        _table_summary("player_performances", perf, [],
                       ["player_id", "season_name", "competition_id", "minutes_played"]),
        _table_summary("player_profiles", profiles, ["date_of_birth"],
                       ["player_id", "date_of_birth", "height", "main_position", "foot"]),
        _table_summary("transfer_history", transfers, ["transfer_date"], ["player_id", "transfer_date"]),
        _table_summary("team_competitions_seasons", tcs, [],
                       ["club_id", "competition_id", "season_id", "season_league_level_level_number"]),
        _table_summary("player_market_value", mv, ["date"], ["player_id", "date", "value"]),
    ]
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(AUDIT_DIR / "table_summary.csv", index=False)
    md += ["## Per-table summary", "", _md_table(summary_df), ""]

    # --- injuries ----------------------------------------------------------
    # Season labels on injury rows are noisy (from_date often falls outside the
    # labeled season); assign seasons by the real onset date instead.
    onset_season = season_start_years(injuries["from_date"])
    label_season = injuries["season_name"].map(season_start_year)
    n_season_mismatch = int((label_season != onset_season).sum())
    snapshot_end = injuries["from_date"].max()
    inj_per_season = (
        injuries.assign(season=onset_season).groupby("season")
        .agg(n_records=("player_id", "size"), min_from=("from_date", "min"), max_from=("from_date", "max"))
        .reset_index()
    )
    inj_per_season.to_csv(AUDIT_DIR / "injuries_per_season.csv", index=False)

    days = injuries["days_missed"]
    days_stats = days.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    n_ge3 = int((days >= 3).sum())
    n_ge7 = int((days >= 7).sum())
    consistent = int(((injuries["end_date"] - injuries["from_date"]).dt.days + 1 == days).sum())

    reason_counts = (
        injuries["injury_reason"].value_counts(dropna=False).rename_axis("injury_reason").to_frame("count")
    )
    reason_counts.to_csv(AUDIT_DIR / "injury_reason_counts.csv")
    n_rare = int((reason_counts["count"] < 100).sum())
    share_rare = float(reason_counts.loc[reason_counts["count"] < 100, "count"].sum() / len(injuries))

    md += [
        "## Injuries",
        "",
        f"- rows: {len(injuries):,}; players: {injuries['player_id'].nunique():,}; "
        f"exact-duplicate rows: {int(injuries.duplicated().sum()):,}",
        f"- snapshot end (max from_date): {snapshot_end.date()}",
        f"- days_missed stats: mean={days_stats['mean']:.1f}, p50={days_stats['50%']:.0f}, "
        f"p90={days_stats['90%']:.0f}, p99={days_stats['99%']:.0f}, max={days_stats['max']:.0f}",
        f"- spells with days_missed >= 3: {n_ge3:,} ({n_ge3 / len(injuries):.1%}); "
        f">= 7: {n_ge7:,} ({n_ge7 / len(injuries):.1%})",
        f"- days_missed == end_date - from_date + 1 on {consistent:,} of "
        f"{int(injuries['end_date'].notna().sum()):,} rows with an end date (end_date = last day missed)",
        f"- distinct injury_reason values: {len(reason_counts)}; "
        f"reasons with < 100 occurrences: {n_rare} covering {share_rare:.2%} of rows",
        f"- rows whose season label disagrees with the onset date: {n_season_mismatch:,} "
        f"({n_season_mismatch / len(injuries):.1%}) — labels are ignored, real dates are used",
        "",
        "### Records per season (onset date)",
        "",
        _md_table(inj_per_season[inj_per_season["season"] >= AUDIT_FIRST_SEASON]),
        "",
    ]
    label_md, events = _labels_section(injuries, mapping)
    md += label_md

    # --- performances ------------------------------------------------------
    perf_per_season = (
        perf.groupby("season").agg(n_records=("player_id", "size")).reset_index()
    )
    perf_per_season.to_csv(AUDIT_DIR / "performances_per_season.csv", index=False)
    dup_perf = int(perf.duplicated(subset=["player_id", "season_name", "competition_id"]).sum())
    dup_perf_team = int(perf.duplicated(subset=["player_id", "season_name", "competition_id", "team_id"]).sum())
    md += [
        "## Performances",
        "",
        f"- rows: {len(perf):,}; players: {perf['player_id'].nunique():,}",
        "- grain: player × season × competition × team (season aggregates, no match-level rows)",
        f"- rows duplicated on (player, season, competition): {dup_perf:,}; "
        f"on (player, season, competition, team): {dup_perf_team:,} — mid-season moves within a league",
        "",
    ]
    md += _minutes_section(perf, cfg["exposure"]["minutes_model_last_season"], cfg["competitions"])

    # --- player overlap ----------------------------------------------------
    p_perf = set(perf["player_id"].unique())
    p_inj = set(injuries["player_id"].unique())
    p_prof = set(profiles["player_id"].unique())
    p_trans = set(transfers["player_id"].unique())
    md += [
        "## Player overlap (all ids are Transfermarkt player ids; no cross-source matching needed)",
        "",
        f"- players in performances: {len(p_perf):,}; injuries: {len(p_inj):,}; "
        f"profiles: {len(p_prof):,}; transfers: {len(p_trans):,}",
        f"- |performances ∩ injuries ∩ profiles|: {len(p_perf & p_inj & p_prof):,}",
        f"- injured players without a profile: {len(p_inj - p_prof):,}",
        "",
    ]

    scope_md, first, holdout = _scope_section(injuries, events, perf, snapshot_end, cfg)
    md += scope_md
    md += _membership_section(transfers, tcs, perf, cfg)

    out = REPORTS_DIR / "stage0_audit.md"
    out.write_text("\n".join(md), encoding="utf-8")
    log.info("wrote %s", out)
    print(f"derived scope: seasons {first}..{holdout} (holdout {holdout}); "
          f"configured: {cfg['seasons']['first']}..{cfg['seasons']['last']}")
    if (first, holdout) != (cfg["seasons"]["first"], cfg["seasons"]["last"]):
        log.warning("configs/data.yaml season scope differs from the audit-derived scope")


if __name__ == "__main__":
    main()
