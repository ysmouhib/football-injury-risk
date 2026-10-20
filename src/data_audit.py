"""Stage 0 data audit: profiles the raw tables and recommends the analysis scope.

Run as: python -m src.data_audit
Writes reports/stage0_audit.md and summary CSVs under data/audit/.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import datasets
from .constants import (
    AUDIT_DIR,
    REPORTS_DIR,
    SEASON_START_MONTH,
    season_start_year,
    season_window,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# A season counts as complete when injury records reach within this many days of
# the season end (Jun 30) and the season ended at least this many days before the
# snapshot. Record-count stability is checked separately against the peak season.
COVERAGE_TOLERANCE_DAYS = 45
# Performance-record stability: a season counts as complete when its record
# count is at least 45% of the peak season and does not drop more than 20%
# against the previous season (the 2025/26 scrape is ~53% of peak with minutes
# coverage collapsing to 0.27, i.e. clearly partial).
MIN_REL_PERFORMANCE_RECORDS = 0.45
MAX_YOY_DROP = 0.80


def _table_summary(name: str, df: pd.DataFrame, date_cols: list[str], key_cols: list[str]) -> dict:
    row = {"table": name, "rows": len(df)}
    for col in date_cols:
        row[f"{col}_min"] = df[col].min()
        row[f"{col}_max"] = df[col].max()
    for col in key_cols:
        row[f"null_{col}"] = float(df[col].isna().mean())
    return row


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, pd.Timestamp):
        return str(v.date())
    return str(v)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("hashing raw csvs")
    datasets.write_hashes()

    injuries = datasets.load_injuries()
    perf = datasets.load_performances()
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
    injuries = injuries.assign(
        season_start_year=(injuries["from_date"].dt.year
                           - (injuries["from_date"].dt.month < SEASON_START_MONTH))
    )
    label_season = injuries["season_name"].map(season_start_year)
    n_season_mismatch = int((label_season != injuries["season_start_year"]).sum())
    snapshot_end = injuries["from_date"].max()
    inj_per_season = (
        injuries.groupby("season_start_year")
        .agg(n_records=("player_id", "size"),
             min_from=("from_date", "min"),
             max_from=("from_date", "max"))
        .reset_index()
    )
    inj_per_season.to_csv(AUDIT_DIR / "injuries_per_season.csv", index=False)

    days = injuries["days_missed"]
    days_stats = days.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    n_ge3 = int((days >= 3).sum())
    n_ge7 = int((days >= 7).sum())

    reason_counts = (
        injuries["injury_reason"].value_counts(dropna=False).rename_axis("injury_reason").to_frame("count")
    )
    reason_counts.to_csv(AUDIT_DIR / "injury_reason_counts.csv")
    reason_counts.head(50).to_csv(AUDIT_DIR / "injury_reasons_top50.csv")
    n_rare = int((reason_counts["count"] < 100).sum())
    share_rare = float(reason_counts.loc[reason_counts["count"] < 100, "count"].sum() / len(injuries))

    md += [
        "## Injuries",
        "",
        f"- rows: {len(injuries):,}; players: {injuries['player_id'].nunique():,}",
        f"- snapshot end (max from_date): {snapshot_end.date()}",
        f"- days_missed stats: mean={days_stats['mean']:.1f}, p50={days_stats['50%']:.0f}, "
        f"p90={days_stats['90%']:.0f}, p99={days_stats['99%']:.0f}, max={days_stats['max']:.0f}",
        f"- spells with days_missed >= 3: {n_ge3:,} ({n_ge3 / len(injuries):.1%}); "
        f">= 7: {n_ge7:,} ({n_ge7 / len(injuries):.1%})",
        f"- distinct injury_reason values: {len(reason_counts)}; "
        f"reasons with < 100 occurrences: {n_rare} covering {share_rare:.2%} of rows",
        f"- rows whose season label disagrees with the onset date: {n_season_mismatch:,} "
        f"({n_season_mismatch / len(injuries):.1%}) — labels are ignored, real dates are used",
        "",
        "### Records per season",
        "",
        _md_table(inj_per_season),
        "",
        "### Top 50 injury reasons",
        "",
        _md_table(reason_counts.head(50).reset_index()),
        "",
    ]

    # --- performances ------------------------------------------------------
    perf = perf.assign(season_start_year=perf["season_name"].map(season_start_year))
    perf_per_season = (
        perf.groupby("season_start_year")
        .agg(n_records=("player_id", "size"),
             minutes_nonnull_rate=("minutes_played", lambda s: float(s.notna().mean())))
        .reset_index()
    )
    perf_per_season.to_csv(AUDIT_DIR / "performances_per_season.csv", index=False)

    comp_names = tcs[["competition_id", "competition_name"]].drop_duplicates(subset=["competition_id"])
    levels = tcs[["competition_id", "season_id", "season_league_level_level_number"]].drop_duplicates(
        subset=["competition_id", "season_id"]
    )
    perf_l = perf.merge(
        levels.rename(columns={"season_id": "season_start_year"}),
        on=["competition_id", "season_start_year"],
        how="left",
    )
    cov = (
        perf_l[perf_l["season_league_level_level_number"] == 1]
        .groupby(["competition_id", "season_start_year"])
        .agg(n_records=("player_id", "size"),
             minutes_nonnull_rate=("minutes_played", lambda s: float(s.notna().mean())))
        .reset_index()
        .merge(comp_names, on="competition_id", how="left")
    )
    cov.to_csv(AUDIT_DIR / "minutes_coverage_level1_by_season.csv", index=False)

    md += [
        "## Performances",
        "",
        f"- rows: {len(perf):,}; players: {perf['player_id'].nunique():,}",
        f"- overall minutes_played non-null rate: {perf['minutes_played'].notna().mean():.1%}",
        "",
        "### Records per season",
        "",
        _md_table(perf_per_season),
        "",
    ]

    # --- duplicates --------------------------------------------------------
    dup_perf = int(perf.duplicated(subset=["player_id", "season_name", "competition_id"]).sum())
    dup_inj = int(injuries.duplicated().sum())
    md += [
        "## Duplicate-key checks",
        "",
        f"- performances duplicated on (player_id, season_name, competition_id): {dup_perf:,}",
        f"- injuries exact-duplicate rows: {dup_inj:,}",
        "",
    ]

    # --- player overlap ----------------------------------------------------
    p_perf = set(perf["player_id"].unique())
    p_inj = set(injuries["player_id"].unique())
    p_prof = set(profiles["player_id"].unique())
    overlap = p_perf & p_inj & p_prof
    md += [
        "## Player overlap",
        "",
        f"- players in performances: {len(p_perf):,}",
        f"- players in injuries: {len(p_inj):,}",
        f"- players in profiles: {len(p_prof):,}",
        f"- |performances ∩ injuries ∩ profiles|: {len(overlap):,}",
        "",
    ]

    # --- positions ----------------------------------------------------------
    pos_counts = profiles["main_position"].value_counts(dropna=False).rename_axis("main_position").to_frame("count")
    pos_counts.to_csv(AUDIT_DIR / "main_position_counts.csv")

    # --- completeness & scope recommendation --------------------------------
    max_perf_records = int(perf_per_season["n_records"].max())
    evidence = inj_per_season.merge(perf_per_season, on="season_start_year", how="outer").sort_values(
        "season_start_year"
    )
    evidence["season_end"] = evidence["season_start_year"].map(lambda y: season_window(int(y))[1])
    evidence["injury_coverage_gap_days"] = (evidence["season_end"] - evidence["max_from"]).dt.days
    evidence["ended_before_snapshot"] = evidence["season_end"] <= snapshot_end - pd.Timedelta(days=COVERAGE_TOLERANCE_DAYS)
    evidence["injuries_reach_june"] = evidence["injury_coverage_gap_days"] <= COVERAGE_TOLERANCE_DAYS
    evidence["perf_records_stable"] = (
        (evidence["n_records_y"] >= MIN_REL_PERFORMANCE_RECORDS * max_perf_records)
        & (evidence["n_records_y"] >= MAX_YOY_DROP * evidence["n_records_y"].shift(1))
    )
    evidence["complete"] = (
        evidence["ended_before_snapshot"] & evidence["injuries_reach_june"] & evidence["perf_records_stable"]
    )
    evidence_out = evidence.drop(columns=["season_end"])
    evidence_out.to_csv(AUDIT_DIR / "season_completeness.csv", index=False)

    complete_years = [int(y) for y in evidence.loc[evidence["complete"], "season_start_year"]]
    first_year, last_year = min(complete_years), max(complete_years)

    # competition names drift with sponsorships; group by competition_id only
    in_scope_cov = cov[cov["season_start_year"].between(first_year, last_year)]
    seasons_per_comp = in_scope_cov.groupby("competition_id")["season_start_year"].nunique()
    weighted = (
        perf_l[(perf_l["season_league_level_level_number"] == 1)
               & perf_l["season_start_year"].between(first_year, last_year)]
        .groupby("competition_id")
        .agg(n_records=("player_id", "size"),
             n_with_minutes=("minutes_played", lambda s: int(s.notna().sum())))
        .reset_index()
    )
    weighted["minutes_nonnull_rate"] = weighted["n_with_minutes"] / weighted["n_records"]
    weighted["seasons"] = weighted["competition_id"].map(seasons_per_comp)
    comp_cov = (
        weighted.merge(comp_names, on="competition_id", how="left")
        .sort_values("minutes_nonnull_rate", ascending=False)
    )
    comp_cov.to_csv(AUDIT_DIR / "minutes_coverage_level1_overall.csv", index=False)

    qualifying_080 = comp_cov[comp_cov["minutes_nonnull_rate"] >= 0.80]
    # The 0.80 target is unattainable in this datalake (max observed coverage is
    # ~0.56; coverage varies smoothly between ~0.41 and ~0.56 and cannot
    # discriminate). Recommended scope: the five major European leagues, whose
    # coverage is in the same band as the best-covered competitions and whose
    # injury reporting is the most complete.
    recommended_ids = ["GB1", "ES1", "IT1", "FR1", "L1"]
    recommended = comp_cov[comp_cov["competition_id"].isin(recommended_ids)]
    n_players_scope = perf_l[
        perf_l["competition_id"].isin(recommended_ids)
        & perf_l["season_start_year"].between(first_year, last_year)
    ]["player_id"].nunique()

    md += [
        "## Season completeness evidence",
        "",
        _md_table(evidence_out.assign(
            min_from=evidence_out["min_from"].dt.strftime("%Y-%m-%d"),
            max_from=evidence_out["max_from"].dt.strftime("%Y-%m-%d"),
        )),
        "",
        "## Recommendation",
        "",
        f"- snapshot_end: **{snapshot_end.date()}** (max from_date in injuries; "
        f"max transfer_date {transfers['transfer_date'].max().date()}, "
        f"max market-value date {mv['date'].max().date()})",
        f"- complete seasons: start years {first_year}..{last_year} "
        f"({first_year}/{str(first_year + 1)[2:]} .. {last_year}/{str(last_year + 1)[2:]})",
        f"- level-1 competitions with >= 80% minutes coverage in those seasons: "
        f"**{len(qualifying_080)}** — the 0.80 target is unattainable "
        f"(max observed {comp_cov['minutes_nonnull_rate'].max():.3f}).",
        f"- recommended scope: {recommended_ids} "
        f"(level 1, coverage {recommended['minutes_nonnull_rate'].min():.3f}.."
        f"{recommended['minutes_nonnull_rate'].max():.3f})",
        f"- players with >= 1 performance record in the recommended scope: {n_players_scope:,}",
        "",
        "### Level-1 competition coverage (in-scope seasons)",
        "",
        _md_table(comp_cov.assign(minutes_nonnull_rate=comp_cov["minutes_nonnull_rate"].round(4))),
        "",
    ]

    out = REPORTS_DIR / "stage0_audit.md"
    out.write_text("\n".join(md), encoding="utf-8")
    log.info("wrote %s", out)
    print(f"complete seasons: {first_year}..{last_year}")
    print(f"qualifying at 0.80: {sorted(qualifying_080['competition_id'])}")
    print(f"recommended competitions: {recommended_ids}")
    print(f"players in scope: {n_players_scope:,}")
    print(f"snapshot_end: {snapshot_end.date()}")


if __name__ == "__main__":
    main()
