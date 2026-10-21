# MODEL_CARD.md — weekly time-loss injury risk, top-5 European leagues (exp-0018)

> **Published copy** of a local working document (snapshot 2026-10-21). Changes from the original: references to the private research pre-plan and the project instruction file are reworded. No number and no entry is changed. Paths `data/experiments/…`, `data/final/…` and `reports/…` correspond to `results/experiments/`, `results/final_run/` and `results/reports/` + `results/figures/`; per-player files under `data/` are not published. Commit ids cited before commit 75437a6 refer to the history before it was rewritten for publication — mapping in [`results/README.md`](README.md#commit-ids-cited-in-the-log).

## Model details

- **What it predicts:** for a player registered with a Premier League, LaLiga, Serie A, Ligue 1 or
  Bundesliga senior team and not currently absent, the probability that a **time-loss injury of ≥ 7
  days** begins within the next **7 days** (y7) or **28 days** (y28), issued every Sunday.
- **Model:** class-weighted logistic regression (scikit-learn, C = 1, balanced weights) on 34
  features, followed by a **Platt calibration map** learned on held-out scores of half of the
  previous season. One model per horizon. Experiment `exp-0018`; final-run code commit `90209ca`.
- **Companion model:** a Random Survival Forest (`exp-0013`) gives a 1–90-day risk curve from the
  same features (injury history + exposure, without the context family); it ranks players almost
  identically and is calibrated in level without a post-hoc step.
- **Developed:** October 2026, thesis research on open-data injury prediction. Selection by an
  evidence-gated champion/challenger protocol (`EXPERIMENTS.md`, 23 logged experiments).

## Inputs

All inputs are known at prediction time t (enforced by leakage tests):
- **Injury history** (from Transfermarkt profiles): days since return from the last injury, return
  within 28 days, injuries and days lost in the last 365 / 730 days and career, prior injuries in the
  last 730 days by body part (hamstring, groin/hip, thigh, knee, ankle, lower leg) and muscle.
- **Exposure, lagged one and two completed seasons:** appearances, starts, substitute appearances,
  squad selections, estimated minutes (from appearance counts — the datalake's minutes field is
  minutes per goal), number of competitions, UEFA club minutes.
- **Demographics and context:** age, height, foot, position group (2025 profile snapshot), league,
  month, weeks into the season.

No within-season training or match load, GPS, wellness or medical data are used.

## Intended use

- Research on the predictability of injuries from public data, and as a **screening aid** that
  ranks player-weeks for closer attention by qualified staff, alongside their own monitoring.
- Population: male senior players in the top-5 European leagues, seasons similar to 2014/15–2024/25.

## Not intended for

- Decisions about an individual player's selection, contract, transfer valuation or medical
  clearance on the basis of the score alone.
- Any league, competition level, women's football, youth football or a different injury definition.
- Inferring causes: feature effects are associational (see "Factors").

## Evaluation — one-shot holdout 2024/25 (673 players never used in development)

Primary population: in-season player-weeks (prediction Sunday September–April). 95% player-cluster
bootstrap CIs.

| | y7 | y28 |
|---|---|---|
| Injury prevalence | 2.45% | 9.13% |
| PR-AUC (primary metric) | **0.046** [0.038, 0.058] | **0.158** [0.134, 0.188] |
| ROC-AUC | 0.657 [0.626, 0.684] | 0.661 [0.630, 0.690] |
| Sensitivity at 90% specificity | 0.22 [0.17, 0.27] | 0.23 [0.18, 0.28] |
| Brier | 0.0237 | 0.0810 |
| Calibration slope / intercept | 0.94 / +0.20 | 0.96 / +0.23 |
| Mean predicted risk | 2.01% | 7.41% |
| Net benefit > both defaults for thresholds | 2–4% | 7–17% |
| vs Stage-1 baseline (Δ PR-AUC) | +0.015 [+0.007, +0.025] | +0.044 [+0.023, +0.068] |
| vs history-only model (Δ PR-AUC) | −0.002 [−0.004, +0.002] | −0.003 [−0.011, +0.004] |

All weeks (secondary): PR-AUC y7 0.042 (prevalence 1.91%), y28 0.145 (7.25%); ROC-AUC 0.70 / 0.70.

**How to read this.** The model is roughly 1.7–1.9× better than random at concentrating injuries in
its top-ranked weeks, and ranks an injured week above an uninjured one about two times in three.
At 90% specificity it catches about one injury in four or five. That is useful for screening and
far from a reliable individual forecast. Across the whole season it does better than a
history-only model **only because it knows the football calendar** (fewer injuries over the summer
break); within the playing season it is no better than injury history plus lagged exposure.

## Operating points (Stage 7, exp-0024; thresholds fixed on 2023/24 before the holdout was read)

In-season holdout, per 25-player squad. Precision is the share of flags followed by an injury;
"caught" means the injury fell in the horizon of at least one flagged week.

| threshold (calibrated risk) | flags per week per 25 players | precision | injuries caught | caught / missed per squad per 90 days |
|---|---|---|---|---|
| 7-day ≥ 0.018 | 12.9 | 0.035 | 74% | 5.8 / 2.1 |
| 7-day ≥ 0.027 | 5.4 | 0.046 | 40% | 3.2 / 4.7 |
| 7-day ≥ 0.051 | 0.6 | 0.068 | 7% | 0.5 / 7.3 |
| 28-day ≥ 0.067 | 12.7 | 0.131 | 77% | 6.6 / 2.0 |
| 28-day ≥ 0.101 | 5.0 | 0.166 | 43% | 3.7 / 4.9 |
| 28-day ≥ 0.180 | 0.6 | 0.229 | 8% | 0.7 / 7.8 |

Scores change slowly from week to week, so a flagged player usually stays flagged. The "lead time"
before an injury is set by the horizon, not by an early-warning ability.

**Workload-ratio rules.** The 7:28-day acute:chronic workload ratio used in practice cannot be computed
from this data (no dated minutes), so the model has **not** been compared with it. A season-scale
analogue (last-season minutes against earlier seasons, flag > 1.5) has no skill on the holdout
(PR-AUC at or below prevalence); the model beats it by +0.074 PR-AUC at 28 days (exp-0028).

## Calibration and uncertainty

- Calibration slope ≈ 1 on the holdout, but the level is under-predicted (intercept +0.15 to +0.23):
  2024/25 had more injuries than the training seasons. Expect the absolute risk to lag a season
  whose injury rate changes; ranking is more stable than level.
- Class-conditional conformal sets (α = 0.10) give risk tiers. 28 days, in-season holdout: low
  tier 22% of weeks with 3.7% injured, high tier 13% with 18.0%. Coverage held for all weeks
  (0.90–0.91) but fell below nominal in-season (0.86 at 7 days, 0.88 at 28 days): treat the tiers as
  approximate under season-to-season shift.
- Tier coverage is guaranteed per class, not per subgroup. At 28 days (α = 0.10) it was 0.82 for
  players aged 29 and over, 0.83 in Serie A and 0.85 for forwards, against 0.95–0.97 for players aged 23
  and under and for goalkeepers (exp-0025, exp-0027).
- Season to season (development 2015/16–2023/24, exp-0026), ranking was stable (lift 1.5–2.5).
  Calibration drifted, most of all around COVID-19: 2019/20 all-weeks slopes were 0.66–0.69, and
  the calibration map learned on that season left 2020/21 under-confident.

## Factors and subgroups

- Attribution (exact SHAP, 2023/24 validation): 70% non-modifiable factors (calendar, injury
  history, age, position, league), 22% lagged exposure/load, 8% return-to-play timing. Risk rises
  with last season's minutes (no U/J-shaped chronic-load curve), falls with time since return from
  injury and, given injury history, with age.
- Holdout in-season y28 PR-AUC by league 0.147–0.174 and by position 0.112 (goalkeepers) to 0.175
  (attack). Goalkeepers and LaLiga are the least well calibrated (slopes 1.32 and 1.46: predictions
  too compressed).
- Stage 7 (exp-0025, exp-0027, 95% CIs): discrimination holds in every position, age band (≤ 23,
  24–28, ≥ 29), last-season-minutes half and league; all PR-AUC CIs lie above prevalence. Risk is
  under-predicted in every subgroup and in every league except **LaLiga, where it is over-predicted**
  (mean predicted 7.6% vs 5.7% observed at 28 days; slope 1.46 [1.05, 2.03]). Read LaLiga scores as
  too high and too uniform.

## Data and known biases

Transfermarkt datalake (public), 1.29 M player-weeks, 9,967 players. Injury reporting is
public-record based: about 18% of injuries have no disclosed type (handled by masking), reporting
density rose until about 2016, and 12.6% of panel players have no injury record at all (never injured
and unscraped cannot be distinguished). National-team load is missing. Profile attributes are a 2025
snapshot. Details: `DATA_CARD.md`.

## Ethical considerations

The data are public but concern identifiable people's health. Scores should not be published per
player, used to stigmatise players, or used without the player's medical staff. All local
explanations in the thesis are anonymised.

## Reproducibility

Feature store `panel_v5` sha256 `7bb6cc0b81eb…`; configs `configs/exp-0018.yaml`,
`configs/final_run.yaml`; seeds 20261020; pinned `requirements.txt`. `python -m src.cli data`,
`features`, `train`, `evaluate`; the final run is `src.final_run` (pre-registered in `FINAL_RUN.md`,
executed once on 2026-10-21).
The deployed objects are re-materialised by `python -m src.cli freeze` and must reproduce every
final-run holdout prediction bit for bit (model hash 688b63775081…). Weekly scores are logged
append-only by `python -m src.cli prospective --date YYYY-MM-DD`, with model and code hashes on
every row. That needs a datalake snapshot covering the date; the current snapshot ends 2026-09-28.
