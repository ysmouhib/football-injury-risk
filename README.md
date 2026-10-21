# Football Injury Risk Prediction from Open Data

Predicting time-loss injuries in professional football players using only
publicly available data: injury histories, appearances, and player info.
No club GPS, medical, or training-load data.

## What it does

- Builds a player-week panel (1.29 M player-weeks, 9,967 players; about 3,000–3,500
  players per season, ~2,200 at risk in a given week) across top-5 European leagues
  (2014/15–2024/25).
- Predicts injury within the next 7 and 28 days (logistic regression with
  Platt calibration) and time-to-injury (random survival forest).
- Honest evaluation: chronological, player-grouped splits; the 2024/25
  season was held out and scored exactly once.
- Holdout, in-season: 28-day PR-AUC 0.158 (prevalence 9.1%), ROC-AUC 0.66,
  calibration slope 0.96. Beats the first-stage logistic baseline (exposure +
  demographics) by +0.044 PR-AUC [0.023, 0.068]; beats treat-all/treat-none
  on decision curves at 7–17% risk thresholds.
  Evidence: [results/](results/README.md).
- configs/ contains every experiment (exp-0001 … exp-0029), one YAML each.

## What it does not do

- Its gain over simple models is modest: in-season it is no better than injury
  history + exposure alone (28-day PR-AUC difference −0.003 [−0.011, +0.004]);
  tree models never beat logistic regression here.
- Short-horizon prediction is weak (7-day PR-AUC 0.046 vs 2.5% prevalence),
  and 7-day conformal coverage fails in-season (86% vs the 90% target).
- Part of what looks like signal across all weeks is the football calendar
  (off-season weeks are near-automatic negatives).
- No medical or performance advice. Research code.

## Layout

- src/      pipeline: data, features, train, evaluate, report
- tests/    78 tests, including data-leakage guards
- configs/  one YAML per experiment
- results/  published evidence: metrics, figures, experiment log, pre-registration, model card

## Usage

    python -m src.cli data
    python -m src.cli features
    python -m src.cli train configs/exp-0018.yaml
    python -m src.cli evaluate configs/exp-0018.yaml
    python -m src.cli freeze                          # rebuild and verify the deployed model
    python -m src.cli analyse configs/exp-0024.yaml   # practitioner analyses (exp-0024 … exp-0028)
    python -m src.cli prospective --date YYYY-MM-DD   # append-only weekly risk log

(Equivalent make targets exist for the first four if you have make installed.)

## Data

Derived from public web records (player injury and appearance data).
Raw and derived datasets are not redistributed in this repo.

## Requirements

Python 3.12 (tested with 3.12.10; the pinned numpy needs 3.12 or later), see requirements.txt.

## For practitioners: six questions, six measured answers

All numbers are out-of-sample: either the one-shot 2024/25 holdout (in-season weeks, 565 players,
~440 at risk per week) or the 2015/16–2023/24 development seasons under rolling-origin refits.
Each answer's evidence is listed in [results/](results/README.md); the analyses re-run from
configs exp-0024 to exp-0029, given the data (which is not redistributed).

**1. If I use it, how many alarms do I get — and how many are real?**

Thresholds were fixed in advance on calibration data (2023/24), then measured once on the
holdout. Per 25-man squad, in-season ([exp-0024](results/reports/exp-0024.md)):

| Setting | Flags / week | Precision | Injuries caught | Median days from first flag to injury |
|---|---|---|---|---|
| 7-day, target 2× base rate | ~5 | 4.6% | 3.2 of 7.9 per 90 days (40%) | 5 |
| 28-day, target 2× base rate | ~5 | 16.6% | 3.7 of 8.5 per 90 days (43%) | 24 |
| 28-day, target 1.5× base rate | ~13 | 13.1% | 6.6 of 8.5 per 90 days (77%) | 24 |
| 28-day, target 3× base rate | ~0.6 | 22.9% | 0.7 of 8.5 per 90 days (8%) | 23 |

The targets were multiples of the 2023/24 base rate. On the holdout, where injuries were more
frequent, the 1.5×, 2× and 3× settings delivered 1.4×, 1.8× and 2.5–2.8× the holdout base rate.

This is a screening signal, not a diagnosis: at the 2× setting you review about five players per
week and catch about two injuries in five. Push precision higher and the model goes quiet; push
recall higher and you review half the squad every week. The last column is not early warning:
scores change slowly, so for 93–97% of injuries caught at 28 days the player was already flagged
in the first week whose window covered the injury, and the number is capped by the horizon (7 or
28 days).

**2. Does it work equally well for every player?**

Mostly yes. Holdout 28-day PR-AUC sits between 0.14 and 0.18 for every outfield position, age band
and half of last-season minutes, and every subgroup's confidence interval lies above its base rate:
no dead subgroup ([exp-0025](results/reports/exp-0025.md)). The open data has no recent workload,
so "usage" here means minutes in the previous season.

Caveats:
- **Goalkeepers are the weakest slice** (0.11; 59 players).
- **The 90% risk tiers are not guaranteed per subgroup.** For players aged ≥29 (82% coverage) and
  forwards (85%), the shortfall is extra false alarms: uninjured weeks placed in the high-risk tier.
  For goalkeepers and players aged ≤23 it is injuries the tiers miss: only 55% and 83% of their
  injury weeks are covered.
- **Risk is under-predicted in every position, age band and minutes half,** by about a fifth
  overall (mean predicted 7.4% vs 9.1% observed at 28 days), because 2024/25 had more injuries than
  the training seasons. A local level adjustment (intercept refit on recent data) is the obvious
  remedy, but it is untested here. The intercept-refit variants tried in development (exp-0019 to
  exp-0023) did not pass the promotion rule, and one squad gives very few injuries to fit on.

**3. Is performance stable across seasons?**

For ranking, yes ([exp-0026](results/reports/exp-0026.md)). Across nine development seasons,
in-season lift over prevalence ranged 1.5–2.5× (7-day) and 1.5–2.2× (28-day), with no dead
season. The 2024/25 holdout (1.9× / 1.7×) was a typical year, not a lucky one.

COVID seasons ranked injuries as well as normal years, but calibration broke. Across all weeks of
2019/20 the slopes fell to ~0.7 (in-season they stayed ~0.9). The yearly recalibration learned on
that season then left 2020/21 under-confident (slope 1.27 at 7 days). Recalibrating on a disrupted
season carries the disruption forward, so check calibration before reusing a map learned on an
abnormal season. Calibration also drifts in normal years: at 28 days the in-season slope's
confidence interval excludes 1 in 2016/17, 2018/19 and 2023/24.

**4. Does it work in every league?**

All top-5 leagues beat their own prevalence, and the confidence intervals overlap too much to rank
them ([exp-0027](results/reports/exp-0027.md)). Two caveats:
- **LaLiga is over-predicted, with scores too uniform** (28-day slope 1.46 [1.05, 2.03]). Its
  recorded injury rate is 5.7% against 8–11% in the other leagues; lower public reporting is one
  possible explanation, not tested.
- **Serie A's 28-day coverage runs below nominal** (83%).

The model already includes the league; a further per-league adjustment was not tested.

**5. Has it been validated prospectively?**

No. That is the one proof still outstanding. The logging harness is built and tested but not yet
running. `python -m src.cli prospective --date YYYY-MM-DD` scores every active player and appends
timestamped, hash-stamped predictions (player, week, both risk scores, model and code hashes).

It needs a data snapshot covering the date. The current snapshot ends 2026-09-28 and later dates
are refused, so live weekly logging starts with the next data refresh. The only run so far replays
19 Oct 2025 (2,202 players) to test the path. A prospective evaluation would be pre-registered
separately after one full season.

Reproducibility checks:
- Rebuilding the panel from raw data for a historical week (2023-10-15, 2,176 players) reproduces
  every stored feature value.
- Rebuilding the frozen champion from a clean checkout reproduces every holdout prediction and both
  calibration maps exactly
  ([manifest](results/final_run/frozen_champion_manifest.json)).

**6. How does it compare to what clubs already do, e.g. ACWR?**

The classic 7:28-day acute:chronic workload ratio needs dated minutes, which this open dataset
does not contain. It is implemented and unit-tested (`src/acwr.py`) but cannot be scored here
([exp-0028](results/reports/exp-0028.md)).

The computable season-level workload ratios show no skill at all:
- **Precision is below the base rate** (7.5% vs 9.1% at 28 days).
- **They never beat treat-all/treat-none on decision curves.**
- **The model beats them** by +0.074 PR-AUC at 28 days [+0.050, +0.100].

A head-to-head against real weekly ACWR (or GPS load) requires a dated-minutes feed, the first
thing to plug in below.

### What real deployment would take

- An official dated data feed (the features are appearances, minutes, injury history and basic
  profile data — they port directly)
- A level (intercept) adjustment on recent local data — likely needed (questions 2 and 4) but untested
  here
- A defined intervention protocol (what staff do when a player is flagged) — a risk score without
  an action has no value
- GDPR Art. 9 handling: injury-risk scores about identified athletes are health data
