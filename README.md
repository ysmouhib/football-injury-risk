# Football Injury Risk Prediction from Open Data

Predicting time-loss injuries in professional football players using only
publicly available data: injury histories, appearances, and player info.
No club GPS, medical, or training-load data.

## What it does

- Builds a player-week panel for ~670 players per season across top-5
  European leagues (2014/15–2024/25).
- Predicts injury within the next 7 and 28 days (logistic regression with
  Platt calibration) and time-to-injury (random survival forest).
- Honest evaluation: chronological, player-grouped splits; the 2024/25
  season was held out and scored exactly once.
- Holdout, in-season: 28-day PR-AUC 0.158 (prevalence 9.1%), ROC-AUC 0.66,
  calibration slope 0.96. Beats the naive baseline; beats treat-all/treat-none
  on decision curves at 7–17% risk thresholds.
- configs/ contains every experiment (exp-0001 … exp-0029), one YAML each.

## What it does not do

- It does not beat simple baselines by much: injury history + exposure carry
  almost all the signal; tree models never beat logistic regression here.
- Short-horizon prediction is weak (7-day PR-AUC 0.046 vs 2.5% prevalence),
  and 7-day conformal coverage fails in-season.
- Part of what looks like signal across all weeks is the football calendar
  (off-season weeks are near-automatic negatives).
- No medical or performance advice. Research code.

## Layout

- src/      pipeline: data, features, train, evaluate, report
- tests/    67 tests, including data-leakage guards
- configs/  one YAML per experiment

## Usage

    python -m src.cli data
    python -m src.cli features
    python -m src.cli train configs/exp-0018.yaml
    python -m src.cli evaluate configs/exp-0018.yaml

(Equivalent make targets exist if you have make installed.)

## Data

Derived from public web records (player injury and appearance data).
Raw and derived datasets are not redistributed in this repo.

## Requirements

Python 3.10+, see requirements.txt.

## For practitioners: six questions, six measured answers

All numbers are out-of-sample: either the one-shot 2024/25 holdout (in-season weeks, 565 players,
~440 at risk per week) or the 2015/16–2023/24 development seasons under rolling-origin refits.
Every figure regenerates from configs exp-0024 to exp-0029.

**1. If I use it, how many alarms do I get — and how many are real?**

Thresholds were fixed in advance on calibration data, then measured once on the holdout.
Per 25-man squad, in-season:

| Setting | Flags / week | Precision | Injuries caught | Median lead time |
|---|---|---|---|---|
| 7-day, precision ≈ 2× base rate | ~5 | 4.6% | 3.2 of 7.9 per 90 days (41%) | 5 days |
| 28-day, precision ≈ 2× base rate | ~5 | 16.6% | 3.7 of 8.5 per 90 days (43%) | 24 days |
| 28-day, precision ≈ 1.5× base rate | ~13 | 13.1% | 6.6 of 8.5 per 90 days (77%) | 24 days |
| 28-day, precision ≈ 3× base rate | ~0.6 | 22.9% | 0.7 of 8.5 per 90 days (8%) | 23 days |

This is a screening signal, not a diagnosis: at the 2× setting you review about five players per
week and catch two injuries out of five, over three weeks ahead. Push precision higher and the
model goes quiet; push recall higher and you review half the squad every week.

**2. Does it work equally well for every player?**

Mostly yes. Holdout 28-day PR-AUC sits between 0.14 and 0.18 for every position, age band and
usage level — no dead subgroup. Three honest caveats: goalkeepers are the weakest slice (0.11,
small n); players aged ≥29 (coverage 82%), forwards (85%) and defenders (87%) get prediction sets
that under-cover against the 90% target, so treat their risk as understated; and risk is slightly
under-predicted almost everywhere, so run a per-squad intercept refit on your own recent data
before use.

**3. Is performance stable across seasons?**

Yes. Across nine development seasons, in-season lift over prevalence ranged 1.5–2.4× (7-day) and
1.5–2.2× (28-day), with no dead season. The 2024/25 holdout (1.9× / 1.7×) was a typical year, not
a lucky one. COVID seasons ranked injuries as well as normal years but calibration sagged (slopes
down to ~0.7) — in disrupted periods, recalibrate more often.

**4. Does it work in every league?**

All top-5 leagues beat their own prevalence, and the confidence intervals overlap too much to rank
them. Two caveats: LaLiga is over-predicted (its recorded injury prevalence is much lower than the
other leagues — a reporting-culture artifact), and Serie A's 28-day coverage runs below nominal.
Do a per-league intercept refit before use.

**5. Has it been validated prospectively?**

That is the one proof still outstanding, and it is now running. `python -m src.cli prospective
--date YYYY-MM-DD` writes append-only, timestamped, hash-stamped predictions (player, week, both
risk scores, model and code hashes) before outcomes are known. Reproducibility is verified:
rebuilding the panel from raw data for an arbitrary week reproduces the stored panel bit for bit,
and rebuilding the frozen champion from a clean checkout reproduces every holdout prediction and
both calibration maps exactly. Live weekly logging starts with the next data refresh; results
after one full season.

**6. How does it compare to what clubs already do, e.g. ACWR?**

The classic 7:28-day acute:chronic workload ratio needs dated minutes, which this open dataset
does not contain — it is implemented and unit-tested (`src/acwr.py`) but cannot be scored here.
The computable season-level workload ratios show no skill at all: precision below the base rate,
never beating treat-all/treat-none on decision curves, and the model beats them with confidence
intervals excluding zero. A head-to-head against real weekly ACWR (or GPS load) requires a
dated-minutes feed — the first thing to plug in below.

### What real deployment would take

- An official dated data feed (the features are just appearances, minutes and injury history —
  they port directly)
- A per-club / per-league intercept refit on recent local data
- A defined intervention protocol (what staff do when a player is flagged) — a risk score without
  an action has no value
- GDPR Art. 9 handling: injury-risk scores about identified athletes are health data
