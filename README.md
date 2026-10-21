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
- configs/ contains every experiment (exp-0001 … exp-0023), one YAML each.

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
