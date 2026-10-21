# Results and evidence

Everything in this folder is aggregate output of the code in this repository: metrics with
bootstrap confidence intervals, decision records, figures and the experiment log. Nothing here
contains raw or derived player data or per-player predictions (see [What is withheld](#what-is-withheld)).

All results use the feature store `panel_v5` (content sha256 `7bb6cc0b81eb7f340bc5f2b9af7f2b07289804264005e3ec497818765da477b7`)
and seed 20261020. Confidence intervals are 95% player-cluster bootstrap intervals (1000 resamples).

## Where each claim is backed

| Claim | Evidence |
|---|---|
| Holdout 2024/25 results (in-season and all weeks, both horizons, comparators, survival model, decision curves, conformal coverage, league and position strata) | [`final_run/holdout_results.json`](final_run/holdout_results.json); [`FINAL_RUN.md`](FINAL_RUN.md) (pre-registration); final-run entry in [`EXPERIMENTS.md`](EXPERIMENTS.md); [`figures/fig10_holdout_reliability.png`](figures/fig10_holdout_reliability.png), [`figures/fig11_holdout_decision_curves.png`](figures/fig11_holdout_decision_curves.png) |
| The holdout was scored once, with a configuration fixed beforehand | [`final_run/holdout_evaluated.json`](final_run/holdout_evaluated.json) (one-shot marker: started 10:08:42, finished 10:18:52, feature-store hash, code commit, clean tree); commit `e75b51e` adds `configs/final_run.yaml` and `src/final_run.py` (commit date 10:05:55); `spec_sha256` in the results equals the hash of that config |
| The final-run code reproduced every stored development prediction before it touched the holdout | `rehearsal_checks` in [`final_run/rehearsal_results.json`](final_run/rehearsal_results.json) |
| The deployed model rebuilt from current code reproduces every holdout prediction and both calibration maps exactly | `reproduces_final_run` in [`final_run/frozen_champion_manifest.json`](final_run/frozen_champion_manifest.json) (task 7.0 in the log) |
| Model-building decisions exp-0001 to exp-0023 (promotions, rejections, seed repeats) | [`EXPERIMENTS.md`](EXPERIMENTS.md); [`experiments/exp-NNNN/metrics.json`](experiments/) and `run.json` (config sha256, data hash, seed, commit, environment); `delta_bootstrap.npz` (bootstrap draws behind pooled decisions); [`reports/experiments_summary.md`](reports/experiments_summary.md); [`figures/fig1_trajectory.png`](figures/fig1_trajectory.png) |
| Calibration, conformal sets and decision curves in development | [`reports/conformal_exp-0018.md`](reports/conformal_exp-0018.md), [`reports/decision_curves_exp-0018.md`](reports/decision_curves_exp-0018.md), [`experiments/exp-0018/`](experiments/exp-0018/), figures 2, 3 and 8 |
| Interpretability (SHAP, dose-response, positions, calendar effect) | [`reports/stage5_interpretability.md`](reports/stage5_interpretability.md), [`experiments/exp-0018/interpret/`](experiments/exp-0018/interpret/), figures 4, 6, 7 and 9 |
| Practitioner questions in the main README | Q1 [`reports/exp-0024.md`](reports/exp-0024.md); Q2 [`reports/exp-0025.md`](reports/exp-0025.md) and the exp-0025 addendum in the log; Q3 [`reports/exp-0026.md`](reports/exp-0026.md); Q4 [`reports/exp-0027.md`](reports/exp-0027.md); Q5 exp-0029 entry in the log and the frozen-model manifest; Q6 [`reports/exp-0028.md`](reports/exp-0028.md) and figure 11. JSON for each: [`experiments/exp-0024/`](experiments/exp-0024/) … [`exp-0028/`](experiments/exp-0028/) |
| What the model is, how it was evaluated and where it should not be used | [`MODEL_CARD.md`](MODEL_CARD.md) |

## How to check

- Every run file records its config path and sha256, the feature-store hash, the seed and the code
  commit. Configs are in [`../configs/`](../configs/).
- To re-run from scratch you need your own copy of the public Transfermarkt datalake (the snapshot
  used here has injury records up to 2026-09-28; raw-file hashes are listed in `FINAL_RUN.md` §2).
  `python -m src.cli data` and `python -m src.cli features` must reproduce the feature-store hash
  above. From there, every experiment re-runs from its config
  (`python -m src.cli train|evaluate|analyse configs/exp-NNNN.yaml`). The code is deterministic, so
  the metrics should match these files exactly. This has been checked within the project (the
  rehearsal and the frozen-model rebuild above); it has not been replicated independently.
- The holdout run itself is `python -m src.final_run --expected-commit e75b51ec06126e36617cc634d687f0c914039b87`
  on a clean checkout of that commit.

## Limits of this evidence

- Commit dates are recorded by the author's machine; they are not a trusted timestamp.
- The pre-registration file was kept local, not committed, under the project's rules at the time.
  It is published here unchanged apart from the rewording noted at its top.
- The repository history was rewritten for publication after the final run. The experiment log
  cites the earlier commit ids; the mapping is below.

## Commit ids cited in the log

Every training run recorded the sha256 of its config. Each old commit id is matched to the
rewritten commit that adds the same config, and the config's sha256 at that commit equals the one
recorded at run time ([`commit_mapping.json`](commit_mapping.json)).

| experiment | commit cited in the log | commit in this history | config sha256 (recorded = found) |
|---|---|---|---|
| exp-0001 | 79887ae | dd2cca1 | e5ebf83ff58b |
| exp-0002 | 13733d7 | 03d6c17 | 315e6965e156 |
| exp-0003 | a37dc81 | 94bb558 | cbd204fdfda6 |
| exp-0004 | 6a747a3 | 7f574dd | 6a25c3cf9c02 |
| exp-0005 | a826632 | 42eab8c | 507c04481afc |
| exp-0006 | c1fa892 | a5f0984 | ddde8abca3ab |
| exp-0007 | e2db5a2 | f287ca5 | 47184cbda531 |
| exp-0008 | b42066e | c27b5dd | 040ec3fe63db |
| exp-0009 | 04ce786 | ff2452c | 41051a4dd451 |
| exp-0010 | 5e26fba | 589dcc4 | 968bebc00f9f |
| exp-0011 | 6cfd428 | 07489fd | 99beba3b59d9 |
| exp-0012 | 00c1e78 | e7c5e09 | ad8065bd0a14 |
| exp-0013 | 5bdbb4b | db820d8 | da1374ae3154 |
| exp-0014 | 5bdbb4b | db820d8 | dbf68b0843b5 |
| exp-0015 | f5f69b7 | 13567f5 | 77ed2a1c508d |
| exp-0016 | 7eee611 | f7e9469 | 96139aba18ce |
| exp-0017 | b72cdc2 | 535daec | e2ba8a9fbfc8 |
| exp-0018 | cdc4c2f | cf4a836 | 4a974b9ef870 |
| exp-0019 | 8493ee5 | e8128cc | 2043983c3177 |
| exp-0020 | b37d187 | dada562 | a405230ade62 |
| exp-0021 | 067ea9d | 1d402fb | 8430482db84d |
| exp-0022 | ebfdf46 | 64c6c42 | 4766f56e2c1d |
| exp-0023 | 59b1756 | e2ab17c | c5917daa649f |

Code-only commits cited in the log:

| commit cited in the log | commit in this history | how matched |
|---|---|---|
| f5f69b7 (arm comparison) | 13567f5 | same commit as the exp-0015 config above |
| b46cc50 (conformal sets, decision curves) | 072e284 | commit content (adds `src/conformal.py`, `src/decision_curves.py`) |
| c8de203 (interpretability, figures) | 456c328 | commit content (adds `src/interpret.py`, `src/figures.py`) |
| 90209ca (final-run harness) | e75b51e | `configs/final_run.yaml` hash equals the run's `spec_sha256` (85712891af11…); the holdout predictions are reproduced bit for bit from later code (frozen-model manifest) |

Commit ids from 75437a6 onward (Stage 7) are in this history as cited.

## What is withheld

- Raw and derived datasets (Transfermarkt-derived; not redistributed).
- Per-player predictions, calibration rows and fitted model files: weekly injury-risk scores for
  identifiable players are health-related data.
- Individual local explanations (thesis figure 5) and individual ICE curves. They are anonymised,
  but age, position, season and outcome together could identify a player. Aggregate dose-response
  curves and bands are published.
- Private working documents cited in the log: the plan with its numbered open decisions
  (`PLAN.md`), the data card (`DATA_CARD.md`) and the evaluation protocol (`EVALUATION_PROTOCOL.md`).
  The decisions that governed the results are summarised in the log's ruling and summary entries.
