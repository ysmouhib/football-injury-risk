# FINAL_RUN.md — pre-registration of the one-shot holdout evaluation

> **Published copy** of a local working document (snapshot 2026-10-21). Changes from the original: references to the private research pre-plan and the project instruction file are reworded. This file was written at 10:06 and kept local (not committed) under the project's rules at the time; the committed evidence that the configuration was fixed before the run is commit e75b51e (`configs/final_run.yaml`, `src/final_run.py`; commit date 10:05:55), and the run's one-shot marker `results/final_run/holdout_evaluated.json` records its start at 10:08:42. No number and no entry is changed. Paths `data/experiments/…`, `data/final/…` and `reports/…` correspond to `results/experiments/`, `results/final_run/` and `results/reports/` + `results/figures/`; per-player files under `data/` are not published. Commit ids cited before commit 75437a6 refer to the history before it was rewritten for publication — mapping in [`results/README.md`](README.md#commit-ids-cited-in-the-log).

Written 2026-10-21 10:06 (+02:00), **before any model has scored a holdout row.** Nothing in this file
may be changed after the run starts; anything learned afterwards goes to `EXPERIMENTS.md` as a new
entry. Rules: `EVALUATION_PROTOCOL.md` §6, project rule 3 (one-shot holdout).

## 1. Holdout status at registration (disclosure)

The holdout is the **test player group (seeded hash, 20%), season 2024/25** (prediction Sundays
2024-07-07 … 2025-06-29). Up to this point the only computations that involved 2024/25 data were:
raw-table audits at Stage 0 (injury-record counts per season, reporting density per top-5
player-season, appearance-record completeness — no panel labels); panel row counts per season; and
the leakage tests, which build holdout *index sets* without reading labels or scoring rows. No model
has produced a prediction for a holdout row; no holdout label statistic (prevalence, PR-AUC, …) has
been computed. The final-run code was rehearsed on the main validation origin (2023/24, validation
group), where it reproduced every stored experiment prediction bit-identically
(`data/final/rehearsal/results.json`).

## 2. Code, data and command

- Code: commit **`90209ca3ba9f48bd85bf00bdd712a8fa5333a105`** (`src/final_run.py`,
  `configs/final_run.yaml`), clean tree required (checked by the script).
- Command (run exactly once):
  `python -m src.final_run --expected-commit 90209ca3ba9f48bd85bf00bdd712a8fa5333a105`
- Data: feature store `panel_v5`, content sha256
  `7bb6cc0b81eb7f340bc5f2b9af7f2b07289804264005e3ec497818765da477b7` (checked by the script);
  data config sha256 `ec6681930d65…`; label mapping `label_map_v1` sha256 `99708c894cae…`; raw
  tables as in `DATA_CARD.md` §1 (injuries `5753d0c2b7e6…`, performances `21eda6556654…`, transfers
  `5c602e631654…`, team competitions `83f06c8108e8…`, profiles `fca92a178848…`, market values
  `4f75a5ed92c7…`).
- Environment: Python 3.12.10, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, lightgbm 4.7.0,
  scikit-survival 0.28.0, pyarrow 26.0.0 (`requirements.txt`).
- Seeds: 20261020 for every model, split, calibration half and bootstrap.
- Leakage tests must pass at run time (checked by the script). A one-shot marker
  (`data/final/holdout_evaluated.json`) is written **before** the first holdout row is scored; the
  script refuses to run again once it exists.

## 3. Champion pipeline (exact)

- Experiment `configs/exp-0018.yaml` (config sha256 `4a974b9ef870…`).
- **Features** — 30 numeric: age, height_cm, prev1_apps, prev1_starts, prev1_sub_apps, prev1_squad,
  prev1_minutes_est, prev1_n_comps, prev1_no_record, prev2_minutes_est, prev2_apps, prev2_no_record,
  days_since_return, no_prior_injury, returned_within_28d, career_injuries, career_days_lost,
  injuries_365d, injuries_730d, days_lost_365d, days_lost_730d, hamstring_injuries_730d,
  groin_hip_injuries_730d, thigh_injuries_730d, knee_injuries_730d, ankle_injuries_730d,
  lower_leg_injuries_730d, muscle_injuries_730d, weeks_into_season, prev1_euro_minutes;
  4 categorical: position_group, foot, league, season_month. Definitions: `DATA_CARD.md` §5–6.
- **Model:** scikit-learn LogisticRegression(C = 1.0, class_weight = "balanced", max_iter = 2000,
  default lbfgs solver); numeric features median-imputed (training medians) with missingness
  indicators and standardised; categoricals one-hot encoded on the training vocabulary
  (unseen → ignored).
- **Training pool:** player groups train + validation, seasons 2014/15–2023/24, rows whose outcome
  window closes ≥ 7 days before the first holdout Sunday 2024-07-07 (last training Sunday: y7
  2024-06-23, y28 2024-06-02), label defined.
- **Calibration (final configuration, owner requirement):** Platt map
  p_cal = σ(a + b · logit p_raw), (a, b) fitted by unpenalised logistic regression on **calibration
  half A**, scored by the **inner model**. Inner model = the same pipeline refitted on training-pool
  rows of seasons ≤ 2022/23 whose outcome windows close ≥ 7 days before 2023-07-02 (last inner
  Sunday: y7 2023-06-18, y28 2023-05-28). Calibration rows = training-pool rows of season 2023/24
  whose outcome windows close ≥ 7 days before 2024-07-07, split by a seeded player hash (salt
  "calibration") into half A (fits the map) and half B (conformal quantiles). The map is applied to
  the deployed model trained on the whole training pool. Separate maps for y7 and y28. The map is
  **not** refitted on, or adjusted with, any holdout information.
- Horizons: y7 and y28 (time-loss injury ≥ 7 days missed, onset in (t, t + 7 d] / (t, t + 28 d];
  unknown-reason ambiguous windows masked).

## 4. Evaluation populations (owner ruling, open decision 18)

- **Primary — in-season, at-risk:** holdout rows whose prediction Sunday falls in September–April
  (panel rows are already available, i.e. not absent at t).
- **Secondary — all weeks** (mandatory sensitivity analysis, reported side by side).

## 5. Metrics (both labels, both populations; player-cluster bootstrap, 1000 resamples, 95% CI)

Champion: PR-AUC (**primary metric**), ROC-AUC (literature comparison only), sensitivity at 90%
specificity, MCC at that point, Brier, calibration slope and intercept, mean predicted vs
prevalence, reliability table (deciles); decision curve (net benefit, pt 0.01–0.30) with CIs for
the champion's NB and its margin over max(treat-all, treat-none) and the useful range; Mondrian
conformal sets at α = 0.05 / 0.10 / 0.20 — coverage overall (CI) and per class, risk tiers —
with quantiles from calibration half B scored by calibrator_A ∘ inner model; strata by league and
position (point estimates).

Comparators, trained on the same pool with their own configs: **exp-0016** (history only + Platt;
ranking identical to exp-0004) and **exp-0001** (Stage-1 logistic baseline, uncalibrated by design):
PR-AUC with CI and the paired Δ (champion − comparator) with CI; point dossiers.

Survival champion **exp-0013** (RSF, `configs/exp-0013.yaml`, sha256 `da1374ae3154…`) trained on the
same pool (90-day purge): C^td (CI), IBS (CI), 7- and 28-day risk PR-AUC and calibration; both
populations; its 28-day risk also enters the decision curve.

## 6. Reporting rules fixed now

- One evaluation; whatever it shows is reported. No re-tuning, recalibration, threshold choice or
  population change after the results are seen.
- Development reference points for drift (main origin 2023/24, rehearsal): champion PR-AUC in-season
  y7 0.0393, y28 0.1510; all weeks y7 0.0365, y28 0.1357; in-season Δ vs exp-0016 y7 +0.0006
  [−0.0013, +0.0030], y28 +0.0008 [−0.0071, +0.0089].
- The final report states (owner ruling) that in-season the context family added nothing in
  development and the champion is effectively equivalent to exp-0004; the holdout in-season Δ vs
  exp-0016 is reported as the out-of-sample check of that statement, whichever way it comes out.
- Literature reference points (EVALUATION_PROTOCOL §6.3) are indicative only: Rossi et al. 2018
  ROC-AUC ≈ 0.76 and Wang & Lee 2026 AUPRC ≈ 0.759 come from different units (single-club,
  session/match level, GPS data, different prevalence); PR-AUC is not comparable across
  prevalences.
- If the run crashes after the marker is written, the holdout counts as touched: the crash, the fix
  and any re-run are logged in full, and no partial metric is reported selectively.
