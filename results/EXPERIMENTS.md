# EXPERIMENTS.md — append-only experiment log

> **Published copy** of a local working document (snapshot 2026-10-21). Changes from the original: references to the private research pre-plan and the project instruction file are reworded; the log is append-only and continues locally. No number and no entry is changed. Paths `data/experiments/…`, `data/final/…` and `reports/…` correspond to `results/experiments/`, `results/final_run/` and `results/reports/` + `results/figures/`; per-player files under `data/` are not published. Commit ids cited before commit 75437a6 refer to the history before it was rewritten for publication — mapping in [`results/README.md`](README.md#commit-ids-cited-in-the-log).

> Template and promotion rule: `EVALUATION_PROTOCOL.md` §3–4. Entries are never edited or deleted;
> corrections are new entries that cite the one they correct.

---

### stage-0 — Foundations (data, labels, feature store v0, split harness)
- Date / Stage: 2026-10-20 / Stage 0 (not an experiment loop: no model trained, no verdict)
- Data version: `panel_v0` sha256 `0c5a1a7ea076aa8863b964bcc461058f1f462d4378527241ab989ef65160db74`
  (1,293,644 player-weeks, 9,967 players, seasons 2014/15–2024/25, top-5 leagues)
- Configs: `configs/data.yaml` (seed 20261020), `configs/label_mapping.yaml` (`label_map_v1`)
- Splits: train/validation/test player groups 60/20/20; validation origins 2021/22, 2022/23,
  2023/24 (main 2023/24); holdout 2024/25 — untouched (only row/player counts computed).
- Reproducibility: wiping `data/` and rerunning `src.cli data` + `src.cli features` reproduces the
  identical panel hash. Leakage + unit tests: 36/36 green; each mandatory leakage test was checked to
  fail on an injected leak (feature timestamp after t, onset on day t, row-level random roles,
  no purge, label copied into a feature).
- Development-row prevalence: y7 1.65%, y28 6.25% (masked 0.10% / 0.40%).
- Findings that changed the plan (PLAN.md §7):
  1. A1 — `minutes_played` is minutes *per goal* for outfield players (null on every row with
     goals = 0). The earlier "~48% minutes coverage" reading and the league-scope rationale built on
     it were artefacts. Exposure is now estimated from appearance counts (R² 0.996 out-of-time).
  2. A2 — the planned weekly allocation of season totals leaked future information (season totals and
     future injury weeks); replaced by last-two-completed-season exposure. Within-season load is not
     observable from this datalake; Stage 2 load families need re-specification (task 2.0).
  3. A3/A3b — population is time-t registration (transfer history) + availability + activity in the
     last two completed seasons. The activity requirement was added after deceased players appeared as
     registered in a draft panel; inactive rows had y28 0.06–0.38% vs 6.3%.
  4. A4 — seasons 2014/15–2024/25 from scrape-completeness and injury-reporting-density rules; the
     2014/15 boundary is threshold-sensitive (density 0.716 vs threshold 0.669).
  5. A6 — unknown-reason injuries (18% of all records) masked rather than labelled negative.
- Open for Stage 1: decision 11 (main-origin vs pooled-origin promotion evaluation, by power
  calculation on the baseline only).

---

### exp-0001 — Class-weighted logistic regression, v0 features
- Date / Stage: 2026-10-20 / Stage 1
- Hypothesis (one sentence): A class-weighted logistic regression on exposure and demographic
  features ranks next-7/28-day time-loss injuries above the prevalence baseline.
- Change vs champion (one controlled change; config diff link): none — first champion of record
  (research pre-plan §2); `configs/exp-0001.yaml` (sha256 e5ebf83ff58b), commit 79887ae (clean)
- Data version (hash) / Split origin(s) / Seed: `panel_v0` 0c5a1a7ea076… / validation origins
  2021/22, 2022/23, 2023/24 (main 2023/24) / 20261020
- Primary metric (main origin 2023/24; player-cluster bootstrap, 1000 resamples):
  champion = n/a | challenger = y7 **0.0207** [0.0175, 0.0264], y28 **0.0778** [0.0658, 0.0950]
  | Δ = n/a | prevalence y7 0.0161, y28 0.0624 (lift 1.29× / 1.25×)
- Secondary metrics (main origin):

  | label | sens@90spec | MCC@90spec | Brier | cal. slope | cal. intercept | mean predicted |
  |---|---|---|---|---|---|---|
  | y7 | 0.142 | 0.017 | 0.2475 | 1.079 | −4.124 | 0.490 |
  | y28 | 0.147 | 0.038 | 0.2478 | 1.037 | −2.726 | 0.492 |

  Per-origin PR-AUC (2021/22, 2022/23, 2023/24): y7 0.0274 / 0.0211 / 0.0207; y28 0.0899 / 0.0788 /
  0.0778. Full dossier, strata, reliability and decision-curve tables: `reports/experiments/exp-0001.md`.
- Calibration slope: champion = n/a | challenger = y7 1.079, y28 1.037 (per-origin range 0.73–1.32)
- Verdict: **CROWNED** — first champion of record (no promotion test applies)
- Notes & surprises:
  - Weak but real signal from lagged exposure + demographics alone; this is the floor the tree line
    must beat.
  - Class weighting inflates probabilities (mean predicted ≈ 0.49 vs prevalence 0.016–0.062,
    intercepts −2.7/−4.1, Brier ≈ 0.248) while slopes stay near 1 — ranking is usable, absolute risk is
    not until Stage 4 recalibration. Decision curves on these raw probabilities are not interpretable.
  - Calibration slope varies strongly across origins (0.73–1.32): season-to-season drift.
  - Open decision 11 resolved by its pre-specified rule on this run: bootstrap SE of PR-AUC, main
    origin vs mean of origins — y7 0.0023 vs 0.0031 (ratio 1.33), y28 0.0076 vs 0.0072 (0.95). Pooling
    does not reduce variance (the same validation players recur across origins and the 2021/22 origin
    is more variable), so promotion decisions use the **main origin 2023/24**; other origins are
    reported as spread.
  - Reproducibility: predictions from a dry run on the uncommitted tree and the official run on
    commit 79887ae are bit-identical.

---

### exp-0002 — Class-weighted LightGBM, v0 features
- Date / Stage: 2026-10-20 / Stage 1
- Hypothesis (one sentence): A class-weighted gradient-boosted tree ensemble on the same v0 features
  improves validation PR-AUC over the logistic champion at both horizons by capturing non-linear age
  and exposure effects and their interactions.
- Change vs champion (one controlled change; config diff link): model block only, LR → LightGBM with
  fixed pre-declared hyperparameters (400 trees, lr 0.03, 31 leaves, min_child_samples 200,
  subsample/colsample 0.8, λ 1, balanced class weights, 4 threads, deterministic) —
  `configs/exp-0002.yaml` vs `configs/exp-0001.yaml`; commit 13733d7 (clean)
- Data version (hash) / Split origin(s) / Seed: `panel_v0` 0c5a1a7ea076… / main origin 2023/24 /
  20261020
- Primary metric (main origin): y7 champion = 0.0207 | challenger = 0.0225 | Δ = +0.0018 |
  bootstrap 95% CI = [−0.0027, 0.0054]; y28 champion = 0.0778 | challenger = 0.0803 | Δ = +0.0025 |
  CI = [−0.0112, 0.0159]
- Secondary metrics (main origin):

  | label | sens@90spec | MCC@90spec | Brier | cal. slope | cal. intercept |
  |---|---|---|---|---|---|
  | y7 | 0.164 (champ 0.142) | 0.027 (0.017) | 0.2220 (0.2475) | 0.846 | −4.017 |
  | y28 | 0.141 (champ 0.147) | 0.033 (0.038) | 0.2267 (0.2478) | 0.806 | −2.627 |

  Origin spread (2021/22, 2022/23, 2023/24): y7 0.0214 / 0.0211 / 0.0225 (champion 0.0274 / 0.0211 /
  0.0207); y28 0.0766 / 0.0755 / 0.0803 (champion 0.0899 / 0.0788 / 0.0778).
- Calibration slope: champion = y7 1.079, y28 1.037 | challenger = y7 0.846, y28 0.806
  (regression 0.075 and 0.157 > 0.05)
- Verdict: **INCONCLUSIVE** (condition 1 holds for both labels, condition 2 fails for both;
  condition 3 also fails) → one seed repeat allowed: exp-0003.
- Notes & surprises: the gain is confined to the main origin; on 2021/22 LightGBM is clearly worse
  and its mean over origins is below the champion's at both horizons. 2021/22 slopes 0.36–0.39:
  the boosted model is overconfident out of time on that season.

---

### exp-0003 — Class-weighted LightGBM, v0 features (seed repeat of exp-0002)
- Date / Stage: 2026-10-20 / Stage 1
- Hypothesis (one sentence): as exp-0002 (protocol repeat after INCONCLUSIVE).
- Change vs champion (one controlled change; config diff link): as exp-0002; vs exp-0002 only the
  seed changes (20261020 → 20261021; drives LightGBM bagging and the bootstrap) —
  `configs/exp-0003.yaml`; commit of config 3c (see `git log configs/exp-0003.yaml`), tree clean
- Data version (hash) / Split origin(s) / Seed: `panel_v0` 0c5a1a7ea076… / main origin 2023/24 /
  20261021
- Primary metric (main origin): y7 champion = 0.0207 | challenger = 0.0214 | Δ = +0.0007 |
  CI = [−0.0041, 0.0045]; y28 champion = 0.0778 | challenger = 0.0808 | Δ = +0.0030 |
  CI = [−0.0112, 0.0182]
- Secondary metrics: sens@90spec y7 0.167, y28 0.138; MCC y7 0.028, y28 0.031; Brier y7 0.2219,
  y28 0.2269. Origin spread y7 0.0231 / 0.0213 / 0.0214; y28 0.0767 / 0.0789 / 0.0808.
- Calibration slope: champion = y7 1.079, y28 1.037 | challenger = y7 0.775, y28 0.788
- Verdict: **INCONCLUSIVE again → loop exp-0002/0003 decided: REJECT** (the one permitted repeat is
  used; conditions 2 and 3 fail at both horizons). Champion remains **exp-0001**.
- Notes & surprises: seed-to-seed change in y7 PR-AUC (0.0225 → 0.0214) is as large as the Δ
  itself — on v0 features the LightGBM advantage, if any, is within seed noise. Consistent with
  the research pre-plan's warning that LR can match ML on weak tabular feature sets; the tree model must be
  re-challenged once richer feature families exist (PLAN task 2.7).

---

### stage-1 summary
- Champion: **exp-0001** (class-weighted logistic regression, v0 features). Cumulative Δ PR-AUC vs
  the Stage-1 baseline: 0 (y7 0.0207, y28 0.0778 on the main origin 2023/24).
- Challengers: 1 loop (exp-0002 + seed repeat exp-0003) → REJECT. Non-promotion streak: 1.
- Harness frozen: metric definitions (`EVALUATION_PROTOCOL.md` §3a), main-origin aggregation
  (decision 11), 1000-resample paired player-cluster bootstrap, intersection rule over y7 and y28.

- Correction (2026-10-20) to the exp-0003 entry: its config was committed in a37dc81; training and
  evaluation ran on that clean commit.

---

### exp-0004 — Champion + injury-history family
- Date / Stage: 2026-10-20 / Stage 2 (loop 2.1)
- Hypothesis (one sentence): Adding injury-history features to the logistic champion improves
  validation PR-AUC at both horizons.
- Change vs champion (one controlled change; config diff link): feature set v0 → v0+history (16
  features, PLAN A8 / DATA_CARD §6) — `configs/exp-0004.yaml` vs `configs/exp-0001.yaml`; model,
  labels, seed unchanged; config commit 6a747a3 (clean)
- Data version (hash) / Split origin(s) / Seed: `panel_v1` f015e0aad741… (rows/labels identical to
  `panel_v0`, enforced by the comparison) / main origin 2023/24 / 20261020
- Primary metric (main origin): y7 champion = 0.0207 | challenger = **0.0307** | Δ = +0.0100 |
  bootstrap 95% CI = [0.0049, 0.0152]; y28 champion = 0.0778 | challenger = **0.1137** |
  Δ = +0.0360 | CI = [0.0196, 0.0521]
- Secondary metrics (main origin; champion in brackets):

  | label | sens@90spec | MCC@90spec | Brier | cal. intercept |
  |---|---|---|---|---|
  | y7 | 0.242 (0.142) | 0.059 (0.017) | 0.2325 (0.2475) | −4.134 |
  | y28 | 0.243 (0.147) | 0.111 (0.038) | 0.2332 (0.2478) | −2.740 |

  Origin spread: y7 0.0360 / 0.0303 / 0.0307 (champion 0.0274 / 0.0211 / 0.0207); y28 0.1264 /
  0.1075 / 0.1137 (champion 0.0899 / 0.0788 / 0.0778) — the gain holds at every origin.
- Calibration slope: champion = y7 1.079, y28 1.037 | challenger = y7 0.996, y28 0.995
- Verdict: **PROMOTE** (conditions 1–3 hold for y7 and y28). New champion **exp-0004**; previous
  champion config `configs/exp-0001.yaml` archived in place (unchanged, still the Stage-1 baseline).
- Notes & surprises: largest single gain so far (y28 PR-AUC ×1.46); slopes also moved closer to 1
  and became stable across origins (0.93–1.00, was 0.73–1.32). Probabilities remain inflated by class
  weighting (intercepts unchanged) — Stage 4. Stage-2 non-promotion streak reset to 0.

---

### exp-0005 — LightGBM on v0+history (model-family re-challenge, PLAN 2.7)
- Date / Stage: 2026-10-20 / Stage 2
- Hypothesis (one sentence): With injury-history features available, a class-weighted LightGBM
  (same fixed settings as exp-0002) improves validation PR-AUC over the logistic champion at both
  horizons.
- Change vs champion (one controlled change; config diff link): model block LR → LightGBM, settings
  identical to exp-0002 (not retuned) — `configs/exp-0005.yaml` vs `configs/exp-0004.yaml`; config
  commit a826632 (clean)
- Data version (hash) / Split origin(s) / Seed: `panel_v1` f015e0aad741… / main origin 2023/24 /
  20261020
- Primary metric (main origin): y7 champion = 0.0307 | challenger = 0.0309 | Δ = +0.0002 |
  CI = [−0.0018, 0.0023]; y28 champion = 0.1137 | challenger = 0.1133 | Δ = −0.0004 |
  CI = [−0.0071, 0.0059]
- Secondary metrics (main origin): sens@90spec y7 0.231 (champion 0.242), y28 0.237 (0.243); MCC y7
  0.054 (0.059), y28 0.106 (0.111); Brier y7 0.2200 (0.2325), y28 0.2231 (0.2332). Origin mean
  PR-AUC: y7 0.0304 (champion 0.0324), y28 0.1096 (0.1159).
- Calibration slope: champion = y7 0.996, y28 0.995 | challenger = y7 1.057, y28 1.040 (y7
  regression 0.053 > 0.05)
- Verdict: **REJECT** (condition 1 fails for y28; condition 3 fails for y7). No repeat: the
  INCONCLUSIVE branch needs condition 1 at both horizons. Champion remains **exp-0004**.
- Notes & surprises: second loss of the tree model family (exp-0002/0003 on v0, now on v0+history).
  With season-level exposure and history counts, the signal appears close to additive on the logit
  scale; the boosted trees add nothing measurable. Thesis-relevant: the research pre-plan expectation that
  tree ensembles dominate is not borne out on this open-data feature set so far. Stage-2
  non-promotion streak: 1.

---

### exp-0006 — Champion + season-load derivative family
- Date / Stage: 2026-10-20 / Stage 2 (loop 2.2)
- Hypothesis (one sentence): Adding season-level load derivatives to the champion improves
  validation PR-AUC at both horizons.
- Change vs champion (one controlled change; config diff link): feature set v0+history →
  v0+history+load (6 features, PLAN A8 incl. the 90-minute denominator rule recorded before the run)
  — `configs/exp-0006.yaml` vs `configs/exp-0004.yaml`; commit c1fa892 (clean)
- Data version (hash) / Split origin(s) / Seed: `panel_v2` e860c93b1b7e… (rows/labels identical to
  v1) / main origin 2023/24 / 20261020
- Primary metric (main origin): y7 champion = 0.0307 | challenger = 0.0310 | Δ = +0.0003 |
  CI = [−0.0004, 0.0011]; y28 champion = 0.1137 | challenger = 0.1161 | Δ = +0.0023 |
  CI = [−0.0012, 0.0066]
- Secondary metrics (main origin): sens@90spec y7 0.239 (champion 0.242), y28 0.254 (0.243); MCC
  y7 0.058 (0.059), y28 0.119 (0.111); Brier y7 0.2328 (0.2325), y28 0.2332 (0.2332). Origin spread
  y7 0.0356 / 0.0340 / 0.0310, y28 0.1242 / 0.1108 / 0.1161 (champion 0.0360 / 0.0303 / 0.0307 and
  0.1264 / 0.1075 / 0.1137).
- Calibration slope: champion = y7 0.996, y28 0.995 | challenger = y7 0.991, y28 0.983
- Verdict: **INCONCLUSIVE** (condition 1 holds for both, condition 2 fails for both; condition 3
  holds) → repeat exp-0007.
- Notes & surprises: small, consistent-in-sign gain at y28 (also on the origin mean, 0.1170 vs
  0.1159), but within noise; season-level load carries little information beyond history + lagged
  exposure.

---

### exp-0007 — Champion + season-load derivative family (seed repeat of exp-0006)
- Date / Stage: 2026-10-20 / Stage 2
- Hypothesis (one sentence): as exp-0006 (protocol repeat after INCONCLUSIVE).
- Change vs champion (one controlled change; config diff link): as exp-0006; vs exp-0006 only the
  seed (20261020 → 20261021) — `configs/exp-0007.yaml`; commit e2db5a2 (clean)
- Data version (hash) / Split origin(s) / Seed: `panel_v2` e860c93b1b7e… / main origin 2023/24 /
  20261021
- Primary metric (main origin): y7 Δ = +0.0003, CI = [−0.0004, 0.0010]; y28 Δ = +0.0023,
  CI = [−0.0011, 0.0060]
- Secondary metrics / calibration: identical to exp-0006 (model predictions bit-identical).
- Verdict: **INCONCLUSIVE → loop exp-0006/0007 decided: REJECT**. Champion remains **exp-0004**.
  Stage-2 non-promotion streak: 2.
- Notes & surprises: the logistic learner is deterministic, so the seed repeat only redraws the
  bootstrap; it cannot add evidence. Governance note (forward-only, protocol text unchanged): the
  permitted repeat is still run for every INCONCLUSIVE result so that the decision to repeat never
  depends on how promising a result looks; for deterministic learners it is a formality.

---

### exp-0008 — Champion + context family
- Date / Stage: 2026-10-20 / Stage 2 (loop 2.3)
- Hypothesis (one sentence): Adding context features (league, season month, weeks into the season,
  European club minutes last season) to the champion improves validation PR-AUC at both horizons.
- Change vs champion (one controlled change; config diff link): feature set v0+history →
  v0+history+context (PLAN A8b; national-team load dropped before the run as not leakage-free) —
  `configs/exp-0008.yaml` vs `configs/exp-0004.yaml`; commit b42066e (clean)
- Data version (hash) / Split origin(s) / Seed: `panel_v3` 7f5b7ca1b1d2… (rows/labels identical) /
  main origin 2023/24 / 20261020
- Primary metric (main origin): y7 champion = 0.0307 | challenger = **0.0365** | Δ = +0.0057 |
  CI = [0.0035, 0.0092]; y28 champion = 0.1137 | challenger = **0.1357** | Δ = +0.0219 |
  CI = [0.0144, 0.0322]
- Secondary metrics (main origin; champion in brackets): sens@90spec y7 0.292 (0.242), y28 0.300
  (0.243); MCC y7 0.079 (0.059), y28 0.153 (0.111); Brier y7 0.2181 (0.2325), y28 0.2198 (0.2332).
  Origin spread y7 0.0431 / 0.0357 / 0.0365, y28 0.1443 / 0.1191 / 0.1357 — above the champion at
  every origin.
- Calibration slope: champion = y7 0.996, y28 0.995 | challenger = y7 1.036, y28 1.061
  (regression y7 0.032, y28 **0.056 > 0.05**)
- Verdict: **REJECT** (conditions 1–2 hold for both labels; condition 3 fails for y28). No repeat
  (the INCONCLUSIVE branch requires condition 2 to fail). Champion remains **exp-0004**. Stage-2
  non-promotion streak: 3.
- Notes & surprises:
  - Largest discrimination gain since the history family, significant at both horizons, rejected
    on a calibration-slope change of +0.006 beyond tolerance, in the *under*-confident direction
    (slope > 1).
  - Post-hoc diagnostic (does not change the verdict): player-cluster bootstrap (300 resamples) of
    the main-origin calibration slope gives SE ≈ 0.09 (champion) / 0.08 (challenger), and the
    slope-regression statistic has a 95% interval of [−0.112, 0.119] for y28 ([−0.095, 0.084] y7).
    The pre-registered 0.05 tolerance on a single-season point estimate is therefore well inside its
    own sampling noise: condition 3 is close to a coin flip for small changes. Recorded as PLAN open
    decision 12; the rule is **not** changed here (changing it after it rejected a challenger would
    be a post-hoc flip). Pre-specified follow-up: families rejected only on condition 3 are
    re-challenged once the Stage-4 calibration layer is part of the champion pipeline (PLAN 4.4).

---

### exp-0009 — Champion + decline-proxy family
- Date / Stage: 2026-10-20 / Stage 2 (loop 2.4)
- Hypothesis (one sentence): Adding performance-decline proxies (log market value at t, its
  12-month change, change in starts share) to the champion improves validation PR-AUC at both
  horizons.
- Change vs champion (one controlled change; config diff link): feature set v0+history →
  v0+history+decline (PLAN A8c, recorded before the run) — `configs/exp-0009.yaml` vs
  `configs/exp-0004.yaml`; commit 04ce786 (clean)
- Data version (hash) / Split origin(s) / Seed: `panel_v4` 1d2c2bb02f7a… (rows/labels identical) /
  main origin 2023/24 / 20261020
- Primary metric (main origin): y7 champion = 0.0307 | challenger = 0.0309 | Δ = +0.0002 |
  CI = [−0.0015, 0.0021]; y28 champion = 0.1137 | challenger = 0.1149 | Δ = +0.0011 |
  CI = [−0.0032, 0.0060]
- Secondary metrics (main origin): sens@90spec y7 0.239 (champion 0.242), y28 0.245 (0.243); MCC y7
  0.058 (0.059), y28 0.113 (0.111). Origin spread y7 0.0380 / 0.0305 / 0.0309, y28 0.1319 / 0.1071 /
  0.1149.
- Calibration slope: champion = y7 0.996, y28 0.995 | challenger = y7 1.018, y28 1.016
- Verdict: **INCONCLUSIVE** → repeat exp-0010.
- Notes & surprises: market value (a playing-level proxy) adds essentially nothing once history and
  lagged exposure are in the model.

---

### exp-0010 — Champion + decline-proxy family (seed repeat of exp-0009)
- Date / Stage: 2026-10-20 / Stage 2
- Change vs exp-0009: seed only (20261020 → 20261021) — `configs/exp-0010.yaml`; commit 5e26fba
- Primary metric (main origin): y7 Δ = +0.0002, CI = [−0.0013, 0.0021]; y28 Δ = +0.0011,
  CI = [−0.0030, 0.0059]; predictions identical to exp-0009 (deterministic learner).
- Verdict: **INCONCLUSIVE → loop exp-0009/0010 decided: REJECT**. Champion remains **exp-0004**.
  Stage-2 non-promotion streak: 4.

---

### exp-0011 — Champion trained from 2016/17 (data-scope loop, PLAN 2.6)
- Date / Stage: 2026-10-20 / Stage 2
- Hypothesis (one sentence): Training the champion only on seasons from 2016/17, after injury
  reporting stabilised, improves validation PR-AUC at both horizons.
- Change vs champion (one controlled change; config diff link): training window starts 2016/17
  instead of 2014/15 (`train_first_season: 2016`); evaluation rows unchanged —
  `configs/exp-0011.yaml` vs `configs/exp-0004.yaml`; commit 6cfd428 (clean)
- Data version (hash) / Split origin(s) / Seed: `panel_v4` 1d2c2bb02f7a… / main origin 2023/24 /
  20261020; training rows at the main origin 506,676 (y7) vs 643,978 for the champion
- Primary metric (main origin): y7 champion = 0.0307 | challenger = 0.0311 | Δ = +0.0004 |
  CI = [−0.0000, 0.0011]; y28 champion = 0.1137 | challenger = 0.1150 | Δ = +0.0012 |
  CI = [−0.0002, 0.0032]
- Secondary metrics (main origin): sens@90spec y7 0.233 (champion 0.242), y28 0.262 (0.243); Brier
  y7 0.2254 (0.2325), y28 0.2257 (0.2332). Origin spread y7 0.0358 / 0.0300 / 0.0311, y28 0.1256 /
  0.1063 / 0.1150.
- Calibration slope: champion = y7 0.996, y28 0.995 | challenger = y7 1.001, y28 1.004
- Verdict: **INCONCLUSIVE** → repeat exp-0012.
- Notes & surprises: dropping the two under-reported seasons (21% fewer training rows) does not hurt
  and may help slightly — consistent with the A4 reporting-density concern.

- **Governance note (2026-10-20, forward-only, recorded before exp-0012 ran):** for a
  *deterministic* learner the protocol's seed repeat produces bit-identical predictions and only
  redraws the bootstrap. Deciding on the repeat's CI alone would give a borderline result a second
  Monte Carlo draw (biased towards promotion). Operational reading of "then decide" from now on:
  if the repeat's predictions are bit-identical to the original's, condition 2 is judged on the
  percentile CI of the **pooled** Δ bootstrap draws of both runs (2 × 1000); conditions 1 and 3 are
  unchanged (identical point estimates). For stochastic learners the decision is the repeat's own
  verdict, as before. Check of earlier deterministic pairs: exp-0006/0007 and exp-0009/0010 are
  re-scored under this reading in the exp-0012 entry; their CIs were far from zero.

---

### exp-0012 — Champion trained from 2016/17 (seed repeat of exp-0011)
- Date / Stage: 2026-10-20 / Stage 2
- Change vs exp-0011: seed only (20261020 → 20261021) — `configs/exp-0012.yaml`; commit 00c1e78
- Primary metric (main origin): y7 Δ = +0.0004, CI = [−0.0001, 0.0011]; y28 Δ = +0.0012,
  CI = [−0.0002, 0.0032]; predictions bit-identical to exp-0011.
- Decision under the governance note above (pooled 2 × 1000 draws): y7 [−0.0001, 0.0011],
  y28 [−0.0002, 0.0032] → condition 2 fails → **loop exp-0011/0012: REJECT**. Champion remains
  **exp-0004**. Stage-2 non-promotion streak: 5.
- Re-scoring of earlier deterministic pairs under the pooled reading (same decisions):
  exp-0006/0007 y7 [−0.0004, 0.0010], y28 [−0.0011, 0.0063] → REJECT; exp-0009/0010 y7
  [−0.0014, 0.0021], y28 [−0.0031, 0.0060] → REJECT.

---

### stage-2 summary
- Champion: **exp-0004** (class-weighted LR, v0 + injury history). Cumulative Δ PR-AUC vs the
  Stage-1 baseline exp-0001 (main origin 2023/24): **y7 +0.0100** (0.0207 → 0.0307), **y28 +0.0360**
  (0.0778 → 0.1137); calibration slopes 0.996 / 0.995.
- Per-family verdicts:

  | Loop | Change | Decision | Note |
  |---|---|---|---|
  | 2.1 | + injury history | **PROMOTE** | significant at both horizons, every origin |
  | 2.7 | LR → LightGBM (on v0+history) | REJECT | tie at both horizons; second tree-model loss |
  | 2.2 | + season-load derivatives | REJECT | inconclusive ×2 (pooled CI straddles 0) |
  | 2.3 | + context | REJECT | large significant PR-AUC gain, failed calibration condition 3 at y28 by 0.006 (PLAN 4.4 re-challenge) |
  | 2.4 | + decline proxies (market value) | REJECT | no measurable gain |
  | 2.6 | training window from 2016/17 | REJECT | borderline (pooled CI lower bounds −0.0001 / −0.0002) |

- Exit: all planned families tested (exit criterion met; plateau streak 5 of 8). 2.5 stays blocked
  (no match-level data).
- Methodological findings for the thesis: (1) the tree ensemble has not beaten logistic regression on
  any open-data feature set tried; (2) calibration condition 3 as pre-registered is noise-dominated
  (PLAN open decision 12); (3) deterministic learners make the protocol's seed repeat a bootstrap
  redraw — handled by the pooled-draw reading recorded before exp-0012.

---

### amendment-2026-10-21 — calibration condition 3 (owner decision; forward only)
- Registered by the project owner after reviewing Stages 0–2 (PLAN open decision 12). Applies to
  experiments from **exp-0013** on; exp-0001…exp-0012 keep their verdicts (incl. exp-0008 REJECT).
- New condition 3: a challenger fails on calibration only if Δ_cal < −0.05 **and** the paired
  player-level bootstrap 95% CI of Δ_cal lies entirely below −0.05. Threshold unchanged.
  Δ_cal = |b_champion − 1| − |b_challenger − 1| (symmetric reading, flagged to the owner).
- The Stage-4 re-test of the context family after the calibration layer (PLAN 4.4) stays scheduled.
- Also confirmed by the owner: intersection rule over y7 and y28; decisions on the main origin
  2023/24 only; deterministic seed repeats decided on pooled bootstrap draws.
- Stage 3 design pre-registered as PLAN amendment A9 (before any survival model was trained).

---

### exp-0013 — Random Survival Forest, v0+history (survival baseline)
- Date / Stage: 2026-10-21 / Stage 3 (task 3.2)
- Hypothesis (one sentence): A Random Survival Forest on the Arm-A champion features orders time to
  the next time-loss injury within 90 days better than chance and yields usable 28-day risks.
- Change vs champion (one controlled change; config diff link): none — first survival champion of
  record (research pre-plan Stage 3; design A9) — `configs/exp-0013.yaml`, commit 5bdbb4b (clean); settings
  as pre-registered (200 trees, 50k-row bootstrap per tree, min_samples_leaf 100, sqrt features)
- Data version (hash) / Split origin(s) / Seed: `panel_v5` 7bb6cc0b81eb… / origins 2021/22, 2022/23,
  2023/24 with 90-day purge (main 2023/24) / 20261020
- Primary metric (main origin; player-cluster bootstrap, 1000 resamples): **C^td = 0.6694**
  [0.6408, 0.6972] (22,384 rows, 3,906 events within 90 days)
- Secondary metrics:

  | origin | C^td | IBS 1–89 d | null-KM IBS | risk28 PR-AUC | risk28 slope | risk28 intercept | mean risk28 / y28 prevalence |
  |---|---|---|---|---|---|---|---|
  | 2021/22 | 0.6284 | 0.0788 | 0.0812 | 0.1217 | 1.004 | −0.161 | 0.072 / 0.062 |
  | 2022/23 | 0.6654 | 0.0731 | 0.0759 | 0.1070 | 1.059 | −0.217 | 0.069 / 0.056 |
  | 2023/24 | 0.6694 | 0.0779 | 0.0815 | 0.1150 | 1.161 | −0.061 | 0.066 / 0.062 |

  IBS main origin 0.0779 [0.0703, 0.0849]; risk7 PR-AUC (vs y7) 0.0308. Null-KM IBS (training KM,
  same curve for every row) computed post hoc as a descriptive reference: Brier skill 3–4%.
  League C^td (main origin): ES1 0.697, GB1 0.694, FR1 0.690, L1 0.636, IT1 0.621; position: GK
  0.747, ATT 0.666, DEF 0.657, MID 0.619.
- Calibration slope (28-day risk vs y28): champion = n/a | challenger = 1.161 (main origin)
- Verdict: **CROWNED** — first survival champion of record (no promotion test applies).
- Notes & surprises: unlike the class-weighted Arm A, the survival model's 28-day risks are close to
  calibrated in the large (intercepts −0.06 to −0.22, mean risk within ~1 pp of prevalence). Its
  28-day PR-AUC (0.1150) is on par with the Arm-A champion's y28 (0.1137) — formal comparison in
  task 3.4. Fit ≈ 3 min per origin.

---

### exp-0014 — XGBoost-AFT, v0+history (survival challenger)
- Date / Stage: 2026-10-21 / Stage 3 (task 3.3)
- Hypothesis (one sentence): An XGBoost accelerated-failure-time model on the same features improves
  the validation time-dependent C-index over the Random Survival Forest champion.
- Change vs champion (one controlled change; config diff link): model block RSF → XGBoost-AFT
  (normal loss, σ 1.0, 400 rounds, lr 0.03, depth 6, min_child_weight 50, subsample/colsample 0.8,
  λ 1; pre-registered in A9 and committed before exp-0013 was evaluated) — `configs/exp-0014.yaml`
  vs `configs/exp-0013.yaml`; commit 5bdbb4b (clean)
- Data version (hash) / Split origin(s) / Seed: `panel_v5` 7bb6cc0b81eb… / main origin 2023/24 /
  20261020
- Primary metric (main origin): champion = 0.6694 | challenger = 0.6667 | Δ = −0.0027 |
  bootstrap 95% CI = [−0.0124, +0.0072]
- Secondary metrics (main origin): IBS 0.0792 vs 0.0779, Δ +0.0012 [+0.0005, +0.0020] (significantly
  worse); risk28 PR-AUC 0.1126 (champion 0.1150); risk28 mean 0.036 vs prevalence 0.062 (intercept
  +0.60); risk7 calibration slope 0.47, intercept +2.82 (median predicted 7-day risk 0.04% vs 1.6%
  observed). Origin spread C^td 0.6264 / 0.6721 / 0.6667 (champion 0.6284 / 0.6654 / 0.6694).
- Calibration slope (28-day risk): champion = 1.161 | challenger = 0.704; Δ_cal = −0.135, bootstrap
  CI [−0.392, +0.187] → condition 3 passes under rule v2 (CI not entirely below −0.05).
- Verdict: **REJECT** (condition 1 fails). Survival champion remains **exp-0013**. Stage-3
  non-promotion streak: 1.
- Notes & surprises:
  - Diagnosis: the pre-declared log-normal loss with σ = 1 imposes a hazard that is near zero in the
    first days and rises later; the observed injury hazard is close to constant. Ranking (C^td) is
    almost unaffected, but the curve shape — hence 7- and 28-day risk and IBS — is wrong.
  - Observation for the owner on the v2 calibration rule: here a point Δ_cal of −0.135 (slope
    1.16 → 0.70) passes because the single-season slope bootstrap CI spans ≈ ±0.3. At this sample size
    rule v2 will rarely bind; calibration is effectively gated by Stage 4 instead.
  - Harness robustness fix made during this evaluation (before any verdict was computed): the
    logistic-recalibration Newton solver diverged on these compressed risks (singular Hessian); it now
    halves a step only when the full step would lower the likelihood. Re-scoring exp-0004, exp-0008
    and exp-0013 gives identical verdicts and values equal to ≤ 1e-8.

---

### exp-0015 — XGBoost-AFT with Weibull loss, v0+history (survival challenger)
- Date / Stage: 2026-10-21 / Stage 3
- Hypothesis (one sentence): An XGBoost AFT model whose loss matches the near-constant injury hazard
  seen in the training data (extreme-value / Weibull, σ 1.1) improves the validation C^td over the
  Random Survival Forest champion.
- Change vs champion (one controlled change; config diff link): model block RSF → XGBoost-AFT; vs
  exp-0014 only `aft_loss_distribution: extreme`, `aft_loss_distribution_scale: 1.1`, fixed from the
  training-row Kaplan–Meier before the run (PLAN A9b) — `configs/exp-0015.yaml`; commit f5f69b7
  (clean)
- Data version (hash) / Split origin(s) / Seed: `panel_v5` 7bb6cc0b81eb… / main origin 2023/24 /
  20261020
- Primary metric (main origin): champion = 0.6694 | challenger = 0.6635 | Δ = −0.0059 |
  bootstrap 95% CI = [−0.0148, +0.0031]
- Secondary metrics (main origin): IBS 0.0784 vs 0.0779, Δ +0.0005 [+0.0001, +0.0010]; risk28
  PR-AUC 0.1116 (champion 0.1150); risk28 mean 0.072 vs prevalence 0.062; risk7 slope 1.07,
  intercept −0.28 (exp-0014: 0.47 / +2.82). Origin spread C^td 0.6237 / 0.6687 / 0.6635.
- Calibration slope (28-day risk): champion = 1.161 | challenger = 1.020 (Δ_cal +0.141, CI
  [−0.164, +0.220])
- Verdict: **REJECT** (condition 1 fails). Survival champion remains **exp-0013**. Stage-3
  non-promotion streak: 2. No further AFT variants (A9b).
- Notes & surprises: the training-derived curve shape repairs the AFT's absolute risks at every
  horizon (now the best-calibrated survival model in slope terms) but costs a little ranking.
  Calibration-vs-discrimination trade-off recorded for Stage 4.

---

### arm-comparison — Arm A exp-0004 vs Arm B exp-0013 (task 3.4, descriptive)
- Date / Stage: 2026-10-21 / Stage 3. Identical main-origin rows (2023/24); paired player-cluster
  bootstrap (1000 resamples, seed 20261020); `src/arm_comparison.py`, commit f5f69b7; report
  `reports/arm_comparison_exp-0004_exp-0013.md`.

  | horizon | prevalence | PR-AUC A | PR-AUC B | Δ (B − A) [95% CI] | slope A / B | intercept A / B | mean predicted A / B | Spearman | top-decile overlap |
  |---|---|---|---|---|---|---|---|---|---|
  | 7 d | 0.0161 | 0.0307 | 0.0308 | +0.0000 [−0.0016, +0.0013] | 0.996 / 1.164 | −4.134 / −0.085 | 0.461 / 0.018 | 0.961 | 0.822 |
  | 28 d | 0.0624 | 0.1137 | 0.1150 | +0.0012 [−0.0037, +0.0057] | 0.995 / 1.161 | −2.740 / −0.061 | 0.467 / 0.066 | 0.963 | 0.836 |

- Findings: (1) the two arms carry the same ranking information (no PR-AUC difference; rank
  correlation 0.96; > 80% shared top decile). (2) The survival arm's absolute risks are calibrated in
  the large without any post-hoc step, the class-weighted classifier's are not (inflated ~7–28×).
  (3) Arm B additionally gives a full 1–90-day risk curve from one model. For the thesis claim
  (research pre-plan §1): the time-to-event formulation adds calibrated absolute risk and timing, not
  discrimination. Both arms' slopes ≈ 1.16 / ≈ 1.0 leave the slope question to Stage 4.

---

### stage-3 summary
- Survival champion: **exp-0013** (Random Survival Forest, v0+history): main-origin C^td 0.6694
  [0.6408, 0.6972], IBS 0.0779 (null-KM 0.0815), 28-day risk slope 1.161, intercept −0.061.
- Challengers: exp-0014 XGBoost-AFT log-normal REJECT (C^td −0.0027, IBS significantly worse, curve
  shape misspecified); exp-0015 XGBoost-AFT Weibull (distribution from training KM) REJECT (C^td
  −0.0059; best calibrated). Streak 2.
- Arm A vs Arm B documented (above). Exit criteria met.
- Rule v2 observation: single-season slope CIs (≈ ±0.3) make calibration condition 3 rarely binding;
  calibration quality is in practice decided in Stage 4.

---

### rulings-2026-10-21 — owner decisions after Stage 3; Stage-4 pre-registration
- Δ_cal = change in |slope − 1|, both directions count — official. Calibration veto power at current
  sample size accepted as is. Promotion = primary metric + calibration veto; IBS and other secondary
  metrics are reported with CIs and never decide. Harness-fix procedure approved.
- Stage-4 design pre-registered as PLAN amendment A10 before any Stage-4 model was fitted.
- Raised for owner ruling (PLAN open decision 15): calibration-layer loops cannot move PR-AUC
  (Platt is rank-preserving), so they are decided on the Brier score with a discrimination guard and
  the unchanged calibration veto; feature/model loops keep PR-AUC as primary.

---

### exp-0016 — Champion + Platt calibration layer
- Date / Stage: 2026-10-21 / Stage 4 (task 4.1, first loop)
- Hypothesis (one sentence): A Platt calibration layer fitted on the temporal calibration split
  lowers the validation Brier score of the champion at both horizons without hurting discrimination.
- Change vs champion (one controlled change; config diff link): adds `calibration: {method: platt}`
  (A10 split: inner model ≤ V − 2, calibrator on half A of season V − 1, training player group only);
  decision metric Brier with PR-AUC guard (open decision 15) — `configs/exp-0016.yaml` vs
  `configs/exp-0004.yaml`; commit 7eee611 (clean)
- Data version (hash) / Split origin(s) / Seed: `panel_v5` 7bb6cc0b81eb… / main origin 2023/24 /
  20261020. Calibration rows at the main origin: half A 39,411 (y7, 615 positives) / 36,908 (y28,
  2,288 positives); inner model on 567,056 / 561,210 rows.
- Decision metric (main origin, paired player-cluster bootstrap): y7 Brier champion = 0.23248 |
  challenger = **0.01576** | Δ = −0.21672 | CI = [−0.22640, −0.20667]; y28 champion = 0.23323 |
  challenger = **0.05726** | Δ = −0.17597 | CI = [−0.18389, −0.16691]
- Guard / primary metric: PR-AUC unchanged by construction (y7 0.0307, y28 0.1137; Δ CI [0, 0]).
- Secondary metrics (main origin): calibration slope y7 1.030, y28 1.046; intercept y7 +0.066, y28
  +0.033; mean predicted y7 0.0151 vs prevalence 0.0161, y28 0.0605 vs 0.0624. Origin spread of slope
  y7 0.989 / 0.933 / 1.030, y28 0.987 / 0.901 / 1.046. Fitted maps (a, b): y7 (−4.08, 0.96),
  (−4.16, 1.04), (−4.19, 0.97); y28 (−2.68, 0.94), (−2.75, 1.08), (−2.76, 0.95) — stable across
  origins.
- Calibration slope: champion = y7 0.996, y28 0.995 | challenger = y7 1.030, y28 1.046 (Δ_cal −0.026 /
  −0.041; veto v2 not triggered)
- Verdict: **PROMOTE** (Brier conditions 1–2 at both horizons, guard and veto pass). New champion
  **exp-0016**; exp-0004 archived in place.
- Notes & surprises: class weighting distorted the intercept, not the slope; Platt essentially
  undoes the weighting (intercepts ≈ log of the prior odds ratio). Stage-4 exit check at the main
  origin: both slopes inside [0.9, 1.1] ✓ (2022/23 y28 0.901 at the edge).

---

### exp-0017 — Champion with isotonic instead of Platt calibration
- Date / Stage: 2026-10-21 / Stage 4 (task 4.1, second loop)
- Hypothesis (one sentence): A non-parametric isotonic calibration map lowers the validation Brier
  score further than the Platt map at both horizons without hurting discrimination.
- Change vs champion (one controlled change; config diff link): calibration method Platt → isotonic
  (same split, same base model) — `configs/exp-0017.yaml` vs `configs/exp-0016.yaml`; commit b72cdc2
  (clean)
- Data version (hash) / Split origin(s) / Seed: `panel_v5` 7bb6cc0b81eb… / main origin 2023/24 /
  20261020
- Decision metric (main origin): y7 Brier champion = 0.01576 | challenger = 0.01577 | Δ = +0.000004 |
  CI = [−0.00001, +0.00002]; y28 champion = 0.05726 | challenger = 0.05726 | Δ = −0.000002 |
  CI = [−0.00018, +0.00014]
- Guard: PR-AUC y7 0.0307 → 0.0295, Δ −0.0012 [−0.0026, −0.0006]; y28 0.1137 → 0.1106, Δ −0.0031
  [−0.0065, −0.0006] → significantly worse at both horizons (guard fails).
- Secondary metrics (main origin): slope y7 0.936, y28 0.980; sens@90spec y7 0.136 (champion 0.242),
  y28 0.239 (0.243) — the step map's ties collapse the high-risk tail at 7 days.
- Calibration slope: champion = y7 1.030, y28 1.046 | challenger = y7 0.936, y28 0.980
- Verdict: **REJECT** (Brier condition 1 fails at y7, condition 2 at both; PR-AUC guard fails).
  Champion remains **exp-0016** (Platt). Stage-4 non-promotion streak: 1.
- Notes & surprises: with a near-correct slope already, the extra flexibility of isotonic buys no
  Brier and costs ranking resolution through ties — the guard did its job.

---

### exp-0018 — Calibrated champion + context family (re-test of exp-0008, PLAN 4.4)
- Date / Stage: 2026-10-21 / Stage 4
- Hypothesis (one sentence): Adding the context family (definitions unchanged from exp-0008) to the
  Platt-calibrated champion improves validation PR-AUC at both horizons.
- Change vs champion (one controlled change; config diff link): feature set v0+history →
  v0+history+context; calibration layer as in exp-0016 (Platt, refitted on the challenger's own inner
  model) — `configs/exp-0018.yaml` vs `configs/exp-0016.yaml`; commit cdc4c2f (clean)
- Data version (hash) / Split origin(s) / Seed: `panel_v5` 7bb6cc0b81eb… / main origin 2023/24 /
  20261020
- Primary metric (main origin): y7 champion = 0.0307 | challenger = **0.0365** | Δ = +0.0057 |
  CI = [0.0035, 0.0092]; y28 champion = 0.1137 | challenger = **0.1357** | Δ = +0.0219 |
  CI = [0.0144, 0.0322] (identical to exp-0008: Platt preserves ranking)
- Secondary metrics (main origin; champion in brackets): Brier y7 0.01574 (0.01576), y28 0.05658
  (0.05726); sens@90spec y7 0.292 (0.242), y28 0.300 (0.243); MCC y7 0.079, y28 0.153.
- Calibration slope: champion = y7 1.030, y28 1.046 | challenger = y7 **1.190**, y28 **1.329**;
  Δ_cal y7 −0.160 [−0.230, +0.121], y28 −0.283 [−0.367, +0.009] → veto v2 not triggered (upper bounds
  above −0.05).
- Verdict: **PROMOTE** (standard rule at both horizons; veto v2 passes). New champion **exp-0018**;
  exp-0016 archived in place. Stage-4 non-promotion streak reset to 0.
- Notes & surprises:
  - The exp-0008 discrimination gain replicates exactly; under rule v2 the family is now promoted.
  - Calibration got worse at the main origin, not better: the inner model (≤ 2021/22) was
    overconfident on the calibration season 2022/23, so Platt learned b = 0.80 (y28) / 0.87 (y7); the
    deployed model on 2023/24 was not (raw slope 1.061 / 1.036), so the shrink made it
    under-confident: 1.061 / 0.799 ≈ 1.33. On the calibration season's held-out half B the calibrated
    slope is fine (0.97 / 0.95). Across calibration seasons b ranges 0.80–1.08: the slope parameter
    carries season-to-season drift. Stage-4 exit check fails for this champion (slopes outside
    [0.9, 1.1]) → addressed by the pre-registered loop exp-0019 (A10b).
  - As noted in Stage 3, rule v2 rarely binds at this sample size: a point Δ_cal of −0.28 passes.

- Correction (2026-10-21) to the exp-0018 entry: main-origin Brier values are y7 **0.01571**
  (written as 0.01574) and y28 **0.05657** (written as 0.05658), read from
  `data/experiments/exp-0018/metrics.json`. No effect on the verdict (decided on PR-AUC).

---

### exp-0019 — Champion with intercept-only calibration
- Date / Stage: 2026-10-21 / Stage 4 (A10b)
- Hypothesis (one sentence): Recalibrating the champion's class-weighted scores by an intercept
  shift only (slope fixed at 1) lowers the validation Brier score at both horizons compared with the
  full Platt map.
- Change vs champion (one controlled change; config diff link): calibration method platt →
  intercept (same split, same base model and features) — `configs/exp-0019.yaml` vs
  `configs/exp-0018.yaml`; commit 8493ee5 (clean)
- Data version (hash) / Split origin(s) / Seed: `panel_v5` 7bb6cc0b81eb… / main origin 2023/24 /
  20261020
- Decision metric (main origin): y7 Brier champion = 0.015710 | challenger = 0.015713 |
  Δ = +0.000003 | CI = [−0.00001, +0.00002]; y28 champion = 0.056566 | challenger = 0.056503 |
  Δ = −0.00006 | CI = [−0.00026, +0.00018]
- Guard / primary metric: PR-AUC identical (rank-preserving map): y7 0.0365, y28 0.1357.
- Secondary metrics — calibration slope per origin (2021/22, 2022/23, 2023/24):

  | label | Platt (champion exp-0018) | intercept-only (exp-0019) |
  |---|---|---|
  | y7 | 1.084 / 0.878 / **1.190** | 0.965 / 0.879 / **1.036** |
  | y28 | 1.042 / 0.911 / **1.329** | 0.916 / 0.902 / **1.061** |

  Mean |slope − 1| over origins: Platt y7 0.132, y28 0.153; intercept-only y7 0.064, y28 0.081.
  Main-origin intercepts (intercept-only): y7 +0.095, y28 +0.107.
- Calibration slope: champion = y7 1.190, y28 1.329 | challenger = y7 1.036, y28 1.061 (Δ_cal +0.154 /
  +0.268)
- Verdict: **REJECT** (Brier condition 1 fails at y7 by +0.000003; condition 2 fails at both).
  Champion remains **exp-0018**. Stage-4 non-promotion streak: 1.
- Notes & surprises:
  - The intercept-only map does exactly what theory predicts — slopes return inside [0.9, 1.1] at the
    main origin and are more stable across origins — but the pre-registered decision metric cannot
    see it: at 1.6–6% prevalence the Brier score is dominated by the base rate and is nearly blind
    to a slope error of this size. Raised as PLAN open decision 16; the verdict is not revisited.
  - Per A10b the Stage-4 calibration exit is closed as **documented**: the champion (exp-0018,
    Platt) has main-origin slopes 1.190 / 1.329, outside [0.9, 1.1].

---

### conformal — split-conformal prediction sets on the champion exp-0018 (task 4.2, descriptive)
- Date / Stage: 2026-10-21 / Stage 4. `src/conformal.py` (marginal LAC verified equal to MAPIE 1.6
  `SplitConformalClassifier`), commit b46cc50; quantiles from calibration half B of each origin
  (calibrator_A ∘ inner model, out of sample for both); coverage on validation rows, player-cluster
  bootstrap CI; full table `reports/conformal_exp-0018.md`.
- Marginal LAC: overall coverage at or slightly above nominal (α = 0.10: 0.913–0.920 for y28 and y7)
  but degenerate — because prevalence < α, class 1 is excluded almost everywhere: injured-player
  coverage 0.000 at α ≥ 0.10 (y28 α = 0.05: 0.09–0.26), and 2–17% empty sets. Not usable for injury
  flagging.
- Class-conditional (Mondrian) LAC, α = 0.10:

  | label | origin | coverage [95% CI] | injured | uninjured | low {0} share / rate | uncertain {0,1} | high {1} |
  |---|---|---|---|---|---|---|---|
  | y28 | 2021/22 | 0.918 [0.903, 0.931] | 0.850 | 0.923 | 0.359 / 0.026 | 0.553 / 0.067 | 0.088 / 0.179 |
  | y28 | 2022/23 | 0.917 [0.901, 0.930] | 0.890 | 0.918 | 0.313 / 0.020 | 0.599 / 0.065 | 0.088 / 0.126 |
  | y28 | 2023/24 | 0.933 [0.919, 0.945] | 0.900 | 0.935 | 0.310 / 0.020 | 0.615 / 0.069 | 0.075 / 0.185 |
  | y7 | 2021/22 | 0.920 [0.905, 0.934] | 0.884 | 0.920 | 0.338 / 0.006 | 0.579 / 0.018 | 0.082 / 0.051 |
  | y7 | 2022/23 | 0.914 [0.897, 0.929] | 0.868 | 0.914 | 0.330 / 0.006 | 0.583 / 0.017 | 0.087 / 0.031 |
  | y7 | 2023/24 | 0.925 [0.910, 0.939] | 0.911 | 0.925 | 0.329 / 0.004 | 0.593 / 0.019 | 0.077 / 0.046 |

- Coverage check: overall Mondrian coverage meets 1 − α at every origin and α (0.05: 0.960–0.967;
  0.20: 0.816–0.851); injured-class coverage is below nominal at the earliest origin (y28 2021/22
  0.850 at α = 0.10) and at nominal at the main origin — temporal shift erodes the guarantee
  modestly, it does not break it. No empty sets.
- Risk tiers (main origin, y28, α = 0.10): low 31% of player-weeks with a 2.0% injury rate, high
  7.5% with 18.5% — a 9× separation; the "uncertain" tier (61%) sits at the base rate. The honest
  reading: the model can rule out about a third of player-weeks and flag a small high-risk group;
  for most player-weeks it cannot decide.

---

### decision-curves — net benefit on the main origin (task 4.3, descriptive)
- Date / Stage: 2026-10-21 / Stage 4. `src/decision_curves.py`, commit b46cc50; champion exp-0018
  vs exp-0004 (uncalibrated), exp-0016 (calibrated, history only), exp-0013 (survival 1 − S(h)),
  treat-all, treat-none; thresholds 0.01–0.30 (§3a grid); player-cluster bootstrap 1000 resamples;
  table `reports/decision_curves_exp-0018.md`.
- y28: champion margin over max(treat-all, treat-none) has a CI above 0 for **pt = 0.01–0.14**; peak
  +0.0191 [+0.0147, +0.0217] at pt = 0.06 (≈ 19 additional net true positives per 1,000 player-weeks);
  no benefit above 0.15 (few risks that high).
- y7: useful range **pt = 0.01–0.03** (peak +0.0037 [+0.0024, +0.0051] at 0.02).
- The uncalibrated exp-0004 coincides with treat-all at every threshold (its inflated
  probabilities exceed every pt) — decision utility requires the calibration layer. exp-0016 and the
  survival model lie below the champion across the useful range (y28 at pt = 0.06: 0.0175 and
  0.0173 vs 0.0216).

---

### stage-4 summary
- Champion: **exp-0018** — class-weighted LR, v0 + history + context, Platt calibration on the
  temporal calibration split. Main-origin PR-AUC y7 0.0365, y28 0.1357; Brier y7 0.01571, y28
  0.05657; calibration slope y7 1.190, y28 1.329 (intercepts +0.087 / +0.089).
- Loops: exp-0016 Platt PROMOTE (Brier −0.217 / −0.176); exp-0017 isotonic REJECT (PR-AUC guard);
  exp-0018 context re-test PROMOTE under veto v2 (PR-AUC +0.0057 / +0.0219); exp-0019 intercept-only
  REJECT (Brier unchanged although slopes returned to 1.04 / 1.06). Streak 1.
- Exit: calibrated champion ✓; slope within [0.9, 1.1] at the main origin ✗ → **documented**
  (exp-0018 entry; open decision 16 for the owner). Conformal (Mondrian) and decision-curve
  analyses documented above.
- Cumulative Δ PR-AUC vs the Stage-1 baseline exp-0001 (main origin): y7 +0.0158 (0.0207 → 0.0365),
  y28 +0.0579 (0.0778 → 0.1357).

---

### rulings-2026-10-21b — owner decisions after Stage 4
- Decision 15 (Brier-decided calibration loops) confirmed as written for exp-0016…exp-0019.
- Decision 16 amendment adopted, forward only (loops from exp-0020 on; exp-0019 stays REJECT):
  calibration-only loops are decided on Δ_cal = change in |slope − 1| (condition 1 Δ_cal > 0,
  condition 2 bootstrap CI entirely above 0, both horizons); guards: no significant PR-AUC drop, no
  significant Brier increase; calibration veto v2 unchanged.
- Intercept-only may re-challenge as exp-0020; one optional loop for a per-season intercept refit
  (exp-0021; reading recorded as PLAN open decision 17). `FINAL_RUN.md` must state the final
  calibration configuration explicitly.
- Stage 5 to follow (SHAP, PDP/ICE incl. the best tree challenger's shapes, position breakdown,
  modifiable vs non-modifiable factors, thesis figures; SoccerMon check only if cheap).

---

### exp-0020 — Champion with intercept-only calibration (re-challenge under decision 16)
- Date / Stage: 2026-10-21 / Stage 4
- Hypothesis (one sentence): Recalibrating the champion's class-weighted scores by an intercept shift
  only brings the validation calibration slope closer to 1 at both horizons than the full Platt map,
  without hurting discrimination or Brier.
- Change vs champion (one controlled change; config diff link): calibration method platt →
  intercept; decision metric Δ|slope − 1| (decision 16) — `configs/exp-0020.yaml` vs
  `configs/exp-0018.yaml`; commit b37d187 (clean). Pipeline identical to exp-0019 (predictions
  bit-identical); only the decision rule differs.
- Data version (hash) / Split origin(s) / Seed: `panel_v5` 7bb6cc0b81eb… / main origin 2023/24 /
  20261020
- Decision metric (main origin, paired player-cluster bootstrap): y7 slope champion = 1.190 |
  challenger = 1.036 | Δ_cal = +0.1544 | CI = [−0.0824, +0.1782]; y28 champion = 1.329 |
  challenger = 1.061 | Δ_cal = +0.2672 | CI = [+0.0482, +0.3097]
- Guards: PR-AUC identical (y7 0.0365, y28 0.1357); Brier Δ y7 +0.000004 [−0.000011, +0.000023],
  y28 −0.000063 [−0.000260, +0.000178] — both guards pass. Veto v2 not triggered (Δ_cal > 0).
- Calibration slope: champion = y7 1.190, y28 1.329 | challenger = y7 1.036, y28 1.061
- Verdict: **INCONCLUSIVE** (condition 1 holds at both horizons; condition 2 holds at y28 only) →
  one repeat allowed; deterministic learner → decided on pooled bootstrap draws (exp-0021).
- Notes & surprises: the y7 interval is wide on the low side because |b − 1| is non-linear near 1:
  in resamples where the champion's y7 slope happens to sit near 1 the improvement vanishes. The
  repeat takes id exp-0021, so the optional per-season intercept loop (open decision 17) becomes
  exp-0022.

---

### exp-0021 — Champion with intercept-only calibration (seed repeat of exp-0020)
- Date / Stage: 2026-10-21 / Stage 4
- Change vs exp-0020: seed only (20261020 → 20261021) — `configs/exp-0021.yaml`; commit 067ea9d;
  predictions bit-identical to exp-0020 (deterministic learner).
- Decision metric (own draws): y7 Δ_cal +0.1544 [−0.0899, +0.1782]; y28 +0.2672 [+0.0549, +0.3082].
  Pooled 2 × 1000 draws (confirmed rule for deterministic repeats): y7 [−0.0899, +0.1782],
  y28 [+0.0486, +0.3088].
- Verdict: **INCONCLUSIVE → loop exp-0020/0021 decided: REJECT** (condition 2 fails at y7 on the pooled
  draws). Champion remains **exp-0018**. Stage-4 non-promotion streak: 2.

---

### exp-0022 — Champion with intercept refitted per season on the deployed model
- Date / Stage: 2026-10-21 / Stage 4 (optional loop, reading in PLAN open decision 17)
- Hypothesis (one sentence): Refitting the calibration intercept each season on the deployed model's
  own scores of the last training season brings the validation slope closer to 1 at both horizons
  than the Platt map learned from the inner model, without hurting discrimination or Brier.
- Change vs champion (one controlled change; config diff link): calibration block → intercept-only,
  `fit_on: deployed` — `configs/exp-0022.yaml` vs `configs/exp-0018.yaml`; commit ebfdf46 (clean)
- Data version (hash) / Split origin(s) / Seed: `panel_v5` 7bb6cc0b81eb… / main origin 2023/24 /
  20261020
- Decision metric (main origin): y7 Δ_cal +0.1544 [−0.0824, +0.1782]; y28 +0.2672 [+0.0482, +0.3097]
  — identical to exp-0020. Guards pass (Brier Δ y7 +0.000003, y28 −0.000075, CIs straddle 0).
- Secondary: main-origin intercepts y7 +0.082, y28 +0.093 (exp-0020: +0.095 / +0.107); fitted
  intercepts within ≈ 0.01–0.02 of the inner-model fits (e.g. y28 2023: −2.804 vs −2.818).
- Verdict: **INCONCLUSIVE** → repeat exp-0023.
- Notes & surprises: structural, not chance — a recalibration slope is invariant to an intercept
  shift, so every intercept-only variant inherits the deployed model's slope (1.036 / 1.061) and the
  decision-16 metric cannot distinguish intercept variants; they differ only in calibration-in-the-
  large, where the deployed-model fit is marginally closer to 0.

---

### exp-0023 — Intercept refitted on the deployed model (seed repeat of exp-0022)
- Change vs exp-0022: seed only — `configs/exp-0023.yaml`; commit 59b1756; predictions bit-identical.
- Pooled 2 × 1000 draws: y7 [−0.0899, +0.1782], y28 [+0.0486, +0.3088] (identical to exp-0020/0021,
  as the slope draws must be).
- Verdict: **INCONCLUSIVE → loop exp-0022/0023 decided: REJECT**. Champion remains **exp-0018**.
  Stage-4 non-promotion streak: 3.

---

### stage-4b summary — calibration loops under decision 16
- Intercept-only (exp-0020/0021) and the deployed-model intercept refit (exp-0022/0023) both improve
  the main-origin slope to 1.036 / 1.061 — significant at y28, not at y7 — and are REJECT under the
  both-horizons rule. Champion and final calibration configuration remain **exp-0018: Platt map
  fitted on the inner model's out-of-time scores of half A of season V − 1**; to be stated in
  `FINAL_RUN.md`.
- Thesis note: at this sample size the y7 slope bootstrap is too wide to certify a 0.15 improvement in
  |slope − 1|; calibration-in-the-large differences between intercept variants are ≈ 0.01 in logit.

---

### stage-5 — interpretability and science output (descriptive; design A11)
- Date / Stage: 2026-10-21 / Stage 5. Code `src/interpret.py`, `src/figures.py`, commit c8de203;
  outputs `data/experiments/exp-0018/interpret/`, figures `reports/figures/fig1–fig9`, narrative
  `reports/stage5_interpretability.md`. Main origin 2023/24.
- Reproduction gate: exp-0018 (LR + Platt), exp-0005 (LightGBM) and exp-0013 (RSF) refitted at the
  main origin reproduce their stored validation predictions bit-identically.
- SHAP (exact linear, calibrated log-odds; additivity error 1.8e-15; equal to shap.LinearExplainer):
  share of mean |φ| for y28 — modifiable exposure/load 22%, partially modifiable return-to-play 8%,
  non-modifiable 70% (y7: 30 / 8 / 62%). Top features y28: season_month 21%, prev1_minutes_est 10%,
  weeks_into_season 7%, days_since_return 7%, league 6%, position 6%.
- **Key finding — context gain is a calendar effect** (post-hoc, descriptive; verdicts unchanged):
  in-season weeks only (Sep–Apr), exp-0018 vs exp-0016 PR-AUC y28 0.1510 vs 0.1502, Δ CI
  [−0.0071, +0.0089]; y7 0.0393 vs 0.0387, CI [−0.0013, +0.0030]. Development-season 28-day rates by
  start month: May–Jun 2.1–2.7%, Jul 4.4%, Sep–Jan ≈ 8%. `season_month` largely encodes whether the
  window contains football. Raised as PLAN open decision 18 (evaluation population before Stage 6).
- Dose-response, pre-registered U/J criterion: no U/J shape for chronic load (prev1_minutes_est) in
  any model — monotone increasing (LightGBM +0.13, RSF +0.10, LR +0.78 log-odds end to end);
  prev2 minutes flat; appearances inverted-U in LightGBM (not U/J); age decreasing (survivor effect);
  days since return steeply decreasing over ≈ 500 days (re-injury window).
- Positions (y28, 2023/24): PR-AUC vs prevalence GK 0.143 / 0.023, DEF 0.161 / 0.074, MID 0.128 /
  0.066, ATT 0.128 / 0.064; goalkeeper calibration weakest (mean slope over origins 1.53). Injury
  site mix differs (hamstring 22% of midfielders' vs 6% of goalkeepers' injuries).
- SoccerMon transfer check (5.4): **skipped, not cheap** — raw release not local (OSF download);
  local `pmsys` tables are third-party derivatives with undocumented processing on two elite female
  teams. Recorded as future work.
- Figures: palette validator not run (Node.js unavailable); documented pre-validated reference slots
  used unchanged; every figure rendered and visually checked, layout fixes applied (legend placement,
  axis-label wrapping, label collisions).

---

### rulings-2026-10-21c — owner decisions before Stage 6
- Decision 17 reading confirmed. Rolling in-season calibration updates and the SoccerMon transfer check
  logged as future work.
- Decision 18 ruled: the **primary** evaluation population for the final run is in-season, at-risk
  player-weeks (prediction Sunday in September–April; a not-at-risk person-time correction registered
  before the holdout is touched); all weeks is a mandatory secondary/sensitivity analysis reported side
  by side. exp-0018 stays champion; the final report must state that in-season the context family adds
  nothing and the champion is effectively equivalent to exp-0004.

---

### final-run pre-registration — 2026-10-21 10:06 (+02:00), before any holdout row was scored
- `FINAL_RUN.md` written; code frozen at commit 90209ca3ba9f48bd85bf00bdd712a8fa5333a105
  (`src/final_run.py`, `configs/final_run.yaml`); data `panel_v5` 7bb6cc0b81eb…; seeds 20261020.
- Champion exp-0018 with the Platt map fitted on the inner model's scores of half A of season
  2023/24 (training pool = train + validation groups); comparators exp-0016 and exp-0001; survival
  champion exp-0013. Primary population in-season (Sep–Apr), secondary all weeks.
- Rehearsal on the main validation origin reproduced every stored prediction bit-identically
  (exp-0018, exp-0016, exp-0001 × y7/y28; exp-0013) and the logged metrics.
- Command: `python -m src.final_run --expected-commit 90209ca3ba9f48bd85bf00bdd712a8fa5333a105`

---

### final-run — one-shot holdout evaluation (2024/25, test player group)
- Date / Stage: 2026-10-21 10:08–10:18 (+02:00) / Stage 6. Command and code as pre-registered
  (commit 90209ca, clean tree, `panel_v5` 7bb6cc0b81eb… verified, leakage tests 8/8 at run time);
  single execution, exit 0; one-shot marker `data/final/holdout_evaluated.json`; results
  `data/final/holdout/results.json`. Holdout: 673 players; y7 23,727 rows (453 positives), y28 23,627
  (1,714); in-season y7 15,435 (378), y28 15,355 (1,402). Training pool 932,763 / 924,546 rows,
  7,404 players. Platt maps: y7 (a −4.138, b 0.946), y28 (a −2.746, b 0.936).
- **Primary population (in-season, Sep–Apr), champion exp-0018, 1000 player-cluster resamples:**

  | metric | y7 | y28 |
  |---|---|---|
  | prevalence | 0.0245 | 0.0913 |
  | **PR-AUC** | **0.0463** [0.0380, 0.0577] | **0.1580** [0.1335, 0.1877] |
  | ROC-AUC | 0.657 [0.626, 0.684] | 0.661 [0.630, 0.690] |
  | sens @ 90% spec | 0.217 [0.172, 0.273] | 0.229 [0.184, 0.278] |
  | MCC @ 90% spec | 0.060 [0.037, 0.087] | 0.118 [0.077, 0.160] |
  | Brier | 0.0237 [0.0209, 0.0267] | 0.0810 [0.0726, 0.0896] |
  | calibration slope | 0.943 [0.765, 1.115] | 0.962 [0.779, 1.149] |
  | calibration intercept | +0.202 [+0.082, +0.310] | +0.234 [+0.105, +0.343] |
  | mean predicted | 0.0201 | 0.0741 |
  | Δ PR-AUC vs exp-0016 (history only) | −0.0015 [−0.0042, +0.0015] | −0.0033 [−0.0109, +0.0044] |
  | Δ PR-AUC vs exp-0001 (Stage-1 baseline) | +0.0151 [+0.0070, +0.0246] | +0.0442 [+0.0229, +0.0675] |
  | decision curve: useful range | pt 0.02–0.04 | pt 0.07–0.17 |
  | Mondrian α = 0.10: coverage (injured / uninjured) | 0.861 [0.839, 0.884] (0.923 / 0.860) | 0.884 [0.866, 0.902] (0.910 / 0.881) |
  | tiers α = 0.10: low / uncertain / high (share, injury rate) | 19.8% 0.9% / 65.8% 2.4% / 14.4% 4.9% | 22.1% 3.7% / 64.7% 9.2% / 13.2% 18.0% |

  Comparators (in-season): exp-0016 PR-AUC y7 0.0478, y28 0.1613; exp-0001 0.0312 / 0.1138 (ROC-AUC
  0.592 / 0.596; Brier 0.259 / 0.256 — uncalibrated by design). Survival champion exp-0013
  in-season: C^td 0.6433 [0.6126, 0.6744], IBS 0.1038 [0.0943, 0.1136], 28-day risk PR-AUC 0.1608,
  slope 1.070, intercept +0.308.
- **Secondary population (all weeks):** champion PR-AUC y7 0.0423 [0.0347, 0.0514] (prevalence
  0.0191), y28 0.1445 [0.1227, 0.1675] (0.0725); ROC-AUC 0.698 / 0.701; slope 1.031 / 1.059,
  intercept +0.150 / +0.163; Δ vs exp-0016 y7 +0.0066 [+0.0032, +0.0107], y28 +0.0187 [+0.0109,
  +0.0278]; Δ vs exp-0001 y7 +0.0181 [+0.0115, +0.0254], y28 +0.0538 [+0.0362, +0.0712]; useful
  range y7 0.01–0.03, y28 0.03–0.15; Mondrian α = 0.10 coverage 0.902 [0.884, 0.917] / 0.910
  [0.896, 0.924]. Survival C^td 0.6453 [0.6184, 0.6724], IBS 0.0887.
- Strata (in-season, y28 PR-AUC / prevalence / slope): ES1 0.158 / 0.057 / 1.46, FR1 0.147 / 0.080 /
  0.93, GB1 0.174 / 0.104 / 0.83, IT1 0.164 / 0.106 / 0.87, L1 0.164 / 0.110 / 0.96; attack 0.175 /
  0.107 / 0.87, defence 0.159 / 0.100 / 0.89, midfield 0.144 / 0.086 / 1.02, goalkeepers 0.112 /
  0.040 / 1.32.
- **Against the pre-registered development reference points:** PR-AUC in-season y7 0.0393 → 0.0463,
  y28 0.1510 → 0.1580; all weeks 0.0365 → 0.0423, 0.1357 → 0.1445 — but prevalence rose more (y28
  in-season 0.0787 → 0.0913), so lift over prevalence fell slightly (y28 in-season 1.92× → 1.73×;
  all weeks 2.17× → 1.99×). Calibration slopes are closer to 1 than in development (0.94–1.06 vs
  1.11–1.33); the level is under-predicted (intercept +0.15 to +0.23) because 2024/25 had more
  injuries than the training seasons. Mondrian coverage at α = 0.10 holds for all weeks but falls
  below nominal in-season at 7 days (CI excludes 0.90) — the shift erodes the conformal guarantee in
  the primary population. Survival C^td dropped from 0.669 to 0.645.
- **Owner-mandated statement, checked out of sample:** in-season the context family added nothing in
  development and adds nothing on the holdout (Δ vs history-only −0.0015 / −0.0033, CIs straddle 0);
  in-season the champion is effectively equivalent to exp-0004 (exp-0016 = exp-0004 + Platt, same
  ranking). The all-weeks advantage replicates and is the calendar effect.
- Literature points (indicative only, different units): ROC-AUC 0.66 in-season / 0.70 all weeks vs
  Rossi et al. 2018 ≈ 0.76 (single club, GPS, session level); PR-AUC 0.158 vs Wang & Lee 2026
  ≈ 0.759 is not comparable across prevalence and unit of analysis.
- Verdict: none (one-shot evaluation, reported as is). No re-run, re-tuning or recalibration.

---

### stage-7 — practitioner evidence pack (owner brief 2026-10-21; analysis only)
- Brief: exp-0024 … exp-0029 in order, one config and one entry each; the deployed champion is
  **frozen**; no model-selection decision from any result; everything reported as it comes out;
  missing artifacts regenerated only through the final-run code path; README not touched.
- Readings recorded in PLAN open decisions 19–24 before any run. (19) The frozen champion is the
  final-run exp-0018 with its Platt map; the "per-season intercept refit" named in the brief is
  exp-0022/0023, which was **rejected** and is not part of the champion. (20) The datalake has no
  dated minutes: the usage band uses last-season minutes, and the specified 7:28-day ACWR cannot be
  scored (implemented and tested, plus a labelled season-scale substitute). (21)–(24) operational,
  subgroup, per-season and prospective definitions.
- Repository history was rewritten by the owner after the final run (publishing); the pre-registered
  final-run commit 90209ca and the commits cited in earlier entries are no longer in the history.
  Task 7.0 re-establishes the link between the current code and the frozen run.

### task 7.0 — frozen champion re-materialised and verified
- 2026-10-21 12:20–12:22, commit 75437a6 (clean), `panel_v5` 7bb6cc0b81eb… verified, leakage tests
  8/8 at run time; `python -m src.cli freeze` (`src/champion.py`).
- exp-0018 refitted through the final run's building blocks (train + validation pool; Platt map from
  the inner model's scores of calibration half A of 2023/24). **Bit-identical to the one-shot run**:
  every holdout key, p, p_raw and y (y7 23,727 rows, y28 23,627) and both Platt maps (y7 a
  −4.137957226124922, b 0.9460885241456899; y28 a −2.7458393131851517, b 0.9364710172481375).
- Stored (`data/final/champion/`): fitted objects per horizon (parameter fingerprints y7 38e00fc1ba3f…,
  y28 aa01e16a0c4d…; deployed-pipeline model hash 688b63775081…), calibration half A/B scores
  (calibrator_A ∘ inner model), manifest.

### exp-0024 — Operational characteristics (alarm burden), in-season holdout
- Date / Stage: 2026-10-21 / Stage 7 (analysis only).
- Question (descriptive, no hypothesis): at operating thresholds fixed in advance on the calibration
  split (precision ≈ 1.5×, 2×, 3× prevalence), what alarm burden, precision, recall, injuries caught
  or missed and lead time does the frozen champion deliver on the in-season holdout?
- Change vs champion: none — frozen exp-0018 (decision 19); `configs/exp-0024.yaml`. Thresholds
  selected on calibration half B only (rule: decision 21) and committed in 03595fe (12:24) **before any
  holdout row was read** (the code refuses otherwise); analysis code 6d4581f; reported run at 4d143f5
  (clean; an earlier run at a dirty tree gave identical numbers).
- Data / split / seed: `panel_v5` 7bb6cc0b81eb…; final-run holdout predictions, test group 2024/25,
  in-season (Sep–Apr): y7 15,435 player-weeks, y28 15,355; 565 players; 35 Sundays, ≈ 440 at-risk
  players per Sunday; seed 20261020, 1000 player-cluster resamples.
- Thresholds (calibration half B 2023/24, in-season; y7 29,675 rows, prevalence 0.0204; y28 29,495 rows,
  0.0772; all targets reached): y7 0.0178 / 0.0268 / 0.0513, y28 0.0671 / 0.1014 / 0.1801.
- Results (holdout; prevalence y7 0.0245, 378 qualifying injuries; y28 0.0913, 408 injuries):

  | horizon, target | flags / Sunday | flags / week per 25 players | precision | recall | injuries caught / missed (share caught) | caught / missed per 25-player squad per 90 d | median lead (d) |
  |---|---|---|---|---|---|---|---|
  | y7, 1.5× | 227.3 [209.8, 243.8] | 12.89 [11.87, 13.81] | 0.0349 [0.0305, 0.0396] | 0.735 [0.688, 0.784] | 278 / 100 (0.735) | 5.79 [4.94, 6.72] / 2.08 [1.68, 2.50] | 5 [4, 5] |
  | y7, 2× | 95.9 [83.9, 107.8] | 5.43 [4.75, 6.12] | 0.0456 [0.0385, 0.0533] | 0.405 [0.343, 0.464] | 153 / 225 (0.405) | 3.19 [2.52, 3.91] / 4.69 [4.09, 5.32] | 5 [4, 5] |
  | y7, 3× | 11.0 [6.8, 15.5] | 0.62 [0.39, 0.87] | 0.0675 [0.0459, 0.0911] | 0.069 [0.036, 0.107] | 26 / 352 (0.069) | 0.54 [0.27, 0.86] / 7.33 [6.41, 8.25] | 4 [3, 5] |
  | y28, 1.5× | 223.3 [206.1, 239.5] | 12.73 [11.72, 13.64] | 0.1311 [0.1154, 0.1477] | 0.731 [0.682, 0.779] | 314 / 94 (0.770 [0.724, 0.814]) | 6.57 [5.59, 7.55] / 1.97 [1.59, 2.42] | 24 [24, 25] |
  | y28, 2× | 86.9 [75.1, 98.8] | 4.95 [4.25, 5.62] | 0.1661 [0.1397, 0.1937] | 0.360 [0.302, 0.416] | 176 / 232 (0.431 [0.370, 0.493]) | 3.68 [2.94, 4.49] / 4.86 [4.22, 5.48] | 24 [23, 25] |
  | y28, 3× | 11.1 [6.9, 15.5] | 0.63 [0.39, 0.88] | 0.2294 [0.1550, 0.3073] | 0.063 [0.033, 0.097] | 34 / 374 (0.083 [0.044, 0.126]) | 0.71 [0.36, 1.09] / 7.83 [6.87, 8.77] | 23 [22, 26] |

- Calibration slope: not applicable (thresholded); champion unchanged.
- Verdict: none (analysis only; no model-selection decision).
- Notes & surprises: (1) Because 2024/25 had more injuries, holdout precision exceeds the absolute
  calibration targets at 1.5× and 2×, but the lift over holdout prevalence is lower than intended:
  y7 1.42× / 1.86× / 2.76×, y28 1.44× / 1.82× / 2.51×. (2) The trade-off is steep. Catching about three
  quarters of injuries means flagging about half the squad every week (≈ 12.8 of 25). The strictest
  threshold flags about 0.6 players per 25 per week and catches 7–8% of injuries, with about one alarm
  in four correct at 28 days. (3) **Lead time is not early warning:** it is bounded by the horizon
  (1–7 / 1–28 days) and reflects score persistence. For 93–97% of caught 28-day injuries the flag was
  already on at the earliest Sunday whose window contained the onset, and 83–96% of their in-window
  weeks were flagged. (4) More qualifying injuries at 28 days (408) than at 7 days (378): an onset in
  early May still falls in the 28-day window of an April Sunday, but its 7-day row is outside the
  in-season population. Report: `reports/experiments/exp-0024.md`; figure
  `reports/figures/exp-0024_operational.png`.

### exp-0025 — Subgroup calibration of the frozen champion (holdout)
- Date / Stage: 2026-10-21 / Stage 7 (analysis only).
- Question (descriptive): within position, age band at season start and usage band, does the frozen
  champion still discriminate, is it calibrated, and does the 28-day Mondrian coverage hold?
- Change vs champion: none — frozen exp-0018; `configs/exp-0025.yaml` (commit e4eca25); definitions
  PLAN decisions 20(a), 22; reported run at 1ca5763 (clean; identical to an earlier run whose console
  print failed after the results were written); figure redrawn at f532a0c (legend placement only).
- Data / split / seed: `panel_v5` 7bb6cc0b81eb…; final-run holdout predictions (test group 2024/25);
  primary in-season (y7 15,435 / y28 15,355 player-weeks, 565 players); all weeks secondary; seed
  20261020, 1000 player-cluster resamples within each subgroup. Usage cut = median last-season
  estimated minutes of the in-season holdout player-weeks = 1,946 min (substitute for prior-28-day
  minutes, decision 20). Mondrian α = 0.10 quantiles from the final run's calibration half B (q0
  0.1156, q1 0.9605); they reproduce the final run's overall in-season coverage exactly (checked in
  code). Unknown position / age / usage: 0 player-weeks in every cell. No cell below 500 player-weeks.
- Primary results (in-season; PR-AUC and slope with 95% CI; intercept; 28-day coverage with CI):

  | subgroup | n y7 / y28 (players) | prevalence y7 / y28 | PR-AUC y7 | PR-AUC y28 | slope y7 | slope y28 | intercept y7 / y28 | coverage y28 (α 0.10) |
  |---|---|---|---|---|---|---|---|---|
  | GK | 1,814 / 1,809 (59) | 0.0105 / 0.0404 | 0.0378 [0.0169, 0.0781] | 0.1116 [0.0525, 0.2004] | 1.46 [0.54, 2.30] | 1.32 [0.49, 2.16] | +0.21 / +0.23 | 0.970 [0.947, 0.988] |
  | DEF | 5,099 / 5,071 (187) | 0.0263 / 0.1002 | 0.0431 [0.0317, 0.0603] | 0.1588 [0.1207, 0.2132] | 0.81 [0.45, 1.14] | 0.89 [0.52, 1.24] | +0.17 / +0.22 | 0.866 [0.827, 0.901] |
  | MID | 4,173 / 4,155 (147) | 0.0240 / 0.0864 | 0.0426 [0.0306, 0.0575] | 0.1436 [0.0994, 0.1997] | 1.01 [0.65, 1.36] | 1.02 [0.59, 1.47] | +0.19 / +0.17 | 0.907 [0.869, 0.938] |
  | FWD | 4,349 / 4,320 (172) | 0.0287 / 0.1069 | 0.0565 [0.0377, 0.0822] | 0.1748 [0.1292, 0.2228] | 0.89 [0.55, 1.22] | 0.87 [0.51, 1.21] | +0.25 / +0.30 | 0.846 [0.804, 0.887] |
  | age ≤ 23 | 5,254 / 5,226 (208) | 0.0225 / 0.0838 | 0.0520 [0.0304, 0.0774] | 0.1563 [0.1016, 0.2167] | 0.94 [0.60, 1.29] | 0.91 [0.54, 1.29] | +0.27 / +0.30 | 0.945 [0.925, 0.963] |
  | age 24–28 | 5,784 / 5,753 (206) | 0.0266 / 0.0994 | 0.0514 [0.0387, 0.0697] | 0.1812 [0.1361, 0.2308] | 1.09 [0.82, 1.34] | 1.18 [0.87, 1.49] | +0.24 / +0.28 | 0.878 [0.843, 0.908] |
  | age ≥ 29 | 4,397 / 4,376 (151) | 0.0241 / 0.0896 | 0.0412 [0.0282, 0.0596] | 0.1439 [0.1063, 0.1893] | 0.86 [0.52, 1.21] | 0.86 [0.53, 1.21] | +0.08 / +0.10 | 0.817 [0.770, 0.861] |
  | minutes bottom half | 7,747 / 7,699 (296) | 0.0232 / 0.0879 | 0.0440 [0.0337, 0.0634] | 0.1557 [0.1233, 0.1939] | 0.99 [0.73, 1.27] | 1.02 [0.75, 1.32] | +0.19 / +0.24 | 0.892 [0.863, 0.918] |
  | minutes top half | 7,688 / 7,656 (269) | 0.0258 / 0.0947 | 0.0490 [0.0363, 0.0642] | 0.1613 [0.1244, 0.2074] | 0.90 [0.67, 1.12] | 0.91 [0.66, 1.17] | +0.21 / +0.23 | 0.875 [0.846, 0.903] |

- Secondary (all weeks; PR-AUC [95% CI], prevalence): GK y7 0.0328 [0.0132, 0.0715] (0.0077), y28
  0.1005 [0.0456, 0.1777] (0.0310); DEF 0.0394 [0.0295, 0.0559] (0.0209), 0.1447 [0.1112, 0.1938]
  (0.0806); MID 0.0401 [0.0286, 0.0533] (0.0188), 0.1356 [0.0971, 0.1806] (0.0714); FWD 0.0508 [0.0346,
  0.0752] (0.0217), 0.1583 [0.1203, 0.1996] (0.0810); ≤ 23 0.0483 [0.0293, 0.0695] (0.0174), 0.1473
  [0.0994, 0.2033] (0.0659); 24–28 0.0466 [0.0362, 0.0631] (0.0206), 0.1625 [0.1260, 0.2051] (0.0779);
  ≥ 29 0.0378 [0.0270, 0.0544] (0.0192), 0.1330 [0.1015, 0.1724] (0.0736); minutes bottom 0.0399
  [0.0315, 0.0553] (0.0181), 0.1417 [0.1127, 0.1731] (0.0694); top 0.0451 [0.0331, 0.0599] (0.0201),
  0.1482 [0.1163, 0.1851] (0.0758).
- Calibration slope: champion unchanged; subgroup slopes above.
- Verdict: none (analysis only; no model-selection decision).
- Notes & surprises: (1) Discrimination holds in every subgroup: every PR-AUC CI lies above its
  subgroup prevalence, at both horizons. (2) No subgroup slope CI excludes 1. Goalkeepers are the
  least certain (59 players; slope 1.46 / 1.32 with CIs from about 0.5 to 2.2). (3) The level is
  under-predicted in **every** subgroup (intercepts +0.08 to +0.30; mean predicted below prevalence
  throughout). It is least under-predicted for players aged 29 and over (+0.08 / +0.10). The 2024/25 rise in
  injuries is broad, not confined to one group. (4) Class-conditional conformal coverage is **not**
  subgroup-conditional: 28-day coverage ranges from 0.817 (age ≥ 29; CI excludes 0.90) and 0.846
  (forwards; CI excludes 0.90) to 0.945 (age ≤ 23) and 0.970 (goalkeepers), around the overall
  0.884. Risk tiers are least reliable for older players and forwards. (5) Usage band is the
  last-season-minutes substitute, not recent workload (decision 20). Report:
  `reports/experiments/exp-0025.md`; figure `reports/figures/exp-0025_subgroups.png`.

### exp-0026 — Per-season stability of the champion pipeline (development seasons only)
- Date / Stage: 2026-10-21 12:31–12:44 / Stage 7 (analysis only).
- Question (descriptive): how stable is the champion pipeline's discrimination season by season
  over 2015/16–2023/24, and how did the COVID-19 seasons behave?
- Change vs champion: none — the exp-0018 pipeline refitted per season as in the rolling-origin
  protocol (decision 23); `configs/exp-0026.yaml`, commit f450db2 (clean); leakage tests 8/8 and
  `panel_v5` hash verified at run time; figure redrawn at 8731509 (legend placement only).
- Data / split / seed: `panel_v5` 7bb6cc0b81eb…; for each season S the validation player group of S
  (test group never read; holdout season never read), training on train-group seasons 2014/15 … S − 1
  with purge + 7-day embargo; Platt map from the inner model on half A of S − 1 for S ≥ 2016/17
  (2015/16 raw: no inner season; PR-AUC identical by construction — checked: |ΔPR-AUC raw vs
  calibrated| = 0 for every calibrated season, all Platt b > 0). **Reproduction:** 2021/22–2023/24
  equal the stored exp-0018 predictions bit for bit, both horizons. Seed 20261020, 1000 resamples.
- Results, in-season (primary; PR-AUC [95% CI], lift = PR-AUC / prevalence, calibration slope [CI]):

  | season | train seasons | y7 prevalence | y7 PR-AUC | lift | y7 slope | y28 prevalence | y28 PR-AUC | lift | y28 slope |
  |---|---|---|---|---|---|---|---|---|---|
  | 2015/16 | 1 | 0.0187 | 0.0341 [0.0280, 0.0440] | 1.82 | — | 0.0679 | 0.1163 [0.0960, 0.1399] | 1.71 | — |
  | 2016/17 | 2 | 0.0217 | 0.0370 [0.0311, 0.0460] | 1.70 | 0.82 [0.64, 1.02] | 0.0809 | 0.1307 [0.1127, 0.1511] | 1.62 | 0.82 [0.65, 0.99] |
  | 2017/18 | 3 | 0.0219 | 0.0333 [0.0286, 0.0408] | 1.52 | 0.85 [0.67, 1.09] | 0.0810 | 0.1225 [0.1071, 0.1438] | 1.51 | 0.92 [0.73, 1.13] |
  | 2018/19 | 4 | 0.0240 | 0.0588 [0.0472, 0.0756] | 2.45 | 1.17 [1.00, 1.35] | 0.0880 | 0.1922 [0.1570, 0.2285] | 2.18 | 1.22 [1.04, 1.40] |
  | 2019/20 (COVID) | 5 | 0.0196 | 0.0409 [0.0339, 0.0507] | 2.08 | 0.95 [0.79, 1.10] | 0.0722 | 0.1412 [0.1191, 0.1683] | 1.96 | 0.90 [0.75, 1.07] |
  | 2020/21 (COVID) | 6 | 0.0220 | 0.0395 [0.0343, 0.0459] | 1.80 | 1.27 [1.06, 1.48] | 0.0814 | 0.1459 [0.1255, 0.1680] | 1.79 | 1.13 [0.93, 1.32] |
  | 2021/22 | 7 | 0.0203 | 0.0469 [0.0356, 0.0609] | 2.31 | 0.98 [0.75, 1.20] | 0.0739 | 0.1525 [0.1185, 0.1872] | 2.06 | 0.96 [0.73, 1.18] |
  | 2022/23 | 8 | 0.0171 | 0.0385 [0.0267, 0.0542] | 2.26 | 0.95 [0.77, 1.10] | 0.0641 | 0.1207 [0.0947, 0.1510] | 1.88 | 0.90 [0.72, 1.09] |
  | 2023/24 | 9 | 0.0204 | 0.0393 [0.0333, 0.0488] | 1.92 | 1.11 [0.91, 1.33] | 0.0787 | 0.1510 [0.1292, 0.1780] | 1.92 | 1.32 [1.09, 1.57] |

  In-season player-weeks per season 13,394–14,454 (validation group, ≈ 400–450 players).
- Results, all weeks (secondary; PR-AUC [CI] y7 / y28; slope y7 / y28): 2015/16 0.0336 [0.0279, 0.0422] /
  0.1085 [0.0911, 0.1281]; 2016/17 0.0344 [0.0294, 0.0414] / 0.1217 [0.1064, 0.1395], 0.82 / 0.85;
  2017/18 0.0321 [0.0277, 0.0382] / 0.1166 [0.1018, 0.1333], 0.97 / 0.99; 2018/19 0.0517 [0.0416,
  0.0648] / 0.1736 [0.1475, 0.2045], 1.01 / 1.11; **2019/20 0.0363 [0.0308, 0.0434] / 0.1263 [0.1090,
  0.1459], 0.69 [0.57, 0.81] / 0.66 [0.55, 0.77]**; 2020/21 0.0354 [0.0309, 0.0413] / 0.1344 [0.1174,
  0.1552], 1.29 [1.10, 1.49] / 1.15 [0.99, 1.32]; 2021/22 0.0431 [0.0325, 0.0549] / 0.1443 [0.1141,
  0.1748], 1.08 / 1.04; 2022/23 0.0357 [0.0261, 0.0482] / 0.1191 [0.0942, 0.1467], 0.88 / 0.91; 2023/24
  0.0365 [0.0306, 0.0441] / 0.1357 [0.1148, 0.1590], 1.19 / 1.33.
- Platt maps per season (b, y7 / y28): 2016/17 0.81 / 0.81, 2017/18 0.73 / 0.73, 2018/19 0.91 / 0.91,
  2019/20 0.94 / 0.98, **2020/21 0.66 / 0.76** (learned on the COVID season 2019/20), 2021/22 0.89 / 0.88,
  2022/23 1.00 / 0.99, 2023/24 0.87 / 0.80.
- Verdict: none (analysis only; no model-selection decision).
- Notes & surprises: (1) **Discrimination is stable in kind, not in size.** In-season lift ranges
  1.52–2.45 (y7, median 1.92) and 1.51–2.18 (y28, median 1.88); every season's PR-AUC CI lies above
  its prevalence. The holdout (y7 1.89×, y28 1.73×) is typical of the development seasons, not an
  outlier. 2018/19 is the best season (y28 0.192, 2.18×); 2017/18 the weakest (1.51×). The first
  three seasons are fitted on 1–3 training seasons (lift 1.51–1.82), so season difficulty and
  training size are confounded there. (2) **COVID-19.** Football stopped from mid-March 2020 (Ligue 1
  did not resume; the other leagues resumed May–June and finished in July–August 2020), and 2020/21
  ran on a compressed calendar without crowds. *Discrimination did not degrade*: 2019/20 in-season
  lift 2.08 / 1.96, 2020/21 1.80 / 1.79, both inside the range of the other seasons. 2019/20 in-season
  prevalence was among the lowest (y28 0.072) because March–April 2020 had no matches.
  *Calibration did degrade:* 2019/20 all weeks has the only slopes clearly below 1 in the series
  (0.69 / 0.66, CIs exclude 1). That is consistent with the calendar features (month, weeks into the
  season) mis-describing a season whose suspension and summer restart moved matches into the
  normally quiet months; the in-season slopes of that season are fine (0.95 / 0.90). The disruption
  then **propagated through the per-season Platt refit**: the map learned on 2019/20 has the flattest
  b of the series (0.66 / 0.76), and 2020/21 was under-confident (in-season y7 slope 1.27 [1.06, 1.48]).
  (3) Slope CIs exclude 1 in non-COVID seasons too (2018/19 y28 1.22; 2023/24 y28 1.32), as seen in
  development (open decision 12): season-to-season calibration drift is the norm, not a COVID artefact.
  Report: `reports/experiments/exp-0026.md`; figure `reports/figures/exp-0026_seasons.png`; predictions
  `data/experiments/exp-0026/predictions.parquet`.

### exp-0027 — Per-league breakdown of the frozen champion (holdout)
- Date / Stage: 2026-10-21 / Stage 7 (analysis only).
- Question (descriptive): how do discrimination, calibration and 28-day Mondrian coverage vary
  across the five leagues on the holdout?
- Change vs champion: none — frozen exp-0018; `configs/exp-0027.yaml`, commit 16d4004 (clean); same
  metric set and rules as exp-0025 (decision 22); league = competition of the registration at t.
- Data / split / seed: `panel_v5` 7bb6cc0b81eb…; final-run holdout predictions, in-season primary,
  all weeks secondary; seed 20261020, 1000 player-cluster resamples within each league; final-run
  Mondrian quantiles (α = 0.10). No unknown league; no cell below 500 player-weeks. In-season point
  estimates equal the final run's league strata.
- Primary results (in-season):

  | league | n y7 / y28 (players) | prevalence y7 / y28 | PR-AUC y7 | PR-AUC y28 | slope y7 | slope y28 | intercept y7 / y28 | mean predicted y28 | coverage y28 (α 0.10) |
  |---|---|---|---|---|---|---|---|---|---|
  | Premier League (GB1) | 2,880 / 2,857 (109) | 0.0278 / 0.1043 | 0.0579 [0.0303, 0.1013] | 0.1737 [0.1093, 0.2548] | 0.88 [0.35, 1.36] | 0.83 [0.29, 1.34] | +0.34 / +0.39 | 0.0734 | 0.917 [0.881, 0.950] |
  | LaLiga (ES1) | 3,172 / 3,161 (108) | 0.0158 / 0.0566 | 0.0587 [0.0260, 0.1010] | 0.1580 [0.0898, 0.2262] | 1.41 [1.03, 1.95] | 1.46 [1.05, 2.03] | −0.27 / −0.32 | 0.0755 | 0.877 [0.829, 0.920] |
  | Serie A (IT1) | 3,567 / 3,524 (144) | 0.0278 / 0.1061 | 0.0418 [0.0316, 0.0545] | 0.1639 [0.1190, 0.2122] | 0.78 [0.48, 1.07] | 0.87 [0.54, 1.20] | +0.24 / +0.32 | 0.0803 | 0.831 [0.783, 0.879] |
  | Ligue 1 (FR1) | 2,908 / 2,905 (111) | 0.0213 / 0.0795 | 0.0389 [0.0243, 0.0663] | 0.1475 [0.0912, 0.2352] | 0.87 [0.33, 1.33] | 0.93 [0.33, 1.51] | +0.37 / +0.39 | 0.0556 | 0.945 [0.913, 0.972] |
  | Bundesliga (L1) | 2,908 / 2,908 (104) | 0.0299 / 0.1100 | 0.0525 [0.0357, 0.0889] | 0.1642 [0.1197, 0.2439] | 1.00 [0.62, 1.43] | 0.96 [0.57, 1.41] | +0.27 / +0.30 | 0.0842 | 0.861 [0.810, 0.908] |

- Secondary (all weeks; PR-AUC [CI], prevalence; y7 then y28): GB1 0.0528 [0.0292, 0.0914] (0.0200),
  0.1573 [0.0993, 0.2368] (0.0749); ES1 0.0523 [0.0246, 0.0876] (0.0126), 0.1457 [0.0906, 0.1974]
  (0.0486); IT1 0.0394 [0.0300, 0.0512] (0.0222), 0.1505 [0.1132, 0.1916] (0.0846); FR1 0.0362
  [0.0239, 0.0589] (0.0173), 0.1386 [0.0914, 0.2162] (0.0670); L1 0.0479 [0.0328, 0.0805] (0.0230),
  0.1484 [0.1084, 0.2187] (0.0865).
- Verdict: none (analysis only; no model-selection decision).
- Notes & surprises: (1) Every league's PR-AUC CI lies above its prevalence at both horizons, though
  the Premier League's y28 lower bound only just clears it (0.109 vs 0.104). With ≈ 100–150 players per
  league, the CIs overlap widely, so no league ranks reliably above another. (2) **LaLiga is the
  calibration outlier.** It has the lowest injury rate (y28 0.057 vs 0.080–0.110 elsewhere), it is the
  only league whose level is over-predicted (intercept −0.27 / −0.32; mean predicted 0.076 vs 0.057),
  and it is the only league with slope CIs excluding 1 (1.41 / 1.46: predictions too compressed).
  The league indicator shifts the level but evidently not enough. A lower public reporting density for
  Spanish clubs is a possible explanation, not tested here. The other four leagues are
  under-predicted (+0.24 to +0.39). (3) 28-day coverage ranges from 0.831 (Serie A; CI excludes 0.90)
  to 0.945 (Ligue 1). As in exp-0025, class-conditional conformal sets do not give league-conditional
  coverage. Report: `reports/experiments/exp-0027.md`; figure `reports/figures/exp-0027_leagues.png`.

### exp-0028 — Current-practice workload-ratio baseline vs the frozen champion (holdout)
- Date / Stage: 2026-10-21 / Stage 7 (baseline comparison; **not a challenger** — no promotion
  decision whatever the outcome).
- Question (descriptive): on the same holdout player-weeks, how does a workload-ratio rule of the
  kind used in practice compare with the frozen champion in PR-AUC and net benefit?
- **Deviation from the brief (decision 20, recorded before the run):** the specified 7:28-day ACWR
  (acute = minutes in the last 7 days, chronic = mean weekly minutes over the last 28 days) **cannot be
  computed**: the datalake holds season totals per competition, no dated appearances. It is
  implemented as specified on a dated minutes table (`src/acwr.py`, 5 unit tests incl. window edges
  and no-future-data) but **not scored**. Scored instead, labelled as a substitute: the season-scale
  analogues defined in Stage 2 before any loop ran (`src/load.py`, columns of `panel_v5`): uncoupled =
  last-season minutes / mean of the two seasons before (primary), coupled = last-season minutes /
  mean of the last three seasons (secondary); both undefined when the chronic term is < 90 minutes
  (ranked lowest, never flagged); flag = ratio > 1.5. **These are not ACWR**; they measure a
  season-to-season jump in playing time.
- Change vs champion: none — frozen exp-0018; `configs/exp-0028.yaml`, commit 32172c7 (clean).
- Data / split / seed: `panel_v5` 7bb6cc0b81eb…; the final run's holdout rows (same player-weeks as
  the champion), in-season primary, all weeks secondary; paired player-cluster bootstrap, 1000
  resamples, seed 20261020.
- Primary metric (in-season; PR-AUC [95% CI]; Δ = champion − baseline [95% CI]):

  | score | y7 PR-AUC (prevalence 0.0245) | Δ y7 | y28 PR-AUC (prevalence 0.0913) | Δ y28 |
  |---|---|---|---|---|
  | champion exp-0018 | 0.0463 [0.0380, 0.0577] | — | 0.1580 [0.1335, 0.1877] | — |
  | season ratio, uncoupled (continuous) | 0.0227 [0.0194, 0.0266] | +0.0236 [+0.0157, +0.0330] | 0.0839 [0.0723, 0.0967] | +0.0741 [+0.0503, +0.1004] |
  | season ratio, uncoupled > 1.5 (flag) | 0.0237 [0.0207, 0.0269] | +0.0226 [+0.0151, +0.0322] | 0.0886 [0.0779, 0.0993] | +0.0694 [+0.0489, +0.0947] |
  | season ratio, coupled (continuous) | 0.0228 [0.0195, 0.0268] | +0.0235 [+0.0154, +0.0328] | 0.0842 [0.0729, 0.0973] | +0.0737 [+0.0498, +0.1005] |
  | season ratio, coupled > 1.5 (flag) | 0.0239 [0.0210, 0.0269] | +0.0225 [+0.0151, +0.0321] | 0.0892 [0.0792, 0.0997] | +0.0688 [+0.0480, +0.0943] |

- Secondary: flags (in-season) — uncoupled > 1.5 flags 20.4% of player-weeks, precision 0.0200 (y7) /
  0.0753 (y28), recall 0.167 / 0.168; coupled > 1.5 flags 12.0%, precision 0.0168 / 0.0670, recall
  0.082 / 0.088. Undefined ratio: 1.1% (uncoupled), 0.3% (coupled) of rows; on defined rows only,
  Δ y7 +0.0237 [+0.0153, +0.0333] / +0.0235 [+0.0149, +0.0332], y28 +0.0741 [+0.0497, +0.0995] /
  +0.0737 [+0.0490, +0.0987]. All weeks: champion 0.0423 / 0.1445 vs uncoupled 0.0180 [0.0159, 0.0209]
  / 0.0679 [0.0600, 0.0769] (Δ +0.0244 [+0.0172, +0.0325] / +0.0766 [+0.0563, +0.0974]); flag 0.0186 /
  0.0707; coupled 0.0182 / 0.0689, flag 0.0186 / 0.0709. Decision curve: the flag rule never beats
  both defaults on pt 0.01–0.30 (largest margin over max(treat all, treat none): y7 −0.0021 /
  −0.0016, y28 −0.0047 / −0.0044 in-season; all weeks −0.0008 / −0.0008, −0.0040 / −0.0033); overlaid
  dashed on `reports/figures/fig11_holdout_decision_curves.png` (uncoupled flag; the pre-overlay
  figure is reproducible from `holdout_figures()` without `overlay`).
- Calibration slope: not applicable (ratios are not probabilities); champion unchanged.
- Verdict: none (baseline comparison; no promotion decision).
- Notes & surprises: (1) The season-scale workload ratios carry **no** signal for injury in the next 7
  or 28 days: their PR-AUC sits at or slightly below prevalence, and flagged weeks are *less* often
  injury weeks than average (precision 0.0200 vs 0.0245; 0.0753 vs 0.0913). A jump in playing time
  over earlier seasons mostly marks players breaking into a side (plausibly younger, with less
  injury history); not tested here. (2) The champion beats every substitute at both horizons and in
  both populations, with CIs far from 0. (3) This is **not** evidence about the 7:28-day ACWR used in
  practice, which needs dated minutes. The comparison against current practice stays open until a
  dated exposure source is approved (`DATA_CARD.md`) — logged as future work. Report:
  `reports/experiments/exp-0028.md`.

### exp-0029 — Prospective logging harness for the frozen champion
- Date / Stage: 2026-10-21 / Stage 7 (harness; no evaluation, no labels read).
- Question: can the frozen champion score every active player as of a given date with exactly the
  training-time feature code, and log the scores append-only with model and code versions?
- Change vs champion: none — frozen artifacts of task 7.0 (model hash 688b63775081…), never refitted;
  `src/prospective.py`, CLI `python -m src.cli prospective --date YYYY-MM-DD`; `configs/exp-0029.yaml`;
  commit 70c2688 (clean). Design: decision 24.
- Behaviour: t = Sunday on or before the date; refused if t is after the snapshot
  (`labels.observed_until` 2026-09-28) or the snapshot lacks t's top-league club table (covered through
  2025/26); features from the unchanged `build_panel` over t's season, rows at t = all active players;
  output appended to `data/prospective/<date>.csv` with player_id, week, p7, p28, model_hash, git_hash,
  scored_at (UTC), features_sha256. The header is written once, existing lines are never rewritten,
  and the file is refused if its header differs.
- Tests (`tests/test_prospective.py`, 6; `tests/test_acwr.py`, 5): Sunday mapping; snapshot guard;
  parameter fingerprint deterministic and sensitive to a 1e-9 change of the Platt intercept; scoring
  equals the frozen model; append-only (second write leaves the first block byte-identical; one header;
  exact float round trip); foreign-file refusal; **feature parity**: the harness rebuilt Sunday
  2023-10-15 from raw data and reproduced the feature store's 2,176 rows exactly (all feature columns,
  same players). Full suite: **78 passed** (was 67), leakage tests green.
- Smoke run: `--date 2025-10-22` → t 2025-10-19: **2,202 active players** scored in 30 s (the feature
  store had 2,170 at 2024-10-20); p7 mean 0.0219 (median 0.0197, range 0.0022–0.1566), p28 mean 0.0833
  (median 0.0763, range 0.0097–0.4484); one model hash, git hash 70c2688…, features sha256
  12e0ea196299…; `--date 2026-10-21` refused (after the snapshot).
- Verdict: none (harness).
- Notes & surprises: (1) The snapshot was taken on 2026-09-29, so scoring a 2025/26 date replays the
  logging path retrospectively: the features use only information dated ≤ t, but records added later
  to the datalake are visible. A live prospective log needs a datalake refresh before each run. (2)
  2025/26 outcomes in the snapshot were not read. A later prospective evaluation of these logs would
  be a new, separately pre-registered analysis. (3) The harness scores all player groups, including
  players the model was trained on; it is a deployment log, not an evaluation set.

### stage-7 summary — practitioner evidence pack
- Champion unchanged and frozen (exp-0018 + Platt; bit-identical to the one-shot run, task 7.0). No
  model-selection decision taken; all six analyses reported as they came out.
- Operating points (exp-0024): thresholds fixed on 2023/24 give lift 1.4–2.8× on the holdout. A
  25-player squad gets ≈ 12.8 / 5.2 / 0.6 flags per week for 74–77% / 40–43% / 7–8% of injuries caught.
  Lead time reflects score persistence, not early warning.
- Subgroups (exp-0025) and leagues (exp-0027): discrimination holds everywhere (all PR-AUC CIs above
  prevalence). The level is under-predicted in every subgroup and in every league except LaLiga,
  which is over-predicted with compressed predictions (slope 1.41 / 1.46, CIs exclude 1). Mondrian
  coverage is not subgroup-conditional: 0.82 (age ≥ 29), 0.83 (Serie A), 0.85 (forwards) to 0.97
  (goalkeepers).
- Per-season stability (exp-0026, development only): in-season lift 1.5–2.5 (y7) and 1.5–2.2 (y28);
  the holdout is typical. COVID-19 did not hurt ranking. It broke calibration in 2019/20 all weeks
  (slopes 0.69 / 0.66), and the per-season Platt refit carried the distortion into 2020/21.
- Current practice (exp-0028): the 7:28-day ACWR is not computable from public season totals. The
  season-scale workload-ratio substitute has no skill (PR-AUC ≤ prevalence; flag never beats the
  default strategies); the champion beats it by +0.024 (y7) and +0.074 (y28) in-season, CIs far from 0.
- Prospective harness (exp-0029): built, tested (feature parity with the training panel), smoke-run.
- Future work added: dated exposure source for a true ACWR comparison (needs owner approval and a
  `DATA_CARD.md` entry); subgroup-conditional (e.g. Mondrian by league or age) conformal calibration;
  pre-registered prospective evaluation of logged scores after a datalake refresh.

### exp-0025 addendum — where the subgroup coverage shortfall comes from (README check, 2026-10-21)
- Descriptive split of exp-0025's 28-day Mondrian coverage (α = 0.10, final-run quantiles q0 0.1156,
  q1 0.9605; in-season holdout) into injured and uninjured weeks; same rows, no new model, no decision.
  Computed while checking the public README against the log.

  | subgroup | coverage (all) | injured weeks | uninjured weeks | share with p > q0 (uninjured excluded from set) | share with p < 1 − q1 (injury excluded) |
  |---|---|---|---|---|---|
  | age ≥ 29 | 0.817 | 0.929 | 0.806 | 0.207 | 0.212 |
  | FWD | 0.846 | 0.907 | 0.838 | 0.175 | 0.166 |
  | DEF | 0.866 | 0.955 | 0.856 | 0.157 | 0.113 |
  | Serie A | 0.831 | 0.906 | 0.822 | 0.192 | 0.206 |
  | age ≤ 23 | 0.945 | 0.831 | 0.956 | 0.053 | 0.297 |
  | GK | 0.970 | 0.548 | 0.988 | 0.012 | 0.781 |

- Reading: where coverage falls short (age ≥ 29, forwards, defenders, Serie A), the misses are
  **uninjured** weeks placed in the high-risk-only set (false alarms). The groups whose **injuries**
  the tiers miss are goalkeepers (injured coverage 0.55) and players aged ≤ 23 (0.83); their overall
  coverage looks high only because uninjured weeks dominate. Defenders' overall coverage CI [0.827,
  0.901] includes 0.90.
