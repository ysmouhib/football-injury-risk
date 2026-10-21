# Stage 5 — Interpretability and science output

All numbers are descriptive analyses of the validation season 2023/24 (main origin) unless stated;
none of them is a promotion decision. Source data: `data/experiments/exp-0018/interpret/`,
`conformal.json`, `decision_curves.json`; design pre-registered as PLAN amendment A11. Figures in
`reports/figures/`.

**Explained models.** Champion exp-0018 (class-weighted logistic regression, v0 + injury history +
context, Platt calibration); best tree challenger of the classification arm exp-0005 (LightGBM,
v0 + history); survival champion exp-0013 (Random Survival Forest, 28-day risk). Each was refitted at
the main origin through the training code and reproduced its stored validation predictions
bit-identically (22,346 / 22,234 / 22,384 rows) before being explained.

## 1. How the champion improved (Figure 1)

`fig1_trajectory.png` — main-origin PR-AUC with player-cluster 95% CIs for every classification
experiment, champion of record as a step line. Two promotions moved discrimination: the injury
history family (exp-0004: y28 0.0778 → 0.1137) and the context family (exp-0018: → 0.1357). Every
other challenger sits inside the champion's interval. Cumulative gain vs the Stage-1 baseline:
y7 +0.0158, y28 +0.0579 (prevalence 0.016 / 0.062).

## 2. The context gain is a calendar effect (Figure 9)

`fig9_context_in_season.png`. Restricted to in-season weeks (September–April), the context family adds
nothing: y28 PR-AUC 0.1510 vs 0.1502 for the history-only model, Δ CI [−0.0071, +0.0089]; y7 0.0393
vs 0.0387, CI [−0.0013, +0.0030]. On all weeks the gain is +0.0219 [+0.0140, +0.0312]. The reason is
in the outcome itself: 28-day injury rates are lowest for windows starting in May–June (2.1–2.7% in
the development seasons, when the window covers the summer break) and July (4.4%), and highest from
September to January (≈ 8%). `season_month` (21% of the champion's attribution) mostly encodes
*whether there is football in the next 28 days*. The champion's headline improvement over exp-0004
is therefore about timing, not about identifying which players get injured; within the playing
season, injury history and lagged exposure carry the signal.

## 3. What drives predicted risk (Figures 4–5)

`fig4_shap_global.png` — exact interventional SHAP on the calibrated log-odds (additivity error
1.8e-15; identical to `shap.LinearExplainer`). Share of mean |SHAP|, 28-day risk:

| class | share (y28) | share (y7) | largest members (sign of effect) |
|---|---|---|---|
| modifiable — exposure / load | 22% | 30% | minutes last season (+), starts last season (−), appearances last season (−) |
| partially modifiable — return to play | 8% | 8% | days since return (− : risk falls with time since return), returned within 28 days (+) |
| non-modifiable | 70% | 62% | season month, weeks into season (+), league, position, no prior injury (−), injury counts (+) |

The opposite signs of minutes (+) and appearances/starts (−) are partly collinearity: for a fixed
number of appearances, more minutes means more full games. They should not be read as separate
causal levers. The classification of factors is associational; "modifiable" means a quantity staff
can influence, not a demonstrated causal effect.

`fig5_shap_local.png` — three anonymised player-weeks: the highest-risk week (risk 37.2%, driven by
two-year and career injury counts and a hamstring history; no injury followed), a median week
(5.1%), and the highest-risk week that was followed by an injury (21.8%, driven by career and recent
injury counts, season month and last season's minutes).

## 4. Dose-response and the workload–injury paradox (Figure 6)

`fig6_dose_response.png` — centred partial dependence (log-odds) with player-cluster bands; the
logistic champion is linear by construction, so the shape question rests on the two tree models.
Pre-registered criterion (A11): "U/J-shaped" if the minimum lies strictly inside the 10th–90th
percentile and both the 5th and 95th percentile values exceed it by ≥ 0.05 log-odds.

- **Chronic load (minutes last season): monotone increasing in all three models, no U/J shape.**
  End-to-end rise: LightGBM +0.13, survival forest +0.10, logistic +0.78 (the logistic slope is
  inflated by the offsetting appearances/starts terms). LightGBM rises mostly below ≈ 1,000 minutes
  and is roughly flat above; the survival forest rises gradually up to ≈ 3,000 minutes.
- Minutes two seasons ago: flat (≤ 0.05).
- Appearances last season: LightGBM rises to ≈ 40 appearances, then falls (inverted U, not U/J);
  the survival forest rises monotonically.
- Age: risk falls with age given injury history (LightGBM −0.63 end to end) — consistent with a
  survivor effect among players who remain at top-5 clubs into their thirties.
- Days since return: steep fall over the first ≈ 500 days in both tree models — the re-injury window.

With season-level exposure, the data do not reproduce the U/J-shaped chronic-load relationship
reported from weekly GPS-based studies. The proxy is coarse (a season total, estimated from
appearances), so this is an absence of evidence at this resolution, not evidence against the
weekly-load hypothesis.

## 5. Positions (Figure 7)

`fig7_positions.png` — 28-day, 2023/24. PR-AUC exceeds prevalence in every position: goalkeepers
0.143 vs 0.023, defenders 0.161 vs 0.074, midfielders 0.128 vs 0.066, attackers 0.128 vs 0.064.
Averaged over the three origins: attack 0.163, defence 0.131, midfield 0.122, goalkeepers 0.085.
Calibration is weakest for goalkeepers (mean slope 1.53 over origins: predictions too compressed).
Injury sites of the 28-day injuries differ by position: hamstring 22% of midfielders' injuries vs
6% of goalkeepers'; groin/hip 19% and upper limb 12% for goalkeepers. Attribution is similar across
outfield positions (season month, last season's minutes, weeks into season, days since return); for
goalkeepers the position term itself dominates.

## 6. Decision utility and risk tiers (Figures 2, 3, 8)

- `fig2_reliability.png` — deciles of predicted risk vs observed rate for the calibrated champion and
  the survival champion at 7 and 28 days.
- `fig3_decision_curves.png` — the champion beats both treat-all and treat-none for thresholds
  1–14% (28 days) and 1–3% (7 days); peak margin +0.019 net benefit at 6% (28 days).
- `fig8_conformal_tiers.png` — class-conditional conformal tiers at α = 0.10: 28 days, low tier 31%
  of player-weeks with 2.0% injured, uncertain 62% with 6.9%, high 8% with 18.5% (coverage 93.3%);
  7 days, 0.4% / 1.9% / 4.6%.

## 7. Not done

- **SoccerMon transfer check (5.4): skipped as not cheap.** The raw SoccerMon release is not local
  (the `pmsys` README requires a download from OSF); the local `pmsys` tables are third-party
  derivatives from student theses with undocumented processing (including "manipulated" variants and
  merged performance scores) on two elite *female* teams. A transfer check on them would be weak
  evidence; a proper check needs the raw release, a DATA_CARD entry and its own design.
- Palette validation: `scripts/validate_palette.js` could not run (Node.js is not installed); the
  figures use the reference palette's documented, pre-validated slots unchanged (slots 1–3 for
  multi-series charts, adjacent slots for the one stacked bar, the blue/red diverging pair and the
  blue ordinal ramp).
