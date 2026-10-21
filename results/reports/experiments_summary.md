# Experiment summary

Aggregation: main; classification decision labels: y7, y28; survival decision metric: C^td (main origin).

## Arm A — classification

| id | title | vs | PR-AUC y7 | PR-AUC y28 | run verdict | loop decision |
|---|---|---|---|---|---|---|
| exp-0001 | Class-weighted logistic regression, v0 features | — | 0.0207 | 0.0778 | BASELINE | BASELINE |
| exp-0002 | Class-weighted LightGBM, v0 features | exp-0001 | 0.0225 | 0.0803 | INCONCLUSIVE | REJECT |
| exp-0003 | Class-weighted LightGBM, v0 features (seed repeat of exp-0002) | exp-0001 | 0.0214 | 0.0808 | INCONCLUSIVE (repeat of exp-0002) | REJECT — repeat verdict |
| exp-0004 | Champion + injury-history family | exp-0001 | 0.0307 | 0.1137 | PROMOTE | PROMOTE |
| exp-0005 | LightGBM on v0+history (model-family re-challenge) | exp-0004 | 0.0309 | 0.1133 | REJECT | REJECT |
| exp-0006 | Champion + season-load derivative family | exp-0004 | 0.0310 | 0.1161 | INCONCLUSIVE | REJECT |
| exp-0007 | Champion + season-load derivative family (seed repeat of exp-0006) | exp-0004 | 0.0310 | 0.1161 | INCONCLUSIVE (repeat of exp-0006) | REJECT — pooled bootstrap y7 [-0.0004, 0.0010], y28 [-0.0011, 0.0063] |
| exp-0008 | Champion + context family | exp-0004 | 0.0365 | 0.1357 | REJECT | REJECT |
| exp-0009 | Champion + decline-proxy family | exp-0004 | 0.0309 | 0.1149 | INCONCLUSIVE | REJECT |
| exp-0010 | Champion + decline-proxy family (seed repeat of exp-0009) | exp-0004 | 0.0309 | 0.1149 | INCONCLUSIVE (repeat of exp-0009) | REJECT — pooled bootstrap y7 [-0.0014, 0.0021], y28 [-0.0031, 0.0060] |
| exp-0011 | Champion trained from 2016/17 (data-scope loop) | exp-0004 | 0.0311 | 0.1150 | INCONCLUSIVE | REJECT |
| exp-0012 | Champion trained from 2016/17 (seed repeat of exp-0011) | exp-0004 | 0.0311 | 0.1150 | INCONCLUSIVE (repeat of exp-0011) | REJECT — pooled bootstrap y7 [-0.0001, 0.0011], y28 [-0.0002, 0.0032] |
| exp-0016 | Champion + Platt calibration layer | exp-0004 | 0.0307 | 0.1137 | PROMOTE | PROMOTE |
| exp-0017 | Champion with isotonic instead of Platt calibration | exp-0016 | 0.0295 | 0.1106 | REJECT | REJECT |
| exp-0018 | Calibrated champion + context family (re-test of exp-0008) | exp-0016 | 0.0365 | 0.1357 | PROMOTE | PROMOTE |
| exp-0019 | Champion with intercept-only calibration | exp-0018 | 0.0365 | 0.1357 | REJECT | REJECT |
| exp-0020 | Champion with intercept-only calibration (re-challenge under decision 16) | exp-0018 | 0.0365 | 0.1357 | INCONCLUSIVE | REJECT |
| exp-0021 | Champion with intercept-only calibration (seed repeat of exp-0020) | exp-0018 | 0.0365 | 0.1357 | INCONCLUSIVE (repeat of exp-0020) | REJECT — pooled bootstrap y7 [-0.0899, 0.1782], y28 [0.0486, 0.3088] |
| exp-0022 | Champion with intercept refitted per season on the deployed model | exp-0018 | 0.0365 | 0.1357 | INCONCLUSIVE | REJECT |
| exp-0023 | Champion with intercept refitted per season on the deployed model (seed repeat of exp-0022) | exp-0018 | 0.0365 | 0.1357 | INCONCLUSIVE (repeat of exp-0022) | REJECT — pooled bootstrap y7 [-0.0899, 0.1782], y28 [0.0486, 0.3088] |

**Current champion: exp-0018.** Cumulative Δ vs baseline exp-0001: PR-AUC y7 +0.0158, PR-AUC y28 +0.0579

## Arm B — survival

| id | title | vs | C^td | IBS | run verdict | loop decision |
|---|---|---|---|---|---|---|
| exp-0013 | Random Survival Forest, v0+history (survival baseline) | — | 0.6694 | 0.0779 | BASELINE | BASELINE |
| exp-0014 | XGBoost-AFT, v0+history (survival challenger) | exp-0013 | 0.6667 | 0.0792 | REJECT | REJECT |
| exp-0015 | XGBoost-AFT with Weibull loss, v0+history (survival challenger) | exp-0013 | 0.6635 | 0.0784 | REJECT | REJECT |

**Current champion: exp-0013.** Cumulative Δ vs baseline exp-0013: C^td +0.0000, IBS +0.0000
