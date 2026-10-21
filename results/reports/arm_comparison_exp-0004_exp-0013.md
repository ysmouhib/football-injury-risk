# Arm A (exp-0004) vs Arm B (exp-0013) — origin 2023

Arm A = class-weighted classifier probability for the horizon; Arm B = 1 − S(h) from the survival model. Identical rows; paired player-cluster bootstrap.

| horizon | prevalence | PR-AUC A | PR-AUC B | Δ (B − A) [95% CI] | slope A / B | intercept A / B | mean predicted A / B | Spearman | top-decile overlap |
|---|---|---|---|---|---|---|---|---|---|
| 7 d | 0.0161 | 0.0307 | 0.0308 | +0.0000 [-0.0016, +0.0013] | 0.996 / 1.164 | -4.134 / -0.085 | 0.461 / 0.018 | 0.961 | 0.822 |
| 28 d | 0.0624 | 0.1137 | 0.1150 | +0.0012 [-0.0037, +0.0057] | 0.995 / 1.161 | -2.740 / -0.061 | 0.467 / 0.066 | 0.963 | 0.836 |
