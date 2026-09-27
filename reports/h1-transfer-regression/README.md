# Transfer into H1: pseudocount sweep, scale fits, and multi-source regression

Run 2026-09-25 with `scripts/evaluation/h1_transfer_regression.py`, reusing the frozen
profiles from `reports/all-source-h1-transfer` (same targets, genes, CPM support, evaluation
space `ln(1+CPM/20)`, panel-target genes excluded). No downloads or new source processing.

## Definitions

- Pseudocount family: `h_hat = (h0 + c) * ((kp + c)/(k0 + c)) - c`; c = 0 is ratio, c = 1
  is the log1p rule, c = inf is additive. Checked: c in {0, 1, inf} reproduce the stored
  predictions.
- Scale, effect space (E): `y_hat = a q`, q the evaluated effect of the direct prediction.
  Least squares on training targets. Checked: the all-target fit equals `<q,y>/||q||^2` from
  the stored decomposition. Best achievable pooled error ratio is `1 - rho^2`.
- Scale, fold-change space (F): `h_hat = (h0 + c) * ((kp + c)/(k0 + c))^a - c`, floored and
  renormalized, with a from a bounded 1-D search in [0, 1.5]. Deployable version.
- Averages are taken in the log-fold-change space for each c (undefined at c = 0):
  equal-source weights, and equal-study weights (Replogle / Nadig / X-Atlas / Zhu / Nourreddine).
- `kolf_increment`: `a * avg + b * (KOLF - avg)`. `multi_ols` / `multi_nnls`: one coefficient per source.
- `h1_mean`: mean H1 effect of the training targets (H1-calibrated common response).
- Wrong-target controls: five derangements of donor-target correspondence, refit per fold.
- Five fixed target folds (seed 20260925); all metrics are held-out. 2,000 target bootstraps
  give the MSE-ratio intervals, which are conditional on these screens.

## Main panel: five anchors, 110 targets, c = 1

| Model | Retrieval | Cosine | MSE ratio [95%] | Fitted coefficient(s) |
|---|---:|---:|---|---|
| unchanged | 0.500 | 0.000 | 1.000 | |
| h1_mean | 0.443 | 0.149 | 0.979 [0.969, 0.989] | |
| K562 GWPS scaled | 0.851 | 0.077 | 0.993 [0.988, 1.000] | a = 0.043 |
| KOLF2.1J scaled | 0.745 | 0.069 | 0.991 [0.986, 0.997] | a = 0.046 |
| CD4T scaled | 0.687 | 0.048 | 0.994 [0.987, 1.001] | a = 0.065 |
| HCT116 scaled | 0.781 | 0.049 | 0.998 [0.996, 1.001] | a = 0.021 |
| HEK293T scaled | 0.758 | 0.041 | 0.999 [0.998, 1.001] | a = 0.021 |
| avg direct | 0.858 | 0.113 | 1.721 [1.341, 2.426] | |
| avg scaled | 0.858 | 0.113 | 0.984 [0.972, 1.003] | a = 0.136 |
| study_avg scaled | 0.858 | 0.113 | 0.979 [0.967, 0.996] | a = 0.151 |
| kolf_increment | 0.851 | 0.108 | 0.982 [0.971, 1.000] | a = 0.137, b = 0.022 |
| multi_nnls (= OLS) | 0.847 | 0.110 | 0.976 [0.963, 0.995] | CD4T .059, KOLF .045, K562 .039, HCT .018, HEK .017 |
| avg + h1_mean | 0.538 | 0.181 | 0.963 [0.949, 0.984] | m = 0.980, a = 0.130 |
| multi + h1_mean | 0.559 | 0.181 | 0.957 [0.942, 0.977] | m = 0.964 |
| avg scaled, wrong target | 0.503 | 0.004 | 1.000 | |
| avg + h1_mean, wrong target | 0.443 | 0.149 | 0.979 | |

Scale fitting leaves retrieval and cosine unchanged and brings every single source to
0.991–0.999, as the `1 - rho^2` bound predicts (`in_sample_bound.csv`). E and F agree
closely. Fold coefficients are stable (range within about ±0.01).

## Pseudocount sweep

Held-out cosine peaks at c of about 3–30 CPM for every source in every panel (tables in
`summary.csv`). On the five anchors, the equal-source average reaches cosine 0.119 at c = 10,
versus 0.113 at c = 1, 0.072 additive, and 0.102 at c = 0.1. F-space scaling fails at c = 0
because a zero ratio stays zero for any a > 0.

## Files

`summary.csv` (all panels, c, models, spaces), `coefficients.csv` (per fold),
`per_target.csv.gz`, `in_sample_bound.csv`, `manifest.json`.
