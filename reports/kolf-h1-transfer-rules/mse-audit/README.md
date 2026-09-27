# Why is the profile error ratio large?

Diagnostic run 2026-09-25, after the original experiment. Predictions and original results are
unchanged. No regression or H1 calibration was fitted.

## Final renormalization is not the cause

The original prediction is floored at zero, then scaled to one million CPM. To isolate this
last operation, score the floored rates before scaling using the same fixed log1p(CPM/20)
transformation. This is an ablation, not an alternative valid normalized profile.

| Rule | Error ratio before final scaling | After final scaling |
|---|---:|---:|
| Additive | 9.0242 | 8.9678 |
| Ratio | 5.8949 | 5.8791 |
| Log1p | 5.0743 | 5.0587 |

The final scaling slightly reduces the error. The CPM-to-CP50K conversion and inverse-log1p
transfer were already independently checked in the original validation. This audit does not
establish that common-panel normalization is biologically ideal, only that the final rescaling
is not generating the large observed error.

## Exact energy decomposition

Let P be all predicted effect entries and Y all true effect entries on the fixed scored axis.
Then `||P-Y||²/||Y||² = 1 + ||P||²/||Y||² - 2<P,Y>/||Y||²`.

| Rule | Prediction energy / truth | Twice overlap / truth | Error ratio |
|---|---:|---:|---:|
| Additive | 8.4365 | 0.4687 | 8.9678 |
| Ratio | 5.3032 | 0.4241 | 5.8791 |
| Log1p | 4.4728 | 0.4140 | 5.0587 |

For log1p, the pooled norm ratio is 2.1149 and pooled cosine is 0.0979. These pooled values
are different aggregations from the original mean per-target norm ratio (3.48) and cosine (0.069).
Large prediction energy has little matching overlap with H1. Calling this a *biological*
amplitude mismatch would be premature: measured effects contain sampling noise.

Error is not concentrated only in zero source entries: these contribute 4.03% of log1p error
and 10.47% of ratio error. About 76% of log1p error occurs at genes with KOLF control expression
between 5 and 100 CPM. The largest single target contributes less than 2% of total error.

## Count-depth audit and technical sampling null

Observed shared-gene pseudobulk depths:

| | KOLF | H1 |
|---|---:|---:|
| Median total UMI per target | 1,054,133 | 22,382,442 |
| Median UMI per cell | 4,350 | 55,956 |
| Median cells per target | 242 | 400 |

To isolate technical count sampling, set the true KOLF perturbation composition equal to its
pooled control composition. For each target, draw a multinomial pseudobulk with that probability
vector and exactly its observed KOLF UMI total. Apply the original transfer rule, zero fallback,
flooring, normalization, and evaluation to these samples. Repeat ten times with seed 20260925.
H1 perturbation truth is used only for diagnostic evaluation and the reference energy denominator.
The noise-free null maps exactly to unchanged H1 controls; all simulated library sums are verified.

| Rule | Null prediction energy / H1 truth (mean) | Null error ratio against actual H1 (mean; range) | Actual transfer error ratio |
|---|---:|---:|---:|
| Additive | 4.3716 | 5.3747 (5.3631–5.3846) | 8.9678 |
| Ratio | 4.2799 | 5.2807 (5.2666–5.3025) | 5.8791 |
| Log1p | 3.4879 | 4.4897 (4.4808–4.4987) | 5.0587 |

Even a donor with no biological perturbation response produces a large apparent response when
sampled at these depths. For log1p, the technical null energy is about 78% of the *scale* of the
observed prediction energy. This is **not an estimate that 78% of the actual prediction is noise**:
actual perturbations change sampling variance and interact with the nonlinear transfer.

This null conditions on observed libraries and treats pooled control probabilities as known.
It ignores control uncertainty, biological heterogeneity, construct differences, and batch effects.
It is not a cell-split reliability estimate or an unbiased variance subtraction. It supports
technical count noise as a substantial contributor and motivates the independent split-half work.
It does not identify the remaining energy as transferable biological signal.

Reproduce:

```sh
pixi run python -m scripts.evaluation.audit_kolf_h1_transfer_mse
```

`decomposition.csv`, `expression_bins.csv`, `per_target.csv`, `source_depth.csv`, and
`count_sampling_null.csv` contain the numbers. `manifest.json` records assumptions, input hashes,
producer hash, seed, and output hashes. Original transfer predictions are not modified.
