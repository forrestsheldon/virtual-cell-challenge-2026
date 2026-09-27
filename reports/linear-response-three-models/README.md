# Linear-response H1 diagnostics

This directory records the revised three-model H1 experiment. It contains
expected-profile diagnostics and frozen local H1 harness scores; neither is an
official competition score. No VCC 2026 A/B/C prediction, package,
authentication, upload, or leaderboard submission was made.

The model side uses the strict 26-guide, 32,616-cell control pool and never reads
benchmark truth cells. The standalone evaluator owns
`reports/vcc2026-h1/reference_cells.csv`, the 126 canonical 400-cell target panels,
and all 38,176 H1 controls. It calls `cell_eval2` 0.16.0's discrimination kernel
with panel-wide target exclusion, midranks, and the `n-1` denominator. It does not
reimplement official MSE, DE, or score scaling.

## Result

All three models are useful negative diagnostics and none is ready for submission.

| arm | median target-excluded cosine | median Spearman | mean PDS | expected-profile error / unchanged control |
|---|---:|---:|---:|---:|
| Model 1: empirical covariance | 0.0090 | 0.0081 | 0.5119 | 1.7902 |
| Model 2: rank-50 covariance | 0.0115 | 0.0140 | 0.5125 | 32.6510 |
| Model 3: Poisson-lognormal | 0.0056 | 0.0094 | 0.5018 | 104.6617 |
| Model 1/2 normalized-decoder null | 0.0001 | -0.0017 | 0.5028 | 1.1274 |
| Model 3 normalized-decoder null | 0.0074 | 0.0019 | 0.5000 | 1.1202 |
| Model 3 fitted posterior null | 0.0084 | 0.0014 | 0.5000 | 1.0078 |

The null rows are measured chance behavior, not denominators for cosine, Spearman,
or PDS. The error column is a paired ratio of sums against unchanged controls.

For Model 1's sampled audit, `model/base = 1.8905`, `null/base = 1.2282`, and
additive null-excess removal gives `1.8905 - (1.2282 - 1) = 1.6623`; the exact
expected-profile value is `1.7902`. The ratio `model/null` is deliberately not
reported as a debiased result. No metric other than this raw MSE audit is debiased.

## Calibration and Model 1

`CELL_TARGET_SUM = 10,000` is the per-cell representation and
`BULK_TARGET_SUM = 50,000` is the calibration/evaluation representation. The old
scale mismatch is large: the median observed on-target shift is -0.6394 at 10,000
and -1.5010 at 50,000, a 2.347-fold change in absolute magnitude.

Knockdown depth is the pooled count-fraction log ratio
`kappa_j = log(f_pert,j / f_ctrl,j)`. All 150 donors are finite, detected, and
negative. Their median is -2.8284 and MAD is 0.2076 (MAD / absolute median =
0.0734). Each benchmark target receives the leave-one-target-out donor median;
restricting donors to targets with at least 400 cells is reported separately.

Amplitude is solved through the exact expected decoder on each fixed balanced
400-control draw. Model 1's maximum absolute closure error is `6.7e-11`. Its
directions are reproducible but wrong: median split-half downstream cosine is
0.9290, while batch-centering changes them by median cosine 0.9993. The top three
shared axes contain 61.30% of direction energy and the effective rank is 11.19.
This supports retiring batch correction, not adding it.

## Model 2

The independently reproduced randomized-SVD ladder is:

| rank | control variance | median cosine to Model 1 | median recovered target diagonal | median target-normalized downstream inflation |
|---:|---:|---:|---:|---:|
| 3 | 5.17% | 0.7767 | 0.0343 | 22.91x |
| 20 | 11.23% | 0.9779 | 0.0932 | 10.54x |
| 50 | 13.48% | 0.9842 | 0.1129 | 8.84x |
| 100 | 15.11% | 0.9854 | 0.1231 | 8.12x |

The recomputed full covariance columns match the direction-only legacy cache at
minimum cosine 0.9999999. Rank 50 therefore is not a distinct response direction,
but target-matched calibration makes it a materially different—and much worse—
magnitude experiment. Its median fitted amplitude is -137.1 versus -12.69 for
Model 1, and its expected error ratio rises to 32.65. The magnitude difference is
not treated as merely notational.

## Model 3

Model 3 is the rank-50 Poisson-lognormal model with observed raw totals as offsets,
population normalization `sum(q) = 1`, exact per-cell Hessians, three deterministic
global starts, and a train-only SVD initialization. The tiny one-dimensional
posterior check agrees with dense quadrature: absolute Laplace mean error is
0.00058 and relative variance error is 0.000034.

The six-step training approximation selects start 0; all three starts are nearly
tied on that approximation. Re-evaluated with converged 30-step modes, held-out
Laplace NLL improves from 561,545.61 to 561,539.17. Final inference has median MAP
gradient norm 0.000068 and q99 0.000228. Held-out posterior-predictive diagnostics
also pass their declared gates:

- observed versus predicted zero fractions: 0.5144 versus 0.5042;
- log1p gene-mean Pearson: 0.99997;
- log1p gene-variance Spearman: 0.99807;
- posterior total-rate ratio q01/median/q99: 0.9750 / 1.0029 / 1.0213;
- prior total-rate ratio q01/median/q99: 0.9712 / 0.9995 / 1.0353;
- `sum(q) = 1.000000`; norm of `q^T L = 0.01364`.

The control fit passes, but target matching exposes the same weak-diagonal failure
as Model 2. Median amplitude is -363.2. Perturbed expected total multipliers have
median 1.75 and maximum 12.60, and expected-profile error reaches 104.66. These
totals are reported without renormalization; applying a softmax would silently turn
the model back into the normalized decoder it was meant to contrast.

The cell generator therefore stops at its unchanged harness-validity gate. A
deterministic replay produces 972 cells above the 1,000,000-count maximum across
23 targets (no zero-total cells), with a maximum total of 3,124,368. No Model 3
H5AD is written, capped, or renormalized. The compact failure record is
`cell_eval/model3/generation_manifest.json`.

A later document-matched implementation tested the fuller joint-Laplace
`LL^T + D` EM formulation. Its 128-cell smoke problem passed, and the independent
`D` and three pure-factor starts completed. The full model completed two outer EM
updates, then stopped during the third E-step when the strict Armijo search failed
for batch 1,426 of 1,516. Because outer iterations were not checkpointed, those
two updates were lost when the process exited. This is reported as a numerical
failure at the predeclared convergence gate, not as a fitted Model 3 result; no
solver relaxation, extra penalty, or silent fallback was introduced.
`model3_full_failure.json` preserves the completed objectives, exact failure
location, surviving lower-rung artifacts, and stop decision.

## Frozen local cell-level H1 scoring

The released scale bundle and reference cache were hash-verified, then copied to
the isolated `data/derived/linear_response/eval_cache`. Model 1, Model 2, and the
normalized-decoder null were scored with the unchanged in-repository frozen
harness. The deterministic control baseline and Model 3 posterior null were
scored with the standalone `vcc2026-h1-benchmark` v0.2.0 interface, which pins the
same CellEval2/pdex versions, benchmark contract, configuration, and scale
bundle. Each sampled prediction is deleted only after its generation and score
manifests agree on the prediction hash.

| arm | local scaled average | PDS | DE LFC NMAE | DE fidelity/yield | DE reach | DE Jaccard | status |
|---|---:|---:|---:|---:|---:|---:|---|
| Model 1 | 0.1083 | 0.0125 | -0.1589 | 0.5383 | 0.0276 | 0.2304 | complete |
| Model 2 | -0.3805 | 0.0121 | -3.1601 | 0.5440 | 0.0101 | 0.3109 | complete |
| Model 3 | — | — | — | — | — | — | stopped at count-total gate |
| normalized-decoder null | 0.1039 | -0.0294 | -0.0781 | 0.5294 | 0.0058 | 0.1957 | complete after clean retry |
| Model 3 posterior null | 0.0176 | -0.0268 | -0.0576 | 0.2055 | -0.0491 | 0.0337 | complete |
| deterministic unchanged-control baseline | -0.0452 | -0.0455 | -0.0763 | -0.0836 | -0.0610 | -0.0049 | complete |

These are scores on the public H1 reconstruction, not leaderboard results. In
particular, Model 2's amplified response is decisively harmful here despite its
raw direction cosine of 0.984 to Model 1. Model 1's sampled H5AD is a deterministic
replay from compact float32 directions; it differs from the earlier one-off
float64 sampled-profile audit by at most 0.00079 in log-bulk space. That small
non-bit-exact replay is recorded rather than hidden.

Model 1 improves the scaled average over its normalized-decoder null by only
0.00441. Its PDS, fidelity, reach, and Jaccard rise, while DE LFC accuracy becomes
worse. The full H1 result therefore does not demonstrate a practically meaningful
gain over the source-panel/generator null.

The fitted Model 3 posterior null scores 0.01761. It improves on the deterministic
unchanged-control baseline (-0.04523), mostly through DE direction fidelity, but
is below the normalized-decoder null (0.10391). These nulls are deliberately not
interchangeable: the deterministic baseline samples size-matched cells from all
38,176 controls, the normalized-decoder null replays Model 1's fixed source rows
through a multinomial decoder, and the posterior null draws Poisson counts from
Model 3's fitted source posterior. The posterior-null H5AD was 3.37 GiB and was
removed after its SHA-256 matched both manifests; its compact scores and per-target
results remain under `cell_eval/model3_posterior_null`.

## Control-generative fit

Control-only tests separate whether the covariance is learned from whether it
predicts perturbations. On a fixed 1,024-gene downstream panel, the learned
covariance columns are far above two structure-destroying ensembles:

| covariance model | matched cosine | wrong-target q95 | randomized-gene q95 | error / zero covariance |
|---|---:|---:|---:|---:|
| Cipher empirical, split half | 0.9704 | 0.1110 | 0.0505 | 0.0378 |
| rank-50 projection, split half | 0.9924 | 0.1123 | 0.0525 | 0.0118 |
| Poisson-lognormal, held-out guides | 0.9417 | 0.1062 | 0.0518 | 0.0770 |

One-sided permutation p-values are below 0.0001 for 10,000 wrong-target
assignments and 0.0005 for 2,000 randomized gene mappings. The rank-50 split-half
test uses a basis fitted on all controls, so it is a denoising check rather than a
strict held-out fit; the Poisson-lognormal comparison uses guides excluded from
global fitting.

Actual prior-predictive counts were also generated for 256 held-out control cells
over three deterministic replicates. A rank-50 Gaussian/multinomial generator
improves downstream covariance error from 1.226 for the matched independent-mean
multinomial to 0.950, covariance cosine from approximately zero to 0.354, and
zero-fraction absolute error from 0.0106 to 0.0015. The Poisson-lognormal prior
improves its matched independent-Poisson covariance error from 1.228 to 1.008 and
cosine from approximately zero to 0.192, but barely improves zero-fraction error
(0.0105 to 0.0102). Cipher itself stores only 150 covariance columns, not a full
positive-definite covariance, so generating from it would require a new,
unimplemented distributional assumption.

On 4,000 cells from two held-out control guides, the fitted Poisson-lognormal
Laplace NLL is 561,539.16 per cell versus 562,339.60 for the train-only independent
Poisson mean. The rank-50 initialization already reaches 561,545.60: 793.996 of
the 800.434-unit improvement comes before optimization, while fitting contributes
6.439. All 4,000 paired cells improve after fitting (descriptive cell-level SE
0.070), but two held-out guides are insufficient for a population-level
significance claim.

No model uses a sparsity prior. Cipher has no penalty; Model 2 has only the rank-50
constraint; Model 3 has a dense L2 loading penalty with coefficient `1e-4`.
`generative_covariance_generalization.csv`, `generative_sampled_control_fit.csv`,
`generative_heldout_likelihood*.csv`, and `generative_regularization_audit.csv`
contain the compact results and controls.

## Stable strong-DE direction diagnostic

The scale-free mechanism diagnostic uses the frozen 10,780-gene DE universe,
excludes the perturbed target, and defines a strong truth effect by adjusted
`p < 0.05` and `abs(log2 fold change) >= 0.5`. A gene is retained when both
independent 200-cell half-panel signs match the full-panel sign in at least four
of five deterministic splits. Of 23,920 target-excluded strong effects, 23,918
pass this sign-stability rule. Seventy-eight perturbations retain at least 10
genes and enter the equal-target-weight aggregate.

For each target with `K` stable strong genes, genes are ranked by absolute model
response and the primary statistic is the fraction of the top `K` that are truth
strong-DE genes with the correct sign. It is invariant to positive scalar
amplitude and penalizes non-DE genes that displace true responses.

| response representation | mean signed top-K recovery | median | wrong-target mean | random-gene mean | wrong-target p |
|---|---:|---:|---:|---:|---:|
| Model 1 decoder tangent | 0.0252 | 0.0086 | 0.0209 | 0.0141 | 0.0429 |
| Model 2 decoder tangent | 0.0246 | 0.0086 | 0.0212 | 0.0141 | 0.0784 |
| Model 1 finite decoder | 0.0239 | 0.0086 | 0.0204 | 0.0141 | 0.0695 |
| Model 1 covariance direction | 0.0186 | 0.0047 | 0.0156 | 0.0142 | 0.0693 |
| Model 1 residual after top 3 axes | 0.0185 | 0.0033 | 0.0151 | 0.0142 | 0.0791 |
| Model 2 rank-50 direction | 0.0182 | 0.0060 | 0.0158 | 0.0142 | 0.1184 |
| Model 2 finite decoder | 0.0167 | 0.0000 | 0.0157 | 0.0140 | 0.2730 |
| Shared top 3 axes alone | 0.0138 | 0.0044 | 0.0165 | 0.0142 | 0.9583 |

The Model 1 decoder tangent retrieves 5.04% of the stable strong genes in its
top-K sets; sign agreement conditional on retrieval is 50.4%, yielding 2.52%
signed recovery. It is clearly above randomized gene labels, but only 0.43
percentage points above assigning another target's response. Its nominal
wrong-target p-value does not survive correction across the eight exploratory
representations (`p = 0.343` after Bonferroni correction). Four individual
targets are nominally above their wrong-target distributions, but none survives
Benjamini-Hochberg correction (minimum `q = 0.774`).

This isolates two conclusions that MSE obscured. First, Model 1 and rank 50 have
similar intrinsic strong-DE direction information; rank 50's much worse MSE is
primarily its finite-amplitude calibration pathology. Second, the top three
shared axes have no detectable strong-DE signal by themselves and perform worse
than assigning another target's direction. The weak enrichment over random genes
is therefore consistent with generic or shared response tendencies rather than
target-specific recovery. The full-panel LFC calculation was independently
reconstructed from the canonical cells and matches the frozen cache to maximum
absolute error `3.4e-14`.

`strong_de_truth_genes.csv`, `strong_de_recovery_per_target.csv`,
`strong_de_recovery_summary.csv`, and `strong_de_recovery_manifest.json` retain
the gene sets, per-target decomposition, randomized controls, hashes, and seeds.

## Cross-fitted forcing-map ceiling

The fixed target force in CIPHER is the assumption `u_g = -e_g`, so that the
response is the negative target covariance column. To test whether the covariance
contains more response information than this single column exposes, a diagnostic
oracle fits target-anchored sparse forcing vectors `C u_g`. For every perturbation,
the forcing support and coefficients are learned from one deterministic half of
the 10,780-gene truth axis and scored only on the other half; the halves are then
reversed. The target coefficient is constrained non-positive. This is a
mechanistic ceiling that reads target truth during fitting, not a zero-shot model.

| response operator | forcing coordinates | held-out signed recovery | wrong-target mean | corrected p |
|---|---:|---:|---:|---:|
| empirical covariance | fixed target | 0.0184 | 0.0158 | 1.0000 |
| empirical covariance | target + 1 | 0.0339 | 0.0154 | 0.0011 |
| empirical covariance | target + 4 | 0.0623 | 0.0215 | 0.0011 |
| empirical covariance | target + 9 | 0.0841 | 0.0246 | 0.0011 |
| rank-50 covariance | fixed target | 0.0182 | 0.0159 | 1.0000 |
| rank-50 covariance | target + 1 | 0.0482 | 0.0181 | 0.0011 |
| rank-50 covariance | target + 4 | 0.1006 | 0.0268 | 0.0011 |
| rank-50 covariance | target + 9 | **0.1507** | 0.0353 | 0.0011 |
| rank-50 mode span | unconstrained 50-dimensional ceiling | **0.1704** | 0.0377 | 0.0011 |

The fixed-column positive control reproduces the earlier 0.0186 recovery to
within 0.0002 despite ranking separately inside the two held-out folds. Ten
rank-50 forcing coordinates recover 15.6% of held-out strong genes and assign the
correct sign to 95.4% of those retrieved, giving 15.1% signed recovery. It improves
65 of 78 eligible targets and loses one relative to the fixed rank-50 column.

This is evidence that the rank-50 control response *space* contains a meaningful
part of the perturbation truth. It does not provide a zero-shot way to choose the
force: about 46% of eligible target-only fits put the required negative target
coefficient at its zero boundary, and even at support 10 this occurs in 39% of
fold fits. Secondary supports are only moderately stable between gene folds
(median Jaccard 0.385 at support 10) and repeatedly select shared-axis genes such
as `CDK1`, `IGFBP2`, `SCGB3A2`, and histones. The result therefore localizes the
failure to the map from intervention identity to forcing vector, while warning
that the oracle partly reconstructs shared response programs.

`forcing_ceiling_summary.csv`, `forcing_ceiling_per_target.csv`,
`forcing_ceiling_support.csv`, and `forcing_ceiling_manifest.json` retain the
cross-fitted scores, selected forces, coefficients, nulls, hashes, and split seed.

## Control-only natural-expression dose response

The second diagnostic asks whether naturally low expression of the target is a
better proxy for CRISPRi than its covariance column. Within each deterministic
control half, target `log1CP10k` is residualized against batch and guide identity.
A sensitivity arm also removes the top three control-PC scores. Linear and
quadratic downstream curves are fitted using controls only, and the predicted
direction is the fitted change from the residual median to its tenth percentile.
The two independent half-directions are averaged before truth evaluation.

| control-only curve | signed recovery | wrong-target mean | corrected p | median half cosine |
|---|---:|---:|---:|---:|
| linear, batch + guide | 0.0187 | 0.0157 | 0.2640 | 0.9378 |
| quadratic, batch + guide | 0.0187 | 0.0162 | 0.3960 | 0.9061 |
| linear, batch + guide + top 3 PCs | **0.0200** | 0.0152 | 0.1208 | 0.8618 |
| quadratic, batch + guide + top 3 PCs | 0.0195 | 0.0148 | 0.1080 | 0.7975 |

The adjusted linear curve closely reproduces the empirical covariance result,
which is a positive control on the implementation. Quadratic curvature provides
no aggregate gain and lowers split-half stability. Removing the top three axes
raises recovery by only 0.0012; neither adjusted result survives correction over
the four curves, and no perturbation survives target-wise Benjamini-Hochberg
correction. Thus finite nonlinearity in the observed natural target-expression
range does not rescue target-specific strong-DE prediction.

`control_dose_response_summary.csv`, `control_dose_response_per_target.csv`,
`control_dose_response_fit.csv`, and `control_dose_response_manifest.json` retain
the scores, residual ranges, split stability, nulls, hashes, and software versions.

## Reliability and retracted fixes

Five deterministic splits divide every canonical target panel into disjoint
200-cell halves and split controls independently. Median target-excluded reliability
is 0.4438 cosine and 0.2587 Spearman. Aggregate PDS is 0.9773 at 200-vs-200 and
0.9885 after the `cell_eval2` Spearman-Brown correction to 400 cells. No nonexistent
second 400-cell panel is claimed.

The following remain retired diagnostics: batch centering, projecting out leading
modes, correlation columns, and diagonal shrinkage. In particular, the apparent
top-mode-complement gain had split-half cosine 0.019 and is sampling noise. The
cell-cycle interpretation of the shared axes remains a reviewer result that was not
independently re-derived here. `retracted_diagnostics.csv` preserves each retired
result and its control; `review_evidence.csv` marks which claims were independently
reproduced.

## Reproduction

```bash
pixi run python scripts/linear_response/build_stage1_manifest.py
pixi run python -m scripts.linear_response.model1_cipher
pixi run python -m scripts.linear_response.model2_pca
pixi run python -m scripts.linear_response.model3_poisson_lognormal
pixi run python -m scripts.linear_response.strong_de_recovery
pixi run python -m scripts.linear_response.forcing_ceiling
pixi run python -m scripts.linear_response.control_dose_response
pixi run python -m scripts.evaluation.profile_h1 \
  data/derived/linear_response/model1_expected_profiles.npz \
  --output reports/linear-response-three-models/profile_scores/model1
```

Model 2 and Model 3 use the same evaluator command with their corresponding NPZ and
output directory. Model-side artifacts contain absolute expected pseudobulk
profiles, target/gene order, source rows, nulls, and model metadata—not truth.
Hashes, seeds, software versions, and independent-reproduction status are recorded
in the manifests and `review_evidence.csv`.

For cell-level replay, run one arm at a time with
`scripts.linear_response.generate_h1_cells`, validate and score it with the frozen
H1 harness while explicitly passing `data/derived/linear_response/eval_cache`,
verify both manifests, then delete the replaceable H5AD. Model 3 remains stopped
at both its approximate model's count-total gate and the fuller model's strict
joint-mode convergence gate. This phase still forbids A/B/C generation,
packaging, authentication, and submission.

The standalone local scorer at
`/Users/forrestsheldon/Projects/vcc2026-h1-benchmark` is tagged `v0.2.0`. It pins
the same CellEval2 commit, pdex version, H1 checksum, configuration checksum, and
126-by-400 benchmark contract used here, while bounding memory through target-wise
pseudobulks and 512-gene prediction-DE chunks. Its local data directory is now set
up and checked from the existing H1 file and the repository's release asset. Its
deterministic unchanged-control baseline reproduces the documented scaled average
of -0.04523068765223792. The standalone scorer is the preferred interface for
future artifacts.
