# How much of a perturbation response transfers between cell contexts?

**Status:** first KOLF→H1 transfer-rule experiment completed 2026-09-25; multi-source/regression work deferred  
**Prepared:** 2026-09-24; revised after plan review on 2026-09-24  
**Scope:** retrospective H1 transfer, averaging, and regression calibration  
**Follow-up:** control-expression matching and unseen-context evaluation; VCC prediction and submission are separate work

## Executed first experiment (2026-09-25)

The subsequent discussion narrowed the first run to three **unfitted KOLF→H1 transfer rules**:
additive CPM changes, pure CPM ratios, and log2(CPM+1) changes. H1 perturbations were evaluation
truth only. Shared-gene CPM normalization, flooring of negative rates, and renormalization were
used; undefined donor ratios defaulted to no change. Count checks used stochastic rounding.
This supersedes the earlier proposed H1-calibrated regression as the immediate first experiment.

Results and exact methods: [KOLF–H1 transfer-rule report](../reports/kolf-h1-transfer-rules/README.md).
Log1p and ratio gave similar retrieval, with log1p reducing error; all three transferred target
identity but worsened profile squared error relative to unchanged controls. No broad-source
average or fitted regression was run. The remaining sections retain the longer-term proposal,
not an instruction to fit coefficients on H1 in the next step.

## 1. Question and place in the blog series

The first regression-baselines post follows *Baselines I: Linear Response*. That post found that
control covariance supplied little useful perturbation information. The next question is:

> If control fluctuations cannot recover a perturbation response, how much can we recover from
> measurements of that same perturbation in other cellular contexts?

Start with individual source responses, then ask whether averaging helps, whether scalar
calibration improves transfer, and whether KOLF contributes information beyond the broad average.
Control-expression matching is a subsequent hypothesis, not a prerequisite for this first post.

Hypotheses to state before the new analysis:

- Correct target identities transfer more information than wrong-target or common-response controls.
- Combining sources may improve prediction, but can also dilute useful context-specific responses.
- KOLF2.1J is a biologically motivated donor for H1 because both are pluripotent; this does not
  establish that it is the best donor.
- Amplitude calibration may improve error without recovering additional response direction.

These are hypotheses, not promised results. H1 has already been explored in this repository;
new target folds do not make it an untouched confirmatory dataset. Record prior analyses that
informed model choices and distinguish reused results from new calculations.

## 2. Prediction object and provenance

For perturbation target `p`, dataset–context source `s`, and response gene `g`, write the source
effect as `x_psg` and the H1 effect as `y_pg`. Exclude the perturbed target gene from fitting and
evaluation. Apply broader metric-specific exclusions consistently, including panel-target
exclusion when using official PDS.

The current anchors are Replogle K562 GWPS, primary CD4 T cells, X-Atlas HCT116, X-Atlas HEK293T,
and KOLF2.1J. Treat each dataset–context pair as one source prediction. Retain study, assay,
construct, control, preprocessing, and artifact provenance separately. Mirrors and alternative
processed copies of the same observations are not independent evidence.

The currently recorded all-source H1 intersection is 110 targets. Verify membership and
eligibility against input manifests, then freeze it for the primary paired comparison. Full
available overlaps are secondary and represent different target populations. Use common
response-gene support for paired comparisons; missing genes or targets are not zeros. Document
how the intersection differs from the full H1 benchmark population.

Use one frozen effect definition across sources. Reproducing the existing epsilon log-fold-change
anchors is a candidate starting point, not an implicit endorsement of that geometry. Record
normalization denominators, pseudocounts, control aggregation, gene support, and decoding. Preserve
raw counts. Standardized linear-rate effects from the split-power work are a separate sensitivity;
do not silently mix geometries or change preprocessing between models.

## 3. Averaging is a prediction choice, not an independence correction

Define the primary broad baseline as the equal-source average:

\[
\bar x_p=\frac{1}{S}\sum_{s=1}^{S}x_{ps},\qquad S=5.
\]

This averages available source effects; it does not identify a universal biological response.
It weights observed contexts equally without claiming that they represent future contexts.

HCT116 and HEK293T share X-Atlas provenance. That matters for correlated errors and interpretation,
but does not imply that their combined prediction weight must equal a single-context study's.
Different studies may also share systematic errors. Distinguish prediction weights from the
independence of evidence used to assess generalization.

As a sensitivity, define the equal-study average:

\[
\bar x_p^{\mathrm{study}}
=\frac{1}{D}\sum_{d=1}^{D}\frac{1}{|S_d|}\sum_{s\in S_d}x_{ps}.
\]

For the current five-source, four-study grouping:

| Source | Equal source weights (primary) | Equal study weights (sensitivity) |
|---|---:|---:|
| K562 | 20% | 25% |
| CD4 T | 20% | 25% |
| KOLF | 20% | 25% |
| HCT116 | 20% | 12.5% |
| HEK293T | 20% | 12.5% |

Equal study weights limit the influence of studies contributing more contexts; equal source
weights preserve equal influence per observed context. Neither fixes uneven biological coverage
or makes sources independent. Verify grouping from provenance and report sensitivity to study
exclusion where coverage permits.

## 4. First-analysis model ladder

Use deterministic five-fold target assignments shared by every fitted model. Fit on training
targets and predict only held-out H1 targets. Freeze preprocessing and metrics before comparison;
outcome-driven tuning requires an inner training split.

### A. Direct transfer

1. **Unchanged:** `y_hat_p = 0`.
2. **Each source:** `y_hat_p = x_ps`, with KOLF the biologically motivated comparison.
3. **Broad average:** `y_hat_p = bar_x_p`.
4. **Study-balanced average:** the weighting sensitivity in Section 3.

This shows what measured source effects supply before H1-fitted calibration.

### B. One scalar per candidate predictor

For each individual source and each fixed average, fit

\[
\widehat y_p=\widehat a q_p,
\]

where `q_p` is that candidate's fixed effect vector. Every candidate receives the same fitting
procedure and held-out folds. Average raw source effects first, then fit one scalar to the
average. Scaling each source separately before averaging learns relative source weights: it is
a supervised ensemble, deferred from this simple amplitude comparison.

### C. Does KOLF add information beyond the average?

After comparing endpoints, fit the two-coefficient regression

\[
\widehat y_p=a\bar x_p+b(x_{p,\mathrm{KOLF}}-\bar x_p).
\]

`b = 0` gives a scaled broad average; `b = a` gives scaled KOLF. For positive `a`,
`0 <= b <= a` interpolates between them; unrestricted coefficients can extrapolate. Judge added
value by held-out prediction, not the sign or significance of `b` alone. KOLF is already in the
average, so these predictor directions are not independent.

A joint coefficient per source is exploratory follow-up. Five coefficients and 110 targets do
not themselves establish overfitting. Inspect predictor correlation, fold coefficient variation,
and held-out performance before deciding whether a larger fit helps or requires regularization.

## 5. Fitting objective: decide before execution

The fitting loss and coefficient constraints remain modeling decisions to discuss before
implementation. Apply the chosen procedure consistently across comparable candidates.

- **Squared effect error:** minimize `sum_p ||y_p - a q_p||^2` over the fixed gene support.
  This gives `a = sum_p <q_p,y_p> / sum_p ||q_p||^2`. It directly optimizes squared prediction
  error, but stronger perturbations have more influence on the fitted slope.
- **Median per-target projection:** take the median of `a_p = <q_p,y_p> / ||q_p||^2`.
  This resists extreme slopes but gives a weak, noisy perturbation the same vote as a well-resolved
  one. Zero predictor norms have undefined projections. It is a robust slope summary, not OLS.
- **A declared target-weighted loss:** can limit strong-target dominance, but changes the estimand
  and can amplify weak-target noise. Do not introduce weights implicitly.

Proposed starting point for discussion: squared effect error for the regression, with median
projections as a robustness diagnostic. Decide whether scalar fits are nonnegative (attenuation
or amplification) or signed (also permitting inversion). Positive scaling preserves a nonzero
vector's direction; negative scaling does not. Define degenerate-fit handling explicitly.

Gene entries may enter a prediction loss but are not independent replicates for inference.
Cross-split errors-in-variables estimates may later inform biological transfer slopes; these
answer a different question from predictive coefficients for noisy observed sources and should
not automatically replace them.

## 6. Controls and evaluation

### Calibration controls

- **Wrong-target transfer:** permute source-to-destination target correspondence. Use the same
  permutation across sources when testing ensembles to preserve source agreement. Exclude both
  destination and shuffled-source target genes, or use a common panel-excluded axis, to avoid
  direct knockdown leakage. Refit calibrated nulls using training folds only; distinguish this
  from inference-only shuffling.
- **Target-independent response:** predict the same mean response for every target. An H1 mean
  uses training targets only and is labelled H1-calibrated. A source-derived mean uses a fixed,
  documented target population. This connects to the common-response baseline in the CRISPRi post.
- **Positive reliability control:** use disjoint H1 cell splits where validated artifacts permit.
  Exact self-transfer checks implementation, not experimental reliability.
- **Null response control:** use independently held-out controls where supported by existing artifacts.

Confirm these controls work before interpreting model differences. Record missing controls and
resolve their scope before execution rather than silently launching additional data acquisition.

### Complementary metric views

Report target retrieval (official PDS where applicable) and target-excluded effect cosine against
wrong-target controls. Keep common-response performance visible. Define treatment of zero vectors
explicitly rather than silently dropping them.

Also report signed top-gene recovery, strong-DE direction/yield, NMAE, norm ratios, and expression
error. Freeze top-gene counts, DE/effect thresholds, and eligibility before comparison. DE
significance and direction/yield depend on amplitude, sample size, and generated variability;
they are not pure direction measures. Improvement on these alone does not establish additional
perturbation-specific directional information.

Begin with expected-effect predictions. If count generation is included, keep the decoder, H1
control baseline, generation procedure, and seeds fixed across candidates. Independently check
expected and emitted means/effects. Distinguish profile diagnostics from the official cell-based
scorer; verify and pin its live contract before official evaluation. Mean prediction does not
establish successful prediction of single-cell distributions.

Use paired target bootstraps on held-out predictions. These intervals are conditional on the
observed screens, controls, and fitted fold models; they do not capture new-laboratory or
new-context variation, or full training uncertainty. Do not report gene-entry p-values. Show
per-target variation alongside aggregate results.

## 7. What the first analysis can establish

| Observation | Supported interpretation | Still unresolved |
|---|---|---|
| Correct-target sources beat shuffled/common responses | Target identity carries transferable information on this panel | Other contexts and target populations |
| Average beats individual sources | Pooling improves prediction for these sources and H1 targets | Noise reduction versus complementary biology or systematic bias |
| KOLF beats the average or adds held-out value | This donor provides useful information beyond this pool | Whether pluripotency, assay, construct quality, or another factor explains it |
| Scalar fitting improves error without direction/retrieval | Amplitude calibration helps | Additional perturbation-specific directional information |
| No clear separation | Comparison unresolved at this panel's precision | Equivalence, inactivity, measurement limits, and inadequate models |

H1-fitted coefficients evaluated on held-out H1 targets establish **H1-calibrated transfer**.
They do not establish zero-shot transfer to an unseen destination context. Applying those frozen
coefficients elsewhere adds the hypothesis that H1 calibration transfers. KOLF was biologically
motivated before this comparison, but should not be described as independently preregistered
given the existing exploratory work.

## 8. Follow-up: do controls predict which responses transfer?

First ask whether smaller control-expression distances predict better response transfer across
source–destination pairs. Examine within-study and across-study comparisons before optimizing
nearest-neighbor ensembles. Pairs sharing a context are not independent evidence.

The measurement model in *Exploring the Data III* makes the limitation explicit: control
expression reflects both gene sensitivity `f_sg` and biological expression `pi_sg`. Normalization
and source-wise gene standardization do not generally separate them. Standard deviations across
only five source contexts are also weakly estimated. Distance, gene support, low-variance handling,
normalization, and panel exclusions require justification; standardized correlation of
`log1p(CPM)` profiles is only a candidate. Retain control-construct identities for QC.

Finding KOLF nearest to H1 is a sanity check, not validation of response matching. If top-k
selection is pursued, rank and average the same dataset–context units throughout, so `k = S`
exactly recovers the broad average. Do not average a study's contexts in ALL but select one of
them in MATCHED. Study-balanced variants must likewise have consistent endpoints.

Evaluate unseen-context transfer with an outer context holdout: no destination responses enter
coefficient fitting, distance selection, or k tuning. Tune on inner training contexts only.
Separately exclude the entire destination study to test transfer across experimental provenance.
With too few contexts, fix a small rule and acknowledge limited generalization evidence.

Any later VCC comparison must freeze its rule before scoring, keep effect representation and
generation identical between arms, document per-target coverage/fallbacks, and record all
input/configuration/output hashes. It is not required to finish the first post. Candidate
generation and submission remain separate decisions; repeated leaderboard tuning is outside scope.

## 9. Diagnostic continuation and deferred models

If transfer is weak or model differences unresolved, investigate reliability before escalating
complexity. The [split-power plan](context-transfer-split-power.md) supplies the methodology:
independent cell splits within constructs, different constructs within a context, then
exact-construct and different-construct transfer across contexts. Distinguish unresolved activity,
scale changes, and reproducible directional disagreement. Responder mixtures require additional
evidence; mean effects and split power do not identify them alone.

Gene coefficients, embeddings, joint source weighting, and hierarchical context/study models are
deferred. Context and study are largely confounded; random-effects assumptions do not create
crossed replication or identify separate biological and technical variance components.

## 10. Deliverables and remaining decisions

Keep this plan's filename for continuity. Proposed results directory:
`reports/average-vs-matched-regression/`, with a README describing the narrower first-analysis scope.

First-analysis outputs:

- source/study/provenance map, verified target intersection, and common gene support;
- frozen effect definition, fitting objective/constraints, target folds, and metric definitions;
- direct and held-out calibrated predictions, coefficients, and per-target metrics;
- wrong-target, common-response, and available positive/null control results;
- paired comparisons, weighting sensitivity, and coefficient stability diagnostics;
- figures for individual-source versus average transfer, calibration, and KOLF's added value;
- input/output hashes, software versions, code commit/dirty state, and prior-analysis references.

Before execution, discuss and settle effect geometry, fitting loss/constraints, exact evaluation
definitions, available reliability controls, and whether count-based scoring belongs in this
first analysis. These are modeling choices, not implementation defaults. Matching-rule selection,
outer-context evaluation, and VCC artifacts are follow-up outputs, not first-post gates.

Large matrices remain gitignored. This revision edits the plan only; it does not initiate analysis,
downloads, cloud operations, submissions, commits, or pushes.

## 11. First-post outline

Working question/title: **How much of a perturbation response transfers between cell contexts?**

1. **Bridge from linear response:** controls supplied little useful response information; now
   use measured responses to the same perturbations elsewhere.
2. **Prediction object:** explain effect representation and biological/experimental differences
   between the five sources and H1, building on the sequencing post.
3. **Direct transfer:** individual donors, broad average, and identity/common-response controls.
4. **First regression:** fit one scalar per predictor; explain what it changes and cannot change.
5. **Biologically motivated donor:** compare KOLF with the average and test incremental held-out value.
6. **Interpretation:** distinguish identity, direction, amplitude, and measurement reliability;
   explain H1-calibrated target holdout versus unseen-context prediction.
7. **Next question:** can controls identify useful donors before seeing destination responses?

Write conclusions from results rather than precommitting to averaging or matching winning.
The first post should establish what simple transferred-response regressions predict and the
limits of that evidence before asking a more flexible model to explain the differences.
