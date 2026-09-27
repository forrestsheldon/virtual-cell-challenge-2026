# What transfers between screens? Split-power analysis

**Status:** revised analysis plan for review; compact cell-split core  
**Prepared:** 2026-09-23; revised 2026-09-24  
**Scope:** retrospective CRISPRi analysis; no VCC submission, download, or model change  
**Artifact contract:** `plans/context-transfer-split-artifacts.md`

## Editorial thesis

Biological perturbation effects may transfer between screens, with genuine changes across
cellular contexts. The current analysis estimates reproducible signal and split-to-split noise for each
perturbation, then asks how those estimates explain construct disagreement. Responder mixtures
are not decomposed in this phase. Paired-context transfer is a subsequent application of the
validated methodology, not a requirement to develop a responder model.

The opening observation is deliberately prior to cross-screen modeling: distinct constructs
assigned to the same target can agree little better than wrong-target controls even within one
context. That result is provisional until it is calibrated against independent cell splits.
The current `P1`, `P2`, and `P1P2` labels are author-provided design classes, not yet verified
promoter annotations, so the analysis will call them **encoded P-classes** until the library
design tables or guide coordinates establish their meaning.

The motivating pooled results are retained in
`reports/context-transfer-blog/metric_calibration_v2_summary.csv`; this plan does not treat those
results as reliability-corrected evidence.

## Questions, in order

1. Does each construct produce a response that replicates across disjoint cells?
2. When two constructs disagree, is one below the experiment's signal floor, are they scaled
   versions of one response, or do they contain reproducible different directions?
3. How do signal/noise estimates change with cell count, exposure, and quarter-versus-half splits?
4. Once calibrated, how much construct agreement and paired-context transfer is supported by
   reproducible signal?

Responder detection and new regression modeling are deferred. They are not gates or deliverables
for establishing this methodology. See `plans/context-transfer-responders-deferred.md`.

Each question is a gate. A later interpretation is not attempted when the earlier signal needed
to define it is absent.

## Artifact scope and what a split measures

The core release pools expression across source batches into four cell splits per exact
construct. It also retains control identity × split sums, batch × control-role × split sums,
and fine-resolution batch counts/exposures **without** fine-resolution expression. Batch-aware
assignment does not require four cells per batch: sparse strata stay in the release and final
support is assessed after pooling. Atomic quarters may be recombined into halves.

Use broad exact shared scPerturb constructs. X-Atlas retains the 415-target union, with an
optional top100/top200 multi-construct-gene augmentation as pseudobulks only. Forrest selected
ranking by the second-best shared construct's weaker-context cell count; panel size remains a
choice. See the artifact proposal and revised preflight for exact ranking, lists and costs.

The first analysis measures cell-split reliability conditional on the observed experiment.
Recorded batches are capture groups or source sample labels, not established biological
replicates. Shared systematic effects may reproduce across cell halves. Whole-batch held-out
validation is a separately approved extension. Responder detection remains deferred. Retain raw responder-ready cell subsets with all eligible
controls under the artifact contract for later research; no automatic two-week expiry. This
retention does not add responder fitting to the current analysis.
No arbitrary batch bootstrap or perturbation-batch reweighting is implied by the core release.

## 1. Fix the estimand before choosing a transformation

The primary effect is a difference of linear count rates. For perturbation `p`, control `0`,
atomic split `r`, and a frozen exposure `E`, define

\[
d_{pr}=\frac{Y_{pr}}{E_{pr}}-\frac{Y_{0r}}{E_{0r}}.
\]

With total UMI as `E`, this is a difference in transcript composition and is CPM up to a constant.
With a frozen sum of cell size factors as `E`, it is expression per unit capture exposure. These
are different estimands and must not be mixed. The first analysis will use total UMI because it is
exactly defined by the audited count artifacts and matches the existing native-universe CPM
analysis.

Conditional on exposure, Poisson, multinomial, and negative-binomial count models all give a
zero-centered rate-estimation error:

\[
d_{pr}=\delta_p+\varepsilon_{pr},\qquad
\mathbb E(\varepsilon_{pr}\mid E)=0.
\]

The split-power argument needs conditional zero mean, cell-disjoint perturbation and baseline
splits, and a common underlying effect across splits. It does not need Gaussian gene counts or
independent genes. Cell disjointness alone does not establish independence of shared technical
or biological batch effects.

For batch matching, let E_pbr be perturbation UMI exposure in batch b and split r and set
w_pbr = E_pbr / E_pr. Subtract the weighted sum of batch × baseline-role × split control rates
from the pooled perturbation rate. Both control rates and the exposure weights are recoverable
from the compact core. Report this matched estimand distinctly from subtraction of a global
pooled control rate. A missing required control batch/split makes the matched estimate unavailable;
the pooled perturbation vector cannot be retrospectively restricted to supported batches.

Weights can differ across splits. Matching baseline batches removes an additive baseline
composition difference under that model; it does not force batch-dependent perturbation effects
to be equal. If split effects differ, the off-diagonal product estimates their shared component,
not automatically the squared norm of a single population effect. Examine batch compositions
and matched nulls before a stronger interpretation. Do not silently filter n<4 strata.

`log2(CPM+1)` remains a named sensitivity and visualization coordinate. It is not primary for
power estimation because its expected value depends on depth through transformation bias. The
epsilon log fold change remains a biological effect display where required, not a split-power
coordinate.

## 2. Estimate signal and noise with off-diagonal products

For `K` disjoint cell splits, estimate unweighted signal power with

\[
\widehat S_p=
\frac{1}{K(K-1)}\sum_{r\ne s}d_{pr}^{\mathsf T}d_{ps}.
\]

Observed power and split noise power are

\[
\widehat Q_p=\frac1K\sum_r d_{pr}^{\mathsf T}d_{pr},\qquad
\widehat V_p=\widehat Q_p-\widehat S_p.
\]

Under the stated independence/common-effect assumptions, the off-diagonal estimator removes
the positive squared-noise contribution that makes an inactive high-dimensional vector look large. Negative finite-sample estimates of `S` are retained; they
mean that positive shared signal is unresolved by the available splits. If true effects differ
across split batch mixtures, V also contains that heterogeneity and must not be labelled pure
measurement noise.

Report, per construct:

- signal power `S`;
- noise power `V`;
- split signal-to-noise ratio `S/V`;
- reliability of one split, `S/Q`;
- reliability of the pooled pseudobulk only under justified split/exposure weighting; do not
  assume a fourfold noise reduction with unequal or heterogeneous splits;
- target-gene repression as a separate activity QC, never as an evaluation coordinate.

Ratios whose estimated signal or noise denominator is non-positive remain undefined. The signed
power estimates and their intervals are reported rather than repaired by truncation.

## 3. Standardize heteroskedastic genes without restoring the noise floor

Let `v_g` be the variance of the perturbation-minus-control rate estimate for gene `g` at a common
reference cell count and exposure. Freeze a shrunken diagonal matrix

\[
D=\operatorname{diag}(\widetilde v_1,\ldots,\widetilde v_G),
\]

where the mean-variance trend and floor are estimated from independent controls. The primary
standardized power is the cross-split Mahalanobis product

\[
\widehat S^{M}_p=
\frac{1}{K(K-1)}\sum_{r\ne s}d_{pr}^{\mathsf T}D^{-1}d_{ps}.
\]

An ordinary same-split Mahalanobis norm has a null expectation near the gene dimension. The
cross-split product has null expectation zero. It therefore handles both heteroskedastic gene
counts and high-dimensional noise without a log transform.

Two versions will be kept distinct:

- **common-reference geometry:** one frozen `D` at a common depth and cell count for biological
  comparisons across contexts;
- **actual-design detectability:** variances at each experiment's observed depth and cell count,
  reported as an experimental-resolution diagnostic.

The diagonal metric is primary. A full covariance inverse is deferred. Residual gene correlation
is calibrated with wrong-target and non-targeting controls; a low-rank-plus-diagonal covariance is
considered only if those controls show that diagonal standardization remains materially distorted.

### Relation to the natural Fisher geometry of count rates

The linear rate difference is also a local approximation to the natural geometry of a
multiplicative count model. For one gene, suppose

\[
Y_p\sim\operatorname{Poisson}(E_p\mu_p),\qquad
Y_0\sim\operatorname{Poisson}(E_0\mu_0),\qquad
\beta=\log(\mu_p/\mu_0).
\]

Near the null `beta = 0`, the linear effect satisfies

\[
\delta=\mu_p-\mu_0\approx\mu_0\beta,
\]

while the rate-difference estimator has variance

\[
v=\mu_0\left(\frac1{E_p}+\frac1{E_0}\right).
\]

Therefore its standardized squared displacement is

\[
\frac{\delta^2}{v}
\approx
\frac{E_pE_0}{E_p+E_0}\mu_0\beta^2
=I_\beta\beta^2,
\]

where `I_beta` is the efficient Fisher information for the log-rate ratio after allowing the
shared baseline rate to be unknown. Thus `D^{-1}` does more than put genes on comparable numerical
scales: locally, it maps linear rate differences into the Fisher-information metric of the
underlying log-rate model. The cross-split product estimates reproducible displacement in that
metric without explicitly taking unstable low-count log ratios.

This equivalence is local and model-dependent. Large fold changes, overdispersion, zero inflation,
gene covariance, compositional exposure, and estimated variance trends can make the Poisson
quadratic approximation inaccurate. The common-reference `D` fixes one tangent-space geometry for
cross-context biological comparison; the actual-design `D` instead measures detectability under
each experiment's exposure. Held-out Poisson or negative-binomial deviance is a natural
multiplicative sensitivity analysis if the local approximation appears inadequate.

## 4. Separate construct scale, direction, and inactivity

For two constructs or encoded P-classes, use only off-diagonal split pairs to estimate shared
signal:

\[
\widehat C_{12}=
\frac{1}{K(K-1)}\sum_{r\ne s}d_{1r}^{\mathsf T}D^{-1}d_{2s}.
\]

Excluding `r=s` prevents shared control noise from entering both effects. When both signal powers
are positive with intervals away from zero, compute

\[
\widehat\rho_M=\frac{\widehat C_{12}}
{\sqrt{\widehat S^M_1\widehat S^M_2}},\qquad
\widehat\beta_{2\leftarrow1}=\frac{\widehat C_{12}}{\widehat S^M_1}.
\]

Interpretation is ordered:

| Evidence | Interpretation |
|---|---|
| `S1 > 0`, `S2` unresolved | construct 2 is inactive or below resolution; its direction is undefined |
| `S1,S2 > 0`, `rho_M` near 1, `beta != 1` | common direction with different scale |
| `S1,S2 > 0`, `rho_M` low | reproducible construct-specific directions |
| both powers unresolved | the experiment cannot adjudicate construct agreement |

Do not noise-correct a cosine by dividing by near-zero reliability. The signal-power gate comes
first. The core supports prespecified cell-split comparisons and paired target-level resampling;
uncertainty for an individual construct is limited by its finite set of retained splits. Do not
claim arbitrary cell bootstrap or batch bootstrap from these pseudobulks. A separately retained
cell object or approved batch-fold extension is needed for those stronger resampling analyses.
Recorded-batch variation is not biological-culture variation without experimental evidence.

## 5. Subsequent application: transfer ladder

Run each comparison first in raw linear-rate geometry and then in the frozen standardized geometry.

1. **Within construct, within context:** independent cell splits. This is the reliability ceiling.
2. **Between constructs, within context:** exact target, different construct or encoded P-class.
3. **Exact construct across paired contexts:** construct reproducibility plus context change.
4. **Different constructs across paired contexts:** gene-level biological transfer.
The current deliverable is the signal/noise methodology and per-perturbation estimates.
Apply this ladder only after calibration passes. Do not claim to identify responder dilution,
construct-specific biology and context-specific biology as separately identifiable additive
components from cell-split power alone. Regression and responder-conditioned transfer remain
future work.

## 6. Calibration controls and gates

### Required controls

- exact self-transfer reconstructs power, scale 1, and direction 1;
- disjoint cell splits of one construct are the positive reliability control;
- wrong-target constructs are the negative direction/retrieval control;
- independently split non-targeting guide identities are the null power control;
- split labels and target labels are permuted as implementation controls;
- shared-control and independent-control calculations demonstrate the expected covariance bias;
- pooled counts exactly equal the sum of atomic split counts;
- identity × split and batch × role × split control margins reconstruct the same control total;
- missing matched-control support is flagged and cannot trigger retrospective perturbation-batch
  filtering or reuse of baseline cells across otherwise independent splits.

### Decision gates

1. **Count-power gate:** positive controls separate from non-targeting and wrong-target nulls.
2. **Construct gate:** at least one member has reproducible signal before scale or direction is
   interpreted.
3. **Transfer gate (subsequent application):** noise removal improves cross-context target retrieval or standardized shared
   signal, not only norm error or expression MSE.

Failure at a gate is a result and stops that branch. It does not trigger another representation,
filter, or model until the failure is understood.

## 7. Compact outputs

The analysis should produce one table per estimand rather than repeated ad hoc reports:

- per-construct power, noise, reliability, depth, and activity QC;
- per-target construct-pair shared power, scale, and direction;
- empirical null and effective-dimension summaries;
- quarter-versus-half signal/noise diagnostics; paired-context shared signal after calibration;
- paired target bootstraps; predefined batch-fold validation only with an explicitly retained extension;
- a manifest linking every result to source and split-artifact hashes.

Core inference is perturbation-level or conditional cell-split reliability; recorded-batch inference requires the corresponding extension. Gene entries are not treated as independent replicates,
and no gene-entry p-values are reported.

Headline figures should show signal, noise, uncertainty and construct agreement before and after
reliability calibration. Ordinary MSE remains an official-contract diagnostic and cannot by
itself establish biological transfer.

## 8. Claims this design can and cannot support

If successful, the analysis can show that a component of CRISPRi response is reproducible and
has measurable shared signal across constructs or paired contexts after accounting for sampling
noise under the stated assumptions. It can distinguish
failure to resolve a perturbation from reproducible directional disagreement.

It cannot prove that an encoded P-class is a biological promoter without external library
annotation; distinguish failed perturbations from responder dilution; establish responder
identities or fractions; or identify absolute transcript changes from UMI composition alone.

## Current reporting sequence

1. Calibrate signal/noise estimation on controls and disjoint cell splits.
2. Characterize per-perturbation signal, noise, depth and uncertainty.
3. Explain construct disagreement where signal is resolved; apply to paired-context transfer.
4. Revisit responder detection only after this methodology is established and separately approved.
