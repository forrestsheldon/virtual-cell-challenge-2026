# Three-model linear-response experiment and VCC submission handoff

**Status:** superseded by `plans/linear-response-h1-diagnostics.md`; retained for history  
**Authority:** historical only; do not execute any instruction below  
**Prepared:** 2026-09-01; revised 2026-09-02  
**Repository:** `/Users/forrestsheldon/Projects/virtual-cell-challenge-2026`  
**Former goal:** benchmark three deliberately small linear-response variants and submit VCC predictions.

## Scope

Implement only these three models:

1. **Normalized CIPHER-style linear response** using empirical covariance columns.
2. **PCA-truncated normalized linear response** using the top control components.
3. **Poisson-lognormal linear response with observed-total offsets** fitted to raw counts, with a low-rank latent covariance and Laplace-MAP inference.

The intended comparison is:

```text
full normalized covariance
    -> top normalized fluctuation modes
    -> count-likelihood-estimated latent modes
```

Do not add an exact conditional multinomial, negative-binomial, Good-Turing, learned residual, foundation-model, or multi-dataset variant in this run. Record tempting extensions under “Deferred work”; do not implement them unless a required model cannot be made valid.

## Scientific question

Can a covariance-column linear-response model learned entirely from unperturbed cells produce useful CRISPRi response directions, and do either low-rank truncation or a count-native latent likelihood improve those directions?

This is a methods/post experiment, not an exhaustive benchmark. The result should support a modest statement about three variants, not a broad claim that one observation model is generally superior.

## Authoritative local context

Read before changing code or assuming challenge details:

- `AGENTS.md`
- `context/vcc2026/README.md`
- `context/vcc2026/challenge.md`
- `context/vcc2026/datasets.md`
- `context/vcc2026/evaluation.md`
- `context/vcc2026/submission.md`
- `context/vcc2026/rules.md`
- `context/vcc2026/tooling.md`
- `data/README.md`
- `submissions/README.md`
- `reports/crispri-h1-exploration/README.md`

Recheck the live CLI help before submission. The installed command is available inside Pixi:

```bash
pixi run vcc --version
pixi run vcc prep --help
pixi run vcc submit --help
```

At plan creation, `pixi run vcc --version` reports `vcc 0.1.0`; bare `vcc` is not on the shell PATH. Always use `pixi run vcc` in repository commands unless the environment changes deliberately.

## Existing data: do not redownload

### VCC 2025 H1

- `data/external/vcc2025_h1/adata_Training.h5ad`
- `data/external/vcc2025_h1/gene_names.csv`
- `data/external/vcc2025_h1/pert_counts_Training.csv`
- 221,273 cells × 18,080 genes, raw UMI counts in sparse `X`
- 150 perturbed targets plus non-targeting controls
- 48 recorded batches
- provenance and checksums are in `data/README.md`

Use `adata.var_names` as the primary H1 gene axis. Verify that the H1 gene CSV is read with the correct header convention and contains exactly 18,080 genes; a naive `pandas.read_csv` call can consume the first gene as a header.

The existing H1 exploration identified a strict pool of 26 apparently inert non-targeting guide identities and five non-inert controls. Use the strict pool for the primary H1 control fit, reading its identifiers from the generated Mixscape/null manifests rather than copying names by hand. Retain an all-NTC sensitivity result only if it is cheap; it is not a fourth model.

Use nominally labelled perturbed cells for the primary benchmark. Do not make Mixscape filtering part of the primary three-model comparison.

### VCC 2026 validation controls

- `data/controls/context_A.h5ad`
- `data/controls/context_B.h5ad`
- `data/controls/context_C.h5ad`
- `data/controls/gene_names.csv`
- `data/controls/pert_counts.csv`
- 18,400 control cells per context
- 18,533 genes in the required order
- 300 targets
- 46 `ntc_id` constructs × 400 cells per context

Preserve `ntc_id` during fitting and QC. Fit a separate susceptibility for A, B, and C; do not pool anonymous contexts.

All 300 validation targets are present on the supplied 2026 gene axis. Only 13 validation targets overlap the 150 H1 perturbed targets, so the submission cannot rely on target-specific H1 effects. Use a single transferable CRISPRi amplitude per model, calibrated from H1 as described below.

## Shared notation and response rule

For target gene `g`, use a one-gene field

\[
u_g = a e_g,
\]

and predict

\[
\delta_g = C u_g = a C_{:g}.
\]

The covariance/susceptibility `C` is context-specific. The scalar amplitude `a` is model-specific because normalized coordinates and latent Poisson log rates have different units.

CRISPRi implies a negative primary amplitude. Constrain the transferred amplitude to `a <= 0` and report any H1 targets whose individually implied amplitude has the opposite sign.

## Shared preprocessing and controls

### Normalized representation for Models 1 and 2

Use one fixed transformation everywhere those models appear:

1. divide each cell by its raw total;
2. multiply by 10,000;
3. apply `log1p`.

Call this `log1CP10k`. Never overwrite raw `X`; process in chunks or write explicitly named derived artifacts.

Do not select HVGs for the response output. All available genes must remain in the covariance column and prediction. PCA computation may use randomized/truncated algorithms, but its loading matrix must span the full gene axis.

### Gene-target exclusion

The official VCC metrics exclude the perturbed target gene, and PDS excludes all panel targets. Do not use the scored downstream genes to calibrate a held-out H1 target’s amplitude. Use the target-only calibration below, then evaluate downstream genes with the target removed.

### Negative baseline

Use the existing control-resampling baseline locally as a sanity reference. Do not spend a VCC scoring submission on another random/control-only artifact unless the user explicitly changes scope.

## Model 1: normalized CIPHER-style LR

For normalized control matrix `Z`, estimate the empirical covariance column for every requested target:

\[
\widehat C_{:g}
=
\frac{1}{n-1} Z_c^T Z_{c,:g},
\]

where `Z_c` is column-centered.

Do not materialize an 18,533 × 18,533 dense covariance. Compute only the 150 H1 or 300 VCC target columns, in blocks if necessary.

Prediction in normalized space:

\[
Z_i' = Z_i + a\widehat C_{:g}.
\]

Use a common raw-count decoder for Models 1 and 2:

1. sample a real control cell and retain its raw total `N_i`;
2. apply the normalized shift to its `log1CP10k` vector;
3. calculate non-negative relative weights with `expm1` and clipping at zero;
4. normalize those weights to probabilities;
5. sample integer counts from `Multinomial(N_i, p_i')`;
6. store sparse CSR counts with no explicit zeros.

If all weights are zero after numerical clipping, fail that target-context generation with a diagnostic rather than silently returning controls.

## Model 2: PCA-truncated normalized LR

Fit PCA/SVD to the same centered normalized controls:

\[
\widehat C_k = V_k\Lambda_kV_k^T.
\]

Use `k = 50` for the primary run. This matches the existing H1 exploratory PCA scale and keeps the comparison legible. A `{20, 100}` control-only stability check is allowed, but do not turn it into extra VCC submissions; freeze `k = 50` unless the control fit is numerically invalid.

For target `g`, compute the direction without forming `C_k`:

\[
d_g = V_k\Lambda_k(V_k^T e_g).
\]

Then apply

\[
Z_i' = Z_i + a d_g
\]

and use exactly the same raw-count decoder as Model 1.

Use deterministic PCA/SVD seeds and record the solver, centering, singular values, explained variance, and versions.

## Model 3: Poisson-lognormal LR with observed-total offsets

Treat each observed raw total `N_i` as a fixed exposure rather than as a response to be modelled:

\[
h_i \sim \mathcal N(0,I_k),
\]

\[
x_i = m + Lh_i,
\]

\[
Y_{ig}\mid N_i,h_i
\sim
\operatorname{Poisson}\left(N_i e^{x_{ig}}\right).
\]

Use `k = 50` to match Model 2. The latent log-rate covariance is

\[
C = LL^T.
\]

This is a Poisson approximation to conditioning on the total. If

\[
S_i = \sum_g e^{x_{ig}}
\]

is close to one, the Poisson and multinomial log likelihoods agree to first order, while the generated total has conditional relative sampling noise approximately `1 / sqrt(N_i)`. At 10,000 counts this is about 1%.

More precisely, their latent-state-dependent difference is

\[
\ell_{\mathrm P}(x_i)-\ell_{\mathrm M}(x_i)
=
N_i\left(\log S_i-S_i\right)+\mathrm{constant}
=
-\frac{N_i}{2}(S_i-1)^2
+O\left((S_i-1)^3\right)
+\mathrm{constant}.
\]

The first-order term cancels; the approximation fails through deviations of `S_i` from one, not through an untracked first-order change.

Anchor the population scale with the lognormal correction

\[
q_g = \exp\left(m_g+\frac12 C_{gg}\right),
\]

and enforce

\[
\sum_g q_g = 1.
\]

An implementation may enforce this after each global update by shifting all entries of `m` by the same log normalizer. Record the exact convention. Also keep latent total-rate variation small: diagnose `S_i` over prior and posterior draws and either constrain or penalize the first-order global loading component `q^T L` if it is not negligible. This condition is what makes the “mild total-count noise” interpretation true rather than assumed.

For target `g`, calculate

\[
d_g = C e_g = L(L_{g:})^T.
\]

The exact perturbed rate is

\[
\lambda_{ij}'
=
N_i\exp\left(m_j+L_{j:}h_i+a d_{g,j}\right),
\]

and generate independently across genes:

\[
Y_{ij}'\sim\operatorname{Poisson}(\lambda_{ij}').
\]

There is no softmax and no small-response approximation in the generator. At the population level,

\[
\mu_j' = \mu_j\exp(aC_{jg}).
\]

Sample offset totals `N_i` from the empirical control totals of the relevant context. The realized Poisson totals may differ mildly from those offsets. For VCC, balance sampled source controls across the 46 `ntc_id` constructs as evenly as practical, following the existing control-resampling baseline’s deterministic allocation.

The perturbation can change the expected total by the factor

\[
R_g(a)=\sum_j q_j\exp(aC_{jg}).
\]

Record this factor for every target. If `q^T L` is controlled, its first-order change is suppressed, but finite-response changes remain possible. Do not silently renormalize perturbed rates; doing so would restore the multinomial/softmax model that this approximation is intended to avoid.

### Primary inference approach

Use Laplace-MAP:

- shrinkage prior/penalty on `L`;
- unique per-cell latent mode from the log-concave posterior;
- inverse latent Hessian for the Laplace correction;
- approximate marginal posterior optimization for `m` and `L`;
- enforcement of the population rate normalization `sum(q) = 1`;
- deterministic initialization and at least two additional starts for the global non-convex factor fit.

Prototype on a small cell/gene subset first. Validate per-cell Laplace moments against MCMC or high-accuracy numerical checks on a tiny latent/gene subproblem before trusting the full fit.

If differentiating or optimizing the full Laplace marginal objective proves genuinely intractable, a Gaussian variational approximation is the allowed fallback. The expected Poisson log likelihood is analytic under a Gaussian variational posterior, which makes this fallback especially natural here. Document the reason, objective, covariance family, and accuracy comparison; do not silently replace Laplace with point-estimated cell factors.

Any new modelling dependency must be added deliberately to `pixi.toml`, locked, and exercised by the smoke tests. Do not install into an unmanaged environment.

## H1 amplitude calibration without target leakage

The response direction comes only from H1 controls. Perturbed H1 cells are used to calibrate a transferable scalar and to evaluate held-out targets.

Use leave-one-perturbation-out calibration:

1. For every H1 target `t`, estimate its observed on-target shift from a raw-count pseudobulk/control comparison in the coordinate system appropriate to the model.
2. For Models 1 and 2, calculate the individually implied scalar from the target coordinate:

   \[
   a_t = \frac{\Delta_{t,t}}{d_{t,t}}.
   \]

3. For Model 3, estimate the observed target shift on the per-cell rate scale `Y_it / N_i`. Under the fitted model,

   \[
   \log\frac{E[Y_{it}/N_i]_{\mathrm{pert}}}{E[Y_{it}/N_i]_{\mathrm{ctrl}}}
   = a_t C_{tt},
   \]

   so use the corresponding one-dimensional target-only likelihood or moment estimate of `a_t`. Document the pseudocount or shrinkage used for a zero or very small target mean; do not fit downstream genes.
4. Exclude only targets with a predeclared numerical failure such as zero control detection, non-finite direction, or effectively zero `d_{t,t}`. Record every exclusion and threshold.
5. For held-out target `t`, set `a` to the median negative `a_j` over all eligible `j != t`.
6. Evaluate target `t` on downstream genes, excluding the perturbed target.
7. For VCC submission, refit the model-specific scalar as the median negative `a_t` over all eligible H1 targets and freeze it for all 300 targets and all A/B/C contexts.

This is intentionally simple. Do not fit target-specific amplitudes for the 13 overlapping VCC targets in the primary submission; doing so would give those targets a qualitatively different information budget.

Report the distribution of `a_t`, median, MAD/bootstrap interval, eligibility count, and sign failures. If the global scalar is unstable under target bootstrap or depends on a few targets, stop and report rather than hiding it with clipping.

## H1 benchmark

### Evaluation set

Use H1 targets with at least 400 nominally perturbed cells for the primary 400-cell truth comparison. The local capacity report suggests 126 targets meet this threshold; recompute from the H5AD rather than hard-coding the count.

For each target:

- sample 400 truth cells deterministically;
- generate 400 prediction cells;
- use the strict control pool as the reference;
- apply leave-one-target-out amplitude calibration;
- exclude the target gene from downstream metrics.

Use several fixed prediction seeds, preferably `{0, 1, 2, 3, 4}`, to separate generator Monte Carlo variation from model differences. Do not refit the susceptibility per seed.

### Primary metrics

Keep the local benchmark compact:

1. cosine similarity of predicted and observed normalized/log1p pseudobulk effects;
2. perturbation-discrimination/PDS-style retrieval across the H1 panel;
3. normalized/log1p pseudobulk MSE relative to the unchanged-control baseline;
4. Spearman correlation of predicted and observed downstream response vectors as an interpretable secondary statistic.

Use the current VCC 2026 normalization conventions where they apply, but label this as an H1 adaptation rather than an official VCC score. The H1 panel and gene axis differ from the 2026 validation contract.

Report per-target results, medians, bootstrap intervals over targets, and response-strength strata. Include the unchanged-control baseline. Do not select the winning model using the VCC leaderboard alone.

### H1 outputs

Write reproducible, small tracked outputs under:

```text
reports/linear-response-three-models/
    README.md
    h1_summary.csv
    h1_per_target.csv
    amplitude_calibration.csv
    control_fit_diagnostics.csv
    run_manifest.json
    figures/
```

Large generated H5ADs, checkpoints, covariance columns, and factor matrices belong under ignored `data/derived/` or `submissions/`, with paths and hashes recorded in the manifest.

## Control-fit and generator checks

Run before any VCC packaging:

- exact gene order and target presence;
- finite response columns and predicted Poisson rates;
- correct CRISPRi target-direction sign for the calibrated response;
- control split-half stability of target covariance columns;
- Model 2 singular spectrum and reconstruction diagnostics;
- Model 3 held-out control log likelihood or deviance;
- Model 3 posterior-predictive zero fractions, means, variances, and dominant modes;
- Model 3 baseline latent-rate sums `S_i`, population normalization `sum(q)`, and excess total-count variation;
- Model 3 per-target expected total multipliers `R_g(a)` before generation;
- generated total-count distribution versus source controls;
- perturbation-specific predictions are not all identical;
- no generated cell exceeds 1,000,000 total counts;
- no explicit stored zeros in sparse output;
- reproducibility under fixed seeds.

If Model 3 fails its control posterior-predictive checks, do not add NB or mixture machinery in this scope. Report the failure and decide with the user whether to submit the misspecified model as the planned third variant.

## VCC 2026 fitting and prediction

For each model and each of contexts A/B/C:

1. Load raw controls and preserve official gene order.
2. Fit/estimate the context-specific control susceptibility using controls only.
3. Apply the frozen model-specific H1 amplitude to each of the 300 targets.
4. Generate 400 raw-count cells per target.
5. Use deterministic, model/context/target-derived seeds.
6. Combine outputs into exactly 360,000 rows × 18,533 columns.
7. Set `obs` to exactly the required `target_gene` and `context` fields unless an additional field is proven accepted and useful.
8. Include no non-targeting/control rows.
9. Store `X` as sparse, finite, non-negative, whole-valued raw counts.
10. Set `var_names` to `data/controls/gene_names.csv` exactly and in order.

Generate in chunks by context and target; do not hold a dense 360,000 × 18,533 matrix in memory.

Suggested artifact layout:

```text
submissions/lr-cipher-normalized-v1/
    prediction.h5ad
    prediction.vcc
    manifest.json
    prep.json
    submit.json
    status.json

submissions/lr-pca50-normalized-v1/
    ...

submissions/lr-poisson-lognormal50-v1/
    ...
```

The entire `submissions/` data area is gitignored. Record enough metadata in the tracked report to reproduce each artifact.

## Packaging and submission

### Preflight

```bash
pixi install --frozen
pixi run check
pixi run vcc --version
pixi run vcc whoami
```

If authentication is missing, ask the user to authenticate in their own terminal with `pixi run vcc login --token-stdin`. Never ask for or record an API token.

For every model, validate without upload first:

```bash
pixi run vcc prep submissions/<run>/prediction.h5ad \
  -g data/controls/gene_names.csv \
  --perts data/controls/pert_counts.csv \
  -o submissions/<run>/prediction.vcc \
  --dry-run --json > submissions/<run>/prep.json
```

After a clean dry run, create the package by removing `--dry-run`. Do not use `--no-verify-targets`, `--no-check-cell-counts`, `--no-require-counts`, or `--allow-controls` to bypass a failure.

### Submission order and quota

The current validation limit is two scoring submissions per UTC day, resetting at 00:00 UTC, with only one submission in flight at a time. Three model submissions therefore require at least two UTC dates.

Submit in increasing-complexity order:

1. `lr-cipher-normalized-v1`
2. `lr-pca50-normalized-v1`
3. `lr-poisson-lognormal50-v1`

Use concise model names and descriptions that record the H1 calibration and context-specific control fit. Submit one, wait for a terminal status, then submit the next:

```bash
pixi run vcc submit submissions/<run>/prediction.vcc \
  -m "<concise model name>" \
  -d "<model, rank, H1 global amplitude, seed, code revision>" \
  --json

pixi run vcc status <entry-id> --wait --json
```

Do not retry an HTTP 409; it means another team submission is still in flight. Use `--resume` only for an interrupted upload. Do not redirect the released CLI to another endpoint.

Record entry ID, UTC submission time, CLI version, artifact checksum, terminal status, and all returned scores in the run manifest and tracked report. A submission is complete only after it reaches scoring/published status.

## Implementation layout

Keep reusable code out of notebooks. A suggested structure is:

```text
src/vcc_lr/
    io.py
    normalize.py
    amplitude.py
    generation.py
    metrics.py
    models/
        cipher_normalized.py
        pca_normalized.py
        poisson_lognormal_latent.py

scripts/linear_response/
    benchmark_h1.py
    fit_vcc_controls.py
    build_submission.py
    summarize_runs.py

configs/linear_response/
    h1.yaml
    vcc2026_validation.yaml

tests/
    test_lr_normalization.py
    test_lr_response_columns.py
    test_lr_generation.py
    test_lr_submission_schema.py
```

This layout is guidance, not a requirement. Preserve unrelated user changes and do not commit or push without explicit instruction.

## Minimum tests

- covariance-column calculation matches a small dense covariance;
- PCA response calculation matches `V_k @ diag(lambda_k) @ V_k.T @ e_g`;
- Model 3 population mean rates `q` sum to one after every saved fit;
- Model 3 Poisson rates remain finite and non-negative under response shifts;
- Model 3 empirical baseline total-count noise agrees with the fitted `S_i` distribution and Poisson variance;
- fixed seeds reproduce generated counts;
- generated counts are non-negative integers; Models 1 and 2 preserve requested totals exactly, while Model 3 has documented Poisson and latent-rate total variation around its offsets;
- response amplitude is fit without the held-out H1 target;
- target gene is excluded from H1 downstream metrics;
- H1 and VCC gene-axis readers handle their different CSV/header conventions;
- output contains exactly 400 cells per target-context pair;
- output contains exactly A/B/C and all 300 targets;
- output gene order exactly matches the official CSV;
- sparse output contains no explicit zeros;
- a tiny synthetic H5AD passes `vcc prep --dry-run` semantics where feasible, followed by full-artifact dry runs.

## Stage gates

### Gate 1: data and split manifest

Proceed only after the H1 control identities, target counts, gene axes, seeds, and VCC context dimensions have been written to a manifest and checked against local documentation.

### Gate 2: H1 model validity

Proceed only after all three models produce finite, perturbation-specific H1 predictions and the global amplitude calibration is stable enough to report.

### Gate 3: count generator validity

Proceed only after generated counts pass raw-count, total-count, sparsity, and reproducibility checks.

### Gate 4: full VCC dry run

Proceed to upload only after every `.h5ad` passes `pixi run vcc prep ... --dry-run` with all default target, count, context, and raw-count checks enabled.

### Gate 5: submissions

Submit all three planned models subject to the two-per-UTC-day and one-in-flight limits. If authentication, quota, or server state prevents progress, preserve ready artifacts and exact commands and report the blocker.

## Deliverable for the eventual post

The final write-up should be able to show:

1. the shared linear-response equation;
2. empirical normalized covariance-column response;
3. top-mode truncation and its spectral interpretation;
4. the count-native Poisson-lognormal formulation with observed totals as offsets;
5. H1 held-out-target results for the three models;
6. A/B/C leaderboard results with identical information boundaries;
7. limitations: global amplitude transfer, use of observed totals as offsets, possible latent/perturbed total-rate drift, population-versus-dynamical covariance, and anonymous-context generalization.

Do not present leaderboard differences as proof of causal regulatory inference.

## Deferred work

- exact conditional logistic-normal multinomial model;
- independently or jointly modelled biological total abundance and capture efficiency;
- negative-binomial observation sensitivity;
- Good-Turing coverage diagnostics;
- target- or feature-dependent amplitude models;
- learned perturbing fields;
- regression or ML residual around `C u`;
- mixtures, flows, or response-dependent covariance;
- additional external CRISPRi datasets.

These are follow-ups after the three-model run, not prerequisites.
