# KOLF → H1: three fixed transfer rules

Run on 2026-09-25. No H1 perturbation responses were used for fitting, scaling, or selecting
parameters. This is a retrospective comparison of predicted mean expression, not a full VCC
score or a model of single-cell distributions.

## Result

| Rule | Mean retrieval ↑ | Mean effect cosine ↑ | Signed top-100 recovery ↑ | Profile squared error / unchanged ↓ |
|---|---:|---:|---:|---:|
| Unchanged H1 controls | 0.5000 | 0.0000 | 0.0000 | 1.0000 |
| Additive | 0.7035 | 0.0574 | 0.0259 | 8.9678 |
| Ratio | 0.7359 | 0.0646 | 0.0296 | 5.8791 |
| Log1p difference | 0.7384 | 0.0690 | 0.0367 | 5.0587 |

All three carry correct-target information: the mean retrieval scores over 20 wrong-target
shuffles were 0.4883, 0.4908, and 0.4911 respectively, and wrong-target mean cosines were
0.0013–0.0020. Target-independent mean-profile controls each scored exactly 0.5 on retrieval.
However, overall alignment remains weak and every rule substantially worsens squared error
relative to unchanged controls. Mean predicted/truth effect-norm ratios are 4.81, 3.79, and 3.48
for additive, ratio, and log1p respectively. This establishes an amplitude mismatch, not its cause.

Log1p minus ratio, paired target-bootstrap differences (95% intervals):

- retrieval: +0.00253 [-0.00033, +0.00540]; no clear separation;
- cosine: +0.00446 [+0.00334, +0.00575];
- signed top-100 recovery: +0.00707 [+0.00463, +0.00984];
- profile squared-error ratio: -0.82036 [-1.17431, -0.60146].

Log1p minus additive retrieval is +0.03492 [+0.00313, +0.07031]. These intervals condition on
these screens and their fixed control estimates; they do not measure generalization to another
context or laboratory. The comparison does not establish a biological saturation mechanism.

## Inputs and support

- KOLF: Nourreddine et al. author-QC-filtered raw UMI counts, locally aggregated into target
  pseudobulks and pooled NTC controls. Source URL, source hashes, aggregation details, and original
  manifest are preserved inside `manifest.json`. All available constructs/batches in the compact
  target artifact are pooled; no new responder, batch, or guide-quality filtering was performed.
- H1: the existing fixed benchmark sample of 400 cells per target, plus all 38,176 NTC cells,
  from VCC 2025 training data. Count sums and cell identities trace to the existing H1 aggregation
  manifest and `reports/vcc2026-h1/reference_cells.csv`.
- 123 common targets. CAST, CHMP3, and TAZ are absent from KOLF overlap. KOLF contributes
  78–636 cells per target (median 242). This is not the smaller five-source 110-target intersection.
- 17,603 shared gene symbols; duplicate symbols summed in raw count space. H1's 477 other genes
  are excluded. **All CPM denominators use this common 17,603-gene support**, rather than each
  dataset's native panel. No HVG selection, batch correction, or model fitting.
- Controls are pooled count sums, not equal-weight averages over control constructs. This is
  the existing artifact definition, not the official competition's construct-balanced baseline.

## Exact transfer process

Let `k0`, `kp`, and `h0` be KOLF control, KOLF perturbed, and H1 control CPM. Each is formed by
summing raw counts, restricting to shared genes, then normalizing the sum to one million.
This is log of pooled expression where applicable, **not mean per-cell log expression**.

1. Additive: `raw = h0 + kp - k0`.
2. Ratio: `raw = h0 * (kp / k0)`. For `k0 == 0`, set the ratio to **1** (no donor evidence).
3. Log1p: transfer `log2(kp + 1) - log2(k0 + 1)` onto `log2(h0 + 1)`, then invert.
   Algebraically: `raw = (h0 + 1) * (kp + 1) / (k0 + 1) - 1`. The pseudocount is **1 CPM**.

For every rule, floor negative **rates** to zero, then renormalize the resulting profile to
one million. Negative log changes are not clipped. `profiles.npz` preserves pre-floor and final
predictions. The normalization includes the target gene; metric exclusion occurs afterward.

### Zeros and boundaries

KOLF controls are zero for 427 shared genes. Across the 123 KOLF perturbations these genes have
only 38 nonzero entries, each one UMI. They include PEG3 and KDM5D, which have appreciable H1
expression and stable H1 responses. They are retained: the ratio fallback supplies no change
before closure; additive/log1p retain the observed singleton changes. Renormalization can still
shift fallback genes along with the rest of the profile. H1 control zeros also remain: pure ratio
transfer cannot activate a gene whose H1 baseline is zero.

| Rule | Median genes floored per target | Median removed negative mass (CPM) | Range of final scaling factor |
|---|---:|---:|---:|
| Additive | 2,991 | 7,992.53 | 0.9757–0.9966 |
| Ratio | 0 | 0 | 0.9762–1.0203 |
| Log1p | 2,455 | 320.50 | 0.9760–1.0212 |

Thus flooring is frequent, though its expression mass is much smaller for log1p. This boundary
handling is part of each tested predictor. Restricting evaluation to donor-control-positive genes
without changing predictions or normalization alters mean retrieval by at most 0.000067 and
cosine by at most 0.000033; see `donor_positive_summary.csv`.

## Evaluation

Every method is scored in the same `ln(1 + CPM/20)` space (log1p counts per 50,000), subtracting
the transformed H1 control profile. All 123 panel-target genes are excluded from **all** primary
metrics, leaving 17,480 response genes. This conservative common axis also prevents shuffled
source targets from contributing direct-knockdown information.

- Retrieval: cosine similarity of each predicted effect to all 123 true effects; matching-target
  midrank converted to `1 - rank/122`, with zero-vector ties scoring 0.5. Actual results were checked
  against the installed cell-eval2 0.16.0 kernel. This is raw retrieval on this restricted H1 panel,
  not a reference-scaled competition score.
- Cosine: matching effect directions; zero predictions score 0 by convention.
- Signed top-100: fraction of the truth's 100 largest absolute effect coordinates also in the
  predicted top 100 with matching nonzero signs. Ties use stable gene-order sorting.
- Profile error ratio: sum of squared predicted-minus-true effects across targets and genes,
  divided by summed truth effect energy. Unchanged equals 1. No jackknife correction or reference
  scaling; this is **not** the official unbiased MSE statistic.
- Additional tables retain absolute-error/norm diagnostics and direction/error on the existing
  stable strong-DE gene set (74 targets with >=10 retained genes). These use the current profile
  effects, not official per-cell DE LFCs or predicted significance calls.
- 2,000 paired target bootstraps. Twenty fixed derangements supply wrong-target diagnostics;
  shuffling predictions is equivalent to shuffling donor perturbations because the H1 baseline
  is identical for every target. Common-response controls average predicted profiles equally
  across source targets, using no H1 perturbation truth.

## Stochastic rounding

Final CPM profiles are converted to expected per-cell counts at the **H1 control mean depth on
shared genes: 56,528.2235 UMIs**. No target-specific H1 perturbation depth is used.
For each gene with expected count `lambda`, 400 independent stochastic-rounded cells have pooled
count `400*floor(lambda) + Binomial(400, lambda-floor(lambda))`. We sample this exact aggregate
law, avoiding a large cell matrix; it is equivalent to independent per-cell/per-gene rounding
for these mean-profile checks. We do not subsequently alter integer counts to force a library total.
Evaluation normalizes their pooled counts as it does the expected profiles.

Across five seeds, aggregate mean signed count error is between -0.000021 and +0.000018 UMI per
gene per cell. Mean absolute error is 0.0113–0.0119, versus 0.1734–0.1780 for deterministic per-cell
rounding. Realized library deviations are at most 0.015%. The rounding error has no detected global
bias; subsequent log/normalization is nonlinear and can still add finite-sample error.

| Rule | Retrieval range over seeds | Profile error ratio range |
|---|---:|---:|
| Additive | 0.7029–0.7037 | 8.9892–8.9902 |
| Ratio | 0.7358–0.7376 | 5.9001–5.9015 |
| Log1p | 0.7378–0.7389 | 5.0812–5.0822 |

Unchanged-control rounding adds sampling noise: its error ratio becomes 1.0224–1.0231 and retrieval
ranges 0.4459–0.5101 across five realizations. It does not create informative predictions. Rounding
supplies integerization noise only, not biological variance, covariance, or responder mixtures.

## Reproduction and checks

```sh
pixi run python -m pytest -q tests/test_kolf_h1_transfer_rules.py
pixi run python -m scripts.evaluation.kolf_h1_transfer_rules
```

Eight tests pass, including transfer identities, inverse-log algebra, zero handling, boundary
accounting, target exclusion, positive/negative retrieval controls, scorer parity, and rounding
expectation. Real-output checks in `validation.json` verify declared hashes, inverse-log transfer,
boundary reconstruction, integer supports, and all four actual retrieval profiles against the
installed scoring kernel. Exact truth transfer is an implementation control, not a replicate ceiling.

Code: `scripts/evaluation/kolf_h1_transfer_rules.py`. Small tables/figures are in this directory;
large profile arrays and one seed's integer pseudobulks are gitignored under
`data/derived/kolf_h1_transfer_rules/`. `manifest.json` records software, source lineage, input and
output SHA-256 hashes, and code state. The native-panel CPM values in the earlier
`zero_control_gene_audit.csv` are preflight diagnostics, not the common-panel inputs used here.
No downloads, cloud operations, commits, pushes, submissions, or full cell-based scoring occurred.

## Follow-up: large-error audit

The [normalization and sampling audit](mse-audit/README.md) shows that final renormalization
slightly reduces error (log1p 5.0743 before versus 5.0587 after). A zero-biological-response
multinomial donor sampled at observed KOLF depths yields mean log1p error ratio 4.4897.
Thus transferred technical count noise is a substantial candidate explanation for the large
prediction energy; it should not be interpreted as purely biological amplitude mismatch.
This diagnostic does not replace independent cell-split reliability estimation.
