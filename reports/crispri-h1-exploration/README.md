# CRISPRi H1 exploration

This report uses the official VCC 2025 H1 training split documented in
[`data/README.md`](../../data/README.md). Raw counts are never overwritten.

## Default UMAP

Run:

```bash
pixi run python scripts/exploration/explore_vcc2025_h1.py
```

The script selects 2,000 Seurat-flavor HVGs from a seeded 30,000-cell sample,
normalizes every cell to 10,000 counts, applies `log1p`, scales and clips at 10,
calculates 50 PCs, builds a 15-neighbor graph from PC1–PC30, and calculates the
default Scanpy UMAP (`min_dist=0.5`, seed 0). All 221,273 cells are included in
PCA and UMAP. These plots give an impression of the data structure; they are not
evidence of perturbation success or faithfully preserved geometry.

The three UMAP figures share one coordinate system:

- `umap_all.png`
- `umap_control.png`
- `umap_perturbed.png`

`embedding.parquet`, `highly_variable_genes.csv`, `pca_variance.csv`, and
`run.json` retain the reproducible derived values and parameters.

## Cell-level differential expression

Run:

```bash
pixi run python scripts/exploration/wilcoxon_h1_de.py
pixi run python scripts/exploration/plot_h1_de.py
```

For every perturbation, the script compares all nominally labelled cells with
all 38,176 non-targeting cells. Raw counts are normalized to 10,000 counts per
cell and log1p transformed. It uses a two-sided Mann–Whitney U/Wilcoxon
rank-sum test and applies Benjamini–Hochberg correction across all 18,080 genes
separately within each perturbation. The implementation uses `pdex` 0.3.0 and
keeps the shared control reference in memory while loading one perturbation at
a time.

At FDR ≤ 0.05, the number of DE genes ranges from 17 to 11,275 (median 2,111).
Twenty-four of 150 perturbations have at most 100 DE genes, 47 have at most 500,
and 57 have at most 1,000. Every perturbation significantly represses its own
target gene, including all 24 perturbations with at most 100 DE genes. The
low-response group therefore cannot be explained by a simple absence of
population-level target knockdown, although cell-level heterogeneity remains
unresolved until Mixscape.

Outputs:

- `wilcoxon_de.parquet`: all 2,712,000 perturbation–gene results;
- `wilcoxon_summary.csv`: DE counts and on-target diagnostics by perturbation;
- `response_spectrum.png`: nested FDR, fold-change, and AUC criteria for every
  perturbation;
- `response_diagnostics.png`: cell count, target-transcript detection, and
  up/down-regulation diagnostics;
- `representative_effects.png`: gene-level effects for LAD1, KLF10, and
  SMARCA4; and
- `wilcoxon_run.json`: transformations, test, threshold, and sample counts.

The p-values describe separation within this pooled screen and treat cells as
the observations. They are not estimates of variation between independent
cultures. DE counts are also sensitive to cell number and very small effects,
so fold changes, detection fractions, and the later pre/post-Mixscape comparison
should accompany them.

### Non-targeting null and effect-size calibration

Run:

```bash
pixi run python scripts/exploration/wilcoxon_h1_de_null.py
pixi run python scripts/exploration/calibrate_h1_de_thresholds.py
```

Each of the 31 non-targeting guide identities is compared with the other 30
using the same normalization, cell-level Wilcoxon test, and within-comparison
BH correction as the target analysis. FDR alone does not control this empirical
null: every pseudo-guide has at least four significant genes, with 1,087 calls
in total. The previous illustrative criterion (`FDR <= 0.05`,
`|log2FC| >= 0.5`, `|AUC - 0.5| >= 0.05`) still calls 27 genes across nine of
31 pseudo-guides.

Five pseudo-guides are also the five failures found independently by the
Mixscape null. With the other 26 guide identities treated as the empirically
strict control set, `FDR <= 0.05`, `|log2FC| >= 0.5`, and
`|AUC - 0.5| >= 0.15` gives zero observed null-guide calls. Applied to the
target comparisons, it retains 17,770 target--gene calls across all 150
perturbations, with a median of eight genes per perturbation. The DE plots now
use this calibrated effect-size threshold.

The five non-inert guide identities still produce 11 calls at that threshold.
Eliminating all calls while pretending that all 31 guides are exchangeable
requires `|AUC - 0.5| >= 0.40`; only 792 target--gene calls and 140 targets then
remain, with a median of one gene. This is evidence for control-guide curation,
not for a harsher universal cutoff. The observed 0/26 is an empirical screen,
not proof of a zero population false-positive rate, and the five guides should
be excluded rather than relabelled. Final pre/post-Mixscape DE should therefore
be rerun against the 26-guide strict control pool.

`wilcoxon_null_de.parquet`, `wilcoxon_null_summary.csv`, and
`wilcoxon_null_run.json` retain the pseudo-guide results. The complete threshold
tradeoff and selected boundary are in `wilcoxon_null_threshold_sweep.csv` and
`wilcoxon_null_threshold_recommendation.json`.

The 26-guide ensemble was then tested again with the five non-inert guides
removed from every pseudo-guide reference. FDR alone still calls 681 genes and
at least four genes for every guide, reflecting the power to detect very small
guide-associated shifts. The calibrated criterion calls zero genes across all
26 leave-one-guide-out comparisons. These stricter results are retained as
`wilcoxon_strict_null_de.parquet`, `wilcoxon_strict_null_summary.csv`, and
`wilcoxon_strict_null_run.json`.

Method reference, accessed 2026-08-30:
[ArcInstitute/pdex](https://github.com/ArcInstitute/pdex).

## Cross-fitted Fisher directions

Run:

```bash
pixi run python scripts/exploration/fisher_h1_directions.py
```

For each perturbation, genes in a stringent-effect subset (`FDR <= 0.05`,
`|log2FC| >= 0.5`, and `|AUC - 0.5| >= 0.05`) form the candidate set. This
exploratory Fisher run predates the null calibration above and has not been
silently redefined. All 150 CRISPR target genes are excluded globally, so the
direction cannot recognize a
perturbation from its target transcript. This label describes the conjunction
of thresholds used by this exploratory analysis; its complement is not a set of
proven non-DE genes. Within each held-out Flex run, genes
are weighted on the other two runs with a ridge-stabilized diagonal Fisher
direction. Raw counts are represented as Pearson residuals (`theta=100`), and
each held-out direction is standardized against 500 controls from the same Flex
run. A labelled cell is called response-like when it exceeds the corresponding
held-out control 95th percentile.

Among the 145 perturbations with at least one eligible downstream gene, 54,966
of 171,163 scored labelled cells (32.1%) remain in the control-like range. The
median perturbation is 12.4% control-like, while the upper quartile begins at
47.0%, exposing a broad spectrum rather than one universal responder class.
The shared-scale examples are MAST2 (weak: 89.6% control-like), TET1 (mixed:
51.0%), and HMGN1 (mostly responding but mixed: 20.3%). Five perturbations are
explicitly excluded because removing all target genes leaves no stringent-effect
candidates.

`fisher_score_distributions.png` shows the pooled and representative score
distributions. Per-cell scores, per-target summaries, exclusions, and run
parameters are retained in the correspondingly named `fisher_*` files.
Candidate discovery still used the full-data DE results, so the plot is
descriptive rather than an independent false-positive call; the Fisher weights
and scores themselves are cross-fitted, and Mixscape provides the next
comparison.

## Mixscape responder detection

The run is split into batch-local signature calculation followed by pooled
guide-level classification:

```bash
pixi run python scripts/exploration/mixscape_h1_signatures.py
pixi run python scripts/exploration/mixscape_h1_classify.py
```

The signature script normalizes raw counts to 10,000 per cell and applies
`log1p` exactly once. It reuses the 50 expression PCs from the default UMAP and
finds 20 nearest non-targeting controls within each of the 48 recorded batches,
using PC1–PC15. Each batch is written to
`data/derived/vcc2025_h1_mixscape/signatures/` with log-normalized `X` and
`layers["X_pert"]`. Five control guide pairs that failed both the DE and
Mixscape null analyses are excluded from the reference by default. The strict
control population therefore contains 32,616 cells from 26 guide pairs; the
5,560 excluded-control cells remain in the per-cell output under the
`excluded_control` label but do not enter neighbor finding or classification.

Classification preserves all 158 targeting guide identities, pools signatures
across batches (`split_by=None`), and uses a seeded, batch-balanced sample of
5,000 strict controls per guide. Guides with fewer than 30 cells are excluded,
not relabelled as controls or nonresponders. The present dataset has no such
guide before classification. The default marker test remains Wilcoxon.

Classification outputs are:

- `mixscape_cells.parquet`: per-cell class and KO posterior;
- `mixscape_guides.csv`: responder counts, fractions, posterior, signature
  magnitude, and cell-count evidence tier;
- `mixscape_excluded.csv`: initial and post-hoc exclusions with reasons; and
- `mixscape_run.json`: control composition and run parameters.

All 48 signature shards completed in about 53 minutes using up to four
batch-level workers and occupy 18 GiB. Pooled classification of all 158
targeting guide groups completed in 35.4 minutes and peaked at 5.0 GB RAM. No
targeting guide was excluded. Mixscape called 142,965 of 183,097 targeting cells
KO-like (78.1%) and 40,132 NP-like (21.9%). The 32,616 strict controls retain the
NT class by construction. Thirteen guide groups had no KO-like cells and 20
were at most 50% KO-like. These are model classifications, not ground-truth
measurements of perturbation success.

Implementation reference: Pertpy 1.2.0 Mixscape, following the local
`mixscape-responder-detection` candidate skill reviewed on 2026-08-31.

Plot the classifications on the unchanged default UMAP and reconstruct
representative Pertpy perturbation-score projections with:

```bash
pixi run python scripts/exploration/plot_h1_mixscape.py
```

`mixscape_umap.png` shows all revised labels and separate NP- and KO-highlighted
panels in the original UMAP coordinates. `mixscape_perturbation_scores.png`
shows raw cell-count histograms along the final Mixscape direction for UBE3C,
ZNF593, and STX4, chosen as well-sampled examples near 30%, 50%, and 90%
KO-like. Each panel compares all nominally perturbed cells with the same 5,000
strict controls used by the classifier. The dashed line is the exact posterior
probability 0.5 boundary from the final Gaussian-mixture iteration. The
histograms are not density-normalized, so the relative cell numbers remain
visible.

### Differential expression after Mixscape

The pre/post comparison uses the same 32,616 strict NT cells on both sides, so
the only change is retaining Mixscape KO-like cells after classification:

```bash
pixi run python scripts/exploration/wilcoxon_h1_mixscape_de.py
```

The pre-Mixscape analysis includes all 183,097 originally labelled cells across
150 targets. The post-Mixscape analysis includes 142,965 KO-like cells across
140 targets; ten targets have no KO-like cells and therefore no post-Mixscape
test. Both full tables retain every tested gene and separate flags for
`FDR <= 0.05`, `|log2FC| >= 0.5`, `|AUC - 0.5| >= 0.15`, their union, and their
intersection.

FDR-only calls increase from 456,948 to 472,832 target--gene pairs, while the
unique-gene universe is already nearly saturated before Mixscape (17,929 versus
17,942 of 18,080 genes). Requiring all three criteria is more informative:
calls increase from 17,816 to 19,699, comprising 17,740 shared, 1,959 gained,
and 76 lost pairs. Among the 140 tested post-Mixscape targets, 102 gain
all-three calls, 37 are unchanged, and only SV2A loses one. All 140 retain
significant on-target knockdown; 134 on-target transcripts pass all three
criteria. `wilcoxon_mixscape_de_comparison.png` summarizes target-level changes
and gene-universe sizes. The two full parquet tables, target summary, and run
manifest use the `wilcoxon_*_mixscape_de` naming pattern.

The post-Mixscape result is descriptive rather than an independent DE estimate:
Mixscape uses expression differences to choose KO-like cells, so enrichment of
the selected signal is expected.

### Baseline target expression and responder fraction

Run:

```bash
pixi run python scripts/exploration/plot_h1_responder_correlates.py
```

Target expression in the same 32,616-cell strict NT pool has a weak positive
rank association with the target-level Mixscape KO-like fraction (Spearman
rho = 0.229, p = 0.00473, n = 150). The control-cell detection fraction gives
a nearly identical result (rho = 0.224, p = 0.00592). This is not a strong
linear relationship: Pearson correlation with log10 control mean is 0.078
(p = 0.342), and Pearson correlation with detection fraction is 0.008
(p = 0.925). Detection is close to saturated for most targets.

The rank result is stable after removing the high-expression, zero-responder
TMSB4X outlier (rho = 0.253) or all ten zero-responder targets (rho = 0.240).
Baseline expression therefore carries some information about responder
fraction, but does not explain the broad variation or the Mixscape failures.

`mixscape_responder_expression.png` shows both relationships and the median
responder fraction in each expression quartile. The joined target table,
correlation and sensitivity results, and run metadata use the corresponding
`mixscape_responder_expression*` filenames.

## Mixscape non-targeting null

Run:

```bash
pixi run python scripts/exploration/mixscape_h1_null_controls.py
```

Each of the 31 retained non-targeting guide identities is treated in turn as a
pseudo-perturbation, with the other 30 guide identities as controls. Every
pseudo-guide has 483--2,749 cells and occurs in all 48 batches. Batch-local
signatures are recalculated with the pseudo-guide excluded from the 20-neighbor
reference pool; classification otherwise retains the main run's 5,000-control
cap, pooled classification, and Wilcoxon marker test.

The strict null fails. Five of 31 pseudo-guides produce a Mixscape direction and
KO calls, totaling 3,374 of 38,176 NT cells (8.84%). Their guide-level KO
fractions are 39.5%, 51.3%, 57.3%, 69.9%, and 78.9%, with KO cells present in
all 48 batches for every failing guide. The other 26 guides produce no direction
and exactly zero KO calls. KO fraction is unrelated to guide cell count
(Spearman rho approximately 0.0003). Reusing Pertpy's original signature shards
for the strongest failure independently gives 993 of 1,287 KO-like cells
(77.2%), closely reproducing the leave-one-guide-out result.

`mixscape_null_guides.csv`, `mixscape_null_cells.parquet`, and
`mixscape_null_run.json` retain the guide summary, cell calls, and exact run
configuration. These findings do not distinguish unintended guide-specific
biology from guide-correlated technical structure, but they show that the
current defaults do not satisfy a non-targeting-guide specificity criterion.

Neighbor matching was tested with:

```bash
pixi run python scripts/exploration/mixscape_h1_null_sweep.py
```

The sweep crossed 10, 20, and 50 neighbors with 10, 15, and 30 PCs. All nine
settings retained all five null-guide failures. Total NT KO calls ranged only
from 3,277 to 3,384 (8.58%--8.86% of all NT cells), compared with 3,374 at the
20-neighbor/15-PC default. The lowest count occurred with 50 neighbors and 15
PCs, but its strongest pseudo-guide remained 78.0% KO-like. Neighbor count and
PCA dimensionality therefore do not explain or solve the specificity failure.
Detailed and setting-level results are in
`mixscape_null_parameter_sweep_guides.csv` and
`mixscape_null_parameter_sweep.csv`.

Finally, the 26 retained guides were rerun with all five non-inert guides
excluded from both the batch-local neighbor reference and pooled classifier
controls:

```bash
pixi run python scripts/exploration/mixscape_h1_null_controls.py \
  --prefix mixscape_strict_null --exclude-guides <five-guide-pair IDs>
```

All 26 leave-one-guide-out tests produced no perturbation direction and zero
KO-like calls among the 32,616 retained control cells. The corresponding
outputs are `mixscape_strict_null_guides.csv`,
`mixscape_strict_null_cells.parquet`, and `mixscape_strict_null_run.json`.
Thus this ensemble passes both the calibrated DE null and the Mixscape null,
although the finite set of 26 controls cannot establish a zero population
false-positive rate.

### DE within KO-like null calls

The 3,374 KO-like cells from the five failed NT guides were compared separately
with the 32,616-cell strict control ensemble:

```bash
pixi run python scripts/exploration/wilcoxon_h1_null_ko_de.py
```

The five subsets contain 300--1,016 KO-like cells. Although 45--138 genes per
guide pass FDR 0.05 alone, only 2--5 pass the calibrated combination of
`FDR <= 0.05`, `|log2FC| >= 0.5`, and `|AUC - 0.5| >= 0.15`. Their leading
repressed transcripts are THOP1, FRAT2, PTMS, POLR3G, and PRKCQ, respectively,
matching the candidates from the original guide-level analysis.
`wilcoxon_null_ko_de.png` shows the five gene-level effect distributions;
the corresponding table, summary, and run parameters use the same filename
prefix. This is descriptive rather than independent validation because
Mixscape used differential expression to select the KO-like cells.

## Differential expression in the cell-eval2 scoring universe

The analysis above normalizes to 10,000 counts and corrects across all 18,080
genes. The competition scores DE in a different universe, so the whole
comparison was rerun inside it:

```bash
pixi run python scripts/exploration/check_celleval2_parity.py
pixi run python scripts/exploration/check_one_vs_rest.py
pixi run python scripts/exploration/celleval2_h1_de.py
pixi run python scripts/exploration/celleval2_h1_null.py
pixi run python scripts/exploration/celleval2_h1_guide_null.py
pixi run python scripts/exploration/compare_celleval2_null.py
```

The universe is `cell-eval2` 0.16.0 under `configs/vcc2026.yaml`: counts
normalized per cell to 1e6, the gene set restricted to genes whose mean CPM in
the reference group exceeds 5, a two-sided Mann-Whitney U on `log1p(CPM)`, log2
fold change from arithmetic CPM means with a 1e-9 pseudocount, and
Benjamini-Hochberg within each comparison over surviving genes only. The gate
keeps **10,780 of 18,080 genes**.

### Verification

`check_celleval2_parity.py` runs `cell_eval2.de_compute.compute_de` and the
reimplementation on the same 10,308-cell slice. They agree on the kept gene set,
and on p-values and log2 fold changes to 0.0 (`p_adj` to 2.2e-16). Reaching that
required three corrections, each of which changes marginal calls:

- cell-eval2 casts to float64 before normalizing; float32 moves log2FC by ~1e-5;
- scanpy divides by `counts/target_sum`, and multiplying by its reciprocal differs by one ulp;
- the test runs on `log1p(CPM)`, which is monotone but not injective in float64. It
  merges near-tied values (3,344 distinct values become 3,343 on the busiest gene),
  changing the tie correction and moving p-values by ~6e-5.

`check_one_vs_rest.py` confirms the one-vs-rest kernel used by both nulls returns
statistics and p-values bit-identical to the pairwise kernel used for the targets,
so null and target counts are on one scale.

### Held-out random control null

`celleval2_h1_null.py` draws random non-targeting pseudo-groups at nine sizes
spanning the real range of target cell counts (25 to 4,800), ten replicates each,
every group held out of its own reference. **Across all 90 comparisons there are 2
significant genes in total**, in 2 comparisons. Benjamini-Hochberg at 0.05 predicts
about 4.5 comparisons with at least one rejection, so the procedure is calibrated
and mildly conservative. The gene universe, recomputed per comparison from each
group's own reference, is stable at 10,780 +/- 1.

Cell-level DE is therefore not inflated by pseudoreplication in this design. A
no-effect group of 4,800 cells calls zero genes. Metadata confirms why the draws
are a fair null rather than an optimistic one: real targets are not
batch-concentrated. The median target holds 3.2% of its cells in its largest batch
and spans 46.8 of 48 effective batches, which random draws at matched size
reproduce closely. No target has more than 9.1% of its cells in one batch.

### Guide-identity null

`celleval2_h1_guide_null.py` repeats the comparison with the real non-targeting
guide pairs as groups. Their sizes (483-2,749) sit inside the range where random
draws called nothing, so size is controlled and guide identity is the only
difference. **221 significant genes across 31 guides, 15 of which call at least
one.** Calls skew downward (159 down, 62 up) and only 5 genes are called by more
than one guide, so there is no shared signature.

The calls separate into two phenomena that a raw count conflates:

| guide pair | DE | strongest call | genes with abs(log2FC) > 0.5 |
|---|---:|---|---:|
| 00018\|00127 | 81 | THOP1 -1.36 | 6 |
| 00062\|02518 | 33 | VAMP4 -2.53 | 3 |
| 00027\|00143 | 27 | MGMT -0.43 | 0 |
| 00047\|00882 | 20 | WASF2 -1.69 | 5 |
| 00050\|03642 | 16 | GRPR +0.62 | 1 |
| 00026\|02263 | 13 | FRAT2 -1.37 | 4 |
| 00121\|00339 | 8 | PRKCQ -3.37 | 3 |

Five guide pairs carry several strongly repressed genes at numerically zero
`p_adj`. These are the same five excluded by the earlier Mixscape and 10,000-count
DE nulls, recovered independently here. The reading is that they have real
repressive activity at specific loci; a 3.4 log2 knockdown with AUC 0.88 is not a
control. `00027|00143` is the other phenomenon: 27 calls, no gene above 0.5, and
the largest cell count in the pool, so power detecting a small shift.

### Targets against the floor

`compare_celleval2_null.py` places the 150 targets against the null at their own
size. **All 150 exceed even the worst null draw at their size.** The weakest is
DZIP3 at 4 DE genes, and it represses its own gene by 4.53 log2. Every one of the
147 targets whose own gene clears the CPM gate significantly represses it, at a
median of -4.35 log2 (KAT2A, MAU2 and SSBP1 are the three whose own transcript
sits below 5 CPM in controls and so cannot be scored).

DE counts run from 4 to 8,865 with a median of 2,065; 31 targets call at most 100
genes. Their Spearman correlation with cell count is **-0.115**, so the response
spectrum is not a power artifact. Against the earlier 10,000-count run the counts
barely move (median 2,111 to 2,065; quartile ratios 0.79/0.92/0.99).

### Consequence for thresholds

The earlier calibration on this page raised thresholds until the guide null was
silent, which retained a median of eight genes per perturbation. That null was
contaminated by the five off-target guide pairs, and the size-matched random null
now separates the two explanations it confounded: the test is calibrated, and the
floor was five bad controls rather than a loose criterion. An effect-size cutoff
cannot fix that, because those guides produce the *largest* effects in the null;
removing them by threshold would need abs(log2FC) > 3.4 and would erase most real
perturbation biology. The five guides are a control-pool curation problem, not a
threshold problem, and FDR < 0.05 alone -- which is what the competition scores --
is defensible in this universe.

### Does the contaminated control pool bias the targets?

The five suspect guide pairs are 5,560 of 38,176 control cells (14.6%), so they
enter every target's reference. Recomputing the pooled control means without them
shifts the gated genes by a median of 0.0016 log2, with a single gene past 0.05
(PRKCQ, 0.053) and a maximum of 0.053; gate membership moves by one gene. Each
suspect pair is only 2-4% of the pool on its own, so its off-target repression is
diluted roughly twentyfold in the pooled mean.

The contamination is therefore consequential where a suspect guide *is* the group
under test, and negligible where it is 4% of the reference. The target run against
all 31 guides stands; it does not need to be repeated against a curated pool.

### Which part of the universe change silenced the old null?

The earlier guide null on this page called 1,087 significant genes and found at
least four for every one of the 31 pseudo-guides. That run and the gated one differ
in grouping (none: both group by guide) and in the gene universe only, so the two
are directly comparable. Scoring the old calls against the gate:

- **823 of 1,087 (75.7%)** lie in genes that fail `reference mean CPM > 5`;
- the median control expression of a called gene is **0.002 CPM**;
- the 264 above-gate calls are close to the gated run's 221.

The old result therefore decomposes into ~823 low-expression artifacts and ~264
real guide-associated calls, and the CPM gate rather than the normalization or the
correction is what removes them. Note the direction of the surprise: shrinking the
BH universe from 18,080 to 10,780 genes makes the threshold *more* lenient for the
survivors, so gating should raise the count among retained genes. It falls by a
factor of five, which means the discarded genes carried nearly all of the signal.

The mechanism, demonstrated by `scripts/exploration/check_tie_variance.py`. The
tie-corrected variance is

    Var[U] = n1 n2 / 12 * [ (N+1) - sum(t^3 - t) / (N(N-1)) ]

and for a gene that is zero in all but k of N cells, the zeros form one tie group of
size N-k that absorbs nearly the whole bracket: it collapses from N+1 to about 3k+1,
so Var[U] ~ n1 n2 k / 4 rather than n1 n2 N / 12. That collapse is *correct* -- with
almost everything tied there really is almost no information. The failure is the
normal approximation built on top of it. The same sparsity makes U violently
discrete: with a group holding 3.4% of cells, the number of nonzero cells landing in
it is nearly Poisson with mean 0.034k, so U has a few atoms and a normal
approximation to a three-atom skewed distribution underestimates tail mass.

For 0/1 data the exact null is hypergeometric, so the asymptotic p-value can be
scored against the truth. Taking a guide-sized group (n1 = 1,287) and the first
enrichment x that the gated BH universe would call significant:

| k | x | asymptotic p | exact p | too small by |
|---:|---:|---|---|---:|
| 4 | 2 | 2.4e-07 | 6.5e-03 | 27,000x |
| 10 | 3 | 3.1e-06 | 3.8e-03 | 1,250x |
| 20 | 5 | 8.3e-08 | 4.4e-04 | 5,300x |
| 50 | 8 | 7.4e-07 | 2.5e-04 | 337x |
| 200 | 19 | 1.5e-06 | 5.2e-05 | 36x |
| 1000 | 60 | 3.1e-06 | 2.5e-05 | 8x |

A gene present in four cells, two of them in the group, is handed p = 2e-07 by the
rank test when the true probability is 0.0065. BH cannot repair this, because BH
assumes valid p-values. The error decays with density and reverses in the depletion
tail, so it is specific to sparse-plus-enriched -- the corner the ungated null was
making its calls in.

The gate is a detection-fraction cut in disguise, and the split is bimodal rather
than arbitrary: genes below `CPM > 5` are detected in a median of 334 of 38,176
cells (0.9%), genes above it in 33,821 (88.6%), with little in between. It is not a
modelling preference in Arc's config; it removes the region where the test Arc chose
is invalid.

### Guide-null structure, exactly

Of 31 non-targeting guide pairs: **16 call nothing**, 7 call only genes below 0.5
log2, and **8 call at least one gene above 0.5**. Of those 8, **5 carry three or
more**, which is where the gap sits (6, 5, 4, 3, 3, then 1, 1, 1). Maximum effect
size alone does not separate them (3.37, 2.53, 1.69, 1.37, 1.36, 0.93, 0.69, 0.62),
so the discriminator is the number of large effects, not the presence of one. The
`>= 3` cut was chosen knowing it reproduces the five guides flagged by the earlier
Mixscape and 10,000-count analyses; the gap in the data is independent of that
choice, but the specific cut is partly hindsight.

### Does any of that survive inside the gated universe?

No. The gate is on mean CPM, so in principle a gene detected in few cells but with
high counts there could pass it and remain in the bad regime. In this dataset none
does. The sparsest gene clearing `CPM > 5` is NPTX1 at **9.9% detection**
(k = 3,767); exactly one gated gene is under 10% detection and five under 20%, and
the median is 88.6%. At those densities the asymptotic p is wrong by **1.7x to
2.9x**, not by orders of magnitude.

The one theoretical margin is that BH's threshold for the top-ranked gene of 10,780
at alpha 0.05 is p < 4.6e-6, and NPTX1's 2.9x error straddles it (asymptotic 3.2e-6,
exact 9.2e-6). Empirically nothing lands there. If the 221 guide-null calls were
residual tie artifact they would concentrate at the sparse end; instead they are
*denser* than the universe they are drawn from (median detection 93.4% against
88.6%), only 2 of 221 sit below 25% detection, and none is in the five sparsest
gated genes. The two random-null calls are at 98.8% and 93.6% detection -- ordinary
BH false positives at the dense end.

This is what licenses reading the five guides as real repressive activity rather
than as an artifact of the test: the calls do not come from the region where the
test misbehaves.

## How much of the panel is scoreable at 400 cells?

```bash
pixi run python scripts/exploration/celleval2_h1_subsample.py
```

Every count above comes from a median of 1,045 cells per target; the 2026 evaluation
supplies 400. `celleval2_h1_subsample.py` draws 400 cells from each target that has
them, five replicates, against the same reference and universe. The 24 targets with
fewer than 400 cells are run whole, once, and flagged. Per-target seeds are `crc32`
of the target name, not `hash()`, whose salt is per-process and would make the
frozen draw irreproducible.

**DE counts fall by about 60%**: median 1,890 to **496** over the 126 subsampled
targets, with retained-fraction quartiles 0.21 / 0.41 / 0.63.

`de_wilcoxon_lfc_nmae` omits a target with fewer than 10 reference-significant genes:

| DE genes | all cells | at 400 cells |
|---|---:|---:|
| under 10 | 5 / 126 | **17 / 126** |
| under 20 | 8 / 126 | 29 / 126 |
| under 50 | 20 / 126 | 40 / 126 |

**Twelve targets lose eligibility**, all with on-target knockdown between -3.8 and
-4.9 log2. TMSB4X is the extreme: 4,760 cells and 190 DE genes at full size, 3 at
400. Six more (CALM3, IDE, WFS1, KIF1B, IGF2R, TWF2) **flip eligibility between
replicate draws**, so whether they can be scored depends on which 400 cells are
drawn.

The smaller control pool the competition uses (18,400 per context against H1's
38,176) does not add to this. For a fixed effect size the rank test's
`z = (AUC - 0.5) * sqrt(12 n1 n2 / (N+1))`, which at `n1 = 400` gives 68.9 against
38,176 controls and 68.5 against 18,400 -- a 0.6% difference. With `n1 << n2` the
perturbed group dominates, so 400 cells is the whole effect and no separate
control-subsampled run is needed.

Outputs: `celleval2_subsample_de.parquet` and `_summary.csv` hold all five
replicates plus the 24 undersized targets. `celleval2_subsample_frozen_de.parquet`
and `_frozen_summary.csv` are replicate 0 of the 126 eligible targets -- a fixed,
reproducible 126 x 400 reference. It is a deterministic *draw from training cells*,
not held-out truth, and it cannot cover the 24 undersized targets.

## Does responder filtering sharpen the DE signal?

```bash
pixi run python scripts/exploration/celleval2_h1_mixscape_de.py
pixi run python scripts/exploration/plot_celleval2_mixscape_de.py
```

The earlier pre/post-Mixscape comparison ran at 10,000-count normalization over all
18,080 genes. Repeated in the scoring universe, with the Mixscape labels carried
over unchanged from the pertpy run. Because the reference is all 31 NT guide pairs,
the pre-Mixscape arm *is* `celleval2_de.parquet` and is read back rather than
recomputed.

A third arm carries the argument. Post-Mixscape uses fewer cells, so a change in
DE-set size confounds "filtering kept the responding cells" with "filtering removed
cells". The matched arm draws the same number of cells at random from the same
target, so the KO arm is read against a loss of power alone.

**Selection wins in 131 of 140 targets.** Median DE genes: pre 2,193, KO-like 2,384,
size-matched random 1,979. The gain over the matched arm is entirely concentrated
where there is something to filter (Spearman against responder fraction **-0.786**):

| Mixscape KO-like fraction | targets | median gain over matched | median KO DE | median matched DE | median Jaccard vs all-cell |
|---|---:|---:|---:|---:|---:|
| under 50% | 7 | **2.6x** (1.355 log2) | 1,045 | 603 | 0.469 |
| 50-75% | 18 | 1.6x (0.692) | 1,009 | 579 | 0.624 |
| 75-90% | 42 | 1.18x (0.244) | 2,112 | 1,510 | 0.771 |
| over 90% | 73 | 1.03x (0.038) | 3,858 | 3,773 | 0.922 |

For a target where only a third of labelled cells respond, testing the Mixscape
subset finds **2.6 times** the DE genes that the same number of randomly chosen cells
finds. Above 90% responders the filtering does nothing, which is the right null: it
is removing almost no cells there.

The earlier reading -- that pre- and post-Mixscape gene sets are similar -- holds only
in aggregate. Jaccard against the all-cell set falls from 0.922 in the
high-responder band to **0.469** below 50%: for the targets where filtering matters,
it changes which genes are called, not merely how many.

`celleval2_mixscape_de_comparison.png` shows all three arms;
`celleval2_mixscape_de_summary.csv` and `celleval2_post_mixscape_de.parquet` hold
the per-target and per-gene results.

### Two independent methods agree on the same weak targets

Ten targets have **no Mixscape KO-like cells at all**: DZIP3, PMS1, LAD1, RAB3B,
PTPN1, CALM3, CAMSAP2, PLCB3, IDE, TMSB4X. Every one of them is also in the
17-target set that falls below `lfc_nmae`'s ten-gene floor at 400 cells -- a
complete overlap, with no exceptions in either direction among the no-KO set.

The two criteria share no machinery: one is a mixture model fitted to per-cell
perturbation signatures, the other is the power of a rank test at a given sample
size. All ten carry on-target knockdown between -4.0 and -4.9 log2. They are
perturbed, silenced, and transcriptionally almost inert -- and both the responder
model and the scoring metric independently decline to say anything about them.

## Mixscape classes against the response bins

```bash
pixi run python scripts/exploration/plot_celleval2_mixscape_umap.py
pixi run python scripts/exploration/plot_celleval2_mixscape_scores.py
```

`celleval2_mixscape_scores.png` histograms the Mixscape perturbation score with the
perturbed population split into the classes it was assigned, rather than shown as one
undivided block against the controls. The classes are the production run's, not ones
refitted for the plot; the refitted threshold agrees with them for **100%** of
perturbed cells, so the two are interchangeable here. Representative guides span the
responder range: UBE3C (32% KO-like), ZNF593 (46%), STX4 (90%). In each the
non-perturbed cells sit on top of the control distribution and the KO-like cells form
the right tail, which is the claim Mixscape makes and the reason its labels are worth
using.

`umap_mixscape_bins.png` puts the same classes on the UMAP, on the panels and axes of
`umap_response_bins.png`. KO and NP occupy separate rows: overlaid in one panel the
larger class hides the smaller, and because alpha compounds with density the panel then
reads as cell count rather than class.

Cell-weighted, the KO-like share rises across the bins -- 65% negligible, 81% subtle,
92% strong -- but the negligible figure needs care, because that bin is not one
population:

- **10 of its 50 targets have no KO-like cells at all** (all ten zero-responder targets
  in the dataset are in this bin), contributing 16,246 of the bin's 69,837 cells;
- the remaining 40 have an ordinary responder fraction, median **88%**.

So a typical negligible perturbation is not one where the cells failed to respond. Most
of its cells respond; they simply produce few differentially expressed genes. Its
KO-like cells also stay inside the control territory on the UMAP, with none of the
satellite clusters the strong bin populates -- the weakness is in the size of the
response, not only in how many cells mount one.

### Responder fraction and response size are different axes

They correlate only moderately (Spearman **0.528**), and every bin spans nearly the
whole responder range:

| response bin | median KO-like fraction | range |
|---|---:|---|
| negligible | 0.86 | 0.00-0.98 |
| subtle | 0.86 | 0.32-1.00 |
| strong | 0.98 | 0.32-1.00 |

The negligible and subtle bins have the *same* median responder fraction. UBE3C is a
strong perturbation (4,071 DE genes) in which only 32% of labelled cells respond;
STX4 is subtle (1,141) with 90% responding. A perturbation can therefore produce a
large transcriptional response in a minority of cells, or a modest one in nearly all
of them, and DE-set size alone does not distinguish the two.

## Systematic variation, and whether anything survives it

```bash
pixi run python scripts/exploration/celleval2_h1_systema.py
pixi run python scripts/exploration/celleval2_h1_specificity.py
```

Systema (Viñas Torné et al., *Nature Biotechnology* 2025,
[doi:10.1038/s41587-025-02777-8](https://doi.org/10.1038/s41587-025-02777-8)) argues
that perturbation responses are dominated by one shared direction, and that
control-referenced metrics therefore mostly measure whether a model reproduced the
*average* perturbation effect. Their statistic: with per-perturbation centroid `O(X)`,
control centroid `O_control`, shift `s_X = O(X) - O_control`, and average perturbation
effect `a = mu_pert - O_control` (`mu_pert` the per-cell mean over all perturbed
cells), **systematic variation = mean over perturbations of cos(s_X, a)**.

Centroids are means over cells of log-normalized expression, so they cannot be read
off pseudobulk means -- `mean(log1p(x))` is not `log1p(mean(x))`. The script streams
the file once in contiguous chunks to accumulate them exactly, over the same
CPM-gated 10,780 genes.

### H1 carries less systematic variation than any dataset Systema examined

| dataset | systematic variation |
|---|---|
| **VCC 2025 H1 (this screen)** | **0.187 +/- 0.145** |
| Systema, mean over ten datasets | 0.41 +/- 0.18 |
| Adamson | 0.76 |
| Norman | 0.50 |
| Replogle K562 | 0.32 |
| Frangieh (their lowest) | ~0.2-0.3 |

Their re-referencing check passes: replacing the control centroid with the unweighted
mean of perturbation centroids moves the cosines to **0.000 +/- 0.009** (Systema report
-0.06). Pairwise cosine between shifts is median **0.018**, so the average pair of
perturbations is close to orthogonal -- but the maximum is **0.974**.

The statistic is robust to the space it is computed in: on log2 fold changes of
arithmetic CPM means rather than log-normalized centroids, the same quantities are
0.202 median and 0.023 pairwise, against 0.185 and 0.018 here.

### The shared direction is pluripotency exit, not generic stress

Genes loading most positively on `a` are **NODAL, LEFTY1, LEFTY2, BMP7, SMAD3**, with
H1F0, KDM6B and LGALS1; **PRDM14** is among the most negative. That is a coherent
Nodal/BMP-driven differentiation signature with loss of a core pluripotency factor, not
the ribosome-biogenesis or unfolded-protein signatures Systema found in K562 and
Adamson. It is exactly the "biological in origin but systematic in effect" case the
paper is careful to allow for: in hESCs, a sufficiently disruptive perturbation nudges
cells out of the pluripotent state, whatever the gene.

Alignment tracks response size (Spearman `cos_shared` vs DE count **0.68**) and is
highest for the transcriptional machinery -- TADA1 0.55, MAU2 0.51, MED13 0.48, MED24
0.45, KAT2A 0.45, MED12 0.45. Some large responses are nevertheless orthogonal to it:
ARID1A (-0.06, 4,928 DE genes) and EIF4B (-0.05, 6,268) move somewhere else entirely.

### Co-complex perturbations give the same response

If the residual after the shared component carried no target-specific information,
perturbations of co-complex members would be no more similar than random pairs. Using
CORUM human complexes (5,628 complexes; 74 contain at least two of our targets, giving
45 co-complex pairs out of 11,175):

| | mean cosine |
|---|---|
| co-complex pairs | **0.2335** |
| all other pairs | 0.0345 |
| difference | **0.1991** |

Against a label permutation that holds both the pair count and the similarity
distribution fixed (20,000 permutations, null 0.0003 +/- 0.0173): **z = 11.5, p < 5e-5**
(the permutation floor). Co-complex membership predicts response similarity at about
seven times the background level.

The ranking is a roll-call of real complexes rather than a statistical curiosity:

| pair | cosine | complex |
|---|---:|---|
| METTL14 / METTL3 | 0.974 | METTL3-METTL14-WTAP (m6A writer) |
| MED13 / MED13L | 0.698 | Mediator |
| MED12 / MED13 | 0.685 | Mediator |
| MED1 / MED24 | 0.616 | THRA-TRAP |
| KAT2A / TADA1 | 0.573 | STAGA |
| NDUFB4 / NDUFB6 | 0.538 | Respiratory chain complex I |
| TADA1 / USP22 | 0.490 | STAGA, SPT3-linked |
| ARID1A / SMARCA4 | 0.318 | NUMAC / BAF |
| BRD9 / SMARCA4 | 0.173 | ncBAF |
| BIRC2 / IKBKG | 0.169 | TNFR1 signaling |

**Conclusion.** Both things are true, and the specific component is the larger one
here. There is a shared axis, it is real, and it grows with response magnitude -- but at
0.187 it accounts for less of this dataset than of any benchmark Systema examined, and
what remains is strongly organised by protein complex membership. On this dataset the
Systema critique lands on the *evaluation* rather than on the biology.

Annotation provenance: CORUM human complexes downloaded 2026-09-03 from the CORUM
FastAPI endpoint (`/public/file/download_current_file?file_id=human&file_format=txt`,
discovered from the site bundle since the legacy `allComplexes.txt.zip` paths now
return the SPA shell), 6.3 MB, 7,813 rows, <https://mips.helmholtz-munich.de/corum/>.

## Responders and non-responders within a perturbation

```bash
pixi run python scripts/exploration/celleval2_h1_fisher_score.py
pixi run python scripts/exploration/plot_celleval2_fisher_score.py
```

Simplified successor to `fisher_h1_directions.py`, which used Pearson residuals
under a negative-binomial model and excluded all 150 targets globally from every
marker set. This uses `log1p(CPM)` on the gated universe -- the same
representation as `celleval2_de.parquet` -- and excludes only each target's own
gene. Markers are the top 50 genes by `|log2FC|` among that target's FDR<0.05
genes (never fewer than the FDR-only set already has, minimum 4; caps a strong
perturbation's thousands of tiny-effect significant genes at a focused set). The
direction is a cross-fitted diagonal Fisher score: weights (per-gene mean
difference over pooled variance, lightly ridge-regularized) are fit on cells from
two of three Flex batch groups and read off the held-out third, so a cell's score
never depends on a direction fit using that same cell.

**Negative control**: an inert non-targeting guide pair (`non-targeting_00006|
non-targeting_00706`, zero calls in the guide-identity null) was scored through
the identical pipeline as if it were a target. Mean AUC 0.503 (chance), false
positive rate 6.0% against an expected ~5% -- the pipeline does not manufacture
separation on its own.

Across the real 150 targets: AUC ranges from 0.614 to 1.00, median 0.959 -- every
target clears the negative control's chance-level 0.50; 121 of 150 clear
AUC >= 0.75 with at least 300 cells. `celleval2_fisher_score_distributions.png`
shows all perturbations pooled plus three representative targets chosen by
responder fraction (not by DE-count bin, which is shown elsewhere) among those with
enough cells to render a smooth histogram (>=1,000): SMARCA5 at 99% responder-like
(control and perturbed barely overlap), KDM2B at 46%, and TRAM2 at 20%. KDM2B and
TRAM2 show the mixture directly -- a fraction of the perturbed population
indistinguishable from control, and a fraction clearly shifted, rather than the
whole labelled population moving together. Even pooled across all 150 targets, the
perturbed distribution stays visibly right-skewed against the controls' unimodal
peak, so the phenomenon is not confined to a handful of examples.

Outputs: `celleval2_fisher_scores.parquet` (per-cell scores, held-out fold and
class), `celleval2_fisher_summary.csv` (per-target AUC, d', responder fraction),
`celleval2_fisher_run.json`.
