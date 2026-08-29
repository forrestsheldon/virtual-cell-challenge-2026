# Minimal HPA rank-correlation test

This test sums all raw control counts into one pseudobulk for each context,
removes genes with `MT-`, `RPL`, or `RPS` prefixes (while retaining the
non-ribosomal `RPS6K` kinase family), selects the 3,000 genes with greatest
median absolute deviation of `log2(nTPM + 1)` across HPA, and computes Spearman
correlation against every Human Protein Atlas cell-line profile.

Run:

```bash
pixi run python scripts/exploration/rank_hpa_cell_lines.py \
  data/controls data/reference/hpa_v25_1/rna_celline.tsv.zip
```

Reference provenance:

- source: `https://www.proteinatlas.org/download/tsv/rna_celline.tsv.zip`
- Human Protein Atlas version: 25.1, Ensembl 109
- accessed: 2026-08-28
- compressed bytes: 205,861,141
- archive member bytes: 1,069,805,463
- SHA-256: `e5b11779308c2720cfe89d9cb19932ad4c0c3fea0fa37e8c61a14d501995852a`
- coverage: 1,206 cell lines

The archive is stored under gitignored `data/reference/`. Correlation is a
nearest-expression-profile test, not cell-line authentication.

## Results

| context | rank 1 | rho | other top-five matches | interpretation |
| --- | --- | ---: | --- | --- |
| A | Jurkat E6.1 | 0.906 | JURKAT, MOLT-4, PF-382, P12-Ichikawa | highly coherent T-ALL match; Jurkat is the leading exact hypothesis |
| B | HeLa | 0.874 | PODO/SVTERT152, SiHa, OV7, LXF 289 | HeLa is clearly first, but the runner-up tissues are mixed |
| C | CAL-33 | 0.874 | CAL-27, PE/CA-PJ15, A-431, PE/CA-PJ34 | coherent squamous epithelial group, especially oral/head-and-neck lines; exact line unresolved |

The top-to-second correlation differences are 0.008 for A, 0.048 for B, and
0.011 for C. A's first two entries are Jurkat variants, so the small numerical
gap reinforces rather than weakens the Jurkat hypothesis. C supports a tissue
class more strongly than a particular line. The C result revises the earlier
bronchial-versus-keratinocyte interpretation toward squamous head-and-neck or
skin epithelium.

All 1,206 reference profiles are positively correlated with each query; the
lowest correlations are 0.662--0.670. Absolute rho therefore is not evidence of
identity in this cross-platform comparison. The informative features are the
relative rank, separation from runners-up, and coherence of their annotated
lineages.

## Variable-gene refinement

Restricting the comparison to 3,000 HPA-variable genes preserved every leading
match and made the background substantially more discriminating:

| context | leading match | rho | median background rho | first-to-second gap | first-to-fifth gap |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Jurkat E6.1 | 0.899 | 0.171 | 0.041 | 0.187 |
| B | HeLa | 0.792 | 0.361 | 0.220 | 0.248 |
| C | CAL-33 | 0.784 | 0.333 | 0.056 | 0.111 |

![The three highest HPA correlations for each VCC context. All panels share the
same correlation scale; blue marks the leading nomination, and each title gives
the first-to-second gap.](hpa-rank/generated/hpa_rank_correlations_3000_variable_genes_top3.png)

For comparison, the all-gene median correlations were 0.727, 0.772, and 0.772,
and the first-to-second gaps were 0.008, 0.048, and 0.011. The refinement
therefore addresses the shared-expression background without changing the
substantive result. Use `--variable-genes 0` to reproduce the all-gene test.
