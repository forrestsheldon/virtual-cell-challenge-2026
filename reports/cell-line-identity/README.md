# Cell-line identity exploration

This analysis is intentionally separate from control-cell QC. It treats the
anonymous A/B/C labels as the comparison groups and aggregates raw counts by
`ntc_id` before calculating log2(CPM + 1).

Run it with:

```bash
pixi run python scripts/exploration/guess_cell_lines.py data/controls
```

The script reports one-versus-rest differential expression, the smaller of the
two pairwise effects, the fraction of cells detecting each gene, and the
fraction of a context's 46 NTC pseudobulks exceeding both other context means.
Because `ntc_id` is not confirmed to be a biological replicate, p-values and FDR
are descriptive; effect size and construct consistency drive interpretation.

Generated files are written to `reports/cell-line-identity/generated/`.

## First-pass interpretation

| context | robust lineage signal | tentative exact candidates | confidence |
| --- | --- | --- | --- |
| A | immature/CD4 T lineage: `TRBC1`, `CD3D/E/G`, `ZAP70`, `LCK`, `DNTT`, `BCL11B` | Jurkat or MOLT-4 | high for lineage; medium-low for exact line |
| B | collagen-rich mesenchymal epithelial cancer: `ITGA11`, `COL3A1`, `COL5A1`, `NNMT`, with `KRT7`, `FOLR1`, `MSLN` | HeLa; SiHa is a weaker alternative | medium |
| C | basal/squamous epithelium with an airway-like component: `KRT5/13/14/15`, `TP63`, `SOX2`, `SCGB1A1` | HBEC3-KT or HaCaT | high for basal epithelium; low for exact line |

These are hypotheses, not de-anonymized labels. A quick cross-platform check used
the archived Human Protein Atlas v20.1 cell-line RNA table. Spearman correlation
over 10,887 expressed and reference-variable genes ranked Jurkat then MOLT-4 for
A (0.796, 0.781), HeLa then SiHa for B (0.705, 0.629), and HBEC3-KT then HaCaT
for C (0.657, 0.653). Marker-set scoring reversed the close A and C pairs,
showing that neither exact call is yet stable.

Reference provenance: Human Protein Atlas v20.1 `rna_celline.tsv.zip`, accessed
2026-08-28 from
`https://v20.proteinatlas.org/download/rna_celline.tsv.zip` (10,651,643 bytes;
the extracted table was 59,332,015 bytes). This archived compact panel contains
69 cell lines and was used only for this exploratory check. The current HPA
v25.1 table covers 1,206 lines but is 205,861,141 compressed bytes, so it was not
downloaded under the repository's 100 MB approval rule.
