# Public perturbation-data metadata

This directory separates three entities:

1. a **biological experiment** (`dataset_catalog.csv`);
2. an accessible **representation** of it (`dataset_variants.csv`);
3. experiment x cell-context **target coverage** (coverage CSVs).

The canonical full CRISPRi atlas is `perturbation_observations.parquet` (with an
inspection-friendly CSV mirror). Each row is one normalized perturbation gene x
one study-qualified `dataset_context_id`. `dataset_contexts.csv` separately maps
those IDs to `canonical_context_id` values so that repeated biological contexts
remain distinct in the source of truth.

`n_cells` counts every cell in the current curated representation assigned to
that perturbation; it is not a biological quality filter. `n_cells_status`,
`n_cells_source`, and `n_cells_scope` distinguish exact values from unresolved
coverage and record the applicable representation. Unresolved values remain
blank rather than being imputed or treated as zero.

The derived atlas views are:

- `gene_x_dataset_context.csv` (primary binary coverage matrix);
- `gene_x_canonical_context.csv` (collapsed biological-context view);
- `context_x_dataset.csv` (study/protocol bridge view);
- `dataset_context_overlap.csv` and `bridge_edges.csv` (weighted context graph);
- `perturbation_context_summary.csv` and
  `perturbation_context_thresholds.csv` (gene-level overlap summaries);
- `benchmark_gene_exposure.csv` (E5 `k_g` and Seen-1/Seen-multi/Unseen annotations, including the experimental-context-only ablation).

The older `target_coverage_*.csv` files remain the VCC-300-specific view and are
not the full benchmark universe.

Read `provenance.md`, `preprocessing_comparison.md`, and
`recommended_variants.md` before selecting a file. Regenerate atlas and VCC
coverage tables with:

```bash
pixi run python scripts/metadata/build_target_coverage.py
pixi run python scripts/metadata/build_perturbation_atlas.py
pixi run python scripts/metadata/validate_metadata.py
```

Target lists in `source_targets/` were obtained from categorical H5AD metadata or
small author supplementary files; they are not expression matrices. Gene matching
uses case-normalized exact symbols and preserves source labels. Alias resolution
is intentionally not guessed. No target has been filtered for cell count, guide
quality, knockdown, response strength, contamination, or responder status.
