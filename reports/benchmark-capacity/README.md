# Nominal benchmark capacity

This report quantifies the public human single-cell CRISPRi opportunity set before
perturbation-quality filtering. It is a planning resource, not a claim that every
listed gene/context is already execution-ready.

The canonical source table contains 20,073 gene x experimental dataset-context
observations across 10,136 genes, 14 response-bearing dataset-contexts, 13
canonical biological contexts, and 5 studies. Study-qualified
`dataset_context_id` values are preserved even when two observations share a
`canonical_context_id`.

The generated files are:

- `capacity_summary.csv`: comparable nominal ladder totals plus the E5 ablation;
- `E0_sampling_capacity.csv`: per-context cell-count and split-size coverage;
- `E1_distribution_capacity.csv`: control/raw-count readiness;
- `E2_same_context_pairs.csv`: same canonical context across studies;
- `E3a_pairwise_within_study.csv`: ordered within-study source-target overlaps;
- `E3b_multisource_within_study_holdouts.csv`: main union and strict all-source-intersection counts;
- `E4_replogle_targets.csv`: per-target Replogle K562 overlap;
- `E5_multisource_holdouts.csv`: canonical-context E5 and experimental-context-only ablation capacity;
- `replogle_anchor_summary.csv`: aggregate Replogle bridge metrics;
- `cell_count_attrition_summary.csv`: thresholded E3--E5 totals at 50, 100, 200, 400, and 800 cells;
- `cell_count_attrition_by_target_context.csv`: thresholded per-target-context totals;
- `E3a_pairwise_cell_thresholds.csv`: thresholded ordered within-study pair counts;
- `cell_count_qualified_exposure.csv`: per-gene E3b/E5 `k_g_qualified` records;
- `cell_count_attrition.md`: count semantics, completeness, compact attrition tables, and unresolved files;
- `evaluation_ladder.md`: interpretation of each benchmark rung;
- `bridge_map.md`: graph-oriented view of the dataset landscape.

Regenerate the computational outputs with:

```bash
pixi run python scripts/metadata/build_perturbation_atlas.py
pixi run python scripts/metadata/validate_metadata.py
```

The build consumes curated local metadata and target lists; it does not depend on
live web scraping or expression-matrix downloads. VCC-300 membership is retained
as an annotation and never used as an atlas inclusion rule.

## Interpretation boundary

These are upper bounds before checks for guide quality, knockdown efficiency,
cell number, response strength, contamination, responder status, label
confidence, and cross-study comparability. E0 split-size columns are descriptive
capacity estimates rather than global eligibility thresholds. Exact per-target
cell counts are now available for all 14 response-bearing dataset-contexts;
`cell_count_attrition.md` records the representation-specific definitions.

H1 is an important challenge/assay bridge, but its currently verified accessible
representation is log-normalized and its assay-family compatibility is unresolved.
The Jiang pathway conditions and McFaline-Figueroa drug environments are collapsed
to cell-line-level contexts in this first atlas and should be refined before a
benchmark is run.
