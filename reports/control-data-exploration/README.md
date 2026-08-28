# Post 1: Exploring cell lines

Status: **first analysis implemented**
Control release: **validation panel `vcc2026-val-1`**

## Scope

Post 1 introduces cell lines and uses the anonymous A/B/C controls to show how
three cultured cell populations can differ at single-cell resolution.

The post will cover:

1. what a cell line is;
2. why tissue of origin, genotype, culture adaptation, passage history, and
   cell state make cell lines different;
3. a minimal QC check of the supplied controls; and
4. a UMAP showing the relationship between contexts A, B, and C.

Detailed 10x Flex technology, CRISPRi, perturbation labels, exact cell-line
identity matching, and baseline submission are deferred to later work.

## Minimal analysis

The complete analysis is in
`scripts/exploration/explore_controls.py`. It deliberately stays below 100
lines so it can be read and edited as part of the post.

It:

- takes the control-data directory as its positional argument;
- verifies the official gene names and order;
- calculates QC metrics for all 55,200 control cells;
- estimates each cell's Good–Turing unseen UMI mass as `singleton genes / total UMIs`;
- saves one five-panel QC figure;
- selects a seeded, balanced subset of cells for the embedding;
- explicitly normalizes, log-transforms, selects highly variable genes, runs
  PCA, builds a neighbor graph, and calculates UMAP; and
- saves the QC panel, Good–Turing distribution, UMAP, and cell-level QC table.

Run it with:

```bash
pixi run python scripts/exploration/explore_controls.py data/controls --seed 0
```

Default outputs are written to
`reports/control-data-exploration/generated/`.

## Interpretation boundary

- The source H5AD files are never overwritten.
- Normalization and feature selection are used only for the derived embedding.
- The UMAP is exploratory; separation or proximity is not proof of cell-line
  identity.
- QC differences are descriptive and do not automatically justify filtering.
- `ntc_id` is a control-construct label, not a confirmed batch label.
