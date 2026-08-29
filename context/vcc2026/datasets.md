---
last_verified: 2026-08-29
status: current
---

# Challenge datasets

## TARGET-DOMAIN DATA

The validation bundle is panel `vcc2026-val-1` and is already present under `data/controls/` (about 662 MB uncompressed on disk; the live page describes the bundle as about 630 MB). It contains:

| file | bytes | contents |
|---|---:|---|
| `context_A.h5ad` | 224,973,316 | 18,400 A controls × 18,533 genes |
| `context_B.h5ad` | 210,966,111 | 18,400 B controls × 18,533 genes |
| `context_C.h5ad` | 226,056,691 | 18,400 C controls × 18,533 genes |
| `gene_names.csv` | 118,786 | required ordered gene axis |
| `pert_counts.csv` | 1,949 | 300 unique target genes |
| `manifest.json` | 715 | release and schema counts |

Backed inspection found sparse CSR `float32` raw UMI counts in `X`, no layers/embeddings/`uns`, and `obs = {target_gene, context, ntc_id}`. A 256-cell sample from each file was non-negative and exactly whole-valued (sample maxima 629/663/905). `var` has the gene-name index and no additional columns. Every supplied cell has `target_gene = non-targeting`. Each context pools **46 distinct NTC constructs × 400 cells = 18,400 controls**. `ntc_id` must therefore be retained: collapsing it prematurely destroys construct-aware baseline and QC options.

The official page states 10x Genomics Flex chemistry, median depth about 20,000 UMI/cell, 400 cells per target in withheld truth, and target selection for greater than 80% on-target knockdown. Final D/E/F controls and their new target panel are scheduled separately.

## EXTERNAL TRAINING / REFERENCE DATA

The live page recommends VCC 2025 H1; Replogle 2022; Nadig 2025; Jiang 2025; Srivatsan 2020; McFaline-Figueroa 2024; Tahoe-100M; and scBaseCount/Arc Virtual Cell Atlas. These are not one homogeneous training corpus:

- Replogle, Nadig, Jiang, and parts of McFaline-Figueroa provide genetic CRISPRi responses.
- Srivatsan, Tahoe-100M, and parts of McFaline-Figueroa provide chemical responses.
- scBaseCount and most of the Arc Virtual Cell Atlas are observational context references.

Access and preprocessing lineage—not just paper identity—are audited in `metadata/datasets/` before any canonical representation is chosen.
