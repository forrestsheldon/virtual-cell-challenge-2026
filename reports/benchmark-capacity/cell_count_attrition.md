# Cell-count coverage and E3--E5 attrition

`n_cells` is the number of cells in the dataset's current curated representation
assigned to a perturbation label. No new biological quality filter is applied.

## Coverage completeness

All 14 dataset-contexts and all 20,073 canonical observation rows now have exact
per-perturbation counts. There are no partially counted or unresolved contexts.

| dataset-context(s) | assignment, controls, and scope |
|---|---|
| `vcc2025_h1::H1` | `obs["perturbation"]`; `control` is control; only single-gene perturbations occur. Counts include every assigned cell retained after scPertEval's documented cell QC. |
| `replogle2022::{K562,RPE1}` | `obs["perturbation"]`; `control` is control (`gene` uses `non-targeting` upstream); `nperts` is 0 for controls and 1 otherwise. Counts include every assigned cell in the scPerturb representation. Duplicate K562 panel targets retain the larger panel count and are not summed. |
| `nadig2025::{HepG2,Jurkat}` | Author `obs["gene"]` is copied to `obs["perturbation"]`; `non-targeting` becomes `control`; `nperts` is 0/1. Counts include every assigned cell in the scPerturb harmonized objects. |
| `jiang2025::{A549,MCF7,HT29,HAP1,BxPC3,K562}` | Author Seurat `meta.data$gene` defines assignment and `cell_type` defines context; `NT` is control. No multiple-gene labels occur. Counts include every assigned cell and are summed across the five pathway objects. |
| `mcfaline2024::{A172,T98G,U87MG}` | Author CDS `colData$gene_id` defines assignment; `NA`, `NTC`, `random`, or missing labels are controls. Comma-separated multiple-gene assignments occur and are excluded from the atlas's single-gene observations. Counts otherwise include every represented cell across the currently collapsed treatment conditions. |

Author-representation totals used for the newly resolved contexts:

| dataset-context | total represented cells | controls/missing | multiple-gene labels | single-target cells counted |
|---|---:|---:|---:|---:|
| Jiang A549 | 207,261 | 9,634 | 0 | 197,627 |
| Jiang MCF7 | 260,545 | 10,819 | 0 | 249,726 |
| Jiang HT29 | 360,963 | 20,023 | 0 | 340,940 |
| Jiang HAP1 | 277,261 | 15,784 | 0 | 261,477 |
| Jiang BxPC3 | 314,758 | 18,092 | 0 | 296,666 |
| Jiang K562 | 207,688 | 9,917 | 0 | 197,771 |
| McFaline A172 | 335,088 | 69,836 | 34,774 | 230,478 |
| McFaline T98G | 325,455 | 77,040 | 40,991 | 207,424 |
| McFaline U87MG | 328,756 | 59,616 | 36,814 | 232,326 |

## Attrition

Main comparison: `n_gene_context_evaluations`.

| n_min | E3a | E3b | E4 | E5 |
|---:|---:|---:|---:|---:|
| 50 | 14,472 | 7,784 | 4,657 | 9,905 |
| 100 | 11,120 | 4,642 | 2,639 | 6,255 |
| 200 | 8,866 | 3,036 | 1,002 | 3,549 |
| 400 | 5,632 | 1,752 | 117 | 1,862 |
| 800 | 2,624 | 697 | 5 | 727 |

Unique genes are reported separately.

| n_min | E3a | E3b | E4 | E5 |
|---:|---:|---:|---:|---:|
| 50 | 2,399 | 2,399 | 2,353 | 2,773 |
| 100 | 1,465 | 1,465 | 1,286 | 1,827 |
| 200 | 848 | 848 | 381 | 945 |
| 400 | 471 | 471 | 44 | 498 |
| 800 | 167 | 167 | 2 | 178 |

The number of nonempty target contexts is 13 for E3a/E3b at every threshold;
11, 11, 11, 11, and 5 for E4; and 14 for E5 at every threshold. H1 is not an
E3 target because it has no second context in the same study.

E4 is the main collapse point: it falls from 1,002 evaluations at 200 cells to
117 at 400 and 5 at 800 because both the Replogle anchor and target must clear
the threshold. E3a remains larger at high thresholds because Jiang supplies six
dense within-study contexts, McFaline supplies three, and every ordered E3a
source-target pair is counted separately. E3b folds permitted sources into one
holdout per target. E3b and E5 additionally require qualified source exposure
(`k_g_qualified` at least one) as well as a qualified target observation.

Full per-target counts are in `cell_count_attrition_by_target_context.csv`;
ordered E3a pair counts are in `E3a_pairwise_cell_thresholds.csv`;
per-gene E3b/E5 `k_g_qualified` is in `cell_count_qualified_exposure.csv`.
