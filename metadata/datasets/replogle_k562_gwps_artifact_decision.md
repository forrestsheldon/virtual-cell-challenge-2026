# Replogle K562 GWPS artifact decision

Last verified: **2026-08-21**. Neither artifact was downloaded. Exact sizes came
from repository APIs; structure and selected values were inspected through
HTTP byte ranges without saving either H5AD.

## Compared artifacts

| property | author/Figshare | scPerturb |
|---|---|---|
| exact file | `K562_gwps_raw_singlecell_01.h5ad`, Figshare file 35775507 | `ReplogleWeissman2022_K562_gwps.h5ad`, Zenodo record 13350497 |
| exact size | **65,830,941,948 bytes** (61.31 GiB) | **8,805,466,154 bytes** (8.20 GiB) |
| cells x genes | **1,989,578 x 8,248** | **1,989,578 x 8,248** |
| expression | dense `float32` raw UMI counts in `X`; contiguous and uncompressed | dense `float32` raw UMI counts in `X`; gzip-compressed HDF5 chunks |
| other expression matrices | none (`X`, `obs`, and `var` are the only root groups) | none; `layers`, `raw`, embeddings, graphs, and precomputed DE are absent |
| raw counts present | yes, in `X` | yes, in `X` |

The size ratio is **7.48x**. The author file costs 57,025,475,794 additional
bytes (53.11 GiB), principally because its dense matrix is uncompressed. The
scPerturb transformation reads the author H5AD, keeps `adata.X`, adds/renames
metadata, and writes gzip-compressed HDF5. It performs no normalization,
log-transformation, cell subsetting, or gene subsetting for this public file.
Three byte-range-inspected rows (first, middle, and last) were exactly equal
between files and were non-negative, whole-valued raw counts.

## Gene universe

Both artifacts contain the same **8,248 gene rows**, already restricted by the
author deposit to genes expressed above 0.01 UMI per cell. The author file uses
Ensembl gene IDs as the `var` index and stores gene symbols in `var.gene_name`.
It has 8,246 distinct symbol categories because `TBCE` and `HSPA14` each map to
two Ensembl rows. scPerturb keeps the Ensembl IDs in `var.ensembl_id`, makes gene
symbols the index, and disambiguates those four rows as:

- `TBCE_ENSG00000284770` and `TBCE_ENSG00000285053`;
- `HSPA14_ENSG00000187522` and `HSPA14_ENSG00000284024`.

This is an identifier change that exact-symbol joins must handle, not a gain or
loss of expression features.

## Perturbation, guide, batch, and replicate metadata

The author `obs` columns are `gem_group`, `gene`, `gene_id`, `transcript`,
`gene_transcript`, `sgID_AB`, `mitopercent`, `UMI_count`, `z_gemgroup_UMI`,
`core_scale_factor`, and `core_adjusted_UMI_count`. They include **9,867**
perturbation categories (including non-targeting), **11,187** guide / gene-guide
categories, transcript annotations, QC/depth covariates, and GEM-group batch.

scPerturb retains every one of those cell-level fields, renaming `gem_group` to
`batch`, `sgID_AB` to `guide_id`, and `mitopercent` to `percent_mito`. It adds
standardized `perturbation`, cell-line/disease/organism fields, `nperts`, and
recomputed count/gene/ribosomal summaries. The perturbation, guide,
gene-transcript, and transcript category sets are unchanged. One unused literal
`nan` category in the author `gene_id` categorical dictionary is not carried
forward; no cell used it.

The author H5AD has no separate biological-replicate column. Its only explicit
batch-like variable is `gem_group`, and scPerturb preserves those values as
`batch`. Neither file therefore supplies independent replicate metadata beyond
the GEM-group field; `batch` must not silently be interpreted as a biological
replicate.

## Information-loss assessment

No scientifically important information present in this author H5AD was found
to be absent from scPerturb:

- no cells or genes are removed;
- raw counts and their cell/gene order are retained;
- target, guide, transcript, depth/QC, and GEM-group metadata are retained;
- the author artifact contains no layers, `raw`, embeddings, graphs, `uns`, or
  precomputed DE results that could be lost.

The only material changes are reversible metadata harmonization, four explicit
duplicate-symbol repairs, derived QC columns, and compressed storage. These
changes matter for identifier bookkeeping but do not justify retaining the
uncompressed author copy.

## Recommendation

**Download scPerturb.** It preserves the author artifact's cells, full 8,248-gene
raw-count matrix, guide/perturbation annotations, and GEM-group batch metadata,
while saving 57.03 GB. The author artifact provides no additional matrix or
scientifically important annotation sufficient to justify the 7.48x download
cost.

## Sources

- [Author Figshare record 20029387](https://plus.figshare.com/articles/dataset/20029387), including its description of raw single-cell counts and the >0.01 UMI/cell gene threshold; [Figshare API metadata](https://api.figshare.com/v2/articles/20029387) for exact file size. Accessed 2026-08-21.
- [scPerturb Zenodo record 13350497](https://zenodo.org/records/13350497) and [record API](https://zenodo.org/api/records/13350497) for exact file size and checksum. Accessed 2026-08-21.
- [scPerturb Replogle transformation at commit b69f72a](https://github.com/sanderlab/scPerturb/blob/b69f72a070a92bcbaf41e7f9897b11598109ab48/dataset_processing/scripts/ReplogleWeissman2022.py) and its [writer implementation](https://github.com/sanderlab/scPerturb/blob/b69f72a070a92bcbaf41e7f9897b11598109ab48/utils.py). Accessed 2026-08-21.
- [Replogle et al., Cell 2022](https://doi.org/10.1016/j.cell.2022.05.013), primary experiment. Accessed 2026-08-21.
