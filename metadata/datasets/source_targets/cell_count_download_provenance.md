# Author-object cell-count extraction provenance

Accessed 2026-08-21/2026-08-22. Large inputs were downloaded under explicit
approval to `/tmp/vcc_counts_20260821`, verified, reduced to the count/audit CSVs
in this directory, and deleted. They are not required to regenerate the atlas
from the retained count tables.

## Jiang 2025

Source record: <https://zenodo.org/records/14518762>

| file | bytes | published and verified MD5 |
|---|---:|---|
| `Seurat_object_IFNB_Perturb_seq.rds` | 4,326,548,669 | `3eb5e7af1601bf562a5b20dea5de3dc9` |
| `Seurat_object_IFNG_Perturb_seq.rds` | 2,915,636,149 | `0fef1f14c36906e9c40e4d1c6aae6926` |
| `Seurat_object_INS_Perturb_seq.rds` | 5,601,176,410 | `c7b830dfcc020545c3f222cad5b13b34` |
| `Seurat_object_TGFB_Perturb_seq.rds` | 2,642,041,433 | `8e9b4d39a95ec5881a30be6a2df541d1` |
| `Seurat_object_TNFA_Perturb_seq.rds` | 4,656,209,976 | `60ed8bff6c749b1250f8fde9c5435c2e` |

Extraction used author `meta.data$gene` grouped by `meta.data$cell_type` and
summed the five pathway objects. `NT` is control. No multiple-gene assignment
labels were present. Every non-control cell mapped to one of the 218 retained
targets.

## McFaline-Figueroa 2024

Source:
<https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM7056nnn/GSM7056149/suppl/GSM7056149_sciPlexGxE_2_preprocessed_cds.list.RDS.gz>

- Exact bytes: 3,619,987,565
- Downloaded-file SHA-256:
  `3e5c6755c737dba099c61db637568f81835e8c692f77cba1cb1f62ac4a794207`
- GEO's outer gzip expands to a 3,641,798,100-byte, internally compressed RDS.

Extraction used author `colData$gene_id` from the A172, T98G, and U87MG CDS
objects. `NA`, `NTC`, `random`, and missing assignments are controls.
Comma-separated multiple-gene labels are present and excluded from the
single-gene atlas counts. Every remaining assigned cell mapped to one of the 522
retained targets; treatment conditions were counted together, matching the
current atlas context definition.

The initially inspected
`GSM7056149_sciPlexGxE_2_cell.annotations.txt.gz` was 115,324,478 bytes with
SHA-256
`7fffab7b4c323bdda5c21d6a71d499d9fde2748fc98e6150ba2342f3ab660156`.
It contains only cell barcode and experiment label, so it cannot establish
`gene_id` or cell-line-specific counts and was not used for the retained values.
