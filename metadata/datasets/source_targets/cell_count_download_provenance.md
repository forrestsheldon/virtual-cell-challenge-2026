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

## X-Atlas/Orion 2025 (Huang et al.)

Accessed 2026-09-15. Source: Hugging Face dataset
`Xaira-Therapeutics/X-Atlas-Orion` (bioRxiv <https://doi.org/10.1101/2025.06.11.659105>;
CC-BY-NC-SA-4.0), files `data/HCT116_Batch*.parquet` (109) and
`data/HEK293T_Batch*.parquet` (223).

No full parquet was downloaded. Using pyarrow over an HTTP-range file object, only
the `gene_target` and `pass_guide_filter` columns were read from each batch; the
`gene_token_id`/`gene_expression` arrays were never fetched. Total transfer was
about 58 MB across all 332 batches (roughly 0.16 MB per 300-500 MB file). The
882 KB `metadata/gene_metadata.parquet` was downloaded once to confirm every
target label is a valid gene symbol.

Counts are cells per `gene_target` with `pass_guide_filter` true (all released rows
passed). The source control label `Non-Targeting` is stored as `control`. Per line:

| line | targets | perturbed cells | control cells | median cells/target | VCC-300 present |
|---|---:|---:|---:|---:|---:|
| HCT116 | 18,293 | 3,243,392 | 165,777 | 150 | 300 / 300 |
| HEK293T | 18,311 | 4,315,461 | 218,838 | 200 | 300 / 300 |

The combined perturbed + control total (7,943,468) matches the paper's "eight
million cells." The library design targets 18,903 genes (41,780 sgRNAs plus 1,026
non-targeting pairs); the smaller per-line target counts are genes with at least
one passing single-sgRNA cell in the released data.

## Nourreddine et al. 2024/2026 (KOLF2.1J iPSC atlas)

Accessed 2026-09-15. Source: Figshare+ article 27261219
(DOI 10.25452/figshare.plus.27261219; Nourreddine et al., *Nature Biotechnology*,
10.1038/s41587-026-03199-w). File `KOLF_Pan_Genome_QC_Filtered.h5ad`
(189,393,177,972 bytes; download `https://ndownloader.figshare.com/files/64650261`).

No full file was downloaded. Using `scripts/metadata/inspect_remote_h5ad.py`
(h5py over a budget-guarded HTTP-range reader), only the `obs/gene_target`
categorical `categories` and `codes` were read — about 12 MB of the 189 GB file.
Counts are cells per `gene_target`; the single control label `NTC` (146,747 cells)
is stored as `control`, and `KNTC1` is a gene.

Result: 11,687 targets, 2,512,462 perturbed cells, median 218 cells/target, and
282 of 300 VCC targets present. Total (2,659,209) matches the h5ad `n_obs`. The
h5ad X representation (raw vs normalized) was not verified.

## Zhu et al. 2026 (primary CD4+ T cell genome-scale Perturb-seq)

Accessed 2026-09-15. Source: Biohub Virtual Cells Platform public S3 bucket
`s3://genome-scale-tcell-perturb-seq/marson2025_data/` (Zhu, Dann, … Marson;
*Cell* 2026; bioRxiv 10.64898/2025.12.23.696273; GEO GSE314342 / SRA SRP643211).
File `GWCD4i.pseudobulk_merged.h5ad` (~44.6 GB).

No full file was downloaded. Using a budget-guarded HTTP-range reader with h5py,
only `obs/perturbed_gene_name` (categories + codes), `obs/guide_type`, and the
per-pseudobulk `obs/n_cells` were read (~12 MB). Per-gene cell counts are the sum
of `n_cells` over all targeting-guide pseudobulks for that gene, collapsed across
4 donors and 3 culture conditions (Rest, Stim8hr, Stim48hr). Non-targeting guides
(939,535 cells) are stored as `control`.

Result: 12,730 targets, 21,056,730 perturbed cells, median 1,508 cells/target, and
297 of 300 VCC targets present. Perturbed + control (21,996,265) matches the
reported ~22 million cells. The cell-level matrices (`D*_*.assigned_guide.h5ad`,
118–173 GB each) were not used; the pseudobulk `n_cells` give exact per-gene totals.
