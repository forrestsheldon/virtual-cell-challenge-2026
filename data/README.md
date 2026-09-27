# Data

Challenge data live here locally and are ignored by git.

When the 2026 release is available, record:

- official source URLs and access date;
- filenames, sizes, and checksums;
- the distinction between raw, provided-preprocessed, and locally derived data;
- any commands used to download or transform the files.

Do not commit challenge data to this repository.

## External perturbation data

### Virtual Cell Challenge 2025 H1 training split

- Local files: `external/vcc2025_h1/adata_Training.h5ad`,
  `external/vcc2025_h1/gene_names.csv`, and
  `external/vcc2025_h1/pert_counts_Training.csv`
- Official source: `gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/`
- Accessed: 2026-08-30
- H5AD object generation: `1765904883947296`
- H5AD exact size: 15,482,497,461 bytes
- H5AD CRC32C: `/Z0row==` (verified against the bucket object)
- H5AD SHA-256: `a09977104fefb622368ca74b50c9d3c1e891733e6c83db07acfca49b0219c02b`
- Expression: 221,273 cells x 18,080 genes; CSR `float32` raw UMI counts
  in `X`, with 1,932,554,688 stored nonzero entries
- Cell metadata: `target_gene`, `guide_id`, and `batch`; 151 target labels
  including non-targeting controls, 189 guide categories, and 48 batches
- Gene metadata: `gene_id`; the downloaded 18,080-row `gene_names.csv`
  exactly matches `var_names` in order
- The object contains no layers, embeddings, neighbor graphs, or `uns`
  payload. Sampled `X` entries were non-negative and exactly whole-valued.
- `pert_counts_Training.csv` contains all 150 perturbed target labels, and its
  cell counts exactly match the downloaded H5AD. The two CSVs were verified
  against their official GCS MD5 hashes.

Locally derived Mixscape signature shards are written under
`derived/vcc2025_h1_mixscape/signatures/` by
`scripts/exploration/mixscape_h1_signatures.py`. Each shard contains one
recorded batch, explicitly normalized/log1p expression in `X`, and the Pertpy
Mixscape perturbation signature in `layers["X_pert"]`; the raw source above is
never modified. All 48 batch shards were generated with Pertpy 1.2.0 on
2026-08-31; together they occupy 18 GiB.

### Replogle 2022 K562 genome-wide Perturb-seq

- Local file: `external/replogle2022/ReplogleWeissman2022_K562_gwps.h5ad`
- Source: `https://zenodo.org/api/records/13350497/files/ReplogleWeissman2022_K562_gwps.h5ad/content`
- Accessed: 2026-08-21
- Upstream representation: scPerturb harmonization of the author-processed Replogle K562 GWPS experiment
- Exact size: 8,805,466,154 bytes
- MD5: `13db594f8f1d2ccb88fec44a13e414dc`
- Expression: 1,989,578 cells x 8,248 genes; dense `float32` raw UMI counts in gzip-compressed `X`
- Download method: resumable HTTP transfer, parallelized over non-overlapping byte ranges; ranges were reassembled in order and the complete file was checked against the Zenodo size and MD5 before the temporary chunks were removed.

## External chromatin-accessibility data

- Local root: `external/chromatin_accessibility/`
- Accessed: 2026-09-20
- Contents: 10 GRCh38/hg38 source files across H1, K562, HepG2, HCT116,
  CD4 T cell, HEK293T, Jurkat, a KOLF2.1J-derived iPSC proxy, and RPE1
- Exact size: 463,227,825 bytes (441.77 MiB)
- Representations: nine peak files and one RPE1 CPM-normalized BigWig signal
  track
- Frozen manifest: `metadata/datasets/chromatin_accessibility_sources.json`
- Download and audit scripts: `scripts/data/download_chromatin_accessibility.py`
  and `scripts/data/audit_chromatin_accessibility.py`
- Full provenance, caveats, and intended modeling boundary:
  [`reports/chromatin-accessibility-atlas/README.md`](../reports/chromatin-accessibility-atlas/README.md)

These are untouched public processed files. They have not been peak-merged,
filtered, lifted over, assigned to genes, normalized together, or treated as
matched measurements from the perturbation experiments.

## Cloud-derived storage

The project uses `gs://bold-bastion-509200-f9-vcc2026` for durable derived
pseudobulks, manifests, and restartable processing checkpoints. It is not a
permanent mirror for large public raw datasets.

- Google Cloud project: `bold-bastion-509200-f9`
- Created: 2026-09-20
- Location: `EUROPE-WEST2` (London), single region
- Default class: Standard
- Uniform bucket-level access: enabled
- Public access prevention: enforced
- Soft-delete retention: 604,800 seconds (seven days)
- Object versioning: not enabled

The first write/read check stored
`audits/xatlas-orion-streaming-audit/README.md` (3,196 bytes, CRC32C
`hT79qQ==`, MD5 `JTmPD8xSmBR/oSubV22ajw==`).

### X-Atlas Orion HCT116 and HEK293T pseudobulks

- Source: <https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion>
- Source revision: `53a5bc98d49247bcf967500292575c3d3602de31`
- Accessed and aggregated: 2026-09-20
- Representation: raw, unnormalized integer UMI sums over all 38,606 source
  genes
- Requested panel: union of 126 H1 and 300 VCC 2026 targets (415 unique)
- Coverage in each context: 414 targets; `TAZ` is missing
- Durable prefix:
  `gs://bold-bastion-509200-f9-vcc2026/derived/xatlas_orion/`
- Final target-, construct-, and batch-control-level H5ADs:
  `derived/xatlas_orion/final/{HCT116,HEK293T}/`
- Restartable per-shard checkpoints: `derived/xatlas_orion/shards/`
- Full provenance, exact hashes, counts, and audit results:
  [`reports/xatlas-orion-streaming-audit/README.md`](../reports/xatlas-orion-streaming-audit/README.md)

The public 117.59 GiB source was scanned but not mirrored. The temporary VM
and its scratch and boot disks were deleted after the durable outputs were
hash-verified.

### Compact cross-context pseudobulk atlas

- Local roots: `derived/context_atlas/final/` and
  `derived/xatlas_orion/final/`
- Downloaded and independently audited: 2026-09-20
- Contexts: CD4T, HepG2, Jurkat, K562 essential, KOLF2.1J, RPE1, HCT116, and
  HEK293T
- Local contents: 19 target-, guide-, and control-level H5ADs plus eight
  manifests and eight cloud audit records
- Exact local size: 1,037,791,141 bytes (0.967 GiB)
- Large target-by-batch and target-by-donor-condition matrices remain in the
  Cloud Storage bucket and were deliberately not downloaded
- Compact audit: `scripts/cloud/audit_compact_pseudobulks.py`
- Results, cloud locations, and storage costs:
  [`reports/context-atlas-cloud/README.md`](../reports/context-atlas-cloud/README.md)


### Cell-split transfer release (2026-09-24)

- Version: `split-v1-20260924`; six contexts: K562 essential, RPE1, HepG2, Jurkat,
  HCT116 and HEK293T. Broad shared scPerturb constructs; X-Atlas union plus top200
  multi-construct genes ranked by their weaker qualifying construct across contexts.
- Local compact destination: `derived/context_transfer_splits/split-v1-20260924/<context>/`.
  Downloaded and verified: 48 compact files, 2,084,075,226 bytes (1.94 GiB).
- Contents: raw construct × four cell-split pseudobulks, batch × control-role × split
  controls, cell assignments, batch/split exposures and matched-control support flags.
- Responder-ready raw cell subsets remain cloud-only, retained without automatic expiry.
- Cloud prefix: `gs://bold-bastion-509200-f9-vcc2026/derived/context_transfer_splits/staging/split-v1-20260924/`.
- Cloud source-reconstruction and transport checks passed for all six contexts.
  Local verification: `scripts/analysis/verify_local_split_release.py --core-only`.
- Full audit, hashes, resource state and costs:
  [`split release completion`](../reports/context-transfer-blog/split-v1-20260924/completion/README.md).
- Native gene axes and raw counts are preserved. Respect `matched_control_support.parquet`;
  86 HepG2 strata have missing baseline support and must not silently use pooled controls.
