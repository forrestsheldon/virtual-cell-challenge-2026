# X-Atlas/Orion streaming audit

Accessed 2026-09-19. This is an acquisition audit, not an expression-data
derivative.

## Sources

- Hugging Face dataset:
  <https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion>
- Hugging Face repository commit reported by the resolver:
  `53a5bc98d49247bcf967500292575c3d3602de31`
- Author-processed Figshare record, version 3:
  <https://doi.org/10.25452/figshare.plus.29190726>
- HCT116 Figshare object: <https://ndownloader.figshare.com/files/55021257>
- HEK293T Figshare object: <https://ndownloader.figshare.com/files/55074802>

## Target coverage

The union of the 126 H1 benchmark targets and 300 VCC 2026 targets contains
415 genes. Both X-Atlas contexts contain 414 of them; `TAZ` is the sole
missing target. This is a pairwise coverage calculation against each panel,
not a three-way-intersection requirement.

More specifically, the panels share 11 genes, so the union is
`126 + 300 - 11 = 415`. X-Atlas contains 125/126 H1 targets and all 300 VCC
2026 targets in both contexts. The shared panel genes are `AKT2`, `MED13`,
`MTA1`, `RNF2`, `SIN3B`, `SMARCA5`, `STAT6`, `TARBP2`, `TRAPPC6A`, `TWF2`,
and `ZNF714`.

## Hugging Face Parquet layout

The repository contains 332 cell-level Parquet shards totaling
126,259,821,037 bytes (117.59 GiB):

| Context | Files | Bytes | GiB |
|---|---:|---:|---:|
| HCT116 | 109 | 46,576,484,789 | 43.38 |
| HEK293T | 223 | 79,683,336,248 | 74.21 |

The files contain sparse raw counts as paired `gene_token_id` and
`gene_expression` list columns. Cells for a target are distributed across
batches, so Hugging Face streaming avoids local storage but does not avoid a
scan of the full 117.59 GiB.

`HCT116_Batch1.parquet` was inspected as a small representative shard. It has
18,549 cells, one row group, 10,076 target labels, and only 346 cells covering
220 genes from the H1/VCC target union. Its SHA-256 is
`a7a6918ecda30343e3f9262bcfb654bf077170d2e99c7bcd0c8b9f963764f13b`.
It is not a usable atlas-level pseudobulk by itself.

## Figshare H5AD layout and selective-read estimate

The author H5ADs contain CSR raw-count matrices with uncompressed `data`,
`indices`, and `indptr` arrays:

| Context | Shape | Remote size |
|---|---:|---:|
| HCT116 | 3,409,169 x 38,606 | 209,354,246,272 bytes |
| HEK293T | 4,534,299 x 38,606 | 350,164,035,901 bytes |

Metadata-only range reads identified 65,900 HCT116 and 88,325 HEK293T cells
for the 414 covered targets. The exact selected sparse-array payload is only
3.76 and 6.33 GiB, respectively. However, target cells are interspersed and
the HDF5 arrays are chunked. Reading all touched chunks would transfer about
57.55 GiB for HCT116 and 97.85 GiB for HEK293T before controls. Adding a
deterministic sample of 10,000 controls raises these estimates to 64.53 and
106.96 GiB. Selective HDF5 range access is therefore no better than scanning
the 117.59 GiB Parquet release.

The metadata audit transferred 50,088,064 bytes for HCT116 and 61,286,717
bytes for HEK293T; it did not retrieve expression arrays.

## Completed cloud aggregation

The approved full scan ran on 2026-09-20 from the pinned Hugging Face commit.
Each Parquet shard was downloaded independently, checked against its LFS
SHA-256, aggregated, uploaded as a restartable checkpoint, and deleted from
the worker. No full source dataset was retained. The outputs contain raw,
unnormalized integer UMI sums on the complete 38,606-gene X-Atlas axis.

| Context | Source shards | Perturbed cells | Control cells | Targets | Guides | Raw UMIs |
|---|---:|---:|---:|---:|---:|---:|
| HCT116 | 109 | 65,900 | 165,777 | 414 | 1,455 | 4,487,063,129 |
| HEK293T | 223 | 88,325 | 218,838 | 414 | 1,456 | 7,278,520,474 |

For each context, three H5ADs were retained: target-pooled pseudobulks,
construct-level pseudobulks, and one non-targeting control pseudobulk per
source batch. They are stored under
`gs://bold-bastion-509200-f9-vcc2026/derived/xatlas_orion/final/`. The six
H5ADs total 275,074,219 bytes. Restartable shard checkpoints and logs remain
under the same `derived/xatlas_orion/` prefix; the full prefix occupies
6,534,842,578 bytes at this checkpoint.

The post-write audit verified every H5AD SHA-256 against its manifest, raw
integer and non-negative values, exact cell and UMI totals, identical gene
axes, exact reconstruction of every target row by pooling its construct rows,
and exact reconstruction of the global control row from batch controls. The
cloud copies were streamed back through SHA-256 after upload and matched all
six local output hashes. Compact manifests and audit results are retained in
[`generated/`](generated/).

The reproducible programs are
[`aggregate_xatlas_shards.py`](../../scripts/cloud/aggregate_xatlas_shards.py),
[`merge_xatlas_shards.py`](../../scripts/cloud/merge_xatlas_shards.py), and
[`audit_xatlas_pseudobulks.py`](../../scripts/cloud/audit_xatlas_pseudobulks.py).
