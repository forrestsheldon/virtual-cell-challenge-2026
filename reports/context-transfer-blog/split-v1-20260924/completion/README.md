# Split release completion and costs

All six contexts completed their independent source-reread validation and cloud transport
checks on 2026-09-24 at 14:44 UTC. The worker automatically stopped at 14:45 UTC.
No canonical data were overwritten and no responder model was fitted.

Project: `bold-bastion-509200-f9`; zone: `europe-west2-b`.
Instance `vcc-splits-20260924` is **TERMINATED** (stopped, not deleted).
The 30 GiB balanced boot disk and 250 GiB standard scratch disk remain attached and retained.
Inventory scope is this release only; unrelated resources were not inspected.

## Validation and artifact locations

The six `acceptance.json` records report exact full-source reaggregation, exact responder
source-row counts, exact reconstruction of applicable existing guide pseudobulks and matching
control-role margins. The `transport-audit.json` hashes are anchored by `status.json`.
Every context's manifest, acceptance and transport audit is copied alongside this report.
HepG2 has 86 flagged missing-baseline strata; the remaining contexts have none. Preserve these
support flags: missing matched controls must not silently become pooled-control substitutions.

Cloud release:
`gs://bold-bastion-509200-f9-vcc2026/derived/context_transfer_splits/staging/split-v1-20260924/`

Local destination:
`data/derived/context_transfer_splits/split-v1-20260924/<context>/`

Compressed analysis artifacts total 1.94 GiB; responder-ready cell objects total 7.29 GiB.
The full active cloud prefix, including launch inputs/logs, is 9.526 GiB. Derived responder
objects have no automatic expiry. Full source files remain on the worker scratch disk only.

Forrest chose compact-only local acquisition on 2026-09-24; responder-ready cells remain
cloud-only. **Local acquisition and verification passed for all six contexts:** 48 files,
2,084,075,226 bytes (1.94 GiB). `local-core-acceptance.json` records the verified local scope; do not treat partial `.gstmp` files as usable artifacts.
Verify with `scripts/analysis/verify_local_split_release.py --core-only` after acquisition completes.
The resumable downloader `scripts/analysis/download_split_core.py` runs this check automatically.

## Cost estimate

USD list prices, not an invoice. Exact inputs and calculations are in `cost-estimate.json`;
regional SKU evidence is in `storage-pricing-skus.json` and the parent `pricing-skus.json`.

| Item | Estimate |
|---|---:|
| VM compute, 5.185345 hours | $2.42 |
| Disks during that interval | $0.11 |
| External IP during run, allowance | $0.03 |
| Total run | $2.55 |
| Retained disks, ongoing | $15.60/month |
| Active cloud release, London Standard at $0.023/GiB-month | $0.22/month |
| Total ongoing while disks retained | $15.82/month |
| Ongoing after approved worker/disk deletion | $0.22/month |

A full 9.23 GiB local download adds approximately $1.11 at the $0.12/GiB Europe/North America
transfer tier, before any applicable free allowance. A compact-only download is approximately
$0.23. Actual destination, account tier, credits and discounts may alter these amounts.
Operations, tax, currency conversion and soft-deleted temporary/versioned objects are excluded.
Disks continue accruing after shutdown until explicitly deleted. No further VM compute charges
accrue while stopped.

## Cleanup boundary

The source-bearing scratch disk and review-retained boot disk have not been deleted.
After the requested local copies pass verification, recommend deleting only this stopped worker
and these two named disks. Durable derived pseudobulks and responder objects must remain.
AGENTS.md requires explicit approval before deleting source data or review-retained disks.

## Validated cloud coverage

| Context | Selected cells | Retained cells in cloud | Compact GiB | Cell-object GiB |
|---|---:|---:|---:|---:|
| K562_essential | 309,610 | 39,291 | 0.236 | 0.328 |
| RPE1 | 209,264 | 35,061 | 0.222 | 0.284 |
| HepG2 | 145,471 | 20,918 | 0.265 | 0.202 |
| Jurkat | 262,950 | 43,776 | 0.266 | 0.357 |
| HCT116 | 346,394 | 171,166 | 0.397 | 2.273 |
| HEK293T | 455,287 | 225,815 | 0.556 | 3.841 |

Local acceptance SHA-256: `2aab52362decee42fb17043fedeaead5f5143f15437244729e895b234026c618`.

All applicable immutable reference constructs reconstructed exactly in the local check.
HepG2’s 86 missing-baseline strata affect 78 constructs / 77 targets in batch 21, splits 2/3;
see `missing-control-support.json`. They remain explicitly unavailable for matched-baseline
comparisons requiring those strata. No replacement controls or transformations were applied.
