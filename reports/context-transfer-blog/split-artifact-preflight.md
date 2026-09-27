# Cell-split artifact preflight — compact revision

**Revised 2026-09-24. Proposal only; no source acquisition or cloud build authorized.**

This revision replaces fine-batch expression with pooled construct splits. Broad scPerturb remains the working scope; X-Atlas keeps the 415-target union with a proposed 100- or 200-gene addition. H1 is unchanged. KOLF is excluded. The original preflight and its hashes are preserved in [archive/split-artifact-preflight-v1](archive/split-artifact-preflight-v1/README.md). Its 48–744 GiB storage scenarios are superseded, not current requirements.

## Compact core

- Raw **construct × four cell-split pseudobulks**, including control identity × split rows, pooled across source batches.
- Raw **batch × frozen control-role × split control pseudobulks** for matching baselines.
- Selected-cell assignments and **counts/UMI exposures** by construct or control identity × batch × split, without a gene-expression vector for each fine stratum.
- Source manifests, canonical logical hashes, file SHA-256s and reconstruction audits.

No construct × batch × split or control identity × batch × split expression is required. Responder analysis is deferred, but raw responder-ready cell subsets are retained separately at Forrest’s request; whole-batch-fold expression remains a separate possible extension. Preserve native gene axes, raw integer counts, source labels and encoded P-classes without normalization or promoter reinterpretation. Details: [artifact proposal](../../plans/context-transfer-split-artifacts.md) and [analysis plan](../../plans/context-transfer-split-power.md).

## X-Atlas panel: weaker-construct criterion selected by Forrest

For each exact shared construct, take min(HCT116 cells, HEK293T cells). Rank each eligible gene by its **second-best** construct on that measure; break ties by total cells across all shared constructs in both contexts, then gene label. There are 1,491 eligible multi-construct genes. Retain all shared constructs for selected genes, not just the two used for ranking. This favors well-sampled surviving perturbations and does not represent an unbiased genome-wide sample.

| Added panel | New genes beyond union | Resulting X-Atlas targets / constructs per context | Minimum qualifying-pair cells, each context | Approx. quarter / half budget at that minimum | Core, all six contexts |
|---|---:|---:|---:|---:|---:|
| Top 100 | 100 | 514 / 639 | 217 | 54.2 / 108.5 | 9.12 GiB |
| Top 200 | 199 | 613 / 842 | 179 | 44.8 / 89.5 | 9.85 GiB |

The count guarantee applies to the two qualifying constructs in each context, not weaker additional constructs. Quarter/half figures divide pooled counts; final batch-aware assignments need not have perfectly equal pooled sizes. Top100 is nested within top200. The original X-Atlas panel covers 414 of the 415 union targets and contains 431 perturbation constructs per context.

**Recommendation:** top200 is a reasonable next proposal: it costs about 0.73 GiB more than top100 under conservative core budgets while preserving at least 179 cells per qualifying construct per context. Panel size still needs approval. [Top100 list](split-artifact-xatlas-top100.csv), [top200 list](split-artifact-xatlas-top200.csv), [all rankings](split-artifact-xatlas-ranked-genes.csv).

For comparison, total-cell ranking gives a minimum qualifying-pair count of only 8 in both the top100 and top200 lists; it can hide a weak partner behind one abundant construct. It overlaps the chosen lists by 53/100 and 124/200 genes. It is retained as a comparison, not the selected criterion.

## Revised storage

All core figures below include all four scPerturb contexts at broad shared-construct scope plus both X-Atlas contexts. Expression figures are **uncompressed CSR upper bounds**, using int64 counts, int32 column indices and int64 row pointers. Metadata sizes are estimates; container metadata and working copies are additional. No compression savings are assumed.

| X-Atlas scope | Core budget | Retained cell CSR estimate | Total estimate |
|---|---:|---:|---:|
| Current union | 8.37 GiB | 31.71 GiB | 40.08 GiB |
| Union + top100 | 9.12 GiB | 31.71 GiB | 40.83 GiB |
| Union + top200 | 9.85 GiB | 31.71 GiB | 41.57 GiB |

Responder analysis remains deferred; **retain raw single-cell subsets** for the 115/135 broad scPerturb multi-construct targets and the original 17 X-Atlas multi-construct targets, with all eligible controls from their batches. Added X-Atlas genes remain pseudobulk-only. This adds **31.71 GiB** of uncompressed CSR estimates, with X-Atlas density uncertain by 0.5x–2x. Preserve native genes, cell locators, source metadata and split assignments; validate subset counts and reconstructed aggregates independently. These are future build outputs, not objects already produced. Retain durably for the anticipated research within two weeks, with **no automatic expiry**; later deletion requires user approval. Iterative responsibilities/DE and held-out calibration remain future development.

Core expression bounds use a partition property: partitions cannot introduce genes absent from the existing pooled row, so sum split nnz is at most four times pooled nnz; cap by input nnz where known. New X-Atlas constructs have no local pooled expression reference and use the full native-gene ceiling. Batch-role controls use the same support bound. Metadata allowances are 160 bytes per selected cell and 80 bytes per fine exposure row, with dictionary-encoded identifiers expected. Budget working margin rather than treating these as measured compressed file sizes.

| Context, with union + top200 | Targets / constructs | Selected cells incl. controls | Median pooled construct cells | Core budget |
|---|---:|---:|---:|---:|
| K562_essential | 2,055 / 2,171 | 309,610 | 116 | 1.00 GiB |
| RPE1 | 2,055 / 2,171 | 209,264 | 68 | 1.01 GiB |
| HepG2 | 2,392 / 2,548 | 145,471 | 43 | 1.26 GiB |
| Jurkat | 2,392 / 2,548 | 262,950 | 80 | 1.19 GiB |
| HCT116 | 613 / 842 | 346,394 | 208 | 2.35 GiB |
| HEK293T | 613 / 842 | 455,287 | 273 | 3.05 GiB |

## What batch information still does

Replogle batch labels were renamed from gem_group. Nadig supplies obs/batch; its experimental meaning remains unverified. X-Atlas sample labels correspond one-to-one with the 109 HCT116 and 223 HEK293T shards. These labels are not established biological replicates.

Assign cells by deterministic hash order and round-robin within batch × exact construct, with a stratum-specific starting rotation; then pool expression across batches. **Do not require four cells in every batch or discard sparse strata.** The earlier occupancy measurements remain valid metadata, but n≥4 per batch is not an analysis gate. Gate on actual pooled split support. Record batch composition and exposures in the small tables.

A matched baseline is the sum of batch × baseline-role × split control rates weighted by that construct split's batch UMI exposure fractions. It is computable from this core. It does not allow arbitrary later perturbation-batch reweighting. If a required control batch/split is missing, flag the estimate as unavailable; pooled perturbation expression cannot be retroactively restricted to the supported batches. Prespecify any alternative aggregate before acquisition rather than silently pooling controls or filtering cells. The illustrative identity-role allocation leaves one HepG2 batch with only three baseline controls; final support must be checked after assignment.

Cell halves measure reliability within the recorded experiment/batch mixture. Shared systematic effects can reproduce, and different split mixtures can change the underlying effect. The off-diagonal product is shared split signal; interpretation as a single biological effect's power needs the corresponding assumptions and null checks. Whole-batch validation or arbitrary batch bootstrap requires additional approved expression retention. No such extension is silently promised by the compact core.

## Preserved provenance and acquisition

The initial Phase A checked all 14 retained H5AD hashes, all 332 X-Atlas shard source identities, every existing construct cell count reconstructed from source metadata, and scPerturb metadata UMI totals. The revision uses these cached metadata and local pooled-gene supports; it performs no new source/network acquisition. Full expression-file SHA-256s must still be verified after approved acquisition. Exact source sizes/URLs/MD5/SHA-256, axes, duplicate groups, control identities, batches and coverage quantiles remain in [the JSON](split-artifact-preflight.json).

The four scPerturb sources are dense compressed float32 raw-count X, totaling 4,927,873,119 bytes (4.59 GiB), absent from repository data/external at preflight. Author processing and scPerturb harmonization are not independent experiments. No extra normalization is introduced. Replogle core_scale_factor is retained verbatim where present; capture-exposure semantics are unresolved.

X-Atlas remains pinned to commit 53a5bc98d49247bcf967500292575c3d3602de31: 109 HCT116 shards (46,576,484,789 bytes) and 223 HEK293T shards (79,683,336,248 bytes), **117.59 GiB total**. Largest shard is 717,945,523 bytes. Under full-shard checksum validation, top100 and top200 require the same source scan. Smaller panels save output storage and aggregation work, not source download volume. Existing checkpoint sums cannot reconstruct cell splits or responders.

Paired common native axes remain 7,226 / 7,632 / 38,606 gene IDs; X-Atlas has 38,584 literal symbols and 21 duplicate groups. Preserve all native columns; freeze any downstream mapping and target exclusion separately. The broader scPerturb pairs retain 115 / 135 multi-construct targets. H1 and the original 415-target union remain unchanged.

## Revised cloud envelope — not authorization

- Project bold-bastion-509200-f9, configuration vcc-2026, region europe-west2, proposed zone europe-west2-b; existing bucket-scoped worker service account and network. Verify boundary again before any mutation.
- Candidate e2-highmem-8: 8 vCPU / 64 GiB; **30 GiB balanced boot + 100 GiB balanced scratch**, replacing the prior 300–2,000 GiB scratch proposals. One context/shard at a time, bounded count blocks. Estimated RAM 8–24 GiB; validate at the first-shard checkpoint. No dense fine-stratum accumulator.
- Core plus retained cells totals about 41.57 GiB for top200, or 62.35 GiB with 50% working margin at central density. At twice the estimated X-Atlas cell density, the same margin reaches about 101 GiB before source/container overhead; the 100 GiB scratch candidate is conditional on bounded output staging and a measured first-shard checkpoint, otherwise revise disk size/cost before launch. Full sources must not accumulate on scratch: temporary source cleanup requires an agreed lifecycle and independently validated durable checkpoints.
- Keep the conservative 4–12-hour runtime allowance until measured; reading/decompression and verification still dominate source acquisition. Prior pricing allowances (not a verified London quote): $0.50/VM-hour, $0.15/balanced-GiB-month, $0.005/address-hour, $0.03/durable-GiB-month. VM + disks + address are about **$0.532/hour**, or **$2.13–$6.38** over 4–12 hours. Confirm regional quote before creation; taxes, operations and transfer are additional.
- If stopped and retained, both proposed disks cost about **$19.50/month** under those allowances. Core durable storage is about **$0.25–$0.30/month**; retained cells add **$0.95/month**, for about **$1.20–$1.25/month** total at central density. Archives/copies add cost. Local egress remains separate (prior allowance $0.15/GiB). These inherit the dated Phase A planning rates, not a new price verification.
- Local free space was about 50 GiB at Phase A. Keep the compact core/audits locally and responder cell objects durably in the bucket; do not assume a full local copy fits with working margin. Recheck free space before the build. Local/cache statistics are dated, not a fresh whole-machine audit.
- Stage only under data/derived/context_transfer_splits/<release>/<context>/ and gs://bold-bastion-509200-f9-vcc2026/derived/context_transfer_splits/staging/<release>/<context>/. Stop compute on completion/safe failure. Independently verify scientific reconstruction and local/cloud hashes before promotion; no canonical overwrite or persistent source mirror. Agree source/scratch retention/deletion before launch.

## Acceptance and remaining decisions

Reconstruct all included existing construct/control rows exactly. Control identity×split and batch×role×split are overlapping margins of the same cells and must reconstruct the same totals. Broad shared scPerturb omits some context-specific constructs: K562 EGLN2/PTCD1/RBM4 and RPE1 C7orf26/FAM136A/ZBTB17 require explicitly partial-target references. Added X-Atlas genes lack an existing expression reference, so require an independent unsplit source aggregation plus metadata counts; retain exact comparisons for every original union-panel row.

Before launch: choose top100 versus top200 (recommend top200), finalize identity roles/missing-control handling, decide whether any batch-fold extension is needed, and approve acquisition/resources/spend/retention. No n≥4 batch filter is proposed. Establish per-perturbation signal/noise in raw total-UMI rate geometry first, then frozen-D calibration and construct agreement. Responder detection is not a current gate or deliverable.

Reproduce cached reassessment with `.pixi/envs/default/bin/python scripts/analysis/reassess_split_storage.py`; the full metadata/local-hash audit script also emits this current revision. The ranking and support-bound calculations have synthetic positive/negative checks; this is not a claim that future builders or scientific gates have passed. [Current audit](split-artifact-preflight-audit.json).

**Resource state for this revision:** no cloud commands, created resources, writes, source downloads, source deletion, commits or pushes. Project-wide resources/costs were not inspected; KOLF remains untouched. No new continuing cloud cost. Prior source evidence and cloud read-only checks remain dated 2026-09-24.
