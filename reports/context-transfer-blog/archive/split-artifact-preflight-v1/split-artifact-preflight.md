# Cell-split artifact preflight

**Phase A complete; Phase B not authorized. Accessed 2026-09-24.**

This report covers K562 essential–RPE1, HepG2–Jurkat, and HCT116–HEK293T. H1 remains the existing 126-target benchmark; no additional H1 splits were inspected. KOLF sources, artifacts, and resources were not inspected or changed. Existing pseudobulks and checkpoints remain immutable. Cross-study comparisons remain secondary and confounded by study/protocol.

## Findings that change the build decision

- The 415-target union is poorly suited to a broad construct-comparison analysis in the essential/Nadig libraries: it contains only 25 and 36 shared targets, respectively, and one multi-construct target per pair. Their full shared libraries provide 2,055 and 2,392 targets.
- Four labels can be assigned deterministically, but most occupied construct × batch strata cannot populate all four. Pooled counts conceal this. A common-batch-support rule and fixed batch weighting must be agreed before split-power interpretation.
- X-Atlas metadata was sufficient to measure coverage for all 109 + 223 shards. Full shared-construct selection would retain almost all 7.94 million source cells. Its storage cost is materially larger than the 415-target release.
- Preserving control-identity × batch × split expression is a substantial storage component. Subsampling responder controls does not eliminate that aggregation requirement.
- Suggested direction for discussion: broaden scPerturb; decide X-Atlas separately. No universe, sparse-stratum filter, control cap, or cloud resource has been approved.

## Evidence, provenance, and selection

Local manifests, audits, aggregation scripts, and all 14 retained H5AD hashes were checked. Source metadata counts independently reconstruct every existing construct cell count. scPerturb metadata UMI totals also match the audited totals. X-Atlas shard selected counts and recorded sample identities match all 332 manifests. Current Zenodo sizes/MD5s and pinned Hugging Face LFS sizes/SHA-256s match the prior provenance. Full source hashes were **not** recomputed from metadata reads.

For scPerturb, the existing builder selects **all author-released cells**, reads raw `X`, uses `obs/gene`, `obs/guide_id`, and `obs/batch`, and preserves `var/ensembl_id` and `var/gene_name`. The source is an author-processed matrix with scPerturb metadata harmonization, not an independent experiment. These four H5ADs are **dense, compressed float32 X**, contrary to the sparse description in the older dataset catalog. Counts were previously audited as integers; Phase B must repeat that check.

For X-Atlas, the existing builder selects `pass_guide_filter` AND (`gene_target` in the 415-target union OR `gene_target == "Non-Targeting"`). Counts are paired `gene_token_id` / `gene_expression` lists. `sample` is the recorded batch: metadata confirms exactly one sample per shard and agreement with the shard manifests. All metadata-read cells pass the guide filter. Shard batches are technical units; independent biological replication has not been established. The Figshare and Hugging Face representations are not independent evidence.

**Construct identity:** proposed `exact_construct_id = JSON([paired-library namespace, verbatim source label])`. The namespace is shared across the paired contexts, not keyed by file hash. All observed labels have exactly two nonempty pipe-separated components, and split/join round-trips exactly. This parses component **labels**, not verified sgRNA sequences. Components are not independent observations. Preserve P1, P2, and P1P2 literally; no promoter interpretation or target-alias remapping.

## Paired-library coverage

Counts below exclude controls unless stated. Shared constructs are matched by exact verbatim labels, with identical target mapping verified. “Union targets” is the paired overlap within the 415-target panel.

| Pair | All shared targets | All shared constructs | Union targets / constructs | Shared control identities | Common native IDs / literal symbols |
|---|---:|---:|---:|---:|---:|
| K562_essential ↔ RPE1 | 2,055 | 2,171 | 25 / 26 | 12 | 7,226 / 7,226 |
| HepG2 ↔ Jurkat | 2,392 | 2,548 | 36 / 37 | 130 | 7,632 / 7,632 |
| HCT116 ↔ HEK293T | 18,285 | 19,812 | 414 / 431 | 1024 | 38,606 / 38,584 |

The native axes contain 8,563 / 8,749 / 9,624 / 8,882 / 38,606 / 38,606 genes in context order below. X-Atlas has 21 duplicate-symbol groups (22 extra columns, including three MKKS IDs); the complete groups are in JSON. The scPerturb literal symbols are unique, but TBCE/HSPA14 have upstream Ensembl-suffixed disambiguations. Do not silently remove these suffixes. Freeze a pairwise gene axis and explicit mapping before D; do not force an unnecessary six-context intersection. Retain the native axis in every source artifact, and exclude the target from downstream evaluation geometry.

## Universe options and responder panels

`union415` retains every observed construct for panel targets in each context, plus all controls; paired comparisons use their exact shared subset. `all_shared` retains only constructs seen in both members of the library, plus **all context-specific controls**, not just shared controls. The responder panel comprises targets with at least two retained constructs. This is a metadata definition, not an activity filter.

| Universe | Context | Targets / constructs | Selected cells incl. controls | Multi-construct / multi-P-class targets | Responder perturbation cells |
|---|---|---:|---:|---:|---:|
| union415 | K562_essential | 25 / 26 | 14,781 | 1 / 1 | 341 |
| union415 | RPE1 | 36 / 37 | 18,990 | 1 / 1 | 244 |
| union415 | HepG2 | 36 / 37 | 7,880 | 1 / 1 | 129 |
| union415 | Jurkat | 36 / 37 | 18,455 | 1 / 1 | 233 |
| union415 | HCT116 | 414 / 431 | 231,677 | 17 / 16 | 5,389 |
| union415 | HEK293T | 414 / 431 | 307,163 | 17 / 16 | 6,989 |
| all_shared | K562_essential | 2,055 / 2,171 | 309,610 | 115 / 111 | 28,600 |
| all_shared | RPE1 | 2,055 / 2,171 | 209,264 | 115 / 111 | 23,576 |
| all_shared | HepG2 | 2,392 / 2,548 | 145,471 | 135 / 114 | 15,942 |
| all_shared | Jurkat | 2,392 / 2,548 | 262,950 | 135 / 114 | 31,763 |
| all_shared | HCT116 | 18,285 / 19,812 | 3,409,140 | 1491 / 1323 | 509,810 |
| all_shared | HEK293T | 18,285 / 19,812 | 4,533,980 | 1491 / 1323 | 673,828 |

The extra coverage of all_shared is especially important for construct calibration: 115 / 135 / 1,491 multi-construct targets per pair, compared with 1 / 1 / 17 under union415. Additional targets widen the assessment; metadata alone does not establish signal or successful responder inference.

## Four-split feasibility

These are exact source-metadata counts. A stratum with n ≥ 4 can populate four splits; n ≥ 8 gives at least two cells per split, and n ≥ 20 gives at least five. These are arithmetic conditions, **not power thresholds**. JSON includes these thresholds, full pooled quantiles, multi-construct strata, and batch/control coverage.

| Universe | Context | Pooled construct cells min / median / max | Occupied construct×batch strata | Strata with ≥4 cells | Perturbation cells in those strata |
|---|---|---:|---:|---:|---:|
| union415 | K562_essential | 41 / 126 / 573 | 1,121 | 453 (40.4%) | 67.0% |
| union415 | RPE1 | 22 / 98 / 3580 | 1,612 | 400 (24.8%) | 71.9% |
| union415 | HepG2 | 3 / 40 / 1213 | 1,048 | 118 (11.3%) | 52.4% |
| union415 | Jurkat | 3 / 92 / 2555 | 1,577 | 367 (23.3%) | 66.4% |
| union415 | HCT116 | 2 / 137 / 620 | 30,645 | 4,644 (15.2%) | 34.3% |
| union415 | HEK293T | 2 / 184 / 724 | 46,255 | 4,944 (10.7%) | 27.7% |
| all_shared | K562_essential | 5 / 116 / 1996 | 88,119 | 31,812 (36.1%) | 64.2% |
| all_shared | RPE1 | 2 / 68 / 3580 | 80,235 | 13,761 (17.2%) | 44.1% |
| all_shared | HepG2 | 1 / 43 / 1213 | 76,268 | 6,394 (8.4%) | 24.7% |
| all_shared | Jurkat | 1 / 80 / 2555 | 99,583 | 19,178 (19.3%) | 45.1% |
| all_shared | HCT116 | 1 / 141 / 1129 | 1,448,225 | 241,409 (16.7%) | 37.0% |
| all_shared | HEK293T | 1 / 188 / 1272 | 2,202,324 | 257,325 (11.7%) | 29.9% |

**Conclusion:** four nonempty splits at every source construct × batch are not viable. Preserve four labels, sparse strata, and all counts in the build. Do not silently use fewer splits or claim the pooled count divided by four is a within-batch guarantee. Restricting analysis to n ≥ 4 strata would retain only 24.7%–64.2% of perturbation cells in the all_shared universes. Even that offers only one cell per split at the boundary.

**Decision:** compare a common-support, batch-standardized estimand with an all-cell pooled estimand. The former needs the same eligible batches and fixed weights across splits; the latter can have different batch mixtures across splits and does not automatically satisfy the identical-effect assumption. Common support for a two-construct comparison must also be shared by both constructs. Metadata-only cell-count weights are one proposal; UMI-derived weights change the estimand and should not be chosen implicitly. Batch folds are not biological replicates.

## Controls and assignment contract

| Context | Recorded batches | Control identities | Control cells | Total controls per batch min–max |
|---|---:|---:|---:|---:|
| K562_essential | 48 | 97 | 10,691 | 116–281 |
| RPE1 | 56 | 113 | 11,485 | 119–260 |
| HepG2 | 56 | 130 | 4,976 | 5–123 |
| Jurkat | 55 | 130 | 12,013 | 39–290 |
| HCT116 | 109 | 1024 | 165,777 | 868–1826 |
| HEK293T | 223 | 1025 | 218,838 | 12–1603 |

- Preserve every selected construct × batch × split **count** and control identity × batch × split count in assignments/manifest, including explicit zero support. Counts alone cannot reconstruct batch expression: the analyzed panel needs batch-resolved expression sums too.
- Proposed control roles are disjoint exact identities: approximately 25% for D training, 25% for null evaluation, and 50% for effect baselines, using the same deterministic role for a label across contexts. Lists must be frozen after coverage review. Null identities never subtract themselves; D is fitted without null-evaluation identities. The illustrative allocation leaves one HepG2 batch with only 3 baseline-role controls, so four baseline splits are impossible there; HEK293T has a batch with only 6 baseline-role controls. These batches remain explicit.
- Compute batch-matched effects with identical perturbation/control batch weights. Preserve atomic control sums so shared-baseline and disjoint-baseline calculations can be compared. Shared control **cells** create measurement covariance; matching a construct label across different contexts does not mean the cells are shared.
- Canonical locator: source SHA-256 + file/shard + zero-based source row index. Keep the source cell barcode/index when available. Assignment rows contain selected cells only; manifest exclusions are summarized by reason.
- Proposed assignment: seed 0, SHA-256 order within context × batch × exact construct, tie-break by locator, then round-robin into four splits. A deterministic hash-derived starting rotation prevents every singleton stratum landing in split 0. This is balanced within strata, not simple hash modulo four on cells. Retain empty-bin counts.
- Assign whole batches to four proposed folds with a separate hash domain and balanced hash rank; never derive batch fold from cell split. Freeze fold count and adequate training/evaluation support before responder fitting.
- Preserve native-axis raw total UMI and cell count. `core_scale_factor` genuinely exists in K562/RPE1 (ranges 0.600–1.233 / 0.493–2.986); preserve it verbatim, but its validity/direction as an exposure is unresolved. No corresponding field appears in the inspected Nadig/X-Atlas schemas. Do not relabel UMI_count, ncounts, or total_counts as size factors.
- Logical hashes cover canonical row keys, gene order, metadata and canonical integer CSR contents, excluding timestamps/compression. File SHA-256 separately protects transfer integrity. Native gene counts and count sums must be exact; no normalization, log transform, HVG selection, batch correction, or unapproved cell filtering.

## Acquisition

The four scPerturb H5ADs are absent from repository data/external by filename search and total 4,927,873,119 bytes (4.59 GiB). No other worker disk was inspected. Their URLs, MD5s, and SHA-256s below are pinned to the prior builds; Zenodo metadata rechecks size and MD5. A full expression acquisition is still required after approval.

- **K562_essential:** [ReplogleWeissman2022_K562_essential.h5ad](https://zenodo.org/api/records/13350497/files/ReplogleWeissman2022_K562_essential.h5ad/content), 1,546,729,675 bytes.
  MD5 `d8cba17576d1a8afc0f7d71b79cad0f7`; SHA-256 `412fd0df8c4ccea9f4db91cd88033c49200838b29d40945e48574be588b48789`.
- **RPE1:** [ReplogleWeissman2022_rpe1.h5ad](https://zenodo.org/api/records/13350497/files/ReplogleWeissman2022_rpe1.h5ad/content), 1,236,886,900 bytes.
  MD5 `cc7f1ec50aeb3a3e1b4a6cfa713d80fa`; SHA-256 `12d6a0cf9378c4f09411e27f0ce07b59f1d97731bd576d6fadf112f6f088d1a4`.
- **HepG2:** [NadigOConner2024_hepg2.h5ad](https://zenodo.org/api/records/13350497/files/NadigOConner2024_hepg2.h5ad/content), 850,590,740 bytes.
  MD5 `af2be47f7477cf32fa6e4bec1c6a4868`; SHA-256 `1af2f7b3e692ad3d077e6027d68a1f800619e29aa3f7a0efa70146fb7223a4bf`.
- **Jurkat:** [NadigOConner2024_jurkat.h5ad](https://zenodo.org/api/records/13350497/files/NadigOConner2024_jurkat.h5ad/content), 1,293,665,804 bytes.
  MD5 `d8b05d00bfbd686d37ffdd4293bc6c8c`; SHA-256 `ade3d83150fd6944212363cd11da77c0d2143fa6df1febb3f15a66d4dab0677f`.

X-Atlas uses [Hugging Face commit 53a5bc98d49247bcf967500292575c3d3602de31](https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion/tree/53a5bc98d49247bcf967500292575c3d3602de31). The 109 HCT116 shards total 46,576,484,789 bytes; the 223 HEK293T shards total 79,683,336,248 bytes: **117.59 GiB** combined. Largest source shard: 717,945,523 bytes. Exact shard URLs and hashes are in JSON. Full SHA-verified shard acquisition would still scan that release for either universe; retaining fewer targets reduces outputs, not that download volume.

The successful metadata pass transferred 58,245,557 / 95,565,880 bytes for the two X-Atlas contexts and 15,425,631 bytes for the final four scPerturb inspections. Including initial scPerturb inspections and two trial X-Atlas reads, source-range traffic was 179,502,975 bytes (171.19 MiB); manifests/API metadata are additional small reads. Each source range reader enforced HTTP 206, exact Content-Range and a per-source cap (8 MiB H5AD, 2 MiB Parquet). No expression array was requested or decoded. These reads cannot substitute for full source checksum validation.

X-Atlas checkpoint archives retain per-shard construct counts/sums and can provide aggregate coverage, but **cannot reconstruct within-batch cell splits or responder cells**. We used pinned-source metadata directly and left checkpoints unchanged.

## Output size and responder-control tradeoff

All sizes below are **uncompressed CSR planning sizes**, not promises about gzip file sizes. Counts use int64, column indices int32, and row pointers int64. scPerturb responder sizes use exact selected upstream ngenes sums (to verify against X at build); X-Atlas uses the full-source mean nnz from Parquet footers. Selected X-Atlas density is unknown: plan for 0.5×–2× the estimates. JSON also gives dense CSR ceilings, atomic row counts and assignment estimates (160 bytes/selected cell before compression).

| X-Atlas universe | Context | Responder perturbation cells | All eligible controls | Cap-128 controls | All-control cell artifact GiB | Cap-128 cell artifact GiB |
|---|---|---:|---:|---:|---:|---:|
| union415 | HCT116 | 5,389 | 165,777 | 13,952 | 9.77 | 1.10 |
| union415 | HEK293T | 6,989 | 218,826 | 28,416 | 16.22 | 2.54 |
| all_shared | HCT116 | 509,810 | 165,777 | 13,952 | 38.57 | 29.90 |
| all_shared | HEK293T | 673,828 | 218,838 | 28,428 | 64.11 | 50.44 |

All controls maximize tail/null calibration and preserve the released control population. A deterministic cap of 128 per eligible batch reduces cell storage but sacrifices rare identity/state coverage and precision. Allocate across frozen roles and identities, record inclusion weights, and calibrate held-out nulls using the same sampling design. A cap cannot retain every control identity within every batch. **128 is an illustrative comparison, not a chosen cap.** All eligible controls still enter full identity-level pseudobulks; responder subsampling must not change exact control reconstruction.

| Build profile | Pseudobulks GiB | Responder GiB, all / capped controls | Assignments GiB | Total GiB, all / capped controls |
|---|---:|---:|---:|---:|
| union415_all_batch | 43.0 | 27.6 / 4.7 | 0.09 | 70.7 / 47.8 |
| broad_scperturb_union415_xatlas_all_batch | 82.0 | 31.7 / 8.8 | 0.22 | 113.9 / 91.0 |
| all_shared_multiconstruct_batch | 184.3 | 108.4 / 85.5 | 1.32 | 294.0 / 271.1 |
| all_shared_all_batch | 633.8 | 108.4 / 85.5 | 1.32 | 743.6 / 720.6 |

Pseudobulk estimates bound nnz by input nonzeros and row×gene dimensions; they do not assume summing cells leaves matrices sparse. The all-shared/all-batch profile is large because very sparse construct×batch strata preserve much of the cell-level support. Selecting only multi-construct batch expression is cheaper but limits batch-matched analysis to that panel. The six contexts need not use the same universe; a hybrid is an explicit additional option.

## Proposed cloud envelope—not authorization

- Boundary verified: project `bold-bastion-509200-f9`, active configuration `vcc-2026`; proposed zone `europe-west2-b`. Use the existing `vcc-worker@bold-bastion-509200-f9.iam.gserviceaccount.com` and existing network. No IAM, quota, billing, or network changes.
- Candidate: **e2-highmem-8, 8 vCPU / 64 GiB**, with 30 GiB pd-balanced boot. Process contexts sequentially and one expression shard at a time. Use disk-backed accumulators/chunked writes, not a global dense construct×batch matrix. Expected peak RAM 16–40 GiB; proposed stop threshold 48 GiB. These are engineering estimates; Phase B synthetic and first-shard measurements must validate them.
- Scratch, pd-balanced: **300 GiB** for union415 or broad-scPerturb/union-X-Atlas; **1,000 GiB** for all-shared with multi-construct batch expression; **2,000 GiB** for all-shared/all-batch. The broad options require quota/headroom verification and may exceed available quota after existing resources; no quota change is authorized. Keep source shards only through agreed validation checkpoints, and avoid a complete source mirror.
- Runtime planning: **4–12 hours** restricted/hybrid; **12–36 hours** broad. Prior X-Atlas aggregation scanned sources in about 41 minutes, but did not write these much larger artifacts; it is not a timing guarantee.
- Conservative USD planning allowances: compute **$0.50/hour**, balanced disk **$0.15/GiB-month**, ephemeral address **$0.005/hour**, durable storage **$0.03/GiB-month**. Thus VM+disks+address are approximately **$0.573 / $0.717 / $0.922 per hour** for the three scratch sizes. Restricted/hybrid running cost is about **$2.29–$6.88**; broad running cost about **$8.60–$33.18**, before transfer, operations, tax and retained storage. These are planning allowances, not a verified London SKU quote; confirm the region-specific quote before creation. [Compute pricing](https://cloud.google.com/products/compute/pricing/general-purpose), [disk pricing](https://cloud.google.com/compute/disks-image-pricing), [storage pricing](https://cloud.google.com/storage/pricing).
- Retained scratch+boot costs approximately **$49.50 / $154.50 / $304.50 per month** under those allowances even with the VM stopped. Durable storage is `0.03 × retained GiB/month`: about $1.4–$3.5/month for restricted/hybrid central estimates and $8–$22/month for broad central estimates. Archives/additional copies multiply storage. Budget local egress separately at up to $0.15/GiB for planning; transferring 100 GiB would add roughly $15. Confirm actual routing/rate before transfer.
- Local free disk was about **50 GiB**. Retain compact audits/metadata locally; full raw split artifacts may not fit. For large outputs, independently stream cloud objects through SHA-256 locally rather than retain the entire output download. Compare against independently verified on-worker file hashes and scientific audits; record exactly which files were locally retained.
- Proposed local staging: `data/derived/context_transfer_splits/<release>/<context>/`; cloud staging: `gs://bold-bastion-509200-f9-vcc2026/derived/context_transfer_splits/staging/<release>/<context>/`. A release name includes approved-config hash. No staging objects have been created.
- Stop compute on completion or safe failure. Retain review-required source/scratch until agreed lifecycle permits deletion and independent validation passes. Promote only after reconstruction and local/cloud hashes agree. Archive an existing canonical artifact before any future replacement; existing compact pseudobulks/checkpoints remain immutable.

## Acceptance and unresolved decisions

1. Choose universe per pair and batch-expression scope. Broad scPerturb is scientifically better suited to the proposed construct comparisons; broad X-Atlas is a separate storage decision.
2. Approve four split labels and choose a common-support analysis rule, minimum cell counts, fixed batch weighting, and responder fold support. Do not treat n≥4 as adequate power.
3. Freeze control roles and missing-control handling; choose responder panel and all-control versus capped retention.
4. Approve exact source acquisition, resource envelope, spend stop, retention/deletion lifecycle and regional quote. No downloads of full source files or cloud mutations before approval.
5. Freeze gene mapping/axis and D only after artifact review; begin with raw total-UMI rate differences. Target repression remains activity QC. Analysis gates remain binding.

Phase B must reconstruct every included audited construct and control exactly, then reconstruct targets under the identical selection universe. For all_shared, K562 targets EGLN2/PTCD1/RBM4 and RPE1 targets C7orf26/FAM136A/ZBTB17 omit a context-specific construct: their correct reference is the exact sum of included immutable construct rows, not the original full-target row. Label this partial-target coverage; compare original full-target rows wherever all source constructs are retained. This is an explicit selection distinction, not relaxed numerical tolerance.

Acceptance also requires nonnegative integer CSR/no stored zeros, exact source-label round-trip, independent control-role disjointness, assignment uniqueness/balance, batch-matched reconstruction, logical reproducibility and independent file hashes. Synthetic controls must expose shared-control covariance and pooled-control batch bias; failure is a result, not a reason to weaken assertions. Phase A validates occupancy and provenance only; it does not claim these future builder/scientific acceptance tests passed.

**Current resource state:** no resources created and no cloud objects written. Queries restricted to proposed `vcc-split*` names returned zero instances and disks. KOLF and other pre-existing resource inventory/costs were deliberately not audited. No new continuing cloud cost; small metadata reads may incur ordinary operations/egress charges. No source deletion, commit, or push.

## Reproduction

`scripts/analysis/preflight_split_metadata.py` performs bounded metadata reads into gitignored `data/derived/context_transfer_preflight/metadata/`; `scripts/analysis/summarize_split_preflight.py` verifies local hashes and metadata reconstruction and writes the JSON calculations. The 332 downloaded cloud JSON manifests total 367,724 bytes and are immutable local evidence. Per-source metadata/cache hashes, exact source URLs, file hashes, control identities, batch lists, duplicate groups, target lists, quantiles and row/storage estimates are in [split-artifact-preflight.json](split-artifact-preflight.json).

Repository base commit: `9f9063cdaeba8aad5f4712610d6125c9c5b2e8c8`; pre-existing worktree changes were present and left intact. scPerturb schema/provenance interpretation also checked against the [pinned Replogle harmonization script](https://github.com/sanderlab/scPerturb/blob/b69f72a070a92bcbaf41e7f9897b11598109ab48/dataset_processing/scripts/ReplogleWeissman2022.py). Nadig upstream filtering/size-factor semantics have not been inferred from the Replogle script.
