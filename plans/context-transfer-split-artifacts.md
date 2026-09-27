# Split-power artifact proposal

**Status:** cloud build and compact local acquisition completed and independently validated 2026-09-24.
Completion and cost record: `reports/context-transfer-blog/split-v1-20260924/completion/README.md`.
Frozen launch record in
`reports/context-transfer-blog/split-v1-20260924/README.md`  
**Revised:** 2026-09-24 — pooled split artifacts replace fine-batch expression

Launch uses top200, the proposed control-role hash, unavailable flags for missing controls,
no whole-batch folds, and durable responder-ready cell retention. The launch record supersedes
the preliminary machine/disk costs and pending decisions below.  
**Analysis:** `plans/context-transfer-split-power.md`  
**Evidence and storage:** `reports/context-transfer-blog/split-artifact-preflight.{md,json}`

## Purpose and scope

Build compact raw-count artifacts for cell-split reliability and paired-context transfer.
Use source batch labels during assignment and control matching without storing a gene vector
for every construct × batch × split. These strata often contain one cell and would approach
single-cell storage. Four cells in every source batch is **not** a requirement.

The working universe is broad exact shared constructs for K562 essential–RPE1 and
HepG2–Jurkat. For HCT116–HEK293T, preserve the 415-target H1–VCC union view and compare
adding 100 or 200 well-sampled multi-construct genes. This augmentation is pseudobulk-only;
responder-ready cells are retained separately for the original multi-construct panels. H1 stays unchanged. Cross-study
comparisons remain secondary and confounded by study/protocol. KOLF sources, workers,
disks and artifacts are out of scope. Existing compact pseudobulks/checkpoints are immutable.

### X-Atlas ranking

For each exact construct shared by both contexts, take the smaller of its HCT116 and
HEK293T pooled cell counts. For each target with at least two shared constructs, use the
**second-largest** such count as the score. Thus the score guarantees that two constructs
have at least that many cells in each context. Tie-break by total cells across all shared
constructs in both contexts, then target label. Forrest selected this criterion on 2026-09-24.

Compare top 100 and top 200 genes **in addition to** the union panel, counting overlap only
once. Retain all shared constructs for selected genes, including weaker additional constructs;
the score is not a minimum-count guarantee for every retained construct. Selection uses
cell-count metadata only, not measured effects or target repression. It enriches well-sampled
surviving perturbations and supports a conditional analysis, not a representative genome-wide
claim. Proposed lists and the total-count-ranking comparison are retained in the preflight.

## Core release per context

Write versioned staging outputs under
`data/derived/context_transfer_splits/<release>/<context>/`.

| Artifact | Stored content | Purpose |
|---|---|---|
| `cell_assignments.parquet` | Selected-cell locator, context, batch, target, exact construct, control role, split, raw UMI, source identifiers | Reproducibility and batch composition; no expression vector |
| `construct_split_pseudobulk.h5ad` | Exact construct × four cell splits; includes control identity × split rows | Reliability, construct comparison, transfer, identity-aware nulls |
| `batch_role_split_controls.h5ad` | Source batch × frozen control role × split | Batch-matched control baselines and control diagnostics |
| `batch_split_exposures.parquet` | Construct/control identity × batch × split cell counts and native-axis UMI totals; explicit zero support | Recover weights and support without fine-stratum expression |
| `manifest.json`, compact audit | Source identity, schema, selections, logical and file hashes, reconstruction | Independent acceptance |

Control roles partition exact identities into D-training, null-evaluation and effect-baseline
pools. Their proportions and lists remain subject to coverage review; the preflight illustrates
25%/25%/50%. The same exact control label receives the same role across contexts. No null
identity contributes to its own baseline, and D-training identities are disjoint from null
and baseline identities. The two control expression views overlap intentionally: they represent
the same cells at different margins and must never be added as if they were distinct cells.

The release stores raw nonnegative integer counts on the complete **native** gene axis.
Each expression row retains `n_cells` and `total_umis`; no normalized source artifacts.
Duplicate symbols are summed only later in raw count space under a frozen mapping.
Preserve `core_scale_factor` verbatim where it genuinely exists in Replogle; do not treat it
as a validated capture exposure until its semantics are established. Do not invent size factors.

## Assignment and identity

- Canonical cell locator: source SHA-256 + original file/shard + zero-based source row index.
  Keep source barcode/obs index where available; do not assume barcodes are globally unique.
- `exact_construct_id`: canonical JSON `[paired-library namespace, verbatim source label]`.
  The paired contexts share the namespace. Retain the original label and encoded P-classes.
  Pipe-separated components may be parsed losslessly but are not independent observations;
  labels do not by themselves establish sequences or promoter biology.
- Assign cells by deterministic SHA-256 order within context × batch × exact construct,
  seed 0, with locator tie-breaks, then round-robin into `{0,1,2,3}`. A domain-separated
  hash supplies the starting rotation for each stratum. This avoids sending every singleton
  to split zero. Do not use per-cell hash modulo four or filter sparse batches.
- Pool expression across batches after assignment. Within-stratum counts differ by at most
  one; pooled split totals need not be perfectly equal. Report realized counts/exposures and
  permit recombination into halves. Empty pooled splits make that comparison unavailable.
- Assignment rows contain selected cells only. Summarize exclusions by reason in the manifest.
  The exposure table records the missing/zero support needed for matching.

## Batch matching and interpretation

For construct p and cell split r, retain pooled counts Y_pr and UMI exposure E_pr. Batch
exposure E_pbr gives weights w_pbr = E_pbr / E_pr. A matched baseline can be computed from
batch × baseline-role × split control rates using those weights. This matches the baseline
to the observed perturbation batch mixture without retaining perturbation expression by batch.

This does **not** allow arbitrary later perturbation-batch reweighting. If a required batch
has no eligible baseline cells in that split, one cannot simply drop its contribution from the
already pooled perturbation vector. Flag the matched calculation as unavailable, or approve
an explicitly prespecified supported-population aggregate before acquisition. Never silently
substitute pooled controls, borrow the same cells across independent splits, or filter batches.

Cell splits measure reliability conditional on the recorded experiment and its batch mixture.
A systematic effect shared by both halves can reproduce. Differences in split batch mixtures
also matter when perturbation responses vary by batch. Record their composition and use
matched-control/null diagnostics; do not claim biological replication or automatic removal of
batch effects. The raw off-diagonal product is shared split signal; interpreting it as power of
one common effect requires the common-effect assumption to be justified.

## Optional extensions: separate decisions and costs

Responder **analysis** remains deferred, but Forrest requested retention of the raw cell objects
on 2026-09-24 for likely use within two weeks. Retain all shared constructs for the 115/135
multi-construct scPerturb targets and the original 17 multi-construct X-Atlas targets, plus
**all eligible controls from their batches**. Added X-Atlas genes remain pseudobulk-only.
Store raw integer CSR counts on native gene axes, cell locators, source metadata, batches,
constructs, roles and split assignments in separate responder-ready objects; no inferred labels
or responsibilities. Independently verify selected-cell coverage and reconstructed aggregates.
The extra CSR estimate is **31.71 GiB**, about **$0.95/month** at the existing planning allowance;
X-Atlas density has 0.5x–2x uncertainty. With top200 the combined central estimate is 41.57 GiB.
Retain derived objects durably with **no automatic expiry**; two weeks is the intended research
horizon, not a deletion deadline. Later deletion requires approval. Prefer cloud retention and
local compact core/audits. See `plans/context-transfer-responders-deferred.md` for future analysis.

1. **Whole-batch validation:** predefine a small number of independent batch folds and retain
   construct × fold × cell-split sums. Entire batches move between folds; fold hashes are
   independent of cell splits. This tests recorded-batch generalization, not necessarily
   biological replication. It is not a prerequisite for the initial cell-split analysis.
2. **Arbitrary batch bootstrap/reweighting:** requires finer expression or single-cell retention.
   The compact core does not support it; retained cells permit it only within their smaller panel.

Do not retain construct × batch × split or identity × batch × split expression by default.
Counts/exposures at that resolution remain inexpensive metadata.

## Acceptance and release

1. Verify full source SHA-256 after approved acquisition; confirm raw integer nonnegative CSR,
   no explicit zeros, native gene order, and exact label round-trip.
2. Verify locator uniqueness, disjoint cell splits, deterministic within-stratum balance,
   realized pooled support, control-role identity disjointness and assignment/exclusion counts.
3. Sum construct splits to reproduce immutable reference construct rows exactly; sum controls
   across identities and separately across batch-role margins to the same audited totals.
   Recover all `n_cells`, UMI totals and available upstream metadata exactly.
4. Compare target sums under the **identical construct universe**. In broad shared scPerturb,
   partially represented targets use sums of included immutable construct rows; compare full
   target rows wherever all original constructs are included. Label partial coverage explicitly.
5. Added X-Atlas genes have no existing union-panel expression reference. Require an independent
   unsplit source aggregation and source-metadata count check for them, and exact reconstruction
   of every original union-panel construct/control. Do not pretend prior audits cover new genes.
6. Synthetic positive/negative controls verify pooled split reconstruction, matched-baseline
   weights, unavailable-control handling, shared-control covariance and pooled-control batch
   bias. Do not interpret these tests as proof that biological batch effects are absent.
7. Canonical logical hashes cover row keys, native genes, metadata and sorted integer CSR
   contents independent of timestamps/compression. File SHA-256 separately verifies transfer.
8. Validate memory/disk use against the approved envelope; stage outputs, independently verify
   scientific reconstruction and local/cloud hashes, then review before any promotion/analysis.

## Execution boundary

The preflight contains revised size estimates and proposed machine/disks, not authorization.
A metadata pass can precede a single expression pass; source files/shards are acquired only
following approval. Use the existing project/configuration/region/service account in AGENTS.md.
No cloud writes, resource creation, full source downloads, source deletion, commit or push is
included in this proposal revision. Agree source/scratch retention and deletion lifecycle before
launch; stop compute when finished or safely failed. Existing canonical artifacts stay intact.
