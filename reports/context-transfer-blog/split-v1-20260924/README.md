# Approved split release — 2026-09-24

**Completed:** all six contexts passed cloud validation on 2026-09-24; worker stopped.
See [completion and costs](completion/README.md) for the final cloud result and local-copy status.

Forrest approved the cloud build after requesting durable responder-ready cell retention.
The frozen release config uses broad paired scPerturb constructs, the original X-Atlas union
plus the top200 weaker-construct panel, and the original 115/135/17 multi-construct responder
panels with all eligible controls. Added X-Atlas genes are pseudobulk-only.

The proposed control-role hash is frozen (approximately 25/25/50 training/null/baseline).
Missing matched controls make that comparison unavailable; no fallback pooling, sparse-batch
filter or optional whole-batch fold is added. Responder modeling and split-power analysis await
artifact review. This launch resolves those choices from the Phase A proposal.

## Resource and retention contract

- Project `bold-bastion-509200-f9`, configuration `vcc-2026`, zone `europe-west2-b`.
- Worker `vcc-splits-20260924`: `e2-highmem-8`, 8 vCPU / 64 GiB.
- New 30 GiB balanced boot and 250 GiB standard scratch disks, retained for review.
- Larger scratch replaces the preliminary 100 GiB candidate: retain all acquired sources
  for independent rereading and review, without assuming compression or deleting source shards.
  The first balanced-scratch attempt exceeded SSD quota and left no resources; standard
  scratch avoids quota changes. Disk I/O may take longer than the original runtime estimate.
- Existing worker service account and default network. No IAM/firewall/bucket policy changes.
- Shutdown on completion or failure, plus Compute Engine STOP after 16 hours; automatic
  restart disabled. No automatic disk deletion. A failed run remains recoverable from disk.
- Sources remain only on this worker disk, not as a permanent bucket mirror. Durable derived
  cell objects have no automatic two-week expiry. Later deletion requires user approval.
- London Billing Catalog prices: E2 CPU $0.0281037/vCPU-hour, RAM $0.00376602/GiB-hour,
  balanced disks $0.12/GiB-month, standard disks $0.048/GiB-month. VM $0.46585488/hour; VM/disks/IP allowance ~$0.492/hour,
  ~$1.97–$5.91 for 4–12 hours; 16-hour limit ~$7.88, excluding operations, transfers and tax.
  Retained disks $15.60/month. Derived storage retains the conservative $0.03/GiB-month
  allowance (~$1.25/month at the 41.57 GiB uncompressed central estimate).
  Exact retrieved SKU evidence is in `pricing-skus.json`; this is not a billing cap.

## Durable locations and validation

Staging prefix:
`gs://bold-bastion-509200-f9-vcc2026/derived/context_transfer_splits/staging/split-v1-20260924/`

- `launch/bundle.tar.gz`: frozen tested scripts, config, metadata coverage and immutable
  reference copies. Bundle and config SHA-256s are recorded in `launch.json`.
- `status.json`: context progress and verified transport-audit hashes.
- `<context>/`: split/control H5ADs, assignment/exposure/support tables, source-sharded
  `responder_cells/`, manifest, independent acceptance, and transport audit.
- `worker.log`: uploaded at shutdown. Serial console supplies progress during the run.

Every source is checked against its pinned SHA-256 (and scPerturb MD5); each context's
independent auditor rereads all source counts, reconstructs split sums, compares all applicable
immutable reference constructs and verifies every retained raw-cell row. Metadata coverage
must match the Phase A scan exactly. Each uploaded artifact is then streamed through SHA-256
and compared to the worker copy. No canonical promotion occurs in this job.

Local checks: five synthetic tests passed for both formats, malformed counts, corrupted
artifacts, deterministic balanced assignment, sparse strata, matched-baseline bias and
shared-control covariance; Ruff and shell syntax checks passed. The worker reruns these tests.
Real-source acceptance remains pending until `acceptance.json` and transport audits pass.

KOLF resources and artifacts remain out of scope. No commits or pushes.

## Launch checkpoint

Worker is RUNNING; cloud synthetic tests passed and HepG2 source acquisition started at
2026-09-24 09:36 UTC. Real-source acceptance and output hashes remain pending. This
inventory covers this job only, not unrelated project resources. `status-at-launch.json`
is a snapshot; the bucket status is authoritative for subsequent progress.

Verified launch bundle SHA-256:
`14b68b2ac17f9782e9f5d581cd4b70937d1dae2171cf24b713358178e6d80e36`.

Release config SHA-256:
`dba62d325198670fb85ed8ac83fb712644b7c9dc999ea501b8f82c0d21560cd6`.
