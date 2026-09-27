# Context-atlas cloud aggregation and compact local copy

Verified 2026-09-21.

## Completed cloud artifacts

Final pseudobulks for CD4T, HepG2, Jurkat, K562 essential, KOLF2.1J, RPE1,
HCT116, and HEK293T are stored under:

- `gs://bold-bastion-509200-f9-vcc2026/derived/context_atlas/final/`
- `gs://bold-bastion-509200-f9-vcc2026/derived/xatlas_orion/final/`

All eight cloud audits report `status: passed`. Together they represent
3,362,998 selected cells and 45,354,845,317 raw UMIs. The target, guide, and
stratified matrices occupy 4.51 GiB for the context atlas and 262.33 MiB for
the X-Atlas finals. Restartable checkpoints, the archived KOLF artifact, and
the staged recovery bring the complete bucket to 10.90 GiB.

## KOLF H1 recovery

The corrected KOLF artifact was rebuilt from the verified 189,393,177,972-byte
author H5AD using an immutable 415-target H1–VCC union panel. It contains 394
available perturbations plus `NTC`: 123/126 H1 targets and 282/300 VCC 2026
targets. The unavailable H1 targets are `CAST`, `CHMP3`, and `TAZ`.

Independent acceptance checks passed for source-metadata cell counts, raw
integer counts, target reconstruction from guides and target-by-batch rows,
control reconstruction from control-channel rows, and all manifest hashes.
Every row in the former VCC-only artifact was reproduced exactly. A downstream
smoke test constructed native-normalized KOLF and H1 effects for all 123 shared
targets over 17,603 shared response genes without fitting a model.

The corrected compact artifact is canonical under
`data/derived/context_atlas/final/KOLF2.1J/`; its explicit H1 view is under
`data/derived/context_atlas/views/KOLF2.1J_H1/`. The former VCC-only artifact is
retained under `data/derived/context_atlas/archive/KOLF2.1J-vcc300-cd3d00f1/`.
Cloud equivalents are stored under `final/`, `views/`, and the versioned
`archive/` prefix, respectively. The acceptance record is
`reports/context-atlas-cloud/kolf_recovery_acceptance.json`.

## Compact local copy

The 35 canonical locally retained files occupy 1,059,723,902 bytes (0.987 GiB) under:

- `data/derived/context_atlas/final/`
- `data/derived/xatlas_orion/final/`

They comprise 19 target-, guide-, and control-level H5ADs plus the eight
source manifests and eight source audit records. Large target-by-batch and
target-by-donor-condition H5ADs remain in Cloud Storage.

The local compact audit verified:

- every retained file's exact byte size and SHA-256 against its cloud
  manifest;
- CSR, raw integer, non-negative count representation;
- recorded row UMI totals and common gene axes;
- exact selected-cell and UMI totals; and
- exact reconstruction of every target row by pooling its guide rows.

Separate control reconstruction also passed for KOLF2.1J, HCT116, and
HEK293T, the contexts with retained channel- or batch-control matrices. The
other five collections include their global control rows in the target and
guide files but did not produce a separate control-stratum artifact. A
truncated-H5AD negative control was correctly rejected.

Re-run with:

```bash
pixi run python scripts/cloud/audit_compact_pseudobulks.py \
  data/derived/context_atlas/final/* \
  data/derived/xatlas_orion/final/*
```

## Resource cleanup and storage cost

Before cleanup, two terminated instances retained four zonal persistent disks
in `europe-west2-b`:

| Disk class | Provisioned GiB | USD/GiB-hour | Cost/hour | Cost/730-hour month |
|---|---:|---:|---:|---:|
| Standard persistent disk | 380 | 0.000054795 | $0.02082 | $15.20 |
| Balanced persistent disk | 180 | 0.000136986 | $0.02466 | $18.00 |
| **Total** | **560** |  | **$0.04548** | **$33.20** |

That would have been approximately $1.09 per day, prorated by the second. The 10.57 GiB
regional Standard Cloud Storage bucket costs approximately $0.21 per month at
the current $0.000027397/GiB-hour list rate, excluding operations and network
transfer. Prices are Google Cloud USD list prices; billing-currency conversion
and tax can differ.

After the compact set passed its independent local audit, `vcc-cd4-worker` and
`vcc-kolf-worker` were deleted with all four attached disks on 2026-09-20.
A post-deletion inventory showed zero VM instances, zero persistent disks, and
zero reserved addresses. The $33.20/month disk charge has therefore stopped;
only the approximately $0.21/month Cloud Storage cost continues.

The KOLF recovery added one stopped `e2-highmem-8` worker with a 30 GiB
balanced boot disk and a 380 GiB standard source disk. Both disks are retained
pending review; compute charges stop with the VM. At current list rates their
storage is approximately $18.20/month ($0.60/day), of which the source disk is
$15.20/month ($0.50/day). The complete 10.90 GiB Cloud Storage bucket is about
$0.22/month, excluding operations and network transfer.
