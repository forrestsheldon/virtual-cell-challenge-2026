# Google Cloud environment

Verified 2026-09-20.

- Project: `bold-bastion-509200-f9`
- CLI configuration: `vcc-2026`
- Primary region: `europe-west2` (London)
- Durable derived-data bucket: `gs://bold-bastion-509200-f9-vcc2026`
- Enabled APIs: Compute Engine and Cloud Storage

## Free Trial quota audit

The effective project-wide CPU ceiling is 12 vCPUs. In `europe-west2`, the
project currently has 2,048 GB of persistent-disk quota, 24 instances, and
zero used resources. The Free Trial has zero preemptible/Spot CPU quota and
zero project-wide GPU quota; quota increases are unavailable until the billing
account is activated.

The following candidate machine types are available in all three London
zones:

| Machine type | vCPU | RAM |
|---|---:|---:|
| `e2-highmem-8` | 8 | 64 GB |
| `n2-highmem-8` | 8 | 64 GB |
| `n2-standard-8` | 8 | 32 GB |

The selected first worker was `e2-highmem-8`. It was an on-demand CPU VM
because Spot was unavailable during the trial.

## Worker identity and network

Created 2026-09-20:

- Service account:
  `vcc-worker@bold-bastion-509200-f9.iam.gserviceaccount.com`
- The service account has `roles/storage.objectAdmin` on only
  `gs://bold-bastion-509200-f9-vcc2026`; it has no project-wide data role and
  no downloaded key.
- Custom VPC: `vcc-net`
- London subnet: `vcc-london`, `10.20.0.0/24`, with Private Google Access
- Inbound firewall rule: `vcc-allow-iap-ssh`
- The sole allowed inbound path is TCP 22 from Google's IAP range
  `35.235.240.0/20`, and only for instances tagged `iap-ssh`.
- Project-wide OS Login is enabled. The signed-in project owner has explicit
  IAP tunnel and OS administrator roles, so no persistent SSH key needs to be
  distributed manually.


## First aggregation worker

`vcc-xatlas-worker` ran in `europe-west2-b` from
2026-09-20T01:11:08Z until deletion was initiated at
2026-09-20T01:52:02Z. It used:

- `e2-highmem-8`: 8 vCPUs and 64 GB RAM;
- a 30 GB balanced boot disk;
- a 500 GB standard temporary scratch disk mounted at `/mnt/vcc`;
- the bucket-scoped worker identity above; and
- an ephemeral external address for outbound downloads, with inbound SSH
  restricted to IAP.

The worker scanned all 332 X-Atlas Parquet shards and wrote audited
pseudobulks and restartable checkpoints to Cloud Storage. It was then deleted;
both attached disks and the ephemeral address were deleted with it. A final
resource listing showed zero VM instances, zero disks, and zero reserved
addresses. Durable results remain in the project bucket.
