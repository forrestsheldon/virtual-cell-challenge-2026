# Agent guidance

This repository is the source of truth for Virtual Cell Challenge 2026 analysis and modeling. Before assuming anything about the Challenge, read `context/vcc2026/`. Current official VCC sources override model memory and stale local notes; rules, evaluation, dates, and tooling may change during the competition.

- Preserve source URLs, access dates, versions, tags, and commits. Record unknowns explicitly.
- Distinguish original experiments, author-processed data, independent reprocessing/harmonization, mirrors, and convenience loaders. Two hosted copies of one processed dataset are not independent evidence.
- Inspect metadata and preprocessing provenance before selecting data. Do not download large datasets speculatively or any individual file over 100 MB without explicit approval.
- Prefer raw counts when analyses require our own biological or preprocessing choices. Processed copies can be valuable, but document their transformations first.
- Never silently normalize, log-transform, filter, batch-correct, or select HVGs. Never overwrite raw data.
- Keep large data and derived matrices gitignored. Make derived outputs reproducible from scripts and configuration where practical.
- Motivate modeling choices by the zero-shot structure of the 2026 task.
- Do not commit or push unless explicitly instructed.

## Google Cloud operations

Use Google Cloud only within the following boundary unless Forrest explicitly
changes it:

- Project: `bold-bastion-509200-f9`
- gcloud configuration: `vcc-2026`
- Primary region: `europe-west2`
- Durable derived-data bucket: `gs://bold-bastion-509200-f9-vcc2026`

Before any mutating cloud command, verify that the active gcloud configuration
and project match this boundary. Pass the project explicitly to Compute Engine
commands. Do not change billing, budgets, quotas, IAM, service accounts,
networking, firewall rules, public access, or bucket policy without explicit
approval. Use the existing bucket-scoped worker service account; do not create
or download persistent service-account keys.

State the proposed machine, disk sizes and classes, region, estimated running
and retained-storage cost, and retention plan before creating materially
billable resources. Creation is allowed only when the current task clearly
authorizes it. A budget alert is not a spending cap. Stop compute promptly when
work finishes or becomes idle, including after a failed job when it is safe to
do so.

Treat cloud artifacts according to these stages:

1. Write new or corrected outputs to a versioned staging prefix.
2. Verify source identity, scientific coverage, reconstruction checks, and
   manifest hashes independently of the producing job.
3. Archive an existing canonical artifact before replacing it.
4. Promote only after the acceptance checks pass and local/cloud checksums
   agree.

Do not use the durable bucket as a permanent mirror for large public source
data without explicit approval. Do not delete source data, review-retained
disks, canonical artifacts, or archives without explicit approval. Temporary
workers and scratch disks may be deleted only when their intended lifecycle was
agreed in advance and their durable outputs have passed independent validation.

At the end of a cloud task, report the exact project, resource inventory and
state, durable artifact locations and hashes, validation result, retained
resources, and continuing cost. Record these details in the repository when
they are needed to reproduce or audit the work.

## Analysis code

These are scientific scripts, not production code. The priority is clarity, readability, and brevity.

- Most scripts are run once. Do not write defensive code, compatibility shims, or configuration for cases that do not exist. Backwards compatibility is not a priority.
- Confirm that each added step in an analysis or numerical implementation is working, with positive and negative controls whenever one is available. A step whose correctness has not been demonstrated is not finished.
- Prefer a short script whose logic fits on one screen over a general framework.

## Working together

- Come to Forrest to think problems through carefully when decisions present themselves. Always aim to understand a problem fully before proposing solutions.
- Name the options and the tradeoff rather than silently resolving a genuine modelling, statistical, or design choice.
