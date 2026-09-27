---
last_verified: 2026-09-04
status: current
---

# Official tooling

The released PyPI package is `vcc-cli` 0.2.0 (Python ≥3.11), published
2026-09-01. Prefer the documented isolated install:

```bash
uv tool install vcc-cli
uv tool upgrade vcc-cli
vcc --version
```

Authentication uses the member portal. Never paste an access token into chat or logs:

```bash
vcc login --token-stdin
vcc whoami
# headless alternative: export VCC_TOKEN=<key>
```

Dataset and submission workflow:

```bash
vcc datasets list
vcc datasets download controls -d data/controls
vcc prep prediction.h5ad -g data/controls/gene_names.csv \
  --perts data/controls/pert_counts.csv -o prediction.vcc --dry-run
vcc submit prediction.vcc -m "model note" --wait
vcc status
```

Downloads are resumable and CRC32C-checked. `vcc sample` can generate a schema-valid random file, but it is not a scientific baseline.

Version 0.2.0 adds `vcc cancel` for abandoning an upload or a queued scoring
job without consuming a daily submission, explains the `superseded` terminal
state, and substantially reduces `vcc prep` peak memory. Packaging now includes
a small `meta.json` member with the submitted shape and stored-nonzero count so
the service can select an appropriate scoring machine without probing the full
archive. These are submission and packaging changes; they do not change the
`cell-eval2` metric profile.

## Official agent skill

The CLI wheel contains an official `vcc/skill/SKILL.md` and references for installation, authentication, data, preparation, submission, and status. Install/update it with `vcc skill install`, and rerun after each CLI upgrade. We reference the released copy rather than duplicating it because the skill is version-coupled to the CLI. The skill's compressed-download estimate (~406 MiB) differs from the live page (~630 MB) and local uncompressed H5AD total (~662 MB); units/compression explain at least part of that difference.

## Local evaluation

The official evaluation repository is `https://github.com/ArcInstitute/cell-eval2`.
As checked on 2026-09-04, PyPI remains at 0.16.0, `pdex` remains at 0.3.0,
and the repository's `main` HEAD remains
`5e64833518a6603a0301cbe28185d49c30f4a986`. Pin the competition config, rule
version, package version, source commit, and real reference bundle fingerprint.
Public validation truth is not supplied, so full official validation scoring is
server-side. Local metric experiments can use external datasets or synthetic
held-out splits, but they are not interchangeable with the leaderboard.
