---
last_verified: 2026-08-29
status: current
---

# Official tooling

The released PyPI package is `vcc-cli` 0.1.0 (Python ≥3.11). Prefer the documented isolated install:

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

## Official agent skill

The CLI wheel contains an official `vcc/skill/SKILL.md` and references for installation, authentication, data, preparation, submission, and status. Install/update it with `vcc skill install`, and rerun after each CLI upgrade. We reference the released copy rather than duplicating it because the skill is version-coupled to the CLI. The skill's compressed-download estimate (~406 MiB) differs from the live page (~630 MB) and local uncompressed H5AD total (~662 MB); units/compression explain at least part of that difference.

## Local evaluation

The official evaluation repository is `https://github.com/ArcInstitute/cell-eval2`; pin the competition config, rule version, package version, source commit, and real reference bundle fingerprint. Public validation truth is not supplied, so full official validation scoring is server-side. Local metric experiments can use external datasets or synthetic held-out splits, but they are not interchangeable with the leaderboard.
