---
last_verified: 2026-08-29
status: current
---

# Submission contract

`prediction.h5ad` must have exactly **360,000 rows × 18,533 columns** for one round: 400 cells for every target-context pair. `obs` must contain `target_gene` and `context`; it must not contain control cells. `var_names` must exactly equal `gene_names.csv` in order.

`X` must be sparse raw counts: finite, non-negative, whole-valued, and with each cell sum ≤1,000,000. The validator limits the stored sparse entries to 4.75 billion (about 13,200/cell), and explicitly stored zeros count toward that cap. The full archive may contain at most 400,000 cells.

Package and validate without uploading:

```bash
vcc prep prediction.h5ad \
  -g data/controls/gene_names.csv \
  --perts data/controls/pert_counts.csv \
  -o prediction.vcc \
  --dry-run
```

Remove `--dry-run` to write the single-H5AD `.vcc` package. Submit with:

```bash
vcc submit prediction.vcc -m "concise model/version note" --wait
```

Use `vcc submit --resume` after an interrupted transfer and `vcc status` to inspect jobs. A submission counts only after it reaches scoring. The limit is two scoring submissions per UTC day, resetting at 00:00 UTC, with one submission in flight at a time.
