---
last_verified: 2026-09-04
status: current
---

# Submission contract

`prediction.h5ad` must have exactly **360,000 rows × 18,533 columns** for one round: 400 cells for every target-context pair. `obs` must contain `target_gene` and `context`; it must not contain control cells. `var_names` must exactly equal `gene_names.csv` in order.

`X` must be sparse raw counts: finite, non-negative, whole-valued, and with each cell sum ≤1,000,000. The validator limits the stored sparse entries to 4.75 billion (about 13,200/cell), and explicitly stored zeros count toward that cap. The full archive may contain at most 400,000 cells.

## Integerizing decoded expected counts

This is modeling guidance, not an additional submission rule. If a decoder
produces a fractional expected count \(\lambda_g\), do not create repeated cells
with \(\operatorname{round}(\lambda_g)\). The per-gene rounding error is then
identical in every cell, so increasing the number of submitted cells does not
remove its bias from the pseudobulk mean or derived LFC. Across many low-count
genes this can produce a large systematic NMAE penalty.

For a mean-only prediction, independently stochastic-round each cell and gene:

\[
X_{ig}=\lfloor\lambda_g\rfloor+
\operatorname{Bernoulli}(\lambda_g-\lfloor\lambda_g\rfloor).
\]

Then \(E[X_{ig}]=\lambda_g\), and the emitted pseudobulk approaches the decoded
fractional profile as the number of cells increases. Validate the realized
per-gene mean and LFC against the pre-integerization profile, and repeat the
check across deterministic seeds. Any subsequent library-size adjustment must
also be checked for reintroduced bias.

Stochastic rounding supplies only independent integerization noise. It does
not reproduce biological heterogeneity, gene-gene covariance, multimodality,
or the true cell distribution. Improvements in mean-based metrics therefore
must not be reported as evidence of successful distributional generation.

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

With `vcc-cli` 0.2.0, use `vcc cancel <entry-id>` to abandon an interrupted
upload or a scoring job that is still queued. A job that has started running
cannot be cancelled. A cancelled job does not consume the daily limit; a
successfully scored submission does. The `superseded` terminal state means the
entry scored but a newer submission from the same team owns the leaderboard
position.
