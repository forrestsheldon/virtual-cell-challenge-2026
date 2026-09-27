# Linear-response ladder

**Status:** accepted implementation plan  
**Prepared:** 2026-09-09  
**Supersedes:** `linear-response-h1-diagnostics.md`

## Scientific question

Test what, if any, target-specific response information is contained in control-cell
correlations. This is an H1-retrospective diagnostic for a blog post, not a submission
optimization exercise.

## Fixed comparison

- Fit on the strict 26-guide controls and the 10,780 control-CPM-above-5 genes, retaining
  all 126 targets as internal drivers.
- Normalize every response to an on-target coefficient of -1 and transfer a separate
  nonnegative downstream magnitude by leave-one-target-out median oracle scale.
- Use signed stable-strong-DE top-K recovery as the primary scale-free statistic, with
  wrong-target, gene-label permutation, and target-bootstrap controls.
- Compare normalized empirical covariance; all-control and raw-count sensitivities;
  SE-scaled Laplace sparse response; factor-plus-diagonal ranks 10, 25, and 50; and
  global-only, model-only, and combined responses.
- Use a deterministic two-fold gene split to measure a truth-informed forcing-vector
  ceiling at target-only, target plus 1, 4, or 9 learned coordinates, and the full
  rank-50 span.
- Select sparse penalty and factor rank using held-out controls only. Stop after the
  factor model. Run no full cell-level score unless matched-target direction fidelity
  significantly exceeds the wrong-target null.

## Output contract

Shared numerical operations live in `scripts/linear_response/kernel.py`, with one model
script, one truth-owning evaluator, and one detached staged launcher. The blog receives
only headings, equations, generated tables and figures, and structured HTML comments.
Every displayed artifact is backed by a compact hashed CSV. No packaging, submission,
publication, commit, or push is authorized.
