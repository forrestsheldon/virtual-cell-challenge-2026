# Context changelog

## 2026-09-04 — CLI 0.2.0 and evaluator rechecked

Updated the project pin and tooling notes from `vcc-cli` 0.1.0 to 0.2.0. The
release adds cancellation, clearer submission-state reporting, and lower-memory
prediction validation and packaging. It does not change scoring semantics.
Rechecked the evaluator independently: PyPI still publishes `cell-eval2` 0.16.0
and `pdex` 0.3.0, and `ArcInstitute/cell-eval2` `main` still resolves to
`5e64833518a6603a0301cbe28185d49c30f4a986`. Existing H1 reference and scale
artifacts therefore remain current and were not rebuilt.

## 2026-08-29 — submission contract rechecked

Rechecked the live dataset and CLI documentation before building the first
control-resampling artifact. The 360,000-cell, 18,533-gene contract and
`vcc-cli` 0.1.0 remain current. The CLI guide has moved from the portal's former
`/cli-guide` route to `https://vcc-cli-wiki.virtualcellchallenge.org/`.

## 2026-08-21 — initial snapshot

Initial local snapshot of the live 2026 challenge pages, CLI 0.1.0, `cell-eval2` 0.16.0 source, metric brief, rules, and locally downloaded validation-control metadata. No historical change is asserted without a verified earlier snapshot.
