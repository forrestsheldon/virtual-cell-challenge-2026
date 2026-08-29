# Construct-balanced control-resampling baseline

## Submission

- Entry: `Td8NrsH2aoBlrcqxbeDc`
- Status: published
- Validation panel: `vcc2026-val-1`
- Anchor set: `vcc2026-valA-r4+vcc2026-valB-r4+vcc2026-valC-r4`
- Rank at publication: 349
- Artifact: `submissions/control-resampling-seed0/prediction.vcc`
- SHA-256: `7eabcccf54d187ce9159f5e4f466aa5b26d9e7d93282e8f67743fe55d0dcd0a4`

## Method

For each of 300 targets in each of three contexts, sample 400 raw control cells. Balance the sample over the 46
non-targeting constructs: 32 constructs contribute nine cells and 14 contribute eight. Sampling is without replacement
within each target-context pair and reproducible with seed 0. No cell filtering or UMI weighting is applied.

## Scores

| Metric | Score |
|---|---:|
| Overall | -0.30369301723919145 |
| Perturbation discrimination (PDS) | -0.004951375760924769 |
| Expression accuracy (MSE) | 0 |
| DE log-fold-change accuracy (NMAE) | -0.005313183600212712 |
| DE direction fidelity (FID) | -1.7214296946257555 |
| DE direction reach | -0.007889270403560343 |
| DE significance overlap (Jaccard) | -0.08257457904469546 |

The near-zero expression score is consistent with the intended null prediction. DE direction fidelity dominates the
negative overall result, as expected for a method that generates no target-specific response.
