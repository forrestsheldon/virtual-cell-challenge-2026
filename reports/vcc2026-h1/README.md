# VCC 2026 scoring on H1

This directory retains the benchmark manifests, scale, and historical model
results used by the blog analysis. The evaluator itself is now released as the
standalone [`vcc2026-h1-benchmark`](https://github.com/forrestsheldon/vcc2026-h1-benchmark)
repository at version `v0.1.0`.

Install and prepare it with:

```bash
uv tool install git+https://github.com/forrestsheldon/vcc2026-h1-benchmark.git@v0.1.0
vcc-h1 setup --h1 data/external/vcc2025_h1/adata_Training.h5ad
vcc-h1 check
```

The setup command verifies the existing H1 object, downloads the small
versioned reference bundle, and locally extracts the control-only count object.
Omit `--h1` to download the original H1 data directly from Arc with resumption
and checksum verification.

## Benchmark contract

This is a deterministic development benchmark on public 2025 H1 training data,
not withheld truth or an official competition score. It contains the 126 H1
targets with at least 400 cells, one fixed 400-cell reference sample per target,
and all 38,176 non-targeting cells as the control reference. The 24 undersized
targets are outside the panel.

The scorer uses Arc's `vcc2026` profile through `cell-eval2==0.16.0`, source
commit `5e64833518a6603a0301cbe28185d49c30f4a986`, and the CPU
`pdex==0.3.0` path. Arc's reference CPM filter, normalization, target exclusion,
aggregation, edge cases, capped MSE policy, generic baseline, and split-half
anchor are preserved.

## Scoring

A prediction has raw, finite, non-negative integer counts on the exact H1 gene
axis, exactly 400 cells for each canonical target, no controls, and one
observation column named `target_gene`.

```bash
vcc-h1 validate prediction.h5ad
vcc-h1 score prediction.h5ad --output results/
```

The scorer appends the locally extracted real controls internally and writes
only compact `per_target.csv`, `aggregates.csv`, `scores.csv`, and
`manifest.json` outputs. The six scaled scores cover PDS, expression MSE, DE
fold-change error, direction fidelity, direction reach, and significant-gene
Jaccard overlap; `avg_score` is their equal-weight average.

The deterministic unchanged-control baseline can be reproduced with:

```bash
vcc-h1 score-control-baseline --output control-baseline/
```

Its expected scaled `avg_score` is `-0.045230687652238`. The retained
`control-baseline-trimmed/` result was reproduced by the standalone package with
byte-identical per-target output and aggregate/scaled differences below
`1e-12`.

Per-perturbation profiles can compare completed score directories without
rerunning DE or loading counts:

```bash
vcc-h1 plot \
  control=reports/vcc2026-h1/control-baseline-trimmed \
  global-shift=reports/vcc2026-h1/shifted-control-baseline \
  --output reports/vcc2026-h1/baseline_metric_profiles.png
```

Add `--targets TARGET1 TARGET2 ...` for a selected subset. The figure and its
companion CSV order perturbations by significant genes in the canonical frozen
reference DE table. `baseline_metric_profiles.png` shows the unchanged-control
and control-plus-global-shift comparison used to test this feature.

The earlier Stage 1 seed-sequence split remains useful for the private
linear-response experiments, but the public benchmark uses the CRC32 split
documented in the standalone repository. No compatibility layer is maintained.
