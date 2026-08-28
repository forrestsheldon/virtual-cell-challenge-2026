# Virtual Cell Challenge 2026

Research code and reproducible analyses for the 2026 Virtual Cell Challenge. The public site explains results; this repository contains the analysis that produces them.

## Environment

Install [Pixi](https://pixi.prefix.dev/), then install the locked environment:

```bash
pixi install --frozen
```

Start JupyterLab:

```bash
pixi run lab
```

Run the lightweight checks:

```bash
pixi run check
```

The initial environment supports AnnData/H5AD, CSV, and Parquet exploration. Add modelling frameworks only when a concrete experiment requires them.

## Repository layout

```text
data/       Local challenge data; ignored by git
notebooks/  Exploratory analyses
src/        Reusable analysis code
tests/      Tests and environment smoke checks
```

Keep raw and derived challenge data out of git. Record download sources, checksums, and transformations in `data/README.md` as the release details become available.
