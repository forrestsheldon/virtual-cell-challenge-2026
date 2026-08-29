# Local submission artifacts

Large `.h5ad` and `.vcc` files in this directory are gitignored. Reproduce the
first negative baseline from validation controls with:

```bash
pixi run python scripts/baselines/resample_controls.py \
  data/controls submissions/control-resampling-seed0/prediction.h5ad --seed 0

pixi run vcc prep submissions/control-resampling-seed0/prediction.h5ad \
  -g data/controls/gene_names.csv \
  --perts data/controls/pert_counts.csv \
  -o submissions/control-resampling-seed0/prediction.vcc --dry-run
```

Remove `--dry-run` only after validation succeeds to write the `.vcc` package.

For every target-context pair, the script samples 400 raw control cells without
replacement. It balances the 46 `ntc_id` constructs as evenly as possible: 32
constructs contribute 9 cells and 14 contribute 8, with the allocation and cell
selection determined by the global seed, context index, and target index. Cells
may be reused for different target-context pairs. The output contains only the
required `target_gene` and `context` columns and preserves the official gene
order.
