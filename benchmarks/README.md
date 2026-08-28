# Zero-shot evaluation ladder

These YAML files define benchmark questions and information boundaries. E3 has distinct pairwise (E3a) and multi-source (E3b) definitions. They are plans, not runnable benchmarks or final inclusion policies. Nominal capacities are regenerated from `metadata/datasets/perturbation_atlas_sources.yaml` by:

```bash
pixi run python scripts/metadata/build_perturbation_atlas.py
```

No perturbation is filtered here for cell count, guide quality, knockdown, response strength, contamination, or responder status. E0 reports descriptive split-size capacities separately; those are not general atlas filters.
