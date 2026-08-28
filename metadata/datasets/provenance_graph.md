# Representation lineage

```text
Replogle 2022 Figshare author H5AD/MTX
├── scPerturb raw-count harmonization
│   └── PertPy exampledata mirror/loader (same prepared representation)
├── scPertEval normalized/log1p, metadata-trimmed benchmark
└── PerturBase independent QC/normalization/Mixscape products (record mapping pending)

Nadig 2025 GEO GSE264667
├── scPerturb raw-count harmonization
└── scPertEval normalized/log1p, metadata-trimmed benchmark

Jiang 2025 author Seurat objects / GSE281048 / Zenodo
└── PerturBench: restore Seurat raw X, curate labels, common gene-subset pipeline

Srivatsan 2020 GEO GSE139944
├── scPerturb count reconstruction + metadata harmonization
│   └── PertPy exampledata mirror/loader
├── PerturBench: scPerturb upstream, human symbol mapping, highest-dose subset, common gene-subset pipeline
└── PerturBase independent products (exact record mapping pending)

McFaline-Figueroa 2024 GEO GSE225775
├── scPerturb dataset-specific reconstruction scripts (public completed product unclear)
├── PerturBench: GEO files, single-gene-only and treatment filtering, common gene-subset pipeline
└── PerturBase independent products (exact record mapping pending)

VCC 2025 H1 author challenge data
└── scPertEval arch1 normalized/log1p, metadata-trimmed benchmark
```

Branches indicate independent transformations, not independent biology. Dataset/context target coverage is counted once regardless of branch count.
