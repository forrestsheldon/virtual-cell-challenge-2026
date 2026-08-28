# Provenance principles and findings

An experiment is evidence; a hosted copy is an access route. `PertPy → scPerturb H5AD → author deposit` is one lineage, not three experimental replications.

Verified findings:

- PertPy 1.2.0's relevant Replogle and Srivatsan loaders explicitly state “Obtained from scperturb,” download `exampledata.scverse.org` H5ADs, and call `read_h5ad` without another transformation.
- scPerturb's public Replogle files start from author raw single-cell H5ADs, preserve raw count `X`, repair/standardize gene and perturbation metadata, and retain guide/batch/QC columns. The public harmonized file is distinct from scPerturb's paper-analysis workflow, which filters, normalizes, log-transforms, selects HVGs, computes PCA, and subsamples.
- For K562 GWPS specifically, the 65,830,941,948-byte author H5AD and 8,805,466,154-byte scPerturb H5AD have the same 1,989,578 cells x 8,248 genes and dense `float32` raw-count `X`. The author matrix is contiguous and uncompressed; scPerturb writes gzip-compressed chunks. The transformation script does not subset cells or genes, and sampled remote rows were identical. All author `obs` fields are retained or renamed, `gem_group` becomes `batch`, `sgID_AB` becomes `guide_id`, and the original Ensembl gene index is retained as `ensembl_id`. See `replogle_k562_gwps_artifact_decision.md` for the download decision.
- scPertEval performs an independent evaluation-oriented conversion: ≥200 genes/cell and ≥3 cells/gene, total-normalize to 10,000, `log1p`, sparse `float32`, collapse controls to `control`, and trim to perturbation-only `obs` and index-only `var`. Raw counts, guide metadata, batches, layers, embeddings, graphs, and `uns` are removed.
- PerturBase independently applies cell/gene/mitochondrial/perturbation QC and total-10,000 `log1p`; optional denoising adds 4,000 HVGs, PCA/neighbors/UMAP/Leiden and Mixscape responder filtering. Its download product must be identified (raw-after-QC versus normalized/HVG/Mixscape) before use.
- PerturBench independently curates benchmark files. Its common pipeline retains a `counts` layer but only after restricting to a union of ~4,000 HVGs, top DE genes, and perturbation targets; whole-transcriptome comparability is therefore lost.

Unknowns are first-class data. Exact PerturBase record inclusion and product names for each VCC study remain unresolved, and no file is assigned a raw-count claim without source code or structural inspection.
