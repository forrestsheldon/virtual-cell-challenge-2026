# Preprocessing comparison

| representation | count/full gene information | label/guide metadata | best uses | material loss/risk |
|---|---|---|---|---|
| author deposit | varies; closest available to acquisition | usually richest | provenance, reprocessing, label audit | format and QC vary; author “processed” may already transform data |
| scPerturb public harmonized | raw-count `X` verified for inspected relevant H5ADs; broad gene axis | rich standardized context, perturbation, guide/batch/QC where upstream supplies it | pseudobulk, count models, responder/guide audit, covariance experiments | harmonized labels and gene-symbol fixes must be traced; multiple-guide policy is dataset-specific |
| PertPy relevant loader | same prepared scPerturb file | same as upstream file | convenient exploratory access | treating it as independent evidence or a distinct preprocessing pipeline |
| scPertEval hosted | total-10k `log1p`, sparse float32; all post-QC genes | only collapsed `perturbation`; index-only `var` | quick exploration and metric/protocol benchmarking | no counts, depth, guides, batches, detailed context, layers, embeddings, graphs, or `uns` |
| PerturBase | raw-after-QC and normalized/denoised products exist | harmonized; exact product-dependent | alternate QC and Mixscape/responder sensitivity | mito and minimum-perturbation filters change populations; denoising can delete NP cells and restrict genes |
| PerturBench | counts layer retained only for selected HVG/DE/target union; `X` total-normalized/log1p | useful covariates and precomputed DE, dataset-specific cleanup | reproducible model benchmark | not whole transcriptome; Jiang/McFaline/Srivatsan exclusions change the biological question |
| Arc atlas/scBaseCount | count-oriented atlas, documented STARsolo Unique and UniqueAndMult | study/context metadata at atlas scale | control/context representation learning | observational, not perturbation response; requester-pays and huge |

## Suitability by scientific requirement

- **Pseudobulk response:** full-gene trustworthy count matrices are preferred. scPertEval can prototype directionality but cannot reproduce count-aware normalization or sampling corrections.
- **Trashpanda/decontamination:** requires counts and depth; scPertEval is unsuitable. Guide/construct metadata strengthens contamination diagnostics.
- **Mixscape/responder analysis:** needs perturbation and matched-control cells, ideally guide and batch fields. A pre-filtered PerturBase Mixscape product is useful as a sensitivity endpoint, not as the sole input.
- **False negatives and label audit:** use author/scPerturb guide-level data. Collapsed labels erase disagreement among guides and NTC constructs.
- **Covariance/linear response:** cell-level full-gene counts or a deliberately chosen residual representation are required. HVG-only/log-only files can manufacture or erase covariance structure.
- **Official metric experiments:** raw-count outputs are ultimately required by VCC, but scPertEval is convenient for algorithmic prototypes. Use the official `cell-eval2` preprocessing contract for metric-faithful runs.
- **Cross-dataset integration:** begin from count-level variants where feasible, then compare representation effects within the same experiment. Do not batch-correct away context before measuring whether context is the transferable signal.

Preprocessing can remove hypotheses, not just noise. Log-only files cannot recover counts or library size; collapsed controls erase construct structure; discarded batch metadata blocks batch-aware effects; HVG restriction distorts whole-transcriptome metrics; and aggressive batch correction may remove the genuine context dependence that defines this zero-shot task.
