# Preprocessing landscape

- **Original deposits** anchor experimental provenance. “Author processed” may still include guide assignment, QC, or normalization and must be read study by study.
- **scPerturb** is a harmonization pipeline. Its inspected relevant public H5ADs preserve sparse raw-count `X` and rich standardized metadata. Its downstream paper-benchmark workflow additionally filters, total-normalizes, log-transforms, selects HVGs, computes PCA, and subsamples; that workflow must not be confused with the public count H5AD.
- **PertPy** is a convenience interface for the relevant Replogle and Srivatsan files. Its loaders explicitly fetch scPerturb-derived H5ADs and do not independently preprocess them.
- **scPertEval** is an independent, evaluation-specific representation: cell/gene light QC, total-10k, `log1p`, sparse float32, collapsed controls, and almost all metadata removed. It is ideal for lightweight metric work and poor for raw-count, guide, batch, depth, or covariance analyses.
- **PerturBase** independently performs cell/gene/mitochondrial/perturbation QC, total-10k `log1p`, and optional HVG/PCA/Mixscape responder filtering. Exact product and study inclusion must be verified before use.
- **PerturBench** independently curates Jiang, McFaline-Figueroa, and Srivatsan for modeling. It retains selected-gene counts but subsets the gene axis to an HVG/DE/target union and makes scientifically material dataset-specific exclusions.
- **Arc Virtual Cell Atlas/scBaseCount** is a very large count-oriented observational layer useful for context representation, not independent perturbation-response evidence.

The practical conclusion is analysis-specific: keep full count/guide variants for biological inference, use lossy variants when their smaller contract exactly matches the experiment, and always compare the same biology across pipelines before attributing differences to cell context.
