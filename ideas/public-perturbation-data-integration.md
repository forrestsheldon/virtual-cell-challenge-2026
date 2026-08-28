# Can public perturbation datasets be one training corpus?

Working question: **Can public perturbation datasets be treated as one training corpus, and what biological information is lost when we make them statistically similar?**

## Audit before integration

For each representation, record reference build and symbol mapping; common gene universe; cell/gene filters; controls and NTC constructs; raw/count/log scale; library depth; perturbation and guide assignment; multiple-guide policy; batch/replicate fields; chemistry and guide capture; time/dose/stimulation; and retained layers/metadata.

## Planned lightweight analyses

1. Align genes with an explicit alias table while retaining source symbols and one-to-many conflicts.
2. Compare library-size, sparsity, control means/variances, and perturbation cell counts before normalization.
3. Fit PCA to controls, then to pseudobulk perturbation-response vectors; label by dataset, cell context, chemistry, and representation.
4. Quantify whether dataset identity, cellular context, or preprocessing version dominates distances and variance components.
5. Test explicit integration against no integration using held-out biological contexts, not only random cells.
6. Measure whether integration removes real context-specific response direction or magnitude.
7. Repeat in counts, log counts, and deliberately defined residual representations; record every transform.

## Critical within-experiment control

Compare the **same biological experiment** before comparing different experiments:

```text
author/author-processed
vs scPerturb
vs PertPy copy
vs scPertEval
vs PerturBase
vs PerturBench (where present)
```

PertPy/scPerturb equality should be verified as a mirror invariant. Replogle and Nadig count-vs-scPertEval comparisons estimate losses from normalization and metadata trimming. Srivatsan and McFaline comparisons isolate gene-universe and condition filtering. Only after these controls should cross-dataset geometry be interpreted biologically.

## Decision criteria

Integration is helpful only if it improves context-held-out perturbation prediction without erasing context-dependent effects, inflating apparent target coverage, or making the VCC count-output distribution unrecoverable. Heavyweight integration is deferred until the large-file choices and raw-vs-harmonized sensitivity set receive human review.
