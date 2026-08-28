# Recommended biological resources and representations

## Perturbation-response training

Start with Replogle K562 genome-wide using the full-gene scPerturb count H5AD: it covers 272/300 current VCC targets. Add Jiang's six lines (9 VCC targets in each) and McFaline-Figueroa's three glioblastoma lines (23 in each) for direct cross-context effects. Nadig HepG2/Jurkat and Replogle RPE1 have no exact overlap with this 300-gene panel, but remain valuable for learning **gene-general context-transfer structure** rather than direct target lookup.

## Context-transfer and representation learning

Use the three VCC control H5ADs as the target-domain anchor. Arc scBaseCount and Tahoe-100M may supply broad context priors, but sampling, provenance, and access costs need a dedicated plan before any large download.

## Responder and label quality

Use author/scPerturb count-level representations with guide and batch metadata. Compare Mixscape calls with PerturBase products only as a sensitivity analysis; do not learn responder biology solely from a representation that already removed non-perturbed cells.

## Covariance and linear response

Use full-gene cell-level counts, then explicitly compare count, log-count, and residual covariance. scPertEval and PerturBench gene-subset files are unsuitable as the sole covariance source.

## Metric benchmarking

Use scPertEval for quick protocol experiments and official `cell-eval2` with count-level held-out data for metric-faithful tests. Legacy `cell-eval` and scPertEval's own protocols are not the VCC 2026 scoring contract.

## Representation comparison priority

Before cross-study integration, compare Replogle and Nadig scPerturb counts against their scPertEval derivatives, and Srivatsan scPerturb/PertPy against PerturBench. This isolates preprocessing-induced geometry from biological context.
