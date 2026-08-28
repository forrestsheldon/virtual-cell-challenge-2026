# Provisional representations by purpose

These choices follow the audited lineages; they are not a single “best dataset” ranking.

| purpose | initial representation | rationale / guardrail |
|---|---|---|
| fast exploration | scPertEval for Replogle/Nadig/H1; PertPy for scPerturb-backed studies | low-friction; never count mirrors as new evidence |
| competition metric experiments | scPertEval for rapid protocol checks; raw-count scPerturb/author data for metric-faithful experiments | final outputs and official evaluator operate on counts |
| mean/pseudobulk response atlas | author or scPerturb full-gene raw-count H5AD | preserves whole transcriptome and label metadata |
| Trashpanda/contamination | author/scPerturb raw counts with guide, depth, and batch metadata | normalized-only/trimmed variants are unsuitable |
| Mixscape/responder analysis | author/scPerturb count-level cells; compare with PerturBase Mixscape product | retain controls and guide-level heterogeneity; avoid circular use of prefiltered responders |
| false-negative/label audit | richest author/scPerturb guide-level variant | guide agreement and NTC distinctions are essential |
| cross-dataset integration | comparable count-level variants, with within-experiment representation controls | quantify preprocessing effects before harmonization |
| covariance/linear response | full-gene cells from author/scPerturb, then explicitly compare counts, log counts, and residuals | no silent representation choice; avoid HVG-only covariance claims |
| representation learning | Arc atlas/scBaseCount/Tahoe plus controls, with study-aware sampling | useful context prior but does not supply genetic effects |
| model benchmarking | PerturBench selected-gene products | reproducible splits/DE; not a source for whole-transcriptome effect atlases |

For Replogle K562 genome-wide, download the 8,805,466,154-byte scPerturb H5AD rather than the 65,830,941,948-byte author H5AD: the audited files have the same 1,989,578 cells, 8,248-gene raw-count matrix, guide annotations, and GEM-group batch information, while scPerturb is gzip-compressed and adds traceable metadata harmonization. See `replogle_k562_gwps_artifact_decision.md`. The two Nadig raw-count harmonizations remain the next large-download review because these three files jointly offer the highest target coverage across four CRISPRi contexts. Download scPertEval variants only when their lossy contract is sufficient.
