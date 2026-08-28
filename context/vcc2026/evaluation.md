---
last_verified: 2026-08-21
status: current
---

# Current 2026 evaluation

The authoritative 2026 implementation is **`ArcInstitute/cell-eval2`**, not legacy `ArcInstitute/cell-eval`. Inspected source: `cell-eval2` 0.16.0, commit `5e64833518a6603a0301cbe28185d49c30f4a986` (2026-08-20). The published VCC metric brief is dated 2026-08-19, specifies rule version 3, and says the reference anchors were produced with 0.15.0. Treat released reference bundles as the numeric anchor and pin both rule and software provenance.

All six metrics exclude the perturbed target gene. PDS additionally excludes all 300 panel targets from every profile. Counts are pseudobulk-normalized to 50,000 and `log1p` for PDS/MSE; DE uses per-cell total normalization to 1,000,000. Wilcoxon tests are two-sided, the reference-control expression gate is >5 CPM, and Benjamini–Hochberg significance is `padj < 0.05` within each perturbation after filtering.

| official name | published statistic | level / aggregation | better; undefined behavior |
|---|---|---|---|
| `pds_cosine` | For real effects \(r_i\) and predicted effect \(p_i\), rank all cosine similarities \(\cos(p_i,r_j)\) with midranks and score \(1-k_i/(n-1)\), then mean over 300 targets. A zero/shared profile scores 0.5. | pseudobulk perturbation effects; cohort retrieval | higher |
| `expr_mse_unbiased_capped_norm` | Jackknife-corrected squared error on normalized/log pseudobulk means; the panel statistic is a ratio of summed numerator to summed denominator (not a mean of per-target ratios). Raw perfection is 0 and a control-like prediction is near 1. | gene × perturbation, panel ratio | lower raw; reference-scaled result alone is clipped to [0,1] |
| `de_wilcoxon_lfc_nmae` | On reference-significant genes, \(\sum_g |\hat L_g-L_g| / \sum_g |L_g|\), where \(L=\log_2((\mu_p+10^{-9})/(\mu_c+10^{-9}))\). Missing predicted LFC is zero. | DE genes per target, then mean over eligible targets | lower; require at least 10 adjudicable reference genes; invalid targets omitted |
| `de_wilcoxon_direction_fidelity_yield_raw` | \(F=K/\max(n_{pred},n_{real})\), with K significant genes shared in the same LFC direction. | DE sets per target, mean over defined targets | higher; implementation-defined empty cases are retained from the official scorer |
| `de_wilcoxon_direction_reach_raw` | Rank reference-significant genes by prediction confidence; find deepest prefix with direction purity ≥0.9; \(R=k^*/n_{real}\). | ranked DE direction per target, mean over nonempty reference sets | higher; empty reference sets omitted |
| `de_wilcoxon_sig_jaccard` | \(|S_{pred}\cap S_{real}|/|S_{pred}\cup S_{real}|\). | significant-gene sets per target, then mean over 300 | higher; empty union = 1 |

The exact edge cases and jackknife operations live in the pinned source and metric PDF; this summary is not a replacement implementation.

## Reference scaling and leaderboard

For raw statistic `u`, control-mean baseline `b`, and replicate anchor `r`, each context score is `s = (u - b) / (r - b)` (with orientation handled by the metric values themselves). `b` is an equal-weight average of per-NTC-construct mean count vectors for constructs passing the official filters. `r` is the mean of five deterministic, disjoint half-split reference replicates, each using its own control half (`n_splits=5`, base seed 0). A score of 0 matches the baseline; 1 matches measured replicate reliability; values outside [0,1] can occur except for the capped MSE score.

The overall score is the unweighted mean of 6 metric scores × 3 contexts. Published validation-anchor ranges provide useful sanity checks: PDS baseline 0.500/replicate 0.927–0.984; MSE 0.986–0.992/0.028–0.045; fidelity 0.505–0.522/0.795–0.832; reach 0.047–0.097/0.958–0.978; Jaccard 0.021–0.037/0.375–0.423; NMAE 1.0009–1.0017/0.369–0.431.

## Implemented protocols and ceilings

`cell-eval2` supports v1/v2 configuration families, streaming/partitioned scoring, baseline construction, `prep-real-bundle`, reference anchors, and a generic split-half ceiling module. VCC 2026 uses `configs/vcc2026.yaml`, v2 names, counts input, Wilcoxon DE, and its dedicated anchor path. The source explicitly warns that the generic Spearman–Brown ceiling extrapolation is not valid for all VCC metrics; the competition uses measured split-half anchors instead.

Legacy `cell-eval` was inspected at v0.8.2, commit `6928cf8bd7a706040ccfd13119e4085726dee64a` (2026-07-27). It is a useful generic evaluator with older profiles and ceiling utilities, but it is **not** the current competition scorer. The live submission validator also requires exactly 400 cells/target, even though generic evaluator documentation can accept arbitrary prediction cell counts.

## Future stress tests (do not run yet)

- correct mean with wrong variance or wrong covariance;
- collapsed multimodal mixtures;
- uniformly shrunk or inflated responses;
- correct direction but wrong magnitude;
- the same average response for every perturbation;
- correct perturbation effect with mismatched library size;
- construct-imbalanced controls;
- target-gene leakage and panel-target leakage.
