"""Repeat response-gene cross-fitting after shared-gene normalization."""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd

from scripts.evaluation.crossfit_empirical_scale import target_folds
from scripts.evaluation.crossfit_replogle_h1_gene_scale import (
    AGGREGATION,
    BULK,
    MIN_STRONG,
    PHASE1,
    PRIOR_UMIS,
    REPORT,
    context_features,
    context_permutation_test,
    crossfit_models,
)
from scripts.evaluation.prepare_replogle_h1_phase1 import truth_arrays
from scripts.evaluation.prepare_replogle_h1_reliability import dirichlet_lfc
from scripts.evaluation.prepare_replogle_h1_transfer import loo_components


def summarize_residuals(
    variant: str,
    predictions: dict[str, np.ndarray],
    source: np.ndarray,
    target: np.ndarray,
    strong: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = {"unscaled": source, **predictions}
    per_target_rows = []
    for model, prediction in models.items():
        for row in range(len(source)):
            selected = strong[row]
            if selected.sum() < MIN_STRONG:
                continue
            difference = prediction[row, selected] - target[row, selected]
            per_target_rows.append(
                {
                    "source_effect": variant,
                    "model": model,
                    "target_row": row,
                    "strong_genes": int(selected.sum()),
                    "strong_residual_mse": float(np.square(difference).mean()),
                    "strong_residual_nmae": float(
                        np.abs(difference).sum()
                        / np.abs(target[row, selected]).sum()
                    ),
                    "strong_residual_sign_agreement": float(
                        (
                            np.sign(prediction[row, selected])
                            == np.sign(target[row, selected])
                        ).mean()
                    ),
                }
            )
    per_target = pd.DataFrame(per_target_rows)
    summary = (
        per_target.groupby(["source_effect", "model"], sort=False)
        .agg(
            eligible_targets=("target_row", "size"),
            mean_strong_residual_mse=("strong_residual_mse", "mean"),
            mean_strong_residual_nmae=("strong_residual_nmae", "mean"),
            mean_strong_residual_sign_agreement=(
                "strong_residual_sign_agreement",
                "mean",
            ),
        )
        .reset_index()
    )
    return per_target, summary


def main() -> None:
    with np.load(AGGREGATION, allow_pickle=False) as saved:
        targets = saved["target_gene"].astype(str).tolist()
        matched = saved["matched_target_gene"].astype(str).tolist()
        genes = saved["shared_gene"].astype(str).tolist()
        raw_source = saved["replogle_lfc_shared"].astype(np.float64)
        h1 = saved["h1_lfc_shared"].astype(np.float64)
        target_counts_all = saved["replogle_count_sum"].astype(np.int64)
        source_control_all = saved["replogle_control_count_sum"].astype(np.int64)
        h1_control_all = saved["h1_control_count_sum"].astype(np.int64)

    bulk = ad.read_h5ad(BULK, backed="r")
    try:
        unique_source_genes = pd.unique(
            bulk.var["gene_name"].astype(str).to_numpy()
        ).tolist()
    finally:
        bulk.file.close()
    source_lookup = {gene: index for index, gene in enumerate(unique_source_genes)}
    shared_source = np.asarray([source_lookup[gene] for gene in genes])
    target_counts = target_counts_all[:, shared_source]
    source_control = source_control_all[shared_source]
    shrunk_source = dirichlet_lfc(
        target_counts,
        target_counts.sum(axis=1),
        source_control,
        source_control.sum(),
        PRIOR_UMIS,
    )

    _, raw_residual = loo_components(raw_source)
    _, shrunk_residual = loo_components(shrunk_source)
    _, h1_residual = loo_components(h1)
    target_lookup = {target: index for index, target in enumerate(targets)}
    matched_rows = np.asarray([target_lookup[target] for target in matched])
    target_residual = h1_residual[matched_rows]

    with np.load(PHASE1, allow_pickle=False) as saved:
        output_genes = saved["output_gene"].astype(str).tolist()
    _, stable_output, _ = truth_arrays(targets, output_genes)
    output_lookup = {gene: index for index, gene in enumerate(output_genes)}
    output_shared = np.asarray([output_lookup[gene] for gene in genes])
    strong = stable_output[matched_rows][:, output_shared]
    h1_control = h1_control_all[output_shared]
    features, mean_expression, _ = context_features(source_control, h1_control)
    outer = target_folds(matched)

    fold_tables = []
    per_target_tables = []
    summaries = []
    permutation_rows = []
    for variant, source, seed in (
        ("raw_shared_normalization", raw_residual, 20260922),
        ("count_shrunk_100000_shared_normalization", shrunk_residual, 20260923),
    ):
        predictions, folds, _ = crossfit_models(
            matched, source, target_residual, strong, features, outer, seed
        )
        folds.insert(0, "source_effect", variant)
        fold_tables.append(folds)
        per_target, summary = summarize_residuals(
            variant, predictions, source, target_residual, strong
        )
        per_target["target_gene"] = np.asarray(matched)[
            per_target.target_row.to_numpy(dtype=int)
        ]
        per_target_tables.append(per_target)
        summaries.append(summary)
        permutation, null = context_permutation_test(
            matched,
            source,
            target_residual,
            strong,
            features,
            mean_expression,
            outer,
            seed + 100,
        )
        permutation_rows.append(
            {"source_effect": variant, "permutations": len(null), **permutation}
        )

    pd.concat(fold_tables, ignore_index=True).to_csv(
        REPORT / "shared_normalization_fold_models.csv", index=False
    )
    pd.concat(per_target_tables, ignore_index=True).to_csv(
        REPORT / "shared_normalization_per_target.csv", index=False
    )
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(REPORT / "shared_normalization_summary.csv", index=False)
    permutation = pd.DataFrame(permutation_rows)
    permutation.to_csv(
        REPORT / "shared_normalization_context_permutation.csv", index=False
    )
    manifest = {
        "normalization": "K562 and H1 counts were each library-normalized within the 7,583-gene shared universe before residual construction",
        "scope": "effect-space sensitivity only; no additional cell candidates or fast scores were generated",
        "source_effects": [
            "raw shared-gene-normalized",
            "100,000-equivalent-UMI count shrinkage with shared-gene totals",
        ],
    }
    (REPORT / "shared_normalization_design.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(summary.to_string(index=False))
    print(permutation.to_string(index=False))


if __name__ == "__main__":
    main()
