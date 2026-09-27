"""Cross-fit perturbation-level K562-to-H1 residual scaling models."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.evaluation.crossfit_empirical_scale import target_folds
from scripts.evaluation.prepare_replogle_h1_phase1 import (
    evaluate,
    expected_realized_effects,
    truth_arrays,
)
from scripts.evaluation.prepare_replogle_h1_reliability import dirichlet_lfc
from scripts.evaluation.prepare_replogle_h1_transfer import loo_components

ROOT = Path(__file__).resolve().parents[2]
BULK = ROOT / "data/external/replogle2022/K562_gwps_raw_bulk_01.h5ad"
AGGREGATION = ROOT / "data/derived/replogle_h1_transfer/effects.npz"
PHASE1 = ROOT / "data/derived/replogle_h1_transfer/phase1_effects.npz"
FEATURES = ROOT / "reports/replogle-h1-transfer/phase2/oracle_scaling_per_target.csv"
DERIVED = ROOT / "data/derived/replogle_h1_transfer/crossfit_scale_effects.npz"
REPORT = ROOT / "reports/replogle-h1-transfer/crossfit-scale"
PRIOR_UMIS = 100_000.0
MIN_STRONG = 10
RIDGE_ALPHAS = np.r_[0.0, np.logspace(-4, 4, 17)]

FEATURE_COLUMNS = (
    "k562_target_control_log2_cpm1p",
    "h1_target_control_log2_cpm1p",
    "h1_minus_k562_target_control_log2_cpm1p",
    "k562_on_target_log2fc",
    "source_cells",
    "source_residual_norm",
)

ARMS = (
    "h1_global_only",
    "h1_global_raw_unscaled",
    "h1_global_raw_global_all",
    "h1_global_raw_global_strong",
    "h1_global_raw_ridge",
    "h1_global_shrunk_unscaled",
    "h1_global_shrunk_global_all",
    "h1_global_shrunk_global_strong",
    "h1_global_shrunk_ridge",
)


def projection_scale(source: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    x, y = source[valid], target[valid]
    denominator = float(x @ x)
    return float(x @ y / denominator) if denominator else np.nan


def fit_ridge(
    features: np.ndarray, outcome: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    medians = np.nanmedian(features, axis=0)
    filled = np.where(np.isfinite(features), features, medians)
    means = filled.mean(axis=0)
    scales = filled.std(axis=0)
    scales[scales == 0] = 1
    design = (filled - means) / scales
    centered = outcome - outcome.mean()
    beta = np.linalg.pinv(
        design.T @ design + alpha * np.eye(design.shape[1])
    ) @ (design.T @ centered)
    return medians, means, scales, float(outcome.mean()), beta


def predict_ridge(
    model: tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray],
    features: np.ndarray,
) -> np.ndarray:
    medians, means, scales, intercept, beta = model
    filled = np.where(np.isfinite(features), features, medians)
    return intercept + ((filled - means) / scales) @ beta


def choose_ridge_alpha(
    features: np.ndarray, outcome: np.ndarray, names: list[str], seed: int
) -> float:
    folds = target_folds(names, seed=seed)
    errors = np.zeros(len(RIDGE_ALPHAS))
    for fold in (0, 1):
        train, valid = folds != fold, folds == fold
        for index, alpha in enumerate(RIDGE_ALPHAS):
            prediction = predict_ridge(
                fit_ridge(features[train], outcome[train], alpha), features[valid]
            )
            errors[index] += np.square(prediction - outcome[valid]).sum()
    return float(RIDGE_ALPHAS[np.argmin(errors)])


def crossfit_scales(
    targets: list[str],
    source: np.ndarray,
    target: np.ndarray,
    strong: np.ndarray,
    features: np.ndarray,
    outer_folds: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    all_oracle = np.empty(len(targets))
    strong_oracle = np.full(len(targets), np.nan)
    strong_count = strong.sum(axis=1)
    for index in range(len(targets)):
        valid = np.ones(source.shape[1], dtype=bool)
        all_oracle[index] = projection_scale(source[index], target[index], valid)
        if strong_count[index] >= MIN_STRONG:
            strong_oracle[index] = projection_scale(
                source[index], target[index], strong[index]
            )

    predictions = {
        "unscaled": np.ones(len(targets)),
        "global_all": np.empty(len(targets)),
        "global_strong": np.empty(len(targets)),
        "ridge": np.empty(len(targets)),
    }
    fold_rows = []
    for heldout_fold in (0, 1):
        train = outer_folds != heldout_fold
        heldout = outer_folds == heldout_fold
        eligible_train = train & np.isfinite(strong_oracle)
        global_all = float(np.nanmean(all_oracle[train]))
        global_strong = float(np.nanmean(strong_oracle[eligible_train]))
        alpha = choose_ridge_alpha(
            features[eligible_train],
            strong_oracle[eligible_train],
            np.asarray(targets)[eligible_train].tolist(),
            seed + heldout_fold,
        )
        ridge = fit_ridge(
            features[eligible_train], strong_oracle[eligible_train], alpha
        )
        predictions["global_all"][heldout] = global_all
        predictions["global_strong"][heldout] = global_strong
        predictions["ridge"][heldout] = predict_ridge(ridge, features[heldout])
        fold_rows.append(
            {
                "heldout_fold": heldout_fold,
                "training_targets": int(train.sum()),
                "training_strong_scale_targets": int(eligible_train.sum()),
                "heldout_targets": int(heldout.sum()),
                "global_all_scale": global_all,
                "global_strong_scale": global_strong,
                "ridge_alpha": alpha,
            }
        )
    predictions["all_oracle"] = all_oracle
    predictions["strong_oracle"] = strong_oracle
    predictions["strong_count"] = strong_count
    return predictions, pd.DataFrame(fold_rows)


def build_intended(
    all_targets: list[str],
    matched: list[str],
    h1_global: np.ndarray,
    raw_residual: np.ndarray,
    shrunk_residual: np.ndarray,
    raw_scales: dict[str, np.ndarray],
    shrunk_scales: dict[str, np.ndarray],
) -> np.ndarray:
    matched_lookup = {target: index for index, target in enumerate(matched)}
    result = np.repeat(h1_global[None, :, :], len(ARMS), axis=0)
    specifications = (
        (1, raw_residual, raw_scales["unscaled"]),
        (2, raw_residual, raw_scales["global_all"]),
        (3, raw_residual, raw_scales["global_strong"]),
        (4, raw_residual, raw_scales["ridge"]),
        (5, shrunk_residual, shrunk_scales["unscaled"]),
        (6, shrunk_residual, shrunk_scales["global_all"]),
        (7, shrunk_residual, shrunk_scales["global_strong"]),
        (8, shrunk_residual, shrunk_scales["ridge"]),
    )
    for target_index, target in enumerate(all_targets):
        if target not in matched_lookup:
            continue
        source_index = matched_lookup[target]
        for arm_index, residual, scale in specifications:
            result[arm_index, target_index] += (
                scale[source_index] * residual[source_index]
            )
    return result


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    with np.load(AGGREGATION, allow_pickle=False) as saved:
        targets = saved["target_gene"].astype(str).tolist()
        matched = saved["matched_target_gene"].astype(str).tolist()
        shared_genes = saved["shared_gene"].astype(str).tolist()
        raw_source = saved["replogle_lfc_native"].astype(np.float64)
        target_counts = saved["replogle_count_sum"].astype(np.int64)
        control_counts = saved["replogle_control_count_sum"].astype(np.int64)
        h1 = saved["h1_lfc_native"].astype(np.float64)

    bulk = ad.read_h5ad(BULK, backed="r")
    try:
        symbols = bulk.var["gene_name"].astype(str).to_numpy()
    finally:
        bulk.file.close()
    unique_symbols = pd.unique(symbols).tolist()
    symbol_lookup = {gene: index for index, gene in enumerate(unique_symbols)}
    shared_indices = np.asarray([symbol_lookup[gene] for gene in shared_genes])
    shrunk_source = dirichlet_lfc(
        target_counts[:, shared_indices],
        target_counts.sum(axis=1),
        control_counts[shared_indices],
        control_counts.sum(),
        PRIOR_UMIS,
    )
    _, raw_residual = loo_components(raw_source)
    _, shrunk_residual = loo_components(shrunk_source)
    h1_global, h1_residual = loo_components(h1)
    target_lookup = {target: index for index, target in enumerate(targets)}
    matched_h1_rows = np.asarray([target_lookup[target] for target in matched])
    h1_residual_matched = h1_residual[matched_h1_rows]

    with np.load(PHASE1, allow_pickle=False) as saved:
        output_genes = saved["output_gene"].astype(str).tolist()
    truth, stable_output, de_valid = truth_arrays(targets, output_genes)
    output_lookup = {gene: index for index, gene in enumerate(output_genes)}
    output_shared = np.asarray([output_lookup[gene] for gene in shared_genes])
    strong_shared = stable_output[matched_h1_rows][:, output_shared]

    feature_frame = pd.read_csv(FEATURES).set_index("target_gene").loc[matched]
    base_features = feature_frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    outer_folds = target_folds(matched)
    raw_scales, raw_fold_table = crossfit_scales(
        matched,
        raw_residual,
        h1_residual_matched,
        strong_shared,
        base_features,
        outer_folds,
        seed=20260917,
    )
    shrunk_features = base_features.copy()
    shrunk_features[:, FEATURE_COLUMNS.index("source_residual_norm")] = np.linalg.norm(
        shrunk_residual, axis=1
    )
    shrunk_scales, shrunk_fold_table = crossfit_scales(
        matched,
        shrunk_residual,
        h1_residual_matched,
        strong_shared,
        shrunk_features,
        outer_folds,
        seed=20260918,
    )
    raw_fold_table.insert(0, "source_effect", "raw")
    shrunk_fold_table.insert(0, "source_effect", "count_shrunk_100000")
    pd.concat([raw_fold_table, shrunk_fold_table]).to_csv(
        REPORT / "fold_models.csv", index=False
    )

    scale_rows = []
    for variant, values in (("raw", raw_scales), ("count_shrunk_100000", shrunk_scales)):
        for index, target in enumerate(matched):
            scale_rows.append(
                {
                    "target_gene": target,
                    "outer_fold": int(outer_folds[index]),
                    "source_effect": variant,
                    **{name: vector[index] for name, vector in values.items()},
                }
            )
    scale_table = pd.DataFrame(scale_rows)
    scale_table.to_csv(REPORT / "per_target_scales.csv", index=False)

    intended = build_intended(
        targets,
        matched,
        h1_global,
        raw_residual,
        shrunk_residual,
        raw_scales,
        shrunk_scales,
    )
    realized, source_rows, full_gene_index = expected_realized_effects(
        intended, targets, output_genes, shared_genes, arm_names=ARMS
    )
    per_target, summary = evaluate(
        intended,
        realized,
        targets,
        output_genes,
        shared_genes,
        full_gene_index,
        set(matched),
        h1,
        truth,
        stable_output,
        de_valid,
        arm_names=ARMS,
    )
    per_target.to_csv(REPORT / "expected_per_target.csv", index=False)
    summary.to_csv(REPORT / "expected_summary.csv", index=False)

    scale_summary = []
    for variant, values in (("raw", raw_scales), ("count_shrunk_100000", shrunk_scales)):
        eligible = np.isfinite(values["strong_oracle"])
        for model in ("unscaled", "global_all", "global_strong", "ridge"):
            observed = values["strong_oracle"][eligible]
            predicted = values[model][eligible]
            correlation = spearmanr(predicted, observed).statistic
            scale_summary.append(
                {
                    "source_effect": variant,
                    "model": model,
                    "eligible_targets": int(eligible.sum()),
                    "strong_oracle_rmse": float(
                        np.sqrt(np.mean(np.square(predicted - observed)))
                    ),
                    "strong_oracle_mae": float(np.mean(np.abs(predicted - observed))),
                    "strong_oracle_spearman": (
                        float(correlation) if np.isfinite(correlation) else np.nan
                    ),
                    "predicted_scale_min": float(predicted.min()),
                    "predicted_scale_median": float(np.median(predicted)),
                    "predicted_scale_max": float(predicted.max()),
                }
            )
    scale_summary = pd.DataFrame(scale_summary)
    scale_summary.to_csv(REPORT / "scale_prediction_summary.csv", index=False)
    np.savez_compressed(
        DERIVED,
        arm=np.asarray(ARMS),
        target_gene=np.asarray(targets),
        matched_target_gene=np.asarray(matched),
        shared_gene=np.asarray(shared_genes),
        output_gene=np.asarray(output_genes),
        full_gene_index=full_gene_index,
        source_rows=source_rows,
        outer_fold=outer_folds,
        intended_lfc=intended.astype(np.float32),
        expected_realized_lfc=realized.astype(np.float32),
    )
    metadata = {
        "eligibility": "all arms are H1-informed diagnostics because they use the leave-one-target-out H1 global response; cross-fitted scales use H1 perturbation truth from training targets only",
        "outer_fold_seed": 20260916,
        "inner_tuning": "two-fold target cross-validation inside each outer training fold",
        "ridge_alpha_grid": RIDGE_ALPHAS.tolist(),
        "ridge_features": list(FEATURE_COLUMNS),
        "raw_source_effect": "primary planned analysis",
        "count_shrunk_source_effect": "post-hoc sensitivity using the previously examined 100,000-equivalent-UMI prior",
        "strong_scale_eligibility": f"at least {MIN_STRONG} stable strong H1 genes measurable in the source",
    }
    (REPORT / "design.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(pd.concat([raw_fold_table, shrunk_fold_table]).to_string(index=False))
    print(scale_summary.to_string(index=False))
    print(summary.query("population == 'all_126'").to_string(index=False))


if __name__ == "__main__":
    main()
