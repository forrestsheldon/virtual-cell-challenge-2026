"""Cross-fit response-gene scaling for K562-to-H1 residual transfer."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.evaluation.crossfit_empirical_scale import target_folds
from scripts.evaluation.crossfit_replogle_h1_scale import PRIOR_UMIS
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
DERIVED = ROOT / "data/derived/replogle_h1_transfer/crossfit_gene_scale_effects.npz"
REPORT = ROOT / "reports/replogle-h1-transfer/crossfit-gene-scale"

MIN_STRONG = 10
GENE_PRIOR_MULTIPLIERS = np.r_[0.0, np.logspace(-4, 4, 17), np.inf]
CONTEXT_RIDGE_MULTIPLIERS = np.r_[0.0, np.logspace(-8, 2, 16), np.inf]
N_BOOTSTRAPS = 500
N_PERMUTATIONS = 500

ARMS = (
    "h1_average_plus_raw_one_scale",
    "h1_average_plus_raw_gene_scales",
    "h1_average_plus_raw_expression_context",
    "h1_average_plus_shrunk_one_scale",
    "h1_average_plus_shrunk_gene_scales",
    "h1_average_plus_shrunk_expression_context",
)


def pooled_scale(source: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.square(source).sum())
    return float((source * target).sum() / denominator) if denominator else 0.0


def gene_scales(
    source: np.ndarray, target: np.ndarray, prior_multiplier: float
) -> tuple[np.ndarray, float, float]:
    """Shrink per-gene slopes toward the pooled slope by source-effect energy."""
    global_scale = pooled_scale(source, target)
    energy = np.square(source).sum(axis=0)
    positive = energy[energy > 0]
    reference_energy = float(np.median(positive)) if len(positive) else 1.0
    if np.isinf(prior_multiplier):
        return np.full(source.shape[1], global_scale), global_scale, np.inf
    prior_energy = prior_multiplier * reference_energy
    numerator = (source * target).sum(axis=0) + prior_energy * global_scale
    denominator = energy + prior_energy
    result = np.full(source.shape[1], global_scale)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result, global_scale, prior_energy


def context_features(
    k562_control_counts: np.ndarray,
    h1_control_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return standardized mean-expression and context-difference features."""
    k562 = np.log2(1 + 1_000_000 * k562_control_counts / k562_control_counts.sum())
    h1 = np.log2(1 + 1_000_000 * h1_control_counts / h1_control_counts.sum())
    mean = (h1 + k562) / 2
    difference = h1 - k562
    features = np.column_stack(
        [
            np.ones(len(mean)),
            (mean - mean.mean()) / mean.std(),
            (difference - difference.mean()) / difference.std(),
        ]
    )
    return features, mean, difference


def fit_context_model(
    source: np.ndarray,
    target: np.ndarray,
    features: np.ndarray,
    ridge_multiplier: float,
) -> tuple[np.ndarray, float]:
    """Fit y_pg = x_pg * feature_g @ beta with slopes ridge-shrunk to zero."""
    source_energy = np.square(source).sum(axis=0)
    cross_product = (source * target).sum(axis=0)
    xtx = features.T @ (source_energy[:, None] * features)
    xty = features.T @ cross_product
    if np.isinf(ridge_multiplier):
        beta = np.zeros(features.shape[1])
        beta[0] = pooled_scale(source, target)
        return beta, np.inf
    reference_energy = float(xtx[0, 0]) if xtx[0, 0] else 1.0
    penalty = ridge_multiplier * reference_energy
    regularizer = np.eye(features.shape[1]) * penalty
    regularizer[0, 0] = 0
    return np.linalg.pinv(xtx + regularizer) @ xty, penalty


def masked_scaled_squared_error(
    source: np.ndarray,
    scales: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, int]:
    """Evaluate only selected entries, avoiding dense validation predictions."""
    eligible = mask.sum(axis=1) >= MIN_STRONG
    rows, columns = np.nonzero(mask & eligible[:, None])
    residual = source[rows, columns] * scales[columns] - target[rows, columns]
    return float(residual @ residual), len(residual)


def inner_folds(names: list[str], seed: int) -> np.ndarray:
    return target_folds(names, seed=seed)


def choose_gene_prior(
    names: list[str],
    source: np.ndarray,
    target: np.ndarray,
    strong: np.ndarray,
    seed: int,
) -> tuple[float, np.ndarray]:
    folds = inner_folds(names, seed)
    errors = np.zeros(len(GENE_PRIOR_MULTIPLIERS))
    counts = np.zeros(len(GENE_PRIOR_MULTIPLIERS), dtype=int)
    for fold in (0, 1):
        train, valid = folds != fold, folds == fold
        for index, multiplier in enumerate(GENE_PRIOR_MULTIPLIERS):
            scales, _, _ = gene_scales(source[train], target[train], multiplier)
            error, count = masked_scaled_squared_error(
                source[valid], scales, target[valid], strong[valid]
            )
            errors[index] += error
            counts[index] += count
    losses = np.divide(errors, counts, out=np.full_like(errors, np.inf), where=counts > 0)
    return float(GENE_PRIOR_MULTIPLIERS[np.argmin(losses)]), losses


def choose_context_ridge(
    names: list[str],
    source: np.ndarray,
    target: np.ndarray,
    strong: np.ndarray,
    features: np.ndarray,
    seed: int,
) -> tuple[float, np.ndarray]:
    folds = inner_folds(names, seed)
    errors = np.zeros(len(CONTEXT_RIDGE_MULTIPLIERS))
    counts = np.zeros(len(CONTEXT_RIDGE_MULTIPLIERS), dtype=int)
    for fold in (0, 1):
        train, valid = folds != fold, folds == fold
        for index, multiplier in enumerate(CONTEXT_RIDGE_MULTIPLIERS):
            beta, _ = fit_context_model(
                source[train], target[train], features, multiplier
            )
            error, count = masked_scaled_squared_error(
                source[valid], features @ beta, target[valid], strong[valid]
            )
            errors[index] += error
            counts[index] += count
    losses = np.divide(errors, counts, out=np.full_like(errors, np.inf), where=counts > 0)
    return float(CONTEXT_RIDGE_MULTIPLIERS[np.argmin(losses)]), losses


def crossfit_models(
    names: list[str],
    source: np.ndarray,
    target: np.ndarray,
    strong: np.ndarray,
    features: np.ndarray,
    outer: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[int, dict[str, object]]]:
    predictions = {
        "one_scale": np.empty_like(source),
        "gene_scales": np.empty_like(source),
        "expression_context": np.empty_like(source),
    }
    rows = []
    models: dict[int, dict[str, object]] = {}
    name_array = np.asarray(names)
    for heldout_fold in (0, 1):
        train, heldout = outer != heldout_fold, outer == heldout_fold
        prior_multiplier, prior_curve = choose_gene_prior(
            name_array[train].tolist(),
            source[train],
            target[train],
            strong[train],
            seed + 10 * heldout_fold,
        )
        ridge_multiplier, ridge_curve = choose_context_ridge(
            name_array[train].tolist(),
            source[train],
            target[train],
            strong[train],
            features,
            seed + 10 * heldout_fold + 1,
        )
        scales, global_scale, prior_energy = gene_scales(
            source[train], target[train], prior_multiplier
        )
        beta, ridge_penalty = fit_context_model(
            source[train], target[train], features, ridge_multiplier
        )
        predictions["one_scale"][heldout] = source[heldout] * global_scale
        predictions["gene_scales"][heldout] = source[heldout] * scales
        predictions["expression_context"][heldout] = source[heldout] * (
            features @ beta
        )
        rows.append(
            {
                "heldout_fold": heldout_fold,
                "training_targets": int(train.sum()),
                "heldout_targets": int(heldout.sum()),
                "global_scale": global_scale,
                "gene_prior_multiplier": prior_multiplier,
                "gene_prior_energy": prior_energy,
                "context_ridge_multiplier": ridge_multiplier,
                "context_ridge_penalty": ridge_penalty,
                "context_intercept": beta[0],
                "context_mean_expression_per_sd": beta[1],
                "context_h1_minus_k562_per_sd": beta[2],
            }
        )
        models[heldout_fold] = {
            "training_mask": train,
            "gene_scales": scales,
            "global_scale": global_scale,
            "gene_prior_multiplier": prior_multiplier,
            "gene_prior_curve": prior_curve,
            "context_beta": beta,
            "context_ridge_multiplier": ridge_multiplier,
            "context_ridge_curve": ridge_curve,
        }
    return predictions, pd.DataFrame(rows), models


def build_intended(
    all_targets: list[str],
    matched: list[str],
    h1_global: np.ndarray,
    raw_predictions: dict[str, np.ndarray],
    shrunk_predictions: dict[str, np.ndarray],
) -> np.ndarray:
    result = np.repeat(h1_global[None, :, :], len(ARMS), axis=0)
    target_lookup = {target: index for index, target in enumerate(all_targets)}
    specifications = (
        (0, raw_predictions["one_scale"]),
        (1, raw_predictions["gene_scales"]),
        (2, raw_predictions["expression_context"]),
        (3, shrunk_predictions["one_scale"]),
        (4, shrunk_predictions["gene_scales"]),
        (5, shrunk_predictions["expression_context"]),
    )
    for source_row, target in enumerate(matched):
        output_row = target_lookup[target]
        for arm, prediction in specifications:
            result[arm, output_row] += prediction[source_row]
    return result


def bootstrap_models(
    source: np.ndarray,
    target: np.ndarray,
    features: np.ndarray,
    models: dict[int, dict[str, object]],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    coefficient_rows = []
    gene_frames = []
    for heldout_fold, model in models.items():
        rows = np.flatnonzero(model["training_mask"])
        beta_samples = np.empty((N_BOOTSTRAPS, features.shape[1]))
        gene_samples = np.empty((N_BOOTSTRAPS, source.shape[1]), dtype=np.float32)
        global_samples = np.empty(N_BOOTSTRAPS)
        for bootstrap in range(N_BOOTSTRAPS):
            sampled = rng.choice(rows, len(rows), replace=True)
            scales, global_scale, _ = gene_scales(
                source[sampled],
                target[sampled],
                float(model["gene_prior_multiplier"]),
            )
            beta, _ = fit_context_model(
                source[sampled],
                target[sampled],
                features,
                float(model["context_ridge_multiplier"]),
            )
            gene_samples[bootstrap] = scales
            global_samples[bootstrap] = global_scale
            beta_samples[bootstrap] = beta
        for column, name in enumerate(
            ("intercept", "mean_expression_per_sd", "h1_minus_k562_per_sd")
        ):
            low, median, high = np.quantile(beta_samples[:, column], (0.025, 0.5, 0.975))
            coefficient_rows.append(
                {
                    "heldout_fold": heldout_fold,
                    "coefficient": name,
                    "q025": low,
                    "median": median,
                    "q975": high,
                    "positive_fraction": float((beta_samples[:, column] > 0).mean()),
                }
            )
        centered = gene_samples - global_samples[:, None]
        gene_frames.append(
            pd.DataFrame(
                {
                    "heldout_fold": heldout_fold,
                    "bootstrap_deviation_q025": np.quantile(centered, 0.025, axis=0),
                    "bootstrap_deviation_median": np.median(centered, axis=0),
                    "bootstrap_deviation_q975": np.quantile(centered, 0.975, axis=0),
                    "bootstrap_deviation_positive_fraction": (centered > 0).mean(axis=0),
                }
            )
        )
    return pd.DataFrame(coefficient_rows), pd.concat(gene_frames, ignore_index=True)


def permute_within_deciles(
    values: np.ndarray, mean_expression: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    bins = pd.qcut(mean_expression, 10, labels=False, duplicates="drop")
    result = values.copy()
    for bin_index in np.unique(bins):
        rows = np.flatnonzero(bins == bin_index)
        result[rows] = rng.permutation(values[rows])
    return result


def crossfit_context_loss(
    names: list[str],
    source: np.ndarray,
    target: np.ndarray,
    strong: np.ndarray,
    features: np.ndarray,
    outer: np.ndarray,
    seed: int,
) -> float:
    total_error, total_count = 0.0, 0
    name_array = np.asarray(names)
    for heldout_fold in (0, 1):
        train, heldout = outer != heldout_fold, outer == heldout_fold
        ridge, _ = choose_context_ridge(
            name_array[train].tolist(),
            source[train],
            target[train],
            strong[train],
            features,
            seed + heldout_fold,
        )
        beta, _ = fit_context_model(source[train], target[train], features, ridge)
        error, count = masked_scaled_squared_error(
            source[heldout], features @ beta, target[heldout], strong[heldout]
        )
        total_error += error
        total_count += count
    return total_error / total_count


def context_permutation_test(
    names: list[str],
    source: np.ndarray,
    target: np.ndarray,
    strong: np.ndarray,
    features: np.ndarray,
    mean_expression: np.ndarray,
    outer: np.ndarray,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    mean_only = features[:, :2]
    reference_loss = crossfit_context_loss(
        names, source, target, strong, mean_only, outer, seed
    )
    observed_loss = crossfit_context_loss(
        names, source, target, strong, features, outer, seed + 100
    )
    observed_improvement = reference_loss - observed_loss
    rng = np.random.default_rng(seed + 200)
    null = np.empty(N_PERMUTATIONS)
    for permutation in range(N_PERMUTATIONS):
        permuted = permute_within_deciles(features[:, 2], mean_expression, rng)
        permuted_features = np.column_stack([features[:, :2], permuted])
        loss = crossfit_context_loss(
            names,
            source,
            target,
            strong,
            permuted_features,
            outer,
            seed + 1000 + permutation,
        )
        null[permutation] = reference_loss - loss
    return (
        {
            "mean_expression_only_strong_residual_mse": reference_loss,
            "observed_full_context_strong_residual_mse": observed_loss,
            "observed_difference_feature_improvement": observed_improvement,
            "null_improvement_mean": float(null.mean()),
            "null_improvement_q95": float(np.quantile(null, 0.95)),
            "one_sided_permutation_p": float(
                (1 + np.sum(null >= observed_improvement)) / (N_PERMUTATIONS + 1)
            ),
        },
        null,
    )


def gene_stability_table(
    genes: list[str],
    source: np.ndarray,
    features: np.ndarray,
    mean_expression: np.ndarray,
    difference: np.ndarray,
    models: dict[int, dict[str, object]],
    bootstrap: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    table = pd.DataFrame(
        {
            "gene": genes,
            "mean_control_log2_cpm1p": mean_expression,
            "h1_minus_k562_control_log2_cpm1p": difference,
            "mean_expression_z": features[:, 1],
            "context_difference_z": features[:, 2],
            "source_energy_all_targets": np.square(source).sum(axis=0),
            "scale_trained_for_fold0": models[0]["gene_scales"],
            "scale_trained_for_fold1": models[1]["gene_scales"],
            "global_trained_for_fold0": models[0]["global_scale"],
            "global_trained_for_fold1": models[1]["global_scale"],
        }
    )
    for fold in (0, 1):
        selected = bootstrap.query("heldout_fold == @fold").reset_index(drop=True)
        for column in selected.columns:
            if column != "heldout_fold":
                table[f"fold{fold}_{column}"] = selected[column].to_numpy()
    first = table.scale_trained_for_fold0.to_numpy()
    second = table.scale_trained_for_fold1.to_numpy()
    first_deviation = first - float(models[0]["global_scale"])
    second_deviation = second - float(models[1]["global_scale"])
    energy = table.source_energy_all_targets.to_numpy()
    high_energy = energy >= np.median(energy[energy > 0])
    same_direction = np.sign(first_deviation) == np.sign(second_deviation)
    stable = high_energy & same_direction & (
        (
            (table.fold0_bootstrap_deviation_positive_fraction >= 0.95)
            & (table.fold1_bootstrap_deviation_positive_fraction >= 0.95)
        )
        | (
            (table.fold0_bootstrap_deviation_positive_fraction <= 0.05)
            & (table.fold1_bootstrap_deviation_positive_fraction <= 0.05)
        )
    )
    table["stable_same_side_of_global"] = stable
    all_correlation = spearmanr(first, second).statistic
    high_energy_correlation = spearmanr(
        first_deviation[high_energy], second_deviation[high_energy]
    ).statistic
    summary = {
        "all_gene_scale_spearman": (
            float(all_correlation) if np.isfinite(all_correlation) else None
        ),
        "high_energy_deviation_spearman": (
            float(high_energy_correlation)
            if np.isfinite(high_energy_correlation)
            else None
        ),
        "high_energy_deviation_same_direction_fraction": float(
            same_direction[high_energy].mean()
        ),
        "stable_same_side_of_global_genes": int(stable.sum()),
        "high_energy_genes": int(high_energy.sum()),
    }
    return table, summary


def source_control_counts(shared_genes: list[str]) -> np.ndarray:
    bulk = ad.read_h5ad(BULK, backed="r")
    try:
        symbols = bulk.var["gene_name"].astype(str).to_numpy()
    finally:
        bulk.file.close()
    unique = pd.unique(symbols).tolist()
    lookup = {gene: index for index, gene in enumerate(unique)}
    with np.load(AGGREGATION, allow_pickle=False) as saved:
        counts = saved["replogle_control_count_sum"].astype(np.float64)
    return counts[np.asarray([lookup[gene] for gene in shared_genes])]


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    with np.load(AGGREGATION, allow_pickle=False) as saved:
        targets = saved["target_gene"].astype(str).tolist()
        matched = saved["matched_target_gene"].astype(str).tolist()
        genes = saved["shared_gene"].astype(str).tolist()
        raw_source = saved["replogle_lfc_native"].astype(np.float64)
        target_counts = saved["replogle_count_sum"].astype(np.int64)
        control_counts = saved["replogle_control_count_sum"].astype(np.int64)
        h1_control_all = saved["h1_control_count_sum"].astype(np.float64)
        h1 = saved["h1_lfc_native"].astype(np.float64)

    bulk = ad.read_h5ad(BULK, backed="r")
    try:
        symbols = bulk.var["gene_name"].astype(str).to_numpy()
    finally:
        bulk.file.close()
    unique_symbols = pd.unique(symbols).tolist()
    source_lookup = {gene: index for index, gene in enumerate(unique_symbols)}
    shared_source = np.asarray([source_lookup[gene] for gene in genes])
    shrunk_source = dirichlet_lfc(
        target_counts[:, shared_source],
        target_counts.sum(axis=1),
        control_counts[shared_source],
        control_counts.sum(),
        PRIOR_UMIS,
    )
    _, raw_residual = loo_components(raw_source)
    _, shrunk_residual = loo_components(shrunk_source)
    h1_global, h1_residual = loo_components(h1)
    target_lookup = {target: index for index, target in enumerate(targets)}
    matched_rows = np.asarray([target_lookup[target] for target in matched])
    target_residual = h1_residual[matched_rows]

    with np.load(PHASE1, allow_pickle=False) as saved:
        output_genes = saved["output_gene"].astype(str).tolist()
    truth, stable_output, de_valid = truth_arrays(targets, output_genes)
    output_lookup = {gene: index for index, gene in enumerate(output_genes)}
    output_shared = np.asarray([output_lookup[gene] for gene in genes])
    strong = stable_output[matched_rows][:, output_shared]

    source_control = source_control_counts(genes)
    h1_control = h1_control_all[output_shared]
    features, mean_expression, difference = context_features(
        source_control, h1_control
    )
    outer = target_folds(matched)

    all_predictions = {}
    all_models = {}
    fold_tables = []
    gene_tables = []
    bootstrap_tables = []
    permutation_rows = []
    stability = {}
    for variant, source, seed in (
        ("raw", raw_residual, 20260920),
        ("count_shrunk_100000", shrunk_residual, 20260921),
    ):
        predictions, folds, models = crossfit_models(
            matched, source, target_residual, strong, features, outer, seed
        )
        folds.insert(0, "source_effect", variant)
        fold_tables.append(folds)
        all_predictions[variant] = predictions
        all_models[variant] = models

        bootstrap_beta, bootstrap_gene = bootstrap_models(
            source, target_residual, features, models, seed + 100
        )
        bootstrap_beta.insert(0, "source_effect", variant)
        bootstrap_tables.append(bootstrap_beta)
        gene_table, stability_summary = gene_stability_table(
            genes,
            source,
            features,
            mean_expression,
            difference,
            models,
            bootstrap_gene,
        )
        gene_table.insert(0, "source_effect", variant)
        gene_tables.append(gene_table)
        stability[variant] = stability_summary

        permutation, null = context_permutation_test(
            matched,
            source,
            target_residual,
            strong,
            features,
            mean_expression,
            outer,
            seed + 200,
        )
        permutation_rows.append(
            {"source_effect": variant, "permutations": len(null), **permutation}
        )

    fold_table = pd.concat(fold_tables, ignore_index=True)
    fold_table.to_csv(REPORT / "fold_models.csv", index=False)
    pd.concat(bootstrap_tables, ignore_index=True).to_csv(
        REPORT / "context_coefficient_bootstrap.csv", index=False
    )
    pd.concat(gene_tables, ignore_index=True).to_csv(
        REPORT / "per_gene_coefficients.csv", index=False
    )
    pd.DataFrame(permutation_rows).to_csv(
        REPORT / "context_difference_permutation.csv", index=False
    )

    intended = build_intended(
        targets,
        matched,
        h1_global,
        all_predictions["raw"],
        all_predictions["count_shrunk_100000"],
    )
    realized, source_rows, full_gene_index = expected_realized_effects(
        intended, targets, output_genes, genes, arm_names=ARMS
    )
    per_target, summary = evaluate(
        intended,
        realized,
        targets,
        output_genes,
        genes,
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
    np.savez_compressed(
        DERIVED,
        arm=np.asarray(ARMS),
        target_gene=np.asarray(targets),
        matched_target_gene=np.asarray(matched),
        shared_gene=np.asarray(genes),
        output_gene=np.asarray(output_genes),
        full_gene_index=full_gene_index,
        source_rows=source_rows,
        outer_fold=outer,
        intended_lfc=intended.astype(np.float32),
        expected_realized_lfc=realized.astype(np.float32),
    )
    design = {
        "eligibility": "all arms are H1-informed diagnostics and ineligible for final zero-shot selection",
        "outer_target_fold_seed": 20260916,
        "inner_tuning_metric": "mean squared H1 residual error on stable strong-DE genes from training targets only",
        "gene_prior_multipliers": [
            None if np.isinf(value) else float(value)
            for value in GENE_PRIOR_MULTIPLIERS
        ],
        "context_ridge_multipliers": [
            None if np.isinf(value) else float(value)
            for value in CONTEXT_RIDGE_MULTIPLIERS
        ],
        "response_gene_features": [
            "intercept",
            "standardized mean K562/H1 control log2(CPM+1)",
            "standardized H1-minus-K562 control log2(CPM+1)",
        ],
        "context_difference_null": f"{N_PERMUTATIONS} permutations within mean-expression deciles, with inner tuning repeated",
        "coefficient_stability": f"{N_BOOTSTRAPS} perturbation bootstraps within each outer training fold",
        "raw_source_effect": "primary planned analysis",
        "count_shrunk_source_effect": "post-hoc sensitivity using a 100,000-equivalent-UMI prior",
        "gene_scale_stability": stability,
    }
    (REPORT / "design.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n"
    )
    print(fold_table.to_string(index=False))
    print(pd.DataFrame(permutation_rows).to_string(index=False))
    print(json.dumps(stability, indent=2, sort_keys=True))
    print(summary.query("population == 'all_126'").to_string(index=False))


if __name__ == "__main__":
    main()
