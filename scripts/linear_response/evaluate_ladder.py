"""Evaluate the restricted linear-response ladder on H1 truth."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl

from scripts.linear_response.kernel import (
    decoder_lfc_tangents,
    expected_restricted_cpm,
    nonnegative_l1_scale,
)
from scripts.linear_response.strong_de_recovery import (
    derangement_null,
    pairwise_wrong_target_scores,
    randomized_gene_null,
    ranked_genes,
    signed_topk,
)

ROOT = Path(__file__).resolve().parents[2]
H1 = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
DE = ROOT / "data/derived/linear_response/eval_cache/de_wilcoxon_table-12f179b9e966af64.parquet"
DERIVED = ROOT / "data/derived/linear_response/ladder"
REPORT = ROOT / "reports/linear-response-ladder"
STRONG = ROOT / "reports/linear-response-three-models/strong_de_truth_genes.csv"
TARGET_COUNTS = ROOT / "reports/linear-response-three-models/h1_target_counts.csv"
EMPIRICAL = DERIVED / "empirical.npz"
SPARSE = DERIVED / "sparse.npz"
FACTOR = DERIVED / "factor.npz"
N_NULL = 10_000
PRIMARY_MODELS = ("empirical", "sparse", "factor_selected")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_truth(
    targets: list[str], genes: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    frame = pl.read_parquet(DE).to_pandas()
    truth = np.empty((len(targets), len(genes)))
    for index, target in enumerate(targets):
        group = frame.loc[frame["target"] == target].set_index("feature")
        truth[index] = group.loc[genes, "log2_fold_change"]
    stable = np.zeros_like(truth, dtype=bool)
    target_lookup = {target: index for index, target in enumerate(targets)}
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    table = pd.read_csv(STRONG)
    for row in table.loc[table["stable_strong"]].itertuples(index=False):
        stable[target_lookup[row.target_gene], gene_lookup[row.feature]] = True
    for index, target in enumerate(targets):
        if target in gene_lookup:
            stable[index, gene_lookup[target]] = False
    return truth, stable


def load_models() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(EMPIRICAL, allow_pickle=False) as empirical, np.load(
        SPARSE, allow_pickle=False
    ) as sparse, np.load(FACTOR, allow_pickle=False) as factor:
        metadata = {
            key: empirical[key]
            for key in [
                "target_gene",
                "output_gene",
                "fit_gene",
                "full_gene_index",
                "target_fit_index",
                "source_rows",
            ]
        }
        ranks = factor["ranks"].astype(int)
        selected = int(factor["selected_rank"])
        models = {
            "empirical": empirical["response"].astype(np.float64),
            "empirical_all31": empirical["all31_response"].astype(np.float64),
            "empirical_raw": empirical["raw_response"].astype(np.float64),
            "sparse": sparse["response"].astype(np.float64),
            **{
                f"factor_rank{rank}": factor["response"][index].astype(np.float64)
                for index, rank in enumerate(ranks)
            },
        }
    models["factor_selected"] = models[f"factor_rank{selected}"]
    return models, metadata


def tangent_predictions(
    models: dict[str, np.ndarray], metadata: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    checkpoint = DERIVED / "tangent_predictions.npz"
    signature = "|".join(sha256(path) for path in [EMPIRICAL, SPARSE, FACTOR])
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if (
                saved["model"].astype(str).tolist() == list(models)
                and str(saved["input_signature"]) == signature
            ):
                return {
                    name: saved["tangent"][index].astype(np.float64)
                    for index, name in enumerate(models)
                }

    indices = metadata["full_gene_index"].astype(int)
    output_count = len(metadata["output_gene"])
    predictions = {
        name: np.empty((len(metadata["target_gene"]), output_count))
        for name in models
    }
    data = ad.read_h5ad(H1, backed="r")
    try:
        for target, rows in enumerate(metadata["source_rows"]):
            raw_full = data.X[np.sort(rows)].tocsr()
            totals = np.asarray(raw_full.sum(axis=1)).ravel()
            raw = raw_full[:, indices]
            directions = np.vstack(
                [response[target] for response in models.values()]
            )
            values = decoder_lfc_tangents(
                raw, directions, full_totals=totals
            )[:, :output_count]
            for index, name in enumerate(models):
                predictions[name][target] = values[index]
            print(f"Ladder tangents {target + 1}/{len(metadata['target_gene'])}")
    finally:
        data.file.close()

    np.savez_compressed(
        checkpoint,
        model=np.asarray(list(models)),
        tangent=np.asarray(list(predictions.values()), dtype=np.float32),
        input_signature=np.asarray(signature),
    )
    return predictions


def scales(
    predictions: dict[str, np.ndarray], truth: np.ndarray, stable: np.ndarray
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    rows = []
    transferred = {}
    oracle_by_model = {}
    eligible = stable.sum(axis=1) >= 10
    for name, prediction in predictions.items():
        oracle = np.full(len(truth), np.nan)
        for target in np.flatnonzero(eligible):
            oracle[target] = nonnegative_l1_scale(
                prediction[target], truth[target], stable[target]
            )
        loo = np.empty(len(truth))
        for target in range(len(truth)):
            donors = eligible.copy()
            donors[target] = False
            loo[target] = np.median(oracle[donors]) if donors.any() else np.nan
        transferred[name] = loo
        oracle_by_model[name] = oracle
        for target in range(len(truth)):
            rows.append(
                {
                    "model": name,
                    "target_index": target,
                    "eligible": eligible[target],
                    "oracle_gamma": oracle[target],
                    "loo_median_gamma": loo[target],
                    "oracle_at_zero": bool(eligible[target] and oracle[target] == 0),
                }
            )
    return pd.DataFrame(rows), transferred, oracle_by_model


def exact_predictions(
    models: dict[str, np.ndarray],
    scales_by_model: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
    checkpoint_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    checkpoint = DERIVED / checkpoint_name
    scale_matrix = np.asarray([scales_by_model[name] for name in models])
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if (
                saved["model"].astype(str).tolist() == list(models)
                and np.allclose(saved["gamma"], scale_matrix)
            ):
                exact = {
                    name: saved["lfc"][index].astype(np.float64)
                    for index, name in enumerate(models)
                }
                larger = {
                    name: saved["larger_lfc"][index].astype(np.float64)
                    for index, name in enumerate(saved["larger_model"].astype(str))
                }
                return exact, larger

    indices = metadata["full_gene_index"].astype(int)
    output_count = len(metadata["output_gene"])
    predictions = {
        name: np.empty((len(metadata["target_gene"]), output_count))
        for name in models
    }
    larger = {
        name: np.empty((len(metadata["target_gene"]), output_count))
        for name in PRIMARY_MODELS
    }
    data = ad.read_h5ad(H1, backed="r")
    try:
        for target, rows in enumerate(metadata["source_rows"]):
            raw_full = data.X[np.sort(rows)].tocsr()
            totals = np.asarray(raw_full.sum(axis=1)).ravel()
            raw = raw_full[:, indices]
            baseline = expected_restricted_cpm(
                raw, totals, np.zeros(len(indices)), 0.0
            )
            for name, response in models.items():
                direction = response[target]
                shifted = expected_restricted_cpm(
                    raw, totals, direction, scales_by_model[name][target]
                )
                predictions[name][target] = np.log2(
                    (shifted[:output_count] + 1e-9)
                    / (baseline[:output_count] + 1e-9)
                )
                if name in PRIMARY_MODELS:
                    shifted_larger = expected_restricted_cpm(
                        raw,
                        totals,
                        direction,
                        1.5 * scales_by_model[name][target],
                    )
                    larger[name][target] = np.log2(
                        (shifted_larger[:output_count] + 1e-9)
                        / (baseline[:output_count] + 1e-9)
                    )
            print(f"Exact ladder profiles {target + 1}/{len(metadata['target_gene'])}")
    finally:
        data.file.close()

    np.savez_compressed(
        checkpoint,
        model=np.asarray(list(models)),
        lfc=np.asarray(list(predictions.values()), dtype=np.float32),
        gamma=scale_matrix,
        larger_model=np.asarray(list(larger)),
        larger_lfc=np.asarray(list(larger.values()), dtype=np.float32),
    )
    return predictions, larger


def refine_scales_exact(
    names: list[str],
    models: dict[str, np.ndarray],
    oracle_by_model: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
    truth: np.ndarray,
    stable: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Transfer exact-grid donor optima without reading the held target's truth."""
    factors = np.asarray([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    indices = metadata["full_gene_index"].astype(int)
    output_count = len(metadata["output_gene"])
    refined_oracle = {name: np.full(len(truth), np.nan) for name in names}
    rows = []
    data = ad.read_h5ad(H1, backed="r")
    try:
        for target, source_rows in enumerate(metadata["source_rows"]):
            if stable[target].sum() < 10:
                continue
            raw_full = data.X[np.sort(source_rows)].tocsr()
            totals = np.asarray(raw_full.sum(axis=1)).ravel()
            raw = raw_full[:, indices]
            baseline = expected_restricted_cpm(
                raw, totals, np.zeros(len(indices)), 0.0
            )[:output_count]
            for name in names:
                center = oracle_by_model[name][target]
                candidates = center * factors if center > 0 else np.asarray([0.0])
                errors = []
                for gamma in candidates:
                    shifted = expected_restricted_cpm(
                        raw, totals, models[name][target], gamma
                    )[:output_count]
                    prediction = np.log2((shifted + 1e-9) / (baseline + 1e-9))
                    errors.append(nmae(prediction, truth[target], stable[target]))
                selected = int(np.argmin(errors))
                refined_oracle[name][target] = candidates[selected]
                rows.append(
                    {
                        "model": name,
                        "target_index": target,
                        "linear_oracle_gamma": center,
                        "exact_oracle_gamma": candidates[selected],
                        "selected_factor": factors[selected] if center > 0 else 0.0,
                        "exact_oracle_nmae": errors[selected],
                        "grid_boundary": bool(
                            center > 0 and selected in {0, len(candidates) - 1}
                        ),
                    }
                )
    finally:
        data.file.close()

    transferred = {}
    for name in names:
        values = refined_oracle[name]
        transferred[name] = np.asarray(
            [np.nanmedian(np.delete(values, target)) for target in range(len(values))]
        )
    return transferred, pd.DataFrame(rows)


def nmae(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    return float(
        np.abs(prediction[mask] - truth[mask]).sum()
        / np.abs(truth[mask]).sum()
    )


def holm(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values)
    order = np.argsort(values)
    adjusted = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * values[order]
    )
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1)
    return result


def score_directions(
    predictions: dict[str, np.ndarray],
    exact: dict[str, np.ndarray],
    larger: dict[str, np.ndarray],
    scales_by_model: dict[str, np.ndarray],
    oracle_by_model: dict[str, np.ndarray],
    truth: np.ndarray,
    stable: np.ndarray,
    targets: list[str],
    genes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_gene_indices = [
        genes.index(target) if target in genes else None for target in targets
    ]
    truth_sign = np.sign(truth).astype(np.int8)
    per_target = []
    summaries = []
    for model_index, (name, prediction) in enumerate(predictions.items()):
        ranking = ranked_genes(prediction)
        pairwise = pairwise_wrong_target_scores(
            prediction, ranking, stable, truth_sign, target_gene_indices
        )
        observed = np.full(len(targets), np.nan)
        model_rows = []
        for target_index, target in enumerate(targets):
            result = signed_topk(
                prediction[target_index],
                stable[target_index],
                truth_sign[target_index],
                target_gene_indices[target_index],
            )
            eligible = int(result["n_strong"]) >= 10
            if eligible:
                observed[target_index] = result["signed_recovery"]
            wrong = np.delete(pairwise[target_index], target_index)
            wrong = wrong[np.isfinite(wrong)]
            model_rows.append(
                {
                    "model": name,
                    "target_gene": target,
                    "eligible": eligible,
                    **result,
                    "strong_energy_fraction": 1 - result["non_strong_energy_fraction"]
                    if eligible
                    else np.nan,
                    "wrong_target_mean": wrong.mean() if eligible else np.nan,
                    "signed_recovery_excess_wrong": result["signed_recovery"]
                    - wrong.mean()
                    if eligible
                    else np.nan,
                    "truth_strong_lfc_energy": np.square(
                        truth[target_index, stable[target_index]]
                    ).sum(),
                    "loo_median_gamma": scales_by_model[name][target_index],
                    "oracle_gamma": oracle_by_model[name][target_index],
                    "oracle_linear_nmae": nmae(
                        oracle_by_model[name][target_index]
                        * prediction[target_index],
                        truth[target_index],
                        stable[target_index],
                    )
                    if eligible
                    else np.nan,
                    "linear_crossfit_nmae": nmae(
                        scales_by_model[name][target_index] * prediction[target_index],
                        truth[target_index],
                        stable[target_index],
                    )
                    if eligible
                    else np.nan,
                    "exact_crossfit_nmae": nmae(
                        exact[name][target_index],
                        truth[target_index],
                        stable[target_index],
                    )
                    if eligible
                    else np.nan,
                }
            )
        per_target.extend(model_rows)
        wrong_null = derangement_null(
            pairwise, np.random.default_rng([0, 81, model_index])
        )
        gene_null = randomized_gene_null(
            prediction,
            ranking,
            stable,
            truth_sign,
            target_gene_indices,
            np.random.default_rng([0, 82, model_index]),
        )
        finite = np.isfinite(observed)
        deltas = np.asarray(
            [row["signed_recovery_excess_wrong"] for row in model_rows if row["eligible"]]
        )
        bootstrap = np.random.default_rng([0, 83, model_index]).choice(
            deltas, size=(N_NULL, len(deltas)), replace=True
        ).mean(axis=1)
        observed_mean = float(np.nanmean(observed))
        summaries.append(
            {
                "model": name,
                "eligible_targets": int(finite.sum()),
                "mean_signed_recovery": observed_mean,
                "mean_unsigned_recovery": np.nanmean(
                    [row["unsigned_recall"] for row in model_rows]
                ),
                "mean_sign_given_recovered": np.nanmean(
                    [row["sign_given_recovered"] for row in model_rows]
                ),
                "mean_strong_energy_fraction": np.nanmean(
                    [row["strong_energy_fraction"] for row in model_rows]
                ),
                "wrong_target_mean": wrong_null.mean(),
                "wrong_target_p": (1 + np.sum(wrong_null >= observed_mean))
                / (N_NULL + 1),
                "random_gene_mean": gene_null.mean(),
                "random_gene_p": (1 + np.sum(gene_null >= observed_mean))
                / (N_NULL + 1),
                "excess_wrong_bootstrap_q025": np.quantile(bootstrap, 0.025),
                "excess_wrong_bootstrap_q975": np.quantile(bootstrap, 0.975),
                "median_crossfit_gamma": np.median(scales_by_model[name]),
                "mean_linear_crossfit_nmae": np.nanmean(
                    [row["linear_crossfit_nmae"] for row in model_rows]
                ),
                "mean_oracle_linear_nmae": np.nanmean(
                    [row["oracle_linear_nmae"] for row in model_rows]
                ),
                "mean_exact_crossfit_nmae": np.nanmean(
                    [row["exact_crossfit_nmae"] for row in model_rows]
                ),
                "mean_abs_selected_linearization_error": np.nanmean(
                    [
                        abs(row["exact_crossfit_nmae"] - row["linear_crossfit_nmae"])
                        for row in model_rows
                    ]
                ),
            }
        )
        if name in larger:
            larger_differences = []
            for target in np.flatnonzero(stable.sum(axis=1) >= 10):
                linear_error = nmae(
                    1.5 * scales_by_model[name][target] * prediction[target],
                    truth[target],
                    stable[target],
                )
                exact_error = nmae(
                    larger[name][target], truth[target], stable[target]
                )
                larger_differences.append(abs(exact_error - linear_error))
            summaries[-1]["mean_abs_larger_linearization_error"] = np.mean(
                larger_differences
            )

    summary = pd.DataFrame(summaries)
    primary = summary["model"].isin(PRIMARY_MODELS)
    summary.loc[primary, "wrong_target_p_holm"] = holm(
        summary.loc[primary, "wrong_target_p"].to_numpy()
    )
    summary["passes_target_signal_gate"] = (
        summary["model"].isin(PRIMARY_MODELS)
        & summary["excess_wrong_bootstrap_q025"].gt(0)
        & summary["wrong_target_p_holm"].lt(0.05)
    )
    summary["requires_exact_scale_refinement"] = summary[
        "mean_abs_selected_linearization_error"
    ].gt(0.02)
    return pd.DataFrame(per_target), summary


def score_global_axes(
    predictions: dict[str, np.ndarray],
    scales_by_model: dict[str, np.ndarray],
    truth: np.ndarray,
    stable: np.ndarray,
    targets: list[str],
    genes: list[str],
) -> pd.DataFrame:
    target_indices = [
        genes.index(target) if target in genes else None for target in targets
    ]
    signs = np.sign(truth).astype(np.int8)
    equal = (truth.sum(axis=0) - truth) / (len(truth) - 1)
    counts = (
        pd.read_csv(TARGET_COUNTS)
        .set_index("target_gene")
        .loc[targets, "n_cells"]
        .to_numpy()
    )
    legacy = np.log2(np.average(np.exp2(truth), axis=0, weights=counts))
    rows = []
    for axis_name, axis in [("equal_target_loo", equal), ("cell_weighted", legacy)]:
        for model in PRIMARY_MODELS:
            model_effect = scales_by_model[model][:, None] * predictions[model]
            arms = {
                "identity": np.zeros_like(truth),
                "global_only": np.broadcast_to(axis, truth.shape)
                if axis.ndim == 1
                else axis,
                "model_only": model_effect,
            }
            arms["model_plus_global"] = arms["model_only"] + arms["global_only"]
            for arm, values in arms.items():
                scores = []
                errors = []
                for target in range(len(targets)):
                    score = signed_topk(
                        values[target], stable[target], signs[target], target_indices[target]
                    )
                    if score["n_strong"] >= 10:
                        scores.append(score["signed_recovery"])
                        errors.append(nmae(values[target], truth[target], stable[target]))
                rows.append(
                    {
                        "global_axis": axis_name,
                        "model": model,
                        "arm": arm,
                        "mean_signed_recovery": np.mean(scores),
                        "mean_strong_de_nmae": np.mean(errors),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    models, metadata = load_models()
    targets = metadata["target_gene"].astype(str).tolist()
    genes = metadata["output_gene"].astype(str).tolist()
    truth, stable = load_truth(targets, genes)
    tangent = tangent_predictions(models, metadata)
    scale_table, transferred, oracle = scales(tangent, truth, stable)
    exact, larger = exact_predictions(
        models, transferred, metadata, "exact_predictions_initial.npz"
    )
    per_target, summary = score_directions(
        tangent,
        exact,
        larger,
        transferred,
        oracle,
        truth,
        stable,
        targets,
        genes,
    )
    refine_names = summary.loc[
        summary["requires_exact_scale_refinement"], "model"
    ].tolist()
    refinement = pd.DataFrame()
    if refine_names:
        refined, refinement = refine_scales_exact(
            refine_names, models, oracle, metadata, truth, stable
        )
        transferred.update(refined)
        for name in refine_names:
            scale_table.loc[
                scale_table["model"] == name, "loo_median_gamma"
            ] = transferred[name]
        exact, larger = exact_predictions(
            models, transferred, metadata, "exact_predictions.npz"
        )
        per_target, summary = score_directions(
            tangent,
            exact,
            larger,
            transferred,
            oracle,
            truth,
            stable,
            targets,
            genes,
        )
    summary["exact_grid_refined"] = summary["model"].isin(refine_names)
    summary["post_refinement_linearization_gap"] = summary[
        "mean_abs_selected_linearization_error"
    ]
    summary["requires_exact_scale_refinement"] &= ~summary["exact_grid_refined"]
    global_table = score_global_axes(
        tangent, transferred, truth, stable, targets, genes
    )

    with np.load(EMPIRICAL, allow_pickle=False) as empirical:
        empirical_variance = empirical["covariance"]
    target_fit = metadata["target_fit_index"].astype(int)
    diagonal = empirical_variance[target_fit, np.arange(len(targets))]
    empirical_scale = scale_table[scale_table["model"] == "empirical"].copy()
    empirical_scale["target_gene"] = targets
    empirical_scale["native_cipher_a_oracle"] = -empirical_scale["oracle_gamma"] / diagonal
    empirical_scale["native_cipher_a_loo"] = -empirical_scale["loo_median_gamma"] / diagonal

    REPORT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "scales": REPORT / "downstream_amplitude_distribution.csv",
        "per_target": REPORT / "direction_fidelity_per_target.csv",
        "summary": REPORT / "direction_fidelity_summary.csv",
        "global": REPORT / "global_axis_comparison.csv",
        "refinement": REPORT / "exact_scale_refinement.csv",
    }
    scale_table["native_cipher_a_oracle"] = np.nan
    scale_table["native_cipher_a_loo"] = np.nan
    scale_table["native_cipher_a_magnitude_oracle"] = np.nan
    scale_table["native_cipher_a_magnitude_loo"] = np.nan
    scale_table["target_gene"] = np.asarray(targets)[
        scale_table["target_index"].to_numpy(dtype=int)
    ]
    scale_table["n_stable_strong_de"] = stable.sum(axis=1)[
        scale_table["target_index"].to_numpy(dtype=int)
    ]
    scale_table["truth_strong_lfc_l1"] = np.asarray(
        [np.abs(values[mask]).sum() for values, mask in zip(truth, stable, strict=True)]
    )[scale_table["target_index"].to_numpy(dtype=int)]
    scale_table["exact_oracle_gamma"] = np.nan
    if not refinement.empty:
        refined_lookup = refinement.set_index(["model", "target_index"])[
            "exact_oracle_gamma"
        ]
        scale_table["exact_oracle_gamma"] = [
            refined_lookup.get((row.model, row.target_index), np.nan)
            for row in scale_table.itertuples(index=False)
        ]
    take = scale_table["model"] == "empirical"
    scale_table.loc[take, "native_cipher_a_oracle"] = empirical_scale[
        "native_cipher_a_oracle"
    ].to_numpy()
    scale_table.loc[take, "native_cipher_a_loo"] = empirical_scale[
        "native_cipher_a_loo"
    ].to_numpy()
    scale_table.loc[take, "native_cipher_a_magnitude_oracle"] = -empirical_scale[
        "native_cipher_a_oracle"
    ].to_numpy()
    scale_table.loc[take, "native_cipher_a_magnitude_loo"] = -empirical_scale[
        "native_cipher_a_loo"
    ].to_numpy()
    scale_table.to_csv(outputs["scales"], index=False)
    per_target.to_csv(outputs["per_target"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    global_table.to_csv(outputs["global"], index=False)
    refinement.to_csv(outputs["refinement"], index=False)
    manifest = {
        "kind": "truth-side evaluation of restricted linear-response ladder",
        "truth_definition": "stable p<0.05, abs LFC>=0.5, target excluded",
        "magnitude": "model-specific median leave-one-target-out downstream L1 scale; local exact donor grid when decoder nonlinearity exceeds 0.02 NMAE",
        "null_replicates": N_NULL,
        "seeds": {
            "wrong_target": "SeedSequence([0,81,model_index])",
            "gene_label_permutation": "SeedSequence([0,82,model_index])",
            "target_bootstrap": "SeedSequence([0,83,model_index])",
        },
        "exact_refinement_factors": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        "passes_full_cell_scoring_gate": bool(
            summary["passes_target_signal_gate"].any()
        ),
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [H1, DE, STRONG, TARGET_COUNTS, EMPIRICAL, SPARSE, FACTOR]
        },
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path) for path in outputs.values()
        },
        "software": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ["anndata", "numpy", "pandas", "polars", "scipy"]
            },
        },
    }
    (REPORT / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
