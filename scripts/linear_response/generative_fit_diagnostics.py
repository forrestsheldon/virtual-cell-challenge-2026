"""Test control-generative fit against independent means and random pairings."""

from __future__ import annotations

import gc
import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from scripts.linear_response.kernel import fit_covariance_columns, log1cp10k
from scripts.linear_response.model3_poisson_lognormal import (
    BATCH_SIZE,
    INFER,
    LOADING_PENALTY,
    control_split,
    dense_batch,
)

ROOT = Path(__file__).resolve().parents[2]
H1 = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
DERIVED = ROOT / "data/derived/linear_response"
REPORT = ROOT / "reports/linear-response-three-models"
TARGET_TABLE = REPORT / "h1_target_counts.csv"
STRICT_CONTROLS = REPORT / "h1_strict_controls.csv"
MODEL1 = DERIVED / "model1_expected_profiles.npz"
MODEL2 = DERIVED / "model2_expected_profiles.npz"
MODEL3_FIT = DERIVED / "model3_fit.npz"
MODEL3_INITIAL = DERIVED / "model3_initialization.npz"
N_SAMPLE = 256
N_REPLICATES = 3
N_DOWNSTREAM_GENES = 1_024
N_RANDOM_PAIRINGS = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def covariance_columns(
    values: np.ndarray, feature_indices: np.ndarray, target_indices: np.ndarray
) -> np.ndarray:
    centered = values - values.mean(axis=0)
    return (centered[:, feature_indices].T @ centered[:, target_indices]) / (
        len(values) - 1
    )


def column_generalization(
    predicted: np.ndarray,
    observed: np.ndarray,
    rng: np.random.Generator,
    *,
    n_random: int = N_RANDOM_PAIRINGS,
    n_random_gene_mappings: int = 0,
) -> dict[str, float]:
    predicted_norm = np.linalg.norm(predicted, axis=0)
    observed_norm = np.linalg.norm(observed, axis=0)
    unit_predicted = np.divide(
        predicted,
        predicted_norm,
        out=np.zeros_like(predicted),
        where=predicted_norm > 0,
    )
    unit_observed = np.divide(
        observed,
        observed_norm,
        out=np.zeros_like(observed),
        where=observed_norm > 0,
    )
    similarities = unit_predicted.T @ unit_observed
    matched = np.diag(similarities)
    null_medians = np.empty(n_random)
    columns = np.arange(predicted.shape[1])
    for index in range(n_random):
        null_medians[index] = np.median(similarities[columns, rng.permutation(columns)])
    matched_median = float(np.median(matched))
    denominator = float(np.square(observed).sum())
    result = {
        "median_matched_cosine": matched_median,
        "random_pairing_median": float(np.median(null_medians)),
        "random_pairing_q95": float(np.quantile(null_medians, 0.95)),
        "random_pairing_p": float(
            (1 + np.count_nonzero(null_medians >= matched_median)) / (n_random + 1)
        ),
        "squared_error_over_zero_covariance": float(
            np.square(predicted - observed).sum() / denominator
        ),
    }
    if n_random_gene_mappings:
        gene_null_medians = np.empty(n_random_gene_mappings)
        features = np.arange(predicted.shape[0])
        for index in range(n_random_gene_mappings):
            remapped = unit_predicted[rng.permutation(features)]
            gene_null_medians[index] = np.median(
                np.sum(remapped * unit_observed, axis=0)
            )
        result.update(
            {
                "random_gene_mapping_median": float(np.median(gene_null_medians)),
                "random_gene_mapping_q95": float(np.quantile(gene_null_medians, 0.95)),
                "random_gene_mapping_p": float(
                    (1 + np.count_nonzero(gene_null_medians >= matched_median))
                    / (n_random_gene_mappings + 1)
                ),
            }
        )
    return result


def moment_metrics(
    generated: np.ndarray,
    observed: np.ndarray,
    feature_indices: np.ndarray,
    target_indices: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    generated_log = log1cp10k(generated)
    observed_log = log1cp10k(observed)
    generated_covariance = covariance_columns(
        generated_log, feature_indices, target_indices
    )
    observed_covariance = covariance_columns(
        observed_log, feature_indices, target_indices
    )
    covariance = column_generalization(
        generated_covariance, observed_covariance, rng, n_random=1_000
    )
    generated_mean = generated.mean(axis=0)
    observed_mean = observed.mean(axis=0)
    generated_variance = generated.var(axis=0)
    observed_variance = observed.var(axis=0)
    return {
        "zero_fraction_abs_error": float(
            abs(np.mean(generated == 0) - np.mean(observed == 0))
        ),
        "log1p_gene_mean_pearson": float(
            pearsonr(np.log1p(generated_mean), np.log1p(observed_mean)).statistic
        ),
        "log1p_gene_variance_spearman": float(
            spearmanr(
                np.log1p(generated_variance), np.log1p(observed_variance)
            ).statistic
        ),
        "downstream_covariance_cosine": covariance["median_matched_cosine"],
        "downstream_covariance_error_over_zero": covariance[
            "squared_error_over_zero_covariance"
        ],
        "downstream_random_pairing_p": covariance["random_pairing_p"],
    }


def multinomial_rows(
    probabilities: np.ndarray, totals: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    return np.vstack(
        [
            rng.multinomial(int(total), row)
            for total, row in zip(totals, probabilities, strict=True)
        ]
    ).astype(np.int32, copy=False)


def independent_multinomial(
    q: np.ndarray, totals: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    probabilities = np.broadcast_to(q, (len(totals), len(q)))
    return multinomial_rows(probabilities, totals, rng)


def pca_multinomial(
    mean: np.ndarray,
    loadings: np.ndarray,
    totals: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    latent = rng.normal(size=(len(totals), loadings.shape[1]))
    log_expression = mean[None, :] + latent @ loadings.T
    weights = np.clip(np.expm1(log_expression), 0, None)
    probabilities = weights / weights.sum(axis=1, keepdims=True)
    return multinomial_rows(probabilities, totals, rng)


def independent_poisson(
    q: np.ndarray, totals: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    return rng.poisson(totals[:, None] * q[None, :]).astype(np.int32)


def poisson_lognormal(
    m: np.ndarray,
    loadings: np.ndarray,
    totals: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    latent = rng.normal(size=(len(totals), loadings.shape[1]))
    rates = totals[:, None] * np.exp(m[None, :] + latent @ loadings.T)
    return rng.poisson(rates).astype(np.int32)


def train_log_mean(data: ad.AnnData, rows: np.ndarray) -> np.ndarray:
    total = np.zeros(data.n_vars)
    for start in range(0, len(rows), 512):
        total += log1cp10k(data.X[rows[start : start + 512]].tocsr()).sum(0)
    return total / len(rows)


def independent_mean_nll(data: ad.AnnData, rows: np.ndarray, q: np.ndarray) -> float:
    """Match Model 3's NLL convention by omitting count-only x log(total)."""
    values = []
    log_q = np.log(q)
    for start in range(0, len(rows), 128):
        counts = data.X[rows[start : start + 128]].toarray()
        totals = counts.sum(axis=1)
        values.extend((totals - (counts * log_q[None, :]).sum(1)).tolist())
    return float(np.mean(values))


def heldout_likelihoods(
    data: ad.AnnData,
    rows: np.ndarray,
    q: np.ndarray,
    initial_m: np.ndarray,
    initial_loadings: np.ndarray,
    fitted_m: np.ndarray,
    fitted_loadings: np.ndarray,
) -> pd.DataFrame:
    records = []
    usable = rows[: len(rows) // BATCH_SIZE * BATCH_SIZE]
    parameters = {
        "initial_rank50_poisson_lognormal": (initial_m, initial_loadings),
        "fitted_rank50_poisson_lognormal": (fitted_m, fitted_loadings),
    }
    for start in range(0, len(usable), BATCH_SIZE):
        batch_rows = usable[start : start + BATCH_SIZE]
        counts, totals = dense_batch(data, batch_rows)
        batch = pd.DataFrame(
            {
                "source_row": batch_rows,
                "guide_id": data.obs.iloc[batch_rows]["guide_id"]
                .astype(str)
                .to_numpy(),
                "batch": data.obs.iloc[batch_rows]["batch"].astype(str).to_numpy(),
                "independent_mean_poisson": totals
                - (counts * np.log(q)[None, :]).sum(1),
            }
        )
        for name, (m, loadings) in parameters.items():
            modes, rates, hessian = INFER(
                jnp.asarray(m),
                jnp.asarray(loadings),
                jnp.asarray(counts),
                jnp.asarray(totals),
            )
            modes = np.asarray(modes)
            rates = np.asarray(rates)
            hessian = np.asarray(hessian)
            linear = m[None, :] + modes @ loadings.T
            negative_joint = (
                rates.sum(1) - (counts * linear).sum(1) + 0.5 * np.square(modes).sum(1)
            )
            batch[name] = negative_joint + 0.5 * np.linalg.slogdet(hessian)[1]
        records.append(batch)
        if (start // BATCH_SIZE + 1) % 25 == 0:
            print(
                f"Held-out likelihood batch {start // BATCH_SIZE + 1}/{len(usable) // BATCH_SIZE}"
            )
    return pd.concat(records, ignore_index=True)


def likelihood_improvements(values: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("independent_mean_poisson", "initial_rank50_poisson_lognormal"),
        ("independent_mean_poisson", "fitted_rank50_poisson_lognormal"),
        ("initial_rank50_poisson_lognormal", "fitted_rank50_poisson_lognormal"),
    ]
    records = []
    for baseline, model in comparisons:
        difference = values[baseline] - values[model]
        standard_error = difference.std(ddof=1) / np.sqrt(len(difference))
        records.append(
            {
                "baseline": baseline,
                "model": model,
                "mean_nll_improvement": difference.mean(),
                "paired_cell_standard_error": standard_error,
                "paired_cell_z": difference.mean() / standard_error,
                "fraction_cells_improved": np.mean(difference > 0),
                "n_cells": len(difference),
                "caveat": "cell-level SE is descriptive; only two control guides were held out",
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    target_table = pd.read_csv(TARGET_TABLE)
    targets = target_table["target_gene"].astype(str).tolist()
    strict_guides = pd.read_csv(STRICT_CONTROLS)["guide_id"].astype(str).tolist()
    with np.load(MODEL1, allow_pickle=False) as model1:
        genes = model1["gene_names"].astype(str).tolist()
        half0 = model1["half0_directions"].astype(np.float64)
        half1 = model1["half1_directions"].astype(np.float64)
    with np.load(MODEL2, allow_pickle=False) as model2:
        components = model2["components"][:50].astype(np.float64)
    with np.load(MODEL3_INITIAL, allow_pickle=False) as initial:
        initial_m = initial["m"].astype(np.float64)
        initial_loadings = initial["loadings"].astype(np.float64)
    with np.load(MODEL3_FIT, allow_pickle=False) as fitted:
        fitted_m = fitted["m"].astype(np.float64)
        fitted_loadings = fitted["loadings"].astype(np.float64)

    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    target_indices = np.asarray([gene_lookup[target] for target in targets])
    rng = np.random.default_rng(0)
    covariance_rows = []
    projected0 = components.T @ (components @ half0)
    projected1 = components.T @ (components @ half1)

    data = ad.read_h5ad(H1, backed="r")
    try:
        if data.var_names.astype(str).tolist() != genes:
            raise ValueError("model and H1 gene axes differ")
        train_rows, heldout_rows, heldout_guides = control_split(
            data.obs, strict_guides
        )
        heldout_covariance, _, _, _ = fit_covariance_columns(
            data, heldout_rows, target_indices
        )
        fitted_covariance = fitted_loadings @ fitted_loadings[target_indices].T

        q = np.exp(initial_m + 0.5 * np.square(initial_loadings).sum(axis=1))
        q /= q.sum()
        mean = train_log_mean(data, train_rows)
        downstream_mask = np.ones(len(genes), dtype=bool)
        downstream_mask[target_indices] = False
        non_targets = np.flatnonzero(downstream_mask)
        feature_indices = non_targets[
            np.argsort(mean[non_targets])[-N_DOWNSTREAM_GENES:]
        ]
        covariance_rows.extend(
            [
                {
                    "model": "cipher_empirical_covariance",
                    "comparison": "control split half 0 versus half 1",
                    **column_generalization(
                        half0[feature_indices],
                        half1[feature_indices],
                        rng,
                        n_random_gene_mappings=2_000,
                    ),
                },
                {
                    "model": "rank50_latent_projection",
                    "comparison": "full-control rank-50 basis applied to independent control halves",
                    **column_generalization(
                        projected0[feature_indices],
                        projected1[feature_indices],
                        rng,
                        n_random_gene_mappings=2_000,
                    ),
                },
                {
                    "model": "poisson_lognormal",
                    "comparison": "train-only fitted log-rate covariance versus held-out guides",
                    **column_generalization(
                        fitted_covariance[feature_indices],
                        heldout_covariance[feature_indices],
                        rng,
                        n_random_gene_mappings=2_000,
                    ),
                },
            ]
        )
        sample_rng = np.random.default_rng(73)
        sample_rows = np.sort(sample_rng.choice(heldout_rows, N_SAMPLE, replace=False))
        observed = data.X[sample_rows].toarray().astype(np.int32)
        totals = observed.sum(axis=1)
        generators = {
            "independent_mean_multinomial": lambda local_rng: independent_multinomial(
                q, totals, local_rng
            ),
            "rank50_gaussian_multinomial": lambda local_rng: pca_multinomial(
                mean, initial_loadings, totals, local_rng
            ),
            "independent_mean_poisson": lambda local_rng: independent_poisson(
                q, totals, local_rng
            ),
            "poisson_lognormal_prior": lambda local_rng: poisson_lognormal(
                fitted_m, fitted_loadings, totals, local_rng
            ),
        }
        generated_rows = []
        for model_index, (name, generator) in enumerate(generators.items()):
            for replicate in range(N_REPLICATES):
                local_rng = np.random.default_rng(
                    np.random.SeedSequence([0, 11, model_index, replicate])
                )
                generated = generator(local_rng)
                metrics = moment_metrics(
                    generated,
                    observed,
                    feature_indices,
                    target_indices,
                    local_rng,
                )
                generated_rows.append(
                    {"generator": name, "replicate": replicate, **metrics}
                )
                del generated
                gc.collect()
                print(f"Generated-control check: {name} replicate {replicate + 1}")

        likelihood_cells = heldout_likelihoods(
            data,
            heldout_rows,
            q,
            initial_m,
            initial_loadings,
            fitted_m,
            fitted_loadings,
        )
    finally:
        data.file.close()

    model3_manifest = json.loads((REPORT / "model3_manifest.json").read_text())
    initial_penalty = LOADING_PENALTY * float(np.square(initial_loadings).sum())
    fitted_penalty = LOADING_PENALTY * float(np.square(fitted_loadings).sum())
    likelihood_means = likelihood_cells.select_dtypes("number").mean()
    if not np.isclose(
        likelihood_means["initial_rank50_poisson_lognormal"] + initial_penalty,
        model3_manifest["initial_heldout_laplace_nll"],
        atol=0.1,
    ):
        raise AssertionError("per-cell initial likelihood does not reproduce manifest")
    if not np.isclose(
        likelihood_means["fitted_rank50_poisson_lognormal"] + fitted_penalty,
        model3_manifest["selected_heldout_laplace_nll"],
        atol=0.1,
    ):
        raise AssertionError("per-cell fitted likelihood does not reproduce manifest")
    likelihood = pd.DataFrame(
        [
            {
                "model": "independent_mean_poisson",
                "heldout_nll_per_cell_without_count_constant": likelihood_means[
                    "independent_mean_poisson"
                ],
                "comparison": "exact conditional Poisson with train-only pooled mean",
            },
            {
                "model": "initial_rank50_poisson_lognormal",
                "heldout_nll_per_cell_without_count_constant": model3_manifest[
                    "initial_heldout_laplace_nll"
                ]
                - initial_penalty,
                "comparison": "30-step Laplace marginal approximation",
            },
            {
                "model": "fitted_rank50_poisson_lognormal",
                "heldout_nll_per_cell_without_count_constant": model3_manifest[
                    "selected_heldout_laplace_nll"
                ]
                - fitted_penalty,
                "comparison": "30-step Laplace marginal approximation",
            },
        ]
    )
    regularization = pd.DataFrame(
        [
            {
                "model": "cipher_empirical_covariance",
                "structural_constraint": "150 selected empirical covariance columns",
                "penalty": "none",
                "sparsity_prior": False,
                "exact_zero_fraction": float(np.mean(half0 == 0)),
            },
            {
                "model": "rank50_latent_projection",
                "structural_constraint": "rank 50",
                "penalty": "none",
                "sparsity_prior": False,
                "exact_zero_fraction": float(np.mean(components == 0)),
            },
            {
                "model": "poisson_lognormal",
                "structural_constraint": "rank 50",
                "penalty": f"L2 loadings coefficient {LOADING_PENALTY}",
                "sparsity_prior": False,
                "exact_zero_fraction": float(np.mean(fitted_loadings == 0)),
            },
        ]
    )

    outputs = {
        "covariance": REPORT / "generative_covariance_generalization.csv",
        "samples": REPORT / "generative_sampled_control_fit.csv",
        "likelihood": REPORT / "generative_heldout_likelihood.csv",
        "likelihood_improvement": REPORT
        / "generative_heldout_likelihood_improvement.csv",
        "likelihood_by_stratum": REPORT
        / "generative_heldout_likelihood_by_guide_batch.csv",
        "regularization": REPORT / "generative_regularization_audit.csv",
    }
    pd.DataFrame(covariance_rows).to_csv(outputs["covariance"], index=False)
    pd.DataFrame(generated_rows).to_csv(outputs["samples"], index=False)
    likelihood.to_csv(outputs["likelihood"], index=False)
    likelihood_improvements(likelihood_cells).to_csv(
        outputs["likelihood_improvement"], index=False
    )
    likelihood_cells.groupby(["guide_id", "batch"], as_index=False).mean(
        numeric_only=True
    ).to_csv(outputs["likelihood_by_stratum"], index=False)
    regularization.to_csv(outputs["regularization"], index=False)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "purpose": "control-only generative fit checks; no perturbation truth read",
        "truth_cells_read": False,
        "heldout_guides": heldout_guides,
        "heldout_cells": len(heldout_rows),
        "sampled_cells": N_SAMPLE,
        "sample_replicates": N_REPLICATES,
        "downstream_gene_panel": {
            "selection": "top train-control mean log1cp10k genes excluding all 150 target genes",
            "n_genes": N_DOWNSTREAM_GENES,
            "genes": [genes[index] for index in feature_indices],
        },
        "random_column_pairings": N_RANDOM_PAIRINGS,
        "random_gene_mappings": 2_000,
        "limitations": [
            "Cipher stores selected covariance columns, not a positive-definite full covariance, so it has no sampled-count arm without adding an unimplemented distribution.",
            "The rank-50 split-half projection uses a basis fitted on all controls; only the Model 3 covariance and likelihood checks use held-out guides excluded from global fitting.",
            "The null ensembles permute target-column identity and gene-to-covariance-row mapping while preserving learned values; they are not a universal random-matrix test.",
            "No fitted model uses a sparsity prior.",
        ],
        "inputs": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in [MODEL1, MODEL2, MODEL3_FIT, MODEL3_INITIAL]
        },
        "outputs": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
            }
            for name, path in outputs.items()
        },
        "software": {
            "python": platform.python_version(),
            "anndata": version("anndata"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scipy": version("scipy"),
        },
    }
    (REPORT / "generative_fit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
