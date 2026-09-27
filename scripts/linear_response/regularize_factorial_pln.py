"""Regularize factorial PLN response columns using control-only replication."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from scripts.linear_response.kernel import normalize_response_columns
from scripts.linear_response.poisson_lognormal_moments import FactorialMoments

ROOT = Path(__file__).resolve().parents[2]
H1 = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
EMPIRICAL = ROOT / "data/derived/linear_response/ladder/empirical.npz"
FACTORIAL = (
    ROOT
    / "data/derived/linear_response/sequencing_model/factorial_moment_response.npz"
)
STRICT_GUIDES = ROOT / "reports/linear-response-three-models/h1_strict_controls.csv"
DERIVED = ROOT / "data/derived/linear_response/sequencing_model"
REPORT = ROOT / "reports/linear-response-sequencing-model"
ARTIFACT = DERIVED / "regularized_factorial_response.npz"
SPLITS = DERIVED / "regularization_control_splits.npz"
N_SPLITS = 5
CHUNK_SIZE = 512
MEAN_LAMBDAS_CPM = np.asarray([5.0, 10.0, 20.0, 50.0, 100.0])
CPM_EDGES = np.asarray([5.0, 10.0, 25.0, 100.0, 500.0, np.inf])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repeated_halves(obs: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    """Five deterministic halves within every guide-by-batch stratum."""
    controls = obs.iloc[rows]
    guide = controls["guide_id"].astype(str).to_numpy()
    batch = controls["batch"].astype(str).to_numpy()
    groups = sorted(set(zip(guide, batch, strict=True)))
    halves = np.empty((N_SPLITS, len(rows)), dtype=np.int8)
    for split in range(N_SPLITS):
        for group, (guide_id, batch_id) in enumerate(groups):
            positions = np.flatnonzero((guide == guide_id) & (batch == batch_id))
            shuffled = np.random.default_rng(
                np.random.SeedSequence([0, 313, split, group])
            ).permutation(positions)
            halves[split, shuffled[::2]] = 0
            halves[split, shuffled[1::2]] = 1
    return halves


def fit_splits(
    data: ad.AnnData,
    rows: np.ndarray,
    halves: np.ndarray,
    fit_indices: np.ndarray,
    target_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream first and factorial-second moments for all split halves."""
    moments = [
        [FactorialMoments.zeros(len(fit_indices), len(target_indices)) for _ in range(2)]
        for _ in range(N_SPLITS)
    ]
    for start in range(0, len(rows), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(rows))
        raw_full = data.X[rows[start:stop]].tocsr().astype(np.float64)
        totals = np.asarray(raw_full.sum(axis=1)).ravel()
        raw = raw_full[:, fit_indices].toarray()
        for split in range(N_SPLITS):
            assignment = halves[split, start:stop]
            for half in (0, 1):
                take = assignment == half
                moments[split][half].update(raw[take], totals[take], target_indices)
        print(f"factorial PLN regularization {stop}/{len(rows)} cells", flush=True)

    covariance = np.empty(
        (N_SPLITS, 2, len(fit_indices), len(target_indices)), dtype=np.float32
    )
    mean_rate = np.empty((N_SPLITS, 2, len(fit_indices)), dtype=np.float32)
    cells = np.empty((N_SPLITS, 2), dtype=int)
    for split in range(N_SPLITS):
        for half in (0, 1):
            mean, second = moments[split][half].finish()
            ratio = second / np.outer(mean, mean[target_indices])
            with np.errstate(divide="ignore", invalid="ignore"):
                covariance[split, half] = np.log(ratio)
            mean_rate[split, half] = mean
            cells[split, half] = moments[split][half].n
    return covariance, mean_rate, cells


def sampling_variance(
    split_covariance: np.ndarray, split_cells: np.ndarray, full_cells: int
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate Var(Sigma_hat) from repeated independent-half differences."""
    difference = split_covariance[:, 0] - split_covariance[:, 1]
    inverse_cells = 1 / split_cells[:, 0] + 1 / split_cells[:, 1]
    scaled = np.square(difference) / inverse_cells[:, None, None]
    finite = np.isfinite(scaled)
    process_variance = np.divide(
        np.where(finite, scaled, 0).sum(axis=0),
        finite.sum(axis=0),
        out=np.full(scaled.shape[1:], np.inf),
        where=finite.sum(axis=0) > 0,
    )
    return process_variance / full_cells, process_variance


def normal_mixture_fit(
    estimate: np.ndarray,
    standard_error: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a zero-centered normal mixture by marginal maximum likelihood."""
    keep = mask & np.isfinite(estimate) & np.isfinite(standard_error) & (standard_error > 0)
    z = estimate[keep]
    se = standard_error[keep]
    rng = np.random.default_rng(np.random.SeedSequence([0, 317]))
    if len(z) > 300_000:
        take = rng.choice(len(z), 300_000, replace=False)
        z, se = z[take], se[take]
    lower = max(float(np.quantile(se, 0.1)) / 4, 1e-4)
    upper = max(float(np.quantile(np.abs(z), 0.999)), 16 * lower)
    scales = np.concatenate([[0.0], np.geomspace(lower, upper, 20)])
    variance = np.square(se[:, None]) + np.square(scales[None, :])
    log_density = -0.5 * (
        np.log(2 * np.pi * variance) + np.square(z[:, None]) / variance
    )
    weights = np.full(len(scales), 1 / len(scales))
    for _ in range(200):
        responsibility = np.exp(
            log_density
            + np.log(weights)[None, :]
            - logsumexp(log_density + np.log(weights)[None, :], axis=1)[:, None]
        )
        updated = responsibility.mean(axis=0)
        if np.max(np.abs(updated - weights)) < 1e-7:
            weights = updated
            break
        weights = updated
    return scales, weights


def normal_mixture_posterior_mean(
    estimate: np.ndarray,
    standard_error: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Posterior mean under a fitted zero-centered normal mixture."""
    estimate = np.asarray(estimate, dtype=np.float64)
    standard_error = np.asarray(standard_error, dtype=np.float64)
    result = np.zeros_like(estimate)
    flat_z = estimate.ravel()
    flat_se = standard_error.ravel()
    flat_result = result.ravel()
    for start in range(0, len(flat_z), 100_000):
        stop = min(start + 100_000, len(flat_z))
        z = flat_z[start:stop]
        se = flat_se[start:stop]
        keep = np.isfinite(z) & np.isfinite(se) & (se > 0)
        variance = np.square(se[keep, None]) + np.square(scales[None, :])
        log_probability = (
            np.log(weights)[None, :]
            - 0.5
            * (
                np.log(2 * np.pi * variance)
                + np.square(z[keep, None]) / variance
            )
        )
        responsibility = np.exp(
            log_probability - logsumexp(log_probability, axis=1)[:, None]
        )
        shrinkage = np.square(scales)[None, :] / variance
        values = np.zeros_like(z)
        values[keep] = z[keep] * np.sum(responsibility * shrinkage, axis=1)
        flat_result[start:stop] = values
    return result


def mean_shrinkage(
    covariance: np.ndarray,
    mean_rate: np.ndarray,
    target_indices: np.ndarray,
    lambda_cpm: float,
) -> np.ndarray:
    """Shrink relative excess moments using abundance pseudocounts."""
    weight = mean_rate / (mean_rate + lambda_cpm / 1_000_000)
    result = np.zeros_like(covariance, dtype=np.float64)
    finite = np.isfinite(covariance)
    relative_excess = np.expm1(np.clip(covariance, -80, 80))
    shrunk = np.log1p(
        relative_excess * np.outer(weight, weight[target_indices])
    )
    result[finite] = shrunk[finite]
    return result


def downstream_mask(
    genes: int, targets: int, target_indices: np.ndarray, output_count: int
) -> np.ndarray:
    mask = np.zeros((genes, targets), dtype=bool)
    mask[:output_count] = True
    mask[target_indices, np.arange(targets)] = False
    return mask


def column_reproducibility(
    left: np.ndarray, right: np.ndarray, target_indices: np.ndarray, output_count: int
) -> tuple[float, float]:
    """Return matched cosine and its excess over wrong-column cosines."""
    keep = np.ones(output_count, dtype=bool)
    keep[target_indices[target_indices < output_count]] = False
    left = np.nan_to_num(left[:output_count][keep].T)
    right = np.nan_to_num(right[:output_count][keep].T)
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    left = np.divide(left, left_norm, out=np.zeros_like(left), where=left_norm > 0)
    right = np.divide(
        right, right_norm, out=np.zeros_like(right), where=right_norm > 0
    )
    similarities = left @ right.T
    matched = np.diag(similarities)
    wrong = np.asarray(
        [np.median(np.delete(row, index)) for index, row in enumerate(similarities)]
    )
    return float(np.median(matched)), float(np.median(matched - wrong))


def split_metrics(
    name: str,
    split_regularized: np.ndarray,
    split_raw: np.ndarray,
    process_variance: np.ndarray,
    split_cells: np.ndarray,
    mask: np.ndarray,
    target_indices: np.ndarray,
    output_count: int,
    full_regularized: np.ndarray,
    full_raw: np.ndarray,
) -> dict[str, float | str]:
    cosines, specificities, squared, standardized = [], [], [], []
    for split in range(N_SPLITS):
        left, right = split_regularized[split]
        matched, specificity = column_reproducibility(
            left, right, target_indices, output_count
        )
        cosines.append(matched)
        specificities.append(specificity)
        raw_left, raw_right = split_raw[split]
        prediction_error = 0.5 * (
            np.square(left - raw_right) + np.square(right - raw_left)
        )
        valid = mask & np.isfinite(prediction_error) & np.isfinite(process_variance)
        squared.append(float(prediction_error[valid].mean()))
        noise = process_variance * (
            1 / split_cells[split, 0] + 1 / split_cells[split, 1]
        )
        informative = valid & (noise > 0)
        standardized.append(float(np.mean(prediction_error[informative] / noise[informative])))
    raw_energy = np.square(full_raw[mask]).sum()
    return {
        "model": name,
        "median_split_half_cosine": float(np.median(cosines)),
        "median_matched_minus_wrong_cosine": float(np.median(specificities)),
        "mean_heldout_log_covariance_mse": float(np.mean(squared)),
        "mean_noise_standardized_error": float(np.mean(standardized)),
        "retained_covariance_energy": float(
            np.square(full_regularized[mask]).sum() / raw_energy
        ),
        "effective_zero_fraction": float(
            np.mean(np.abs(full_regularized[mask]) < 1e-3)
        ),
    }


def main() -> None:
    started = time.perf_counter()
    with np.load(EMPIRICAL, allow_pickle=False) as empirical, np.load(
        FACTORIAL, allow_pickle=False
    ) as factorial:
        target_names = empirical["target_gene"].astype(str)
        output_genes = empirical["output_gene"].astype(str)
        fit_genes = empirical["fit_gene"].astype(str)
        fit_indices = empirical["full_gene_index"].astype(int)
        target_indices = empirical["target_fit_index"].astype(int)
        source_rows = empirical["source_rows"].astype(int)
        factorial_names = factorial["model"].astype(str).tolist()
        model_index = factorial_names.index("control_factorial_pln")
        full_covariance = factorial["covariance"][model_index].astype(np.float64)
        full_mean = factorial["mean_rate"][model_index].astype(np.float64)

    strict_guides = set(pd.read_csv(STRICT_GUIDES)["guide_id"].astype(str))
    data = ad.read_h5ad(H1, backed="r")
    try:
        labels = data.obs["target_gene"].astype(str).to_numpy()
        guides = data.obs["guide_id"].astype(str).to_numpy()
        rows = np.flatnonzero(
            (labels == "non-targeting") & np.isin(guides, list(strict_guides))
        )
        halves = repeated_halves(data.obs, rows)
        if SPLITS.exists():
            with np.load(SPLITS, allow_pickle=False) as saved:
                split_covariance = saved["covariance"].astype(np.float64)
                split_mean = saved["mean_rate"].astype(np.float64)
                split_cells = saved["cells"].astype(int)
        else:
            split_covariance, split_mean, split_cells = fit_splits(
                data, rows, halves, fit_indices, target_indices
            )
            DERIVED.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                SPLITS,
                covariance=split_covariance,
                mean_rate=split_mean,
                cells=split_cells,
                half_assignment=halves,
                control_rows=rows,
            )
    finally:
        data.file.close()

    output_count = len(output_genes)
    reconstruction_rows = []
    for split in range(N_SPLITS):
        reconstructed_mean = np.average(
            split_mean[split], axis=0, weights=split_cells[split]
        )
        reconstructed_second = np.average(
            [
                np.outer(
                    split_mean[split, half],
                    split_mean[split, half, target_indices],
                )
                * np.exp(split_covariance[split, half])
                for half in (0, 1)
            ],
            axis=0,
            weights=split_cells[split],
        )
        reconstructed_covariance = np.log(
            reconstructed_second
            / np.outer(reconstructed_mean, reconstructed_mean[target_indices])
        )
        reconstruction_rows.append(
            {
                "split": split,
                "maximum_mean_error": np.max(
                    np.abs(reconstructed_mean - full_mean)
                ),
                "maximum_covariance_error": np.max(
                    np.abs(reconstructed_covariance - full_covariance)
                ),
                "median_covariance_error": np.median(
                    np.abs(reconstructed_covariance - full_covariance)
                ),
            }
        )
    if max(row["maximum_covariance_error"] for row in reconstruction_rows) > 1e-6:
        raise AssertionError("split factorial moments do not reconstruct the full fit")

    mask = downstream_mask(
        len(fit_genes), len(target_names), target_indices, output_count
    )
    full_sampling_variance, process_variance = sampling_variance(
        split_covariance, split_cells, len(rows)
    )
    full_standard_error = np.sqrt(full_sampling_variance)
    scales, weights = normal_mixture_fit(
        full_covariance, full_standard_error, mask
    )
    eb_full = normal_mixture_posterior_mean(
        full_covariance, full_standard_error, scales, weights
    )
    eb_splits = np.empty_like(split_covariance)
    for split in range(N_SPLITS):
        for half in (0, 1):
            eb_splits[split, half] = normal_mixture_posterior_mean(
                split_covariance[split, half],
                np.sqrt(process_variance / split_cells[split, half]),
                scales,
                weights,
            )

    diagonal = full_covariance[target_indices, np.arange(len(target_indices))]
    eb_full[target_indices, np.arange(len(target_indices))] = diagonal
    for split in range(N_SPLITS):
        for half in (0, 1):
            eb_splits[split, half, target_indices, np.arange(len(target_indices))] = (
                split_covariance[
                    split, half, target_indices, np.arange(len(target_indices))
                ]
            )

    rows_summary = [
        split_metrics(
            "raw_factorial_pln",
            split_covariance,
            split_covariance,
            process_variance,
            split_cells,
            mask,
            target_indices,
            output_count,
            full_covariance,
            full_covariance,
        )
    ]
    mean_models: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for lambda_cpm in MEAN_LAMBDAS_CPM:
        full = mean_shrinkage(
            full_covariance, full_mean, target_indices, lambda_cpm
        )
        split = np.asarray(
            [
                [
                    mean_shrinkage(
                        split_covariance[repeat, half],
                        split_mean[repeat, half],
                        target_indices,
                        lambda_cpm,
                    )
                    for half in (0, 1)
                ]
                for repeat in range(N_SPLITS)
            ]
        )
        full[target_indices, np.arange(len(target_indices))] = diagonal
        mean_models[lambda_cpm] = (full, split)
        result = split_metrics(
            f"mean_shrink_{lambda_cpm:g}cpm",
            split,
            split_covariance,
            process_variance,
            split_cells,
            mask,
            target_indices,
            output_count,
            full,
            full_covariance,
        )
        result["lambda_cpm"] = lambda_cpm
        rows_summary.append(result)
    rows_summary.append(
        split_metrics(
            "adaptive_eb",
            eb_splits,
            split_covariance,
            process_variance,
            split_cells,
            mask,
            target_indices,
            output_count,
            eb_full,
            full_covariance,
        )
    )
    summary = pd.DataFrame(rows_summary)
    mean_rows = summary[summary["model"].str.startswith("mean_shrink")]
    selected_lambda = float(
        mean_rows.loc[mean_rows["mean_noise_standardized_error"].idxmin(), "lambda_cpm"]
    )
    mean_full, mean_splits = mean_models[selected_lambda]

    models = {
        "control_factorial_pln_mean_shrunk": mean_full,
        "control_factorial_pln_adaptive_eb": eb_full,
    }
    responses = []
    for covariance in models.values():
        response, _ = normalize_response_columns(covariance, target_indices)
        responses.append(response.T)

    cpm_rows = []
    cpm = full_mean * 1_000_000
    cpm_models = {
        "raw_factorial_pln": (full_covariance, split_covariance),
        f"mean_shrink_{selected_lambda:g}cpm": (mean_full, mean_splits),
        "adaptive_eb": (eb_full, eb_splits),
    }
    for lower, upper in pairwise(CPM_EDGES):
        gene_mask = (cpm[:output_count] >= lower) & (cpm[:output_count] < upper)
        gene_mask[target_indices[target_indices < output_count]] = False
        bin_mask = np.zeros_like(mask)
        bin_mask[:output_count] = gene_mask[:, None]
        for name, (full, split) in cpm_models.items():
            cosines = [
                column_reproducibility(
                    pair[0] * bin_mask,
                    pair[1] * bin_mask,
                    target_indices,
                    output_count,
                )[0]
                for pair in split
            ]
            cpm_rows.append(
                {
                    "model": name,
                    "cpm_lower": lower,
                    "cpm_upper": upper,
                    "genes": int(gene_mask.sum()),
                    "median_abs_covariance": float(np.median(np.abs(full[bin_mask]))),
                    "median_standard_error": float(
                        np.median(full_standard_error[bin_mask])
                    ),
                    "median_split_half_cosine": float(np.median(cosines)),
                    "retained_energy": float(
                        np.square(full[bin_mask]).sum()
                        / np.square(full_covariance[bin_mask]).sum()
                    ),
                }
            )

    DERIVED.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ARTIFACT,
        model=np.asarray(list(models)),
        response=np.asarray(responses, dtype=np.float32),
        covariance=np.asarray(list(models.values()), dtype=np.float32),
        standard_error=full_standard_error.astype(np.float32),
        target_gene=target_names,
        output_gene=output_genes,
        fit_gene=fit_genes,
        full_gene_index=fit_indices,
        target_fit_index=target_indices,
        source_rows=source_rows,
        selected_mean_lambda_cpm=np.asarray(selected_lambda),
    )
    summary_path = REPORT / "regularization_summary.csv"
    cpm_path = REPORT / "regularization_by_cpm.csv"
    mixture_path = REPORT / "regularization_mixture.csv"
    reconstruction_path = REPORT / "regularization_split_reconstruction.csv"
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(cpm_rows).to_csv(cpm_path, index=False)
    pd.DataFrame({"scale": scales, "weight": weights}).to_csv(
        mixture_path, index=False
    )
    pd.DataFrame(reconstruction_rows).to_csv(reconstruction_path, index=False)

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "kind": "control-only regularization of factorial PLN covariance columns",
        "splits": "five deterministic guide-by-batch balanced cell halves",
        "uncertainty": "repeated-half difference estimate of log-covariance sampling variance",
        "adaptive_prior": "zero-centered normal scale mixture fitted by marginal maximum likelihood",
        "selected_mean_lambda_cpm": selected_lambda,
        "truth_cells_read": False,
        "runtime_seconds": time.perf_counter() - started,
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [H1, EMPIRICAL, FACTORIAL, STRICT_GUIDES]
        },
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                ARTIFACT,
                SPLITS,
                summary_path,
                cpm_path,
                mixture_path,
                reconstruction_path,
            ]
        },
        "software": {
            "python": platform.python_version(),
            **{name: version(name) for name in ["anndata", "numpy", "pandas", "scipy"]},
        },
    }
    (REPORT / "regularization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
