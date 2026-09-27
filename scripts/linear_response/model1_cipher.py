"""Fit Model 1 and emit model-only H1 expected-profile diagnostics."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scripts.linear_response.kernel import (
    BULK_TARGET_SUM,
    CELL_TARGET_SUM,
    collect_raw_pseudobulks,
    decode_multinomial,
    expected_decoded_sum,
    fit_covariance_columns,
    intended_target_shift,
    knockdown_depths,
    log1cp10k,
    log_pseudobulk,
    sample_balanced_controls,
    solve_output_matched_amplitude,
    split_control_halves,
)

ROOT = Path(__file__).resolve().parents[2]
H1_PATH = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
REPORT_DIR = ROOT / "reports/linear-response-three-models"
DERIVED_DIR = ROOT / "data/derived/linear_response"
TARGET_TABLE = REPORT_DIR / "h1_target_counts.csv"
STRICT_CONTROLS = REPORT_DIR / "h1_strict_controls.csv"
MODEL_INDEX = 0
H1_SHA256 = "a09977104fefb622368ca74b50c9d3c1e891733e6c83db07acfca49b0219c02b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else 0.0


def control_fit_diagnostics(
    target_order: list[str],
    target_indices: list[int],
    full: np.ndarray,
    half0: np.ndarray,
    half1: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for column, (target, gene_index) in enumerate(
        zip(target_order, target_indices, strict=True)
    ):
        mask = np.ones(full.shape[0], dtype=bool)
        mask[gene_index] = False
        first = half0[mask, column]
        second = half1[mask, column]
        norms = [np.linalg.norm(first), np.linalg.norm(second)]
        rows.append(
            {
                "target_gene": target,
                "source_order": column,
                "direction_diagonal": full[gene_index, column],
                "full_downstream_norm": np.linalg.norm(full[mask, column]),
                "half0_downstream_norm": norms[0],
                "half1_downstream_norm": norms[1],
                "half_cosine_downstream": cosine(first, second),
                "half_norm_ratio": min(norms) / max(norms),
            }
        )
    return pd.DataFrame(rows)


def fit_batch_centered_columns(
    data: ad.AnnData, rows: np.ndarray, target_indices: list[int]
) -> np.ndarray:
    """Fit pooled within-batch covariance columns in two streaming passes."""
    labels = data.obs.iloc[rows]["batch"].astype(str).to_numpy()
    batches = sorted(np.unique(labels))
    lookup = {batch: index for index, batch in enumerate(batches)}
    codes = np.asarray([lookup[label] for label in labels], dtype=int)
    counts = np.bincount(codes, minlength=len(batches))
    sums = np.zeros((len(batches), data.n_vars), dtype=np.float64)
    chunk_size = 1_024
    for start in range(0, len(rows), chunk_size):
        values = log1cp10k(data.X[rows[start : start + chunk_size]].tocsr())
        chunk_codes = codes[start : start + chunk_size]
        for code in np.unique(chunk_codes):
            sums[code] += values[chunk_codes == code].sum(axis=0)
    means = sums / counts[:, None]
    cross = np.zeros((data.n_vars, len(target_indices)), dtype=np.float64)
    for start in range(0, len(rows), chunk_size):
        values = log1cp10k(data.X[rows[start : start + chunk_size]].tocsr())
        chunk_codes = codes[start : start + chunk_size]
        centered = values - means[chunk_codes]
        cross += centered.T @ centered[:, target_indices]
    return cross / (len(rows) - len(batches))


def direction_geometry(
    directions: np.ndarray, target_indices: list[int], target_order: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    downstream = directions.copy()
    downstream[np.asarray(target_indices), np.arange(len(target_indices))] = 0
    norms = np.linalg.norm(downstream, axis=0)
    unit = np.divide(downstream, norms, out=np.zeros_like(downstream), where=norms > 0)
    similarities = unit.T @ unit
    upper = similarities[np.triu_indices(len(target_order), k=1)]
    singular = np.linalg.svd(unit, compute_uv=False)
    energy = singular**2
    probabilities = energy / energy.sum()
    effective_rank = float(np.exp(-(probabilities * np.log(probabilities)).sum()))
    summary = pd.DataFrame(
        [
            {"statistic": "pairwise_cosine_q10", "value": np.quantile(upper, 0.1)},
            {"statistic": "pairwise_cosine_median", "value": np.median(upper)},
            {"statistic": "pairwise_cosine_q90", "value": np.quantile(upper, 0.9)},
            {"statistic": "top3_axis_energy_fraction", "value": probabilities[:3].sum()},
            {"statistic": "effective_rank", "value": effective_rank},
        ]
    )
    nearest = similarities.copy()
    np.fill_diagonal(nearest, -np.inf)
    nearest_rows = pd.DataFrame(
        {
            "target_gene": target_order,
            "nearest_target": [target_order[index] for index in nearest.argmax(axis=1)],
            "nearest_cosine": nearest.max(axis=1),
        }
    )
    return summary, nearest_rows


def add_restricted_loo(table: pd.DataFrame, target_counts: pd.DataFrame) -> pd.DataFrame:
    result = table.merge(
        target_counts[["target_gene", "n_cells", "benchmark_candidate"]],
        on="target_gene",
        validate="one_to_one",
    )
    observed = result["observed_knockdown_depth"].to_numpy()
    eligible = result["eligible_knockdown_donor"].to_numpy() & result[
        "benchmark_candidate"
    ].to_numpy()
    restricted = np.full(len(result), np.nan)
    for index in range(len(result)):
        donors = eligible.copy()
        donors[index] = False
        if donors.any():
            restricted[index] = np.median(observed[donors])
    result["leave_one_out_depth_400plus"] = restricted
    return result


def main() -> None:
    target_table = pd.read_csv(TARGET_TABLE)
    target_order = target_table["target_gene"].astype(str).tolist()
    benchmark = target_table[target_table["benchmark_candidate"]].copy()
    strict_guides = pd.read_csv(STRICT_CONTROLS)["guide_id"].astype(str).tolist()
    data = ad.read_h5ad(H1_PATH, backed="r")
    try:
        genes = data.var_names.astype(str).tolist()
        gene_lookup = {gene: index for index, gene in enumerate(genes)}
        target_indices = [gene_lookup[target] for target in target_order]
        strict_mask = data.obs["target_gene"].astype(str).eq("non-targeting") & data.obs[
            "guide_id"
        ].astype(str).isin(strict_guides)
        strict_rows = np.flatnonzero(strict_mask.to_numpy())
        halves = split_control_halves(data.obs, strict_rows, strict_guides)
        directions, half0, half1, control_mean = fit_covariance_columns(
            data, strict_rows, target_indices, halves
        )
        if half0 is None or half1 is None:
            raise AssertionError("split-half covariance fit did not return both halves")
        control_counts, perturbed_counts = collect_raw_pseudobulks(
            data, target_order, strict_guides
        )
        calibration = add_restricted_loo(
            knockdown_depths(
                target_order,
                target_indices,
                control_counts,
                perturbed_counts,
            ),
            target_table,
        )
        eligible_depths = calibration.loc[
            calibration["eligible_knockdown_donor"], "observed_knockdown_depth"
        ]
        observed_median = eligible_depths.median()
        observed_mad = (eligible_depths - observed_median).abs().median()
        loo_depths = calibration["leave_one_out_knockdown_depth"]
        loo_median = loo_depths.median()
        restricted_change = (
            loo_depths - calibration["leave_one_out_depth_400plus"]
        ).abs()
        calibration_summary = pd.DataFrame(
            {
                "statistic": [
                    "eligible_knockdown_donors",
                    "ineligible_or_sign_failure_donors",
                    "observed_knockdown_depth_median",
                    "observed_knockdown_depth_mad",
                    "observed_mad_over_abs_median",
                    "leave_one_out_depth_median",
                    "leave_one_out_depth_mad",
                    "median_abs_loo_change_restricting_to_400plus",
                    "maximum_abs_loo_change_restricting_to_400plus",
                ],
                "value": [
                    calibration["eligible_knockdown_donor"].sum(),
                    (~calibration["eligible_knockdown_donor"]).sum(),
                    observed_median,
                    observed_mad,
                    observed_mad / abs(observed_median),
                    loo_median,
                    (loo_depths - loo_median).abs().median(),
                    restricted_change.median(),
                    restricted_change.max(),
                ],
            }
        )

        n_benchmark = len(benchmark)
        expected_profiles = np.empty((n_benchmark, data.n_vars), dtype=np.float32)
        null_profiles = np.empty_like(expected_profiles)
        sampled_profiles = np.empty_like(expected_profiles)
        sampled_null_profiles = np.empty_like(expected_profiles)
        source_rows = np.empty((n_benchmark, 400), dtype=np.int64)
        closure_rows = []
        noise_rows = []

        calibration_lookup = calibration.set_index("target_gene")
        full_control_10k = log_pseudobulk(control_counts, CELL_TARGET_SUM)
        full_control_50k = log_pseudobulk(control_counts, BULK_TARGET_SUM)
        perturbed_10k = log_pseudobulk(perturbed_counts, CELL_TARGET_SUM)
        perturbed_50k = log_pseudobulk(perturbed_counts, BULK_TARGET_SUM)

        for benchmark_index, row in enumerate(benchmark.itertuples(index=False)):
            column = int(row.source_order)
            target = str(row.target_gene)
            gene_index = target_indices[column]
            source_rng = np.random.default_rng(
                np.random.SeedSequence([0, 1, MODEL_INDEX, column, 0])
            )
            selected = np.sort(
                sample_balanced_controls(data.obs, strict_guides, source_rng)
            )
            raw = data.X[selected].tocsr()
            source_rows[benchmark_index] = selected
            raw_sum = np.asarray(raw.sum(axis=0)).ravel()
            control_fraction = raw_sum[gene_index] / raw_sum.sum()
            transferred = float(
                calibration_lookup.loc[target, "leave_one_out_knockdown_depth"]
            )
            intended = intended_target_shift(control_fraction, transferred)
            amplitude, expected_shift = solve_output_matched_amplitude(
                raw, directions[:, column], gene_index, intended
            )
            expected_sum = expected_decoded_sum(raw, directions[:, column], amplitude)
            null_sum = expected_decoded_sum(raw, directions[:, column], 0.0)
            expected_profiles[benchmark_index] = log_pseudobulk(expected_sum)
            null_profiles[benchmark_index] = log_pseudobulk(null_sum)

            model_rng = np.random.default_rng(
                np.random.SeedSequence([0, 8, MODEL_INDEX, column, 0, 0])
            )
            null_rng = np.random.default_rng(
                np.random.SeedSequence([0, 8, MODEL_INDEX, column, 0, 1])
            )
            sampled = decode_multinomial(raw, directions[:, column], amplitude, model_rng)
            sampled_null = decode_multinomial(raw, directions[:, column], 0.0, null_rng)
            sampled_profiles[benchmark_index] = log_pseudobulk(sampled)
            sampled_null_profiles[benchmark_index] = log_pseudobulk(sampled_null)
            source_profile = log_pseudobulk(raw_sum)
            sampled_shift = sampled_profiles[benchmark_index, gene_index] - source_profile[
                gene_index
            ]
            closure_rows.append(
                {
                    "target_gene": target,
                    "source_order": column,
                    "observed_knockdown_depth": calibration_lookup.loc[
                        target, "observed_knockdown_depth"
                    ],
                    "leave_one_out_knockdown_depth": transferred,
                    "leave_one_out_depth_400plus": calibration_lookup.loc[
                        target, "leave_one_out_depth_400plus"
                    ],
                    "intended_target_shift": intended,
                    "expected_realized_shift": expected_shift,
                    "sampled_realized_shift": sampled_shift,
                    "amplitude": amplitude,
                    "direction_diagonal": directions[gene_index, column],
                }
            )
            noise_rows.append(
                {
                    "target_gene": target,
                    "model_sample_vs_expected_sq_error": float(
                        np.square(sampled_profiles[benchmark_index] - expected_profiles[benchmark_index]).sum()
                    ),
                    "null_sample_vs_expected_sq_error": float(
                        np.square(sampled_null_profiles[benchmark_index] - null_profiles[benchmark_index]).sum()
                    ),
                    "expected_null_vs_source_max_abs": float(
                        np.max(np.abs(null_profiles[benchmark_index] - source_profile))
                    ),
                }
            )
            print(f"Model 1 expected profile {benchmark_index + 1}/{n_benchmark}: {target}")

        scale_audit = pd.DataFrame(
            {
                "target_gene": target_order,
                "source_order": np.arange(len(target_order)),
                "observed_target_shift_10k": perturbed_10k[
                    np.arange(len(target_order)), target_indices
                ]
                - full_control_10k[target_indices],
                "observed_target_shift_50k": perturbed_50k[
                    np.arange(len(target_order)), target_indices
                ]
                - full_control_50k[target_indices],
            }
        )
        diagnostics = control_fit_diagnostics(
            target_order, target_indices, directions, half0, half1
        )
        batch_centered = fit_batch_centered_columns(data, strict_rows, target_indices)
        batch_rows = []
        for column, (target, gene_index) in enumerate(
            zip(target_order, target_indices, strict=True)
        ):
            mask = np.ones(data.n_vars, dtype=bool)
            mask[gene_index] = False
            batch_rows.append(
                {
                    "target_gene": target,
                    "full_vs_batch_centered_cosine": cosine(
                        directions[mask, column], batch_centered[mask, column]
                    ),
                }
            )
        geometry, nearest = direction_geometry(directions, target_indices, target_order)
    finally:
        data.file.close()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "calibration": REPORT_DIR / "model1_knockdown_calibration.csv",
        "calibration_summary": REPORT_DIR / "calibration_summary.csv",
        "closure": REPORT_DIR / "model1_on_target_closure.csv",
        "scale_audit": REPORT_DIR / "calibration_scale_audit.csv",
        "control_fit": REPORT_DIR / "model1_control_fit_diagnostics.csv",
        "batch": REPORT_DIR / "batch_sensitivity.csv",
        "geometry": REPORT_DIR / "model1_shared_axis_geometry.csv",
        "nearest": REPORT_DIR / "model1_nearest_directions.csv",
        "noise": REPORT_DIR / "generator_noise_decomposition.csv",
    }
    calibration.to_csv(outputs["calibration"], index=False)
    calibration_summary.to_csv(outputs["calibration_summary"], index=False)
    pd.DataFrame(closure_rows).to_csv(outputs["closure"], index=False)
    scale_audit.to_csv(outputs["scale_audit"], index=False)
    diagnostics.to_csv(outputs["control_fit"], index=False)
    pd.DataFrame(batch_rows).to_csv(outputs["batch"], index=False)
    geometry.to_csv(outputs["geometry"], index=False)
    nearest.to_csv(outputs["nearest"], index=False)
    pd.DataFrame(noise_rows).to_csv(outputs["noise"], index=False)

    artifact = DERIVED_DIR / "model1_expected_profiles.npz"
    np.savez_compressed(
        artifact,
        target_gene=np.asarray(benchmark["target_gene"].astype(str).tolist()),
        source_order=benchmark["source_order"].to_numpy(dtype=np.int64),
        gene_names=np.asarray(genes),
        expected_log_bulk=expected_profiles,
        null_log_bulk=null_profiles,
        sampled_log_bulk=sampled_profiles,
        sampled_null_log_bulk=sampled_null_profiles,
        source_rows=source_rows,
        directions=directions.astype(np.float32),
        half0_directions=half0.astype(np.float32),
        half1_directions=half1.astype(np.float32),
        control_mean=control_mean.astype(np.float32),
    )
    manifest = {
        "model": "normalized empirical-covariance linear response",
        "truth_cells_read": False,
        "cell_target_sum": CELL_TARGET_SUM,
        "bulk_target_sum": BULK_TARGET_SUM,
        "control_cells": len(strict_rows),
        "control_guides": len(strict_guides),
        "benchmark_targets": len(benchmark),
        "calibration": "leave-one-target-out median negative log-fraction knockdown; exact expected-decoder output matching",
        "seeds": {
            "source_rows": [0, 1, MODEL_INDEX, "source_order", 0],
            "model_decode": [0, 8, MODEL_INDEX, "source_order", 0, 0],
            "null_decode": [0, 8, MODEL_INDEX, "source_order", 0, 1],
        },
        "inputs": {
            "h1": {"path": relative(H1_PATH), "sha256": H1_SHA256},
            "target_table": {
                "path": relative(TARGET_TABLE),
                "sha256": sha256(TARGET_TABLE),
            },
            "strict_controls": {
                "path": relative(STRICT_CONTROLS),
                "sha256": sha256(STRICT_CONTROLS),
            },
        },
        "artifact": {"path": relative(artifact), "sha256": sha256(artifact)},
        "outputs": {
            name: {"path": relative(path), "sha256": sha256(path)}
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
    (REPORT_DIR / "model1_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
