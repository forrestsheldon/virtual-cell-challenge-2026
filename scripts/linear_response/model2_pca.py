"""Fit the truncated-control-PCA response diagnostic and emit H1 profiles."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

from scripts.linear_response.kernel import (
    BULK_TARGET_SUM,
    CELL_TARGET_SUM,
    expected_decoded_sum,
    intended_target_shift,
    log1cp10k,
    log_pseudobulk,
    pca_covariance_columns,
    solve_output_matched_amplitude,
)

ROOT = Path(__file__).resolve().parents[2]
H1_PATH = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
REPORT_DIR = ROOT / "reports/linear-response-three-models"
DERIVED_DIR = ROOT / "data/derived/linear_response"
MODEL1_ARTIFACT = DERIVED_DIR / "model1_expected_profiles.npz"
TARGET_TABLE = REPORT_DIR / "h1_target_counts.csv"
STRICT_CONTROLS = REPORT_DIR / "h1_strict_controls.csv"
CALIBRATION = REPORT_DIR / "model1_knockdown_calibration.csv"
CACHED_DIRECTIONS = DERIVED_DIR / "cipher_h1_direction_cache.npz"
RANKS = (3, 20, 50, 100)
MODEL_RANK = 50
MODEL_INDEX = 1
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


def centered_control_memmap(
    data: ad.AnnData,
    rows: np.ndarray,
    mean: np.ndarray,
    path: Path,
) -> tuple[np.memmap, float]:
    """Materialize one temporary centered matrix for the randomized SVD."""
    matrix = np.memmap(path, dtype=np.float32, mode="w+", shape=(len(rows), data.n_vars))
    squared_norm = 0.0
    for start in range(0, len(rows), 512):
        stop = min(start + 512, len(rows))
        centered = log1cp10k(data.X[rows[start:stop]].tocsr()) - mean
        matrix[start:stop] = centered
        squared_norm += float(np.square(centered).sum())
        print(f"Control matrix {stop}/{len(rows)}")
    matrix.flush()
    return matrix, squared_norm


def spectral_diagnostics(
    components: np.ndarray,
    singular_values: np.ndarray,
    total_squared_norm: float,
    n_cells: int,
    target_table: pd.DataFrame,
    target_indices: list[int],
    full_directions: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    benchmark_mask = target_table["benchmark_candidate"].to_numpy(dtype=bool)
    rows = []
    target_rows = []
    model_directions = None
    for rank in RANKS:
        directions = pca_covariance_columns(
            components[:rank], singular_values[:rank], n_cells, target_indices
        )
        if rank == MODEL_RANK:
            model_directions = directions
        cosines = []
        diagonal_fractions = []
        inflations = []
        for column, (target, gene_index) in enumerate(
            zip(target_table["target_gene"], target_indices, strict=True)
        ):
            mask = np.ones(directions.shape[0], dtype=bool)
            mask[gene_index] = False
            full = full_directions[:, column]
            truncated = directions[:, column]
            raw_cosine = cosine(truncated[mask], full[mask])
            recovered_diagonal_fraction = float(
                truncated[gene_index] / full[gene_index]
            )
            normalized_full = np.linalg.norm(full[mask] / full[gene_index])
            normalized_truncated = np.linalg.norm(
                truncated[mask] / truncated[gene_index]
            )
            inflation = float(normalized_truncated / normalized_full)
            cosines.append(raw_cosine)
            diagonal_fractions.append(recovered_diagonal_fraction)
            inflations.append(inflation)
            target_rows.append(
                {
                    "rank": rank,
                    "target_gene": target,
                    "benchmark_target": benchmark_mask[column],
                    "full_direction_cosine": raw_cosine,
                    "recovered_diagonal_fraction": recovered_diagonal_fraction,
                    "target_normalized_downstream_norm_inflation": inflation,
                }
            )
        rows.append(
            {
                "rank": rank,
                "control_variance_fraction": float(
                    np.square(singular_values[:rank]).sum() / total_squared_norm
                ),
                "median_full_direction_cosine": float(np.median(cosines)),
                "median_recovered_diagonal_fraction": float(
                    np.median(diagonal_fractions)
                ),
                "median_target_normalized_downstream_norm_inflation": float(
                    np.median(inflations)
                ),
            }
        )
    if model_directions is None:
        raise AssertionError("rank-50 directions were not constructed")
    return pd.DataFrame(rows), pd.DataFrame(target_rows), model_directions


def main() -> None:
    target_table = pd.read_csv(TARGET_TABLE)
    target_order = target_table["target_gene"].astype(str).tolist()
    benchmark = target_table[target_table["benchmark_candidate"]].copy()
    strict_guides = pd.read_csv(STRICT_CONTROLS)["guide_id"].astype(str).tolist()
    calibration = pd.read_csv(CALIBRATION).set_index("target_gene")
    with np.load(MODEL1_ARTIFACT) as model1:
        full_directions = model1["directions"].astype(np.float64)
        control_mean = model1["control_mean"].astype(np.float64)
        source_rows = model1["source_rows"].copy()
        model1_targets = model1["target_gene"].astype(str).tolist()
        model1_genes = model1["gene_names"].astype(str).tolist()
    if model1_targets != benchmark["target_gene"].astype(str).tolist():
        raise ValueError("Model 1 target order differs from the benchmark")

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    work_matrix_path = DERIVED_DIR / "model2_centered_controls.f32"
    data = ad.read_h5ad(H1_PATH, backed="r")
    try:
        genes = data.var_names.astype(str).tolist()
        if genes != model1_genes:
            raise ValueError("Model 1 gene axis differs from H1")
        gene_lookup = {gene: index for index, gene in enumerate(genes)}
        target_indices = [gene_lookup[target] for target in target_order]
        strict_mask = data.obs["target_gene"].astype(str).eq("non-targeting") & data.obs[
            "guide_id"
        ].astype(str).isin(strict_guides)
        strict_rows = np.flatnonzero(strict_mask.to_numpy())
        work_matrix, total_squared_norm = centered_control_memmap(
            data, strict_rows, control_mean, work_matrix_path
        )
        _, singular_values, components = randomized_svd(
            work_matrix,
            n_components=max(RANKS),
            n_oversamples=20,
            n_iter=4,
            random_state=0,
            flip_sign=True,
        )
        spectral, per_target_spectral, directions = spectral_diagnostics(
            components,
            singular_values,
            total_squared_norm,
            len(strict_rows),
            target_table,
            target_indices,
            full_directions,
        )

        n_benchmark = len(benchmark)
        expected_profiles = np.empty((n_benchmark, data.n_vars), dtype=np.float32)
        null_profiles = np.empty_like(expected_profiles)
        closure_rows = []
        for benchmark_index, row in enumerate(benchmark.itertuples(index=False)):
            column = int(row.source_order)
            target = str(row.target_gene)
            gene_index = target_indices[column]
            selected = np.sort(source_rows[benchmark_index])
            raw = data.X[selected].tocsr()
            raw_sum = np.asarray(raw.sum(axis=0)).ravel()
            transferred = float(
                calibration.loc[target, "leave_one_out_knockdown_depth"]
            )
            intended = intended_target_shift(
                raw_sum[gene_index] / raw_sum.sum(), transferred
            )
            amplitude, realized = solve_output_matched_amplitude(
                raw, directions[:, column], gene_index, intended
            )
            expected_sum = expected_decoded_sum(raw, directions[:, column], amplitude)
            null_sum = expected_decoded_sum(raw, directions[:, column], 0.0)
            expected_profiles[benchmark_index] = log_pseudobulk(expected_sum)
            null_profiles[benchmark_index] = log_pseudobulk(null_sum)
            closure_rows.append(
                {
                    "target_gene": target,
                    "source_order": column,
                    "observed_knockdown_depth": calibration.loc[
                        target, "observed_knockdown_depth"
                    ],
                    "leave_one_out_knockdown_depth": transferred,
                    "intended_target_shift": intended,
                    "expected_realized_shift": realized,
                    "amplitude": amplitude,
                    "direction_diagonal": directions[gene_index, column],
                }
            )
            print(f"Model 2 expected profile {benchmark_index + 1}/{n_benchmark}: {target}")
    finally:
        data.file.close()
        del work_matrix
        work_matrix_path.unlink(missing_ok=True)

    cached = np.load(CACHED_DIRECTIONS)["directions"]
    positive = []
    for column, target in enumerate(target_order):
        positive.append(
            {
                "target_gene": target,
                "recomputed_vs_cached_covariance_cosine": cosine(
                    full_directions[:, column], cached[:, column]
                ),
            }
        )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "spectral": REPORT_DIR / "model2_spectral_truncation.csv",
        "per_target": REPORT_DIR / "model2_spectral_per_target.csv",
        "closure": REPORT_DIR / "model2_on_target_closure.csv",
        "positive_control": REPORT_DIR / "model2_covariance_positive_control.csv",
    }
    spectral.to_csv(outputs["spectral"], index=False)
    per_target_spectral.to_csv(outputs["per_target"], index=False)
    pd.DataFrame(closure_rows).to_csv(outputs["closure"], index=False)
    pd.DataFrame(positive).to_csv(outputs["positive_control"], index=False)

    artifact = DERIVED_DIR / "model2_expected_profiles.npz"
    np.savez_compressed(
        artifact,
        target_gene=np.asarray(benchmark["target_gene"].astype(str).tolist()),
        source_order=benchmark["source_order"].to_numpy(dtype=np.int64),
        gene_names=np.asarray(genes),
        expected_log_bulk=expected_profiles,
        null_log_bulk=null_profiles,
        source_rows=source_rows,
        directions=directions.astype(np.float32),
        singular_values=singular_values.astype(np.float32),
        components=components.astype(np.float32),
        total_squared_norm=np.asarray(total_squared_norm),
    )
    manifest = {
        "model": "rank-50 truncated control-PCA covariance response",
        "truth_cells_read": False,
        "rank": MODEL_RANK,
        "candidate_ranks": list(RANKS),
        "cell_target_sum": CELL_TARGET_SUM,
        "bulk_target_sum": BULK_TARGET_SUM,
        "randomized_svd": {
            "random_state": 0,
            "n_oversamples": 20,
            "n_iter": 4,
        },
        "calibration": "Model 1 leave-one-target-out knockdown; exact expected-decoder output matching",
        "source_rows": "identical to Model 1",
        "inputs": {
            "h1": {"path": relative(H1_PATH), "sha256": H1_SHA256},
            "model1_artifact": {
                "path": relative(MODEL1_ARTIFACT),
                "sha256": sha256(MODEL1_ARTIFACT),
            },
            "calibration": {
                "path": relative(CALIBRATION),
                "sha256": sha256(CALIBRATION),
            },
        },
        "cached_positive_control": {
            "path": relative(CACHED_DIRECTIONS),
            "sha256": sha256(CACHED_DIRECTIONS),
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
            "scikit-learn": version("scikit-learn"),
            "scipy": version("scipy"),
        },
    }
    (REPORT_DIR / "model2_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
