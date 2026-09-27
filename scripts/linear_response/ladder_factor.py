"""Fit restricted rank-k covariance responses with a diagonal residual."""

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
    log1cp10k,
    pca_covariance_columns,
    split_control_halves,
)

ROOT = Path(__file__).resolve().parents[2]
H1 = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
EMPIRICAL = ROOT / "data/derived/linear_response/ladder/empirical.npz"
STRICT_GUIDES = ROOT / "reports/linear-response-three-models/h1_strict_controls.csv"
DERIVED = ROOT / "data/derived/linear_response/ladder"
REPORT = ROOT / "reports/linear-response-ladder"
RANKS = (10, 25, 50)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_svd(
    data: ad.AnnData,
    rows: np.ndarray,
    feature_indices: np.ndarray,
    mean: np.ndarray,
    name: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    checkpoint = DERIVED / f"factor_svd_{name}.npz"
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            return (
                saved["components"].astype(np.float64),
                saved["singular_values"].astype(np.float64),
                float(saved["squared_norm"]),
            )

    matrix_path = DERIVED / f"factor_controls_{name}.f32"
    matrix = np.memmap(
        matrix_path,
        dtype=np.float32,
        mode="w+",
        shape=(len(rows), len(feature_indices)),
    )
    squared_norm = 0.0
    for start in range(0, len(rows), 512):
        stop = min(start + 512, len(rows))
        values = log1cp10k(data.X[rows[start:stop]].tocsr())[:, feature_indices]
        centered = values - mean
        matrix[start:stop] = centered
        squared_norm += float(np.square(centered).sum())
        print(f"Factor {name} matrix {stop}/{len(rows)}")
    matrix.flush()
    _, singular_values, components = randomized_svd(
        matrix,
        n_components=max(RANKS),
        n_oversamples=10,
        n_iter=3,
        random_state=seed,
        flip_sign=True,
    )
    del matrix
    matrix_path.unlink()
    np.savez_compressed(
        checkpoint,
        components=components.astype(np.float32),
        singular_values=singular_values.astype(np.float32),
        squared_norm=np.asarray(squared_norm),
    )
    return components, singular_values, squared_norm


def factor_response(
    components: np.ndarray,
    singular_values: np.ndarray,
    cells: int,
    empirical_variance: np.ndarray,
    target_indices: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return factor-plus-diagonal response, covariance columns, and residual D."""
    vectors = components[:rank]
    eigenvalues = np.square(singular_values[:rank]) / (cells - 1)
    covariance = pca_covariance_columns(
        vectors, singular_values[:rank], cells, target_indices
    )
    factor_diagonal = np.square(vectors).T @ eigenvalues
    residual = np.maximum(empirical_variance - factor_diagonal, 0.0)
    target_variance = factor_diagonal[target_indices] + residual[target_indices]
    response = -covariance / target_variance
    response[target_indices, np.arange(len(target_indices))] = -1.0
    return response, covariance, residual


def covariance_error(
    predicted: np.ndarray,
    observed: np.ndarray,
    target_indices: np.ndarray,
) -> float:
    ratios = []
    for column, target in enumerate(target_indices):
        keep = np.arange(predicted.shape[0]) != target
        denominator = np.square(observed[keep, column]).sum()
        ratios.append(
            np.square(predicted[keep, column] - observed[keep, column]).sum()
            / denominator
        )
    return float(np.mean(ratios))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else 0.0


def main() -> None:
    with np.load(EMPIRICAL, allow_pickle=False) as empirical:
        targets = empirical["target_gene"]
        fit_genes = empirical["fit_gene"]
        output_genes = empirical["output_gene"]
        feature_indices = empirical["full_gene_index"].astype(int)
        target_indices = empirical["target_fit_index"].astype(int)
        source_rows = empirical["source_rows"]
        half_covariance = [
            empirical["half0_covariance"].astype(np.float64),
            empirical["half1_covariance"].astype(np.float64),
        ]
        variance = empirical["variance"].astype(np.float64)
        half_variance = [
            empirical["half0_variance"].astype(np.float64),
            empirical["half1_variance"].astype(np.float64),
        ]
        mean = empirical["mean"].astype(np.float64)
        half_mean = [
            empirical["half0_mean"].astype(np.float64),
            empirical["half1_mean"].astype(np.float64),
        ]
        full_cells = int(empirical["strict_cells"])
        half_cells = [int(empirical["half0_cells"]), int(empirical["half1_cells"])]

    guides = pd.read_csv(STRICT_GUIDES)["guide_id"].astype(str).tolist()
    data = ad.read_h5ad(H1, backed="r")
    try:
        labels = data.obs["target_gene"].astype(str).to_numpy()
        guide_labels = data.obs["guide_id"].astype(str).to_numpy()
        strict_rows = np.flatnonzero(
            (labels == "non-targeting") & np.isin(guide_labels, guides)
        )
        halves = split_control_halves(data.obs, strict_rows, guides)
        fitted = {
            "full": fit_svd(data, strict_rows, feature_indices, mean, "full", 0),
            "half0": fit_svd(
                data, strict_rows[halves == 0], feature_indices, half_mean[0], "half0", 1
            ),
            "half1": fit_svd(
                data, strict_rows[halves == 1], feature_indices, half_mean[1], "half1", 2
            ),
        }
    finally:
        data.file.close()

    rows = []
    responses = []
    for rank in RANKS:
        full_response, _, residual = factor_response(
            fitted["full"][0],
            fitted["full"][1],
            full_cells,
            variance,
            target_indices,
            rank,
        )
        responses.append(full_response.T)
        half_models = [
            factor_response(
                fitted[f"half{half}"][0],
                fitted[f"half{half}"][1],
                half_cells[half],
                half_variance[half],
                target_indices,
                rank,
            )
            for half in (0, 1)
        ]
        errors = [
            covariance_error(half_models[0][1], half_covariance[1], target_indices),
            covariance_error(half_models[1][1], half_covariance[0], target_indices),
        ]
        cosines = []
        for column, target in enumerate(target_indices):
            keep = np.arange(len(fit_genes)) != target
            cosines.append(
                cosine(
                    half_models[0][0][keep, column],
                    half_models[1][0][keep, column],
                )
            )
        rows.append(
            {
                "rank": rank,
                "heldout_covariance_error_ratio": np.mean(errors),
                "forward_error_ratio": errors[0],
                "reverse_error_ratio": errors[1],
                "control_variance_fraction": np.square(
                    fitted["full"][1][:rank]
                ).sum()
                / fitted["full"][2],
                "median_split_half_response_cosine": np.median(cosines),
                "residual_variance_clipped_fraction": np.mean(residual == 0),
            }
        )

    selection = pd.DataFrame(rows)
    selected_rank = int(
        selection.loc[
            selection["heldout_covariance_error_ratio"].idxmin(), "rank"
        ]
    )
    DERIVED.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    artifact = DERIVED / "factor.npz"
    np.savez_compressed(
        artifact,
        target_gene=targets,
        output_gene=output_genes,
        fit_gene=fit_genes,
        full_gene_index=feature_indices,
        target_fit_index=target_indices,
        source_rows=source_rows,
        ranks=np.asarray(RANKS),
        selected_rank=np.asarray(selected_rank),
        response=np.asarray(responses, dtype=np.float32),
        components=fitted["full"][0].astype(np.float32),
        singular_values=fitted["full"][1].astype(np.float32),
    )
    table = REPORT / "factor_selection.csv"
    selection.to_csv(table, index=False)
    manifest = {
        "model": "restricted randomized-SVD factor covariance plus diagonal residual",
        "truth_cells_read": False,
        "candidate_ranks": list(RANKS),
        "selected_rank": selected_rank,
        "selection": "symmetric held-out control covariance-column error",
        "randomized_svd": {"n_iter": 3, "n_oversamples": 10, "seeds": [0, 1, 2]},
        "inputs": {
            str(EMPIRICAL.relative_to(ROOT)): sha256(EMPIRICAL),
            str(STRICT_GUIDES.relative_to(ROOT)): sha256(STRICT_GUIDES),
        },
        "outputs": {
            str(artifact.relative_to(ROOT)): sha256(artifact),
            str(table.relative_to(ROOT)): sha256(table),
        },
        "software": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ["anndata", "numpy", "pandas", "scikit-learn"]
            },
        },
    }
    (REPORT / "factor_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
