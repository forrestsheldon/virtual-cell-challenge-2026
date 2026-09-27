"""Fit a standard-error-thresholded empirical response."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.linear_response.kernel import normalize_response_columns

ROOT = Path(__file__).resolve().parents[2]
EMPIRICAL = ROOT / "data/derived/linear_response/ladder/empirical.npz"
DERIVED = ROOT / "data/derived/linear_response/ladder"
REPORT = ROOT / "reports/linear-response-ladder"
TAUS = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regression_coefficients(
    covariance: np.ndarray,
    variance: np.ndarray,
    target_indices: np.ndarray,
    cells: int,
    tau: float,
) -> np.ndarray:
    """Simple-regression MAP coefficients under a Laplace prior scaled by SE."""
    target_variance = variance[target_indices]
    coefficients = covariance / target_variance
    residual = np.maximum(
        variance[:, None] - np.square(covariance) / target_variance, 0.0
    )
    standard_error = np.sqrt(
        residual / ((cells - 2) * target_variance[None, :])
    )
    coefficients = np.sign(coefficients) * np.maximum(
        np.abs(coefficients) - tau * standard_error, 0.0
    )
    coefficients[target_indices, np.arange(len(target_indices))] = 1.0
    return coefficients


def heldout_error(
    coefficients: np.ndarray,
    covariance: np.ndarray,
    variance: np.ndarray,
    target_indices: np.ndarray,
) -> float:
    """Equal-target residual-variance ratio on an independent control half."""
    ratios = []
    for column, target in enumerate(target_indices):
        keep = np.arange(len(variance)) != target
        residual = (
            variance
            - 2 * coefficients[:, column] * covariance[:, column]
            + np.square(coefficients[:, column]) * variance[target]
        )
        ratios.append(residual[keep].sum() / variance[keep].sum())
    return float(np.mean(ratios))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else 0.0


def main() -> None:
    with np.load(EMPIRICAL, allow_pickle=False) as empirical:
        targets = empirical["target_gene"]
        fit_genes = empirical["fit_gene"]
        output_genes = empirical["output_gene"]
        target_indices = empirical["target_fit_index"].astype(int)
        source_rows = empirical["source_rows"]
        covariance = empirical["covariance"].astype(np.float64)
        half_covariance = [
            empirical["half0_covariance"].astype(np.float64),
            empirical["half1_covariance"].astype(np.float64),
        ]
        variance = empirical["variance"].astype(np.float64)
        half_variance = [
            empirical["half0_variance"].astype(np.float64),
            empirical["half1_variance"].astype(np.float64),
        ]
        half_cells = [
            int(empirical["half0_cells"]),
            int(empirical["half1_cells"]),
        ]
        full_cells = int(empirical["strict_cells"])

    rows = []
    for tau in TAUS:
        coefficients = [
            regression_coefficients(
                half_covariance[half],
                half_variance[half],
                target_indices,
                half_cells[half],
                tau,
            )
            for half in (0, 1)
        ]
        errors = [
            heldout_error(
                coefficients[0], half_covariance[1], half_variance[1], target_indices
            ),
            heldout_error(
                coefficients[1], half_covariance[0], half_variance[0], target_indices
            ),
        ]
        cosines = []
        for target_index, target in enumerate(target_indices):
            keep = np.arange(len(fit_genes)) != target
            cosines.append(
                cosine(
                    coefficients[0][keep, target_index],
                    coefficients[1][keep, target_index],
                )
            )
        downstream = np.ones_like(coefficients[0], dtype=bool)
        downstream[target_indices, np.arange(len(target_indices))] = False
        rows.append(
            {
                "tau": tau,
                "heldout_error_ratio": np.mean(errors),
                "forward_error_ratio": errors[0],
                "reverse_error_ratio": errors[1],
                "downstream_zero_fraction": np.mean(
                    coefficients[0][downstream] == 0
                ),
                "median_split_half_cosine": np.median(cosines),
            }
        )

    selection = pd.DataFrame(rows)
    selected_tau = float(
        selection.loc[selection["heldout_error_ratio"].idxmin(), "tau"]
    )
    selected_coefficients = regression_coefficients(
        covariance, variance, target_indices, full_cells, selected_tau
    )
    response, diagonal = normalize_response_columns(
        selected_coefficients, target_indices
    )
    if not np.allclose(diagonal, 1):
        raise AssertionError("sparse target coefficients changed before normalization")

    DERIVED.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    artifact = DERIVED / "sparse.npz"
    np.savez_compressed(
        artifact,
        target_gene=targets,
        output_gene=output_genes,
        fit_gene=fit_genes,
        target_fit_index=target_indices,
        source_rows=source_rows,
        response=response.T.astype(np.float32),
        selected_tau=np.asarray(selected_tau),
    )
    table = REPORT / "sparse_control_selection.csv"
    selection.to_csv(table, index=False)
    manifest = {
        "model": "SE-scaled Laplace MAP response",
        "truth_cells_read": False,
        "tau_grid": TAUS.tolist(),
        "selected_tau": selected_tau,
        "selection": "symmetric held-out-control residual variance ratio",
        "inputs": {str(EMPIRICAL.relative_to(ROOT)): sha256(EMPIRICAL)},
        "outputs": {
            str(artifact.relative_to(ROOT)): sha256(artifact),
            str(table.relative_to(ROOT)): sha256(table),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
        },
    }
    (REPORT / "sparse_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
