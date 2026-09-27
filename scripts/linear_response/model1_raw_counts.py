"""Raw-count covariance ablation for the normalized CIPHER-style response.

This is a preprocessing negative control, not a submission model. It replaces
Model 1's per-cell log1p10k covariance with naive covariance of unnormalized
counts, while holding its target calibration, source rows, and decoder fixed.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scripts.linear_response.kernel import (
    expected_decoded_sum,
    intended_target_shift,
    log_pseudobulk,
    solve_output_matched_amplitude,
    split_control_halves,
)

ROOT = Path(__file__).resolve().parents[2]
H1_PATH = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
REPORT = ROOT / "reports/linear-response-three-models"
DERIVED = ROOT / "data/derived/linear_response"
TARGET_TABLE = REPORT / "h1_target_counts.csv"
STRICT_CONTROLS = REPORT / "h1_strict_controls.csv"
CALIBRATION = REPORT / "model1_knockdown_calibration.csv"
MODEL1 = DERIVED / "model1_expected_profiles.npz"
ARTIFACT = DERIVED / "model1_raw_count_expected_profiles.npz"
CHUNK_SIZE = 1_024


@dataclass
class RawMoments:
    n: int
    total: np.ndarray
    cross: np.ndarray

    @classmethod
    def zeros(cls, genes: int, targets: int) -> RawMoments:
        return cls(0, np.zeros(genes), np.zeros((genes, targets)))

    def update(self, values, target_indices: np.ndarray) -> None:
        self.n += values.shape[0]
        self.total += np.asarray(values.sum(0)).ravel()
        self.cross += (values.T @ values[:, target_indices]).toarray()

    def covariance(self, target_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = self.total / self.n
        centered = (
            self.cross - np.outer(self.total, self.total[target_indices]) / self.n
        )
        return centered / (self.n - 1), mean


def fit_raw_covariance_columns(
    data: ad.AnnData,
    rows: np.ndarray,
    target_indices: np.ndarray,
    halves: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream naive raw-count covariance columns and two fixed split halves."""
    moments = [RawMoments.zeros(data.n_vars, len(target_indices)) for _ in range(3)]
    for start in range(0, len(rows), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(rows))
        values = data.X[rows[start:stop]].tocsr().astype(np.float64)
        moments[0].update(values, target_indices)
        chunk_halves = halves[start:stop]
        moments[1].update(values[chunk_halves == 0], target_indices)
        moments[2].update(values[chunk_halves == 1], target_indices)
    full, mean = moments[0].covariance(target_indices)
    first, _ = moments[1].covariance(target_indices)
    second, _ = moments[2].covariance(target_indices)
    return full, first, second, mean


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else 0.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    target_table = pd.read_csv(TARGET_TABLE)
    target_order = target_table["target_gene"].astype(str).tolist()
    benchmark = target_table[target_table["benchmark_candidate"]].copy()
    strict_guides = pd.read_csv(STRICT_CONTROLS)["guide_id"].astype(str).tolist()
    calibration = pd.read_csv(CALIBRATION).set_index("target_gene")
    with np.load(MODEL1, allow_pickle=False) as normalized:
        genes = normalized["gene_names"].astype(str).tolist()
        normalized_targets = normalized["target_gene"].astype(str).tolist()
        normalized_source_rows = normalized["source_rows"]
        normalized_directions = normalized["directions"].astype(np.float64)
    if normalized_targets != benchmark["target_gene"].astype(str).tolist():
        raise ValueError("raw ablation and normalized Model 1 target orders differ")

    data = ad.read_h5ad(H1_PATH, backed="r")
    try:
        if data.var_names.astype(str).tolist() != genes:
            raise ValueError("raw ablation and normalized Model 1 gene axes differ")
        lookup = {gene: index for index, gene in enumerate(genes)}
        target_indices = np.asarray([lookup[target] for target in target_order])
        strict = data.obs["target_gene"].astype(str).eq("non-targeting") & data.obs[
            "guide_id"
        ].astype(str).isin(strict_guides)
        strict_rows = np.flatnonzero(strict.to_numpy())
        halves = split_control_halves(data.obs, strict_rows, strict_guides)
        directions, half0, half1, raw_mean = fit_raw_covariance_columns(
            data, strict_rows, target_indices, halves
        )
        expected = np.empty((len(benchmark), data.n_vars), dtype=np.float32)
        null = np.empty_like(expected)
        closure = []
        diagnostics = []
        for output_index, row in enumerate(benchmark.itertuples(index=False)):
            target = str(row.target_gene)
            column = int(row.source_order)
            gene_index = target_indices[column]
            selected = np.sort(normalized_source_rows[output_index])
            raw = data.X[selected].tocsr()
            raw_sum = np.asarray(raw.sum(0)).ravel()
            intended = intended_target_shift(
                raw_sum[gene_index] / raw_sum.sum(),
                float(calibration.loc[target, "leave_one_out_knockdown_depth"]),
            )
            amplitude, realized = solve_output_matched_amplitude(
                raw, directions[:, column], gene_index, intended
            )
            expected[output_index] = log_pseudobulk(
                expected_decoded_sum(raw, directions[:, column], amplitude)
            )
            null[output_index] = log_pseudobulk(
                expected_decoded_sum(raw, directions[:, column], 0.0)
            )
            mask = np.ones(data.n_vars, dtype=bool)
            mask[gene_index] = False
            raw_downstream = directions[mask, column]
            normalized_downstream = normalized_directions[mask, column]
            raw_diagonal = directions[gene_index, column]
            normalized_diagonal = normalized_directions[gene_index, column]
            closure.append(
                {
                    "target_gene": target,
                    "source_order": column,
                    "intended_target_shift": intended,
                    "expected_realized_shift": realized,
                    "amplitude": amplitude,
                    "direction_diagonal": raw_diagonal,
                    "sampled_realized_shift": np.nan,
                }
            )
            diagnostics.append(
                {
                    "target_gene": target,
                    "source_order": column,
                    "split_half_downstream_cosine": cosine(
                        half0[mask, column], half1[mask, column]
                    ),
                    "normalized_model1_downstream_cosine": cosine(
                        raw_downstream, normalized_downstream
                    ),
                    "raw_direction_diagonal_fraction": abs(raw_diagonal)
                    / max(np.linalg.norm(directions[:, column]), 1e-15),
                    "poisson_mean_over_raw_diagonal": raw_mean[gene_index]
                    / max(raw_diagonal, 1e-15),
                    "downstream_mean_expression_cosine": cosine(
                        raw_downstream, raw_mean[mask]
                    ),
                    "raw_target_normalized_downstream_norm": np.linalg.norm(
                        raw_downstream / raw_diagonal
                    ),
                    "normalized_target_normalized_downstream_norm": np.linalg.norm(
                        normalized_downstream / normalized_diagonal
                    ),
                }
            )
    finally:
        data.file.close()

    DERIVED.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ARTIFACT,
        target_gene=benchmark["target_gene"].astype(str).to_numpy(),
        source_order=benchmark["source_order"].to_numpy(dtype=np.int64),
        gene_names=np.asarray(genes),
        expected_log_bulk=expected,
        null_log_bulk=null,
        source_rows=normalized_source_rows,
        directions=directions.astype(np.float32),
        half0_directions=half0.astype(np.float32),
        half1_directions=half1.astype(np.float32),
        raw_control_mean=raw_mean.astype(np.float32),
    )
    outputs = {
        "closure": REPORT / "model1_raw_count_on_target_closure.csv",
        "diagnostics": REPORT / "model1_raw_count_diagnostics.csv",
    }
    pd.DataFrame(closure).to_csv(outputs["closure"], index=False)
    pd.DataFrame(diagnostics).to_csv(outputs["diagnostics"], index=False)
    manifest = {
        "model": "naive raw-count empirical-covariance preprocessing ablation",
        "status": "diagnostic_only",
        "truth_cells_read": False,
        "covariance_representation": "unnormalized untransformed per-cell counts",
        "held_fixed_from_model1": [
            "strict control pool",
            "target order",
            "source rows",
            "leave-one-target-out knockdown depth",
            "exact expected multinomial decoder",
        ],
        "known_confounders": [
            "library-size covariance",
            "Poisson sampling variance on the diagonal",
        ],
        "artifact": {"path": relative(ARTIFACT), "sha256": sha256(ARTIFACT)},
        "inputs": {
            relative(path): sha256(path)
            for path in [H1_PATH, TARGET_TABLE, STRICT_CONTROLS, CALIBRATION, MODEL1]
        },
        "outputs": {
            name: {"path": relative(path), "sha256": sha256(path)}
            for name, path in outputs.items()
        },
        "software": {
            "python": platform.python_version(),
            **{name: version(name) for name in ["anndata", "numpy", "pandas", "scipy"]},
        },
    }
    (REPORT / "model1_raw_count_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
