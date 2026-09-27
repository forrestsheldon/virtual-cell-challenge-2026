"""Fit the restricted-panel empirical response and its two sensitivities."""

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
from scipy import sparse

from scripts.linear_response.kernel import (
    log1cp10k,
    normalize_response_columns,
    split_control_halves,
)

ROOT = Path(__file__).resolve().parents[2]
H1 = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
PANEL = ROOT / "reports/crispri-h1-exploration/generated/celleval2_shift_genes.csv"
MODEL1 = ROOT / "data/derived/linear_response/model1_expected_profiles.npz"
STRICT_GUIDES = ROOT / "reports/linear-response-three-models/h1_strict_controls.csv"
DERIVED = ROOT / "data/derived/linear_response/ladder"
REPORT = ROOT / "reports/linear-response-ladder"
H1_SHA256 = "a09977104fefb622368ca74b50c9d3c1e891733e6c83db07acfca49b0219c02b"


@dataclass
class Moments:
    n: int
    total: np.ndarray
    squares: np.ndarray
    cross: np.ndarray | None

    @classmethod
    def empty(cls, genes: int, targets: int, *, covariance: bool) -> Moments:
        cross = np.zeros((genes, targets)) if covariance else None
        return cls(0, np.zeros(genes), np.zeros(genes), cross)

    def update(self, values, targets: np.ndarray) -> None:
        self.n += values.shape[0]
        self.total += np.asarray(values.sum(axis=0)).ravel()
        if sparse.issparse(values):
            self.squares += np.asarray(values.power(2).sum(axis=0)).ravel()
        else:
            self.squares += np.square(values).sum(axis=0)
        if self.cross is not None:
            product = values.T @ values[:, targets]
            self.cross += product.toarray() if sparse.issparse(product) else product

    def finish(self, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        variance = (self.squares - np.square(self.total) / self.n) / (self.n - 1)
        if self.cross is None:
            return variance, None
        covariance = (
            self.cross - np.outer(self.total, self.total[targets]) / self.n
        ) / (self.n - 1)
        return variance, covariance


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else 0.0


def main() -> None:
    panel = sorted(pd.read_csv(PANEL)["gene"].astype(str))
    strict_guides = pd.read_csv(STRICT_GUIDES)["guide_id"].astype(str).tolist()
    with np.load(MODEL1, allow_pickle=False) as old:
        targets = old["target_gene"].astype(str)
        source_order = old["source_order"].astype(int)
        source_rows = old["source_rows"].astype(np.int64)
        old_genes = old["gene_names"].astype(str).tolist()
        strict_covariance = old["directions"].astype(np.float64)[:, source_order]
        half0_covariance = old["half0_directions"].astype(np.float64)[:, source_order]
        half1_covariance = old["half1_directions"].astype(np.float64)[:, source_order]

    fit_genes = [*panel, *(target for target in targets if target not in set(panel))]
    old_lookup = {gene: index for index, gene in enumerate(old_genes)}
    fit_indices = np.asarray([old_lookup[gene] for gene in fit_genes])
    target_local = np.asarray([fit_genes.index(target) for target in targets])
    strict_covariance = strict_covariance[fit_indices]
    half0_covariance = half0_covariance[fit_indices]
    half1_covariance = half1_covariance[fit_indices]

    data = ad.read_h5ad(H1, backed="r")
    try:
        labels = data.obs["target_gene"].astype(str).to_numpy()
        guides = data.obs["guide_id"].astype(str).to_numpy()
        control_rows = np.flatnonzero(labels == "non-targeting")
        strict_rows = np.flatnonzero(
            (labels == "non-targeting") & np.isin(guides, strict_guides)
        )
        halves = split_control_halves(data.obs, strict_rows, strict_guides)
        half_by_row = np.full(data.n_obs, -1, dtype=np.int8)
        half_by_row[strict_rows] = halves

        all_normalized = Moments.empty(len(fit_genes), len(targets), covariance=True)
        strict_normalized = Moments.empty(
            len(fit_genes), len(targets), covariance=False
        )
        half_normalized = [
            Moments.empty(len(fit_genes), len(targets), covariance=False)
            for _ in range(2)
        ]
        strict_raw = Moments.empty(len(fit_genes), len(targets), covariance=True)

        for start in range(0, len(control_rows), 512):
            rows = control_rows[start : start + 512]
            raw = data.X[rows].tocsr().astype(np.float64)
            normalized = log1cp10k(raw)[:, fit_indices]
            all_normalized.update(normalized, target_local)
            take = half_by_row[rows] >= 0
            if take.any():
                strict_values = normalized[take]
                strict_normalized.update(strict_values, target_local)
                strict_raw.update(raw[take][:, fit_indices], target_local)
                for half in (0, 1):
                    in_half = half_by_row[rows[take]] == half
                    half_normalized[half].update(
                        strict_values[in_half], target_local
                    )
            print(f"Empirical moments {min(start + 512, len(control_rows))}/{len(control_rows)}")
    finally:
        data.file.close()

    strict_variance, _ = strict_normalized.finish(target_local)
    half0_variance, _ = half_normalized[0].finish(target_local)
    half1_variance, _ = half_normalized[1].finish(target_local)
    all_variance, all_covariance = all_normalized.finish(target_local)
    raw_variance, raw_covariance = strict_raw.finish(target_local)
    assert all_covariance is not None and raw_covariance is not None

    strict_response, strict_diagonal = normalize_response_columns(
        strict_covariance, target_local
    )
    half0_response, _ = normalize_response_columns(half0_covariance, target_local)
    half1_response, _ = normalize_response_columns(half1_covariance, target_local)
    all_response, all_diagonal = normalize_response_columns(
        all_covariance, target_local
    )
    raw_response, raw_diagonal = normalize_response_columns(
        raw_covariance, target_local
    )

    diagnostics = []
    for index, target in enumerate(targets):
        downstream = np.arange(len(fit_genes)) != target_local[index]
        diagnostics.append(
            {
                "target_gene": target,
                "strict_variance": strict_diagonal[index],
                "all31_variance": all_diagonal[index],
                "raw_variance": raw_diagonal[index],
                "strict_split_half_cosine": cosine(
                    half0_response[downstream, index],
                    half1_response[downstream, index],
                ),
                "all31_vs_strict_cosine": cosine(
                    all_response[downstream, index],
                    strict_response[downstream, index],
                ),
                "raw_vs_normalized_cosine": cosine(
                    raw_response[downstream, index],
                    strict_response[downstream, index],
                ),
            }
        )

    DERIVED.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    artifact = DERIVED / "empirical.npz"
    np.savez_compressed(
        artifact,
        target_gene=targets,
        output_gene=np.asarray(panel),
        fit_gene=np.asarray(fit_genes),
        full_gene_index=fit_indices,
        target_fit_index=target_local,
        source_rows=source_rows,
        strict_cells=np.asarray(strict_normalized.n),
        half0_cells=np.asarray(half_normalized[0].n),
        half1_cells=np.asarray(half_normalized[1].n),
        covariance=strict_covariance.astype(np.float32),
        half0_covariance=half0_covariance.astype(np.float32),
        half1_covariance=half1_covariance.astype(np.float32),
        variance=strict_variance.astype(np.float32),
        half0_variance=half0_variance.astype(np.float32),
        half1_variance=half1_variance.astype(np.float32),
        mean=(strict_normalized.total / strict_normalized.n).astype(np.float32),
        half0_mean=(half_normalized[0].total / half_normalized[0].n).astype(
            np.float32
        ),
        half1_mean=(half_normalized[1].total / half_normalized[1].n).astype(
            np.float32
        ),
        all31_variance=all_variance.astype(np.float32),
        raw_variance=raw_variance.astype(np.float32),
        response=strict_response.T.astype(np.float32),
        all31_response=all_response.T.astype(np.float32),
        raw_response=raw_response.T.astype(np.float32),
    )
    table = REPORT / "empirical_fit_diagnostics.csv"
    pd.DataFrame(diagnostics).to_csv(table, index=False)
    manifest = {
        "model": "restricted empirical covariance response",
        "truth_cells_read": False,
        "controls": {"strict": len(strict_rows), "all": len(control_rows)},
        "genes": {"output": len(panel), "fit": len(fit_genes)},
        "targets": len(targets),
        "normalization": "per-cell log1p CP10K; raw-count sensitivity unnormalized",
        "response": "minus covariance column divided by target variance",
        "inputs": {
            str(H1.relative_to(ROOT)): H1_SHA256,
            str(PANEL.relative_to(ROOT)): sha256(PANEL),
            str(MODEL1.relative_to(ROOT)): sha256(MODEL1),
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
                for name in ["anndata", "numpy", "pandas", "scipy"]
            },
        },
    }
    (REPORT / "empirical_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
