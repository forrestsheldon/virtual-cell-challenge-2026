"""Audit the local datasets and freeze the linear-response benchmark split."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
H1_DIR = ROOT / "data/external/vcc2025_h1"
EXPLORATION_DIR = ROOT / "reports/crispri-h1-exploration/generated"
REPORT_DIR = ROOT / "reports/linear-response-three-models"
REFERENCE_CELLS = ROOT / "reports/vcc2026-h1/reference_cells.csv"
REFERENCE_MANIFEST = ROOT / "reports/vcc2026-h1/benchmark_manifest.json"

H1_CELLS = 221_273
H1_GENES = 18_080
H1_TARGETS = 150
H1_SHA256 = "a09977104fefb622368ca74b50c9d3c1e891733e6c83db07acfca49b0219c02b"
CONTROL_LABEL = "non-targeting"
CELLS_PER_TARGET = 400


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_h1_genes(path: Path) -> list[str]:
    genes = pd.read_csv(path, header=None).iloc[:, 0].astype(str).tolist()
    assert len(genes) == H1_GENES
    assert len(genes) == len(set(genes))
    return genes


def h1_target_counts(obs: pd.DataFrame, target_order: list[str]) -> pd.DataFrame:
    labels = obs["target_gene"].astype(str)
    target_cells = labels[labels != CONTROL_LABEL]
    counts = target_cells.value_counts()
    assert set(counts.index) == set(target_order)
    return pd.DataFrame(
        {
            "target_gene": target_order,
            "source_order": range(len(target_order)),
            "n_cells": [int(counts[target]) for target in target_order],
            "calibration_candidate": True,
            "benchmark_candidate": [counts[target] >= 400 for target in target_order],
        }
    )


def strict_control_counts(
    obs: pd.DataFrame, strict_guides: list[str]
) -> pd.DataFrame:
    controls = obs[
        (obs["target_gene"] == CONTROL_LABEL) & obs["guide_id"].isin(strict_guides)
    ]
    grouped = controls.groupby("guide_id", observed=True)
    summary = pd.DataFrame(
        {
            "guide_id": strict_guides,
            "n_cells": grouped.size().reindex(strict_guides).astype(int).tolist(),
            "n_batches": grouped["batch"]
            .nunique()
            .reindex(strict_guides)
            .astype(int)
            .tolist(),
        }
    )
    assert summary["n_cells"].sum() == len(controls)
    return summary


def audit_h1() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    h5ad_path = H1_DIR / "adata_Training.h5ad"
    gene_path = H1_DIR / "gene_names.csv"
    target_path = H1_DIR / "pert_counts_Training.csv"
    strict_path = EXPLORATION_DIR / "mixscape_strict_null_guides.csv"
    strict_run_path = EXPLORATION_DIR / "mixscape_strict_null_run.json"

    genes = read_h1_genes(gene_path)
    target_source = pd.read_csv(target_path)
    target_order = target_source["target_gene"].astype(str).tolist()
    assert len(target_order) == H1_TARGETS
    assert len(target_order) == len(set(target_order))
    strict_guides = pd.read_csv(strict_path)["pseudo_guide"].astype(str).tolist()
    excluded_guides = json.loads(strict_run_path.read_text())["excluded_guides"]
    assert len(strict_guides) == 26
    assert len(excluded_guides) == 5

    data = ad.read_h5ad(h5ad_path, backed="r")
    try:
        assert data.shape == (H1_CELLS, H1_GENES)
        assert data.var_names.astype(str).tolist() == genes
        assert data.X.format == "csr"
        assert str(data.X.dtype) == "float32"
        counts = h1_target_counts(data.obs, target_order)
        assert counts["n_cells"].tolist() == target_source["n_cells"].astype(int).tolist()
        assert counts["benchmark_candidate"].sum() == 126
        strict = strict_control_counts(data.obs, strict_guides)
        assert strict["n_cells"].sum() == 32_616
        assert strict["n_batches"].eq(48).all()
        authoritative = json.loads(REFERENCE_MANIFEST.read_text())
        assert authoritative["inputs"]["h1"]["sha256"] == H1_SHA256
        metadata = {
            "source_url": "gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/",
            "accessed": "2026-08-30",
            "h5ad": {
                "path": relative(h5ad_path),
                "bytes": h5ad_path.stat().st_size,
                "sha256": H1_SHA256,
                "hash_verified_by": relative(REFERENCE_MANIFEST),
                "shape": list(data.shape),
                "obs_columns": data.obs.columns.tolist(),
                "var_columns": data.var.columns.tolist(),
                "representation": "sparse raw UMI counts",
                "x_format": data.X.format,
                "x_dtype": str(data.X.dtype),
            },
            "gene_axis": {
                "path": relative(gene_path),
                "count": len(genes),
                "sha256": sha256(gene_path),
                "matches_var_names": True,
            },
            "targets": {
                "path": relative(target_path),
                "count": len(target_order),
                "sha256": sha256(target_path),
                "order": target_order,
            },
            "batches": int(data.obs["batch"].nunique()),
            "controls": {
                "all_ntc_cells": int((data.obs["target_gene"] == CONTROL_LABEL).sum()),
                "all_ntc_guides": int(
                    data.obs.loc[
                        data.obs["target_gene"] == CONTROL_LABEL, "guide_id"
                    ].nunique()
                ),
                "strict_manifest": relative(strict_path),
                "strict_manifest_sha256": sha256(strict_path),
                "strict_guides": strict_guides,
                "strict_cells": int(strict["n_cells"].sum()),
                "strict_batches": int(
                    data.obs.loc[
                        data.obs["guide_id"].isin(strict_guides), "batch"
                    ].nunique()
                ),
                "excluded_manifest": relative(strict_run_path),
                "excluded_manifest_sha256": sha256(strict_run_path),
                "excluded_guides": excluded_guides,
            },
            "cohorts": {
                "calibration_targets": int(counts["calibration_candidate"].sum()),
                "benchmark_targets": int(counts["benchmark_candidate"].sum()),
                "reference_cells": "owned by reports/vcc2026-h1/reference_cells.csv",
            },
        }
    finally:
        data.file.close()
    return metadata, counts, strict


def seed_namespace() -> dict:
    return {
        "base_seed": 0,
        "model_index": {"model1": 0, "model2": 1, "model3": 2},
        "h1_source_rows": {
            "model1_template": [0, 1, 0, "source_order", 0],
            "model2": "identical to Model 1",
            "model3_template": [0, 1, 2, 0, 0],
        },
        "model1_decode": {
            "model_template": [0, 8, 0, "source_order", 0, 0],
            "null_template": [0, 8, 0, "source_order", 0, 1],
        },
        "model2_randomized_svd": 0,
        "model3": {
            "control_split": 0,
            "initialization_svd": 0,
            "starts": [0, 1, 2],
        },
        "control_split_half": {
            "template": [0, 7, "guide_index", "batch_index"]
        },
        "profile_ceiling": "five uint32 states from SeedSequence(0)",
    }


def write_outputs(
    manifest: dict,
    target_counts: pd.DataFrame,
    strict_controls: pd.DataFrame,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    target_counts.to_csv(REPORT_DIR / "h1_target_counts.csv", index=False)
    strict_controls.to_csv(REPORT_DIR / "h1_strict_controls.csv", index=False)
    (REPORT_DIR / "data_split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def main() -> None:
    h1, target_counts, strict_controls = audit_h1()
    reference = pd.read_csv(REFERENCE_CELLS)
    benchmark_targets = target_counts.loc[
        target_counts["benchmark_candidate"], "target_gene"
    ].astype(str)
    assert reference["target_gene"].nunique() == 126
    assert reference.groupby("target_gene").size().eq(CELLS_PER_TARGET).all()
    assert set(reference["target_gene"].astype(str)) == set(benchmark_targets)
    manifest = {
        "purpose": "H1-only linear-response three-model diagnostics",
        "h1": h1,
        "reference_cells": {
            "path": relative(REFERENCE_CELLS),
            "sha256": sha256(REFERENCE_CELLS),
            "targets": 126,
            "cells_per_target": CELLS_PER_TARGET,
            "owned_by": "scripts/evaluation/profile_h1.py",
        },
        "seeds": seed_namespace(),
        "software": {
            "python": platform.python_version(),
            "anndata": version("anndata"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
        },
    }
    write_outputs(manifest, target_counts, strict_controls)
    print(f"H1 calibration targets: {h1['cohorts']['calibration_targets']}")
    print(f"H1 benchmark targets: {h1['cohorts']['benchmark_targets']}")
    print(f"H1 strict controls: {h1['controls']['strict_cells']:,}")
    print(f"Wrote {relative(REPORT_DIR)}")


if __name__ == "__main__":
    main()
