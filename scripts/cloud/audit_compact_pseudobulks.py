"""Verify locally synced target/guide/control pseudobulks against cloud manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def pool_guides(target: ad.AnnData, guide: ad.AnnData) -> sparse.csr_matrix:
    lookup = {label: index for index, label in enumerate(target.obs_names)}
    rows = np.asarray([lookup[label] for label in guide.obs["gene_target"]])
    membership = sparse.csr_matrix(
        (np.ones(guide.n_obs, dtype=np.int8), (rows, np.arange(guide.n_obs))),
        shape=(target.n_obs, guide.n_obs),
    )
    pooled_cells = np.bincount(
        rows, weights=guide.obs["n_cells"], minlength=target.n_obs
    )
    assert np.array_equal(pooled_cells, target.obs["n_cells"])
    return membership @ guide.X


def audit(directory: Path) -> dict:
    manifest_path = next(directory.glob("*_manifest.json"))
    source_audit = json.loads(next(directory.glob("*_audit.json")).read_text())
    manifest = json.loads(manifest_path.read_text())
    assert source_audit["status"] == "passed"

    declared = manifest["outputs"]
    present = {name: directory / name for name in declared if (directory / name).exists()}
    expected_missing = {
        name
        for name in declared
        if "target_batch" in name or "target_donor_condition" in name
    }
    assert set(declared) - set(present) == expected_missing

    for name, path in present.items():
        assert path.stat().st_size == declared[name]["size"]
        assert sha256(path) == declared[name]["sha256"]

    target_path = next(path for name, path in present.items() if "_target_pseudobulk" in name)
    guide_path = next(path for name, path in present.items() if "_guide_pseudobulk" in name)
    target = ad.read_h5ad(target_path)
    guide = ad.read_h5ad(guide_path)
    optional = [
        ad.read_h5ad(path)
        for name, path in present.items()
        if "_control_" in name or "_batch_control_" in name
    ]

    for data in [target, guide, *optional]:
        assert sparse.isspmatrix_csr(data.X)
        assert np.issubdtype(data.X.dtype, np.integer)
        assert data.X.data.min(initial=0) >= 0
        assert np.array_equal(data.obs["total_umis"], np.asarray(data.X.sum(1)).ravel())
        assert data.var_names.equals(target.var_names)
        assert data.n_vars == manifest["genes"]

    assert (pool_guides(target, guide) != target.X).nnz == 0
    assert int(target.obs["n_cells"].sum()) == manifest["selected_cells"]
    assert int(target.X.sum()) == manifest["selected_umis"]

    if optional:
        control_row = target.obs_names.get_loc(manifest["control_label"])
        assert np.array_equal(
            np.asarray(optional[0].X.sum(0)).ravel(),
            target.X[control_row].toarray().ravel(),
        )

    return {
        "directory": str(directory),
        "status": "passed",
        "files_verified": len(present),
        "omitted_stratified_files": sorted(expected_missing),
        "target_rows": target.n_obs,
        "guide_rows": guide.n_obs,
        "genes": target.n_vars,
        "selected_cells": int(target.obs["n_cells"].sum()),
        "selected_umis": int(target.X.sum()),
        "target_equals_pooled_guides": True,
        "control_reconstruction_checked": bool(optional),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([audit(directory) for directory in args.directories], indent=2))


if __name__ == "__main__":
    main()
