"""Verify pooled context H5ADs against their manifest and one another."""

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


def pool_rows(data: ad.AnnData, target_names: list[str]) -> sparse.csr_matrix:
    lookup = {label: index for index, label in enumerate(target_names)}
    rows = np.asarray([lookup[label] for label in data.obs["gene_target"]])
    membership = sparse.csr_matrix(
        (np.ones(data.n_obs, dtype=np.int8), (rows, np.arange(data.n_obs))),
        shape=(len(target_names), data.n_obs),
    )
    return membership @ data.X


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    manifest_path = next(args.directory.glob("*_manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    files = {name: args.directory / name for name in manifest["outputs"]}
    for name, details in manifest["outputs"].items():
        assert files[name].stat().st_size == details["size"]
        assert sha256(files[name]) == details["sha256"]

    target_name = next(
        name for name in files if name.endswith("_target_pseudobulk.h5ad")
    )
    guide_name = next(name for name in files if name.endswith("_guide_pseudobulk.h5ad"))
    stratified_name = next(
        name
        for name in files
        if "target_batch" in name or "target_donor_condition" in name
    )
    target = ad.read_h5ad(files[target_name])
    guide = ad.read_h5ad(files[guide_name])
    stratified = ad.read_h5ad(files[stratified_name])
    datasets = [target, guide, stratified]
    control_name = next((name for name in files if "control_channel" in name), None)
    control = ad.read_h5ad(files[control_name]) if control_name else None
    if control is not None:
        datasets.append(control)

    for data in datasets:
        assert sparse.isspmatrix_csr(data.X)
        assert np.issubdtype(data.X.dtype, np.integer)
        assert data.X.data.min(initial=0) >= 0
        assert np.array_equal(data.obs["total_umis"], np.asarray(data.X.sum(1)).ravel())
        assert data.var_names.equals(target.var_names)
        assert data.n_vars == manifest["genes"]

    target_names = target.obs_names.tolist()
    assert (pool_rows(guide, target_names) != target.X).nnz == 0
    assert (pool_rows(stratified, target_names) != target.X).nnz == 0
    if control is not None:
        control_row = target.obs_names.get_loc(manifest["control_label"])
        assert np.array_equal(
            np.asarray(control.X.sum(0)).ravel(),
            target.X[control_row].toarray().ravel(),
        )

    assert int(target.obs["n_cells"].sum()) == manifest["selected_cells"]
    assert int(target.X.sum()) == manifest["selected_umis"]
    audit = {
        "dataset": manifest["dataset"],
        "status": "passed",
        "hashes_verified": len(files),
        "target_rows": target.n_obs,
        "guide_rows": guide.n_obs,
        "stratified_rows": stratified.n_obs,
        "genes": target.n_vars,
        "selected_cells": int(target.obs["n_cells"].sum()),
        "selected_umis": int(target.X.sum()),
        "target_equals_pooled_guides": True,
        "target_equals_pooled_strata": True,
        "control_equals_pooled_channels": control is not None,
        "integer_nonnegative_counts": True,
    }
    output = args.directory / manifest_path.name.replace("manifest", "audit")
    output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
