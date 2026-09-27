"""Verify final X-Atlas pseudobulks against their merge manifest."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    manifest_path = next(args.directory.glob("*_manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    context = manifest["context"]
    paths = {name: args.directory / name for name in manifest["outputs"]}
    for name, details in manifest["outputs"].items():
        assert paths[name].stat().st_size == details["size"]
        assert sha256(paths[name]) == details["sha256"]

    target = ad.read_h5ad(paths[f"{context}_target_pseudobulk.h5ad"])
    guide = ad.read_h5ad(paths[f"{context}_guide_pseudobulk.h5ad"])
    control = ad.read_h5ad(paths[f"{context}_batch_control_pseudobulk.h5ad"])
    for data in [target, guide, control]:
        assert sparse.isspmatrix_csr(data.X)
        assert np.issubdtype(data.X.dtype, np.integer)
        assert data.X.data.min(initial=0) >= 0
        assert np.array_equal(data.obs["total_umis"], np.asarray(data.X.sum(1)).ravel())
        assert data.var_names.equals(target.var_names)
        assert data.n_vars == manifest["genes"]

    target_lookup = {label: index for index, label in enumerate(target.obs_names)}
    rows = np.array([target_lookup[label] for label in guide.obs["gene_target"]])
    pooling = sparse.csr_matrix(
        (np.ones(guide.n_obs, dtype=np.int8), (rows, np.arange(guide.n_obs))),
        shape=(target.n_obs, guide.n_obs),
    )
    assert (pooling @ guide.X != target.X).nnz == 0
    assert np.array_equal(
        np.bincount(rows, weights=guide.obs["n_cells"], minlength=target.n_obs),
        target.obs["n_cells"],
    )

    control_row = target.obs_names.get_loc(manifest["control_label"])
    assert np.array_equal(
        np.asarray(control.X.sum(0)).ravel(), target.X[control_row].toarray().ravel()
    )
    assert int(control.obs["n_cells"].sum()) == manifest["controls"]
    assert int(target.obs["n_cells"].sum()) == manifest["selected_cells"]
    assert int(target.X.sum()) == manifest["selected_umis"]

    audit = {
        "context": context,
        "status": "passed",
        "hashes_verified": len(paths),
        "target_rows": target.n_obs,
        "guide_rows": guide.n_obs,
        "batch_control_rows": control.n_obs,
        "genes": target.n_vars,
        "selected_cells": int(target.obs["n_cells"].sum()),
        "selected_umis": int(target.X.sum()),
        "target_equals_pooled_guides": True,
        "control_equals_pooled_batches": True,
        "integer_nonnegative_counts": True,
    }
    output = args.directory / f"{context}_audit.json"
    output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
