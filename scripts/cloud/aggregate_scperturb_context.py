"""Aggregate a full raw-count scPerturb context by target, guide, and batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy
from scipy import sparse

CONTROL = "non-targeting"


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [x.decode() if isinstance(x, bytes) else str(x) for x in values], dtype=str
    )


def column(handle: h5py.File, path: str) -> np.ndarray:
    item = handle[path]
    if isinstance(item, h5py.Group):
        if "categories" in item:
            categories = decode(item["categories"][...])
            codes = item["codes"][...].astype(np.int32)
            if np.any(codes < 0):
                raise ValueError(f"Missing categorical values in {path}")
            return categories[codes]
        if "values" in item and "mask" in item:
            if np.any(item["mask"][...]):
                raise ValueError(f"Missing string values in {path}")
            return decode(item["values"][...])
        raise ValueError(f"Unsupported AnnData column encoding in {path}")
    values = item[...]
    return decode(values) if values.dtype.kind in "OS" else values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def group_codes(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, inverse = np.unique(labels.astype(str), return_inverse=True)
    return unique, inverse.astype(np.int32)


def add_grouped(
    destination: np.ndarray, block: sparse.csr_matrix, codes: np.ndarray
) -> None:
    unique, inverse = np.unique(codes, return_inverse=True)
    membership = sparse.csr_matrix(
        (
            np.ones(len(codes), dtype=np.int8),
            (inverse, np.arange(len(codes))),
        ),
        shape=(len(unique), len(codes)),
    )
    destination[unique] += (membership @ block).toarray().astype(np.int64)


def blocks(
    matrix: h5py.Dataset | h5py.Group, n_cells: int, n_genes: int, block_size: int
):
    if isinstance(matrix, h5py.Dataset):
        for first in range(0, n_cells, block_size):
            last = min(first + block_size, n_cells)
            values = matrix[first:last]
            if np.any(values < 0) or np.any(values != np.floor(values)):
                raise ValueError(f"Non-count value in rows {first}:{last}")
            yield first, last, sparse.csr_matrix(values.astype(np.int64))
        return

    if matrix.attrs["encoding-type"] != "csr_matrix":
        raise ValueError("Expected dense or CSR X")
    indptr = matrix["indptr"][...].astype(np.int64)
    for first in range(0, n_cells, block_size):
        last = min(first + block_size, n_cells)
        offset = int(indptr[first])
        values = matrix["data"][offset : int(indptr[last])]
        if np.any(values < 0) or np.any(values != np.floor(values)):
            raise ValueError(f"Non-count value in rows {first}:{last}")
        yield first, last, sparse.csr_matrix(
            (
                values.astype(np.int64),
                matrix["indices"][offset : int(indptr[last])].astype(np.int32),
                indptr[first : last + 1] - offset,
            ),
            shape=(last - first, n_genes),
        )


def write_h5ad(
    path: Path,
    counts: np.ndarray,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    provenance: dict[str, object],
) -> None:
    data = ad.AnnData(X=sparse.csr_matrix(counts), obs=obs, var=var)
    data.uns["provenance"] = provenance
    data.write_h5ad(path, compression="gzip", compression_opts=4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--context", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-md5", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=2048)
    args = parser.parse_args()

    with h5py.File(args.input, "r") as handle:
        matrix = handle["X"]
        if isinstance(matrix, h5py.Dataset):
            n_cells, n_genes = matrix.shape
            source_representation = "dense float32 X"
        else:
            n_cells, n_genes = map(int, matrix.attrs["shape"])
            source_representation = "CSR X"

        targets = column(handle, "obs/gene")
        guides = column(handle, "obs/guide_id")
        batches = column(handle, "obs/batch").astype(str)
        ncounts = column(handle, "obs/ncounts")
        if CONTROL not in targets:
            raise ValueError(f"Missing {CONTROL} controls")
        target_labels, target_codes = group_codes(targets)
        guide_labels, guide_codes = group_codes(guides)
        batch_labels, batch_codes = group_codes(batches)
        tb_key = target_codes.astype(np.int64) * len(batch_labels) + batch_codes
        observed_tb, target_batch_codes = np.unique(tb_key, return_inverse=True)
        tb_target = observed_tb // len(batch_labels)
        tb_batch = observed_tb % len(batch_labels)

        guide_target_pairs = np.unique(
            np.column_stack([guide_codes, target_codes]), axis=0
        )
        if len(guide_target_pairs) != len(guide_labels):
            raise ValueError("A guide maps to more than one gene target")
        guide_gene_targets = np.empty(len(guide_labels), dtype=target_labels.dtype)
        guide_gene_targets[guide_target_pairs[:, 0]] = target_labels[
            guide_target_pairs[:, 1]
        ]

        sums = {
            "target": np.zeros((len(target_labels), n_genes), dtype=np.int64),
            "guide": np.zeros((len(guide_labels), n_genes), dtype=np.int64),
            "target_batch": np.zeros((len(observed_tb), n_genes), dtype=np.int64),
        }
        for first, last, block in blocks(
            matrix, n_cells, n_genes, args.block_size
        ):
            if not np.array_equal(np.asarray(block.sum(1)).ravel(), ncounts[first:last]):
                raise ValueError(f"X and obs/ncounts disagree at rows {first}:{last}")
            add_grouped(sums["target"], block, target_codes[first:last])
            add_grouped(sums["guide"], block, guide_codes[first:last])
            add_grouped(
                sums["target_batch"], block, target_batch_codes[first:last]
            )
            print(f"rows {first}:{last}", flush=True)

        gene_ids = column(handle, "var/ensembl_id")
        gene_names = column(handle, "var/gene_name")

    assert len(targets) == n_cells
    assert len(gene_ids) == n_genes and len(set(gene_ids)) == n_genes
    assert np.array_equal(sums["target"].sum(0), sums["guide"].sum(0))
    assert np.array_equal(sums["target"].sum(0), sums["target_batch"].sum(0))
    target_cells = np.bincount(target_codes, minlength=len(target_labels))
    guide_cells = np.bincount(guide_codes, minlength=len(guide_labels))
    tb_cells = np.bincount(target_batch_codes, minlength=len(observed_tb))
    provenance: dict[str, object] = {
        "dataset": args.dataset,
        "context": args.context,
        "source_url": args.source_url,
        "source_file": args.input.name,
        "source_size": args.input.stat().st_size,
        "source_md5": args.source_md5,
        "source_sha256": args.source_sha256,
        "source_representation": source_representation,
        "output_representation": "raw UMI count sums",
        "control_label": CONTROL,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    var = pd.DataFrame({"gene_name": gene_names}, index=gene_ids)
    var.index.name = "gene_id"
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "target": (
            sums["target"],
            pd.DataFrame(
                {
                    "gene_target": target_labels,
                    "n_cells": target_cells,
                    "total_umis": sums["target"].sum(1),
                },
                index=target_labels,
            ),
        ),
        "guide": (
            sums["guide"],
            pd.DataFrame(
                {
                    "guide_target": guide_labels,
                    "gene_target": guide_gene_targets,
                    "n_cells": guide_cells,
                    "total_umis": sums["guide"].sum(1),
                },
                index=guide_labels,
            ),
        ),
        "target_batch": (
            sums["target_batch"],
            pd.DataFrame(
                {
                    "gene_target": target_labels[tb_target],
                    "batch": batch_labels[tb_batch],
                    "n_cells": tb_cells,
                    "total_umis": sums["target_batch"].sum(1),
                },
                index=[
                    f"{target_labels[t]}|{batch_labels[b]}"
                    for t, b in zip(tb_target, tb_batch, strict=True)
                ],
            ),
        ),
    }
    files = {}
    for level, (counts, obs) in outputs.items():
        path = args.output / f"{args.context}_{level}_pseudobulk.h5ad"
        write_h5ad(path, counts, obs, var, provenance)
        files[path.name] = {"size": path.stat().st_size, "sha256": sha256(path)}

    control_index = int(np.flatnonzero(target_labels == CONTROL)[0])
    manifest = {
        **provenance,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "anndata": version("anndata"),
            "h5py": h5py.__version__,
        },
        "observed_targets": len(target_labels) - 1,
        "selected_cells": n_cells,
        "perturbed_cells": int(n_cells - target_cells[control_index]),
        "controls": int(target_cells[control_index]),
        "guides": len(guide_labels),
        "batches": len(batch_labels),
        "genes": n_genes,
        "selected_umis": int(sums["target"].sum()),
        "outputs": files,
    }
    manifest_path = args.output / f"{args.context}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
