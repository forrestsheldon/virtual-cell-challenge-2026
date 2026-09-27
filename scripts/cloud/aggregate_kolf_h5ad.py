"""Pool selected KOLF2.1J CRISPRi cells from the author count layer."""

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

CONTROL = "NTC"
SOURCE_URL = "https://ndownloader.figshare.com/files/64650261"


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [x.decode() if isinstance(x, bytes) else str(x) for x in values], dtype=str
    )


def categorical(handle: h5py.File, path: str) -> tuple[np.ndarray, np.ndarray]:
    group = handle[path]
    codes = group["codes"][...].astype(np.int32)
    if np.any(codes < 0):
        raise ValueError(f"Missing categorical values in {path}")
    return decode(group["categories"][...]), codes


def string_array(handle: h5py.File, path: str) -> np.ndarray:
    item = handle[path]
    if isinstance(item, h5py.Dataset):
        return decode(item[...])
    if np.any(item["mask"][...]):
        raise ValueError(f"Missing strings in {path}")
    return decode(item["values"][...])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_targets(paths: list[Path]) -> set[str]:
    targets: set[str] = set()
    for path in paths:
        targets.update(pd.read_csv(path)["target_gene"].astype(str))
    return targets


def remap_categories(
    categories: np.ndarray, codes: np.ndarray, keep: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.unique(codes[keep])
    labels = categories[observed]
    lookup = np.full(len(categories), -1, dtype=np.int32)
    lookup[observed] = np.arange(len(observed), dtype=np.int32)
    return labels, lookup[codes]


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
    parser.add_argument("--targets", type=Path, action="append", required=True)
    parser.add_argument("--source-md5", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gene-block", type=int, default=32)
    args = parser.parse_args()

    wanted = load_targets(args.targets)
    with h5py.File(args.input, "r") as handle:
        matrix = handle["layers/counts"]
        if matrix.attrs["encoding-type"] != "csc_matrix":
            raise ValueError("Expected CSC counts layer")
        n_cells, n_genes = map(int, matrix.attrs["shape"])

        target_categories, source_target_codes = categorical(handle, "obs/gene_target")
        if CONTROL not in target_categories:
            raise ValueError(f"Missing {CONTROL} controls")
        requested_categories = np.flatnonzero(np.isin(target_categories, list(wanted)))
        control_category = int(np.flatnonzero(target_categories == CONTROL)[0])
        keep = np.isin(source_target_codes, requested_categories) | (
            source_target_codes == control_category
        )

        target_labels, target_codes = remap_categories(
            target_categories, source_target_codes, keep
        )
        guide_categories, source_guide_codes = categorical(handle, "obs/gRNA")
        guide_labels, guide_codes = remap_categories(
            guide_categories, source_guide_codes, keep
        )
        guide_target_pairs = np.unique(
            np.column_stack([guide_codes[keep], target_codes[keep]]), axis=0
        )
        if len(guide_target_pairs) != len(guide_labels):
            raise ValueError("A KOLF guide maps to more than one gene target")
        guide_gene_targets = np.empty(len(guide_labels), dtype=target_labels.dtype)
        guide_gene_targets[guide_target_pairs[:, 0]] = target_labels[
            guide_target_pairs[:, 1]
        ]
        batch_categories, batch_codes = categorical(handle, "obs/batch")
        channel_categories, channel_codes = categorical(handle, "obs/channel")

        tb_keys = (
            target_codes[keep].astype(np.int64) * len(batch_categories)
            + batch_codes[keep]
        )
        observed_tb, tb_inverse = np.unique(tb_keys, return_inverse=True)
        target_batch_codes = np.full(n_cells, -1, dtype=np.int32)
        target_batch_codes[keep] = tb_inverse
        tb_target = observed_tb // len(batch_categories)
        tb_batch = observed_tb % len(batch_categories)

        control_rows = source_target_codes == control_category
        observed_channels, channel_inverse = np.unique(
            channel_codes[control_rows], return_inverse=True
        )
        control_channel_codes = np.full(n_cells, -1, dtype=np.int32)
        control_channel_codes[control_rows] = channel_inverse
        control_channel_labels = channel_categories[observed_channels]

        groups = {
            "target": (target_codes, len(target_labels)),
            "guide": (guide_codes, len(guide_labels)),
            "target_batch": (target_batch_codes, len(observed_tb)),
            "control_channel": (
                control_channel_codes,
                len(control_channel_labels),
            ),
        }
        sums = {
            name: np.zeros((size, n_genes), dtype=np.int64)
            for name, (_, size) in groups.items()
        }

        indptr = matrix["indptr"][...].astype(np.int64)
        for first in range(0, n_genes, args.gene_block):
            last = min(first + args.gene_block, n_genes)
            offset = int(indptr[first])
            rows = matrix["indices"][offset : int(indptr[last])].astype(np.int32)
            values = matrix["data"][offset : int(indptr[last])]
            if np.any(values < 0) or np.any(values != np.floor(values)):
                raise ValueError(f"Non-count value in genes {first}:{last}")
            values = values.astype(np.int64)
            local_ptr = indptr[first : last + 1] - offset
            for local_gene, gene in enumerate(range(first, last)):
                start, stop = map(int, local_ptr[local_gene : local_gene + 2])
                gene_rows = rows[start:stop]
                gene_values = values[start:stop]
                selected = keep[gene_rows]
                gene_rows = gene_rows[selected]
                gene_values = gene_values[selected]
                for name, (codes, size) in groups.items():
                    selected_codes = codes[gene_rows]
                    selected = selected_codes >= 0
                    sums[name][:, gene] = np.bincount(
                        selected_codes[selected],
                        weights=gene_values[selected],
                        minlength=size,
                    ).astype(np.int64)
            print(f"genes {first}:{last}", flush=True)

        gene_names = string_array(handle, "var/_index")
        gene_ids = string_array(handle, "var/gene_ids")

    assert len(source_target_codes) == n_cells
    assert len(gene_ids) == n_genes and len(set(gene_ids)) == n_genes
    assert np.array_equal(sums["target"].sum(0), sums["guide"].sum(0))
    assert np.array_equal(sums["target"].sum(0), sums["target_batch"].sum(0))
    control_index = int(np.flatnonzero(target_labels == CONTROL)[0])
    assert np.array_equal(sums["target"][control_index], sums["control_channel"].sum(0))

    target_cells = np.bincount(target_codes[keep], minlength=len(target_labels))
    guide_cells = np.bincount(guide_codes[keep], minlength=len(guide_labels))
    tb_cells = np.bincount(target_batch_codes[keep], minlength=len(observed_tb))
    channel_cells = np.bincount(
        control_channel_codes[control_rows], minlength=len(control_channel_labels)
    )
    observed_targets = set(target_labels) - {CONTROL}
    panel_hash = hashlib.sha256(("\n".join(sorted(wanted)) + "\n").encode()).hexdigest()
    provenance: dict[str, object] = {
        "dataset": "Nourreddine et al. KOLF2.1J pan-genome CRISPRi",
        "source_url": SOURCE_URL,
        "source_file": args.input.name,
        "source_size": args.input.stat().st_size,
        "source_md5": args.source_md5,
        "source_sha256": args.source_sha256,
        "source_representation": "layers/counts CSC float32",
        "output_representation": "raw UMI count sums",
        "control_label": CONTROL,
        "target_panel_sha256": panel_hash,
        "missing_requested_targets": sorted(wanted - observed_targets),
        "created_utc": datetime.now(UTC).isoformat(),
    }
    var = pd.DataFrame(
        {"gene_name": gene_names, "source_index": gene_names}, index=gene_ids
    )
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
                    "batch": batch_categories[tb_batch],
                    "n_cells": tb_cells,
                    "total_umis": sums["target_batch"].sum(1),
                },
                index=[
                    f"{target_labels[t]}|{batch_categories[b]}"
                    for t, b in zip(tb_target, tb_batch, strict=True)
                ],
            ),
        ),
        "control_channel": (
            sums["control_channel"],
            pd.DataFrame(
                {
                    "gene_target": CONTROL,
                    "channel": control_channel_labels,
                    "n_cells": channel_cells,
                    "total_umis": sums["control_channel"].sum(1),
                },
                index=control_channel_labels,
            ),
        ),
    }
    files = {}
    for level, (counts, obs) in outputs.items():
        path = args.output / f"KOLF2.1J_{level}_pseudobulk.h5ad"
        write_h5ad(path, counts, obs, var, provenance)
        files[path.name] = {"size": path.stat().st_size, "sha256": sha256(path)}

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
        "requested_targets": len(wanted),
        "observed_targets": len(observed_targets),
        "selected_cells": int(keep.sum()),
        "perturbed_cells": int(keep.sum() - control_rows.sum()),
        "controls": int(control_rows.sum()),
        "guides": len(guide_labels),
        "batches": len(batch_categories),
        "control_channels": len(control_channel_labels),
        "genes": n_genes,
        "selected_umis": int(sums["target"].sum()),
        "outputs": files,
    }
    manifest_path = args.output / "KOLF2.1J_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
