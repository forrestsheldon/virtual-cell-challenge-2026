"""Pool selected Zhu/Marson CD4 CRISPRi author pseudobulks."""

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
SOURCE_URL = (
    "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/"
    "marson2025_data/GWCD4i.pseudobulk_merged.h5ad"
)
SOURCE_ETAG = "010c14e0af0dccbc2524529d28ca517e-5313"
SOURCE_VERSION_ID = "BWCjgMRhH80BOFIid2.0kbCr2o8wNVmn"


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
        if np.any(item["mask"][...]):
            raise ValueError(f"Missing strings in {path}")
        return decode(item["values"][...])
    values = item[...]
    return decode(values) if values.dtype.kind in "OS" else values


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


def group_codes(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, inverse = np.unique(labels, return_inverse=True)
    return unique, inverse.astype(np.int32)


def contiguous_runs(rows: np.ndarray) -> list[tuple[int, int]]:
    breaks = np.flatnonzero(np.diff(rows) != 1) + 1
    return [
        (int(part[0]), int(part[-1]) + 1)
        for part in np.split(rows, breaks)
        if len(part)
    ]


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
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    wanted = load_targets(args.targets)
    with h5py.File(args.input, "r") as handle:
        matrix = handle["X"]
        if matrix.attrs["encoding-type"] != "csr_matrix":
            raise ValueError("Expected CSR pseudobulk matrix")
        n_source_rows, n_genes = map(int, matrix.attrs["shape"])
        targets = column(handle, "obs/perturbed_gene_name")
        guides = column(handle, "obs/guide_id")
        guide_types = column(handle, "obs/guide_type")
        donors = column(handle, "obs/donor_id")
        conditions = column(handle, "obs/culture_condition")
        n_cells_raw = column(handle, "obs/n_cells")
        total_counts = column(handle, "obs/total_counts")
        keep_for_de = column(handle, "obs/keep_for_DE").astype(bool)
        keep_effective = column(handle, "obs/keep_effective_guides").astype(bool)
        if np.any(n_cells_raw < 0) or np.any(n_cells_raw != np.floor(n_cells_raw)):
            raise ValueError("obs/n_cells is not non-negative integer-valued")
        n_cells = n_cells_raw.astype(np.int64)
        if np.any((targets == CONTROL) != (guide_types == "non-targeting")):
            raise ValueError("NTC labels and non-targeting guide types disagree")

        selected = np.isin(targets, list(wanted)) | (targets == CONTROL)
        rows = np.flatnonzero(selected)
        target_labels, target_codes = group_codes(targets[selected])
        guide_labels, guide_codes = group_codes(guides[selected])
        donor_condition = np.char.add(
            np.char.add(donors[selected].astype(str), "|"),
            conditions[selected].astype(str),
        )
        dc_labels, dc_codes = group_codes(donor_condition)
        tdc_key = target_codes.astype(np.int64) * len(dc_labels) + dc_codes
        observed_tdc, tdc_codes = np.unique(tdc_key, return_inverse=True)
        tdc_target = observed_tdc // len(dc_labels)
        tdc_dc = observed_tdc % len(dc_labels)

        guide_target_pairs = np.unique(
            np.column_stack([guide_codes, target_codes]), axis=0
        )
        if len(guide_target_pairs) != len(guide_labels):
            raise ValueError("A CD4 guide maps to more than one gene target")
        guide_gene_targets = np.empty(len(guide_labels), dtype=target_labels.dtype)
        guide_gene_targets[guide_target_pairs[:, 0]] = target_labels[
            guide_target_pairs[:, 1]
        ]

        sums = {
            "target": np.zeros((len(target_labels), n_genes), dtype=np.int64),
            "guide": np.zeros((len(guide_labels), n_genes), dtype=np.int64),
            "target_donor_condition": np.zeros(
                (len(observed_tdc), n_genes), dtype=np.int64
            ),
        }
        indptr = matrix["indptr"][...].astype(np.int64)
        selected_position = {row: pos for pos, row in enumerate(rows)}
        for first, last in contiguous_runs(rows):
            offset = int(indptr[first])
            values = matrix["data"][offset : int(indptr[last])]
            indices = matrix["indices"][offset : int(indptr[last])].astype(np.int32)
            if np.any(values < 0) or np.any(values != np.floor(values)):
                raise ValueError(f"Non-count value in source rows {first}:{last}")
            local_indptr = indptr[first : last + 1] - offset
            block = sparse.csr_matrix(
                (values.astype(np.int64), indices, local_indptr),
                shape=(last - first, n_genes),
            )
            positions = np.asarray(
                [selected_position[row] for row in range(first, last)]
            )
            if not np.array_equal(
                np.asarray(block.sum(1)).ravel(), total_counts[first:last]
            ):
                raise ValueError(f"X and obs/total_counts disagree at {first}:{last}")
            add_grouped(sums["target"], block, target_codes[positions])
            add_grouped(sums["guide"], block, guide_codes[positions])
            add_grouped(sums["target_donor_condition"], block, tdc_codes[positions])
            print(f"source rows {first}:{last}", flush=True)

        gene_ids = column(handle, "var/gene_ids")
        gene_names = column(handle, "var/gene_name")

    assert n_source_rows == len(targets)
    assert len(gene_ids) == n_genes and len(set(gene_ids)) == n_genes
    assert np.array_equal(sums["target"].sum(0), sums["guide"].sum(0))
    assert np.array_equal(sums["target"].sum(0), sums["target_donor_condition"].sum(0))
    target_cells = np.bincount(
        target_codes, weights=n_cells[selected], minlength=len(target_labels)
    ).astype(np.int64)
    guide_cells = np.bincount(
        guide_codes, weights=n_cells[selected], minlength=len(guide_labels)
    ).astype(np.int64)
    tdc_cells = np.bincount(
        tdc_codes, weights=n_cells[selected], minlength=len(observed_tdc)
    ).astype(np.int64)
    guide_rows = np.bincount(guide_codes, minlength=len(guide_labels))
    guide_keep_de = (
        np.bincount(
            guide_codes, weights=keep_for_de[selected], minlength=len(guide_labels)
        )
        / guide_rows
    )
    guide_keep_effective = (
        np.bincount(
            guide_codes, weights=keep_effective[selected], minlength=len(guide_labels)
        )
        / guide_rows
    )

    observed_targets = set(target_labels) - {CONTROL}
    panel_hash = hashlib.sha256(("\n".join(sorted(wanted)) + "\n").encode()).hexdigest()
    provenance: dict[str, object] = {
        "dataset": "Zhu/Marson primary CD4+ T-cell genome-scale CRISPRi",
        "source_url": SOURCE_URL,
        "source_file": args.input.name,
        "source_size": args.input.stat().st_size,
        "source_sha256": args.source_sha256,
        "source_etag": SOURCE_ETAG,
        "source_version_id": SOURCE_VERSION_ID,
        "source_representation": "author guide x donor x condition pseudobulk CSR",
        "output_representation": "raw UMI count sums",
        "row_filtering": "requested targets plus all NTC; author keep flags retained as summaries but not applied",
        "control_label": CONTROL,
        "target_panel_sha256": panel_hash,
        "missing_requested_targets": sorted(wanted - observed_targets),
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
                    "n_source_pseudobulks": guide_rows,
                    "fraction_keep_for_DE": guide_keep_de,
                    "fraction_keep_effective_guides": guide_keep_effective,
                    "n_cells": guide_cells,
                    "total_umis": sums["guide"].sum(1),
                },
                index=guide_labels,
            ),
        ),
        "target_donor_condition": (
            sums["target_donor_condition"],
            pd.DataFrame(
                {
                    "gene_target": target_labels[tdc_target],
                    "donor_id": [dc_labels[x].split("|", 1)[0] for x in tdc_dc],
                    "culture_condition": [
                        dc_labels[x].split("|", 1)[1] for x in tdc_dc
                    ],
                    "n_cells": tdc_cells,
                    "total_umis": sums["target_donor_condition"].sum(1),
                },
                index=[
                    f"{target_labels[t]}|{dc_labels[d]}"
                    for t, d in zip(tdc_target, tdc_dc, strict=True)
                ],
            ),
        ),
    }
    files = {}
    for level, (counts, obs) in outputs.items():
        path = args.output / f"CD4T_{level}_pseudobulk.h5ad"
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
        "requested_targets": len(wanted),
        "observed_targets": len(observed_targets),
        "selected_source_pseudobulks": int(selected.sum()),
        "selected_cells": int(n_cells[selected].sum()),
        "perturbed_cells": int(target_cells.sum() - target_cells[control_index]),
        "controls": int(target_cells[control_index]),
        "guides": len(guide_labels),
        "donors": len(set(donors[selected])),
        "conditions": len(set(conditions[selected])),
        "genes": n_genes,
        "selected_umis": int(sums["target"].sum()),
        "outputs": files,
    }
    manifest_path = args.output / "CD4T_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
