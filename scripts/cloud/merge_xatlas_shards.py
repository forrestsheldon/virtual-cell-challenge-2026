"""Merge restartable X-Atlas shard sums into annotated raw-count pseudobulks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

CONTROL = "Non-Targeting"
N_GENES = 38_606


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix(archive: np.lib.npyio.NpzFile, prefix: str) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (
            archive[f"{prefix}_data"],
            archive[f"{prefix}_indices"],
            archive[f"{prefix}_indptr"],
        ),
        shape=archive[f"{prefix}_shape"],
    )


def guide_gene_target(guide: str) -> str:
    if guide.lower().startswith("non-targeting"):
        return CONTROL
    # Most constructs use SYMBOL_P1/P2; three use SYMBOL_ENST... instead.
    match = re.match(r"(.+?)_(?:P\d|ENST\d)", guide.split("|")[0])
    if not match:
        raise ValueError(f"Cannot recover target from guide label: {guide}")
    return match[1]


def label_order(labels: set[str]) -> list[str]:
    return ([CONTROL] if CONTROL in labels else []) + sorted(labels - {CONTROL})


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
    parser.add_argument("--context", choices=["HCT116", "HEK293T"], required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--gene-metadata", type=Path, required=True)
    parser.add_argument("--targets", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shard_paths = sorted(
        args.shards.glob(f"{args.context}_Batch*.npz"),
        key=lambda path: int(re.search(r"Batch(\d+)", path.name)[1]),
    )
    if not shard_paths:
        raise ValueError(f"No shard checkpoints in {args.shards}")

    target_labels: set[str] = set()
    guide_labels: set[str] = set()
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as archive:
            target_labels.update(archive["target_labels"].astype(str))
            guide_labels.update(archive["guide_labels"].astype(str))

    targets = label_order(target_labels)
    guides = sorted(guide_labels, key=lambda guide: (guide_gene_target(guide), guide))
    target_lookup = {label: index for index, label in enumerate(targets)}
    guide_lookup = {label: index for index, label in enumerate(guides)}

    target_sum = np.zeros((len(targets), N_GENES), dtype=np.int64)
    target_cells = np.zeros(len(targets), dtype=np.int64)
    guide_sum = np.zeros((len(guides), N_GENES), dtype=np.int64)
    guide_cells = np.zeros(len(guides), dtype=np.int64)
    batch_control_sum = np.zeros((len(shard_paths), N_GENES), dtype=np.int64)
    batch_control_cells = np.zeros(len(shard_paths), dtype=np.int64)
    manifests: list[dict[str, object]] = []

    for batch, path in enumerate(shard_paths):
        manifest_path = path.with_suffix(".json")
        manifest = json.loads(manifest_path.read_text())
        if sha256(path) != manifest["output_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {path}")
        manifests.append(manifest)
        with np.load(path, allow_pickle=False) as archive:
            local_targets = archive["target_labels"].astype(str)
            local_guides = archive["guide_labels"].astype(str)
            target_index = np.asarray([target_lookup[x] for x in local_targets])
            guide_index = np.asarray([guide_lookup[x] for x in local_guides])
            target_sum[target_index] += matrix(archive, "target").toarray()
            guide_sum[guide_index] += matrix(archive, "guide").toarray()
            target_cells[target_index] += archive["target_cells"]
            guide_cells[guide_index] += archive["guide_cells"]
            batch_control_sum[batch] = archive["control_sum"]
            control = np.flatnonzero(local_targets == CONTROL)
            if len(control) != 1:
                raise ValueError(f"Expected one control row in {path}")
            batch_control_cells[batch] = archive["target_cells"][control[0]]

    wanted = set()
    for path in args.targets:
        wanted.update(pd.read_csv(path)["target_gene"].astype(str))
    observed = set(targets) - {CONTROL}
    missing = sorted(wanted - observed)
    extra = sorted(observed - wanted)

    control_index = target_lookup[CONTROL]
    selected_cells = sum(int(x["n_selected_cells"]) for x in manifests)
    selected_umis = sum(int(x["selected_umis"]) for x in manifests)
    assert int(target_cells.sum()) == int(guide_cells.sum()) == selected_cells
    assert int(target_sum.sum()) == int(guide_sum.sum()) == selected_umis
    assert np.array_equal(target_sum[control_index], batch_control_sum.sum(axis=0))
    assert int(target_cells[control_index]) == int(batch_control_cells.sum())
    if extra:
        raise ValueError(f"Unexpected target labels: {extra}")

    genes = pd.read_parquet(args.gene_metadata).sort_values("gene_token_id")
    if genes["gene_token_id"].tolist() != list(range(N_GENES)):
        raise ValueError("Gene metadata is not a complete ordered token axis")
    if not genes["ensembl_id"].is_unique:
        raise ValueError("Ensembl identifiers are not unique")
    var = genes.set_index("ensembl_id")[["gene_name", "gene_token_id"]]

    revision = {str(x["revision"]) for x in manifests}
    panel_hash = {str(x["target_panel_sha256"]) for x in manifests}
    if len(revision) != 1 or len(panel_hash) != 1:
        raise ValueError("Shard provenance is inconsistent")
    provenance = {
        "dataset": "Xaira-Therapeutics/X-Atlas-Orion",
        "revision": revision.pop(),
        "context": args.context,
        "representation": "raw UMI count sums",
        "control_label": CONTROL,
        "target_panel_sha256": panel_hash.pop(),
        "source_shards": len(shard_paths),
        "selected_cells": selected_cells,
        "selected_umis": selected_umis,
        "missing_requested_targets": missing,
        "created_utc": datetime.now(UTC).isoformat(),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    target_path = args.output / f"{args.context}_target_pseudobulk.h5ad"
    guide_path = args.output / f"{args.context}_guide_pseudobulk.h5ad"
    control_path = args.output / f"{args.context}_batch_control_pseudobulk.h5ad"
    write_h5ad(
        target_path,
        target_sum,
        pd.DataFrame(
            {
                "gene_target": targets,
                "n_cells": target_cells,
                "total_umis": target_sum.sum(axis=1),
            },
            index=targets,
        ),
        var,
        provenance,
    )
    write_h5ad(
        guide_path,
        guide_sum,
        pd.DataFrame(
            {
                "guide_target": guides,
                "gene_target": [guide_gene_target(x) for x in guides],
                "n_cells": guide_cells,
                "total_umis": guide_sum.sum(axis=1),
            },
            index=guides,
        ),
        var,
        provenance,
    )
    batches = [path.stem for path in shard_paths]
    write_h5ad(
        control_path,
        batch_control_sum,
        pd.DataFrame(
            {
                "sample": batches,
                "gene_target": CONTROL,
                "n_cells": batch_control_cells,
                "total_umis": batch_control_sum.sum(axis=1),
            },
            index=batches,
        ),
        var,
        provenance,
    )

    outputs = {}
    for path in [target_path, guide_path, control_path]:
        outputs[path.name] = {
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }
    manifest = {
        **provenance,
        "requested_targets": len(wanted),
        "observed_targets": len(observed),
        "controls": int(target_cells[control_index]),
        "perturbed_cells": int(selected_cells - target_cells[control_index]),
        "guides": len(guides),
        "genes": N_GENES,
        "outputs": outputs,
    }
    (args.output / f"{args.context}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
