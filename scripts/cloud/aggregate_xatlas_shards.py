"""Aggregate X-Atlas/Orion Parquet shards into restartable pseudobulk shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
from scipy import sparse

pa.set_cpu_count(int(os.environ.get("VCC_ARROW_THREADS", "2")))

REPO = "Xaira-Therapeutics/X-Atlas-Orion"
API = f"https://huggingface.co/api/datasets/{REPO}"
CONTROL = "Non-Targeting"
N_GENES = 38_606
COLUMNS = [
    "gene_token_id",
    "gene_expression",
    "gene_target",
    "guide_target",
    "sample",
    "pass_guide_filter",
]


def load_targets(paths: list[Path]) -> set[str]:
    targets: set[str] = set()
    for path in paths:
        frame = pd.read_csv(path)
        if "target_gene" not in frame:
            raise ValueError(f"{path} has no target_gene column")
        targets.update(frame["target_gene"].astype(str))
    return targets


def dataset_files(context: str) -> tuple[str, list[dict[str, object]]]:
    with urllib.request.urlopen(API) as response:
        revision = json.load(response)["sha"]
    tree_url = f"{API}/tree/{revision}/data?recursive=true&expand=false&limit=1000"
    with urllib.request.urlopen(tree_url) as response:
        files = json.load(response)
    selected = [
        item
        for item in files
        if item["type"] == "file"
        and Path(item["path"]).name.startswith(f"{context}_Batch")
        and item["path"].endswith(".parquet")
    ]
    selected.sort(key=lambda item: int(re.search(r"Batch(\d+)", item["path"])[1]))
    return revision, selected


def group_sum(
    cells: sparse.csr_matrix, labels: np.ndarray
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    unique, codes = np.unique(labels, return_inverse=True)
    membership = sparse.csr_matrix(
        (
            np.ones(len(labels), dtype=np.int8),
            (codes, np.arange(len(labels))),
        ),
        shape=(len(unique), len(labels)),
    )
    counts = np.bincount(codes, minlength=len(unique)).astype(np.int64)
    return (membership @ cells).tocsr(), unique, counts


def aggregate_table(
    table: pa.Table, wanted: set[str]
) -> tuple[dict[str, object], dict[str, object]]:
    targets = np.asarray(table["gene_target"].to_pylist(), dtype=str)
    passed = table["pass_guide_filter"].to_numpy(zero_copy_only=False).astype(bool)
    keep = passed & (np.isin(targets, list(wanted)) | (targets == CONTROL))
    selected = table.filter(pa.array(keep)).combine_chunks()
    selected_targets = targets[keep]
    selected_guides = np.asarray(selected["guide_target"].to_pylist(), dtype=str)
    samples = np.asarray(selected["sample"].to_pylist(), dtype=str)

    tokens = selected["gene_token_id"].chunk(0)
    expression = selected["gene_expression"].chunk(0)
    offsets = tokens.offsets.to_numpy(zero_copy_only=False)
    gene_index = tokens.values.to_numpy(zero_copy_only=False).astype(np.int32)
    values_float = expression.values.to_numpy(zero_copy_only=False)
    if np.any(values_float < 0) or not np.all(values_float == np.floor(values_float)):
        raise ValueError("gene_expression is not non-negative integer-valued")
    values = values_float.astype(np.int64)
    cell_index = np.repeat(np.arange(len(selected), dtype=np.int32), np.diff(offsets))
    cells = sparse.csr_matrix(
        (values, (cell_index, gene_index)), shape=(len(selected), N_GENES)
    )

    target_matrix, target_labels, target_cells = group_sum(cells, selected_targets)
    guide_matrix, guide_labels, guide_cells = group_sum(cells, selected_guides)
    control_matrix = cells[selected_targets == CONTROL].sum(axis=0)

    arrays: dict[str, object] = {}
    for prefix, matrix in [("target", target_matrix), ("guide", guide_matrix)]:
        arrays[f"{prefix}_data"] = matrix.data
        arrays[f"{prefix}_indices"] = matrix.indices
        arrays[f"{prefix}_indptr"] = matrix.indptr
        arrays[f"{prefix}_shape"] = np.asarray(matrix.shape, dtype=np.int64)
    arrays.update(
        target_labels=target_labels,
        target_cells=target_cells,
        guide_labels=guide_labels,
        guide_cells=guide_cells,
        control_sum=np.asarray(control_matrix).ravel().astype(np.int64),
    )
    audit = {
        "n_input_cells": len(table),
        "n_pass_guide_filter": int(passed.sum()),
        "n_selected_cells": len(selected),
        "n_selected_perturbation_cells": int((selected_targets != CONTROL).sum()),
        "n_control_cells": int((selected_targets == CONTROL).sum()),
        "n_target_labels": len(target_labels),
        "n_guide_labels": len(guide_labels),
        "samples": sorted(set(samples)),
        "selected_umis": int(cells.sum()),
    }
    return arrays, audit


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def process_shard(
    item: dict[str, object],
    revision: str,
    wanted: set[str],
    scratch: Path,
    destination: str,
) -> dict[str, object]:
    name = Path(str(item["path"])).stem
    work = scratch / name
    work.mkdir(parents=True, exist_ok=True)
    source = work / f"{name}.parquet"
    output = work / f"{name}.npz"
    manifest = work / f"{name}.json"
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{item['path']}"

    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "5",
            "--retry-all-errors",
            "--continue-at",
            "-",
            url,
            "--output",
            str(source),
        ],
        check=True,
    )
    source_hash = sha256(source)
    expected_hash = item.get("lfs", {}).get("oid")  # type: ignore[union-attr]
    if expected_hash and source_hash != expected_hash:
        raise ValueError(
            f"SHA-256 mismatch for {name}: {source_hash} != {expected_hash}"
        )

    table = pq.read_table(source, columns=COLUMNS)
    arrays, audit = aggregate_table(table, wanted)
    np.savez_compressed(output, **arrays)
    record = {
        "dataset": REPO,
        "revision": revision,
        "target_panel_sha256": hashlib.sha256(
            ("\n".join(sorted(wanted)) + "\n").encode()
        ).hexdigest(),
        "source_path": item["path"],
        "source_url": url,
        "source_size": item["size"],
        "source_sha256": source_hash,
        "output_size": output.stat().st_size,
        "output_sha256": sha256(output),
        "created_utc": datetime.now(UTC).isoformat(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "scipy": scipy.__version__,
        },
        **audit,
    }
    manifest.write_text(json.dumps(record, indent=2) + "\n")

    remote = f"{destination.rstrip('/')}/{name}"
    subprocess.run(
        ["gcloud", "storage", "cp", str(output), f"{remote}.npz"], check=True
    )
    subprocess.run(
        ["gcloud", "storage", "cp", str(manifest), f"{remote}.json"], check=True
    )
    source.unlink()
    output.unlink()
    manifest.unlink()
    work.rmdir()
    return record


def completed_shards(destination: str) -> set[str]:
    result = subprocess.run(
        ["gcloud", "storage", "ls", f"{destination.rstrip('/')}/*.json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return set()
    return {Path(line).stem for line in result.stdout.splitlines()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", choices=["HCT116", "HEK293T"], required=True)
    parser.add_argument("--targets", type=Path, action="append", required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-shards", type=int)
    args = parser.parse_args()

    wanted = load_targets(args.targets)
    revision, files = dataset_files(args.context)
    done = completed_shards(args.destination)
    pending = [item for item in files if Path(str(item["path"])).stem not in done]
    if args.max_shards is not None:
        pending = pending[: args.max_shards]
    print(
        json.dumps(
            {
                "context": args.context,
                "revision": revision,
                "target_union": len(wanted),
                "total_shards": len(files),
                "completed_shards": len(done),
                "scheduled_shards": len(pending),
            },
            indent=2,
        ),
        flush=True,
    )

    args.scratch.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_shard,
                item,
                revision,
                wanted,
                args.scratch,
                args.destination,
            ): item
            for item in pending
        }
        for future in as_completed(futures):
            record = future.result()
            print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
