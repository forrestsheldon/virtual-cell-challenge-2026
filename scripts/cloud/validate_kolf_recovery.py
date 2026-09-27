"""Validate corrected KOLF pseudobulks and build the explicit H1 analysis view."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

CONTROL = "NTC"
H1_CONTROL = "non-targeting"
TARGET_SUM = 1_000_000.0
EPSILON = 1e-9
EXPECTED_MISSING_H1 = {"CAST", "CHMP3", "TAZ"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def find_file(directory: Path, suffix: str) -> Path:
    matches = list(directory.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise ValueError(f"expected one *{suffix} in {directory}, found {matches}")
    return matches[0]


def assert_rows_equal(old: ad.AnnData, new: ad.AnnData) -> None:
    if not old.var_names.equals(new.var_names):
        raise ValueError("old and corrected KOLF gene axes differ")
    missing = old.obs_names.difference(new.obs_names)
    if len(missing):
        raise ValueError(f"corrected KOLF is missing old rows: {missing.tolist()}")
    aligned = new[old.obs_names]
    if (old.X != aligned.X).nnz:
        raise ValueError("old and corrected KOLF counts differ")
    for column in ("gene_target", "n_cells", "total_umis"):
        if column in old.obs and not old.obs[column].equals(aligned.obs[column]):
            raise ValueError(f"old and corrected KOLF {column} values differ")


def effect(counts: np.ndarray, control: np.ndarray) -> np.ndarray:
    normalized = counts.astype(np.float64)
    baseline = control.astype(np.float64)
    normalized *= TARGET_SUM / normalized.sum(axis=1, keepdims=True)
    baseline *= TARGET_SUM / baseline.sum()
    return np.log2((normalized + EPSILON) / (baseline[None, :] + EPSILON))


def aggregate_h1(
    path: Path, reference: pd.DataFrame, targets: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = ad.read_h5ad(path, backed="r")
    try:
        genes = data.var_names.astype(str).tolist()
        counts = np.empty((len(targets), data.n_vars), dtype=np.int64)
        for index, target in enumerate(targets):
            rows = np.sort(
                reference.loc[reference.target_gene == target, "source_row"].to_numpy(
                    dtype=int
                )
            )
            counts[index] = np.asarray(data.X[rows].sum(axis=0)).ravel().astype(np.int64)
        labels = data.obs["target_gene"].astype(str).to_numpy()
        control_rows = np.flatnonzero(labels == H1_CONTROL)
        control = np.zeros(data.n_vars, dtype=np.int64)
        for start in range(0, len(control_rows), 4096):
            control += np.asarray(
                data.X[control_rows[start : start + 4096]].sum(axis=0)
            ).ravel().astype(np.int64)
    finally:
        data.file.close()
    return counts, control, genes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--h1-targets", type=Path, required=True)
    parser.add_argument("--source-counts", type=Path, required=True)
    parser.add_argument("--h1-data", type=Path, required=True)
    parser.add_argument("--view-output", type=Path, required=True)
    parser.add_argument("--effects-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = find_file(args.new, "_manifest.json")
    audit_path = find_file(args.new, "_audit.json")
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "passed":
        raise ValueError("source aggregation audit did not pass")

    new_target_path = find_file(args.new, "_target_pseudobulk.h5ad")
    new_guide_path = find_file(args.new, "_guide_pseudobulk.h5ad")
    new_target_batch_path = find_file(args.new, "_target_batch_pseudobulk.h5ad")
    new_control_path = find_file(args.new, "_control_channel_pseudobulk.h5ad")
    new_target = ad.read_h5ad(new_target_path)
    new_guide = ad.read_h5ad(new_guide_path)
    new_target_batch = ad.read_h5ad(new_target_batch_path)
    new_control = ad.read_h5ad(new_control_path)
    if new_target.n_obs != 395:
        raise ValueError(f"corrected KOLF target rows: {new_target.n_obs} != 395")

    reference = pd.read_csv(args.h1_targets)
    h1_requested = set(reference.target_gene.astype(str))
    h1_available = h1_requested & (set(new_target.obs_names) - {CONTROL})
    if len(h1_available) != 123 or h1_requested - h1_available != EXPECTED_MISSING_H1:
        raise ValueError("corrected KOLF does not contain the expected 123 H1 targets")

    source_cells = pd.read_csv(args.source_counts).set_index("source_target_gene")[
        "n_cells"
    ]
    source_cells.loc[CONTROL] = source_cells.loc["control"]
    observed_cells = new_target.obs["n_cells"].astype(int)
    expected_cells = source_cells.reindex(new_target.obs_names)
    if expected_cells.isna().any():
        raise ValueError(
            "source metadata is missing corrected KOLF labels: "
            f"{expected_cells.index[expected_cells.isna()].tolist()}"
        )
    expected_cells = expected_cells.astype(int)
    if not observed_cells.equals(expected_cells):
        raise ValueError("corrected target cell counts differ from source metadata")

    comparisons = {}
    for suffix, new_data in (
        ("_target_pseudobulk.h5ad", new_target),
        ("_guide_pseudobulk.h5ad", new_guide),
        ("_target_batch_pseudobulk.h5ad", new_target_batch),
        ("_control_channel_pseudobulk.h5ad", new_control),
    ):
        old_data = ad.read_h5ad(find_file(args.old, suffix))
        assert_rows_equal(old_data, new_data)
        comparisons[suffix] = old_data.n_obs

    h1_order = [
        target
        for target in reference.drop_duplicates("target_gene").target_gene.astype(str)
        if target in h1_available
    ]
    view_rows = [CONTROL, *h1_order]
    h1_view = new_target[view_rows].copy()
    if h1_view.n_obs != 124:
        raise ValueError(f"KOLF H1 view rows: {h1_view.n_obs} != 124")
    h1_view.uns["parent_union"] = {
        "manifest_sha256": sha256(manifest_path),
        "target_pseudobulk_sha256": sha256(new_target_path),
    }
    args.view_output.mkdir(parents=True, exist_ok=True)
    view_path = args.view_output / "KOLF2.1J_H1_target_pseudobulk.h5ad"
    h1_view.write_h5ad(view_path, compression="gzip", compression_opts=4)
    view_manifest = {
        "kind": "KOLF2.1J H1 target-pseudobulk view",
        "parent_manifest_sha256": sha256(manifest_path),
        "parent_target_pseudobulk_sha256": sha256(new_target_path),
        "targets": 123,
        "controls": 1,
        "genes": h1_view.n_vars,
        "missing_h1_targets": sorted(EXPECTED_MISSING_H1),
        "output": {
            "path": view_path.name,
            "size": view_path.stat().st_size,
            "sha256": sha256(view_path),
        },
    }
    view_manifest_path = args.view_output / "KOLF2.1J_H1_manifest.json"
    view_manifest_path.write_text(json.dumps(view_manifest, indent=2) + "\n")

    kolf_genes = new_target.var["gene_name"].astype(str).tolist()
    if len(kolf_genes) != len(set(kolf_genes)):
        raise ValueError("corrected KOLF gene symbols are not unique")
    kolf_lookup = {gene: index for index, gene in enumerate(kolf_genes)}
    kolf_control = new_target[CONTROL].X.toarray().ravel().astype(np.int64)
    kolf_counts = new_target[h1_order].X.toarray().astype(np.int64)
    h1_counts, h1_control, h1_genes = aggregate_h1(
        args.h1_data, reference, h1_order
    )
    h1_lookup = {gene: index for index, gene in enumerate(h1_genes)}
    shared = [gene for gene in h1_genes if gene in kolf_lookup]
    if len(shared) != 17603:
        raise ValueError(f"KOLF-H1 shared response genes: {len(shared)} != 17603")
    kolf_columns = np.asarray([kolf_lookup[gene] for gene in shared])
    h1_columns = np.asarray([h1_lookup[gene] for gene in shared])
    kolf_effect = effect(kolf_counts, kolf_control)[:, kolf_columns]
    h1_effect = effect(h1_counts, h1_control)[:, h1_columns]
    if not np.isfinite(kolf_effect).all() or not np.isfinite(h1_effect).all():
        raise ValueError("non-finite KOLF-H1 smoke-test effects")
    args.effects_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.effects_output,
        target_gene=np.asarray(h1_order),
        shared_gene=np.asarray(shared),
        kolf_lfc_native=kolf_effect,
        h1_lfc_native=h1_effect,
    )

    report = {
        "status": "passed",
        "canonical_target_rows": new_target.n_obs,
        "h1_view_rows": h1_view.n_obs,
        "h1_targets": len(h1_order),
        "shared_response_genes": len(shared),
        "source_cell_counts_match": True,
        "old_rows_exactly_reproduced": comparisons,
        "manifest_sha256": sha256(manifest_path),
        "target_pseudobulk_sha256": sha256(new_target_path),
        "guide_pseudobulk_sha256": sha256(new_guide_path),
        "target_batch_pseudobulk_sha256": sha256(new_target_batch_path),
        "control_channel_pseudobulk_sha256": sha256(new_control_path),
        "h1_view_sha256": sha256(view_path),
        "effects_sha256": sha256(args.effects_output),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
