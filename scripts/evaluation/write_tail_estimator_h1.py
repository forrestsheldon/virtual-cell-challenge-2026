"""Write the selected H1 tail cells as a transductive scoring artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scripts.evaluation.scan_empirical_scale import sha256
from scripts.linear_response.kernel import write_prediction

ROOT = Path(__file__).resolve().parents[2]
H1 = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
REFERENCE = ROOT / "reports/vcc2026-h1/reference_cells.csv"
SELECTED = ROOT / "reports/tail-estimator-h1/selected_source_rows.npy"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selected", type=Path, default=SELECTED)
    parser.add_argument("--kind", default="transductive bottom-tail prediction")
    args = parser.parse_args()

    targets = (
        pd.read_csv(REFERENCE)
        .sort_values(["source_order", "source_row"])
        .drop_duplicates("source_order")["target_gene"]
        .astype(str)
        .tolist()
    )
    selected = np.load(args.selected)
    if selected.shape != (len(targets), 400):
        raise AssertionError("expected 400 selected cells for each target")

    data = ad.read_h5ad(H1, backed="r")
    try:
        counts = {
            target: data.X[np.sort(selected[index])].tocsr()
            for index, target in enumerate(targets)
        }
        genes = data.var_names.astype(str).tolist()
    finally:
        data.file.close()
    write_prediction(counts, genes, args.output)

    transductive = args.selected.resolve() == SELECTED.resolve()
    payload = {
        "kind": args.kind,
        "same_cells_used_for_blinded_selection_and_reference_truth": transductive,
        "labels_used_for_selection": False,
        "labels_used_to_construct_balanced_pool": transductive,
        "selected_source_rows": {
            "path": str(args.selected.relative_to(ROOT)),
            "sha256": sha256(args.selected),
        },
        "prediction": {
            "path": str(args.output.relative_to(ROOT)),
            "sha256": sha256(args.output),
            "bytes": args.output.stat().st_size,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
