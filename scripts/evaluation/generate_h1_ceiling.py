"""Self-consistency ceiling for the H1 harness: real H1 cells as the prediction.

For each target, 400 of its H1 perturbed cells outside the harness's fixed 400-cell reference
sample. Targets with at least 400 such cells are sampled without replacement (fully disjoint
from the reference); the rest are resampled with replacement from the cells available and are
flagged, since duplicated cells inflate prediction-side DE significance. The score of this
prediction is what a perfect model of the mean and distribution could reach, given sampling.
Run: pixi run python -m scripts.evaluation.generate_h1_ceiling OUT.h5ad [--seed S]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from scripts.evaluation.generate_replogle_h1_phase1 import sha256
from scripts.evaluation.h1_generation import ROOT, canonical_axes
from scripts.linear_response.kernel import write_prediction

H1 = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
REFERENCE = ROOT / "reports/vcc2026-h1/reference_cells.csv"
REPORT = ROOT / "reports/h1-harness-review/ceiling"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    targets, genes, _ = canonical_axes()
    reference = set(pd.read_csv(REFERENCE).obs_name.astype(str))
    h1 = ad.read_h5ad(H1, backed="r")
    assert h1.var_names.astype(str).tolist() == genes.tolist()
    labels = h1.obs["target_gene"].astype(str).to_numpy()
    names = h1.obs_names.astype(str).to_numpy()
    in_reference = np.isin(names, list(reference))
    rng = np.random.default_rng(args.seed)
    picks, support = {}, []
    for t in targets:
        pool = np.flatnonzero((labels == t) & ~in_reference)
        disjoint = len(pool) >= 400
        chosen = np.sort(rng.choice(pool, 400, replace=not disjoint))
        picks[t] = chosen
        support.append({"target_gene": t, "available": len(pool), "disjoint": disjoint})
    unique = np.unique(np.concatenate(list(picks.values())))
    block = h1.X[unique]  # one sorted backed read
    block = sparse.csr_matrix(block)
    assert np.all(block.data >= 0) and np.array_equal(block.data, np.round(block.data))
    block = block.astype(np.int32)
    h1.file.close()
    where = {row: i for i, row in enumerate(unique)}
    blocks = {t: block[[where[r] for r in picks[t]]] for t in targets}
    write_prediction(blocks, genes.tolist(), Path(args.output))
    REPORT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(support).to_csv(REPORT / "support.csv", index=False)
    info = {
        "seed": args.seed,
        "disjoint_targets": int(sum(s["disjoint"] for s in support)),
        "prediction_sha256": sha256(Path(args.output)),
    }
    (REPORT / "generation.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
