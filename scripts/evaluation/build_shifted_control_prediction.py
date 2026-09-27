"""Build the 'control + global perturbation axis' prediction for the H1 benchmark.

The unchanged-control baseline (`vcc-h1 score-control-baseline`) relabels 400
real control cells as each target and scores ~0 -- it contains no perturbation
signal. This baseline adds the ONE global perturbation direction to those same
cells: predicted_cell = control_cell + (mean_perturbed - mean_control), applied in
linear CPM space and converted back to integer counts at the cell's own library
size. The predicted per-cell MEAN therefore equals the global perturbed mean (the
Systema average-effect / "perturbed mean" baseline), while each cell keeps the
control population's heterogeneity.

It uses the SAME 400 control cells per target as score-control-baseline (identical
CRC32 seed), so the two scores differ only by the shift -- a paired estimate of
what a single shared-axis correction buys, and a proxy for the same baseline on the
withheld 2026 panel.
"""

import argparse
import zlib
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

CONTROL = "non-targeting"
CELLS_PER_TARGET = 400
SEED_NAMESPACE = "h1-control-baseline-v1"   # must match the public vcc-h1 scorer


def cpm_mean_by_group(h1_path, chunk=20_000):
    """Per-gene mean CPM over perturbed cells and over control cells (one pass)."""
    adata = ad.read_h5ad(h1_path, backed="r")
    is_control = adata.obs["target_gene"].astype(str).to_numpy() == CONTROL
    n_genes = adata.n_vars
    sums = {"pert": np.zeros(n_genes), "ctrl": np.zeros(n_genes)}
    counts = {"pert": 0, "ctrl": 0}
    for start in range(0, adata.n_obs, chunk):
        stop = min(start + chunk, adata.n_obs)
        block = adata.X[start:stop].tocsr().astype(np.float64)
        libr = np.asarray(block.sum(axis=1)).ravel()
        cpm = block.multiply(1e6 / libr[:, None]).tocsr()          # per-cell CPM
        ctrl = is_control[start:stop]
        sums["ctrl"] += np.asarray(cpm[ctrl].sum(axis=0)).ravel()
        sums["pert"] += np.asarray(cpm[~ctrl].sum(axis=0)).ravel()
        counts["ctrl"] += int(ctrl.sum())
        counts["pert"] += int((~ctrl).sum())
    return sums["pert"] / counts["pert"], sums["ctrl"] / counts["ctrl"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1", type=Path, default=Path("data/external/vcc2025_h1/adata_Training.h5ad"))
    parser.add_argument("--controls", type=Path, default=Path("data/derived/vcc2026_h1/h1_controls.h5ad"))
    parser.add_argument("--reference-cells", type=Path, default=Path("reports/vcc2026-h1/reference_cells.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/derived/vcc2026_h1/shifted_control_prediction.h5ad"))
    parser.add_argument("--scale", type=float, default=1.0, help="multiplier on the global shift")
    args = parser.parse_args()

    import pandas as pd
    mu_pert, mu_ctrl = cpm_mean_by_group(args.h1)
    delta = args.scale * (mu_pert - mu_ctrl)                       # global perturbation axis, linear CPM
    print(f"global shift: {np.count_nonzero(delta)} genes move; "
          f"L1 {np.abs(delta).sum():.0f} CPM, max |delta| {np.abs(delta).max():.1f}")

    controls = ad.read_h5ad(args.controls, backed="r")
    control_rows = np.arange(controls.n_obs)
    targets = pd.read_csv(args.reference_cells).drop_duplicates("target_gene")["target_gene"].astype(str).tolist()
    assert len(targets) == 126

    blocks, labels = [], []
    for target in targets:
        seed = zlib.crc32(f"{SEED_NAMESPACE}:{target}".encode())
        picked = np.sort(control_rows[np.random.default_rng(seed).choice(
            len(control_rows), CELLS_PER_TARGET, replace=False)])
        X = controls.X[picked].tocsr().astype(np.float64)
        libr = np.asarray(X.sum(axis=1)).ravel()                  # each cell's own library size
        cpm = np.asarray(X.todense()) * (1e6 / libr[:, None])     # 400 x 18080, dense per target
        shifted = np.clip(cpm + delta[None, :], 0.0, None)
        counts = np.rint(shifted * (libr[:, None] / 1e6)).astype(np.int64)
        blocks.append(sp.csr_matrix(counts))
        labels.extend([target] * CELLS_PER_TARGET)

    X = sp.vstack(blocks, format="csr")
    obs = pd.DataFrame({"target_gene": labels})
    pred = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=controls.var_names))
    pred.X = pred.X.astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pred.write_h5ad(args.output)
    print(f"wrote {args.output}: {pred.shape}, {X.nnz} nonzeros, "
          f"all integer/non-negative: {(X.data >= 0).all() and np.allclose(X.data, np.rint(X.data))}")


if __name__ == "__main__":
    main()
