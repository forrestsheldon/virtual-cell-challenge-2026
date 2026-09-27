"""Harness-review predictions: unbiased global shift and depth-inflated nulls.

shift:   the control + global perturbation axis baseline of build_shifted_control_prediction
         (additive shift in per-cell CPM, same 400 canonical control cells), but integerized
         by stochastic rounding, floor + Bernoulli(fraction), instead of np.rint. The rint
         version erased most of the shift with an error shared by every target.
depth K: no perturbation signal. Every cell is drawn as Poisson(K * N_c * p), with N_c the
         canonical control cell's library size and p the pooled control profile. K = 1
         isolates sampling from a mean profile; K = 2, 4 add depth. Any DE "gain" is artifact.
Run: pixi run python -m scripts.evaluation.generate_h1_review_nulls {shift|depth} OUT [--k K]
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from scripts.evaluation.build_shifted_control_prediction import cpm_mean_by_group
from scripts.evaluation.generate_replogle_h1_phase1 import CONTROLS, sha256
from scripts.evaluation.h1_generation import ROOT, canonical_axes, harness_log
from scripts.linear_response.kernel import write_prediction

H1 = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
DELTA = ROOT / "data/derived/vcc2026_h1/global_shift_delta.npz"
REPORT = ROOT / "reports/h1-harness-review"


def global_delta():
    if not DELTA.exists():
        mu_pert, mu_ctrl = cpm_mean_by_group(H1)
        np.savez_compressed(DELTA, delta=mu_pert - mu_ctrl)
    with np.load(DELTA) as a:
        return a["delta"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["shift", "depth"])
    parser.add_argument("output")
    parser.add_argument("--k", type=float, default=1.0)
    args = parser.parse_args()
    targets, genes, rows = canonical_axes()
    controls = ad.read_h5ad(CONTROLS, backed="r")
    assert controls.var_names.astype(str).tolist() == genes.tolist()
    label = "global_shift_stochastic" if args.kind == "shift" else f"depth_null_x{args.k:g}"
    if args.kind == "shift":
        delta = global_delta()
    else:
        pooled = np.zeros(len(genes))
        for start in range(0, controls.n_obs, 5000):
            pooled += np.asarray(controls.X[start : start + 5000].sum(axis=0)).ravel()
        profile = pooled / pooled.sum()
    blocks, errors, cosines = {}, [], []
    try:
        for k, t in enumerate(targets):
            rng = np.random.default_rng(zlib.crc32(f"{label}:{t}".encode()))
            x = controls.X[rows[k]].toarray().astype(float)
            lib = x.sum(axis=1, keepdims=True)
            if args.kind == "shift":
                expected = np.clip(x / lib * 1e6 + delta[None, :], 0, None) * lib / 1e6
                low = np.floor(expected)
                counts = low + (rng.random(expected.shape) < expected - low)
                base = harness_log(x.sum(axis=0))
                ideal, got = harness_log(expected.sum(axis=0)), harness_log(counts.sum(axis=0))
                errors.append(got - ideal)
                want, real = ideal - base, got - base
                cosines.append(want @ real / np.linalg.norm(want) / np.linalg.norm(real))
            else:
                counts = rng.poisson(args.k * lib * profile[None, :])
            blocks[t] = sparse.csr_matrix(counts.astype(np.int32))
    finally:
        controls.file.close()
    summary = {"candidate": label}
    if errors:
        e = np.array(errors)
        summary["median_cosine"] = float(np.median(cosines))
        summary["constant_error_share"] = float(
            np.square(e.mean(axis=0)).sum() * len(e) / np.square(e).sum()
        )
        assert summary["median_cosine"] > 0.95, summary
    write_prediction(blocks, genes.tolist(), Path(args.output))
    summary["prediction_sha256"] = sha256(Path(args.output))
    (REPORT / label).mkdir(parents=True, exist_ok=True)
    (REPORT / label / "generation.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
