"""Shared cell generation for H1 harness candidates, with a built-in fidelity check.

A candidate is a per-gene log2 factor for each of the 126 targets on the 18,080-gene H1
axis, relative to the pooled control profile. It is applied to the canonical 400 H1 control
cells per target (the same cells as `vcc-h1 score-control-baseline`).

Defaults: `match_sample` re-aims each target's factor at its own 400-cell sample, so the
generated pseudobulk targets the intended profile rather than the profile times that
sample's deviation from the pooled mean; `rounding="gene"` integerizes so every gene's
pseudobulk is exact to within one count (see decode_lfc_genewise). `rounding="cell"` is
per-cell dependent rounding, which keeps library sizes exactly. The earlier deterministic
largest-remainder decoder erased sub-count changes identically in every cell and is not
used here.

Fidelity compares, per target, the generated pseudobulk with the same cells' unrounded
expectation in the harness space log1p(pseudobulk / total * 50,000). Generation fails
unless the median realized-vs-intended cosine clears MIN_COSINE.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from scripts.evaluation.generate_replogle_h1_phase1 import CONTROLS, sha256
from scripts.linear_response.kernel import decode_lfc_dependent_round, write_prediction

ROOT = Path(__file__).resolve().parents[2]
PHASE1 = ROOT / "data/derived/replogle_h1_transfer/phase1_effects.npz"
MIN_COSINE = 0.95


def harness_log(pseudobulk):
    return np.log1p(pseudobulk / pseudobulk.sum() * 5e4)


def canonical_axes():
    with np.load(PHASE1) as a:
        return (
            a["target_gene"].astype(str),
            a["output_gene"].astype(str),
            a["source_rows"].astype(int),
        )


def expected_counts(raw, lfc):
    x = raw.toarray().astype(float)
    w = x * np.exp2(lfc)[None, :]
    return x.sum(axis=1, keepdims=True) * w / w.sum(axis=1, keepdims=True)


def pooled_control_cpm(controls):
    total = np.zeros(controls.n_vars)
    for start in range(0, controls.n_obs, 5000):
        total += np.asarray(controls.X[start : start + 5000].sum(axis=0)).ravel()
    return total / total.sum() * 1e6


def decode_lfc_genewise(raw, lfc, rng):
    """Integerize an LFC shift so every gene's pseudobulk is exact to within one count.

    Per cell, the expected counts are N_c * w / sum(w) with w = counts * 2**lfc, as in
    dependent rounding. Each entry is rounded down, and the leftover fractions are then
    assigned down each gene's column: cells are visited in an independent random order per
    gene, and a cell gets one extra count where the running sum of fractions crosses
    u, u + 1, u + 2, ... for a uniform offset u. Every entry is unbiased (its extra count has
    probability equal to its fraction), and each gene's 400-cell total differs from its
    expectation by less than one. Cell library sizes are preserved only on average.
    """
    expected = expected_counts(raw, lfc)
    low = np.floor(expected)
    frac = expected - low
    order = np.argsort(rng.random(frac.shape), axis=0)
    running = np.cumsum(np.take_along_axis(frac, order, axis=0), axis=0)
    offset = rng.random(frac.shape[1])
    crossed = np.floor(running - offset[None, :])
    extra = np.diff(crossed, axis=0, prepend=np.full((1, frac.shape[1]), -1.0))
    counts = low.copy()
    np.put_along_axis(counts, order, np.take_along_axis(low, order, axis=0) + extra, axis=0)
    gap = counts.sum(axis=0) - expected.sum(axis=0)
    assert np.all(np.abs(gap) < 1 + 1e-9), "gene-wise rounding missed a column total"
    return sparse.csr_matrix(counts.astype(np.int32))


def generate(name, factor, output, report_dir, match_sample=True, rounding="gene"):
    """Decode a (126, 18080) log2-factor matrix, check fidelity, write the prediction.

    Factors are relative to the pooled control profile. With match_sample, each target's
    factor is re-aimed at its own 400-cell sample, so the generated pseudobulk targets the
    intended profile instead of the intended profile times that sample's deviation from the
    pooled mean (genes absent from all 400 cells stay absent).
    """
    targets, genes, rows = canonical_axes()
    assert factor.shape == (len(targets), len(genes))
    factor = np.where(np.isfinite(factor), factor, -60.0)  # log2(0): gene switched off
    controls = ad.read_h5ad(CONTROLS, backed="r")
    assert controls.var_names.astype(str).tolist() == genes.tolist()
    if match_sample:
        pooled = pooled_control_cpm(controls)
    blocks, stats = {}, []
    errors, intended = [], []
    try:
        for k, t in enumerate(targets):
            raw = controls.X[rows[k]].tocsr()
            if match_sample:
                sample = np.asarray(raw.sum(axis=0)).ravel()
                sample = sample / sample.sum() * 1e6
                with np.errstate(divide="ignore"):
                    shift = np.where(sample > 0, np.log2(pooled / sample), 0.0)
                factor[k] = factor[k] + np.where(np.isfinite(shift), shift, 0.0)
            rng = np.random.default_rng(zlib.crc32(f"{name}:{t}".encode()))
            if not np.any(factor[k]):
                block = sparse.csr_matrix(raw.astype(np.int32))
            elif rounding == "gene":
                block = decode_lfc_genewise(raw, factor[k], rng)
            else:
                block = decode_lfc_dependent_round(raw, factor[k], rng)
            blocks[t] = block
            if not np.any(factor[k]):
                stats.append({"target_gene": t, "changed": False})
                continue
            base = harness_log(np.asarray(raw.sum(axis=0)).ravel())
            ideal = harness_log(expected_counts(raw, factor[k]).sum(axis=0))
            got = harness_log(np.asarray(block.sum(axis=0)).ravel())
            want, real = ideal - base, got - base
            cos = float(want @ real / (np.linalg.norm(want) * np.linalg.norm(real)))
            stats.append(
                {
                    "target_gene": t,
                    "changed": True,
                    "cosine": cos,
                    "norm_ratio": float(np.linalg.norm(real) / np.linalg.norm(want)),
                    "rounding_error_ratio": float(np.linalg.norm(got - ideal) / np.linalg.norm(want)),
                }
            )
            errors.append(got - ideal)
            intended.append(want)
    finally:
        controls.file.close()
    changed = [s for s in stats if s["changed"]]
    summary = {"candidate": name, "targets_changed": len(changed)}
    if changed:
        e = np.array(errors)
        summary.update(
            {
                "median_cosine": float(np.median([s["cosine"] for s in changed])),
                "median_norm_ratio": float(np.median([s["norm_ratio"] for s in changed])),
                "median_rounding_error_ratio": float(
                    np.median([s["rounding_error_ratio"] for s in changed])
                ),
                # Near 1/n for independent errors; near 1 for a bias shared by every target.
                "constant_error_share": float(
                    np.square(e.mean(axis=0)).sum() * len(e) / np.square(e).sum()
                ),
            }
        )
        if summary["median_cosine"] < MIN_COSINE:
            raise AssertionError(f"generation fidelity failed: {summary}")
    write_prediction(blocks, genes.tolist(), Path(output))
    summary["prediction_sha256"] = sha256(Path(output))
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "generation.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
    return summary
