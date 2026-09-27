"""Audit closure and effect energy; quantify a fixed-control multinomial sampling null.

This diagnostic adds no fitted coefficients and does not alter the transfer experiment.
Run: pixi run python -m scripts.evaluation.audit_kolf_h1_transfer_mse
"""

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluation.context_transfer_blog import sha256
from scripts.evaluation.context_transfer_h1 import (
    H1_AGGREGATION,
    H1_PHASE1,
    SOURCE_SPECS,
    compact_source,
    h1_context,
)
from scripts.evaluation.kolf_h1_transfer_rules import (
    RULES,
    close_prediction,
    cpm,
    log_profile,
    transfer,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/kolf-h1-transfer-rules/mse-audit"
INPUT = ROOT / "data/derived/kolf_h1_transfer_rules/profiles.npz"


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    with np.load(INPUT) as a:
        data = {key: a[key] for key in a.files}
    keep = data["metric_gene_mask"]
    h0 = data["h1_control"]
    k0 = data["kolf_control"]
    y = (log_profile(data["h1_truth"]) - log_profile(h0))[:, keep]
    Y = np.square(y).sum()
    decomposition = []
    by_gene = []
    by_target = []
    for rule in RULES:
        p = (log_profile(data[rule]) - log_profile(h0))[:, keep]
        pre = (
            log_profile(np.maximum(data[rule + "_before_floor"], 0)) - log_profile(h0)
        )[:, keep]
        P = np.square(p).sum()
        C = np.sum(p * y)
        error = np.square(p - y)
        ratio = error.sum() / Y
        np.testing.assert_allclose(ratio, 1 + P / Y - 2 * C / Y)
        decomposition.append(
            {
                "rule": rule,
                "truth_energy": Y,
                "prediction_energy": P,
                "prediction_energy_over_truth": P / Y,
                "twice_dot_over_truth": 2 * C / Y,
                "pooled_norm_ratio": np.sqrt(P / Y),
                "pooled_cosine": C / np.sqrt(P * Y),
                "error_ratio": ratio,
                "without_final_closure_error_ratio": np.square(pre - y).sum() / Y,
                "closure_change_energy_over_truth": np.square(p - pre).sum() / Y,
                "source_perturbed_zero_error_fraction": error[
                    data["kolf_perturbed"][:, keep] == 0
                ].sum()
                / error.sum(),
            }
        )
        bins = [0, 0.1, 1, 5, 20, 100, np.inf]
        for low, high in pairwise(bins):
            mask = (k0[keep] >= low) & (k0[keep] < high)
            by_gene.append(
                {
                    "rule": rule,
                    "kolf_control_CPM_min": low,
                    "kolf_control_CPM_max": high,
                    "genes": int(mask.sum()),
                    "error_share": error[:, mask].sum() / error.sum(),
                    "truth_energy_share": np.square(y[:, mask]).sum() / Y,
                }
            )
        for i, t in enumerate(data["targets"]):
            by_target.append(
                {
                    "rule": rule,
                    "target_gene": t,
                    "error_share": error[i].sum() / error.sum(),
                    "prediction_energy": np.square(p[i]).sum(),
                    "truth_energy": np.square(y[i]).sum(),
                }
            )
    pd.DataFrame(decomposition).to_csv(REPORT / "decomposition.csv", index=False)
    pd.DataFrame(by_gene).to_csv(REPORT / "expression_bins.csv", index=False)
    pd.DataFrame(by_target).to_csv(REPORT / "per_target.csv", index=False)
    spec = SOURCE_SPECS["KOLF2.1J"]
    k = compact_source(spec)
    ki = [k.genes.index(g) for g in data["genes"]]
    kr = [k.targets.index(t) for t in data["targets"]]
    libraries = np.asarray(k.counts[kr][:, ki].sum(axis=1)).ravel().astype(np.int64)
    cells = k.cells[kr]
    h = h1_context()
    hi = [h.genes.index(g) for g in data["genes"]]
    hr = [h.targets.index(t) for t in data["targets"]]
    h_libraries = np.asarray(h.counts[hr][:, hi].sum(axis=1)).ravel()
    pd.DataFrame(
        {
            "target_gene": data["targets"],
            "kolf_shared_UMIs": libraries,
            "kolf_cells": cells,
            "kolf_UMIs_per_cell": libraries / cells,
            "h1_shared_UMIs": h_libraries,
            "h1_cells": h.cells[hr],
            "h1_UMIs_per_cell": h_libraries / h.cells[hr],
        }
    ).to_csv(REPORT / "source_depth.csv", index=False)
    # With no sampling noise, an unchanged source must yield an unchanged destination.
    for rule in RULES:
        p, _ = close_prediction(
            transfer(np.broadcast_to(k0, (len(libraries), len(k0))), k0, h0, rule)
        )
        np.testing.assert_allclose(p, np.broadcast_to(h0, p.shape), atol=1e-10)
    rows = []
    rng = np.random.default_rng(20260925)
    probs = k0 / k0.sum()
    for repeat in range(10):
        # Condition on each observed target's total UMI count. All genes are sampled
        # jointly from the pooled KOLF control composition: zero true source response.
        counts = np.vstack([rng.multinomial(int(n), probs) for n in libraries])
        np.testing.assert_array_equal(counts.sum(axis=1), libraries)
        for rule in RULES:
            prediction, _ = close_prediction(transfer(cpm(counts), k0, h0, rule))
            p = (log_profile(prediction) - log_profile(h0))[:, keep]
            rows.append(
                {
                    "repeat": repeat,
                    "rule": rule,
                    "null_prediction_energy_over_H1_truth": np.square(p).sum() / Y,
                    "error_ratio_against_actual_H1": np.square(p - y).sum() / Y,
                    "twice_dot_over_H1_truth": 2 * np.sum(p * y) / Y,
                }
            )
    null = pd.DataFrame(rows)
    null.to_csv(REPORT / "count_sampling_null.csv", index=False)
    summary = null.groupby("rule").agg(["mean", "min", "max"])
    summary.to_csv(REPORT / "count_sampling_null_summary.csv")
    inputs = [INPUT, spec.target_path, spec.manifest_path, H1_AGGREGATION, H1_PHASE1]
    manifest = {
        "seed": 20260925,
        "replicates": 10,
        "null": "Independent perturbation pseudobulks sampled multinomially at observed KOLF shared-gene UMI totals, with probabilities fixed to pooled KOLF control composition. No true donor response. Controls treated as known.",
        "limits": "Technical count-sampling diagnostic, not biological cell-split reliability. Ignores cell heterogeneity, control uncertainty, and changes in sampling variance under actual perturbations. Do not subtract it as an unbiased biological variance estimate.",
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
        "producer": {str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__))},
        "outputs": {str(p.relative_to(ROOT)): sha256(p) for p in REPORT.glob("*.csv")},
    }
    (REPORT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pd.DataFrame(decomposition).to_string(index=False))
    print(summary.to_string())


if __name__ == "__main__":
    main()
