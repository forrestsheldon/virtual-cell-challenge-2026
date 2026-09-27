"""Baselines II: c = 20 (the official PDS/MSE scoring geometry) against c = 10 and 30, and the
Wiener-shrinkage prediction of the optimal pseudocount.

* H1, five-source panel: direct transfer per source, equal-screen and equal-study averages,
  plus the cross-fitted effect-space scale for the average.
* Donor to donor (no H1 data): each five-source screen as destination, average of the others.
* Four-source panels (K562-expressed genes, all shared genes).
* Wiener prediction: treating the pseudocount as a per-gene shrinkage weight y0 / (y0 + c) on
  log changes, the noise-optimal value is c* = 1e6 / (N * s2), with N the donor pseudobulk
  depth and s2 the per-target signal variance of log changes (genes with control CPM > 50,
  Poisson noise of the perturbed pseudobulk subtracted).
Run: pixi run python -m scripts.evaluation.baselines_ii_c20
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.evaluation.baselines_ii_four_source as four
import scripts.evaluation.h1_transfer_regression as reg
from scripts.evaluation.donor_pair_pseudocount import NO_STRONG

PSEUDOCOUNTS = (10.0, 20.0, 30.0)
REPORT = reg.ROOT / "reports/baselines-ii"
DEPTH = reg.ROOT / "reports/all-source-h1-transfer/five_anchors/depth.csv"


def direct_rows(panel, label, extra_models=True):
    names = list(panel.sources)
    rows = []
    for c in PSEUDOCOUNTS:
        d = {
            s: reg.direction(a["source_perturbed"], a["source_control"], c)
            for s, a in panel.sources.items()
        }
        d["avg"] = sum(d[s] for s in names) / len(names)
        if extra_models:
            study = pd.Series({s: reg.STUDY[s] for s in names})
            w = {s: 1 / study.nunique() / (study == study[s]).sum() for s in names}
            d["study_avg"] = sum(w[s] * d[s] for s in names)
        for name, change in d.items():
            q = panel.effect(reg.realize(change, panel.h0, c))
            frame = panel.score(q)
            row = {
                "panel": label,
                "pseudocount": c,
                "model": name,
                "retrieval": frame.retrieval.mean(),
                "cosine": frame.cosine.mean(),
                "profile_mse_ratio": frame.squared_error.sum() / frame.truth_energy.sum(),
                "pooled_rho": float(np.sum(q * panel.y) / np.sqrt(np.sum(q * q) * panel.energy)),
            }
            if name == "avg" and extra_models:
                _, scaled, _ = reg.run_effect_model(panel, c, "avg", [q], ["a"], "ols")
                row["scaled_mse_ratio"] = scaled["profile_mse_ratio"]
            rows.append(row)
    return rows


def donor_rows():
    sources, common = reg.load_panel("five_anchors")
    rows = []
    for dest, arr in sources.items():
        fake = {**common, "h1_control": arr["source_control"], "h1_truth": arr["source_perturbed"]}
        others = {s: v for s, v in sources.items() if s != dest}
        panel = reg.Panel(f"donor:{dest}", others, fake, NO_STRONG)
        rows += [
            {**r, "destination": dest}
            for r in direct_rows(panel, "donor_to_donor", extra_models=False)
            if r["model"] == "avg"
        ]
    return rows


def wiener_rows():
    depth = pd.read_csv(DEPTH).set_index(["source", "target_gene"]).shared_UMIs
    sources, common = reg.load_panel("five_anchors")
    targets, genes = common["targets"].astype(str), common["genes"].astype(str)
    rows = []
    for s, arr in sources.items():
        sp, s0 = arr["source_perturbed"], arr["source_control"]
        n = depth.loc[s].reindex(targets).to_numpy()
        keep = (s0 > 50) & ~np.isin(genes, targets)
        lfc = np.log(np.maximum(sp[:, keep], 1e-3) / s0[keep])
        noise = (1e6 / n[:, None]) / np.maximum(sp[:, keep], 1e-3)
        s2 = np.maximum((lfc**2).mean(axis=1) - noise.mean(axis=1), 1e-6)
        cstar = 1e6 / (n * s2)
        rows.append(
            {
                "source": s,
                "median_depth": np.median(n),
                "median_signal_var": np.median(s2),
                "median_c_star": np.median(cstar),
                "c_star_q25": np.percentile(cstar, 25),
                "c_star_q75": np.percentile(cstar, 75),
            }
        )
    return rows


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    strong = pd.read_csv(reg.STRONG).query("stable_strong")
    rows = direct_rows(reg.Panel("five_anchors", *reg.load_panel("five_anchors"), strong), "h1_five_source")
    rows += donor_rows()
    raw, depth, h1, targets, all_genes, k562_genes = four.load_inputs(strong)
    for name, genes in (("h1_four_source_k562_genes", k562_genes), ("h1_four_source_all_genes", all_genes)):
        panel, _ = four.build(genes, targets, raw, depth, h1, strong, name)
        rows += direct_rows(panel, name, extra_models=False)
    table = pd.DataFrame(rows)
    table.to_csv(REPORT / "pseudocount_c10_c20_c30.csv", index=False)
    wiener = pd.DataFrame(wiener_rows())
    wiener.to_csv(REPORT / "wiener_pseudocount_prediction.csv", index=False)
    avg = table[table.model == "avg"].groupby(["panel", "pseudocount"])[["retrieval", "cosine", "profile_mse_ratio", "pooled_rho", "scaled_mse_ratio"]].mean()
    print(avg.round(4).to_string(), flush=True)
    single = table[(table.panel == "h1_five_source") & ~table.model.isin(["avg", "study_avg"])]
    print(single.pivot_table(index="model", columns="pseudocount", values="cosine").round(4).to_string())
    print(wiener.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
