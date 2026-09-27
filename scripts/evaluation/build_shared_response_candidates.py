"""Donor shared-response candidates on the full H1 axis (all 126 targets).

Each anchor source's shared response m_s is its mean c = 10 CPM log change over the H1
targets it measures, excluding each target's own gene. Sources are averaged per gene over
those that measure it, giving m. The target-specific part is the available-source average
minus m (zero where no source covering the target measures the gene). Two fold-change-
space models are cross-fitted over five folds of the 126 targets:

* shared_only:          d_t = b m
* shared_plus_specific: d_t = a (dbar_t - m) + b m

No H1 perturbation data enters m; H1 targets are used only to fit a and b.
Run: pixi run python -m scripts.evaluation.build_shared_response_candidates OUTDIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import scripts.evaluation.h1_transfer_regression as reg
from scripts.evaluation.build_available_average_candidate import ANCHORS, H1, PHASE1, C
from scripts.evaluation.h1_generation import generate
from scripts.evaluation.kolf_h1_transfer_rules import cpm

REPORT = reg.ROOT / "reports/h1-transfer-regression/shared_response"


def load():
    with np.load(PHASE1) as a:
        targets = a["target_gene"].astype(str)
        genes = a["output_gene"].astype(str)
    with np.load(H1) as a:
        assert (a["target_gene"].astype(str) == targets).all()
        h0, hp = cpm(a["h1_control_count_sum"]), cpm(a["h1_count_sum"])
    ti = {t: i for i, t in enumerate(targets)}
    gi = {g: i for i, g in enumerate(genes)}
    total, count = np.zeros(hp.shape), np.zeros(hp.shape, dtype=int)
    m_sum, m_n = np.zeros(len(genes)), np.zeros(len(genes), dtype=int)
    for s in ANCHORS:
        with np.load(reg.SOURCE / f"available_{s}/{s}.npz") as z:
            st = z["targets"].astype(str)
            sg = z["genes"].astype(str)
            change = reg.direction(z["source_perturbed"], z["source_control"], C)
        r = np.array([ti[t] for t in st])
        g = np.array([gi[x] for x in sg])
        total[np.ix_(r, g)] += change
        count[np.ix_(r, g)] += 1
        own = sg[None, :] == st[:, None]  # each target's own gene
        n = (~own).sum(axis=0)
        m_s = np.where(own, 0.0, change).sum(axis=0) / n
        m_sum[g] += m_s
        m_n[g] += 1
    dbar = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    m = np.divide(m_sum, m_n, out=np.zeros_like(m_sum), where=m_n > 0)
    specific = np.where(count > 0, dbar - m[None, :], 0.0)
    return targets, genes, h0, hp, np.broadcast_to(m, hp.shape), specific


def cross_fit(panel, h0, m, specific, free_a):
    profiles = np.empty_like(panel.hp)
    params = []
    for f in range(reg.FOLDS):
        train, test = panel.fold != f, panel.fold == f
        ytr = panel.y[train]

        def loss(x, train=train, ytr=ytr):
            a, b = (x if free_a else (0.0, x[0]))
            d = a * specific[train] + b * m[train]
            return np.square(panel.effect(reg.realize(d, h0, C)) - ytr).sum()

        x0 = [0.13, 0.13] if free_a else [0.13]
        bounds = [(0.0, 2.0)] * len(x0)
        x = minimize(loss, x0, method="L-BFGS-B", bounds=bounds).x
        a, b = (x if free_a else (0.0, x[0]))
        params.append({"fold": f, "a": a, "b": b})
        profiles[test] = reg.realize(a * specific[test] + b * m[test], h0, C)
    return profiles, pd.DataFrame(params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir")
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    targets, genes, h0, hp, m, specific = load()
    strong = pd.read_csv(reg.STRONG).query("stable_strong")
    panel = reg.Panel("shared_response", {}, {"targets": targets, "genes": genes, "h1_control": h0, "h1_truth": hp}, strong)

    # Diagnostic only: does the donor shared response point along H1's own mean response?
    shared_eff = panel.effect(reg.realize(m[:1], h0, C))[0]
    h1_mean = panel.y.mean(axis=0)
    cos = shared_eff @ h1_mean / np.linalg.norm(shared_eff) / np.linalg.norm(h1_mean)
    print(f"cosine(donor shared response, H1 mean response) = {cos:.3f}", flush=True)

    rows_out, fits = [], []
    for name, free_a in (("shared_only", False), ("shared_plus_specific", True)):
        profiles, params = cross_fit(panel, h0, m, specific, free_a)
        frame = panel.score(panel.effect(profiles))
        rows_out.append(
            {
                "model": name,
                **frame.drop(columns=["squared_error", "truth_energy", "strong_genes"]).mean(),
                "profile_mse_ratio": frame.squared_error.sum() / frame.truth_energy.sum(),
            }
        )
        fits.append(params.assign(model=name))
        print(rows_out[-1]["model"], {k: round(rows_out[-1][k], 4) for k in ("retrieval", "cosine", "profile_mse_ratio")}, flush=True)
        print(params.round(4).to_string(index=False), flush=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(h0 > 0, np.log2(profiles / h0), 0.0)
        generate(name, factor, Path(args.outdir) / f"{name}.h5ad", reg.ROOT / "reports/h1-transfer-regression/cell_eval_v3" / name)
    pd.DataFrame(rows_out).assign(shared_vs_h1_mean_cosine=cos).to_csv(REPORT / "summary.csv", index=False)
    pd.concat(fits).to_csv(REPORT / "coefficients.csv", index=False)


if __name__ == "__main__":
    main()
