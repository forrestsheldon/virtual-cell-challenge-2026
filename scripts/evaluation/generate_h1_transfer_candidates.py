"""Generate frozen transfer candidates for the H1 harness (vcc-h1).

Candidates use the five-anchor panel (110 targets, 7,446 genes) at transfer pseudocount
c = 10 CPM, with the per-fold coefficients already fitted by h1_transfer_regression
(each target uses the fold in which it was held out). Each candidate is turned into a
per-gene log2 factor on the 18,080-gene H1 axis and decoded by h1_generation (dependent
rounding on the canonical 400 H1 control cells, with a fidelity check).
Out-of-panel genes and the 16 targets outside the panel keep factor 1 (unchanged).
Run: pixi run python -m scripts.evaluation.generate_h1_transfer_candidates NAME OUT.h5ad
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import scripts.evaluation.h1_transfer_regression as reg
from scripts.evaluation.h1_generation import canonical_axes, generate
from scripts.evaluation.kolf_h1_transfer_rules import cpm

C = 10.0
COEFS = reg.ROOT / "reports/h1-transfer-regression/coefficients.csv"
REPORT = reg.ROOT / "reports/h1-transfer-regression/cell_eval_v3"
CANDIDATES = {
    "avg_direct": ("avg:direct", "E"),
    "avg_scaled": ("avg:scaled", "F"),
    "study_avg_scaled": ("study_avg:scaled", "F"),
    "multi_nnls": ("multi_nnls", "E"),
    "h1_mean": ("h1_mean", "E"),
    "avg_h1_mean": ("avg:h1_mean+scaled", "E"),
}


def fold_coefficients(model, space):
    c = pd.read_csv(COEFS)
    pc = 0 if model == "h1_mean" else C
    c = c.query("panel == 'five_anchors' and model == @model and space == @space")
    c = c[np.isclose(c.pseudocount, pc)]
    return {
        f: dict(zip(g.coefficient, g.value, strict=True)) for f, g in c.groupby("fold")
    }


def panel_profiles(name):
    """Predicted CPM on the panel genes for each of the 110 panel targets."""
    model, space = CANDIDATES[name]
    strong = pd.read_csv(reg.STRONG).query("stable_strong")
    p = reg.Panel("five_anchors", *reg.load_panel("five_anchors"), strong)
    names = list(p.sources)
    d = {
        s: reg.direction(a["source_perturbed"], a["source_control"], C)
        for s, a in p.sources.items()
    }
    studies = pd.Series({s: reg.STUDY[s] for s in names})
    w = {s: 1 / studies.nunique() / (studies == studies[s]).sum() for s in names}
    d["avg"] = sum(d[s] for s in names) / len(names)
    d["study_avg"] = sum(w[s] * d[s] for s in names)
    base = reg.log_profile(p.h0)

    def full_effect(profile):  # effect on every panel gene, not only scored ones
        return reg.log_profile(profile) - base

    q = {k: full_effect(reg.realize(v, p.h0, C)) for k, v in d.items()}
    y_full = full_effect(p.hp)
    y_full[:, ~p.keep] = 0  # H1 mean never carries other targets' own knockdowns
    coefs = fold_coefficients(model, space)
    out = np.empty_like(p.hp)
    for f in range(reg.FOLDS):
        test, train = p.fold == f, p.fold != f
        b = coefs[f]
        if model == "avg:direct":
            out[test] = reg.realize(d["avg"][test], p.h0, C)
            continue
        if space == "F":
            out[test] = reg.realize(d[model.split(":")[0]][test], p.h0, C, b["a"])
            continue
        if model == "multi_nnls":
            eff = sum(b[s] * q[s][test] for s in names)
        elif model == "h1_mean":
            eff = np.broadcast_to(y_full[train].mean(axis=0), (test.sum(), len(base)))
        else:  # avg:h1_mean+scaled
            eff = b["m"] * y_full[train].mean(axis=0) + b["a"] * q["avg"][test]
        out[test] = cpm(np.maximum(20 * np.expm1(base + eff), 0))
    return p, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=sorted(CANDIDATES))
    parser.add_argument("output")
    parser.add_argument("--tag", default="", help="suffix changing the rounding seed")
    args = parser.parse_args()
    p, profiles = panel_profiles(args.name)
    # Positive control: held-out scores match the regression summary.
    frame = p.score(p.effect(profiles))
    summary = pd.read_csv(reg.REPORT / "summary.csv")
    model, space = CANDIDATES[args.name]
    pc = 0 if model == "h1_mean" else C
    row = summary[(summary.panel == "five_anchors") & (summary.model == model) & (summary.space == space)]
    row = row[np.isclose(row.pseudocount, pc)].iloc[0]
    got = frame.squared_error.sum() / frame.truth_energy.sum()
    print(f"MSE ratio {got:.5f} vs regression {row.profile_mse_ratio:.5f}", flush=True)
    assert abs(got - row.profile_mse_ratio) < 5e-3
    targets, genes, _ = canonical_axes()
    gi = {g: i for i, g in enumerate(genes)}
    cols = np.array([gi[g] for g in p.genes])
    with np.errstate(divide="ignore", invalid="ignore"):
        panel_factor = np.where(p.h0 > 0, np.log2(profiles / p.h0), 0.0)
    factor = np.zeros((len(targets), len(genes)))
    for i, t in enumerate(p.targets):
        factor[list(targets).index(t), cols] = panel_factor[i]
    label = args.name + args.tag
    generate(label, factor, args.output, REPORT / label)


if __name__ == "__main__":
    main()
