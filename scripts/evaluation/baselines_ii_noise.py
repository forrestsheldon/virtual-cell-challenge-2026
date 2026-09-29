"""Baselines II: noise-only check at c = 10, inverse-noise averaging, cosine vs. noise figure.

Five-source panel (110 H1 targets, 7,446 genes), direct transfer (a = 1) at c = 10 CPM,
scored with the same held-out-free profile metrics as h1_transfer_regression.

* Noise-only check: each source's perturbed pseudobulk is replaced by a multinomial draw
  from its own pooled control composition at the observed per-target UMI total (no biological
  response), then transferred. Prediction energy of that null, over the actual prediction
  energy, is the source's noise fraction. The same is done for the equal-screen average,
  with every source drawn independently.
* Inverse-noise averages (no H1 data): per source, weights proportional to 1 / null energy;
  per gene, weights proportional to 1 / Var[log(y_p + c)] from the delta method with
  Poisson CPM variance y_p * 1e6 / L_p (control variance neglected: pooled controls are
  far deeper than any target pseudobulk).
Run: pixi run python -m scripts.evaluation.baselines_ii_noise
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

import scripts.evaluation.h1_transfer_regression as reg
from scripts.evaluation.kolf_h1_transfer_rules import cpm

matplotlib.use("Agg")
from matplotlib import pyplot as plt

C = 20.0  # frozen for Baselines II: the official PDS/MSE scoring geometry ln(1 + CPM/20)
REPEATS = 10
REPORT = reg.ROOT / "reports/baselines-ii"
DEPTH = reg.ROOT / "reports/all-source-h1-transfer/five_anchors/depth.csv"
COLORS = {  # dataviz reference palette, categorical slots 1-5, fixed per source
    "K562_GWPS": "#2a78d6",
    "HCT116": "#eb6834",
    "HEK293T": "#1baf7a",
    "KOLF2.1J": "#eda100",
    "CD4T": "#e87ba4",
}


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    strong = pd.read_csv(reg.STRONG).query("stable_strong")
    panel = reg.Panel("five_anchors", *reg.load_panel("five_anchors"), strong)
    depth = pd.read_csv(DEPTH).set_index(["source", "target_gene"]).shared_UMIs
    libs = {s: depth.loc[s].reindex(panel.targets).to_numpy() for s in panel.sources}
    run(panel, libs, f"c{C:g}", check=True)


def run(panel, libs, tag, check=False, sweep=()):
    """Noise-only check, averages, and figure for one panel; optional pseudocount sweep."""
    names = list(panel.sources)
    d = {
        s: reg.direction(a["source_perturbed"], a["source_control"], C)
        for s, a in panel.sources.items()
    }

    def energy(change, c=C):
        return np.square(panel.effect(reg.realize(change, panel.h0, c))).sum() / panel.energy

    # Positive control: at c = 1 the null reproduces the stored all-source noise-only check.
    if check:
        _null_check(panel, libs, energy)
    return _rest(panel, libs, tag, names, d, energy, sweep)


def _null_check(panel, libs, energy):
    stored = pd.read_csv(DEPTH.parent / "count_sampling_null.csv")
    s, s0 = "KOLF2.1J", panel.sources["KOLF2.1J"]["source_control"]
    check_rng = np.random.default_rng(0)
    counts = np.vstack([check_rng.multinomial(int(n), s0 / s0.sum()) for n in libs[s]])
    got = energy(reg.direction(cpm(counts), s0, 1.0), c=1.0)
    want = stored.query("source == @s and rule == 'log1p'").null_prediction_energy_over_truth.mean()
    print(f"c = 1 null check, KOLF: {got:.3f} vs stored {want:.3f}", flush=True)
    assert abs(got / want - 1) < 0.05


def _rest(panel, libs, tag, names, d, energy, sweep):
    # Noise-only check, per source and for the equal-screen average.
    rng = np.random.default_rng(reg.SEED)
    null = {s: [] for s in [*names, "avg"]}
    for _ in range(REPEATS):
        draws = {}
        for s in names:
            s0 = panel.sources[s]["source_control"]
            counts = np.vstack([rng.multinomial(int(n), s0 / s0.sum()) for n in libs[s]])
            draws[s] = reg.direction(cpm(counts), s0, C)
        for s in names:
            null[s].append(energy(draws[s]))
        null["avg"].append(energy(sum(draws.values()) / len(names)))

    # Sampling noise around each target's measured perturbed pseudobulk (parametric bootstrap):
    # resample at the observed depth with the measured composition, transfer, and take the
    # energy of the deviation from the actual prediction.
    def effect(change):
        return panel.effect(reg.realize(change, panel.h0, C))

    actual = {s: effect(d[s]) for s in names}
    actual["avg"] = effect(sum(d.values()) / len(names))
    boot = {s: [] for s in [*names, "avg"]}
    for _ in range(REPEATS):
        draws = {}
        for s in names:
            sp, s0 = panel.sources[s]["source_perturbed"], panel.sources[s]["source_control"]
            counts = np.vstack([rng.multinomial(int(n), row / row.sum()) for n, row in zip(libs[s], sp, strict=True)])
            draws[s] = reg.direction(cpm(counts), s0, C)
        for s in names:
            boot[s].append(np.square(effect(draws[s]) - actual[s]).sum() / panel.energy)
        boot["avg"].append(np.square(effect(sum(draws.values()) / len(names)) - actual["avg"]).sum() / panel.energy)

    # Predictors: single screens, equal-screen, per-source inverse-noise, per-gene inverse-variance.
    predictors = {s: d[s] for s in names}
    predictors["avg"] = sum(d.values()) / len(names)
    study = pd.Series({s: reg.STUDY[s] for s in names})
    w_study = {s: 1 / study.nunique() / (study == study[s]).sum() for s in names}
    predictors["study_avg"] = sum(w_study[s] * d[s] for s in names)
    inv = {s: 1 / np.mean(null[s]) for s in names}
    w_src = {s: inv[s] / sum(inv.values()) for s in names}
    predictors["inv_noise_source"] = sum(w_src[s] * d[s] for s in names)
    var = {
        s: panel.sources[s]["source_perturbed"] * 1e6 / libs[s][:, None]
        / (panel.sources[s]["source_perturbed"] + C) ** 2
        for s in names
    }
    w_gene = {s: 1 / np.maximum(var[s], 1e-12) for s in names}
    total = sum(w_gene.values())
    predictors["inv_var_gene"] = sum(w_gene[s] / total * d[s] for s in names)

    rows = []
    for name, change in predictors.items():
        q = panel.effect(reg.realize(change, panel.h0, C))
        frame = panel.score(q)
        row, _ = reg.summarize(panel, frame, C, name, "direct", [], [])
        row["prediction_energy"] = np.square(q).sum() / panel.energy
        if name in null:
            row["null_energy"] = float(np.mean(null[name]))
            row["perturbed_null_energy"] = float(np.mean(boot[name]))
        if name in names:  # closed form (delta method) for the same sampling noise
            y0, x0 = panel.sources[name]["source_control"][panel.keep], panel.h0[panel.keep]
            per_gene = ((x0 + C) / (x0 + 20)) ** 2 * y0 / (y0 + C) ** 2
            row["closed_form_null_energy"] = float((1e6 / libs[name]).sum() * per_gene.sum() / panel.energy)
            yp = panel.sources[name]["source_perturbed"][:, panel.keep]
            per_entry = ((x0 + C) / (x0 + 20))[None, :] ** 2 * yp / (yp + C) ** 2
            row["closed_form_perturbed_null_energy"] = float(((1e6 / libs[name])[:, None] * per_entry).sum() / panel.energy)
            row["noise_fraction"] = row["null_energy"] / row["prediction_energy"]
        if name in w_src:
            row["inverse_noise_weight"] = w_src[name]
        wrong = [panel.score(q[perm]) for perm in panel.wrong]  # deranged donor targets
        row["wrong_target_retrieval"] = float(np.mean([w.retrieval.mean() for w in wrong]))
        row["wrong_target_cosine"] = float(np.mean([w.cosine.mean() for w in wrong]))
        rows.append(row)
    table = pd.DataFrame(rows)
    cols = ["model", "retrieval", "cosine", "cosine_q025", "cosine_q975", "signed_top100", "profile_mse_ratio", "mse_q025",
            "mse_q975", "prediction_energy", "null_energy", "noise_fraction", "inverse_noise_weight",
            "closed_form_null_energy", "perturbed_null_energy",
            "closed_form_perturbed_null_energy", "wrong_target_retrieval", "wrong_target_cosine"]
    table[cols].to_csv(REPORT / f"noise_and_averaging_{tag}.csv", index=False)
    print(table[cols].round(4).to_string(index=False), flush=True)

    # Figure: cosine vs. noise fraction for the single screens.
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    single = table[table.model.isin(names)]
    for r in single.itertuples():
        ax.scatter(r.noise_fraction, r.cosine, s=70, color=COLORS[r.model], zorder=3,
                   edgecolor="white", linewidth=2, label=r.model.replace("_", " "))
        ax.annotate(r.model.replace("_", " "), (r.noise_fraction, r.cosine),
                    xytext=(8, 4), textcoords="offset points", fontsize=9, color="#333333")
    ax.set_xlabel("Noise-only prediction energy / actual prediction energy")
    ax.set_ylabel("Mean cosine with H1 response")
    ax.set_title(f"Direct transfer at c = {C:g} ({tag}): alignment vs. transferred noise", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.1, single.cosine.max() * 1.25))
    ax.grid(color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(REPORT / f"cosine_vs_noise_{tag}.png", dpi=200)
    fig.savefig(REPORT / f"cosine_vs_noise_{tag}.svg")
    plt.close(fig)
    sweep_rows = []
    for c in sweep:  # direct transfer by pseudocount, single screens and equal-screen average
        dc = {s: reg.direction(panel.sources[s]["source_perturbed"], panel.sources[s]["source_control"], c) for s in names}
        dc["avg"] = sum(dc.values()) / len(names)
        for name, change in dc.items():
            q = panel.effect(reg.realize(change, panel.h0, c))
            frame = panel.score(q)
            rho = float(np.sum(q * panel.y) / np.sqrt(np.sum(q * q) * panel.energy))
            sweep_rows.append({"panel": tag, "pseudocount": c, "model": name, "pooled_rho": rho,
                               "retrieval": frame.retrieval.mean(), "cosine": frame.cosine.mean(),
                               "profile_mse_ratio": frame.squared_error.sum() / frame.truth_energy.sum()})
    if sweep_rows:
        pd.DataFrame(sweep_rows).to_csv(REPORT / f"pseudocount_sweep_{tag}.csv", index=False)
    return table


if __name__ == "__main__":
    main()
