"""Compare three unfitted KOLF -> H1 mean-expression transfer rules.

Run from the repository root: pixi run python -m scripts.evaluation.kolf_h1_transfer_rules
No cell generation, H1 fitting, downloads, or official composite scoring.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scripts.evaluation.context_transfer_blog import sha256
from scripts.evaluation.context_transfer_h1 import (
    H1_AGGREGATION,
    H1_CONTROLS,
    H1_PHASE1,
    SOURCE_SPECS,
    compact_source,
    h1_context,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/kolf-h1-transfer-rules"
DERIVED = ROOT / "data/derived/kolf_h1_transfer_rules"
STRONG = ROOT / "reports/linear-response-three-models/strong_de_truth_genes.csv"
SEED = 20260925
RULES = ("additive", "ratio", "log1p")
METRICS = (
    "retrieval",
    "cosine",
    "signed_top100",
    "normalized_absolute_error",
    "norm_ratio",
    "strong_direction",
    "strong_normalized_absolute_error",
)


def cpm(counts):
    counts = np.asarray(counts, dtype=np.float64)
    return counts * (1e6 / counts.sum(axis=-1, keepdims=True))


def transfer(kp, k0, h0, rule):
    """Return the unbounded CPM prediction, before common boundary handling."""
    if rule == "additive":
        return h0 + kp - k0
    if rule == "ratio":
        ratio = np.divide(kp, k0, out=np.ones_like(kp, dtype=float), where=k0 > 0)
        return h0 * ratio
    if rule == "log1p":
        return (h0 + 1) * ((kp + 1) / (k0 + 1)) - 1
    raise ValueError(rule)


def close_prediction(raw):
    clipped = np.maximum(raw, 0)
    totals = clipped.sum(axis=1)
    assert np.all(totals > 0)
    audit = pd.DataFrame(
        {
            "negative_genes": (raw < 0).sum(axis=1),
            "negative_mass_cpm": -np.minimum(raw, 0).sum(axis=1),
            "raw_total_cpm": raw.sum(axis=1),
            "clipped_total_cpm": totals,
            "renormalization_factor": 1e6 / totals,
        }
    )
    return cpm(clipped), audit


def log_profile(values):
    """Common VCC-style mean-expression space, not a full official score."""
    return np.log1p(np.asarray(values) / 20)  # CPM -> counts per 50,000


def unit_rows(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x, dtype=float), where=norm > 0)


def retrieval(pred, truth):
    sim = unit_rows(pred) @ unit_rows(truth).T
    ranks = rankdata(-sim, method="average", axis=1) - 1
    return 1 - np.diag(ranks) / (len(truth) - 1)


def evaluate(prediction, truth, baseline, keep, strong):
    """All metrics share the panel-excluded gene axis; zeros have cosine 0."""
    p = (log_profile(prediction) - log_profile(baseline))[:, keep]
    y = (log_profile(truth) - log_profile(baseline))[:, keep]
    stable = strong[:, keep]
    scores = retrieval(p, y)
    rows = []
    for i, (x, t) in enumerate(zip(p, y, strict=True)):
        xnorm, tnorm = np.linalg.norm(x), np.linalg.norm(t)
        xtop = np.argsort(-np.abs(x), kind="stable")[:100]
        ttop = np.argsort(-np.abs(t), kind="stable")[:100]
        recovered = np.isin(xtop, ttop) & (np.sign(x[xtop]) == np.sign(t[xtop]))
        recovered &= (x[xtop] != 0) & (t[xtop] != 0)
        mask = stable[i]
        nstrong = int(mask.sum())
        rows.append(
            {
                "retrieval": scores[i],
                "cosine": float(x @ t / (xnorm * tnorm)) if xnorm * tnorm else 0.0,
                "signed_top100": float(recovered.sum() / min(100, len(t))),
                "normalized_absolute_error": np.abs(x - t).sum() / np.abs(t).sum(),
                "norm_ratio": xnorm / tnorm,
                "squared_error": np.square(x - t).sum(),
                "truth_energy": np.square(t).sum(),
                "strong_genes": nstrong,
                "strong_direction": float(np.mean(np.sign(x[mask]) == np.sign(t[mask])))
                if nstrong >= 10
                else np.nan,
                "strong_normalized_absolute_error": np.abs(x[mask] - t[mask]).sum()
                / np.abs(t[mask]).sum()
                if nstrong >= 10
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def stochastic_pseudobulk(rates, rng, n_cells=400):
    """Exact pooled law of independent per-cell/per-gene stochastic rounding.

    Sum of 400 Bernoulli fractions is Binomial(400, fraction). This avoids
    materializing cells; it does not model biological single-cell variability.
    """
    lower = np.floor(rates).astype(np.int64)
    return n_cells * lower + rng.binomial(n_cells, rates - lower)


def rounding_check(predictions, depth, truth, baseline, keep, strong, targets):
    rows, metrics, pooled = [], [], {}
    for seed in range(SEED, SEED + 5):
        for name, prediction in predictions.items():
            # Same random seed across arms; exact common draws are not asserted
            # because the binomial sampler may consume different random numbers.
            rates = prediction * depth / 1e6
            counts = stochastic_pseudobulk(rates, np.random.default_rng(seed))
            realized = counts / 400
            fraction = rates - np.floor(rates)
            variance = fraction * (1 - fraction) / 400
            error = realized - rates
            se = np.sqrt(variance.sum()) / error.size
            assert abs(error.mean()) < 7 * se
            rows.append(
                {
                    "model": name,
                    "seed": seed,
                    "depth": depth,
                    "mean_signed_UMI_error": error.mean(),
                    "mean_error_standard_error": se,
                    "mean_absolute_UMI_error": np.abs(error).mean(),
                    "max_absolute_UMI_error": np.abs(error).max(),
                    "ordinary_rounding_mean_absolute_UMI_error": np.abs(
                        np.rint(rates) - rates
                    ).mean(),
                    "post_rounding_total_relative_error_max": np.max(
                        np.abs(realized.sum(axis=1) / depth - 1)
                    ),
                }
            )
            f = evaluate(cpm(counts), truth, baseline, keep, strong)
            metrics.append(f.assign(model=name, seed=seed, target_gene=targets))
            if seed == SEED:
                pooled[name] = counts
    pd.DataFrame(rows).to_csv(REPORT / "rounding_audit.csv", index=False)
    f = pd.concat(metrics, ignore_index=True)
    f.to_csv(REPORT / "rounded_per_target.csv", index=False)
    summary = f.groupby(["model", "seed"])[list(METRICS)].mean()
    energy = f.groupby(["model", "seed"])[["squared_error", "truth_energy"]].sum()
    summary["profile_mse_ratio"] = energy.squared_error / energy.truth_energy
    summary.to_csv(REPORT / "rounded_summary.csv")
    np.savez_compressed(DERIVED / "stochastic_pseudobulks_seed_20260925.npz", **pooled)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    spec = SOURCE_SPECS["KOLF2.1J"]
    source_manifest = json.loads(spec.manifest_path.read_text())
    assert (
        sha256(spec.target_path)
        == source_manifest["outputs"][spec.target_path.name]["sha256"]
    )
    h1_manifest_path = ROOT / "reports/replogle-h1-transfer/checkpoint1_manifest.json"
    h1_manifest = json.loads(h1_manifest_path.read_text())
    assert (
        sha256(H1_AGGREGATION)
        == h1_manifest["outputs"][str(H1_AGGREGATION.relative_to(ROOT))]
    )
    k, h = compact_source(spec), h1_context()
    common = sorted(set(k.genes) & set(h.genes))
    targets = sorted(set(k.targets) & set(h.targets))
    ki = np.array([k.genes.index(g) for g in common])
    hi = np.array([h.genes.index(g) for g in common])
    kr = np.array([k.targets.index(t) for t in targets])
    hr = np.array([h.targets.index(t) for t in targets])
    selected = k.control_counts[ki] > 0
    gene_table = pd.DataFrame(
        {
            "gene": common,
            "included": True,
            "kolf_control_positive": selected,
            "kolf_control_counts": k.control_counts[ki],
            "h1_control_counts": h.control_counts[hi],
        }
    )
    gene_table.to_csv(REPORT / "gene_support.csv", index=False)
    genes = np.array(common)
    kp = cpm(k.counts[kr][:, ki].toarray())
    k0 = cpm(k.control_counts[ki])
    hp = cpm(h.counts[hr][:, hi].toarray())
    h0 = cpm(h.control_counts[hi])
    keep = ~np.isin(genes, targets)
    assert len(targets) == 123 and len(genes) == 17603
    strong = np.zeros_like(hp, dtype=bool)
    ti, gi = {t: i for i, t in enumerate(targets)}, {g: i for i, g in enumerate(genes)}
    for row in pd.read_csv(STRONG).query("stable_strong").itertuples():
        if row.target_gene in ti and row.feature in gi:
            strong[ti[row.target_gene], gi[row.feature]] = True
    predictions = {"unchanged": np.broadcast_to(h0, hp.shape).copy()}
    audits = []
    raw_predictions = {}
    for rule in RULES:
        raw_predictions[rule] = transfer(kp, k0, h0, rule)
        prediction, audit = close_prediction(raw_predictions[rule])
        predictions[rule] = prediction
        audits.append(audit.assign(rule=rule, target_gene=targets))
    audit = pd.concat(audits, ignore_index=True)
    audit.to_csv(REPORT / "boundary_audit.csv", index=False)
    frames = []
    for name, pred in predictions.items():
        frames.append(
            evaluate(pred, hp, h0, keep, strong).assign(model=name, target_gene=targets)
        )
    positive_frames = []
    for name, pred in predictions.items():
        positive_frames.append(
            evaluate(pred, hp, h0, keep & selected, strong).assign(
                model=name, target_gene=targets
            )
        )
    positive = pd.concat(positive_frames, ignore_index=True)
    positive.to_csv(REPORT / "donor_positive_per_target.csv", index=False)
    positive_summary = positive.groupby("model")[list(METRICS)].mean()
    pe = positive.groupby("model")[["squared_error", "truth_energy"]].sum()
    positive_summary["profile_mse_ratio"] = pe.squared_error / pe.truth_energy
    positive_summary.to_csv(REPORT / "donor_positive_summary.csv")
    # Target-independent prediction: equal mean of each rule's predicted profiles.
    for rule in RULES:
        pred = np.broadcast_to(predictions[rule].mean(axis=0), hp.shape)
        frames.append(
            evaluate(pred, hp, h0, keep, strong).assign(
                model=rule + "_common", target_gene=targets
            )
        )
    per_target = pd.concat(frames, ignore_index=True)
    per_target.to_csv(REPORT / "per_target.csv", index=False)
    summary = per_target.groupby("model")[list(METRICS)].mean()
    energy = per_target.groupby("model")[["squared_error", "truth_energy"]].sum()
    summary["profile_mse_ratio"] = energy.squared_error / energy.truth_energy
    summary["strong_eligible_targets"] = per_target.groupby(
        "model"
    ).strong_direction.count()
    summary.to_csv(REPORT / "summary.csv")
    rng = np.random.default_rng(SEED)
    boot = rng.integers(len(targets), size=(2000, len(targets)))
    comparisons = []
    for left, right in [
        ("additive", "unchanged"),
        ("ratio", "unchanged"),
        ("log1p", "unchanged"),
        ("ratio", "additive"),
        ("log1p", "additive"),
        ("log1p", "ratio"),
    ]:
        a = per_target.query("model == @left").set_index("target_gene").loc[targets]
        b = per_target.query("model == @right").set_index("target_gene").loc[targets]
        for metric in (*METRICS, "profile_mse_ratio"):
            if metric == "profile_mse_ratio":
                diff = (a.squared_error - b.squared_error).to_numpy()
                denom = a.truth_energy.to_numpy()
                samples = diff[boot].sum(axis=1) / denom[boot].sum(axis=1)
                estimate = diff.sum() / denom.sum()
            else:
                diff = (a[metric] - b[metric]).to_numpy()
                samples = np.nanmean(diff[boot], axis=1)
                estimate = np.nanmean(diff)
            lo, hi_ci = np.quantile(samples, [0.025, 0.975])
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "metric": metric,
                    "difference": estimate,
                    "ci_low": lo,
                    "ci_high": hi_ci,
                }
            )
    pd.DataFrame(comparisons).to_csv(REPORT / "paired_differences.csv", index=False)
    # Shuffling complete profiles is equivalent to shuffling donor perturbations:
    # every target uses the same H1 baseline and normalization rule.
    nulls = []
    permutations = []
    for repeat in range(20):
        perm = rng.permutation(len(targets))
        while np.any(perm == np.arange(len(targets))):
            perm = rng.permutation(len(targets))
        permutations.extend(
            {"repeat": repeat, "target_gene": t, "source_target": targets[j]}
            for t, j in zip(targets, perm, strict=True)
        )
        for rule in RULES:
            f = evaluate(predictions[rule][perm], hp, h0, keep, strong)
            row = f[list(METRICS)].mean().to_dict()
            row.update(
                repeat=repeat,
                model=rule,
                profile_mse_ratio=f.squared_error.sum() / f.truth_energy.sum(),
            )
            nulls.append(row)
    pd.DataFrame(nulls).to_csv(REPORT / "wrong_target_summary.csv", index=False)
    pd.DataFrame(permutations).to_csv(
        REPORT / "wrong_target_assignments.csv", index=False
    )
    pd.DataFrame(
        {"target_gene": targets, "kolf_cells": k.cells[kr], "h1_cells": h.cells[hr]}
    ).to_csv(REPORT / "targets.csv", index=False)
    np.savez_compressed(
        DERIVED / "profiles.npz",
        genes=genes,
        targets=targets,
        kolf_control=k0,
        kolf_perturbed=kp,
        h1_control=h0,
        h1_truth=hp,
        metric_gene_mask=keep,
        strong_mask=strong,
        **predictions,
        **{name + "_before_floor": raw for name, raw in raw_predictions.items()},
    )
    # Positive and negative implementation controls in the identical metric space.
    oracle = evaluate(hp, hp, h0, keep, strong)
    assert np.allclose(oracle.retrieval, 1) and np.allclose(oracle.cosine, 1)
    assert np.allclose(oracle.squared_error, 0)
    assert np.allclose(summary.loc["unchanged", "retrieval"], 0.5)
    assert np.allclose(summary.loc["unchanged", "profile_mse_ratio"], 1)
    for rule in RULES:
        assert np.allclose(summary.loc[rule + "_common", "retrieval"], 0.5)
        np.testing.assert_allclose(predictions[rule].sum(axis=1), 1e6)
    controls = ad.read_h5ad(H1_CONTROLS, backed="r")
    n_controls = controls.n_obs
    assert list(controls.var_names.astype(str)) == h.genes
    controls.file.close()
    depth = float(h.control_counts[hi].sum() / n_controls)
    rounding_check(predictions, depth, hp, h0, keep, strong, targets)
    plot(summary, pd.DataFrame(nulls), audit)
    inputs = [
        spec.target_path,
        spec.manifest_path,
        spec.audit_path,
        H1_AGGREGATION,
        H1_PHASE1,
        H1_CONTROLS,
        h1_manifest_path,
        STRONG,
        ROOT / "reports/vcc2026-h1/reference_cells.csv",
    ]
    code = [
        Path(__file__),
        ROOT / "scripts/evaluation/context_transfer_h1.py",
        ROOT / "scripts/evaluation/context_transfer_blog.py",
    ]
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "targets": len(targets),
        "genes": len(genes),
        "metric_genes": int(keep.sum()),
        "zero_kolf_control_genes_retained": int((~selected).sum()),
        "ratio_zero_handling": "Ratio=1 if KOLF control is zero; baseline unchanged before closure",
        "rounding": "Exact aggregate of 400 independently stochastic-rounded cells, five seeds; Binomial(400, fractional rate) + 400*floor(rate); no full cell matrix",
        "rounding_depth": depth,
        "h1_control_cells": n_controls,
        "missing_h1_targets": sorted(set(h.targets) - set(targets)),
        "normalization": "Pooled raw count sums; duplicate symbols summed; all 17,603 shared genes including donor control zeros; CPM over this fixed support",
        "rules": {
            "additive": "h0 + kp - k0",
            "ratio": "h0 * kp/k0",
            "log1p": "(h0+1)*(kp+1)/(k0+1)-1",
        },
        "boundaries": "floor negative values at zero, then close to 1e6 CPM in every arm",
        "evaluation": "log1p(CPM/20) effects; all 123 target genes excluded from every primary metric; no fitted parameters or generated cells; not full official scoring",
        "strong_gene_definition": "Existing H1 padj<0.05, abs(log2FC)>=0.5, concordant full sign in both halves in >=4/5 splits; >=10 retained genes per target. Diagnostic effects use current profile space, not official DE LFC.",
        "bootstrap": "2000 paired target bootstraps; fixed experiments and controls; no cell or study uncertainty",
        "controls": "exact truth recovery and zero/common-response null checks passed; 20 deranged target shuffles; self-transfer is not a replicate ceiling",
        "lineage": {"kolf": source_manifest, "h1": h1_manifest},
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
        "code": {str(p.relative_to(ROOT)): sha256(p) for p in code},
        "software": {
            p: version(p)
            for p in ["numpy", "pandas", "scipy", "anndata", "matplotlib", "cell-eval2"]
        },
        "python": platform.python_version(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "outputs": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [
                *REPORT.glob("*.csv"),
                *REPORT.glob("*.png"),
                *DERIVED.glob("*.npz"),
            ]
        },
    }
    (REPORT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string())


def plot(summary, nulls, audit):
    names = ["unchanged", *RULES]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ["retrieval", "cosine", "profile_mse_ratio"],
        [
            "Target retrieval (higher better)",
            "Effect cosine (higher better)",
            "Profile MSE / unchanged (lower better)",
        ],
        strict=True,
    ):
        ax.bar(
            names,
            summary.loc[names, metric],
            color=["#999999", "#277da8", "#e59036", "#47986c"],
        )
        if metric in ["retrieval", "cosine"]:
            means = nulls.groupby("model")[metric].mean()
            ax.scatter(
                RULES,
                means.loc[list(RULES)],
                color="black",
                marker="_",
                s=120,
                label="Wrong-target mean",
            )
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("KOLF → H1: fixed transfer rules, no H1 fitting")
    fig.savefig(REPORT / "comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), layout="constrained")
    for ax, col, title in zip(
        axes,
        ["negative_genes", "renormalization_factor"],
        ["Genes floored at zero per target", "Post-floor CPM rescaling"],
        strict=True,
    ):
        ax.boxplot([audit.query("rule == @r")[col] for r in RULES], tick_labels=RULES)
        ax.set_title(title)
    fig.savefig(REPORT / "boundaries.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
