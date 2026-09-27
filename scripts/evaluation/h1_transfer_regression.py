"""Pseudocount sweep, cross-fitted scale fits, and multi-source regression into H1.

Reuses the frozen count profiles written by all_source_h1_transfer; no downloads.
Two definitions of the scale `a` are kept distinct:

* effect space ("E"): the evaluated effect q = ln(1+pred/20) - ln(1+h0/20) is
  multiplied by a. Closed-form least squares; the clean object for the 1 - rho^2 bound.
* fold-change space ("F"): the transferred log fold change is multiplied by a before
  realizing a CPM profile, (h0 + c) * exp(a * log((kp + c)/(k0 + c))) - c, then floored
  and renormalized. This is the deployable version that generates counts.

Coefficients are fit on training targets (five fixed folds) and evaluated on held-out
targets only. Run: pixi run python -m scripts.evaluation.h1_transfer_regression
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize_scalar, nnls

from scripts.evaluation.kolf_h1_transfer_rules import (
    close_prediction,
    evaluate,
    log_profile,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/derived/all_source_h1_transfer"
REPORT = ROOT / "reports/h1-transfer-regression"
STRONG = ROOT / "reports/linear-response-three-models/strong_de_truth_genes.csv"
SEED = 20260925
EVAL_C = 20  # CPM pseudocount of the evaluation space ln(1 + CPM/EVAL_C)
FOLDS = 5
N_WRONG = 5
N_BOOT = 2000
PSEUDOCOUNTS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, np.inf)
STUDY = {
    "K562_GWPS": "Replogle2022",
    "K562_essential": "Replogle2022",
    "RPE1": "Replogle2022",
    "HepG2": "Nadig2025",
    "Jurkat": "Nadig2025",
    "HCT116": "XAtlasOrion",
    "HEK293T": "XAtlasOrion",
    "CD4T": "Zhu2026",
    "KOLF2.1J": "Nourreddine2026",
}
MULTI_PANELS = ("five_anchors", "all_nine")
STORED_RULE = {1.0: "log1p", 0.0: "ratio", np.inf: "additive"}


def load_panel(panel):
    sources, common = {}, None
    for path in sorted((SOURCE / panel).glob("*.npz")):
        with np.load(path) as a:
            shared = {k: a[k] for k in ("targets", "genes", "h1_control", "h1_truth")}
            if common is None:
                common = shared
            for k, v in shared.items():
                np.testing.assert_array_equal(v, common[k])
            sources[path.stem] = {
                k: a[k]
                for k in ("source_control", "source_perturbed", *STORED_RULE.values())
            }
    return sources, common


def direction(sp, s0, c):
    """Transferred change in the space selected by pseudocount c."""
    if np.isinf(c):
        return sp - s0
    if c == 0:
        ratio = np.divide(sp, s0, out=np.ones_like(sp), where=s0 > 0)
        with np.errstate(divide="ignore"):
            return np.log(ratio)
    return np.log((sp + c) / (s0 + c))


def realize(d, h0, c, a=1.0):
    if np.isinf(c):
        return close_prediction(h0 + a * d)[0]
    finite = np.where(np.isneginf(d), 0.0, d)
    scaled = np.where(np.isneginf(d), 0.0 if a > 0 else 1.0, np.exp(a * finite))
    return close_prediction((h0 + c) * scaled - c)[0]


def folds_for(n):
    fold = np.empty(n, dtype=int)
    fold[np.random.default_rng(SEED).permutation(n)] = np.arange(n) % FOLDS
    return fold


def derangements(n):
    rng = np.random.default_rng(SEED + 1)
    out = []
    while len(out) < N_WRONG:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            out.append(perm)
    return out


def gram(xs, rows):
    return np.array([[np.sum(a[rows] * b[rows]) for b in xs] for a in xs])


def fit(xs, y, rows, kind):
    if kind == "fixed":
        return np.ones(len(xs))
    g = gram(xs, rows)
    b = np.array([np.sum(x[rows] * y[rows]) for x in xs])
    if kind == "ols":
        return np.linalg.solve(g, b)
    r = cholesky(g)  # G = R^T R, so ||X beta - y||^2 = ||R beta - R^-T b||^2 + const
    return nnls(r, solve_triangular(r, b, trans="T"))[0]


class Panel:
    def __init__(self, name, sources, common, strong_table):
        self.name = name
        self.sources = sources
        self.targets = common["targets"].astype(str)
        self.genes = common["genes"].astype(str)
        self.h0 = common["h1_control"]
        self.hp = common["h1_truth"]
        self.keep = ~np.isin(self.genes, self.targets)
        self.strong = np.zeros(self.hp.shape, dtype=bool)
        ti = {t: i for i, t in enumerate(self.targets)}
        gi = {g: i for i, g in enumerate(self.genes)}
        for r in strong_table.itertuples():
            if r.target_gene in ti and r.feature in gi:
                self.strong[ti[r.target_gene], gi[r.feature]] = True
        self.n = len(self.targets)
        self.fold = folds_for(self.n)
        self.y = self.effect(self.hp)
        self.energy = np.square(self.y).sum()
        self.boot = np.random.default_rng(SEED).integers(self.n, size=(N_BOOT, self.n))
        self.wrong = derangements(self.n)

    def effect(self, profile):
        return (log_profile(profile) - log_profile(self.h0))[:, self.keep]

    def score(self, effect):
        """Evaluate an effect-space prediction with the shared evaluate()."""
        full = np.zeros(self.hp.shape)
        full[:, self.keep] = effect
        pred = EVAL_C * np.expm1(log_profile(self.h0) + full)
        return evaluate(pred, self.hp, self.h0, self.keep, self.strong)


def summarize(panel, frame, c, model, space, coefs, names):
    se, en = frame.squared_error.to_numpy(), frame.truth_energy.to_numpy()
    cos = frame.cosine.to_numpy()
    mse = se[panel.boot].sum(axis=1) / en[panel.boot].sum(axis=1)
    cm = cos[panel.boot].mean(axis=1)
    row = {
        "panel": panel.name,
        "pseudocount": c,
        "model": model,
        "space": space,
        "targets": panel.n,
        **frame.drop(columns=["squared_error", "truth_energy", "strong_genes"])
        .mean()
        .to_dict(),
        "profile_mse_ratio": se.sum() / en.sum(),
        "mse_q025": np.quantile(mse, 0.025),
        "mse_q975": np.quantile(mse, 0.975),
        "cosine_q025": np.quantile(cm, 0.025),
        "cosine_q975": np.quantile(cm, 0.975),
    }
    coef_rows = [
        {
            "panel": panel.name,
            "pseudocount": c,
            "model": model,
            "space": space,
            "fold": f,
            "coefficient": n,
            "value": v,
        }
        for f, vec in enumerate(coefs)
        for n, v in zip(names, vec, strict=True)
    ]
    return row, coef_rows


def run_effect_model(panel, c, model, xs, names, kind, y=None):
    """Cross-fit linear effect-space model; xs may depend on the training fold."""
    y = panel.y if y is None else y
    pred = np.zeros_like(panel.y)
    coefs = []
    for f in range(FOLDS):
        train, test = panel.fold != f, panel.fold == f
        design = xs(train) if callable(xs) else xs
        beta = fit(design, y, train, kind)
        coefs.append(beta)
        pred[test] = sum(b * x[test] for b, x in zip(beta, design, strict=True))
    frame = panel.score(pred)
    return frame, *summarize(panel, frame, c, model, "E", coefs, names)


def run_fold_change_scale(panel, c, model, d):
    """Cross-fit a single fold-change-space scale by bounded 1-D search."""
    pred = np.zeros_like(panel.y)
    coefs = []
    for f in range(FOLDS):
        train, test = panel.fold != f, panel.fold == f
        h0, ytr = panel.h0, panel.y[train]

        def loss(a, train=train, ytr=ytr, h0=h0):
            return np.square(panel.effect(realize(d[train], h0, c, a)) - ytr).sum()

        a = minimize_scalar(
            loss, bounds=(0.0, 1.5), method="bounded", options={"xatol": 1e-4}
        ).x
        coefs.append([a])
        pred[test] = panel.effect(realize(d[test], panel.h0, c, a))
    frame = panel.score(pred)
    return frame, *summarize(panel, frame, c, model, "F", coefs, ["a"])


def run_panel(panel, multi):
    rows, coef_rows, per_target = [], [], []

    def record(result, c, model, space):
        frame, row, coefs = result
        rows.append(row)
        coef_rows.extend(coefs)
        per_target.append(
            frame.assign(
                panel=panel.name,
                pseudocount=c,
                model=model,
                space=space,
                target_gene=panel.targets,
            )
        )

    unchanged = panel.score(np.zeros_like(panel.y))
    assert np.allclose(unchanged.squared_error, unchanged.truth_energy)
    record((unchanged, *summarize(panel, unchanged, 0, "unchanged", "-", [], [])), 0, "unchanged", "-")

    def h1_mean(train):
        return [np.broadcast_to(panel.y[train].mean(axis=0), panel.y.shape)]

    record(run_effect_model(panel, 0, "h1_mean", h1_mean, ["m"], "fixed"), 0, "h1_mean", "E")

    names = list(panel.sources)
    for c in PSEUDOCOUNTS:
        t0 = time.time()
        d, q = {}, {}
        for s, arr in panel.sources.items():
            sp, s0 = arr["source_perturbed"], arr["source_control"]
            d[s] = direction(sp, s0, c)
            direct = realize(d[s], panel.h0, c)
            if c in STORED_RULE:  # positive control: reproduces the frozen predictions
                np.testing.assert_allclose(direct, arr[STORED_RULE[c]], rtol=1e-9, atol=1e-6)
            q[s] = panel.effect(direct)
        for s in names:
            record(run_effect_model(panel, c, f"{s}:direct", [q[s]], ["a"], "fixed"), c, f"{s}:direct", "E")
            record(run_effect_model(panel, c, f"{s}:scaled", [q[s]], ["a"], "ols"), c, f"{s}:scaled", "E")
            record(run_fold_change_scale(panel, c, f"{s}:scaled", d[s]), c, f"{s}:scaled", "F")
            record(
                run_effect_model(panel, c, f"{s}:h1_mean+scaled", lambda tr, s=s: [*h1_mean(tr), q[s]], ["m", "a"], "ols"),
                c,
                f"{s}:h1_mean+scaled",
                "E",
            )
        if not multi or c == 0:  # log of a zero ratio has no finite average
            continue
        studies = pd.Series({s: STUDY[s] for s in names})
        weights = {
            "avg": {s: 1 / len(names) for s in names},
            "study_avg": {
                s: 1 / studies.nunique() / (studies == studies[s]).sum() for s in names
            },
        }
        for label, w in weights.items():
            assert np.isclose(sum(w.values()), 1)
            dbar = sum(w[s] * d[s] for s in names)
            qbar = panel.effect(realize(dbar, panel.h0, c))
            q[label] = qbar
            record(run_effect_model(panel, c, f"{label}:direct", [qbar], ["a"], "fixed"), c, f"{label}:direct", "E")
            record(run_effect_model(panel, c, f"{label}:scaled", [qbar], ["a"], "ols"), c, f"{label}:scaled", "E")
            record(run_fold_change_scale(panel, c, f"{label}:scaled", dbar), c, f"{label}:scaled", "F")
        record(
            run_effect_model(panel, c, "avg:h1_mean+scaled", lambda tr: [*h1_mean(tr), q["avg"]], ["m", "a"], "ols"),
            c,
            "avg:h1_mean+scaled",
            "E",
        )
        if "KOLF2.1J" in names:
            record(
                run_effect_model(
                    panel, c, "kolf_increment", [q["avg"], q["KOLF2.1J"] - q["avg"]], ["a", "b"], "ols"
                ),
                c,
                "kolf_increment",
                "E",
            )
        for kind in ("ols", "nnls"):
            record(
                run_effect_model(panel, c, f"multi_{kind}", [q[s] for s in names], names, kind),
                c,
                f"multi_{kind}",
                "E",
            )
            record(
                run_effect_model(
                    panel, c, f"h1_mean+multi_{kind}", lambda tr: [*h1_mean(tr), *(q[s] for s in names)], ["m", *names], kind
                ),
                c,
                f"h1_mean+multi_{kind}",
                "E",
            )
        # Negative controls: shuffled donor-target correspondence, refit on training folds.
        for r, perm in enumerate(panel.wrong):
            record(run_effect_model(panel, c, f"avg:scaled:wrong{r}", [q["avg"][perm]], ["a"], "ols"), c, f"avg:scaled:wrong{r}", "E")
            record(
                run_effect_model(panel, c, f"avg:h1_mean+scaled:wrong{r}", lambda tr, p=perm: [*h1_mean(tr), q["avg"][p]], ["m", "a"], "ols"),
                c,
                f"avg:h1_mean+scaled:wrong{r}",
                "E",
            )
        print(f"{panel.name} c={c}: {time.time() - t0:.1f}s", flush=True)
    return rows, coef_rows, per_target


def in_sample_checks(panel):
    """The all-target OLS scale must equal <q,y>/||q||^2 from the stored decomposition."""
    out = []
    dec = pd.read_csv(ROOT / f"reports/all-source-h1-transfer/{panel.name}/decomposition.csv")
    for s, arr in panel.sources.items():
        q = panel.effect(arr["log1p"])
        a = np.sum(q * panel.y) / np.sum(q * q)
        r = dec.query("source == @s and rule == 'log1p'").iloc[0]
        a_dec = r.twice_dot_over_truth / 2 / r.prediction_energy_over_truth
        np.testing.assert_allclose(a, a_dec, rtol=1e-6)
        rho2 = np.sum(q * panel.y) ** 2 / (np.sum(q * q) * panel.energy)
        out.append({"panel": panel.name, "source": s, "a_in_sample": a, "pooled_rho": np.sqrt(rho2), "mse_bound": 1 - rho2})
    return out


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    strong = pd.read_csv(STRONG).query("stable_strong")
    panels = [p.name for p in sorted(SOURCE.iterdir()) if p.is_dir()]
    panels = [*MULTI_PANELS, *(p for p in panels if p not in MULTI_PANELS)]
    rows, coef_rows, per_target, checks = [], [], [], []
    for name in panels:
        panel = Panel(name, *load_panel(name), strong)
        print(f"{name}: {panel.n} targets, {panel.keep.sum()} scored genes, {list(panel.sources)}", flush=True)
        checks.extend(in_sample_checks(panel))
        r, c, p = run_panel(panel, name in MULTI_PANELS)
        rows.extend(r)
        coef_rows.extend(c)
        per_target.extend(p)
        pd.DataFrame(rows).to_csv(REPORT / "summary.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(REPORT / "coefficients.csv", index=False)
    pd.concat(per_target, ignore_index=True).to_csv(REPORT / "per_target.csv.gz", index=False)
    pd.DataFrame(checks).to_csv(REPORT / "in_sample_bound.csv", index=False)
    (REPORT / "manifest.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "seed": SEED,
                "folds": FOLDS,
                "pseudocounts_cpm": [str(c) for c in PSEUDOCOUNTS],
                "inputs": str(SOURCE.relative_to(ROOT)),
                "definitions": __doc__,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
