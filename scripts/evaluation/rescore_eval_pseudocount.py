"""Rescore the transfer-pseudocount sweeps in evaluation spaces ln(1 + CPM/c_eval).

Tests whether the best transfer pseudocount (about 10-30 CPM when c_eval = 20) tracks
the evaluation geometry. c_eval = 20 must reproduce the earlier summaries.
Run: pixi run python -m scripts.evaluation.rescore_eval_pseudocount
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.evaluation.donor_pair_pseudocount as donor
import scripts.evaluation.h1_transfer_regression as reg
import scripts.evaluation.kolf_h1_transfer_rules as kolf

REPORT = reg.ROOT / "reports/eval-pseudocount-rescore"
EVAL_PSEUDOCOUNTS = (20.0, 1.0, 100.0)


def set_eval(c_eval):
    def log_profile(values):
        return np.log1p(np.asarray(values) / c_eval)

    kolf.log_profile = log_profile
    reg.log_profile = log_profile
    reg.EVAL_C = c_eval


def h1_sweep(c_eval, strong):
    rows = []
    for name in reg.MULTI_PANELS:
        panel = reg.Panel(name, *reg.load_panel(name), strong)
        names = list(panel.sources)
        for c in reg.PSEUDOCOUNTS:
            d = {
                s: reg.direction(a["source_perturbed"], a["source_control"], c)
                for s, a in panel.sources.items()
            }
            if c != 0:
                d["avg"] = sum(d[s] for s in names) / len(names)
            for label, dd in d.items():
                q = panel.effect(reg.realize(dd, panel.h0, c))
                direct = panel.score(q)
                _, scaled, _ = reg.run_effect_model(panel, c, label, [q], ["a"], "ols")
                rows.append(
                    {
                        "eval_pseudocount": c_eval,
                        "panel": name,
                        "donor": label,
                        "pseudocount": c,
                        "retrieval": direct.retrieval.mean(),
                        "cosine": direct.cosine.mean(),
                        "scaled_mse_ratio": scaled["profile_mse_ratio"],
                    }
                )
        print(f"c_eval={c_eval}: H1 {name} done", flush=True)
    return rows


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    strong = pd.read_csv(reg.STRONG).query("stable_strong")
    rows = []
    for c_eval in EVAL_PSEUDOCOUNTS:
        set_eval(c_eval)
        rows.extend(h1_sweep(c_eval, strong))
        pd.DataFrame(rows).to_csv(REPORT / "h1_sweep.csv", index=False)
        donor.REPORT = REPORT / f"donor_pairs_ceval{c_eval:g}"
        donor.main()
    # Positive control: c_eval = 20 reproduces the original H1 regression summary.
    old = pd.read_csv(reg.ROOT / "reports/h1-transfer-regression/summary.csv")
    old = old[(old.space == "E") & old.model.str.endswith(":scaled")]
    new = pd.DataFrame(rows).query("eval_pseudocount == 20")
    new = new.assign(model=new.donor + ":scaled")
    m = new.merge(old, on=["panel", "model", "pseudocount"], suffixes=("", "_old"))
    assert len(m) == len(new)
    np.testing.assert_allclose(m.cosine, m.cosine_old, rtol=1e-9)
    np.testing.assert_allclose(m.scaled_mse_ratio, m.profile_mse_ratio, rtol=1e-9)
    print("c_eval=20 reproduces h1-transfer-regression", flush=True)


if __name__ == "__main__":
    main()
