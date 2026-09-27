"""Choose the transfer pseudocount without H1: donor-to-donor transfer.

Each source in the frozen five_anchors / all_nine panels is used in turn as the
destination (its control CPM plays h0, its perturbed CPM plays the truth). Predictions
come from each other single source and from the equal-weight average of all other
sources. No H1 profile of any kind is used; targets are the H1-overlap targets only
because those are the frozen panels.
Run: pixi run python -m scripts.evaluation.donor_pair_pseudocount
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.evaluation.h1_transfer_regression import (
    PSEUDOCOUNTS,
    ROOT,
    STUDY,
    Panel,
    direction,
    load_panel,
    realize,
    run_effect_model,
)

REPORT = ROOT / "reports/donor-pair-pseudocount"
NO_STRONG = pd.DataFrame({"target_gene": [], "feature": []})


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    rows = []
    for panel_name in ("five_anchors", "all_nine"):
        sources, common = load_panel(panel_name)
        for dest, arr in sources.items():
            fake = {
                "targets": common["targets"],
                "genes": common["genes"],
                "h1_control": arr["source_control"],
                "h1_truth": arr["source_perturbed"],
            }
            panel = Panel(f"{panel_name}:{dest}", sources, fake, NO_STRONG)
            donors = [s for s in sources if s != dest]
            for c in PSEUDOCOUNTS:
                d = {
                    s: direction(sources[s]["source_perturbed"], sources[s]["source_control"], c)
                    for s in donors
                }
                inputs = dict(d)
                if c != 0:
                    inputs["avg_others"] = sum(d.values()) / len(donors)
                for donor, dd in inputs.items():
                    q = panel.effect(realize(dd, panel.h0, c))
                    direct = panel.score(q)
                    _, scaled, coefs = run_effect_model(panel, c, donor, [q], ["a"], "ols")
                    same_study = donor != "avg_others" and STUDY[donor] == STUDY[dest]
                    rows.append(
                        {
                            "panel": panel_name,
                            "destination": dest,
                            "donor": donor,
                            "same_study": same_study,
                            "pseudocount": c,
                            "targets": panel.n,
                            "retrieval": direct.retrieval.mean(),
                            "cosine": direct.cosine.mean(),
                            "direct_mse_ratio": direct.squared_error.sum()
                            / direct.truth_energy.sum(),
                            "scaled_mse_ratio": scaled["profile_mse_ratio"],
                            "a_mean": np.mean([r["value"] for r in coefs]),
                        }
                    )
            print(f"{panel_name}: destination {dest} done", flush=True)
            pd.DataFrame(rows).to_csv(REPORT / "pairs.csv", index=False)


if __name__ == "__main__":
    main()
