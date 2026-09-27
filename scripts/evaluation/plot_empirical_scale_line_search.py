"""Plot the all-target oracle line search for vanilla linear response."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/linear-response-ladder/empirical-scale-crossfit"
CHECKPOINT = REPORT / "exact_native_grid_checkpoint.npz"
MIN_DE = 10


with np.load(CHECKPOINT) as saved:
    a = -saved["grid"]
    eligible = saved["all_count"] >= MIN_DE
    nmae = np.nanmean(saved["all_nmae"][eligible], axis=0)

curve = pd.DataFrame(
    {"a": a, "mean_de_lfc_nmae": nmae, "eligible_perturbations": eligible.sum()}
)
curve.to_csv(REPORT / "all_target_line_search.csv", index=False)

optimum = int(np.argmin(nmae))
shown = a >= -20
figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
axis.plot(a[shown], nmae[shown], color="#3572A5", linewidth=2.2)
axis.axhline(1, color="0.55", linestyle="--", linewidth=1.2)
axis.scatter(a[optimum], nmae[optimum], color="#D97706", s=48, zorder=3)
axis.scatter(0, 1, facecolor="white", edgecolor="#3572A5", linewidth=1.8, s=48, zorder=3)
axis.annotate(
    f"minimum: $a={a[optimum]:.2f}$\nNMAE = {nmae[optimum]:.5f}",
    (a[optimum], nmae[optimum]),
    xytext=(-8.4, 1.0068),
    arrowprops={"arrowstyle": "-", "color": "0.35"},
    ha="center",
)
axis.annotate(
    "$a=0$\nNMAE = 1",
    (0, 1),
    xytext=(-2.0, 1.0037),
    arrowprops={"arrowstyle": "-", "color": "0.35"},
    ha="center",
)
axis.set(
    xlim=(-20, 0.6),
    ylim=(0.9985, 1.053),
    xlabel=r"global linear-response scale $a$",
    ylabel="mean DE-LFC NMAE",
)
axis.spines[["top", "right"]].set_visible(False)
axis.text(
    0.02,
    0.97,
    f"{eligible.sum()} perturbations; at least {MIN_DE} downstream DE genes",
    transform=axis.transAxes,
    ha="left",
    va="top",
    color="0.35",
    fontsize=9,
)
figure.savefig(REPORT / "all_target_line_search.png", dpi=220)
plt.close(figure)
