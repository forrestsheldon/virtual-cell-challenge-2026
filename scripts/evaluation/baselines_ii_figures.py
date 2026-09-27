"""Baselines II figures: pseudocount sweep and per-screen direct transfer at c = 20.

Reads saved results only (no recomputation):
* reports/h1-transfer-regression/summary.csv          H1 sweep, five-source panel, direct transfer
* reports/donor-pair-pseudocount/pairs.csv            donor-to-donor sweep (no H1 data)
* reports/baselines-ii/pseudocount_c10_c20_c30.csv    the c = 20 points for both
* reports/baselines-ii/noise_and_averaging_c20.csv    per-screen metrics at c = 20
Colors: dataviz reference palette, categorical slots 1-5 in fixed order per source; the
equal-screen average is drawn in ink so it reads as a combination, not a sixth screen.
Run: pixi run python -m scripts.evaluation.baselines_ii_figures
"""

from __future__ import annotations

import shutil

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from scripts.evaluation.baselines_ii_noise import COLORS
from scripts.evaluation.h1_transfer_regression import ROOT

REPORT = ROOT / "reports/baselines-ii"
POST = ROOT.parent / "forrestsheldon.github.io/virtual-cell/posts/draft1-baselines-ii-regression/figures"
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"
ORDER = ["K562_GWPS", "HCT116", "HEK293T", "KOLF2.1J", "CD4T"]
LABEL = {"K562_GWPS": "K562 GWPS", "HCT116": "HCT116", "HEK293T": "HEK293T",
         "KOLF2.1J": "KOLF2.1J", "CD4T": "CD4+ T", "avg": "Average"}
X_ZERO, X_INF = np.log10(0.03), np.log10(3000)  # plotting positions for c = 0 and c = inf


def xpos(c):
    return X_ZERO if c == 0 else X_INF if np.isinf(c) else np.log10(c)


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def sweep_axis(ax):
    ticks = [0, 0.1, 1, 10, 100, np.inf]
    ax.set_xticks([xpos(c) for c in ticks])
    ax.set_xticklabels(["0\nratio", "0.1", "1", "10", "100", "∞\nadditive"])
    ax.set_xlim(X_ZERO - 0.25, X_INF + 0.25)
    for x in (np.mean([X_ZERO, -1]), np.mean([np.log10(300), X_INF])):
        ax.axvline(x, color=GRID, linewidth=6, zorder=0)  # visual break around the limits
    ax.axvline(np.log10(20), color=MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.set_xlabel("Transfer pseudocount c (CPM)", fontsize=9, color=INK)


def draw_curve(ax, series, color, width, label, zorder):
    series = series.sort_index()
    finite = series[(series.index > 0) & np.isfinite(series.index)]
    ax.plot([xpos(c) for c in finite.index], finite.values, color=color, linewidth=width, zorder=zorder)
    for end in (0.0, np.inf):
        if end in series.index and not np.isnan(series[end]):
            edge = finite.index.min() if end == 0 else finite.index.max()
            ax.plot([xpos(end), xpos(edge)], [series[end], finite[edge]], color=color,
                    linewidth=1, linestyle=":", zorder=zorder)
            ax.scatter(xpos(end), series[end], s=18, color=color, zorder=zorder + 1)
    last = finite.index.max()
    ax._end_labels = [*getattr(ax, "_end_labels", []), (finite[last], label, xpos(last))]


def place_end_labels(ax):
    """Direct labels at each curve's right end, nudged apart so none overlap."""
    items = sorted(ax._end_labels)
    low, high = ax.get_ylim()
    gap = 0.045 * (high - low)
    ys = [y for y, _, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for (y0, label, x), y in zip(items, ys, strict=True):
        ax.annotate(label, (x, y0), xytext=(x + 0.08, y), textcoords="data", fontsize=7.5,
                    color=INK, va="center",
                    arrowprops={"arrowstyle": "-", "color": GRID, "lw": 0.8} if abs(y - y0) > gap / 3 else None)


def sweep_figure():
    s = pd.read_csv(ROOT / "reports/h1-transfer-regression/summary.csv")
    s = s[(s.panel == "five_anchors") & (s.space == "E") & s.model.str.endswith(":direct")]
    s = s.assign(model=s.model.str.replace(":direct", "", regex=False))
    c20 = pd.read_csv(REPORT / "pseudocount_c10_c20_c30.csv")
    h1 = pd.concat([s[["model", "pseudocount", "cosine"]],
                    c20[(c20.panel == "h1_five_source") & (c20.pseudocount == 20)][["model", "pseudocount", "cosine"]]])
    pairs = pd.read_csv(ROOT / "reports/donor-pair-pseudocount/pairs.csv")
    pairs = pairs[(pairs.panel == "five_anchors") & (pairs.donor == "avg_others")]
    donor = pd.concat([pairs[["destination", "pseudocount", "cosine"]],
                       c20[(c20.panel == "donor_to_donor") & (c20.pseudocount == 20)][["destination", "pseudocount", "cosine"]]])

    figs = {}
    fig, a = plt.subplots(figsize=(6.2, 4.2))
    for m in ORDER:
        draw_curve(a, donor[donor.destination == m].set_index("pseudocount").cosine, COLORS[m], 1.6, LABEL[m], 3)
    mean = donor[donor.pseudocount > 0].groupby("pseudocount").cosine.mean()
    draw_curve(a, mean, INK, 2.6, "Mean", 4)
    a.set_title("Each screen predicted from the other four (no H1)", fontsize=10, color=INK, loc="left")
    figs["donor_pseudocount_sweep"] = fig
    fig, b = plt.subplots(figsize=(6.2, 4.2))
    for m in ORDER:
        draw_curve(b, h1[h1.model == m].set_index("pseudocount").cosine, COLORS[m], 1.6, LABEL[m], 3)
    draw_curve(b, h1[h1.model == "avg"].set_index("pseudocount").cosine, INK, 2.6, "Average", 4)
    b.set_title("Into H1: each screen and their average", fontsize=10, color=INK, loc="left")
    figs["h1_pseudocount_sweep"] = fig
    for ax in (a, b):
        ax.set_ylabel("Mean cosine with the true response", fontsize=9, color=INK)
        style(ax)
        sweep_axis(ax)
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.08)
        ax.text(np.log10(20) + 0.04, ax.get_ylim()[1], "c = 20", fontsize=7.5, color=MUTED, va="top", ha="left")
        place_end_labels(ax)
        ax.figure.tight_layout()
    return figs


def per_screen_figure():
    t = pd.read_csv(REPORT / "noise_and_averaging_c20.csv").set_index("model")
    rows = [*ORDER, "avg"]
    colors = [COLORS[m] for m in ORDER] + [INK]
    x = np.arange(len(rows))
    wrong = t.loc[ORDER].wrong_target_retrieval.mean()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    panels = [
        ("retrieval", None, None, "Retrieval (dashed: wrong target)", wrong, (0.4, 0.9)),
        ("cosine", "cosine_q025", "cosine_q975", "Mean cosine, 95% interval", 0.0, None),
        ("profile_mse_ratio", "mse_q025", "mse_q975", "MSE ratio, 95% interval (dashed: unchanged)", 1.0, None),
    ]
    for ax, (col, lo, hi, title, ref, ylim) in zip(axes, panels, strict=True):
        vals = t.loc[rows, col].to_numpy()
        if lo:
            for xi, v, l, h, c in zip(x, vals, t.loc[rows, lo], t.loc[rows, hi], colors, strict=True):
                ax.plot([xi, xi], [l, h], color=c, linewidth=2, solid_capstyle="round", zorder=3)
        ax.scatter(x, vals, s=64, color=colors, edgecolor="white", linewidth=1.5, zorder=4)
        ax.axhline(ref, color=MUTED, linewidth=1, linestyle="--", zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[m] for m in rows], rotation=35, ha="right", fontsize=8)
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        style(ax)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def main():
    POST.mkdir(parents=True, exist_ok=True)
    for name, fig in {**sweep_figure(), "per_screen_c20": per_screen_figure()}.items():
        for ext in ("png", "svg"):
            path = REPORT / f"{name}.{ext}"
            fig.savefig(path, dpi=200 if ext == "png" else None)
            shutil.copy(path, POST / path.name)
        plt.close(fig)
    for name in ("cosine_vs_noise_c20.png", "cosine_vs_noise_c20.svg"):
        shutil.copy(REPORT / name, POST / name)
    print(f"figures written to {REPORT} and {POST}")


if __name__ == "__main__":
    main()
