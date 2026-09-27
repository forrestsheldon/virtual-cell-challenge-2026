"""Create the compact tables, figures, and comment-only blog scaffold."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/linear-response-ladder"
OLD_REPORT = ROOT / "reports/linear-response-three-models"
CONTROL_PANEL = ROOT / "reports/crispri-h1-exploration/generated/celleval2_shift_genes.csv"
INITIAL_PANEL = ROOT / "data/derived/linear_response/eval_cache/de_wilcoxon_table-12f179b9e966af64.parquet"
FIGURES = REPORT / "figures"
BLOG = ROOT.parent / "forrestsheldon.github.io/virtual-cell/posts/draft6-baselines I: Linear Response"
BLOG_FIGURES = BLOG / "figures"
MODELS = [
    "empirical",
    "empirical_all31",
    "empirical_raw",
    "sparse",
    "factor_rank10",
    "factor_rank25",
    "factor_rank50",
]
LABELS = {
    "empirical": "Empirical",
    "empirical_all31": "Empirical, all controls",
    "empirical_raw": "Raw-count sensitivity",
    "sparse": "Sparse MAP",
    "factor_rank10": "Factor, k=10",
    "factor_rank25": "Factor, k=25",
    "factor_rank50": "Factor, k=50",
    "factor_selected": "Selected factor",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(figure: plt.Figure, name: str) -> Path:
    path = FIGURES / f"{name}.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    shutil.copy2(path, BLOG_FIGURES / path.name)
    return path


def audit_panel_provenance() -> Path:
    current = sorted(pd.read_csv(CONTROL_PANEL)["gene"].astype(str).unique())
    initial = sorted(
        pd.read_parquet(INITIAL_PANEL, columns=["feature"])["feature"]
        .astype(str)
        .unique()
    )
    path = REPORT / "panel_source_equivalence.csv"
    pd.DataFrame(
        [
            {
                "initial_panel_source": str(INITIAL_PANEL.relative_to(ROOT)),
                "initial_sha256": sha256(INITIAL_PANEL),
                "control_only_panel_source": str(CONTROL_PANEL.relative_to(ROOT)),
                "control_only_sha256": sha256(CONTROL_PANEL),
                "initial_gene_count": len(initial),
                "control_only_gene_count": len(current),
                "ordered_lists_identical": initial == current,
                "symmetric_difference": len(set(initial) ^ set(current)),
                "final_model_code_reads_truth": False,
            }
        ]
    ).to_csv(path, index=False)
    return path


def local_score_table() -> pd.DataFrame:
    current = REPORT / "full_cell_scores.csv"
    if current.exists():
        return pd.read_csv(current)
    arms = {
        "Unchanged controls": "standalone_control_baseline",
        "Multinomial generator null": "normalized_null",
        "Initial empirical response": "model1",
    }
    rows = []
    for label, directory in arms.items():
        scores = pd.read_csv(OLD_REPORT / f"cell_eval/{directory}/scores.csv")
        row = {"arm": label}
        row.update(dict(zip(scores["metric"], scores["from_replicate"], strict=True)))
        rows.append(row)
    columns = [
        "arm",
        "pds_cosine",
        "de_wilcoxon_lfc_nmae",
        "de_wilcoxon_direction_fidelity_yield_raw",
        "de_wilcoxon_direction_reach_raw",
        "de_wilcoxon_sig_jaccard",
        "avg_score",
    ]
    table = pd.DataFrame(rows)[columns]
    null = table.loc[table["arm"] == "Multinomial generator null"].iloc[0]
    model = table.loc[table["arm"] == "Initial empirical response"].iloc[0]
    difference = {"arm": "Initial response minus generator null"}
    difference.update(
        {column: model[column] - null[column] for column in columns[1:]}
    )
    return pd.concat([table, pd.DataFrame([difference])], ignore_index=True)


def make_tables(
    summary: pd.DataFrame,
    scales: pd.DataFrame,
    forcing: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    empirical = summary.loc[summary["model"] == "empirical"].iloc[0]
    naive = pd.DataFrame(
        [
            {
                "arm": "Identity / expected generator null",
                "signed_DE_direction_fidelity": 0.0,
                "strong_DE_LFC_NMAE": 1.0,
            },
            {
                "arm": "Cross-fitted empirical response",
                "signed_DE_direction_fidelity": empirical["mean_signed_recovery"],
                "strong_DE_LFC_NMAE": empirical["mean_exact_crossfit_nmae"],
            },
            {
                "arm": "Response minus null",
                "signed_DE_direction_fidelity": empirical["mean_signed_recovery"],
                "strong_DE_LFC_NMAE": empirical["mean_exact_crossfit_nmae"] - 1,
            },
        ]
    )
    amplitude = (
        scales[scales["eligible"] & scales["model"].isin(MODELS)]
        .groupby("model", as_index=False)
        .agg(
            median_oracle_gamma=("oracle_gamma", "median"),
            oracle_gamma_MAD=(
                "oracle_gamma",
                lambda value: np.median(np.abs(value - np.median(value))),
            ),
            median_LOO_gamma=("loo_median_gamma", "median"),
            zero_oracle_fraction=("oracle_at_zero", "mean"),
        )
    )
    variants = summary[summary["model"].isin(MODELS)].copy()
    variants.insert(1, "label", variants["model"].map(LABELS))
    magnitude = variants[[
        "model",
        "label",
        "median_crossfit_gamma",
        "mean_oracle_linear_nmae",
        "mean_linear_crossfit_nmae",
        "mean_exact_crossfit_nmae",
        "mean_abs_selected_linearization_error",
        "requires_exact_scale_refinement",
    ]].copy()
    forcing_columns = [
        "model",
        "mean_heldout_lfc_cosine",
        "mean_heldout_lfc_spearman",
        "mean_heldout_lfc_nmae",
        "mean_heldout_lfc_explained_energy",
        "mean_signed_recovery",
        "wrong_target_mean",
        "random_operator_mean",
    ]
    requested_forcing = forcing[
        forcing["model"].str.contains(r"anchored_(?:1|2|5|10)$|dense_mode")
    ].copy()
    metric = pd.DataFrame(
        [
            ["Gene universe", "control CPM > 5"],
            ["Truth set", "padj < 0.05; |LFC| >= 0.5; stable sign >= 4/5"],
            ["Per-target K", "number of stable strong-DE genes"],
            ["Primary statistic", "correct-sign truth genes among top-K predictions / K"],
            ["Target gene", "excluded"],
        ],
        columns=["component", "definition"],
    )
    calibration = variants[[
        "label",
        "mean_signed_recovery",
        "mean_unsigned_recovery",
        "mean_sign_given_recovered",
        "mean_strong_energy_fraction",
        "wrong_target_mean",
        "wrong_target_p",
        "random_gene_mean",
        "random_gene_p",
    ]].copy()
    random_controls = requested_forcing[[
        "model",
        "mean_signed_recovery",
        "wrong_target_mean",
        "wrong_target_p_upper",
        "random_operator_mean",
        "random_operator_p_upper",
    ]].copy()
    return {
        "naive_expected_comparison": naive,
        "existing_cell_score_audit": local_score_table(),
        "forcing_ceiling_blog": requested_forcing[forcing_columns],
        "amplitude_summary": amplitude,
        "magnitude_transfer": magnitude,
        "metric_definition": metric,
        "direction_fidelity_calibration": calibration,
        "random_operator_controls": random_controls,
        "variant_summary": variants,
    }


def plot_naive(table: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    colors = ["#999999", "#31688e", "#35b779"]
    axes[0].bar(table["arm"], table["signed_DE_direction_fidelity"], color=colors)
    axes[1].bar(table["arm"], table["strong_DE_LFC_NMAE"], color=colors)
    axes[0].set_ylabel("Signed DE direction fidelity")
    axes[1].set_ylabel("Strong-DE LFC NMAE")
    for axis in axes:
        axis.tick_params(axis="x", rotation=18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return save(figure, "naive_comparison")


def plot_forcing(forcing: pd.DataFrame) -> Path:
    selected = forcing[
        forcing["model"].str.contains(r"anchored_(?:1|2|5|10)$|dense_mode")
    ].copy()
    selected["operator"] = np.where(
        selected["model"].str.startswith("empirical"), "Empirical", "Rank-50"
    )
    selected["support"] = selected["model"].str.extract(r"anchored_(\d+)")[0]
    selected.loc[selected["model"].str.contains("dense"), "support"] = "dense"
    order = ["1", "2", "5", "10", "dense"]
    positions = {support: index for index, support in enumerate(order)}
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for operator, group in selected.groupby("operator"):
        group = group.set_index("support").reindex(order).dropna(subset=["model"])
        x = [positions[support] for support in group.index]
        axes[0].plot(x, group["mean_heldout_lfc_cosine"], marker="o", label=operator)
        axes[1].plot(x, group["mean_signed_recovery"], marker="o", label=operator)
    axes[0].set_ylabel("Held-out LFC cosine")
    axes[1].set_ylabel("Held-out signed recovery")
    for axis in axes:
        axis.set_xlabel("Forcing support")
        axis.set_xticks(
            range(len(order)),
            ["target", "target + 1", "target + 4", "target + 9", "rank-50 span"],
            rotation=20,
        )
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    return save(figure, "forcing_ceiling")


def plot_amplitude(scales: pd.DataFrame, diagnostics: pd.DataFrame) -> Path:
    primary = scales[scales["eligible"] & scales["model"].isin([*MODELS])]
    groups = [
        primary.loc[primary["model"] == model, "oracle_gamma"].dropna()
        for model in MODELS
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].boxplot(
        groups,
        tick_labels=[LABELS[model] for model in MODELS],
        showfliers=False,
    )
    axes[0, 0].set_ylabel("Per-target oracle γ")
    axes[0, 0].tick_params(axis="x", rotation=45)
    empirical = scales[(scales["model"] == "empirical") & scales["eligible"]].copy()
    joined = empirical.merge(diagnostics, on="target_gene")
    native_magnitude = -joined.loc[
        joined["native_cipher_a_oracle"] < 0, "native_cipher_a_oracle"
    ]
    axes[0, 1].hist(
        native_magnitude,
        bins=np.geomspace(native_magnitude.min(), native_magnitude.max(), 24),
        color="#31688e",
        alpha=0.8,
    )
    axes[0, 1].axvline(
        9.72,
        color="#d62728",
        linestyle="--",
        label="old −9.72 reference",
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("Positive-oracle native CIPHER magnitude −a")
    axes[0, 1].set_ylabel("Perturbations")
    axes[0, 1].legend(frameon=False, fontsize=8)
    positive = joined["oracle_gamma"] > 0
    axes[1, 0].scatter(
        joined.loc[positive, "strict_variance"],
        joined.loc[positive, "oracle_gamma"],
        s=16,
        alpha=0.7,
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel(r"$C_{tt}$")
    axes[1, 0].set_ylabel(r"Oracle $\gamma$")
    axes[1, 1].scatter(
        joined["truth_strong_lfc_l1"], joined["oracle_gamma"], s=16, alpha=0.7
    )
    axes[1, 1].set_xlabel("Strong-DE |LFC| sum")
    axes[1, 1].set_ylabel(r"Oracle $\gamma$")
    for axis in axes.ravel():
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return save(figure, "downstream_amplitude")


def plot_magnitude(table: pd.DataFrame) -> Path:
    figure, axis = plt.subplots(figsize=(7.5, 4))
    positions = np.arange(len(table))
    for column, label in [
        ("mean_oracle_linear_nmae", "Per-target oracle, linear"),
        ("mean_linear_crossfit_nmae", "LOO transfer, linear"),
        ("mean_exact_crossfit_nmae", "LOO transfer, exact"),
    ]:
        axis.plot(positions, table[column], "o-", label=label)
    axis.set_xticks(positions, table["label"], rotation=35, ha="right")
    axis.set_ylabel("Strong-DE LFC NMAE")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return save(figure, "crossfit_oracle_magnitude")


def plot_per_target_overview(table: pd.DataFrame) -> Path:
    figure, axis = plt.subplots(figsize=(6.5, 4))
    points = axis.scatter(
        table["truth_strong_lfc_energy"],
        table["signed_recovery_excess_wrong"],
        c=table["exact_crossfit_nmae"],
        cmap="viridis_r",
        s=24,
        alpha=0.8,
    )
    axis.axhline(0, color="#777777", linewidth=1)
    axis.set_xscale("log")
    axis.set_xlabel("Strong-DE LFC energy")
    axis.set_ylabel("Matched signed recovery − wrong-target mean")
    axis.spines[["top", "right"]].set_visible(False)
    figure.colorbar(points, ax=axis, label="Exact cross-fitted LFC NMAE")
    figure.tight_layout()
    return save(figure, "per_target_overview")


def plot_variants(summary: pd.DataFrame) -> Path:
    table = summary.set_index("model").loc[MODELS]
    center = table["mean_signed_recovery"] - table["wrong_target_mean"]
    lower = center - table["excess_wrong_bootstrap_q025"]
    upper = table["excess_wrong_bootstrap_q975"] - center
    figure, axis = plt.subplots(figsize=(8.5, 4))
    axis.errorbar(
        np.arange(len(table)),
        center,
        yerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=3,
        color="#31688e",
    )
    axis.axhline(0, color="#777777", linewidth=1)
    axis.set_xticks(np.arange(len(table)), [LABELS[name] for name in table.index], rotation=35, ha="right")
    axis.set_ylabel("Signed recovery − wrong-target mean")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return save(figure, "variant_direction_fidelity")


def plot_control_selection() -> Path:
    sparse = pd.read_csv(REPORT / "sparse_control_selection.csv")
    factor = pd.read_csv(REPORT / "factor_selection.csv")
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(sparse["tau"], sparse["heldout_error_ratio"], marker="o")
    axes[0].set_xlabel("Sparse threshold τ")
    axes[0].set_ylabel("Held-out control error ratio")
    axes[1].plot(factor["rank"], factor["heldout_covariance_error_ratio"], marker="o")
    axes[1].set_xlabel("Factor rank")
    axes[1].set_ylabel("Held-out covariance error ratio")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return save(figure, "control_only_selection")


def plot_global(global_table: pd.DataFrame) -> Path:
    arms = ["global_only", "model_only", "model_plus_global"]
    table = global_table[
        (global_table["model"] == "empirical")
        & global_table["arm"].isin(arms)
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for position, (axis_name, group) in enumerate(table.groupby("global_axis")):
        group = group.set_index("arm").loc[arms].reset_index()
        x = np.arange(len(group)) + 0.38 * position
        axes[0].bar(x, group["mean_signed_recovery"], width=0.36, label=axis_name)
        axes[1].bar(x, group["mean_strong_de_nmae"], width=0.36, label=axis_name)
    labels = ["Global", "Model", "Model + global"]
    for axis in axes:
        axis.set_xticks(np.arange(3) + 0.19, labels, rotation=20)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Signed DE direction fidelity")
    axes[1].set_ylabel("Strong-DE LFC NMAE")
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return save(figure, "global_axis")


def plot_targets(per_target: pd.DataFrame) -> Path:
    table = per_target[per_target["model"].isin(PRIMARY_MODELS) & per_target["eligible"]]
    matrix = table.pivot(
        index="target_gene", columns="model", values="signed_recovery_excess_wrong"
    )
    strength = table.groupby("target_gene")["truth_strong_lfc_energy"].first()
    matrix = matrix.loc[strength.sort_values().index]
    figure, axis = plt.subplots(figsize=(5.4, 16))
    limit = np.nanmax(np.abs(matrix.to_numpy()))
    image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set_xticks(np.arange(len(matrix.columns)), [LABELS[name] for name in matrix.columns], rotation=30, ha="right")
    axis.set_yticks(np.arange(len(matrix)), matrix.index, fontsize=5)
    axis.set_ylabel("Perturbation, ordered by strong-DE energy")
    figure.colorbar(image, ax=axis, label="Signed recovery − wrong-target mean")
    figure.tight_layout()
    return save(figure, "per_target_fidelity")


PRIMARY_MODELS = ("empirical", "sparse", "factor_selected")


def markdown(table: pd.DataFrame, digits: int = 4) -> str:
    formatted = table.copy()
    for column in formatted.select_dtypes(include="number"):
        if pd.api.types.is_integer_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: "—" if pd.isna(value) else str(value)
            )
        else:
            formatted[column] = formatted[column].map(
                lambda value: "—" if pd.isna(value) else f"{value:.{digits}f}"
            )
    formatted = formatted.fillna("—").astype(str).map(
        lambda value: value.replace("|", "\\|").replace("\n", " ")
    )
    header = "| " + " | ".join(formatted.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in formatted.to_numpy()]
    return "\n".join([header, separator, *rows])


def comment(
    question: str,
    observed: str,
    sources: list[Path],
    caveats: str = (
        "H1-retrospective; cross-fitted scale reads other perturbations; "
        "expected profiles are not official cell-level scores."
    ),
) -> str:
    provenance = "; ".join(
        f"{path.relative_to(ROOT)} [{sha256(path)[:12]}]" for path in sources
    )
    return (
        "<!--\n"
        f"QUESTION: {question}\n"
        f"OBSERVED: {observed}\n"
        "INTERPRETATION: TODO — distinguish covariance direction, transferable magnitude, and shared/global response.\n"
        f"CAVEATS: {caveats}\n"
        f"PROVENANCE: {provenance}\n"
        "SUGGESTED TEXT: TODO\n"
        "-->"
    )


def write_blog(tables: dict[str, pd.DataFrame], figures: dict[str, Path]) -> Path:
    full_scores = REPORT / "full_cell_scores.csv"
    cell_sources = (
        [full_scores, REPORT / "full_cell_manifest.json"]
        if full_scores.exists()
        else [
            OLD_REPORT / "cell_eval/model1/scores.csv",
            OLD_REPORT / "cell_eval/normalized_null/scores.csv",
        ]
    )
    sections = [
        "## 1. Naive control comparison",
        comment(
            "Does the simplest calibrated empirical response improve on no perturbation?",
            tables["naive_expected_comparison"].to_dict("records").__repr__(),
            [REPORT / "direction_fidelity_summary.csv"],
        ),
        markdown(tables["naive_expected_comparison"]),
        "![Controls and empirical linear response](figures/naive_comparison.png)",
        comment(
            "How much of the cell-level result is model-minus-matched-generator?",
            tables["existing_cell_score_audit"].to_dict("records").__repr__(),
            cell_sources,
            "Local public-H1 benchmark, not withheld validation; model and null "
            "use identical source rows with independent decode streams.",
        ),
        markdown(tables["existing_cell_score_audit"]),
        *(
            [
                "![Full H1 score comparison](figures/full_cell_score_comparison.png)",
                "![Full H1 per-perturbation metrics](figures/full_cell_per_target.png)",
            ]
            if full_scores.exists()
            else []
        ),
        "## 2. Per-perturbation results",
        comment(
            "Does target-specific recovery depend on perturbation strength?",
            "Matched-minus-wrong-target recovery is plotted against strong-DE energy.",
            [REPORT / "direction_fidelity_per_target.csv"],
        ),
        markdown(tables["per_target_overview"]),
        "![Per-perturbation empirical results](figures/per_target_overview.png)",
        "## 3. Oracle forcing-vector ceiling",
        comment(
            "Can truth-fitted forcing coordinates extract held-out response information from C?",
            "See forcing-support table and matched/null columns.",
            [OLD_REPORT / "forcing_ceiling_summary.csv"],
        ),
        markdown(tables["forcing_ceiling_blog"]),
        "![Cross-fitted forcing-vector ceiling](figures/forcing_ceiling.png)",
        "## 4. Random-operator controls",
        comment(
            "Do oracle forcing results exceed wrong-target and gene-label-permuted operators?",
            tables["random_operator_controls"].to_dict("records").__repr__(),
            [OLD_REPORT / "forcing_ceiling_summary.csv"],
        ),
        markdown(tables["random_operator_controls"]),
        "## 5. Downstream oracle-scale distribution",
        comment(
            "Is target-knockdown calibration larger than downstream-DE calibration?",
            tables["amplitude_summary"].to_dict("records").__repr__(),
            [REPORT / "downstream_amplitude_distribution.csv"],
        ),
        markdown(tables["amplitude_summary"]),
        "![Downstream amplitude distributions](figures/downstream_amplitude.png)",
        "## 6. Cross-fitted versus oracle magnitude",
        comment(
            "What is lost by transferring one scale from the other H1 perturbations?",
            tables["magnitude_transfer"].to_dict("records").__repr__(),
            [REPORT / "direction_fidelity_summary.csv"],
        ),
        markdown(tables["magnitude_transfer"]),
        "![Cross-fitted and oracle magnitude](figures/crossfit_oracle_magnitude.png)",
        "## 7. DE direction-fidelity calibration",
        comment(
            "Which scale-free statistic isolates strong downstream DE direction?",
            "Metric contract frozen before variant comparison.",
            [OLD_REPORT / "strong_de_recovery_manifest.json"],
        ),
        markdown(tables["metric_definition"]),
        comment(
            "How do signed recovery, unsigned recovery, sign, energy, and nulls relate?",
            tables["direction_fidelity_calibration"].to_dict("records").__repr__(),
            [REPORT / "direction_fidelity_summary.csv"],
        ),
        markdown(tables["direction_fidelity_calibration"]),
        "## 8. Empirical, sparse, and factor comparisons",
        comment(
            "Do control-pool, raw-count, sparse, or factor variants add target-specific information?",
            "See matched-minus-wrong-target intervals and control-only selection curves.",
            [REPORT / "direction_fidelity_summary.csv", REPORT / "sparse_control_selection.csv", REPORT / "factor_selection.csv"],
        ),
        markdown(
            tables["variant_summary"][[
                "label",
                "mean_signed_recovery",
                "wrong_target_mean",
                "random_gene_mean",
                "mean_exact_crossfit_nmae",
                "passes_target_signal_gate",
            ]]
        ),
        "![Variant direction fidelity](figures/variant_direction_fidelity.png)",
        "![Control-only model selection](figures/control_only_selection.png)",
        "## 9. Global-response decomposition",
        comment(
            "Does a shared perturbation axis explain performance separately from target-specific covariance?",
            "Equal-target leave-one-out and cell-weighted axes shown separately.",
            [REPORT / "global_axis_comparison.csv"],
        ),
        markdown(pd.read_csv(REPORT / "global_axis_comparison.csv")),
        "![Global and target-specific response](figures/global_axis.png)",
        "## 10. Per-target successes and failures",
        comment(
            "For which perturbations does matched covariance beat another target's covariance?",
            tables["successes_failures"].to_dict("records").__repr__(),
            [REPORT / "direction_fidelity_per_target.csv"],
        ),
        markdown(tables["successes_failures"]),
        "![Per-perturbation target-specific fidelity](figures/per_target_fidelity.png)",
    ]
    path = BLOG / "ladder-results.qmd"
    path.write_text("\n\n".join(sections) + "\n")
    return path


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    BLOG_FIGURES.mkdir(parents=True, exist_ok=True)
    panel_audit = audit_panel_provenance()
    summary = pd.read_csv(REPORT / "direction_fidelity_summary.csv")
    per_target = pd.read_csv(REPORT / "direction_fidelity_per_target.csv")
    scales = pd.read_csv(REPORT / "downstream_amplitude_distribution.csv")
    global_table = pd.read_csv(REPORT / "global_axis_comparison.csv")
    forcing = pd.read_csv(OLD_REPORT / "forcing_ceiling_summary.csv")
    diagnostics = pd.read_csv(REPORT / "empirical_fit_diagnostics.csv")
    tables = make_tables(summary, scales, forcing)
    tables["per_target_overview"] = per_target.loc[
        (per_target["model"] == "empirical") & per_target["eligible"],
        [
            "target_gene",
            "n_strong",
            "truth_strong_lfc_energy",
            "signed_recovery",
            "wrong_target_mean",
            "signed_recovery_excess_wrong",
            "exact_crossfit_nmae",
        ],
    ].sort_values("truth_strong_lfc_energy")
    extremes = []
    for model in PRIMARY_MODELS:
        eligible = per_target[
            (per_target["model"] == model) & per_target["eligible"]
        ].sort_values("signed_recovery_excess_wrong")
        for outcome, rows in [("failure", eligible.head(5)), ("success", eligible.tail(5))]:
            selected = rows[[
                "target_gene",
                "signed_recovery",
                "wrong_target_mean",
                "signed_recovery_excess_wrong",
                "truth_strong_lfc_energy",
                "exact_crossfit_nmae",
            ]].copy()
            selected.insert(0, "outcome", outcome)
            selected.insert(0, "model", LABELS[model])
            extremes.append(selected)
    tables["successes_failures"] = pd.concat(extremes, ignore_index=True)
    for name, table in tables.items():
        table.to_csv(REPORT / f"{name}.csv", index=False)
    figures = {
        "naive": plot_naive(tables["naive_expected_comparison"]),
        "overview": plot_per_target_overview(tables["per_target_overview"]),
        "forcing": plot_forcing(forcing),
        "amplitude": plot_amplitude(scales, diagnostics),
        "magnitude": plot_magnitude(tables["magnitude_transfer"]),
        "variants": plot_variants(summary),
        "selection": plot_control_selection(),
        "global": plot_global(global_table),
        "targets": plot_targets(per_target),
    }
    for name in ["full_cell_score_comparison", "full_cell_per_target"]:
        path = FIGURES / f"{name}.png"
        if path.exists():
            figures[name] = path
    blog = write_blog(tables, figures)
    manifest_inputs = [
        REPORT / "direction_fidelity_summary.csv",
        REPORT / "direction_fidelity_per_target.csv",
        REPORT / "downstream_amplitude_distribution.csv",
        REPORT / "global_axis_comparison.csv",
        OLD_REPORT / "forcing_ceiling_summary.csv",
    ]
    if (REPORT / "full_cell_scores.csv").exists():
        manifest_inputs.extend(
            [REPORT / "full_cell_scores.csv", REPORT / "full_cell_manifest.json"]
        )
    manifest = {
        "kind": "linear-response ladder blog tables and figures",
        "blog": str(blog),
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in manifest_inputs
        },
        "figures": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in figures.items()
        },
        "tables": {
            name: {
                "path": str((REPORT / f"{name}.csv").relative_to(ROOT)),
                "sha256": sha256(REPORT / f"{name}.csv"),
            }
            for name in tables
        },
        "panel_provenance_audit": {
            "path": str(panel_audit.relative_to(ROOT)),
            "sha256": sha256(panel_audit),
        },
        "software": {
            "python": platform.python_version(),
            **{name: version(name) for name in ["matplotlib", "numpy", "pandas"]},
        },
    }
    (REPORT / "blog_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
