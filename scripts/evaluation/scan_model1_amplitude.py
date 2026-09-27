"""Sequentially score a shared Model 1 amplitude on the local H1 benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scripts.linear_response.forcing_ceiling import load_truth
from scripts.linear_response.kernel import expected_decoded_sum, log_pseudobulk
from scripts.linear_response.strong_de_recovery import signed_topk

ROOT = Path(__file__).resolve().parents[2]
H1 = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
MODEL = ROOT / "data/derived/linear_response/model1_expected_profiles.npz"
DEFAULT_OUTPUT = ROOT / "reports/linear-response-three-models/model1_amplitude_scan"
DEFAULT_PREDICTION = (
    ROOT / "data/derived/linear_response/model1_amplitude_scan.h5ad"
)
GRID = (0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50)
MEDIAN_AMPLITUDE = -9.72
N_SCORED_METRICS = 6
MIN_STRONG_GENES = 10


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def point_name(multiplier: float) -> str:
    return f"multiplier_{multiplier:.2f}".replace(".", "p")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def completed_score(point: Path) -> bool:
    paths = [
        point / "aggregates.csv",
        point / "per_target.csv",
        point / "scores.csv",
        point / "manifest.json",
    ]
    if not all(path.is_file() for path in paths):
        return False
    scores = pd.read_csv(point / "scores.csv")
    return (
        (scores["metric"] == "avg_score").sum() == 1
        and (scores["metric"] != "avg_score").sum() == N_SCORED_METRICS
    )


def expected_strong_de_scan(output: Path) -> pd.DataFrame:
    """Compute the predeclared |truth LFC| >= 0.5 score without sampled cells."""
    with np.load(MODEL, allow_pickle=False) as model:
        targets = model["target_gene"].astype(str).tolist()
        genes = model["gene_names"].astype(str).tolist()
        source_order = model["source_order"].astype(np.int64)
        source_rows = model["source_rows"].astype(np.int64)
        directions = model["directions"].astype(np.float64)
    _, de_indices, truth_lfc, stable = load_truth(targets, genes)
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    de_lookup = {gene_index: index for index, gene_index in enumerate(de_indices)}
    excluded = [de_lookup.get(gene_lookup.get(target, -1)) for target in targets]

    effects = {multiplier: [] for multiplier in GRID}
    data = ad.read_h5ad(H1, backed="r")
    try:
        for index, column in enumerate(source_order):
            raw = data.X[np.sort(source_rows[index])].tocsr()
            null = log_pseudobulk(np.asarray(raw.sum(axis=0)).ravel())
            for multiplier in GRID:
                amplitude = MEDIAN_AMPLITUDE * multiplier
                shifted = log_pseudobulk(
                    expected_decoded_sum(raw, directions[:, column], amplitude)
                )
                effects[multiplier].append((shifted - null)[de_indices])
            print(
                f"expected strong-DE profiles {index + 1}/{len(targets)}: "
                f"{targets[index]}",
                flush=True,
            )
    finally:
        data.file.close()

    rows = []
    signs = np.sign(truth_lfc).astype(np.int8)
    for multiplier in GRID:
        predictions = np.asarray(effects[multiplier])
        target_scores = []
        for index in range(len(targets)):
            result = signed_topk(
                predictions[index], stable[index], signs[index], excluded[index]
            )
            if int(result["n_strong"]) >= MIN_STRONG_GENES:
                target_scores.append(float(result["signed_recovery"]))
        rows.append(
            {
                "multiplier": multiplier,
                "amplitude": MEDIAN_AMPLITUDE * multiplier,
                "eligible_targets": len(target_scores),
                "expected_strong_de_signed_topk": float(np.mean(target_scores)),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output / "expected_strong_de_fidelity.csv", index=False)
    return table


def consolidate(output: Path, strong: pd.DataFrame) -> None:
    rows = []
    summaries = []
    strong_lookup = strong.set_index("multiplier")
    for multiplier in GRID:
        point = output / point_name(multiplier)
        if not completed_score(point):
            continue
        scores = pd.read_csv(point / "scores.csv")
        aggregates = pd.read_csv(point / "aggregates.csv").set_index("metric")
        average = float(
            scores.loc[scores["metric"] == "avg_score", "from_replicate"].iloc[0]
        )
        metric_scores = scores.loc[scores["metric"] != "avg_score"].copy()
        for row in metric_scores.itertuples(index=False):
            rows.append(
                {
                    "multiplier": multiplier,
                    "amplitude": MEDIAN_AMPLITUDE * multiplier,
                    "metric": row.metric,
                    "raw_value": float(aggregates.loc[row.metric, "raw_value"]),
                    "scaled_score": float(row.from_replicate),
                    "scaled_contribution": float(row.from_replicate)
                    / N_SCORED_METRICS,
                    "avg_score": average,
                }
            )
        contribution_sum = sum(
            row["scaled_contribution"]
            for row in rows
            if row["multiplier"] == multiplier
        )
        if not np.isclose(contribution_sum, average, atol=1e-12):
            raise AssertionError("scaled metric contributions do not sum to avg_score")
        summaries.append(
            {
                "multiplier": multiplier,
                "amplitude": MEDIAN_AMPLITUDE * multiplier,
                "avg_score": average,
                "expected_strong_de_signed_topk": float(
                    strong_lookup.loc[multiplier, "expected_strong_de_signed_topk"]
                ),
                "eligible_strong_de_targets": int(
                    strong_lookup.loc[multiplier, "eligible_targets"]
                ),
            }
        )
    pd.DataFrame(rows).to_csv(output / "scaled_metric_contributions.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "scan_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gene-chunk", type=int, default=512)
    parser.add_argument("--de-threads", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    status_path = args.output / "status.json"
    strong_path = args.output / "expected_strong_de_fidelity.csv"
    if strong_path.is_file():
        strong = pd.read_csv(strong_path)
    else:
        write_json(
            status_path,
            {"updated_utc": utc_now(), "stage": "expected_strong_de_fidelity"},
        )
        strong = expected_strong_de_scan(args.output)

    command_provenance = {
        "scorer": str(args.scorer.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "gene_chunk": args.gene_chunk,
        "de_threads": args.de_threads,
    }
    for multiplier in GRID:
        amplitude = MEDIAN_AMPLITUDE * multiplier
        point = args.output / point_name(multiplier)
        point.mkdir(parents=True, exist_ok=True)
        generation_manifest = point / "generation_manifest.json"
        if completed_score(point):
            args.prediction.unlink(missing_ok=True)
            consolidate(args.output, strong)
            continue

        reusable_prediction = False
        if args.prediction.is_file() and generation_manifest.is_file():
            generation = json.loads(generation_manifest.read_text())
            reusable_prediction = np.isclose(
                generation.get("amplitude", {}).get("value", np.nan), amplitude
            )
        if not reusable_prediction:
            args.prediction.unlink(missing_ok=True)
            write_json(
                status_path,
                {
                    "updated_utc": utc_now(),
                    "stage": "generating_prediction",
                    "multiplier": multiplier,
                    "amplitude": amplitude,
                    **command_provenance,
                },
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.linear_response.generate_h1_cells",
                    "model1",
                    str(args.prediction),
                    "--manifest",
                    str(generation_manifest),
                    "--shared-amplitude",
                    str(amplitude),
                ],
                cwd=ROOT,
                check=True,
            )

        write_json(
            status_path,
            {
                "updated_utc": utc_now(),
                "stage": "scoring_prediction",
                "multiplier": multiplier,
                "amplitude": amplitude,
                **command_provenance,
            },
        )
        subprocess.run(
            [
                str(args.scorer),
                "score",
                str(args.prediction),
                "--output",
                str(point),
                "--gene-chunk",
                str(args.gene_chunk),
                "--de-threads",
                str(args.de_threads),
                "--data-dir",
                str(args.data_dir),
            ],
            cwd=args.scorer.resolve().parents[1],
            check=True,
        )
        if not completed_score(point):
            raise RuntimeError(f"incomplete scorer outputs at {point}")
        score_manifest = json.loads((point / "manifest.json").read_text())
        generation = json.loads(generation_manifest.read_text())
        if score_manifest["prediction"]["sha256"] != generation["prediction"]["sha256"]:
            raise AssertionError("scored prediction hash differs from generated prediction")
        args.prediction.unlink()
        consolidate(args.output, strong)
        write_json(
            status_path,
            {
                "updated_utc": utc_now(),
                "stage": "point_complete",
                "multiplier": multiplier,
                "amplitude": amplitude,
                **command_provenance,
            },
        )

    write_json(
        status_path,
        {
            "updated_utc": utc_now(),
            "stage": "complete",
            "points": len(GRID),
            **command_provenance,
            "software": {
                "python": platform.python_version(),
                **{
                    package: version(package)
                    for package in ["anndata", "numpy", "pandas"]
                },
            },
        },
    )
    print(f"Completed {len(GRID)} shared-amplitude scores in {args.output}")


if __name__ == "__main__":
    main()
