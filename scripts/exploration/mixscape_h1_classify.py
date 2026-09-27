"""Pool H1 signature shards and classify responders with Mixscape."""

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pertpy as pt
from scipy import sparse


def subset(shard, rows):
    rows = np.asarray(rows)
    if rows.dtype == bool:
        rows = np.flatnonzero(rows)
    return shard[rows].to_memory()


def evidence_tier(n):
    if n < 10:
        return "insufficient"
    if n <= 30:
        return "exploratory"
    if n <= 100:
        return "usable"
    return "standard"


def signature_norm(matrix):
    if sparse.issparse(matrix):
        return np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    return np.linalg.norm(matrix, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--signatures",
        type=Path,
        default=Path("data/derived/vcc2025_h1_mixscape/signatures"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/crispri-h1-exploration/generated")
    )
    parser.add_argument("--control-cells", type=int, default=5_000)
    parser.add_argument("--min-cells", type=int, default=30)
    parser.add_argument("--test-method", default="wilcoxon")
    parser.add_argument("--guide")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.signatures.glob("*.h5ad"))
    if len(paths) != 48 and not args.allow_partial:
        raise ValueError(f"Expected 48 signature shards, found {len(paths)}")
    shards = [ad.read_h5ad(path, backed="r") for path in paths]
    obs = pd.concat(
        [shard.obs.assign(cell_id=shard.obs_names) for shard in shards],
        ignore_index=True,
    )
    counts = (
        obs.loc[obs.target_gene != "non-targeting"]
        .groupby(["mixscape_perturbation", "target_gene"], observed=True)
        .size()
    )
    guides = [args.guide] if args.guide else counts.index.get_level_values(0).tolist()

    rng = np.random.default_rng(0)
    per_batch = int(np.ceil(args.control_cells / len(shards)))
    control_parts = []
    for shard in shards:
        rows = np.flatnonzero(shard.obs.mixscape_perturbation.to_numpy() == "NT")
        rows = rng.choice(rows, min(per_batch, len(rows)), replace=False)
        control_parts.append(subset(shard, rows))
    controls = ad.concat(control_parts, join="inner")[: args.control_cells].copy()

    calls = []
    summaries = []
    exclusions = []
    mixscape = pt.tl.Mixscape()
    started = time.monotonic()
    for number, guide in enumerate(guides, 1):
        target_gene = counts.loc[guide].index[0]
        n_cells = int(counts.loc[(guide, target_gene)])
        if n_cells < args.min_cells:
            exclusions.append(
                {
                    "guide_id": guide,
                    "target_gene": target_gene,
                    "n_cells": n_cells,
                    "reason": f"fewer than {args.min_cells} cells",
                }
            )
            print(f"[{number}/{len(guides)}] {guide}: excluded", flush=True)
            continue

        parts = [
            subset(shard, shard.obs.mixscape_perturbation.to_numpy() == guide)
            for shard in shards
            if (shard.obs.mixscape_perturbation == guide).any()
        ]
        target = ad.concat(parts, join="inner")
        work = ad.concat([target, controls], join="inner")
        try:
            mixscape.mixscape(
                work,
                pert_key="mixscape_perturbation",
                control="NT",
                layer="X_pert",
                test_method=args.test_method,
                split_by=None,
                perturbation_type="KO",
                random_state=0,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            exclusions.append(
                {
                    "guide_id": guide,
                    "target_gene": target_gene,
                    "n_cells": n_cells,
                    "reason": f"post-hoc failure: {type(error).__name__}: {error}",
                }
            )
            print(f"[{number}/{len(guides)}] {guide}: failed", flush=True)
            continue

        result = work.obs.iloc[: len(target)].copy()
        result["cell_id"] = result.index
        calls.append(
            result[
                [
                    "cell_id",
                    "mixscape_class",
                    "mixscape_class_global",
                    "mixscape_class_p_ko",
                ]
            ]
        )
        responder = result.mixscape_class_global == "KO"
        norms = signature_norm(work.layers["X_pert"][: len(target)])
        summaries.append(
            {
                "guide_id": guide,
                "target_gene": target_gene,
                "n_cells": n_cells,
                "n_responder": int(responder.sum()),
                "responder_fraction": responder.mean(),
                "median_posterior": result.loc[
                    responder, "mixscape_class_p_ko"
                ].median(),
                "effect_magnitude": norms[responder].mean()
                if responder.any()
                else np.nan,
                "evidence_tier": evidence_tier(int(responder.sum())),
            }
        )
        elapsed = time.monotonic() - started
        eta = elapsed / number * (len(guides) - number) / 60
        print(f"[{number}/{len(guides)}] {guide}: ETA {eta:.1f} min", flush=True)

    cells = obs[
        ["cell_id", "target_gene", "guide_id", "batch", "mixscape_perturbation"]
    ].copy()
    cells["mixscape_class"] = np.select(
        [cells.mixscape_perturbation == "NT", cells.target_gene == "non-targeting"],
        ["NT", "excluded_control"],
        default="excluded",
    )
    cells["mixscape_class_global"] = cells.mixscape_class
    cells["mixscape_class_p_ko"] = 0.0
    if calls:
        classified = pd.concat(calls).set_index("cell_id")
        cells = cells.set_index("cell_id")
        cells.update(classified)
        cells = cells.reset_index()

    pd.DataFrame(summaries).to_csv(args.output / "mixscape_guides.csv", index=False)
    pd.DataFrame(
        exclusions, columns=["guide_id", "target_gene", "n_cells", "reason"]
    ).to_csv(args.output / "mixscape_excluded.csv", index=False)
    cells.to_parquet(args.output / "mixscape_cells.parquet", index=False)
    (args.output / "mixscape_run.json").write_text(
        json.dumps(
            {
                "signature_shards": len(paths),
                "control_cells": len(controls),
                "control_composition": controls.obs.batch.value_counts().to_dict(),
                "excluded_control_guides": sorted(
                    obs.loc[
                        obs.mixscape_perturbation == "excluded_control", "guide_id"
                    ].astype(str).unique()
                ),
                "excluded_control_cells": int(
                    (obs.mixscape_perturbation == "excluded_control").sum()
                ),
                "min_cells": args.min_cells,
                "test_method": args.test_method,
                "classification_split": None,
                "seed": 0,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
