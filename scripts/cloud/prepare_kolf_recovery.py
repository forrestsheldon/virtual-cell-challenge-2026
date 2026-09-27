"""Build and independently validate the KOLF H1/VCC recovery panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

CONTROL = "NTC"
EXPECTED = {
    "h1_requested": 126,
    "vcc_requested": 300,
    "panel_overlap": 11,
    "union_requested": 415,
    "h1_available": 123,
    "vcc_available": 282,
    "union_available": 394,
}
EXPECTED_MISSING_H1 = {"CAST", "CHMP3", "TAZ"}
EXPECTED_MISSING_UNION = {
    "CAST",
    "CHKB",
    "CHMP3",
    "DTNBP1",
    "EPHB2",
    "IFNGR2",
    "IL10RB",
    "LENG1",
    "MTM1",
    "PBLD",
    "PSMB9",
    "RSRC1",
    "SNRK",
    "STK3",
    "TAZ",
    "TBC1D19",
    "TBCK",
    "TESK2",
    "TRIP4",
    "TTBK2",
    "ZC2HC1A",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_targets(path: Path, column: str = "target_gene") -> set[str]:
    return set(pd.read_csv(path)[column].astype(str))


def coverage(panel: pd.DataFrame, available: set[str]) -> dict[str, object]:
    h1 = set(panel.loc[panel.in_h1, "target_gene"])
    vcc = set(panel.loc[panel.in_vcc2026, "target_gene"])
    union = set(panel.target_gene)
    return {
        "h1_requested": len(h1),
        "vcc_requested": len(vcc),
        "panel_overlap": len(h1 & vcc),
        "union_requested": len(union),
        "h1_available": len(h1 & available),
        "vcc_available": len(vcc & available),
        "union_available": len(union & available),
        "missing_h1": sorted(h1 - available),
        "missing_union": sorted(union - available),
    }


def assert_expected(summary: dict[str, object]) -> None:
    for key, expected in EXPECTED.items():
        if summary[key] != expected:
            raise ValueError(f"KOLF recovery {key}: {summary[key]} != {expected}")
    if set(summary["missing_h1"]) != EXPECTED_MISSING_H1:
        raise ValueError(f"unexpected missing H1 targets: {summary['missing_h1']}")
    if set(summary["missing_union"]) != EXPECTED_MISSING_UNION:
        raise ValueError(f"unexpected missing union targets: {summary['missing_union']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h1", type=Path, required=True)
    parser.add_argument("--vcc", type=Path, required=True)
    parser.add_argument("--source-counts", type=Path, required=True)
    parser.add_argument("--panel-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()

    h1 = read_targets(args.h1)
    vcc = read_targets(args.vcc)
    available = read_targets(args.source_counts, "source_target_gene") - {CONTROL}
    panel = pd.DataFrame({"target_gene": sorted(h1 | vcc)})
    panel["in_h1"] = panel.target_gene.isin(h1)
    panel["in_vcc2026"] = panel.target_gene.isin(vcc)
    summary = coverage(panel, available)
    assert_expected(summary)

    args.panel_output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.panel_output, index=False)
    summary["panel_sha256"] = file_sha256(args.panel_output)
    summary["panel_path"] = str(args.panel_output)
    summary["source_counts_path"] = str(args.source_counts)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
