#!/usr/bin/env python3
"""Audit checksums and basic interval structure for accessibility sources."""

import gzip
import json
from pathlib import Path

from download_chromatin_accessibility import MANIFEST, ROOT, verify


def audit_bed(path: Path) -> tuple[int, int, int, int]:
    opener = gzip.open if path.suffix == ".gz" else open
    records, min_columns, max_columns, bad = 0, 1_000, 0, 0
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip().split("\t")
            records += 1
            min_columns = min(min_columns, len(fields))
            max_columns = max(max_columns, len(fields))
            try:
                start, end = int(fields[1]), int(fields[2])
                bad += len(fields) < 3 or start < 0 or end <= start
            except (IndexError, ValueError):
                bad += 1
    return records, min_columns, max_columns, bad


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    data_root = ROOT / manifest["data_root"]
    print("id\tbytes\trecords\tmin_columns\tmax_columns\tbad_intervals\tformat_check")
    for source in manifest["sources"]:
        path = data_root / source["path"]
        verify(path, source)
        if path.suffix == ".bed" or path.name.endswith((".bed.gz", ".narrowPeak.gz")):
            records, min_columns, max_columns, bad = audit_bed(path)
            if records != source["records"] or bad:
                raise ValueError(f"interval audit failed: {path}")
            result = (records, min_columns, max_columns, bad, "BED_OK")
        else:
            with path.open("rb") as handle:
                magic = handle.read(4)
            if magic not in (bytes.fromhex("26fc8f88"), bytes.fromhex("888ffc26")):
                raise ValueError(f"not a BigWig file: {path}")
            result = ("NA", "NA", "NA", "NA", "BIGWIG_MAGIC_OK")
        print(source["id"], path.stat().st_size, *result, sep="\t")


if __name__ == "__main__":
    main()
