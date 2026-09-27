#!/usr/bin/env python3
"""Download and checksum the frozen chromatin-accessibility source collection."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "metadata/datasets/chromatin_accessibility_sources.json"


def hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def verify(path: Path, source: dict) -> None:
    if path.stat().st_size != source["expected_bytes"]:
        raise ValueError(f"size mismatch: {path}")
    md5, sha256 = hashes(path)
    if md5 != source["md5"] or sha256 != source["sha256"]:
        raise ValueError(f"checksum mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", action="append", help="download/verify one manifest id")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    data_root = ROOT / manifest["data_root"]
    wanted = set(args.only or [])
    sources = [s for s in manifest["sources"] if not wanted or s["id"] in wanted]
    if wanted - {s["id"] for s in sources}:
        raise ValueError(f"unknown ids: {sorted(wanted - {s['id'] for s in sources})}")

    for source in sources:
        path = data_root / source["path"]
        if not path.exists():
            if args.verify_only:
                raise FileNotFoundError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_name(path.name + ".part")
            subprocess.run(
                ["curl", "-fL", "--retry", "4", "--continue-at", "-", "-o", part, source["url"]],
                check=True,
            )
            part.replace(path)
        verify(path, source)
        print(f"verified\t{source['id']}\t{path.stat().st_size}\t{path}")


if __name__ == "__main__":
    main()
