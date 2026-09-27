"""Copy the released, verified H1 reference cache for linear-response scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/derived/vcc2026_h1/reference_cache"
DESTINATION = ROOT / "data/derived/linear_response/eval_cache"
BENCHMARK = ROOT / "reports/vcc2026-h1/benchmark_manifest.json"
SCALE = ROOT / "reports/vcc2026-h1/scale/manifest.json"
OUTPUT = ROOT / "reports/linear-response-three-models/eval_cache_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if args.source.resolve() == args.destination.resolve():
        raise ValueError("source and destination must differ")
    lock = args.source / "manifest.json.lock"
    if lock.exists() and lock.stat().st_size:
        raise RuntimeError(f"nonempty source lock: {lock}")

    benchmark = json.loads(BENCHMARK.read_text())
    expected = benchmark["artifacts"]["reference_cache_files"]
    for name, digest in expected.items():
        path = args.source / name
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"benchmark cache mismatch: {name}")

    cache_manifest = json.loads((args.source / "manifest.json").read_text())
    filenames = {
        artifact["filename"] for artifact in cache_manifest["artifacts"].values()
    }
    missing = filenames - {
        path.name for path in args.source.iterdir() if path.is_file()
    }
    if missing:
        raise RuntimeError(f"cache manifest files missing: {sorted(missing)}")

    args.destination.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in args.destination.iterdir() if path.is_file()}
    allowed = filenames | {"manifest.json", "manifest.json.lock"}
    if existing - allowed:
        raise RuntimeError(
            f"unexpected destination files: {sorted(existing - allowed)}"
        )
    for name in sorted(filenames | {"manifest.json"}):
        source = args.source / name
        destination = args.destination / name
        if destination.exists() and sha256(destination) != sha256(source):
            raise RuntimeError(f"refusing to overwrite different cache file: {name}")
        if not destination.exists():
            shutil.copy2(source, destination)

    copied = {
        name: sha256(args.destination / name)
        for name in sorted(filenames | {"manifest.json"})
    }
    source_hashes = {
        name: sha256(args.source / name)
        for name in sorted(filenames | {"manifest.json"})
    }
    if copied != source_hashes:
        raise RuntimeError("copied cache hashes do not match source")

    payload = {
        "created_utc": datetime.now(UTC).isoformat(),
        "purpose": "isolated immutable snapshot for sequential linear-response H1 scoring",
        "source": args.source.resolve().relative_to(ROOT).as_posix(),
        "destination": args.destination.resolve().relative_to(ROOT).as_posix(),
        "benchmark_manifest_sha256": sha256(BENCHMARK),
        "scale_manifest_sha256": sha256(SCALE),
        "files": copied,
        "verification": {
            "canonical_benchmark_files_match": True,
            "cache_manifest_complete": True,
            "copy_hashes_match": True,
            "source_lock_holder_checked_externally": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Seeded {len(copied)} cache files at {args.destination}")


if __name__ == "__main__":
    main()
