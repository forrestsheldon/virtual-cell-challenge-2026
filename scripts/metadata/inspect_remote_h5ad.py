#!/usr/bin/env python3
"""Inspect HDF5 metadata over HTTP byte ranges without downloading the matrix."""

from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.request
from collections import OrderedDict

import h5py
import numpy as np


class HTTPRangeReader(io.RawIOBase):
    """Seekable, cached HTTP reader with a hard network-transfer budget."""

    def __init__(self, url: str, block_size: int, max_bytes: int) -> None:
        self.url = url
        self.block_size = block_size
        self.max_bytes = max_bytes
        self.position = 0
        self.transferred = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request) as response:
            self.size = int(response.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def _block(self, index: int) -> bytes:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        start = index * self.block_size
        end = min(start + self.block_size, self.size) - 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{end}"}
        )
        with urllib.request.urlopen(request) as response:
            payload = response.read()
        self.transferred += len(payload)
        if self.transferred > self.max_bytes:
            raise RuntimeError(
                f"Remote-read budget exceeded: {self.transferred} > {self.max_bytes}"
            )
        self.cache[index] = payload
        while len(self.cache) > 32:
            self.cache.popitem(last=False)
        return payload

    def readinto(self, buffer: bytearray) -> int:
        if self.position >= self.size:
            return 0
        wanted = min(len(buffer), self.size - self.position)
        written = 0
        while written < wanted:
            block_index = self.position // self.block_size
            block_offset = self.position % self.block_size
            block = self._block(block_index)
            count = min(wanted - written, len(block) - block_offset)
            buffer[written : written + count] = block[
                block_offset : block_offset + count
            ]
            self.position += count
            written += count
        return written


def decode(values: object) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values  # type: ignore[union-attr]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--categories", help="HDF5 category dataset path")
    parser.add_argument(
        "--codes",
        help="Optional categorical codes path; writes category,n_cells CSV",
    )
    parser.add_argument("--output", help="Write categories or category counts")
    parser.add_argument("--block-mib", type=int, default=1)
    parser.add_argument("--max-mib", type=int, default=95)
    args = parser.parse_args()

    reader = HTTPRangeReader(args.url, args.block_mib * 1024**2, args.max_mib * 1024**2)
    with h5py.File(reader, "r") as handle:
        if args.categories:
            values = decode(handle[args.categories][...])
            if args.output:
                if args.codes:
                    codes = np.asarray(handle[args.codes][...], dtype=np.int64)
                    counts = np.bincount(codes[codes >= 0], minlength=len(values))
                    with open(args.output, "w", newline="", encoding="utf-8") as stream:
                        writer = csv.writer(stream, lineterminator="\n")
                        writer.writerow(["source_target_gene", "n_cells"])
                        writer.writerows(zip(values, counts, strict=True))
                else:
                    with open(args.output, "w", encoding="utf-8") as stream:
                        stream.write("\n".join(values) + "\n")
            result = {
                "url": args.url,
                "path": args.categories,
                "codes_path": args.codes,
                "n_categories": len(values),
                "output": args.output,
            }
        else:
            result = {
                "url": args.url,
                "shape": list(handle["X"].attrs.get("shape", [])),
                "root_keys": sorted(handle.keys()),
                "obs_keys": sorted(handle["obs"].keys()),
                "var_keys": sorted(handle["var"].keys()),
                "layers": sorted(handle.get("layers", {}).keys()),
                "obsm": sorted(handle.get("obsm", {}).keys()),
                "uns": sorted(handle.get("uns", {}).keys()),
            }
    result["bytes_transferred"] = reader.transferred
    result["remote_size"] = reader.size
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
