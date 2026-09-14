#!/usr/bin/env python3
"""Verify a DeepSeek-V4.1-Flash checkpoint directory.

Checks, for every shard referenced by model.safetensors.index.json:
  1. the file exists and has no aria2 control file next to it,
  2. the safetensors header parses,
  3. the file size equals 8 + header + last tensor offset,
  4. (optional, --sha256) the SHA-256 matches ModelScope's file listing.

    python scripts/download/verify_shards.py /data/models/DeepSeek-V4.1-Flash [--sha256]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.request

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"


def header_ok(path: str) -> tuple[bool, str]:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        end = max(v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__")
        return (8 + n + end) == size, f"size {size} expected {8 + n + end}"
    except Exception as e:  # noqa: BLE001
        return False, f"header error: {e}"


def modelscope_sha(files_wanted: set[str]) -> dict[str, str]:
    url = (f"https://www.modelscope.cn/api/v1/models/{REPO}/repo/files"
           "?Revision=master&Recursive=true")
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.load(r)
    out = {}
    for item in data.get("Data", {}).get("Files", []):
        if item.get("Path") in files_wanted and item.get("Sha256"):
            out[item["Path"]] = item["Sha256"]
    return out


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as fh:
        for chunk in iter(lambda: fh.read(64 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--sha256", action="store_true", help="also hash every shard (slow, ~510 GB)")
    args = ap.parse_args()

    idx = json.load(open(os.path.join(args.model_dir, "model.safetensors.index.json")))
    files = sorted(set(idx["weight_map"].values()))
    bad = []
    for f in files:
        p = os.path.join(args.model_dir, f)
        if not os.path.exists(p):
            bad.append((f, "missing")); continue
        if os.path.exists(p + ".aria2"):
            bad.append((f, "aria2 control file present (download unfinished)")); continue
        ok, why = header_ok(p)
        if not ok:
            bad.append((f, why))
    print(f"{len(files) - len(bad)}/{len(files)} shards pass structural checks")
    for f, why in bad:
        print(f"  BAD {f}: {why}")

    if args.sha256 and not bad:
        want = modelscope_sha(set(files))
        for f in files:
            digest = sha256(os.path.join(args.model_dir, f))
            ref = want.get(f)
            status = "OK" if ref == digest else ("NO-REF" if ref is None else "MISMATCH")
            print(f"  {status:8s} {f} {digest[:16]}")
            if status == "MISMATCH":
                bad.append((f, "sha256 mismatch"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
