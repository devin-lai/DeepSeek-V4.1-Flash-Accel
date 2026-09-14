#!/usr/bin/env python3
"""Cache/shape sanity: one 32K request and eight concurrent 8K requests.

The answer is placed at the end of the prompt. This is not an evaluation of
long-range retrieval or model quality. Requires a 32,768-token server cap.
"""

import argparse
import concurrent.futures
import json
import math
import pathlib
import time
import urllib.request

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--dir", required=True)
ap.add_argument("--base-url", default="http://127.0.0.1:8000")
ap.add_argument("--served-model", default="deepseek-v4.1-flash")
args = ap.parse_args()
root = pathlib.Path(args.dir)
root.mkdir(parents=True, exist_ok=True)
if (root / "long-context.json").exists():
    ap.error("use a fresh directory: long-context.json already exists")
base = args.base_url.rstrip("/")
model = args.served_model


def post(path, body):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def tokenize(text):
    return post(
        "/tokenize", {"model": model, "prompt": text, "add_special_tokens": False}
    )["tokens"]


unit = tokenize(
    "The archive contains dated reports about books, maps, scientific instruments, and observations. Each item has a catalog number and a description.\n"
)
tail = tokenize(
    "\nThe final verification code is violet-lantern-7329.\nQuestion: What is the final verification code?\nAnswer:"
)


def run(length, idx):
    tokens = (unit * ((length - len(tail)) // len(unit) + 1))[
        : length - len(tail)
    ] + tail
    start = time.monotonic()
    try:
        d = post(
            "/v1/completions",
            {
                "model": model,
                "prompt": tokens,
                "max_tokens": 32,
                "temperature": 0.0,
                "logprobs": 1,
                "cache_salt": f"context-{length}-{idx}-{time.time_ns()}",
            },
        )
        c = d["choices"][0]
        lp = c["logprobs"]["token_logprobs"]
        return {
            "length": length,
            "request": idx,
            "elapsed_s": time.monotonic() - start,
            "usage": d["usage"],
            "text": c["text"],
            "all_logprobs_finite": bool(lp)
            and all(x is not None and math.isfinite(x) for x in lp),
            "retrieval_ok": "violet-lantern-7329" in c["text"],
        }
    except Exception as e:
        return {"length": length, "request": idx, "error": str(e)}


rows = [run(32000, 0)]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    rows += list(pool.map(lambda i: run(8192, i), range(8)))
(root / "long-context.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2), flush=True)
raise SystemExit(
    0
    if all(x.get("all_logprobs_finite") and x.get("retrieval_ok") for x in rows)
    else 1
)
