#!/usr/bin/env python3
"""Load-test a running vLLM server and print the table this repo uses.

Measures, per concurrency level: output tokens/s, total tokens/s, time to first
token, and time per output token -- streaming, so TTFT is the real thing rather
than a whole-response timing.

    python deploy/bench.py --endpoint http://127.0.0.1:8000 --streams 1 8 32
    python deploy/bench.py --input-len 8192 --output-len 256 --streams 4
    python deploy/bench.py --json results.json --markdown

Requests carry random token-ish text of the requested length so prefix caching
cannot flatter the numbers.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request


def make_prompt(tokens: int, rng: random.Random) -> str:
    # ~0.75 words/token for English; random words defeat prefix caching.
    words = [f"w{rng.randrange(100000)}" for _ in range(max(1, int(tokens * 0.75)))]
    return " ".join(words)


def one_request(endpoint, model, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        # Ask for the real token counts rather than inferring them from the
        # prompt text -- "1024 tokens" of random words is only approximately
        # 1024 tokens, and the error would land in every total.
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"{endpoint}/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    n = 0
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - t0
                try:
                    chunk = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if (chunk.get("choices") or [{}])[0].get("text"):
                    n += 1
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return {"error": str(exc)[:120]}
    total = time.perf_counter() - t0
    return {
        "ttft": ttft or total,
        "total": total,
        "tokens": (usage or {}).get("completion_tokens", n),
        "prompt_tokens": (usage or {}).get("prompt_tokens", 0),
    }


def run_level(endpoint, model, streams, input_len, output_len, timeout, seed):
    rng = random.Random(seed)
    prompts = [make_prompt(input_len, rng) for _ in range(streams)]
    results: list[dict] = [None] * streams  # type: ignore[list-item]

    def worker(i):
        results[i] = one_request(endpoint, model, prompts[i], output_len, timeout)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(streams)]
    wall0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall0

    errs = [r for r in results if "error" in r]
    good = [r for r in results if "error" not in r]
    if not good:
        return {"streams": streams, "error": errs[0]["error"] if errs else "no result"}
    out_tokens = sum(r["tokens"] for r in good)
    in_tokens = sum(r["prompt_tokens"] for r in good) or input_len * len(good)
    return {
        "streams": streams,
        "failed": len(errs),
        "wall_s": wall,
        "prompt_tokens": in_tokens,
        "out_tok_s": out_tokens / wall,
        "total_tok_s": (out_tokens + in_tokens) / wall,
        "ttft_ms": statistics.median(r["ttft"] for r in good) * 1000,
        "tpot_ms": statistics.median(
            (r["total"] - r["ttft"]) / max(1, r["tokens"] - 1) for r in good
        ) * 1000,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
    ap.add_argument("--model")
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--json")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    model = args.model
    if not model:
        with urllib.request.urlopen(f"{args.endpoint}/v1/models", timeout=10) as r:
            model = json.load(r)["data"][0]["id"]
    print(f"endpoint={args.endpoint} model={model} "
          f"in={args.input_len} out={args.output_len}\n")

    for _ in range(args.warmup):
        one_request(args.endpoint, model, make_prompt(64, random.Random(0)), 8, 60)

    rows = []
    for s in args.streams:
        row = run_level(args.endpoint, model, s, args.input_len, args.output_len,
                        args.timeout, args.seed)
        rows.append(row)
        if "error" in row:
            print(f"  {s:>3} streams: FAILED {row['error']}")
        else:
            print(f"  {s:>3} streams: {row['out_tok_s']:8.1f} out tok/s  "
                  f"{row['total_tok_s']:9.1f} total tok/s  "
                  f"TTFT {row['ttft_ms']:7.1f} ms  TPOT {row['tpot_ms']:6.2f} ms"
                  + (f"  ({row['failed']} failed)" if row["failed"] else ""))

    if args.markdown:
        print("\n| streams | out tok/s | total tok/s | TTFT (ms) | TPOT (ms) |")
        print("| ---: | ---: | ---: | ---: | ---: |")
        for r in rows:
            if "error" in r:
                print(f"| {r['streams']} | failed | | | |")
            else:
                print(f"| {r['streams']} | {r['out_tok_s']:.1f} | {r['total_tok_s']:.0f} "
                      f"| {r['ttft_ms']:.0f} | {r['tpot_ms']:.2f} |")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"endpoint": args.endpoint, "model": model,
                       "input_len": args.input_len, "output_len": args.output_len,
                       "rows": rows}, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
