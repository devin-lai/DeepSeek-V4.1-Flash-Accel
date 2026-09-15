#!/usr/bin/env python3
"""Summarise results.jsonl from run_matrix.py.

For every shape, the compile-time kernel at the same page size (variant `pristine`,
falling back to `mode0`) is the control.  Prints median kernel time, ratio to the
control, and whether outputs are bit-identical to the control.
"""
import json, sys, collections, pathlib

rows = [json.loads(l) for l in pathlib.Path(sys.argv[1]).read_text().splitlines() if l.strip()]
rows = [r for r in rows if "error" not in r]

def shape_key(r):
    return (r["dual"], r["cm"], r["nh"], r["topk"], r.get("extra_topk", 0), r.get("extra_pbs", 0), r["tokens"], r["idx"])

ctrl = {}
for r in rows:
    if r["variant"] in ("pristine", "mode0"):
        k = shape_key(r) + (r["pbs"],)
        if k not in ctrl or r["variant"] == "pristine":
            ctrl[k] = r

variants = sorted({r["variant"] for r in rows}, key=lambda v: (v != "pristine", v))
by = collections.defaultdict(dict)
for r in rows:
    by[shape_key(r) + (r["pbs"],)][r["variant"]] = r

print(f"{'shape':46s} {'pbs':>4s} | " + " | ".join(f"{v:>18s}" for v in variants))
print("median us (ratio vs compile-time control at same pbs; * = output hash differs from control)")
for k in sorted(by):
    dual, cm, nh, topk, xk, xp, tokens, idx, pbs = k
    label = f"{'dual ' if dual else ''}{cm} nh={nh} topk={topk}{f' x{xk}@{xp}' if dual else ''} T={tokens} {idx}"
    c = ctrl.get(k)
    cells = []
    for v in variants:
        r = by[k].get(v)
        if r is None:
            cells.append(f"{'-':>18s}")
            continue
        us = r["us_median"]
        if c is not None:
            ratio = us / c["us_median"]
            same = (r["hash_out"] == c["hash_out"]) and (r["hash_lse"] == c["hash_lse"])
            cells.append(f"{us:9.1f} ({ratio:4.2f}){'' if same else '*'}".rjust(18))
        else:
            cells.append(f"{us:9.1f} (  n/a)".rjust(18))
    print(f"{label:46s} {pbs:4d} | " + " | ".join(cells))

# Runtime-variant page sizes without a compile-time control (16, 128): report raw and vs the pbs=64 control.
print("\nruntime-only page sizes vs the compile-time pbs=64 control of the same shape:")
for k in sorted(by):
    *sk, pbs = k
    if pbs in (32, 64):
        continue
    c64 = ctrl.get(tuple(sk) + (64,))
    if c64 is None:
        continue
    dual, cm, nh, topk, xk, xp, tokens, idx = sk
    label = f"{'dual ' if dual else ''}{cm} nh={nh} topk={topk} T={tokens}"
    cells = []
    for v in variants:
        r = by[k].get(v)
        if r is None:
            continue
        cells.append(f"{v}: {r['us_median']:.1f}us ({r['us_median']/c64['us_median']:.2f}x pbs64)")
    print(f"  {label:40s} pbs={pbs:3d}  " + "; ".join(cells))
