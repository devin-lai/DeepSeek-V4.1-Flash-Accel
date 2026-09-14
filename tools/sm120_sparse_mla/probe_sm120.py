#!/usr/bin/env python3
"""Probe and validate FlashInfer's SM120 sparse-MLA DSv4 decode kernel.

Two jobs:

1. **Oracle check.**  Run a shape FlashInfer already ships -- ``(heads=8,
   topk=128, pbs=64)`` -- against the PyTorch reference in :mod:`dsv4_ref`.  If
   those agree, the reference is trustworthy and the KV byte layout in that
   module is correct.
2. **Coverage sweep.**  Walk a grid of ``(num_heads, topk, page_block_size)``
   and record, per shape, whether the kernel dispatches at all and -- when it
   does -- how far it lands from the reference.

The shape DeepSeek-V4.1-Flash actually asks for on 8x RTX 5090 (TP8) is
``num_heads=8, topk=1152, page_block_size=32``; that is the row to watch.

    python probe_sm120.py --oracle
    python probe_sm120.py --sweep
    python probe_sm120.py --shape 8,1152,32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv4_ref import (  # noqa: E402
    BYTES_PER_TOKEN,
    D_NOPE,
    D_QK,
    D_ROPE,
    D_V,
    pack_pages,
    sparse_mla_dsv4_ref,
)

# The shape V4.1-Flash requests on TP8 sm_120: 128-token SWA window carried in a
# buffer sized for the widest layer (2*index_topk + window = 2*512 + 128).
TARGET = (8, 1152, 32)


def build_case(
    num_tokens: int,
    num_heads: int,
    topk: int,
    pbs: int,
    *,
    extra_topk: int = 0,
    extra_pbs: int = 64,
    with_sink: bool = True,
    with_lengths: bool = True,
    seed: int = 0,
    device: str = "cuda",
):
    """Synthesise one decode call plus everything the reference needs."""
    g = torch.Generator(device=device).manual_seed(seed)
    rand = lambda *s: torch.randn(*s, generator=g, device=device, dtype=torch.float32)  # noqa: E731

    # Enough pages that indices are spread over many blocks.
    kv_tokens = max(topk * 4, pbs * 8)
    kv_tokens += (-kv_tokens) % pbs
    cache, effective = pack_pages(rand(kv_tokens, D_NOPE) * 0.5, rand(kv_tokens, D_ROPE) * 0.5, pbs)

    q = (rand(num_tokens, num_heads, D_QK) * 0.5).to(torch.bfloat16)
    indices = torch.randint(
        0, kv_tokens, (num_tokens, topk), generator=g, device=device, dtype=torch.int32
    )
    lengths = None
    if with_lengths:
        lengths = torch.randint(
            topk // 2, topk + 1, (num_tokens,), generator=g, device=device, dtype=torch.int32
        )
    sink = (rand(num_heads) * 0.1) if with_sink else None

    extra = None
    if extra_topk:
        ex_tokens = max(extra_topk * 4, extra_pbs * 8)
        ex_tokens += (-ex_tokens) % extra_pbs
        ex_cache, ex_eff = pack_pages(
            rand(ex_tokens, D_NOPE) * 0.5, rand(ex_tokens, D_ROPE) * 0.5, extra_pbs
        )
        ex_idx = torch.randint(
            0, ex_tokens, (num_tokens, extra_topk), generator=g, device=device, dtype=torch.int32
        )
        ex_len = torch.randint(
            1, extra_topk + 1, (num_tokens,), generator=g, device=device, dtype=torch.int32
        )
        extra = dict(cache=ex_cache, effective=ex_eff, indices=ex_idx, lengths=ex_len)

    return dict(
        q=q,
        cache=cache,
        effective=effective,
        indices=indices,
        lengths=lengths,
        sink=sink,
        extra=extra,
        sm_scale=1.0 / (D_QK**0.5),
    )


def run_kernel(case, num_tokens, num_heads, topk, extra_topk=0):
    from flashinfer.mla._sparse_mla_sm120 import (
        _BI,
        _sparse_mla_sm120_paged_attention,
    )

    num_splits = (topk + _BI - 1) // _BI + (extra_topk + _BI - 1) // _BI
    dev = case["q"].device
    out = torch.zeros(num_tokens, num_heads, D_V, dtype=torch.bfloat16, device=dev)
    out_lse = torch.zeros(num_tokens, num_heads, dtype=torch.float32, device=dev)
    mid_out = torch.zeros(num_tokens, num_heads, num_splits, D_V, dtype=torch.bfloat16, device=dev)
    mid_lse = torch.zeros(num_tokens, num_heads, num_splits, dtype=torch.float32, device=dev)

    ex = case["extra"]
    _sparse_mla_sm120_paged_attention(
        case["q"],
        case["cache"],
        case["indices"],
        out,
        out_lse,
        case["sm_scale"],
        # d_qk=512 (DSv4) pins the scale format; only d_qk=576 accepts a choice.
        topk_length=case["lengths"],
        attn_sink=case["sink"],
        extra_kv_cache=ex["cache"] if ex else None,
        extra_indices=ex["indices"] if ex else None,
        extra_topk_length=ex["lengths"] if ex else None,
        mid_out=mid_out,
        mid_lse=mid_lse,
    )
    return out, out_lse


def compare(case, out, out_lse):
    ex = case["extra"]
    ref, ref_lse = sparse_mla_dsv4_ref(
        case["q"],
        case["effective"],
        case["indices"],
        case["sm_scale"],
        topk_length=case["lengths"],
        attn_sink=case["sink"],
        extra_kv_effective=ex["effective"] if ex else None,
        extra_indices=ex["indices"] if ex else None,
        extra_topk_length=ex["lengths"] if ex else None,
    )
    got = out.float()
    denom = ref.abs().amax().clamp_min(1e-6)
    import math
    # The kernel returns log-sum-exp in **base 2**, which neither the docstring
    # nor the parameter name says. Reporting both makes that visible instead of
    # showing up as a mysterious constant offset that grows with topk.
    return {
        "max_abs": (got - ref).abs().amax().item(),
        "rel": ((got - ref).abs().amax() / denom).item(),
        "cos": torch.nn.functional.cosine_similarity(
            got.flatten(), ref.flatten(), dim=0
        ).item(),
        "lse_max_abs": (out_lse - ref_lse).abs().amax().item(),
        "lse_max_abs_log2": (out_lse - ref_lse / math.log(2)).abs().amax().item(),
    }


def probe(num_heads, topk, pbs, *, num_tokens=8, extra_topk=0, verbose=False):
    row = {"num_heads": num_heads, "topk": topk, "pbs": pbs, "num_tokens": num_tokens,
           "extra_topk": extra_topk}
    try:
        case = build_case(num_tokens, num_heads, topk, pbs, extra_topk=extra_topk)
        out, out_lse = run_kernel(case, num_tokens, num_heads, topk, extra_topk)
        torch.cuda.synchronize()
        row.update(status="ok", **compare(case, out, out_lse))
    except Exception as exc:  # noqa: BLE001 - a probe: every failure is a datum
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}".split("\n")[0][:300]
        if verbose:
            traceback.print_exc()
    return row


def fmt(row):
    head = f"h={row['num_heads']:<4} topk={row['topk']:<5} pbs={row['pbs']:<4}"
    if row.get("extra_topk"):
        head += f" extra={row['extra_topk']:<5}"
    if row["status"] == "ok":
        ok = row["rel"] < 0.05 and row["cos"] > 0.999
        return (f"{head}  {'PASS' if ok else 'MISMATCH'}  rel={row['rel']:.2e} "
                f"cos={row['cos']:.6f} lse2={row['lse_max_abs_log2']:.2e}")
    return f"{head}  DISPATCH-FAIL  {row['error']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true", help="validate the reference on a shipped shape")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--shape", help="num_heads,topk,pbs")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--extra-topk", type=int, default=0)
    ap.add_argument("--json", help="write rows here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    torch.cuda.init()
    print(f"device: {torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability(0)}")
    import flashinfer

    print(f"flashinfer {flashinfer.__version__}\n")

    rows = []
    if args.oracle or not (args.sweep or args.shape):
        print("== oracle check: shapes FlashInfer already ships ==")
        for h, k, p in [(8, 128, 64), (8, 512, 64), (16, 256, 64), (8, 1024, 64)]:
            row = probe(h, k, p, num_tokens=args.tokens, verbose=args.verbose)
            rows.append(row)
            print("  " + fmt(row))
        print("\n== oracle check with an extra segment (dual cache) ==")
        row = probe(8, 128, 64, num_tokens=args.tokens, extra_topk=128, verbose=args.verbose)
        rows.append(row)
        print("  " + fmt(row))

    if args.shape:
        h, k, p = (int(x) for x in args.shape.split(","))
        row = probe(h, k, p, num_tokens=args.tokens, extra_topk=args.extra_topk,
                    verbose=args.verbose)
        rows.append(row)
        print(fmt(row))

    if args.sweep:
        print("\n== coverage sweep ==")
        for p in (64, 32, 128, 16):
            for h in (8, 16, 32, 64, 128):
                for k in (128, 192, 256, 512, 1024, 1152, 2048):
                    row = probe(h, k, p, num_tokens=args.tokens, verbose=args.verbose)
                    rows.append(row)
                    print("  " + fmt(row))

    print(f"\n== target shape for DeepSeek-V4.1-Flash on TP8 sm_120: "
          f"h={TARGET[0]} topk={TARGET[1]} pbs={TARGET[2]} ==")
    row = probe(*TARGET, num_tokens=args.tokens, verbose=args.verbose)
    rows.append(row)
    print("  " + fmt(row))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
