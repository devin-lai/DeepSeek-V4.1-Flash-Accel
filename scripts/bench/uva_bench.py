#!/usr/bin/env python3
"""Micro-benchmarks for the two host-memory access patterns V4.1-Flash relies on.

1. Engram-style row gather: random rows (256 B each) from a pinned FP8 table
   read by the GPU through UVA (zero-copy).
2. Expert streaming: k whole experts (MXFP4 w13, 2*2304 x 5120 bytes/2 ~ 11.8 MB
   each, we use 23.6 MB to cover w13+w2) fetched from a pinned table through
   UVA, vs. an explicit pinned H2D copy of the same bytes, vs. HBM.

Run under numactl to compare local vs remote NUMA:
    numactl --cpunodebind=0 --membind=0 python uva_bench.py --gpu 0
    numactl --cpunodebind=1 --membind=1 python uva_bench.py --gpu 0   # remote

Outputs one JSON line per measurement (stdout) and a summary table (stderr).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch


def uva_view(t: torch.Tensor) -> torch.Tensor:
    try:
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
        return get_accelerator_view_from_cpu_tensor(t)
    except Exception:  # noqa: BLE001 - fall back to torch's own zero-copy path
        return t.cuda(non_blocking=True) if not t.is_pinned() else torch.as_tensor(t).cuda()


def timeit(fn, iters: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def emit(rec: dict) -> None:
    print(json.dumps(rec), flush=True)
    print(f"  {rec}", file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--table-gib", type=float, default=4.0)
    ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--expert-bytes", type=int, default=3 * 2304 * 5120 // 2)  # w13+w2, MXFP4
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    dev = torch.device("cuda", args.gpu)
    tag = {"gpu": args.gpu}

    # --- 1. Engram-style gather --------------------------------------------
    rows = int(args.table_gib * (1 << 30)) // 256
    table = torch.empty((rows, 256), dtype=torch.uint8, pin_memory=True)
    table_uva = uva_view(table)
    for n_rows in (48, 48 * 8, 48 * 64, 48 * 512):
        idx = torch.randint(0, rows, (n_rows,), device=dev)
        t = timeit(lambda: table_uva[idx], iters=50)
        emit({**tag, "bench": "engram_gather_uva", "rows": n_rows, "us": t * 1e6,
              "rows_per_s": n_rows / t, "MBps": n_rows * 256 / t / 1e6})
    del table_uva, table

    # --- 2. Expert streaming ------------------------------------------------
    E, B = args.experts, args.expert_bytes
    W = torch.empty((E, B), dtype=torch.uint8, pin_memory=True)
    W_uva = uva_view(W)
    W_hbm = torch.empty((64, B), dtype=torch.uint8, device=dev)
    for k in (1, 6, 12, 24, 48, 96, 192):
        idx = torch.randperm(E, device=dev)[:k]
        idx_cpu = idx.cpu()
        t_uva = timeit(lambda: W_uva.index_select(0, idx), iters=10)
        t_h2d = timeit(lambda: W.index_select(0, idx_cpu).pin_memory().to(dev, non_blocking=True), iters=5)
        kk = min(k, 64)
        idx_h = torch.randperm(64, device=dev)[:kk]
        t_hbm = timeit(lambda: W_hbm.index_select(0, idx_h), iters=20)
        emit({**tag, "bench": "expert_stream", "k": k, "bytes": k * B,
              "uva_ms": t_uva * 1e3, "uva_GBps": k * B / t_uva / 1e9,
              "h2d_copy_ms": t_h2d * 1e3, "h2d_GBps": k * B / t_h2d / 1e9,
              "hbm_ms_per_expert": t_hbm * 1e3 / kk, "hbm_GBps": kk * B / t_hbm / 1e9})

    # --- 3. plain pinned H2D bandwidth ----------------------------------------
    blob = torch.empty((1 << 30,), dtype=torch.uint8, pin_memory=True)
    dst = torch.empty((1 << 30,), dtype=torch.uint8, device=dev)
    t = timeit(lambda: dst.copy_(blob, non_blocking=True), iters=10)
    emit({**tag, "bench": "h2d_1GiB", "ms": t * 1e3, "GBps": (1 << 30) / t / 1e9})


if __name__ == "__main__":
    main()
