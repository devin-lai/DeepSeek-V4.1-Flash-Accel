#!/usr/bin/env python3
"""Is a CPU expert bank (the KTransformers design) better than UVA offload here?

Both designs solve the same problem -- the routed experts do not fit in GPU
memory -- and both are bandwidth problems, not compute problems, at decode.
So the question has a measurable answer on any given box:

  * **UVA offload** (what vLLM does): the expert weights stay in pinned host
    RAM and the GPU reads them over PCIe on every decode step.  Cost per step =
    bytes touched / PCIe bandwidth.
  * **CPU experts** (what KTransformers does): the weights stay in host RAM and
    the *CPU* multiplies by them, sending only activations across PCIe.  Cost
    per step = bytes touched / DRAM bandwidth, plus the latency of two small
    transfers.

Whichever side has more bandwidth wins, and the bytes touched are identical.
This script measures the CPU side directly -- a real gathered expert GEMV at
the model's true shapes -- and prints it against the measured PCIe number.

    python tools/expert_quant/cpu_experts.py
    python tools/expert_quant/cpu_experts.py --experts 6 --hidden 5120 --inter 2304
    python tools/expert_quant/cpu_experts.py --dtype int8 --threads 64
"""

from __future__ import annotations

import argparse
import os
import time

import torch

GIB = 1024**3


def bench(fn, warmup=3, iters=10) -> float:
    """`fn` takes the iteration number, so a caller can vary the access pattern.

    That matters here: repeating a step with the *same* expert indices leaves
    the working set in LLC and reports a bandwidth the real model never sees.
    """
    for i in range(warmup):
        fn(i)
    t0 = time.perf_counter()
    for i in range(iters):
        fn(warmup + i)
    return (time.perf_counter() - t0) / iters


def stream_bandwidth(gib: float, threads: int) -> float:
    """Plain large-buffer read bandwidth -- the ceiling for any expert scheme."""
    n = int(gib * GIB // 4)
    a = torch.empty(n, dtype=torch.float32)
    a.uniform_(-1, 1)
    dt = bench(lambda _: a.sum(), warmup=1, iters=3)
    return a.numel() * 4 / dt / 1e9


def expert_gemv(num_experts_total, active, hidden, inter, dtype, tokens, threads):
    """One decode step's routed-expert work, laid out the way it really is.

    The bank is a single contiguous tensor; the active experts are gathered by
    index, so the access pattern is the strided, cache-hostile one a real MoE
    produces -- not a dense pass over a small matrix.
    """
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32, "int8": torch.int8}[dtype]
    bytes_per = {"bf16": 2, "fp16": 2, "fp32": 4, "int8": 1}[dtype]

    # Size the resident bank to something that cannot sit in LLC but still fits
    # comfortably: 64 experts is ~4 GiB at bf16 and is representative.
    bank_experts = min(num_experts_total, 64)
    w13 = torch.empty(bank_experts, 2 * inter, hidden, dtype=torch_dtype)
    w2 = torch.empty(bank_experts, hidden, inter, dtype=torch_dtype)
    if torch_dtype.is_floating_point:
        w13.normal_(0, 0.02)
        w2.normal_(0, 0.02)
    else:
        w13.random_(-127, 127)
        w2.random_(-127, 127)

    x = torch.randn(tokens, hidden, dtype=torch.float32)
    g = torch.Generator().manual_seed(0)
    # A fresh routing draw per iteration. Reusing one draw would let the 6
    # active experts sit in cache and inflate the result several-fold.
    n_draws = 32
    draws = [torch.randint(0, bank_experts, (tokens, active), generator=g)
             for _ in range(n_draws)]

    compute_dtype = torch.bfloat16 if torch_dtype is torch.int8 else torch_dtype
    xc = x.to(compute_dtype)

    def step(it):
        idx = draws[it % n_draws]
        out = torch.zeros(tokens, hidden, dtype=compute_dtype)
        for t in range(tokens):
            for e in idx[t].tolist():
                a = w13[e].to(compute_dtype) if torch_dtype is torch.int8 else w13[e]
                b = w2[e].to(compute_dtype) if torch_dtype is torch.int8 else w2[e]
                h = a @ xc[t]
                gate, up = h[:inter], h[inter:]
                out[t] += b @ (torch.nn.functional.silu(gate.float()).to(compute_dtype) * up)
        return out

    dt = bench(step, warmup=2, iters=12)
    bytes_touched = tokens * active * 3 * inter * hidden * bytes_per
    return dt, bytes_touched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=5120, help="V4.1-Flash hidden_size")
    ap.add_argument("--inter", type=int, default=2304, help="moe_intermediate_size")
    ap.add_argument("--experts", type=int, default=6, help="num_experts_per_tok")
    ap.add_argument("--n-routed", type=int, default=384)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--tokens", type=int, default=1, help="decode batch size")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32", "int8"])
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--pcie-gbs", type=float, default=51.3,
                    help="measured per-GPU UVA read bandwidth (scripts/bench/uva_bench.py)")
    ap.add_argument("--hbm-gbs", type=float, default=1300.0,
                    help="per-GPU HBM read bandwidth for the resident experts")
    ap.add_argument("--offload-fraction", type=float, default=1.0,
                    help="share of expert bytes that live in host RAM (1.0 = all)")
    ap.add_argument("--gpu-bits-per-weight", type=float, default=4.25,
                    help="what the GPU side actually stores (MXFP4 = 4.25); the CPU "
                         "side is measured at --dtype, so the two must be priced "
                         "separately or the comparison is not like for like")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--skip-stream", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    print(f"threads={args.threads}  dtype={args.dtype}  "
          f"hidden={args.hidden} inter={args.inter} active={args.experts}/{args.n_routed}\n")
    if args.dtype == "int8":
        print("NOTE: torch has no fused int8 gathered expert GEMV on CPU, so this path\n"
              "      materialises a bf16 copy of each expert before multiplying. The\n"
              "      number below measures that conversion, not a real low-precision\n"
              "      kernel -- which is exactly the thing KTransformers hand-writes.\n"
              "      Use the bf16 row plus the bandwidth-bound extrapolation instead.\n")

    if not args.skip_stream:
        bw = stream_bandwidth(8.0, args.threads)
        print(f"host DRAM read bandwidth (8 GiB sequential): {bw:.1f} GB/s")

    dt, touched = expert_gemv(args.n_routed, args.experts, args.hidden, args.inter,
                              args.dtype, args.tokens, args.threads)
    eff = touched / dt / 1e9
    print(f"gathered expert GEMV, one layer, {args.tokens} token(s): "
          f"{dt * 1e3:.2f} ms, {touched / 1e6:.1f} MB touched, {eff:.1f} GB/s effective\n")

    cpu_ms = dt * 1e3 * args.layers
    # Each rank reads its 1/gpus shard: the offloaded share over PCIe, the rest
    # from HBM, in parallel across ranks.  The GPU stores the experts at
    # --gpu-bits-per-weight, not at the dtype the CPU benchmark used, so the
    # byte count has to be rescaled or the GPU is charged for bytes it never
    # moves.
    weights = args.tokens * args.experts * 3 * args.inter * args.hidden
    gpu_touched = weights * args.gpu_bits_per_weight / 8
    per_rank = gpu_touched / args.gpus
    f = args.offload_fraction
    uva_ms = ((per_rank * f) / (args.pcie_gbs * 1e9)
              + (per_rank * (1 - f)) / (args.hbm_gbs * 1e9)) * 1e3 * args.layers
    print(f"whole model, {args.layers} MoE layers, per decode step:")
    print(f"  CPU experts, all {args.n_routed} on host ({args.dtype}): "
          f"{cpu_ms:8.1f} ms  -> {1000 / cpu_ms:6.1f} tok/s ceiling")
    print(f"  GPU experts @{args.gpu_bits_per_weight} bpw, {f * 100:.0f}% offloaded "
          f"({args.gpus} GPUs, {args.pcie_gbs} GB/s PCIe + {args.hbm_gbs:.0f} GB/s HBM): "
          f"{uva_ms:8.1f} ms  -> {1000 / uva_ms:6.1f} tok/s ceiling")
    print(f"\n  ratio: CPU experts are {cpu_ms / uva_ms:.2f}x the GPU time "
          f"({'slower' if cpu_ms > uva_ms else 'faster'})")
    bpw_cpu = {"bf16": 16, "fp16": 16, "fp32": 32, "int8": 8}[args.dtype]
    if bpw_cpu != args.gpu_bits_per_weight:
        scaled = cpu_ms * args.gpu_bits_per_weight / bpw_cpu
        print(f"  if the CPU side also stored {args.gpu_bits_per_weight} bpw and stayed "
              f"bandwidth-bound: {scaled:6.1f} ms -> {1000 / scaled:.1f} tok/s, "
              f"{scaled / uva_ms:.2f}x the GPU time")
    # Where does the answer flip?  CPU experts win when one memory system beats
    # N PCIe links: gpus * pcie_gbs < cpu_effective_gbs.
    crossover = eff / args.pcie_gbs
    print(f"\ncrossover: CPU experts win below {crossover:.1f} GPUs "
          f"({eff:.0f} GB/s of gathered host bandwidth vs {args.pcie_gbs} GB/s per PCIe link). "
          f"\nThis box has {args.gpus}, so its aggregate PCIe bandwidth "
          f"({args.gpus * args.pcie_gbs:.0f} GB/s) is the larger number and the experts "
          f"belong on the GPUs.")
    print("\nBoth ceilings ignore attention, the dense path and all overlap; they bound "
          "\nonly the routed-expert term, which is the term the two designs disagree about.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
