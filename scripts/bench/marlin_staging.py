#!/usr/bin/env python3
"""Compare UVA, staged and resident MXFP4 Marlin GEMMs on one GPU.

Synthetic packed weights, BF16 activations, 48 local / 384 global experts,
top-6 routing, and the V4.1 gate/up and down projection dimensions. This is
a kernel crossover experiment, not a serving throughput estimate. Flush L2
before each timed call and exclude the flush from CUDA-event timing.
Run without concurrent serving or other GPU benchmarks.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from vllm_dsv41_opt.staging import staged_marlin_gemm

STAGING_KIND = "staged"


def benchmark(projection, tokens, device, trials):
    local_experts, global_experts, topk = 48, 384, 6
    k, n = (5120, 4608) if projection == "gate_up" else (2304, 5120)
    host = torch.empty(
        (local_experts, k // 16, n * 2), dtype=torch.int32, pin_memory=True
    )
    host.random_(-2147483648, 2147483647)
    weight = get_accelerator_view_from_cpu_tensor(host)
    weight._vllm_is_uva_offloaded = True
    resident = weight.clone()
    scales = torch.full(
        (local_experts, k // 32, n), 122, dtype=torch.uint8, device=device
    ).view(torch.float8_e8m0fnu)
    workspace = marlin_make_workspace_new(device, 4)
    expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device=device)
    expert_map[:local_experts] = torch.arange(
        local_experts, device=device, dtype=torch.int32
    )
    cache_flush = torch.empty(128 * 1024**2, device=device, dtype=torch.uint8)
    rows = []
    for count in tokens:
        m = count if projection == "gate_up" else count * topk
        x = torch.randn((m, k), dtype=torch.bfloat16, device=device)
        ids = (
            torch.rand((count, global_experts), device=device)
            .topk(topk, dim=-1)
            .indices.to(torch.int32)
        )
        routing = torch.full(
            (count, topk), 1 / topk, dtype=torch.float32, device=device
        )
        for block_m in [8, 16, 32, 48, 64]:
            if count * topk / global_experts / block_m < 0.9:
                break
        sorted_ids, expert_ids, padded = moe_align_block_size(
            ids, block_m, global_experts, expert_map, ignore_invalid_experts=True
        )
        output = torch.zeros((count * topk, n), dtype=torch.bfloat16, device=device)
        args = [
            x,
            output,
            resident,
            None,
            scales,
            None,
            None,
            None,
            workspace,
            sorted_ids,
            expert_ids,
            padded,
            routing,
        ]
        kwargs = {
            "moe_block_size": block_m,
            "top_k": topk if projection == "gate_up" else 1,
            "mul_topk_weights": projection == "down",
            "b_q_type": scalar_types.float4_e2m1f,
            "size_m": m,
            "size_n": n,
            "size_k": k,
            "use_atomic_add": False,
            "use_fp32_reduce": True,
            "is_zp_float": False,
        }
        ops.moe_wna16_marlin_gemm(*args, **kwargs)
        expected = output.clone()
        variants = [
            ("uva", ops.moe_wna16_marlin_gemm, weight),
            (STAGING_KIND, staged_marlin_gemm(ops.moe_wna16_marlin_gemm, 1), weight),
            ("resident", ops.moe_wna16_marlin_gemm, resident),
        ]
        for kind, gemm, source in variants:
            args[2] = source
            gemm(*args, **kwargs)
            # Compare BF16 storage, including the sign bit of zero.
            assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
            for _ in range(2):
                gemm(*args, **kwargs)
            times = []
            for _ in range(trials):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                cache_flush.zero_()
                start.record()
                gemm(*args, **kwargs)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            row = {
                "projection": projection,
                "tokens": count,
                "block_m": block_m,
                "weight_bytes": weight.numel() * weight.element_size(),
                "kind": kind,
                "median_ms": statistics.median(times),
                "trials_ms": times,
                "bitwise_equal": True,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    torch.cuda.synchronize()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[6, 32, 128, 256, 512, 2048, 4096, 8192],
    )
    parser.add_argument("--projection", choices=["gate_up", "down"], action="append")
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("use a fresh output path")
    if args.trials < 1 or min(args.tokens) < 1:
        parser.error("tokens and trials must be positive")
    torch.cuda.set_device(args.device)
    torch.manual_seed(20260915)
    device = torch.device("cuda", args.device)
    rows = []
    for projection in args.projection or ["gate_up", "down"]:
        rows.extend(benchmark(projection, args.tokens, device, args.trials))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
