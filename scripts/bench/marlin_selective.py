#!/usr/bin/env python3
"""Rejected selective-copy prototype for the Marlin kernel benchmark.

Copy only expert matrices referenced by the aligned local expert IDs. Uses
synthetic weights and the same fixed-input bitwise checks as marlin_staging.
This SM-driven copy was slightly slower than direct UVA on the tested small
batches; it is not installed as a serving hook. Run with serving stopped.
"""

import marlin_staging as benchmark
import torch
import triton
import triton.language as tl


@triton.jit
def active_experts(
    expert_ids, padded, active, N: tl.constexpr, BM: tl.constexpr, S: tl.constexpr
):
    positions = tl.arange(0, S)
    blocks = tl.cdiv(tl.load(padded), BM)
    ids = tl.load(
        expert_ids + positions, (positions < N) & (positions < blocks), other=-1
    )
    experts = tl.arange(0, 64)
    hits = tl.sum((experts[:, None] == ids[None, :]).to(tl.int32), 1) > 0
    tl.store(active + experts, hits)


@triton.jit
def copy_selected(source, dest, active, PER: tl.constexpr, B: tl.constexpr):
    expert = tl.program_id(1)
    if tl.load(active + expert):
        offsets = tl.program_id(0) * B + tl.arange(0, B)
        value = tl.load(source + expert * PER + offsets, offsets < PER, other=0)
        tl.store(dest + expert * PER + offsets, value, offsets < PER)


def factory(original, min_tokens):
    cache = {}

    def gemm(*args, **kwargs):
        weight = args[2]
        key = (weight.device, weight.shape)
        if key not in cache:
            cache[key] = (
                torch.zeros_like(weight),
                torch.empty(64, dtype=torch.int32, device=weight.device),
            )
        temp, active = cache[key]
        ids, padded = args[10], args[11]
        block_m = kwargs["moe_block_size"]
        active_experts[(1,)](
            ids,
            padded,
            active,
            N=ids.numel(),
            BM=block_m,
            S=triton.next_power_of_2(ids.numel()),
            num_warps=8,
        )
        per = weight.numel() // weight.shape[0]
        copy_selected[(triton.cdiv(per, 8192), weight.shape[0])](
            weight, temp, active, PER=per, B=8192, num_warps=8
        )
        return original(*args[:2], temp, *args[3:], **kwargs)

    return gemm


if __name__ == "__main__":
    benchmark.staged_marlin_gemm = factory
    benchmark.STAGING_KIND = "selective"
    benchmark.main()
