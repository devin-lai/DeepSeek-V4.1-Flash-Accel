"""Opt-in, rank-zero routing counters for selecting a layer offload policy.

Use the worker extension RPC while serving is idle to snapshot and reset.
Snapshot once after warmup and again after the profiling workload.
Timing with this instrumentation enabled must not be used as a speed result.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import triton
import triton.language as tl

_snapshot_callback = None


class RoutingProfileWorkerExtension:
    def snapshot_dsv41_routing(self):
        if torch.distributed.get_rank() != 0:
            return None
        if _snapshot_callback is None:
            raise RuntimeError("DSV41_ROUTING_PROFILE must be set")
        return _snapshot_callback()


@triton.jit
def _count_routes(ids, counters):
    # Eight ranks with 48 consecutive experts each; padded to 64 lanes/rank.
    lanes = tl.arange(0, 512)
    experts = (lanes // 64) * 48 + lanes % 64
    slots = tl.arange(0, 64)
    selected = tl.load(ids + slots, slots < 36, other=-1)
    active = tl.sum((experts[:, None] == selected[None, :]).to(tl.int32), 1) > 0
    active = active & (lanes % 64 < 48)
    per_rank = tl.sum(tl.reshape(active.to(tl.int32), (8, 64)), 1)
    worst = tl.max(per_rank, 0)
    tl.atomic_add(counters, 1)
    tl.atomic_add(counters + 1 + tl.arange(0, 8), per_rank.to(tl.int64))
    tl.atomic_add(counters + 9, worst.to(tl.int64))
    tl.atomic_add(counters + 10 + worst, 1)


def install_routing_profile(output_path: str) -> None:
    from vllm.logger import init_logger
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

    global _snapshot_callback
    original = Mxfp4MoEMethod.apply
    counters = {}
    generation = 0
    reported = False
    logger = init_logger("vllm.dsv41_opt")

    @torch.inference_mode()
    def snapshot_and_reset():
        nonlocal generation
        torch.cuda.synchronize()
        generation += 1
        rows = []
        for layer, tensor in sorted(counters.items()):
            values = tensor.cpu().tolist()
            steps = values[0]
            rows.append(
                {
                    "layer": layer,
                    "steps": steps,
                    "rank_unique_sums": values[1:9],
                    "max_rank_unique_sum": values[9],
                    "max_rank_unique_histogram": values[10:],
                    "mean_max_rank_unique": values[9] / steps if steps else None,
                }
            )
            tensor.zero_()
        torch.cuda.synchronize()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "generation": generation,
                    "query_tokens": 6,
                    "topk": 6,
                    "global_experts": 384,
                    "placement": "linear",
                    "rows": rows,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        temporary.replace(path)
        logger.info(
            "Routing snapshot %d written to %s; counters reset", generation, path
        )
        return {"generation": generation, "path": str(path), "layers": len(rows)}

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        nonlocal reported
        if (
            layer.global_num_experts == 384
            and layer.w13_weight.shape[0] == 48
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() == 8
            and torch.distributed.get_rank() == 0
        ):
            match = re.search(r"\.layers\.(\d+)\.", layer.layer_name)
            index = int(match[1]) if match else -1
            if 0 <= index < 40:
                if index not in counters:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "routing counters need eager warmup before capture"
                        )
                    counters[index] = torch.zeros(
                        59, device=x.device, dtype=torch.int64
                    )
                if not reported:
                    reported = True
                    logger.info(
                        "Routing profile enabled; use snapshot_dsv41_routing worker RPC"
                    )
                if topk_ids.shape == (6, 6):
                    _count_routes[(1,)](topk_ids, counters[index], num_warps=8)
        return original(
            self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
        )

    _snapshot_callback = snapshot_and_reset
    Mxfp4MoEMethod.apply = apply
