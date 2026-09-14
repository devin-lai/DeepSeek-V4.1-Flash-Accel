#!/usr/bin/env python3
"""Plan GPU and host memory for a large MoE from the checkpoint alone.

Reads every safetensors header in the model directory -- no tensor data, so it
runs in a second on a 500 GB checkpoint -- groups tensors by role, and prints
what each rank will actually need.  It then says how much must be offloaded to
host RAM, whether the host can pay for it, and which flags to pass.

Two corrections separate this from dividing the checkpoint size by the GPU
count.  Both were measured here, and both are large enough to decide whether a
configuration exists:

**MXFP4 pads the per-rank intermediate size.**  Tensor parallelism splits each
expert's intermediate dimension, and the MoE backends then pad that shard up to
their tile width.  For DeepSeek-V4.1-Flash (`moe_intermediate_size` 2304, 40
layers) the same weights cost 44.8 GiB per rank under TP8+Marlin and 33.6 under
EP8, because expert parallelism never splits an expert and so never pads it.

**Pinned host allocations round up to the next power of two.**  Not
configurable -- neither `roundup_power2_divisions` in `PYTORCH_CUDA_ALLOC_CONF`
nor the same key in `PYTORCH_HOST_ALLOC_CONF` has any effect.  V4.1's Engram
tables are 16 shards of 11.5 GiB, each of which consumes 16, so they cost
264 GiB of host RAM rather than the checkpoint's 189.

Usage:
    python tools/plan_memory.py /path/to/DeepSeek-V4.1-Flash --tp 8 --ep
    python tools/plan_memory.py MODEL --tp 8 --moe-backend marlin --json plan.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from collections import defaultdict

GIB = 1024**3

# Per-rank intermediate-size padding applied by each MXFP4 MoE backend.
# Measured against vLLM's Mxfp4MoEMethod on sm_120.
MOE_TILE = {"marlin": 128, "triton": 64, "deepgemm": 128, "trtllm": 128, "none": 1}

ENGRAM = "engram (FP8 n-gram tables + proj)"
EXPERTS = "routed experts (quantised + block scales)"
VISION = "vision encoder + projector"

# Weights per rank exceed what the checkpoint headers imply, because the MoE
# backend repacks the expert bank into its own layout and keeps workspaces
# beside it. Measured on 8x RTX 5090, EP8 + Marlin, DeepSeek-V4.1-Flash:
# 24.09 GiB resident + 12.39 GiB offloaded = 36.5, against 34.9 predicted.
BACKEND_MARGIN_GIB = 1.6
DSPARK = "dspark drafter (MTP layers)"


def read_header(path: str) -> dict:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def classify(name: str) -> str:
    if ".engram." in name or name.startswith("engram"):
        return ENGRAM
    if "dspark" in name or ".mtp." in name:
        return DSPARK
    if "shared_experts" in name:
        return "shared experts"
    if ".experts." in name:
        return EXPERTS
    if name.startswith("vision") or ".vision" in name or "projector" in name:
        return VISION
    if "embed_tokens" in name or "lm_head" in name or name.startswith(("embed", "head")):
        return "embeddings + lm_head"
    return "attention / dense / router / norms"


def roundup_pow2(x: float) -> float:
    """What a pinned allocation of `x` bytes actually consumes."""
    return 0.0 if x <= 0 else 2.0 ** math.ceil(math.log2(x))


def expert_bytes_per_rank(cfg: dict, ranks: int, expert_parallel: bool,
                          backend: str, bits_per_weight: float) -> tuple[float, float]:
    """(padded, unpadded) routed-expert bytes per rank, in bytes."""
    t = cfg.get("text_config", cfg)
    layers = t["num_hidden_layers"]
    experts = t["n_routed_experts"]
    inter = t["moe_intermediate_size"]
    hidden = t["hidden_size"]
    tile = MOE_TILE.get(backend, 128)

    if expert_parallel:
        # Whole experts per rank: the intermediate dim is never split, so the
        # tile divides it already and nothing is padded.
        per_rank_experts = experts / ranks
        eff_inter = inter
    else:
        per_rank_experts = experts
        eff_inter = inter / ranks
    padded_inter = math.ceil(eff_inter / tile) * tile if tile > 1 else eff_inter

    def total(i):
        return layers * per_rank_experts * 3 * i * hidden * bits_per_weight / 8

    return total(padded_inter), total(eff_inter)


def pinned_offload_bytes(cfg: dict, ranks: int, expert_parallel: bool,
                         bits_per_weight: float, budget_bytes: float):
    """Host bytes a per-rank offload budget actually consumes.

    vLLM's UVA offloader pins one buffer per parameter, and pinned allocations
    round up to the next power of two (PT-001). Expert matrices are nowhere
    near powers of two, so the rounding is not a rounding error: measured on
    8x RTX 5090, a 12.4 GiB/rank budget occupied about 24 GiB of host RAM per
    rank, and the naive sum under-predicted total host use by ~100 GiB.

    Walks layers in order taking `w13_weight` then `w2_weight` -- the order the
    offloader itself uses -- until the budget is spent, rounding each buffer.
    Returns (bytes, one-line explanation).
    """
    try:
        text = cfg.get("text_config", cfg)
        layers = int(text["num_hidden_layers"])
        experts = int(text["n_routed_experts"])
        hidden = int(text["hidden_size"])
        inter = int(text["moe_intermediate_size"])
    except (KeyError, TypeError, ValueError):
        return budget_bytes, ""

    per_rank_experts = experts if expert_parallel else experts
    inter_per_rank = inter if expert_parallel else max(1, inter // ranks)
    if expert_parallel:
        per_rank_experts = max(1, experts // ranks)
    b = bits_per_weight / 8.0
    w13 = per_rank_experts * (2 * inter_per_rank) * hidden * b
    w2 = per_rank_experts * hidden * inter_per_rank * b

    spent = 0.0
    pinned = 0.0
    buffers = 0
    for _ in range(layers):
        for size in (w13, w2):
            if spent >= budget_bytes:
                break
            spent += size
            pinned += roundup_pow2(size)
            buffers += 1
        if spent >= budget_bytes:
            break
    if not buffers:
        return 0.0, ""
    factor = pinned / spent if spent else 1.0
    return pinned, (f"{buffers} pinned buffers/rank, {spent / GIB:.1f} GiB of "
                    f"weights -> {pinned / GIB:.1f} GiB pinned "
                    f"({factor:.2f}x, power-of-two rounding)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--gpu-gib", type=float, default=31.4, help="usable GiB per card (5090 = 31.4)")
    ap.add_argument("--host-gib", type=float, default=0.0, help="host RAM; 0 = read /proc/meminfo")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--ep", action="store_true", help="--enable-expert-parallel")
    ap.add_argument("--moe-backend", default="marlin", choices=sorted(MOE_TILE))
    ap.add_argument("--bits-per-weight", type=float, default=4.25,
                    help="expert storage: MXFP4 = 4.25, NVFP4 = 4.5")
    ap.add_argument("--gpu-util", type=float, default=0.93)
    ap.add_argument("--reserve-gib", type=float, default=4.0,
                    help="per rank for KV cache, CUDA graphs, activations, workspace")
    ap.add_argument("--engram-on-gpu", action="store_true")
    ap.add_argument("--no-vision", action="store_true", help="--language-model-only")
    ap.add_argument("--no-margin", action="store_true",
                    help=f"drop the measured {BACKEND_MARGIN_GIB} GiB/rank backend layout margin")
    ap.add_argument("--json")
    args = ap.parse_args()

    index_path = os.path.join(args.model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"no model.safetensors.index.json in {args.model_dir}", file=sys.stderr)
        return 2
    with open(index_path) as fh:
        files = sorted(set(json.load(fh)["weight_map"].values()))

    groups: dict[str, int] = defaultdict(int)
    missing = []
    for f in files:
        p = os.path.join(args.model_dir, f)
        try:
            hdr = read_header(p)
        except OSError:
            missing.append(f)
            continue
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            groups[classify(k)] += v["data_offsets"][1] - v["data_offsets"][0]

    total = sum(groups.values())
    print(f"checkpoint: {args.model_dir}")
    print(f"shards: {len(files) - len(missing)}/{len(files)}"
          + (f"   MISSING {len(missing)}" if missing else ""))
    print(f"\n{'component':45s} {'GiB':>9s} {'%':>6s}")
    for g, b in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"{g:45s} {b / GIB:9.1f} {100 * b / total:6.1f}")
    print(f"{'TOTAL':45s} {total / GIB:9.1f}")

    with open(os.path.join(args.model_dir, "config.json")) as fh:
        cfg = json.load(fh)

    ranks = args.tp * args.pp
    engram = groups.get(ENGRAM, 0)
    vision = groups.get(VISION, 0)
    experts_ckpt = groups.get(EXPERTS, 0)

    # --- GPU side --------------------------------------------------------
    # Everything that is not a routed expert and is not offloaded to the host.
    non_expert = total - experts_ckpt
    if not args.engram_on_gpu:
        non_expert -= engram
    if args.no_vision:
        non_expert -= vision
    try:
        padded, unpadded = expert_bytes_per_rank(cfg, ranks, args.ep, args.moe_backend,
                                                 args.bits_per_weight)
    except KeyError:
        # Not a config this tool knows how to model; fall back to plain division.
        padded = unpadded = experts_ckpt / ranks

    # The vision tower is replicated on every rank, not sharded: vLLM's
    # multimodal encoder defaults to replicate, not TP.
    per_rank_dense = (non_expert - vision) / ranks + (0 if args.no_vision else vision)

    # Measured correction. On 8x RTX 5090 with EP8 + Marlin, V4.1-Flash reports
    # 24.09 GiB resident per rank alongside 12.39 GiB offloaded -- 36.5 GiB of
    # weights against the 34.9 this arithmetic predicts. The gap is the
    # backend's repacked layout and its workspaces, which are not in the
    # checkpoint headers and so cannot be derived here. Under-predicting is the
    # expensive direction: it costs a five-minute boot that dies in
    # "No available memory for the cache blocks", so the margin is applied by
    # default and can be removed with --no-margin.
    margin = 0.0 if args.no_margin else BACKEND_MARGIN_GIB * GIB
    per_rank_weights = per_rank_dense + padded + margin
    usable = args.gpu_gib * args.gpu_util - args.reserve_gib
    offload_needed = max(0.0, per_rank_weights / GIB - usable)

    print(f"\n--- GPU, per rank ({ranks} ranks: tp={args.tp} pp={args.pp}"
          f"{' ep' if args.ep else ''}, {args.moe_backend}) ---")
    print(f"routed experts, padded:       {padded / GIB:7.1f} GiB"
          + (f"   (+{100 * (padded / unpadded - 1):.0f}% padding)" if padded > unpadded * 1.001 else "   (no padding)"))
    print(f"everything else on GPU:       {per_rank_dense / GIB:7.1f} GiB"
          + ("" if args.no_vision else "   (vision tower replicated, not sharded)"))
    if margin:
        print(f"backend layout margin:        {margin / GIB:7.1f} GiB"
              "   (measured; --no-margin to drop)")
    print(f"weights per rank:             {per_rank_weights / GIB:7.1f} GiB")
    print(f"usable per rank:              {usable:7.1f} GiB"
          f"   ({args.gpu_gib:.1f} x {args.gpu_util} - {args.reserve_gib:.1f} reserved)")

    # --- host side -------------------------------------------------------
    # Engram is stored as 2 tables x ranks shards; each shard is pinned
    # separately and each rounds up to a power of two.
    shards = 2 * ranks
    engram_shard = engram / shards if engram else 0.0
    engram_pinned = roundup_pow2(engram_shard) * shards if engram else 0.0
    # The offloaded experts are pinned the same way, one buffer per parameter,
    # so each one rounds up too -- and they are far from powers of two. The
    # offloader walks layers in order taking `w13_weight` then `w2_weight`
    # until the budget is spent, so model exactly that sequence.
    off_per_rank_actual, off_detail = pinned_offload_bytes(cfg, ranks, args.ep,
                                                           args.bits_per_weight,
                                                           offload_needed * GIB)
    offload_pinned = off_per_rank_actual * ranks
    host_needed = (0 if args.engram_on_gpu else engram_pinned) + offload_pinned

    host_total = args.host_gib
    if host_total <= 0:
        try:
            for line in open("/proc/meminfo"):
                if line.startswith("MemTotal"):
                    host_total = int(line.split()[1]) / (1024 * 1024)
                    break
        except OSError:
            host_total = 0.0

    print("\n--- host RAM (pinned) ---")
    if engram and not args.engram_on_gpu:
        print(f"engram: {shards} shards x {engram_shard / GIB:.2f} GiB"
              f" -> {roundup_pow2(engram_shard) / GIB:.0f} GiB each (power-of-two rounding)"
              f" = {engram_pinned / GIB:.0f} GiB")
    if offload_needed > 0:
        if off_detail:
            print(f"offloaded experts: {off_detail}")
            print(f"                   x {ranks} ranks = {offload_pinned / GIB:.0f} GiB")
        else:
            print(f"offloaded experts: {ranks} x {offload_needed:.1f} GiB"
                  f" = {offload_pinned / GIB:.0f} GiB")
    print(f"total pinned:                 {host_needed / GIB:7.0f} GiB"
          + (f"   of {host_total:.0f} GiB installed" if host_total else ""))

    # --- verdict ---------------------------------------------------------
    print("\n--- verdict ---")
    ok = True
    if offload_needed <= 0:
        print(f"fits on GPU with {usable - per_rank_weights / GIB:.1f} GiB per rank to spare; "
              "no expert offload needed")
    else:
        print(f"needs --cpu-offload-gb {math.ceil(offload_needed * 2) / 2:.1f}  (per rank)")
        if host_total and host_needed / GIB > host_total * 0.95:
            budget = max(0.0, (host_total * 0.95 * GIB - engram_pinned) / ranks / GIB)
            print(f"  DOES NOT FIT: {host_needed / GIB:.0f} GiB pinned on a {host_total:.0f} GiB host. "
                  f"The host can only pay for {budget:.1f} GiB/rank.")
            ok = False
    if not args.ep and args.tp > 1 and padded > unpadded * 1.001:
        print(f"  --enable-expert-parallel would save {(padded - unpadded) / GIB:.1f} GiB per rank "
              "of pure padding, and measured +79 % single-stream throughput here")
    if missing:
        print(f"  {len(missing)} shards missing -- run scripts/download/verify_shards.py")
        ok = False

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({
                "model": args.model_dir,
                "components_gib": {g: b / GIB for g, b in groups.items()},
                "total_gib": total / GIB,
                "ranks": ranks, "expert_parallel": args.ep, "moe_backend": args.moe_backend,
                "expert_gib_per_rank_padded": padded / GIB,
                "expert_gib_per_rank_unpadded": unpadded / GIB,
                "dense_gib_per_rank": per_rank_dense / GIB,
                "usable_gib_per_rank": usable,
                "offload_gib_per_rank": offload_needed,
                "engram_pinned_gib": engram_pinned / GIB,
                "host_pinned_gib": host_needed / GIB,
                "host_total_gib": host_total,
                "fits": ok,
            }, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
