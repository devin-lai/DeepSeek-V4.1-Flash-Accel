# Deployment design for 8× RTX 5090 (32 GB)

This guide explains the reference V4.1 deployment and the measurements behind
its presets. Other GPU counts and memory sizes need their own validation;
adding host RAM alone does not establish that a smaller GPU setup will work.

## The constraint

| Resource or allocation | GiB |
| --- | ---: |
| Usable GPU memory, 8 × 31.4 | ≈ 251 |
| Checkpoint on disk | ≈ 476 |
| Configured expert-offload budget per rank | 12 |
| Engram pinned host allocation, including allocator rounding | ≈ 264 |
| Total pinned host memory in the recorded deployment | ≈ 452 |
| Installed host RAM | 503 |

Checkpoint bytes do not equal runtime allocation. Engram's roughly 189 GiB of
checkpoint data pins about 264 GiB, and expert offload also incurs pinned-buffer
rounding. Host RAM is a capacity constraint; PCIe reads affect speed. GPU
headroom must also cover caches, capture, workspaces, and repacked weights.
The [memory planner](../tools/plan_memory.py) includes these costs, and the
[initial serving report](../benchmarks/results/2026-09-14-v41-first-serve.md#the-offload-floor)
records why the reference preset uses a 12 GiB/rank offload budget.

## Options considered

This table records the September 2026 investigation. Alternative engines and
formats were not validated as working presets here; their current support
must be checked separately before planning a deployment.

| Option | Verdict |
| --- | --- |
| **vLLM main, TP-8 + EP, Engram on host, UVA-offload ~1/3 of routed experts** | **Chosen.** Production server, OpenAI API, tool/reasoning parsers. It was picked expecting zero patches; it needed six edits across vLLM and FlashInfer, all in `upstream/`, none of them a fork. Cost: a PCIe read for every offloaded expert that a token touches, and local VL-013 / FI-004 patches to enable CUDA graphs. |
| vLLM prefetch offloader (`--offload-group-size`) | Copies *whole layers* (all 384 experts, 6.9 GiB) to the GPU per step. At batch 1 that is ~100× more traffic than the 6 experts a token needs. Wrong tool for MoE. |
| SGLang `dev-dsv41` | Enterprise-GPU images only; no sm_120 kernels published; needs Docker. Revisit when it lands in a release. |
| KTransformers (CPU AMX experts) | Best-in-class hybrid design and this box has AMX, but V4.1 (CED, CSA2, Engram, DSpark) is unsupported as of 2026-09-13. Tracked as the future "high-concurrency" path. |
| DeepSeek reference `inference/` | Needs the full checkpoint on GPU per TP rank (64 GB per rank at MP=8). Educational only. |
| EXL3 3.0-3.5 bpw experts (community quants) | Lets the experts fit on GPU without offload on 4×128 GB Sparks, but at TP-8 on 32 GB cards it is still ~0-5 GiB short *and* needs a second 250 GB download plus the `cuda-exl3` vLLM plugin. Listed as an alternative recipe, not the default. |
| Further quantizing the FP8 dense parts to FP4 | Saves < 5 GiB total. Not worth the quality risk. |

## Parallelism: expert parallelism, not pipeline parallelism

The reference investigation compared three layouts. Expert-byte counts are
layout calculations; the outcome column describes the deployment observations.

| layout | expert bytes/rank | verdict |
| --- | ---: | --- |
| TP-8, Marlin (2304/8 = 288 padded to 384) | 44.8 GiB | needs ~22 GiB/rank offloaded; host RAM cannot pay for it |
| TP-4 × PP-2 (576 padded to 640) | 74.7 GiB for 40 layers, 20 per stage | loads, then every PP-1 worker dies: V4.1's MoE gate needs `input_ids`, which vLLM does not send across a pipeline boundary |
| **TP-8 + `--enable-expert-parallel`** (whole experts, 2304 unpadded) | **33.6 GiB** | **the working reference layout** |

Expert parallelism matters here for a reason that has nothing to do with
communication: it stops the MXFP4 kernels from padding. When each rank owns a
*slice* of every expert, 2304 intermediate columns become 288 and Marlin rounds
that to 384, inflating the whole expert bank by a third. When each rank owns 48
*whole* experts, the intermediate size is the original 2304 and nothing is
padded. That is 11 GiB per rank of pure waste removed.

Pipeline parallelism is blocked in the pinned stack by the
[`input_ids` handoff issue](../upstream/vllm/ISSUE-pp-input-ids.md).

## Which experts to spill

The UVA offloader walks layers in order and offloads matching parameters until
the byte budget is exhausted, so a plain `--cpu-offload-gb` spills the
*encoder's* first ~14 layers of experts. Under CED, prefill runs **only the
encoder**, so those layers see every prompt token and stream their experts over
PCIe on every prefill chunk. The decoder layers (20-39) are paid for during
decode either way.

Measured three ways at the same 12 GiB/rank budget, everything else identical:

| placement | 8K prefill total tok/s | 8K TTFT | out tok/s @8 |
| --- | ---: | ---: | ---: |
| stock walk order | 2 424.0 | 6 289 ms | 37.3 |
| **`DSV41_OFFLOAD_LAYERS=20-39`** | **2 654.5** | **5 741 ms** | **42.0** |
| `DSV41_OFFLOAD_LAYERS=0-19` | 2 424.7 | 6 279 ms | 38.3 |

The stock-order and encoder-only rows are close, consistent with stock order
offloading encoder experts first. Decoder-only placement records **+9.5%
prefill throughput and −8.7% TTFT** in this eager-mode experiment. Decode
results are mixed across concurrency levels; the short runs do not establish
a consistent decode improvement or its statistical uncertainty.

So `OFFLOAD_LAYERS=20-39` is in the preset. It is a CED-specific lever: on
V4-Flash, a plain decoder with no such split, the same restriction measured
55.7 against 56.1 tok/s — nothing, exactly as expected.

[Saved placement results](../benchmarks/results/2026-09-14-v41/ladder2-offload-placement.json).

## Speculative decoding

DSpark remains disabled in the presets. No post-patch DSpark run is recorded
here. A useful experiment would measure acceptance rate, draft and verification
cost, throughput, and output quality against the same non-speculative baseline.
Its potential benefit is a research question, not a published speedup.

## The sm_120 constraints, and why the preset looks like it does

The [text preset](../deploy/presets/v41-flash.env) combines required compatibility
fixes with settings chosen for the recorded workload:

- `--block-size 64` — indexer states per block, scaled per layer by its
  compression ratio (VL-009),
- `--language-model-only` — on a stock FlashInfer the dual-cache sparse-MLA
  prefill kernel is instantiated at `topk == 128` only, the sliding window with
  no image tokens (FI-003, VL-011); with the FI-003 edit applied,
  `v41-flash-vision.env` drops this flag and serves images,
- CUDA graphs **on** — with the VL-013 (null block scrubbed after capture)
  and FI-004 (masked-index gathers) patches applied. Eager mode is available
  for diagnosis but still needs the dispatch and cache-layout patches,
- the patches in `upstream/`, all of which `deploy/preflight.py --v41` checks
  for by marker before the launcher starts anything.

Use the [runbook](../deploy/README.md#local-configuration) to create an ignored
local preset when changing tuning values. Preset assignments take precedence
over caller environment variables for these settings.

## Do not use `--numa-bind`

The obvious advice on a two-socket box is to pin each worker to its GPU's NUMA
node. Here it is actively harmful. Engram plus the offloaded experts need more
pinned memory than one node has (251 GiB), so the four workers of a socket run
that node dry and the kernel OOM-kills one with
`oom-kill:constraint=CONSTRAINT_MEMORY_POLICY,nodemask=0`. And it buys nothing:
UVA reads from pinned host memory measured 51.3 GB/s local versus 51.1 GB/s
remote on this machine.

## Memory knobs, in order of leverage

1. `--cpu-offload-gb` (per rank): the only knob that makes the model fit, and
   the floor is sharp. Measured at EP-8 + Marlin: **12 GiB serves, 8 GiB fails
   with `No available memory for the cache blocks`, 4 GiB OOMs during the weight
   load.** `tools/plan_memory.py` reports 12 for this configuration, which it
   gets right only by counting the vision tower as replicated and adding a
   measured 1.6 GiB/rank for the MoE backend's repacked layout — bytes that are
   not in the checkpoint headers.
2. `--gpu-memory-utilization`: the current presets use **0.93**. Changes need
   memory profiling and successful CUDA-graph capture on the target machine.
3. `--max-num-seqs`: the presets allow **16** sequences. Higher client
   concurrency can queue; changing the server cap also changes memory needs.
4. `--max-model-len`: the serving presets cap context at **32,768 tokens**.
   Compact KV storage does not validate longer contexts; test memory,
   correctness, and latency before increasing the cap.
5. `--language-model-only`: restricts the text preset's input path. Check the
   actual loaded allocation before deducting vision weights from a memory
   estimate; the planner accounts for replication when the tower is present.

## What "optimal" means here

For an agentic workload (long prompts, short-to-medium generations, 4-32
concurrent sessions) the goal ordering is:
prefill throughput (tokens/s) → per-stream decode latency → aggregate decode
throughput → startup time. The design above optimizes them in that order:
encoder expert weights stay on GPU during prefill, while Engram still accesses
host memory. Decode also reads offloaded expert weights over PCIe. The cost
depends on the experts touched by each workload; measure both latency and
aggregate throughput when evaluating a new placement or concurrency setting.
