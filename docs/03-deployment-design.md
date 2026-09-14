# Deployment design for 8× RTX 5090 (32 GB)

This is the decision record. It explains *why* the launch script looks the
way it does, so you can adapt it to 8× 4090-48G, 8× RTX PRO 6000, 4× 5090 with
more host RAM, and so on.

## The constraint

| | GiB |
| --- | ---: |
| HBM, 8 × 31.4 | 251 |
| minus CUDA context, CUDA graphs, FlashInfer workspace, KV cache (≈ 4 per rank) | −32 |
| **available for weights** | **≈ 219** |
| GPU-resident weights needed (everything except Engram) | 286 |
| **shortfall** | **≈ 67 → 12 GiB per rank, confirmed by measurement** |

Engram (189 GiB) goes to host RAM regardless. 503 GiB of DRAM leaves room for
Engram + ~95 GiB of spilled experts + page cache, so host memory is not the
bottleneck either. PCIe bandwidth is.

## Options considered

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

Measured on the box, three layouts and only one of them works:

| layout | expert bytes/rank | verdict |
| --- | ---: | --- |
| TP-8, Marlin (2304/8 = 288 padded to 384) | 44.8 GiB | needs ~22 GiB/rank offloaded; host RAM cannot pay for it |
| TP-4 × PP-2 (576 padded to 640) | 74.7 GiB for 40 layers, 20 per stage | loads, then every PP-1 worker dies: V4.1's MoE gate needs `input_ids`, which vLLM does not send across a pipeline boundary |
| **TP-8 + `--enable-expert-parallel`** (whole experts, 2304 unpadded) | **33.6 GiB** | **the only layout that fits** |

Expert parallelism matters here for a reason that has nothing to do with
communication: it stops the MXFP4 kernels from padding. When each rank owns a
*slice* of every expert, 2304 intermediate columns become 288 and Marlin rounds
that to 384, inflating the whole expert bank by a third. When each rank owns 48
*whole* experts, the intermediate size is the original 2304 and nothing is
padded. That is 11 GiB per rank of pure waste removed.

Pipeline parallelism is not an option at all until vLLM forwards `input_ids`
past the first stage.

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

The first and third rows are the same numbers, which is the control: stock walk
order *is* encoder offload. The second is **+9.5 % prefill throughput and
−8.7 % TTFT**, which is the mechanism working as argued. Decode moves less and
less cleanly (+12.6 % at 8 streams, −5 % at 1, which is inside this stack's
run-to-run variation); the concurrent gain is consistent with chunked prefill
sharing steps with decode.

So `OFFLOAD_LAYERS=20-39` is in the preset. It is a CED-specific lever: on
V4-Flash, a plain decoder with no such split, the same restriction measured
55.7 against 56.1 tok/s — nothing, exactly as expected.

Raw numbers: `benchmarks/results/2026-09-14-v41-first-serve.md`.

## Speculative decoding

DSpark would be enabled with 5 draft tokens and adaptive verification: at batch
1 a 5090 is nowhere near compute-bound on this model, so accepted draft tokens
should be almost free. **It is off in the preset** because it is unmeasured: adaptive verification
needs full CUDA graphs, which only became correct on sm_120 with the VL-013 /
FI-004 patches, and no DSpark run has been made since.

## The sm_120 constraints, and why the preset looks like it does

Four settings in `deploy/presets/v41-flash.env` are not tuning choices; without
any one of them the model does not serve, or serves garbage:

- `--block-size 64` — indexer states per block, scaled per layer by its
  compression ratio (VL-009),
- `--language-model-only` — on a stock FlashInfer the dual-cache sparse-MLA
  prefill kernel is instantiated at `topk == 128` only, the sliding window with
  no image tokens (FI-003, VL-011); with the FI-003 edit applied,
  `v41-flash-vision.env` drops this flag and serves images,
- CUDA graphs **on** — with the VL-013 (null block scrubbed after capture)
  and FI-004 (masked-index gathers) patches applied; `--enforce-eager` is the
  fallback for an unpatched stack and costs most of the decode throughput,
- the patches in `upstream/`, all of which `deploy/preflight.py --v41` checks
  for by marker before the launcher starts anything.

The first three are the reason this is a deployment repository and not a
one-line command: each was found by running into it, and none of them is
discoverable from an error message.

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
2. `--gpu-memory-utilization 0.92`: 5090 D has no display attached; 0.95 is
   possible once CUDA-graph capture sizes are fixed.
3. `--max-num-seqs` and `--cuda-graph-sizes`: graphs cost ~2.5 GiB at the
   default size ladder; a short ladder (1, 2, 4, 8, 16, 32) saves ~1 GiB.
4. `--max-model-len`: KV is 890 B/token; 1M tokens per request costs < 1 GiB
   per rank. Not a memory lever on this model — leave it at 262144 or higher.
5. `--language-model-only`: skips the 0.8 GiB vision tower when you only
   serve text.

## What "optimal" means here

For an agentic workload (long prompts, short-to-medium generations, 4-32
concurrent sessions) the goal ordering is:
prefill throughput (tokens/s) → per-stream decode latency → aggregate decode
throughput → startup time. The design above optimizes them in that order:
prefill never touches PCIe, decode pays a bounded PCIe tax, and aggregate
throughput is capped by that tax growing with the number of unique experts per
step (see the cost model in `01-model-anatomy.md`).
