# DeepSeek-V4.1-Flash on 8× RTX 5090: engineering report

[Project overview](../README.md) · [Benchmark protocol](../benchmarks/README.md)

Recorded investigation and measurements through 2026-09-14. Commands below
assume the repository root as the working directory.

Getting **DeepSeek-V4.1-Flash** — 552B backbone, 196B Engram, 1M context — to
serve on eight consumer Blackwell cards, and the tooling that came out of it.

The checkpoint is 476 GiB. The cards hold 251 GiB between them, have no NVLink,
and are sm_120, which the datacenter kernel paths mostly skip. As of vLLM main
@`8c1d1c297` the model did not start. It loaded all 476 GiB, pinned 452 GiB of
host memory, passed memory profiling, and then died six minutes in, in a
FlashInfer kernel that had no specialisation for its shapes. Removing that
revealed a DeepGEMM assertion underneath it, and removing *that* revealed three
more.

**It serves now, with CUDA graphs on.** Six defects, in vLLM, in FlashInfer,
and in what DeepGEMM demands of both — the sixth, the one that had forced eager
mode and cost most of the decode throughput, turned out to be a null-block
contract broken from both sides. All six are [patched here](../upstream/README.md)
as a stock wheel plus a readable diff, all six are
[written up for upstream](../upstream/), and the
[fault inventory](../faults/inventory.toml) carries the log signature of each, so
[`tools/faultscan.py`](../tools/faultscan.py) can name them from a server log.

```console
$ curl -s localhost:8000/v1/completions -d '{"model":"dsv41",
    "prompt":"The capital of France is","max_tokens":8,"temperature":0}' | jq -r .choices[0].text
 Paris. The Eiffel Tower is
```

## Status, stated plainly

| | |
| --- | --- |
| **DeepSeek-V4.1-Flash, text** | **serves, sanity-checked** — latest graph-enabled run: 5/6 greedy probes, perplexity 2.662 on a short English passage |
| DeepSeek-V4.1-Flash, images | **serves** with `PRESET=v41-flash-vision`: the dual-cache sparse-MLA prefill is instantiated at the width image tokens need ([FI-003](../docs/05-fault-inventory.md), patched); a red circle is described as a red circle, and the text probes on that server still pass |
| DeepSeek-V4.1-Flash, CUDA graphs | **on, sanity-checked.** Capture's dummy forwards were writing V4.1 state into the null block, and FlashInfer's sm_120 sparse-MLA kernels gather that block for every masked index; patched on both sides ([VL-013](../docs/05-fault-inventory.md), [FI-004](../docs/05-fault-inventory.md)). Full quality parity remains untested. |
| DeepSeek-V4-Flash (NVFP4) | serves well, CUDA graphs and all; [section 5](#5-the-v4-flash-tuning-matrix) is its full tuning matrix |
| Other GPUs | sm_90 and sm_100 should serve V4.1 without most of this — [why, and what still applies](../docs/04-optimization-guide.md#what-changes-on-other-gpus), read from the dispatch tables rather than measured |

The eager-mode caveat that shaped the first release is gone: V4.1 decode was
**~180 ms per step regardless of batch size** because 40 layers of attention,
indexer, Engram gather and MoE launched one kernel at a time.
[Section 1.6](#16-cuda-graphs-corrupted-the-model-through-the-null-block-from-both-sides)
is how that was found, and [section 2](#2-what-v41-flash-costs-today) what it
was worth.

## Quickstart

```bash
# 1. checkpoint (510 GB; resumable, two mirrors in parallel)
MODEL_DIR=/data/models/DeepSeek-V4.1-Flash bash scripts/download/download_model.sh
python scripts/download/verify_shards.py /data/models/DeepSeek-V4.1-Flash

# 2. environment: install, patch, clear the JIT-cache shadow, preflight
VENV=/data/venvs/vllm-dsv41 MODEL=/data/models/DeepSeek-V4.1-Flash \
  bash scripts/env/setup.sh

source /data/venvs/vllm-dsv41/bin/activate

# 3. serve
HOST=127.0.0.1 MODEL=/data/models/DeepSeek-V4.1-Flash \
  PRESET=v41-flash deploy/serve.sh

# 4. check it is right, not just up
deploy/healthcheck.sh && python deploy/verify.py
```

`deploy/verify.py` exists because on this hardware "up" and "correct" came
apart twice. Both failure modes answered every request promptly, with the same
token id forever, which decodes to an empty string and reads like an empty
response rather than a broken model — `/health` returns 200 throughout and a
load generator counts those tokens as throughput.

One caveat on quality numbers from this stack: **greedy decoding is not
run-to-run reproducible here.** Two runs of the identical configuration gave
different continuations and perplexities 15 % apart, both sane. MXFP4 expert
arithmetic is not associative and the expert-parallel all-to-all does not fix a
reduction order, so a near-tie in the logits can flip.

---

## 1. The six defects between the checkpoint and a served token

Each is a separate project, each has a report in [`upstream/`](../upstream/), and
each is applied by a patch script that is reversible and `--check`-able.

### 1.1 FlashInfer had no sm_120 decode kernel for V4.1's shapes

```
ValueError: SM120 sparse-MLA has no decode kernel for this shape:
num_tokens=8, num_heads=8, topk=1152, d_qk=512, page_block_size=32, ...
```

`launch_sparse_mla_decode_dsv4` dispatched through a hand-written grid of 25
compile-time specialisations — `NUM_HEADS` × `TOPK` — and rejected any
`page_block_size != 64` outright. Inside the kernel neither parameter is
structural: `TOPK` is a default, a clamp and the index-row stride;
`PAGE_BLOCK_SIZE` is the divisor that splits a token index into (page, offset).
Neither sizes shared memory, bounds an unroll, or reaches a tensor-core tile.

[The patch](../upstream/flashinfer/apply_patch.py) makes both runtime. The
instantiation grid **shrinks from 25 kernels to 5** and every shape dispatches.
Verified against a PyTorch reference that reproduces the packed FP8 KV layout
([`tools/sm120_sparse_mla/`](../tools/sm120_sparse_mla/README.md)) — the four
shapes FlashInfer already shipped are the control:

```
141 ok, 0 failed of 141          # pbs {16,32,64,128} × heads {8..128} × topk {128..2048}
worst rel: h=64 topk=128 pbs=128 rel=3.672e-02
numerical mismatches: 0
```

### 1.2 No `--block-size` could serve the model — because the block size is uniform and should not be

With the kernel available, every value of `--block-size` failed, each somewhere
different. The DSA indexer's states-per-block is `block_size //
tokens_per_state`, and `tokens_per_state` is the layer's compression ratio.
V4.1-Flash uses more than one — straight from its `config.json`:

```
compress_ratios = [0, 0,  2 ×18 (layers 2-19),  1 ×20 (layers 20-39),  0, 0, 0]
```

The CED encoder half compresses two tokens per indexer state, the decoder half
one, and DeepGEMM wants 64 states from both:

| `--block-size` | ratio-2 layers | ratio-1 layers | what breaks |
| ---: | ---: | ---: | --- |
| 128 | 64 | **128** | the decoder half: `attention.hpp:262` accepts only 32 or 64 |
| 64 | **32** | 64 | the encoder half: `:320` on sm_120 + FP8 accepts only 64 |

**The fix is to stop requiring one block size.** What the kernel constrains is
*states* per block, not tokens per block, so scale each layer's block size by
its own compression ratio:

```python
block_size = cache_config.block_size * compress_ratio   # ratio > 1
```

Every layer then stores 64 states. The two halves land in separate KV cache
groups by themselves — `UniformTypeKVCacheSpecs.is_uniform_type` returns False
as soon as block sizes differ — and nothing in the allocator changes; this
model already runs four different block sizes, since its sliding-window cache
asks for 32 and its compressor for 8. Bytes per token are unchanged.

It generalises rather than special-casing V4.1: upstream's own V4 indexer
backend declares `[256]` with the comment *"C4 indexer pages hold 64 rows"*,
which is exactly `64 × compress_ratio`.

`--block-size` for this model now means **indexer states per block**. On
sm_120 with an FP8 indexer cache, 64 is the only legal value, and
[the patch](../upstream/vllm/apply_patch.py) teaches the block-size negotiation to
say so at startup instead of aborting in a C++ assert during graph capture.

> An earlier version of [the report](../upstream/vllm/ISSUE-block-size.md) proposed
> per-group *kernel* block sizes instead, using machinery vLLM already
> half-builds. That was implemented and it cannot work here:
> `compute_layout_strides` asserts *"Padded KV pages do not support kernel block
> splitting"*, and V4.1's pages are always padded — 576-byte alignment against a
> 584-byte MLA state and a 132-byte indexer row.

### 1.3 The sliding-window cache asks for a page size no sm_120 kernel implements

`DeepseekV4MLAAttention` builds its SWA cache with a literal `block_size=32`.
Every sm_120 sparse-MLA kernel is written for a 64-token page: the prefill
dispatch bakes `PAGE_BLOCK_SIZE = 64` into all 30 of its DSV4 single-cache
launches (5 head counts × 6 topks), and the decode dispatch rejected anything
else outright with `if (mt != ModelType::DSV4 || page_block_size != 64) return
false;` — which is half of 1.1's error message. The SWA backend already declares `MultipleOf(32)`, so
asking for 64 is legal, and a 128-token window then spans two pages instead of
four.

### 1.4 sm_120 sparse-MLA *prefill* is a second dispatch grid, and V4.1 misses it too

```
Unsupported sparse-MLA prefill configuration: model=DSV4 num_heads=8
topk=1152 page_block_size=32 topk_extra=0 extra_page_block_size=0
```

Same shape, different file. `topk` here is vLLM's SWA prefill index width,
`sliding_window + vision_max_n_token` = 128 + 1024, and the kernel instantiates
`TOPK` from `{128, 192, 256, 512, 1024, 2048}`.

`TOPK` is not a work bound. The tile count comes from the *runtime* length —
`actual_ni = ceil(topk_length / BI)`, `BI = 64` — and the loops run
`ti < actual_ni`. So rounding the index buffer up to an instantiated width
costs an allocation and nothing else, and 1152 is a whole number of tiles, so
none straddles the real data. That, plus 1.3, makes the single-cache
(sliding-window-only) layers dispatch.

### 1.5 Text-only serving still sizes the index for images that cannot arrive

The dual-cache prefill path — the one every *compressed* layer takes — is
instantiated at `topk == 128` **only**. 128 is the sliding window with no image
tokens, so it is reachable exactly when the engine is text-only. But
`--language-model-only` does not make it so:

```python
self.max_image_tokens = (
    getattr(hf_config, "vision_max_n_token", 0)
    if getattr(hf_config, "vision_n_layers", 0) > 0
    else 0
)
```

`vision_n_layers > 0` says the *checkpoint* has a vision tower. It does not say
this engine will ever be handed an image. Text-only mode lives in the
multimodal config — `get_limit_per_prompt("image") == 0`, which
`--language-model-only` also forces — and neither flag touches the hf_config
attribute. So a text-only server still builds 1152-wide index rows per token,
on every platform, to describe visibility inside image spans that cannot occur.

Gating the widening on whether images can actually arrive is a win everywhere —
9× fewer column writes per prefill token, 36 MiB less index buffer at
`max_num_batched_tokens = 8192` — and on sm_120 it is what made the model
serve text before the kernel was widened.

For images themselves, the dual-cache path is now instantiated at `topk 2048`
as well (the FI-003 half of `upstream/flashinfer/apply_patch.py`): ten more
instantiations of a kernel whose `TOPK` is only a row stride and a clamp.
`PRESET=v41-flash-vision deploy/serve.sh` loads the vision tower and accepts
images; every prefill index row is 2048 wide in that mode, text included.


### 1.6 CUDA graphs corrupted the model through the null block, from both sides

With graphs on, every request came back as non-finite logits — the same token
id forever, an empty completion — and the *first* request of a fresh server,
an eager prefill above every captured size, was already wrong. Seven
hypotheses had been eliminated by measurement before this repo's first release
without finding it. What found it was a throwaway plugin that records
per-layer finiteness for every forward and can switch one startup step off at
a time, one boot of the real checkpoint per row:

| configuration | first request |
| --- | --- |
| eager | correct |
| graphs, memory-profiling capture skipped | non-finite |
| graphs, `capture_model` skipped | correct |
| graphs, only capture's dummy **warm-up** forwards, no graph recorded, every real batch eager | **non-finite** |
| graphs, full capture, then block 0 of every KV cache zeroed | correct |

So the graphs were innocent; the forwards run to warm them up were not. And
the first non-finite tensor was the raw output of the layer-0 sliding-window
attention kernel — with q, kv, weights and RoPE cache all finite — for 126 of
132 tokens. The six finite ones were 63, 127 and 128..131: exactly the queries
whose index row fills whole 64-entry tiles. Everyone else's row ends in a
partial tile padded with `-1`.

Two halves, both needed:

- **FlashInfer reads block 0 for every `-1`.** `io_bulk_gather_tile`,
  `io_gather_scales`, `prefill_kv_entry_base` and the decode gather all do
  `idx = (idx >= 0) ? idx : 0`, copy row 0 of block 0 into shared memory, force
  the entry's score to `-1e30` and rely on the zero softmax weight to cancel
  it. `0 × NaN` is `NaN`.
- **vLLM writes block 0.** Capture's dummy forwards use an all-zero block table
  and a `PAD_SLOT_ID` slot mapping. Plain KV inserts honour the pad; V4.1's
  compressed-KV, indexer-K and fp32 compressor-state writes derive their slots
  from the block table, so they land in block 0 — the null block, never
  allocated to a request, `torch.zeros` at start. Hybrid KV groups overlay one
  buffer, so the fp32 state ring lands on layer 0's sliding-window cache. Read
  back after capture: 8143 non-zero bytes in rows 0..7, two NaN bytes in row 0.
  In eager mode the block is all zeros.

Both sides are patched and either alone is enough: vLLM zeroes block 0 at the
end of `capture_model`, and the kernels gather the tile's first (valid) entry
instead of row 0 ([VL-013](../upstream/vllm/ISSUE-v41-cudagraphs.md),
[FI-004](../upstream/flashinfer/ISSUE-null-block.md)). DeepSeek-V4 has neither
half in this shape, which is why it was correct with graphs on the same build
all along — the "not a regression from the patches" control was right, and
misleading.

---

## 2. What V4.1-Flash costs today

`PRESET=v41-flash deploy/serve.sh` exactly as the quickstart runs it — TP-8 +
expert parallel, Engram on the host, 12 GiB/rank of decoder experts offloaded,
`--block-size 64`, text-only, **CUDA graphs on**. `vllm bench serve`, random
1024-in / 128-out. The eager column is the same kit before section 1.6 was
found, so the difference is the graphs and nothing else.

| | 1 stream | 8 streams | 32 streams | 8K prefill |
| --- | ---: | ---: | ---: | ---: |
| output tok/s, **graphs** | **33.7** | **92.4** | **115.8** | — |
| output tok/s, eager | 6.0 | 38.2 | 73.7 | — |
| total tok/s, graphs | 303.0 | 831.3 | 1 042.6 | **2 650** |
| median TTFT, graphs | 692 ms | 2 167 ms | 19 688 ms | 5 739 ms |
| median TPOT, **graphs** | **24.0 ms** | **72.3 ms** | **128.9 ms** | — |
| median TPOT, eager | 159.5 ms | 194.6 ms | 194.1 ms | — |

Single-stream output throughput is 5.6× the patched eager baseline, and median TPOT went from 160 ms to
24 ms; prefill is unchanged, because prefill was never graphed. Startup is
~285 s with the page cache warm (48 shards in 42 s, the Marlin repack, 452 GiB
pinned), then capture adds six seconds and 0.09 GiB per rank. `deploy/verify.py`
against this same server: 5/6 greedy continuations (the miss is "the water
will freeze at zero degrees", a legitimate continuation without the word the
probe wants), perplexity 2.66, and the word problem answered. Raw runs in
`benchmarks/results/2026-09-14-v41-cudagraphs.md`.

**TPOT still rises with concurrency** — 24 → 72 → 129 ms from 1 to 32 streams
— and that is now the real shape of the model on this box: the 12 GiB/rank of
experts behind PCIe are touched more as batch grows ([VL-007](../docs/05-fault-inventory.md)),
and every decode step still crosses the UPI for its all-reduces. For scale,
DeepSeek-V4-Flash with nothing offloaded does 117.9 tok/s single-stream on
the same cards.

### Offload the decoder half, not the encoder half

CED means prefill runs layers 0-19 only, so where the offloaded experts live
decides whether prefill touches PCIe at all. Same 12 GiB/rank budget, three
placements, each verified correct first:

| placement | 8K prefill total tok/s | 8K TTFT | out tok/s @8 |
| --- | ---: | ---: | ---: |
| stock walk order | 2 424.0 | 6 289 ms | 37.3 |
| **`DSV41_OFFLOAD_LAYERS=20-39`** | **2 654.5** | **5 741 ms** | **42.0** |
| `DSV41_OFFLOAD_LAYERS=0-19` | 2 424.7 | 6 279 ms | 38.3 |

Rows one and three are the same numbers, which is the control: vLLM's offloader
spends the budget on the first layers, so its default *is* encoder offload.
Moving it to the decoder half is **+9.5 % prefill throughput and −8.7 % TTFT**.

It is a CED-specific lever and worth nothing without that split — on V4-Flash,
a plain decoder, the same restriction measured 55.7 against 56.1 tok/s.

### The original preset needed 12 GiB/rank

| offload per rank | result |
| ---: | --- |
| 12 GiB | serves |
| 8 GiB | `No available memory for the cache blocks` |
| 4 GiB | CUDA OOM during weight load |

These are the original preset results, not a hardware lower bound. The newer
[text presets](../deploy/README.md#choose-a-preset) change GPU utilization,
graph capture sizes and the host allocator.

`tools/plan_memory.py` predicted 12.0 for this configuration, which was the
value that works. It gets there by counting the vision tower as replicated
rather than sharded, and by adding a measured 1.6 GiB/rank for the MoE
backend's repacked layout and workspaces — bytes that are not in the checkpoint
headers and so cannot be derived from them. Under-predicting is the expensive
direction: it costs a five-minute boot that dies at the very end.

---

## 3. Reusable regardless of which model you are serving

- a **memory model** for 500 GB-class MoE checkpoints on small GPUs that says
  how much has to live in host RAM, mostly from the checkpoint headers and
  partly from two corrections that are not in them
  ([`tools/plan_memory.py`](../tools/plan_memory.py), section 4),
- a **fault inventory** of 19 failure modes with log signatures, and a scanner
  that matches a server log against them and prints the fix,
- a **correctness gate** ([`deploy/verify.py`](../deploy/verify.py)) that fails a
  deployment producing fluent garbage, not just a dead one,
- a **tiny DeepSeek-V4.1** ([`tools/tiny/`](../tools/tiny/README.md)) that keeps
  the CED split, the two compression ratios and the sparse-MLA head geometry
  and throws away the weights, so shape and configuration bugs reproduce in a
  minute instead of six,
- a **PyTorch reference** for sm_120 sparse MLA, including the packed FP8 KV
  layout, validated against the shapes FlashInfer already ships,
- a **deployment kit** — preflight, launcher, presets, healthcheck, systemd —
  where every default encodes a measurement,
- an **expert-quantisation study** on the real checkpoint, and a measurement
  that settles whether the KTransformers design would have been better here.

## 4. The memory model

```bash
python tools/plan_memory.py /path/to/DeepSeek-V4.1-Flash --tp 8 --ep
```

| component | GiB | where it has to live |
| --- | ---: | --- |
| routed experts, MXFP4 + block scales | 275.7 | GPU, partly spilled to host |
| Engram tables + projections, FP8 | 189.1 | host RAM, read by hash lookup |
| attention / dense / routers / norms, FP8 | 5.8 | GPU |
| embeddings + LM head, BF16 | 2.5 | GPU |
| shared experts, FP8 | 1.4 | GPU |
| vision encoder + projector | 0.8 | GPU |
| **total** | **475.2** | |

Dividing that by eight is wrong in two directions, and both corrections were
measured here.

**GPU side — MXFP4 pads the per-rank intermediate size.** Tensor parallelism
splits each expert's intermediate dimension and the MoE backend pads that shard
up to its tile width:

| layout | intermediate/rank | padded to | expert bytes/rank |
| --- | ---: | ---: | ---: |
| TP8, Marlin / DeepGEMM / TRT-LLM | 288 | 384 (1.33×) | 44.8 GiB |
| TP8, Triton | 288 | 320 (1.11×) | 37.4 GiB |
| TP4, Marlin | 576 | 640 (1.11×) | 74.7 GiB |
| **EP8, whole experts** | **2304** | **2304 (1.00×)** | **33.6 GiB** |

**Host side — the stock pinned allocator rounds to powers of two.** The
allocator environment settings tested here did not change that behavior.
The revised planner counts distinct weight and scale allocations:

- Engram's 16 weight shards round to 16 GiB each, and its 16 FP8 scale
  shards round to 0.5 GiB each: **264 GiB** total. Those smaller arrays are
  scales, not projection weights.
- A 12 GiB/rank expert budget spills whole parameters: 16 `w13_weight` plus
  15 `w2_weight` buffers, **12.3926 GiB payload → 23.5 GiB pinned per rank**.
  Scales stay on GPU and must not be counted toward that offload budget.
- The total is **264 + 8 × 23.5 = 452 GiB**, matching about 453 GiB shared
  memory once IPC is included. The earlier 29-buffer / 432 GiB estimate
  incorrectly charged scales against the budget and is superseded.

Plugin 0.2.0's opt-in exact allocator removes this padding for persistent
weights. At the same expert budget shared RAM falls to about 289 GiB; the
new DSpark preset uses about 279 GiB. See the
[measured optimization report](../benchmarks/results/2026-09-14-v41-optimization.md).

## 5. The V4-Flash tuning matrix

DeepSeek-V4-Flash-NVFP4, eight configurations, each a fresh server, `vllm bench
serve` on random 1024-in / 256-out plus long-prompt cases. Output tokens/s.
These are V4-Flash measurements, a separate model with a smaller checkpoint.
They are not V4.1-Flash results or predictions; section 2 reports V4.1 with CUDA graphs enabled.

| config | 1 stream | 8 streams | 32 streams | 8K prefill (total tok/s) | TPOT @1 | startup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP8 baseline | 65.7 | 298.6 | 1 079.9 | 14 808 | 7.64 ms | 180 s |
| **TP8 + expert parallel** | **117.9** | **580.4** | **1 385.3** | 14 499 | 7.84 ms | 120 s |
| TP8 + EP + `--numa-bind` | 118.0 | 580.3 | 1 375.5 | 14 599 | 7.83 ms | 120 s |
| TP8 + EP, 6 GiB/rank offloaded | 56.1 | 148.0 | 268.6 | 5 136 | 16.61 ms | 125 s |
| TP8 + EP, 12 GiB/rank offloaded | 36.2 | 84.1 | 148.9 | 2 842 | 25.52 ms | 130 s |
| TP8 + MTP, 1 draft token | 101.6 | 542.9 | 586.6 | 3 858 | 8.79 ms | 220 s |

What it says:

- **Expert parallelism is the single largest win: +79 % single-stream, +94 % at
  8 streams, +28 % at 32**, with prefill unchanged and a third off the startup
  time. It is not about communication — it stops MXFP4 from padding. Turn it on.
- **Offloading is brutal and roughly linear in offloaded bytes.** Offload only
  what you must to make the model fit.
- **`--numa-bind` did not improve that V4-Flash case** (118.0 vs 117.9),
  and a separate host-heavy run OOM-killed a worker under strict binding.
  These observations do not establish NUMA independence for other workloads.
- **MTP speculative decoding is a losing trade against expert parallelism.**

### Host-memory access, the foundation of every offload decision

`scripts/bench/uva_bench.py`, pinned host buffers read by the GPU through UVA:

| | Engram-style gather, 3072 rows | 6 experts (70 MB) | plain H2D, 1 GiB |
| --- | ---: | ---: | ---: |
| pinned pages on the GPU's own NUMA node | 19 µs | 51.3 GB/s | 56.3 GB/s |
| pinned pages on the remote node | 18 µs | 51.1 GB/s | 56.1 GB/s |

These single-GPU reads reach 91% of plain host-to-device bandwidth. The
conclusion previously drawn here that NUMA placement does not matter was too
broad. Eight simultaneous readers measured 260.3 GB/s aggregate locally and
197.4 GB/s remotely. See the [concurrent test](../benchmarks/results/2026-09-14-v41-optimization.md).

## 6. Failure modes, and what to do about them

[`docs/05-fault-inventory.md`](../docs/05-fault-inventory.md) is the long form,
generated from [`faults/inventory.toml`](../faults/inventory.toml). The inventory
is machine-readable because the useful thing to do with it is match it against
a log:

```console
$ python tools/faultscan.py /data/logs/serve-20260914.log
2 known fault(s) matched:

FI-003  [blocker]  SM120 sparse-MLA prefill is a second dispatch grid, and V4.1 misses it on both axes
  fix: round the prefill index width up to an instantiated TOPK, and ask the SWA cache for 64-token pages
VL-013  [major]    CUDA-graph capture writes V4.1 state into the null block, which the sm_120 sparse-MLA kernels gather for every masked index
  fix: zero block 0 after capture (vLLM) and gather a finite row for masked indices (FlashInfer)
```

It exits 1 on a blocker or major fault, so it drops into CI or a restart loop.
`--list` prints the table, `--show VL-009` one entry in full, `--markdown`
regenerates the doc.

The ones that cost the most time, beyond section 1:

- **`nvcc` off `PATH` silently disables every FlashInfer backend**, and vLLM
  reports it as a missing kernel build pointing at a FlashInfer PR. The fix is
  `export PATH=/usr/local/cuda/bin:$PATH`.
- **A prebuilt `flashinfer-jit-cache` wheel shadows any edit to `csrc/`**, and
  clearing the user JIT cache does not help. Worth knowing before you spend an
  hour debugging a kernel change that was never compiled.
- **FlashInfer autotuning inside the startup collective hangs multi-GPU
  launches** for 40 minutes before a peer drops.
- **Pinned host allocations round to a power of two**, the single largest term
  in V4.1's host budget.
- **Pipeline parallelism is unusable for V4.1**: stage-1 workers raise
  `DeepSeek V4 vision MoE routing requires input_ids`.

## 7. What would change the picture

- **Adaptive DSpark verification on SM120.** Static DSpark is now measured in
  the latency preset. Adaptive verification remains blocked by the indexer
  backend support contract; full CUDA graphs alone are insufficient.
- **Broader vision validation.** FI-003 is patched and a simple image probe
  passes. Multi-image inputs, resolutions and representative vision tasks
  still need a validation matrix.
- **Higher-precision V4.1 weights.** Every quantisation result here is a
  *requantisation* of an already-4-bit release, so it carries MXFP4's error and
  its own. A 3-bit build from the original weights would be materially better
  than the 3-bit row measured here, and no deployer can produce one.
- **`input_ids` across pipeline stages** would make TP4 × PP2 available, which
  halves the number of ranks that must each hold a shard of the expert bank.

## 8. Expert quantisation, and whether KTransformers was the better idea

[`docs/06-expert-quantization.md`](../docs/06-expert-quantization.md) measures what
storing the experts in fewer bits is worth, on the real checkpoint. The control
that makes it trustworthy: the shipped weights are already MXFP4, so
re-encoding them must be a no-op, and it is — `0.0000` on all three metrics.

- **Error feedback matters more than the format.** GPTQ-style error feedback —
  pushing each column's rounding error into the columns not yet reached,
  weighted by that projection's activation Hessian — cuts expert-output error by
  2.8× to 3.5× at every bit budget. A larger effect than the quantiser, the
  rotation, or a whole extra bit.
- **The memory estimate suggests expert offload could be removed.**
  `gptq-vq2/3-g128` is 2.46 bits per weight, projects 19.4 GiB per rank, and lands at 0.149
  expert-output error — where a 4.5-bit NVFP4 requantisation lands while still
  needing 15 GiB per rank in host RAM. This is expert-level quantization
  evidence, not a validated full-model serving configuration.
- **Converting MXFP4 to NVFP4 is a pure loss** — more bits, more memory, *and*
  more error, because FP8 block scales cannot represent the power-of-two scales
  MXFP4 already used.
- **CPU expert execution remains an alternative to benchmark end to end.**
  Gathered expert GEMV measured 110 GB/s on the two Xeons. Multiplying the
  single-GPU UVA result by eight overstated aggregate bandwidth: concurrent
  local readers reached 260.3 GB/s. These different microbenchmarks do not
  settle a serving-engine comparison or establish a GPU-count crossover.

## 9. Repository layout

```
deploy/                         preflight, launcher, presets, healthcheck, verify, systemd, bench
faults/inventory.toml           19 failure modes, machine-readable, with log signatures
tools/faultscan.py              match a server log against them; --list, --show, --markdown
tools/plan_memory.py            memory plan from the checkpoint headers
tools/tiny/                     a DeepSeek-V4.1 that boots in a minute, for shape bugs
tools/sm120_sparse_mla/         PyTorch reference for sm_120 sparse MLA + shape probe
tools/expert_quant/             requantisation explorer, CPU-vs-PCIe benchmark
upstream/flashinfer/            the kernel patch (three edit groups), and three reports with their evidence
upstream/vllm/                  the vLLM patch, and five reports
docs/01-model-anatomy.md        what CED, CSA2, Engram and DSpark mean for a deployer
docs/02-hardware-topology.md    the box, its interconnect, and measured bandwidths
docs/03-deployment-design.md    the decision record, with the rejected options
docs/04-optimization-guide.md   levers ranked by effect, for any hardware
docs/05-fault-inventory.md      the fault inventory, long form
docs/06-expert-quantization.md  the quantisation study and the KTransformers measurement
scripts/                        download + verify, environment setup, benchmarks
vllm_dsv41_opt/                 the vLLM plugin (single-copy offload, layer ranges)
benchmarks/results/             saved benchmark summaries and selected probe outputs
```

## The machine

2× Xeon Gold 6530 (64 cores), 503 GiB DDR5, 8× RTX 5090 (sm_120, 32 GB per GPU),
PCIe Gen5 ×16, no NVLink, GPU P2P disabled. vLLM main @`8c1d1c297`,
FlashInfer 0.6.18.post1, torch 2.13 + cu130, driver 595.71.05, CUDA 13.2.

The deployment measurements in this report come from that machine. Saved
benchmark summaries and selected probe outputs are in `benchmarks/results/`;
some diagnostic logs referenced by the upstream reports remain on the original host.

## Licence

[MIT](../LICENSE), matching the model.

The patches in [`upstream/`](../upstream/) are diffs against vLLM (Apache-2.0) and
FlashInfer (Apache-2.0) and carry those projects' licences where they apply;
they are distributed here as patch scripts rather than as forked source.
