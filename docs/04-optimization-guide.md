# Optimization guide for DeepSeek-V4.1-Flash (any hardware)

The 8× 5090 recipe is one point in a design space. This guide lists the levers
that matter for V4.1-Flash specifically, ordered by how much they move
throughput or latency, with the reasoning behind each so you can re-derive the
right setting for your own box. Measured deltas on the reference machine are
in `benchmarks/`.

## 1. Put the model where it belongs

### Engram on the host, always

The two n-gram tables are 189 GiB of FP8 that are *indexed*, not multiplied.
A token reads 48 rows × 256 B = 12 KB. The tables account for about 40% of the
checkpoint; placing them on the host frees GPU memory for the compute weights. vLLM:
`--engram-config '{"cpu_offload": true}'` (TP-sharded pinned tables, UVA
gather). The tech report's own serving stack prefetches Engram rows from host
DRAM over RDMA.

Host-side details that matter:

- **Do not pass `--numa-bind`.** The obvious advice on a two-socket box is
  wrong here, twice over. It buys nothing — a UVA read from the remote node
  measured 51.1 GB/s against 51.3 GB/s local, a 0.4 % difference, because the
  read is bounded by the PCIe link and not by the memory controller — and it
  costs a worker: Engram plus offloaded experts exceed one socket's 251 GiB,
  so the kernel OOM-kills a rank with
  `oom-kill:constraint=CONSTRAINT_MEMORY_POLICY`. Measured end to end, it
  changes throughput by 0.1 % (118.0 vs 117.9 tok/s) when it does not kill the
  server.
- Reserve host RAM: Engram + offloaded experts + a few GiB per worker. No
  swap. Watch `/proc/meminfo` `Unevictable` (pinned pages) — and remember that
  pinned allocations round up to a power of two, so Engram's 189 GiB of
  tensors occupy 264 GiB of RAM.

### Spill experts, never attention

If GPU memory is still short after Engram is on the host, the remaining
choice is which *routed experts* to spill. Attention, norms, routers, shared
experts and the LM head run for every token and are small (≈ 10 GiB total);
offloading them costs PCIe bandwidth for every token for no memory gain.

vLLM's UVA offloader makes this selectable: `--cpu-offload-params
w13_weight w2_weight` limits eligibility to the fused expert matrices; the
scales stay resident.

### Spill *decoder* experts (this model only)

CED means prefill runs layers 0-19 only. Spilling encoder experts makes every
prefill chunk stream the experts of each spilled layer over PCIe; spilling
decoder experts costs nothing during prefill and the same during decode. The
`vllm_dsv41_opt` plugin (`DSV41_OFFLOAD_LAYERS=20-39`) implements this in one
environment variable.

Measured at 12 GiB/rank: **+9.5 % prefill throughput, −8.7 % TTFT on 8K
prompts**, against vLLM's stock walk order — which spends the budget on the
first layers and so is encoder offload by default. The control is that
explicitly setting `0-19` reproduces the stock numbers to within 0.03 %.

This is CED-specific. On a plain decoder the same restriction is worth nothing:
V4-Flash measured 55.7 against 56.1 tok/s with and without it.

### How much to spill

`tools/plan_memory.py` computes it from the checkpoint and your GPU size.
Rule of thumb for MXFP4 experts: each layer holds 6.9 GiB of routed experts
(with scales); per TP rank that is 6.9 / TP GiB. Spill the smallest number of
layers that leaves ≥ 4 GiB per rank for KV, CUDA graphs and workspace.

## 2. Speculative decoding: DSpark

**Untested here.** Adaptive verification requires full CUDA graphs, which were
only made correct on sm_120 by the VL-013 / FI-004 patches, and no DSpark run
has been measured on the reference machine since. What follows is read off the
model's config and vLLM's implementation, not off a benchmark; the MTP row in
the V4-Flash matrix is the only speculative-decoding *measurement* in this
repository, and it was a losing trade against expert parallelism.

Decode on this model is latency-bound: 16B active parameters per token is a
few tens of milliseconds of kernel launches, all-reduces and (if spilled)
PCIe reads. DSpark drafts 5 tokens in one pass through 3 small blocks; each
accepted token saves one full step. With adaptive verification the draft
length shrinks automatically at high concurrency when verification compute
stops being free.

- `"num_speculative_tokens": 5` matches the trained block size
  (`dspark_block_size = 5`). Larger values gain nothing.
- `"draft_sample_method": "probabilistic"` for `temperature > 0` workloads
  (the recommended sampling is T = 1.0, top-p 0.95).
- Adaptive verification requires full CUDA graphs (no `--enforce-eager`) and
  no pipeline parallelism.
- Expect ~2-4 accepted tokens per step on code, less on free prose.

## 3. Parallelism on PCIe-only boxes

- **Turn expert parallelism on** (`--enable-expert-parallel`). This is the
  single largest lever measured here: +79 % single-stream, +94 % at 8 streams,
  +28 % at 32, with prefill unchanged and a third off the startup time.
  The reason has nothing to do with communication, which is why the usual
  "no NVLink, so no EP" reasoning gets it backwards — see below.
- **Pipeline parallelism is not available** for V4.1: stage-1 workers raise
  `DeepSeek V4 vision MoE routing requires input_ids`, because the MoE gate
  keeps separate expert-selection biases for text and image tokens and vLLM
  does not carry token ids across a pipeline boundary. TP-8 is the only
  8-GPU layout that runs.
- Data parallel (`--data-parallel-size 2` with TP-4) doubles aggregate
  throughput at the cost of two copies of the weights: only for cards large
  enough to hold one copy per group.

### Why expert parallelism wins on a box with no NVLink

Tensor-parallelism splits each expert's intermediate dimension, and the MXFP4
MoE kernel then pads that shard up to its tile width. At TP-8 each rank owns
288 of an expert's 2304 intermediate columns, which Marlin rounds to 384 — a
33 % inflation of the largest tensor in the checkpoint. Expert parallelism
gives each rank 48 *whole* experts instead, so the intermediate size is the
original 2304 and nothing is padded:

| layout | intermediate/rank | padded to | expert bytes/rank |
| --- | ---: | ---: | ---: |
| TP-8, Marlin / DeepGEMM / TRT-LLM | 288 | 384 (1.33x) | 44.8 GiB |
| TP-8, Triton | 288 | 320 (1.11x) | 37.4 GiB |
| **EP-8, whole experts** | **2304** | **2304 (1.00x)** | **33.6 GiB** |

11 GiB per rank of pure padding removed is worth far more than the all-to-all
costs, and on a memory-starved box it is often the difference between
offloading and not. The advice to avoid EP applies to boxes where the experts
already fit and the all-to-all is the only term that changes; it does not
apply here.

## 4. Consumer Blackwell (sm_120) specifics

Everything above is architecture-neutral. These are not: they are the flags
V4.1 needs on sm_120 and the reasons, and each one is a fault entry with a log
signature in [`05-fault-inventory.md`](05-fault-inventory.md).

| flag | why | fault |
| --- | --- | --- |
| `--block-size 64` | it counts indexer *states* per block, scaled by each layer's compression ratio. DeepGEMM's paged MQA logits accepts 32 or 64, and on sm_120 with an FP8 indexer cache, only 64 | VL-009 |
| `--language-model-only` (unless the FI-003 FlashInfer edit is applied) | the stock dual-cache sparse-MLA prefill kernel is instantiated at `topk == 128` only, the sliding window with no image tokens; the patch adds 2048 and `v41-flash-vision.env` serves images | FI-003, VL-011 |
| CUDA graphs on (no `--enforce-eager`) | needs the VL-013 patch (null block scrubbed after capture) and FI-004 (kernels stop gathering block 0 for masked indices); on an unpatched stack capture's dummy forwards leave NaN bytes in block 0 and every partial-tile query returns NaN | VL-013, FI-004 |
| `export PATH=/usr/local/cuda/bin:$PATH` | vLLM gates every FlashInfer backend on `shutil.which("nvcc")` unless `flashinfer-cubin` is installed, and reports its absence as a missing kernel build | VL-010 |
| `--kernel-config '{"enable_flashinfer_autotune": false}'` | autotuning JIT-compiles inside the startup collective; ranks finish minutes apart and a peer drops | VL-003 |

`deploy/preflight.py --v41` checks all of them, plus every patch marker, before
the launcher starts anything.

### What changes on other GPUs

**Read from the source, not measured** — this repository has one machine, and it
is sm_120. Everything below is derived from DeepGEMM's `csrc/apis/attention.hpp`
and FlashInfer's sm_120 dispatch files, and should be checked before being
relied on. The point is that the *shape* of the constraint is portable even
though its value is not.

DeepGEMM's paged MQA logits has two predicates on states-per-block. The
metadata builder accepts 32 or 64 everywhere. The kernel narrows it:

```
(arch_major == 10 and block_kv in {32, 64, 128})
 or (arch_major ==  9 and block_kv in {32, 64})
 or (arch_major == 12 and ((is_fp4 and block_kv in {32, 64})
                           or (not is_fp4 and block_kv == 64)))
```

Intersecting the two, for V4.1's mix of compression ratios 2 and 1:

| architecture | legal states/block | a single `--block-size` that works? |
| --- | --- | --- |
| sm_90 (H100/H200) | 32 or 64 | **yes, 64** — ratio-2 gives 32, ratio-1 gives 64, both legal, and the backends already declare 64 there |
| sm_100 (B200/GB200) | 32 or 64 | **yes, 64** — but only *after* the backends are made to offer it. Unpatched they declare `[64 if is_device_capability_family(90) else 128]`, so 64 cannot be negotiated and 128 fails the ratio-1 half |
| **sm_120 (RTX 5090, RTX PRO 6000)** | **exactly 64**, unless the indexer cache is MXFP4 | **no** — this is [VL-009](05-fault-inventory.md) |

So sm_90 serves this model with a uniform block size *by luck*: its accepted set
has two elements and the two halves happen to land on one each. sm_100 has the
same luck but cannot reach it, because the backend declaration is written as
"64 on sm_90, else 128" — one line of the patch here fixes that and nothing else
is needed. sm_120 has a one-element set, and a model with two compression ratios
cannot hit it twice; that is the half of the patch that scales the block size
per ratio.

Scaling per ratio is correct on all three architectures and only load-bearing
on the third.

The same reading says the sm_120 restriction may not be a correctness one at
all: the sm_100 clause accepts 32, 64 and 128 for the same kernel, and the
sm_120 FP4 branch accepts 32. If either is true, widening one clause upstream
removes the problem outright — which is why
[the report](../upstream/vllm/ISSUE-block-size.md) asks the question rather than
assuming.

The FlashInfer-side constraints are narrower. The text-only requirement
(FI-003) is a consequence of an sm_120-only dispatch table and does not apply
on sm_90 or sm_100, which have their own kernels. VL-013 / FI-004 is different
in kind: the vLLM half (dummy forwards writing block 0) is not sm_120-specific,
and whether it bites on another GPU depends only on whether that GPU's sparse
kernels read the null block for masked indices. sm_120's do.

**The generalisable lesson.** Three of the five defects behind these flags are
the same shape: a kernel's compile-time dispatch table meeting a model whose
shapes were chosen for different hardware. `TOPK` and `PAGE_BLOCK_SIZE` were
template parameters that the kernel only used for arithmetic; the block size
was assumed uniform across layers by code that already supported it varying.
If you are bringing a new large model up on a GPU its kernels were not written
for, look there first — and look at what the parameter is actually *used for*
inside the kernel before concluding it cannot be made runtime.

## 5. Memory that is not weights

- KV cache: the architecture estimate is 890 B/token of global KV plus FP8
  SWA. This does not validate long-context serving: the current presets use
  `--max-model-len 32768`, and longer contexts need quality and memory checks.
- CUDA graphs: the default capture ladder costs ~2.5 GiB per rank. Trim with
  `--cuda-graph-sizes` or `--max-num-seqs` if you need the last GiB.
- FlashInfer sparse-MLA workspace: fixed, ~0.5 GiB.
- `--gpu-memory-utilization`: the reference V4.1 presets use **0.93**. Profile
  allocations and capture on your machine before changing it.
- `--language-model-only` selects text serving. Verify the actual loaded
  allocation before assuming that the flag removes the vision tower's memory.

## 6. Startup time

The checkpoint is approximately 476 GiB; the recorded deployment pinned about
452 GiB of host RAM. The initial eager startup took about 285 seconds with a
warm OS page cache. Marlin repacking and FlashInfer JIT add work on the first
start; those figures are not a cold-start guarantee.

- Keep `VLLM_ENGINE_READY_TIMEOUT_S=3600`.
- Install `flashinfer-cubin` / `flashinfer-jit-cache` for your CUDA version so
  sm_120 kernels are prebuilt; point `FLASHINFER_WORKSPACE_BASE` and
  `VLLM_CACHE_ROOT` at a persistent disk so JIT/compile caches survive.
- Put the checkpoint on NVMe. Record the page-cache state when comparing
  startup times; available cache space changes as pinned allocations grow.

## 7. Things that do not help on this model

- Prefix caching aggressiveness: KV is so small that hit-rate barely changes
  memory; keep it on for TTFT, that is all.
- Quantizing the FP8 dense layers to FP4: saves < 5 GiB, costs accuracy in the
  attention path where the model is most sensitive.
- `--kv-cache-dtype fp8`: main KV is already FP4 and SWA KV is trained FP8.
- torch.compile: unsupported for this architecture in vLLM (custom ops);
  CUDA graphs are what you want and they are on by default.

## 8. Measuring

```bash
vllm bench serve --backend openai-chat --base-url http://127.0.0.1:8000 \
  --model deepseek-v4.1-flash --dataset-name random \
  --random-input-len 4096 --random-output-len 512 \
  --num-prompts 32 --max-concurrency 8 --save-result
```

Report TTFT, TPOT and output tokens/s at concurrency 1, 8 and 32, with and
without DSpark, and note the accepted-tokens-per-step from the server log
(`spec decode ... acceptance`). Prefill throughput is best measured with
16K-64K prompts and `--random-output-len 1`.

**Check correctness before you believe a throughput number.** On this hardware
"up" and "correct" came apart twice, and both times the server answered every
request promptly, at full speed, with the same token id forever — which decodes
to an empty string and reads like an empty response rather than a broken model.
A benchmark harness counts those tokens happily.

```bash
python deploy/verify.py          # exits non-zero if the model is not sane
```

It runs greedy continuations with one obvious answer, a teacher-forced
perplexity on held-out English (which also fails loudly when logits are
non-finite, because the JSON encoder refuses them), and one word problem
through the chat template.
