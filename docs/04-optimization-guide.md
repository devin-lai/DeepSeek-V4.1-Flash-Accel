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

- Check NUMA placement under concurrent GPU load. Eight local readers reached
  260.3 GB/s aggregate, versus 197.4 GB/s for remote readers. The old one-GPU
  51.3/51.1 GB/s comparison missed this contention. Exact-pinned workers were
  already allocating locally on the reference host. Strict `--numa-bind`
  remains off in the presets because the original allocator could exhaust one
  socket; an end-to-end gain from adding it has not been demonstrated.
- Use `DSV41_EXACT_PINNED=1` with plugin 0.2.0 to register persistent weight
  mappings at page granularity. Engram tables and scales occupy 188.83 GiB,
  versus 264 GiB with PyTorch's power-of-two rounding. The same change also
  removes expert-buffer padding. Reserve additional RAM for workers and IPC.
  [Allocator scope and tests](../vllm_dsv41_opt/README.md).

### Spill experts, never attention

If GPU memory is still short after Engram is on the host, the remaining
choice is which *routed experts* to spill. Attention, norms, routers, shared
experts and the LM head run for every token and are small (≈ 10 GiB total);
offloading them costs PCIe bandwidth for every token for no memory gain.

vLLM's UVA offloader makes this selectable: `--cpu-offload-params
w13_weight w2_weight` limits eligibility to the fused expert matrices; the
scales stay resident.

### Spill *decoder* experts (this model only)

The presets restrict offload to layers 20-39 with the `vllm_dsv41_opt` plugin
(`DSV41_OFFLOAD_LAYERS=20-39`). The earlier version of this section claimed
that CED prefill runs layers 0-19 only, so that spilling decoder experts would
cost nothing during prefill. **That is not what the deployed stack does.**
Kernel traces from 2026-09-15 show layers 20-39 executing on every prefill
chunk, with every offloaded expert of those layers read from host memory once
per chunk (about 0.85 GiB per layer per rank). Prefill and decode are both
bound by these reads; see [where the time goes](08-pcie-bound-serving.md).

The earlier eager-mode experiment measured **+9.5 % prefill throughput and
−8.7 % TTFT on 8K prompts** for `20-39` against vLLM's stock walk order at
12 GiB/rank, with `0-19` reproducing the stock numbers to within 0.03 %. Keep
it as a four-request measurement, not as a mechanism: the trace does not show
a reason for decoder placement to beat encoder placement, and the difference
has not been reproduced with the graph-enabled presets. What does follow from
the trace is that the *amount* offloaded, and the prefill chunk size, are what
move prefill.

On a plain decoder the restriction is worth nothing either way: V4-Flash
measured 55.7 against 56.1 tok/s with and without it.

### How much to spill

`tools/plan_memory.py` computes it from the checkpoint and your GPU size.
Rule of thumb for MXFP4 experts: each layer holds 6.9 GiB of routed experts
(with scales); per TP rank that is 6.9 / TP GiB. Spill the smallest number of
layers that leaves sufficient room for KV, CUDA graphs and workspace. The
planner defaults to a conservative 4 GiB reserve; the measured text presets
use smaller reserves and validate their cache capacity with actual requests.

## 2. Speculative decoding: DSpark

**Static DSpark is measured and available in `v41-flash-latency`.** It uses
five draft tokens, probabilistic draft sampling, and
`"enable_adaptive_verification": false`. The reference SM120
`DeepseekV41IndexerBackend` rejects device-decided query lengths required by
adaptive verification. Full CUDA graphs alone do not make that path supported;
do not remove the backend's support check.

The latency preset captures at most 96 query tokens (16 sequences × 6),
reducing graph memory, and moves another 1.3 GiB/rank of expert weights onto
the GPUs compared with the 12 GiB preset. This is a combined configuration
change, not an isolated measurement of DSpark's contribution. Acceptance and
speed depend on the generated text; random-token inputs and ordinary prompts
are reported separately in the [optimization report](../benchmarks/results/2026-09-14-v41-optimization.md).

DSpark uses the checkpoint's existing drafter and unchanged target weights.
Small continuation and log-probability probes pass; full task-quality and
stochastic-distribution equivalence remain unmeasured. Test your own prompts
and sampling settings before drawing capacity or quality conclusions.

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
