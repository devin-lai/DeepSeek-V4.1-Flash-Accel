# Measured on 8x RTX 5090, 2026-09-13

Machine: 2x Xeon Gold 6530, 503 GiB DDR5, 8x RTX 5090 (31.4 GiB usable each),
PCIe Gen5 x16, no NVLink, GPU P2P disabled. vLLM main @ 8c1d1c297 (cu130,
torch 2.13.0), driver 595.71.05.

## 1. Host-memory access, the thing every offload design depends on

`scripts/bench/uva_bench.py`, one GPU at a time, pinned host tensors read
through UVA (zero-copy), `numactl` used to place the pinned pages.

| GPU (NUMA) | pinned pages on | Engram-style gather, 3072 rows x 256 B | 6 experts (70 MB) via UVA | 96 experts via UVA | plain H2D, 1 GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 2 (node 0) | node 0 (local) | 19 us | 51.3 GB/s | 51.3 GB/s | 56.3 GB/s |
| 3 (node 0) | node 1 (remote) | 18 us | 51.1 GB/s | 51.2 GB/s | 56.1 GB/s |
| 6 (node 1) | node 1 (local) | 17 us | 51.3 GB/s | 51.3 GB/s | 56.3 GB/s |
| 7 (node 1) | node 0 (remote) | 18 us | 51.3 GB/s | 51.3 GB/s | 56.3 GB/s |

Two results that shaped the deployment design:

- **UVA reads run at 51 GB/s, ~91 % of the plain pinned H2D rate.** Zero-copy
  expert reads are not a second-class path on PCIe Gen5; they cost about what
  an explicit copy costs, without the staging buffer.
- **NUMA placement of pinned pages does not matter here** (51.1 vs 51.3 GB/s).
  This contradicts the usual advice and it is the reason the working
  configuration does *not* use `--numa-bind`: binding gains nothing and it
  caps each socket's four workers at that node's 251 GiB, which is not enough
  for Engram plus offloaded experts (see section 3).
- An explicit `index_select` on the CPU then `.pin_memory().to(cuda)` reaches
  only 7-8 GB/s: the CPU-side gather, not the bus, is the bottleneck. Let the
  GPU do the gathering.

## 2. MXFP4 experts + UVA offload on sm_120, measured with GPT-OSS-120B

GPT-OSS-120B is the closest available stand-in for V4.1-Flash's expert path:
MXFP4 routed experts that vLLM runs through the same Marlin kernel on sm_120.
Random 1024-in / 256-out, `vllm bench serve`.

| config | GPUs | offloaded/rank | out tok/s @c1 | @c8 | @c32 | TTFT c1 | TPOT c1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP4, no offload | 0-3 | 0 | 251.0 | 1116.9 | 2607.7 | 54.6 ms | 3.39 ms |
| TP8, no offload | 0-7 | 0 | 250.6 | 1229.2 | 2564.2 | 51.8 ms | 3.79 ms |
| TP4, offload 4 GiB | 4-7 | 4 GiB | 113.8 | 235.8 | 392.5 | 249.9 ms | 7.46 ms |
| TP2, offload 6 GiB | 0-1 | (failed) | | | | | |
| TP2, offload 10 GiB | 2-3 | 10.12 GiB | 82.9 | 157.2 | 252.4 | 384.9 ms | 10.18 ms |
| TP2, offload 14 GiB | 4-5 | 14.06 GiB | 57.4 | 93.1 | 169.8 | 620.4 ms | 15.97 ms |
| TP2, offload 10 GiB, upper layers only | 6-7 | 10.12 GiB | 75.5 | 122.8 | 202.1 | 391.9 ms | 12.29 ms |
| TP2, offload 10 GiB, no `--numa-bind` | 0-1 | 10.12 GiB | 74.9 | 157.4 | 259.3 | 445.1 ms | 11.51 ms |

What this says:

- **TP8 buys nothing over TP4 for this model size** on a PCIe box: 250.6 vs
  251.0 tok/s single stream. Communication, not compute, is the decode cost,
  and doubling the group does not reduce it.
- **Offload is expensive and scales with how much you offload.** TP4 goes from
  251 to 114 tok/s single-stream when only 4 GiB/rank moves to host. Going
  from 10 to 14 GiB/rank on TP2 costs another 31 % (82.9 -> 57.4).
  Offloaded bytes are paid *per decode step*, so the tax grows with
  concurrency too: at c32, offload 10 GiB gives 252 tok/s where the same GPUs
  without offload would be far higher.
- **`--numa-bind` made no measurable difference** (74.9/157.4/259.3 bound vs
  82.9/157.2/252.4 unbound, different GPU pairs, same offload). Consistent
  with the micro-benchmark.
- Restricting offload to the upper half of the layers was *not* a win for
  GPT-OSS (75.5 vs 82.9 tok/s at c1). GPT-OSS is a plain decoder, so upper and
  lower layers cost the same; the idea only pays off on an encoder-decoder
  model like V4.1-Flash where prefill skips the decoder half. It is the wrong
  control to draw a V4.1 conclusion from, and it is reported here only to show
  the mechanism works.

## 3. Why the first DeepSeek-V4.1-Flash boots failed

- Engram host offload works exactly as documented: each rank pinned
  **11.80 GiB per table, 2 tables, 23.6 GiB/rank, 189 GiB total**, in ~8 s.
- With `--numa-bind`, the four workers of a socket must fit Engram plus
  offloaded experts plus loader buffers inside that node's 251 GiB. The kernel
  OOM-killed a worker with
  `oom-kill:constraint=CONSTRAINT_MEMORY_POLICY,nodemask=0`
  (`shmem-rss` 61.5 GiB on the victim). Dropping `--numa-bind` is the fix, and
  section 1 shows it costs nothing.
- vLLM picks `DEEPGEMM_MXFP4` for DeepSeek-V4 on sm_120, where GPT-OSS gets
  `MARLIN`. The DeepGEMM path OOM'd on the GPU while *constructing* the expert
  weights (`mxfp4.py` `create_weights`), before any offload could take effect,
  at 30.1 GiB of 31.4 GiB. MXFP4 kernels round the per-rank intermediate size
  up (TP8 gives 288 columns, padded to 384), so the resident expert bytes are
  ~1.3x the checkpoint's. The ladder therefore forces
  `--kernel-config '{"moe_backend": "marlin"}'`.

## 4. DeepSeek-V4-Flash-NVFP4 matrix: not run

All 13 configurations failed at startup with
`ValueError: No common block size for 128.` -- `--block-size 128` is required
by the SM120 sparse-MLA backend for V4.1 but rejected for the V4 NVFP4
checkpoint. One flag, to be re-run.

## 5. Fitting DeepSeek-V4.1-Flash itself: what actually constrains it

The checkpoint finished downloading at 14:51 (48/48 shards, 476 GiB on disk).
Eleven boot attempts followed. Each failed for a different, identifiable
reason, and together they pin down the real constraints.

### 5.1 Host RAM, not GPU memory, is the binding constraint

Pinned host allocations are **rounded up to a power of two** by PyTorch's
caching host allocator, and the rounding is not configurable
(`PYTORCH_CUDA_ALLOC_CONF` / `PYTORCH_HOST_ALLOC_CONF` `roundup_power2_divisions`
have no effect on it). Measured on this box:

| requested | host RAM consumed |
| ---: | ---: |
| 8.00 GiB | 8.00 GiB |
| 3.00 GiB | 4.00 GiB |
| 11.80 GiB | 16.00 GiB |
| 0.70 GiB | 1.00 GiB |
| 0.35 GiB | 0.50 GiB |

Consequences for this model:

- **Engram costs 264 GiB of host RAM, not 189 GiB.** At TP8 each rank pins one
  11.5 GiB shard plus a 0.36 GiB scale per table; those round to 16 + 0.5 GiB,
  and there are 8 ranks x 2 tables.
- **Offloaded experts cost ~1.43x their budget**, since each expert tensor
  (0.70 and 0.35 GiB per layer per rank) rounds to 1.0 and 0.5 GiB.
  *(Later correction: 1.43 is specific to these tensor sizes, not a constant.
  At EP8 on V4.1 the same rounding gives 1.78x -- 0.56 and 0.28 GiB rounding to
  1.0 and 0.5. `tools/plan_memory.py` now walks the buffers rather than
  applying a factor; see `2026-09-14-v41-first-serve.md`.)*
- With 503 GiB of RAM and no swap, the offload budget must stay at or below
  about **14 GiB per rank**. Above that the kernel OOM-killer takes a worker.

Also measured: the offloader's own double copy. Stock vLLM does
`p.data.to("cpu")` then `.pin_memory()`, holding two host buffers per tensor.
The `vllm_dsv41_opt` plugin allocates the pinned buffer once and copies into
it, halving the peak. Without it, `--cpu-offload-gb 24` at TP8 drove `Shmem`
to 462 GiB of 503 GiB before the engine finished loading.

### 5.2 GPU side: expert parallelism removes the MXFP4 padding tax

MXFP4 kernels round the per-rank intermediate size up. Expert bytes per rank
for the 40 layers:

| layout | intermediate per rank | padded to | expert bytes/rank |
| --- | ---: | ---: | ---: |
| TP8, Marlin / DeepGEMM / TRT-LLM | 288 | 384 (1.33x) | 44.8 GiB |
| TP8, Triton | 288 | 320 (1.11x) | 37.4 GiB |
| TP4, Marlin | 576 | 640 (1.11x) | 74.7 GiB |
| **EP8, whole experts** | **2304** | **2304 (1.00x)** | **33.6 GiB** |

A rank can hold roughly 26 GiB of weights after KV, CUDA graphs and workspace.
So TP8+Marlin needs ~22 GiB/rank offloaded (host RAM says no more than 14),
while **EP8 needs only ~10-12 GiB/rank**, which fits. `--enable-expert-parallel`
is therefore not a throughput tweak on this box; it is what makes the model
loadable at all.

### 5.3 Pipeline parallelism is unusable for this model

TP4 x PP2 loads and profiles fine, then every PP-stage-1 worker raises
`ValueError: DeepSeek V4 vision MoE routing requires input_ids.` V4.1's MoE
gate keeps separate expert-selection biases for text and image tokens, so it
needs `input_ids`; vLLM does not forward them across a pipeline boundary. This
is independent of `--language-model-only` (tried both ways).

### 5.4 The remaining blocker: no SM120 sparse-MLA decode kernel for this shape

With EP8 + Marlin + 12 GiB/rank offload the engine loads all 476 GiB, offloads
12.39 GiB/rank, pins Engram, profiles, and then dies during CUDA-graph capture:

```
ValueError: SM120 sparse-MLA has no decode kernel for this shape:
num_tokens=8, num_heads=8, topk=1152, d_qk=512, page_block_size=32,
model_type=1, extra_topk=0
```

Reading `flashinfer/mla/_sparse_mla_sm120.py`, both decode paths require
`page_block_size == 64`, and the DSv4 table only contains
`(num_heads, topk)` with `topk` in {128, 192, 256, 512, 1024}. vLLM hands the
kernel `page_block_size = 32` and `topk = 1152` (= 2 x `index_topk` + the
128-token window, for the ratio-2 CSA2 layers). Note the startup validation
checks a *different* shape (the SWA specialisation, `(8, 128)`), which is why
the run gets all the way to graph capture before failing.

This is an upstream gap in vLLM main @ 8c1d1c297 with flashinfer 0.6.18.post1,
not a configuration mistake. Probes of the two knobs that move those numbers
(`--block-size`, `--kv-cache-dtype`, and shrinking `index_topk` via
`--hf-overrides`) are in `benchmarks/results/probe-*`.


## 6. DeepSeek-V4-Flash-NVFP4 matrix, 8 GPUs, 2026-09-13 21:03-21:50

All eight configurations booted (FlashInfer autotune disabled). Output tok/s.

| config | c1 | c8 | c32 | 8K prefill total | 16K total | TPOT c1 | startup | peak Shmem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP8 baseline | 65.7 | 298.6 | 1079.9 | 14807.9 | 12811.3 | 7.64 ms | 180 s | 0.4 GiB |
| TP8 + EP | 117.9 | 580.4 | 1385.3 | 14498.5 | 12719.0 | 7.84 ms | 120 s | 0.4 GiB |
| TP8 + EP + numa-bind | 118.0 | 580.3 | 1375.5 | 14599.1 | 12725.8 | 7.83 ms | 120 s | 0.4 GiB |
| TP8 + EP, offload 6 | 56.1 | 148.0 | 268.6 | 5136.0 | 4440.2 | 16.61 ms | 125 s | 48.4 GiB |
| TP8 + EP, offload 6, upper layers | 55.7 | 149.8 | 271.9 | 4977.3 | 4543.0 | 16.53 ms | 125 s | 48.4 GiB |
| TP8 + EP, offload 6, stock two-copy | 56.0 | 148.8 | 266.2 | 5055.8 | 4442.7 | 16.63 ms | 130 s | 48.4 GiB |
| TP8 + EP, offload 12 | 36.2 | 84.1 | 148.9 | 2842.3 | 2558.4 | 25.52 ms | 130 s | 96.4 GiB |
| TP8 + MTP (1 token) | 101.6 | 542.9 | 586.6 | 3857.5 | 12013.0 | 8.79 ms | 220 s | 0.4 GiB |

Model weights are 20.78 GiB per rank; KV cache 5.78 GiB per rank at the
baseline. Offloaded host bytes track `8 x --cpu-offload-gb` exactly
(48 -> 48.4 GiB, 96 -> 96.4 GiB), so for this model the offload path costs
1:1 in host RAM.

### Null results worth recording

- `--numa-bind` is within noise of unbound (118.0 vs 117.9 tok/s).
- Restricting offload to the upper half of the layers is within noise
  (55.7 vs 56.1 tok/s): V4-Flash runs every layer during prefill, so there is
  no encoder half to protect. This is a V4.1-only optimisation.
- The plugin's single-copy offload matches the stock two-copy path on both
  throughput and peak `Shmem`.

### Earlier failure, now explained

Every run in this matrix failed on the first attempt after ~40 minutes with
`RuntimeError: ... gloo ... Connection closed by peer`. The cause is
`kernel_warmup` -> `flashinfer_autotune`, which JIT-compiles sm_120 kernels
inside a collective barrier; ranks finish minutes apart and a peer drops.
`--kernel-config '{"enable_flashinfer_autotune": false}'` fixes it and cuts
startup to 120-180 s.
