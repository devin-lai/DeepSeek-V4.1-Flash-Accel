# Where the time goes: DeepSeek-V4.1-Flash on 8× RTX 5090 is PCIe-bound

Profiling notes and optimization exploration, 2026-09-15. Everything here was
measured on the reference machine with the shipped `v41-flash-latency` and
`v41-flash-throughput` presets (torch profiler traces, rank 0; one trial per
benchmark case unless stated). Numbers are from the raw evidence in
[`benchmarks/results/2026-09-15-pcie-bound`](../benchmarks/results/2026-09-15-pcie-bound/).

## 1. A decode step, kernel by kernel

Single request, static DSpark-5 (6 query tokens per step), `v41-flash-latency`.
A step takes about 45 ms on rank 0 and the GPU is busy for 44.7 ms of it, so the
CPU is not the bottleneck.

| what | per step | share | note |
| --- | ---: | ---: | --- |
| Marlin MoE GEMMs in the 14 offloaded decoder layers (20–33) | ~21 ms | ~46% | weights read from pinned host RAM through UVA: 0.3–1.6 ms per launch |
| all-reduce waits after those layers | ~5 ms | ~11% | NCCL kernel time spent waiting for the slowest rank |
| 88 all-reduces per step, pure collective cost | ~3.5 ms | ~8% | 37–40 µs each, NCCL `RING_LL` over host shared memory (GPUs have no peer access) |
| dense FP8 GEMMs (~230 launches, 12–30 µs each) | ~4 ms | ~9% | attention/indexer/o-proj projections at M = 6 |
| everything else (sparse-MLA attention, mHC, norms, routing, sampler) | ~11 ms | ~25% | sparse-MLA decode is 0.5 ms of it |

The 26 layers whose experts are resident spend 35 + 17 µs in Marlin per layer.
The shared expert runs on a second stream and is hidden behind the routed
experts.

Reading the same trace for the whole server rather than one rank: every layer
waits at its all-reduce for the rank whose activated experts happened to be
offloaded and numerous. Expert-parallel decode runs at the speed of the
slowest rank in every layer.

## 2. Prefill is the same story

An isolated 7,510-token prompt took 3.54 s (about 2.1k tok/s in four
2048-token chunks). Kernel time by category on rank 0:

| category | time | share |
| --- | ---: | ---: |
| Marlin MoE GEMMs | 1,633 ms | 46% |
| NCCL all-reduce (21 MB messages, mostly waiting for stragglers) | 1,373 ms | 39% |
| dense FP8 GEMMs | 92 ms | 2.6% |
| sparse-MLA prefill attention (`sparse_mla_prefill_mg_dual_kernel`) | 60 ms | 1.7% |

The Marlin time is almost entirely the 14 offloaded layers: 28 launches per
chunk at 3–46 ms each, while the 26 resident layers take 0.1–1 ms per launch.
**Decoder layers 20–39 execute on every prefill chunk**, and each chunk reads
every offloaded expert of those layers again (about 0.85 GiB per layer per
rank). This contradicts the earlier assumption in this repository that CED
prefill runs only layers 0–19; the profiler trace is the direct evidence
(Marlin launches for layers 20–33 reading host memory inside prefill chunks).
The placement of the offload budget therefore does not remove PCIe traffic
from prefill; the +9.5% recorded for `DSV41_OFFLOAD_LAYERS=20-39` in the
earlier eager-mode experiment is not explained by that mechanism and should be
re-measured before it is relied on.

## 3. The cost model

Host-memory reads by one GPU saturate at **51 GB/s** whether the kernel uses
plain 16-byte loads, TMA bulk copies or the copy engine (`scripts/bench/uva_bw.cu`,
1 GiB pinned, NUMA-local). With all eight GPUs reading at once the aggregate
was 260 GB/s in the earlier NUMA test, i.e. about 32 GB/s per GPU. So:

```
decode step ≈ (distinct offloaded experts hit per rank × 17.7 MB) / ~30 GB/s + ~10 ms
prefill      ≈ offloaded-expert reads roughly proportional to prompt tokens
               (measured; see the chunk-size result below) + all-reduce + compute
```

Expert hit rates follow from the routing: `s` (token, expert) slots over 384
experts activate a given local expert with probability 1 − (1 − 1/384)^s.

| concurrency | query tokens/step | slots | local experts hit (of 48) | bytes per offloaded layer |
| ---: | ---: | ---: | ---: | ---: |
| 1, DSpark-5 | 6 | 36 | 4.3 | 76 MB |
| 8, DSpark-5 | 48 | 288 | 25 | 450 MB |
| 32, no spec | 32 | 192 | 19 | 336 MB |
| 32, DSpark-5 | 192 | 1152 | 46 | 810 MB |

This is why static DSpark helps single-stream latency almost 2× (three tokens
per step for 76 MB of reads) but does not help the 32-stream case: the extra
draft tokens raise the hit rate from 39% to 95%, so bytes per step grow 2.4×
while tokens per step grow 3.3×, and the measured 32-stream throughput with
DSpark and 11 GiB/rank of offload was 156 tok/s against 178 tok/s for the
existing throughput preset (9 GiB/rank, no speculation).

Consequences:

- Every GiB/rank of offload costs about 30 ms per prefill chunk and, at high
  concurrency, 10–30 ms per decode step. GPU memory is the currency.
- Larger prefill chunks were expected to amortise the reads (an 8K prompt
  touches the offloaded layers four times with 2048-token chunks and once with
  8192-token chunks). Measured, they did not: 8192-token chunks changed 8K
  prefill throughput by +2.7% (section 4). Marlin's weight traffic evidently
  scales with the number of 64-row M-tiles it processes, not with the number
  of chunks, so prefill reads are roughly proportional to tokens either way.
- The access method does not matter; the bandwidth is a host/PCIe ceiling.
  A DMA prefetch pipeline would not help.
- Fewer, faster all-reduces do help: 88 per step, and every one of them is a
  synchronisation point that exposes the straggler rank.

## 4. Levers evaluated

One trial per case (`scripts/bench/v41_bench.py --repeat 1`), same six cases
as the [optimization report](../benchmarks/results/2026-09-14-v41-optimization.md),
same day, same checkpoint. The reference row is the shipped latency preset
measured in the same session.

| configuration | code/prose c1 | code/prose c8 | random c1 | random c8 | random c32 | 8K prefill | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `v41-flash-latency` (reference, same session) | 71.5 | 148.6 | 76.2 | — | 128.6 | 2,589 | 11 GiB/rank, 16 seqs, DSpark-5 |
| `v41-flash-throughput` (published, 3-trial mean) | 46.0 | 139.7 | 39.5 | 113.2 | 178.2 | 3,008 | 9 GiB/rank, 32 seqs, no speculation |
| throughput + DSpark-5, 11 GiB/rank, 32 seqs, graphs to 192 | 73.6 | 152.9 | 59.8 | 114.3 | 156.2 | 2,592 | **not adopted**: hit rate 39% → 95% at c32; 1.6 GiB of graph memory |
| latency + HostAR all-reduce (decode messages only) | 72.6 | 153.5 | 77.0 | 107.7 | 133.9 | 2,572 | +1.5% to +4% on decode; prefill's 21 MB messages fell back to NCCL |
| latency + HostAR, 32 MiB hand-off (prefill messages too) | 72.7 | 152.1 | 70.3 | 112.6 | 125.4 | 2,214 | **worse**: HostAR is slower than NCCL at 21 MB; keep the default 2 MiB hand-off |
| long-prompt: 8192-token chunks, 15.5 GiB/rank, 16 seqs, DSpark-5, util 0.95 | 54.7 | 110.9 | 47.6 | 80.8 | 100.8 | crashed | decode −25% as predicted for 4.5 GiB/rank more offload; the 8K requests hit CUDA OOM (see below) |
| long-prompt retry, same but util 0.90 | 53.3 | — | — | — | — | 2,659 | **not adopted**: prefill +2.7% only; chunk size does not amortise the reads the way the model predicted |

Output tok/s; "random" is the 1K-in/128-out synthetic case, "8K prefill" is
total tok/s at concurrency 2.

**Host-memory all-reduce (HostAR).** The GPUs in this machine have no peer
access, so NCCL stages every all-reduce through host shared memory with a
ring of dependent hops. HostAR, a vLLM general plugin from the
`pcie-blackwell-lab` research tree that shares this server, replaces small
tensor-parallel all-reduces with a single-kernel one-/two-shot reduction over
NUMA-local registered host memory. Installed as an editable copy into the
serving venv (inert unless `PBLAB_HOSTAR=1`), it took the 88 all-reduces per
decode step (10 KB–1.1 MB messages) and left the 21 MB prefill messages to
NCCL under its default 2 MiB hand-off. Decode gained 1.5–4% in one trial,
consistent with all-reduce being about 8% of a decode step. Raising the
hand-off to 32 MiB so that the prefill messages also went through HostAR made
8K prefill 14% slower: the two-shot host reduction moves each 21 MB message
through host memory several times and loses to NCCL's ring at that size, so
the default routing (small messages only) is the right one. HostAR sums in
FP32 in rank order and then rounds once, so its results can differ from
NCCL's in the last bit; the sanity checks passed unchanged. The plugin is
part of an unpublished research tree, so the presets do not depend on it;
`PBLAB_HOSTAR=1` in the launcher environment enables it where it is
installed.

**Larger prefill chunks.** With 8192-token chunks an 8K prompt reads the
offloaded layers once instead of four times, so the model predicts a large
time-to-first-token gain for long prompts, paid for with activation memory
(vLLM's profile run measured 2.39 GiB at 8192 tokens against 1.47 GiB at
2048) and therefore more offload and slower decode. The decode side behaved
as predicted (−25% at 15.5 GiB/rank). The prefill side did not run at
`gpu_memory_utilization 0.95`: vLLM handed the residual 3.33 GiB to the KV
cache, and the real 8K prefill then failed to allocate 320 MiB more than the
profiled peak, killing the engine (`torch.OutOfMemoryError`, all eight ranks).
The profile-run estimate of activation memory is not a safe bound at this
chunk size; the retry leaves headroom with utilization 0.90.

The retry completed and settled the question: 8K prefill reached 2,659 total
tok/s against 2,589 for the reference preset (+2.7%, within single-trial
noise) while single-stream decode fell from 71.5 to 53.3 tok/s. The
chunk-amortisation prediction in section 3 was wrong for this kernel path, and
the preset is not worth its decode cost.

**Speculative decoding at high concurrency.** The third row is the direct test
of the cost model. DSpark's acceptance stayed at 3.3 tokens per step, but at
32 sequences the 192 query tokens hit 95% of the offloaded experts instead of
39%, so each step read 2.4× the bytes while producing 3.3× the tokens, and the
two extra GiB/rank of offload (graph pool 1.6 GiB, KV 0.41 GiB) ate the
remainder. Speculation is a single-stream lever on this machine.

## 5. Precision

Configuration changes above do not alter the target model's arithmetic: the
checkpoint, MXFP4 experts, FP8 dense layers and BF16 activations are the
same in every row, and static DSpark with greedy verification is lossless at
temperature 0. Each server was checked with `deploy/verify.py` before timing
(six greedy continuations, teacher-forced perplexity on a short passage, one
chat answer).

| configuration | continuations | perplexity | chat |
| --- | ---: | ---: | --- |
| throughput + DSpark-5, 11 GiB/rank | 6/6 | 2.86 | `$8` |
| latency + HostAR all-reduce | 6/6 | 2.80 | `$8` |
| latency + HostAR, 32 MiB hand-off | 6/6 | 2.75 | `$8` |
| long-prompt, 8192-token chunks, util 0.95 | 6/6 | 2.81 | `$8` |
| long-prompt, 8192-token chunks, util 0.90 | 6/6 | 2.90 | `$8` |

The perplexity value moves between identical servers (2.6–3.0 across the
2026-09-14 runs), so it is a sanity check, not a quality measurement; see the
[optimization report](../benchmarks/results/2026-09-14-v41-optimization.md#correctness-checks-and-evidence-limits)
for the open question about run-to-run variance.

## 6. What to run

- **Interactive text:** keep `v41-flash-latency`. None of the variants tried
  beat it on single-stream or eight-stream code/prose output; HostAR adds
  1.5–4% where the plugin is installed.
- **Batch throughput:** keep `v41-flash-throughput` (no speculation). Adding
  DSpark at 32 sequences does not pay because it raises the offloaded-expert
  hit rate to 95%.
- **Long prompts:** do not trade decode for larger prefill chunks. 8192-token
  chunks bought +2.7% prefill throughput for −25% decode. If you do raise
  `MAX_BATCHED_TOKENS`, leave GPU memory headroom (utilization 0.90), because
  the profile-run activation estimate is not a safe bound at that chunk size
  and the engine dies on the first long request otherwise.
- **Every GiB/rank kept on the GPU is worth about 10–30 ms per decode step at
  high concurrency and 30 ms per prefill chunk.** The CUDA-graph ladder,
  activation peak and KV residual are the places to look; the 29-size ladder
  captured for 192 query tokens cost 1.6 GiB/rank and is the reason the
  speculative throughput experiment had to give up two more GiB of experts.

## 7. Not pursued, and why

- **Computing offloaded experts on the CPU.** The earlier measurement in
  `benchmarks/results/2026-09-13-cpu-experts.txt` puts gathered expert GEMV at
  110 GB/s of host bandwidth for the whole socket; that beats one GPU's PCIe
  link but not eight of them, and at 32 streams every rank reads every
  offloaded expert anyway.
- **Hot/cold expert placement.** Would only help at low concurrency (at
  c ≥ 8 nearly every expert is hit). Needs per-expert placement inside the
  fused `[E, …]` Marlin weight tensors; not attempted.
- **Native FP4 tensor-core MoE (FP8 activations).** Changes activation
  precision; Marlin keeps BF16 activations, which is at or above the
  checkpoint's reference numerics. Decode is not compute-bound here, so the
  gain would be limited to prefill.
- **FP8 KV cache.** The main KV is already FP4/FP8 packed; nothing to gain.
