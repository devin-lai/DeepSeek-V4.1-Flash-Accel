# Exact pinned weights, static DSpark, and wider batching on 8× RTX 5090

This experiment extends the repository's patched, CUDA-graph-enabled V4.1
server. It delivers two text presets without changing checkpoint weights or
quantization: `v41-flash-latency` for interactive generation and
`v41-flash-throughput` for concurrent work. This report compares the complete
configurations; it does not attribute the combined gain to one flag.

## Configuration and method

One dedicated host: eight RTX 5090 D GPUs (the RTX 5090 family used throughout
this repository), 32 GB each, no NVLink, PCIe Gen5 ×16, two Xeon Gold 6530
sockets, 503 GiB host RAM, no swap. GPUs 0–3 are local to NUMA node 0 and 4–7
to node 1. The stack is vLLM `8c1d1c2974ee42757ee2e93cc898932edfd9d265`
plus repository patches, FlashInfer 0.6.18.post1, PyTorch 2.13 + cu130,
CUDA 13.2 toolkit, driver 595.71.05. Existing model shards were reused.

Common settings: TP8 + expert parallelism, Marlin, CPU Engram, offload only
`w13_weight` / `w2_weight` in decoder layers 20–39, block size 64, text only,
32,768-token context cap, chunked prefill with 2,048 batched tokens, CUDA
graphs enabled, OMP_NUM_THREADS=8. Strict `--numa-bind` is off.

| Setting | Previous `v41-flash` | Latency preset | Throughput preset |
| --- | ---: | ---: | ---: |
| Exact pinned allocator | Off | On | On |
| Requested expert offload, GiB/rank | 12 | 11 | 9 |
| GPU memory utilization | 0.93 | 0.95 | 0.95 |
| Maximum active sequences | 16 | 16 | 32 |
| Speculative decoding | None | Static DSpark, 5 drafts | None |
| Graph capture sizes | vLLM default | Explicit sizes through 96 | vLLM default |
| Approximate shared host RAM | 453 GiB | 279 GiB | 264 GiB |

The previous preset already includes the single-copy offloader, decoder-only
placement and attention/cache patches. It is not upstream stock vLLM or eager
execution. The latency preset fixes adaptive verification off and uses
probabilistic draft sampling. See the [preset files](../../deploy/presets/).

The primary comparison uses three sequential trials per case, seeds 0/1/2,
temperature 0, `ignore_eos`, and a fresh cache salt for every trial. Each CLI
invocation performs its initial test request before timing. Workload order is
fixed, not randomized; trials within a configuration share a server process.
These are short reproducibility tests, not independent machine replicates or
production capacity estimates. Throughput includes prefill and queueing.

| Workload | Input tokens | Output tokens | Requests / concurrency |
| --- | ---: | ---: | ---: |
| Random c1 | 1,024 | 128 | 4 / 1 |
| Random c8 | 1,024 | 128 | 16 / 8 |
| Random c32 | 1,024 | 128 | 64 / 32 |
| Prefill | 8,192 | 1 | 4 / 2 |
| Interactive c1 | 58–72 after chat template | 256 | 8 / 1 |
| Interactive c8 | Same eight prompts | 256 | 8 / 8 |

The [interactive dataset](../workloads/interactive.jsonl) contains original
code and prose prompts. Thinking is disabled in the model's chat template.
It is a performance workload, not a scored application benchmark. The harness
rejects failed requests, missing output tokens and invalid timing metrics;
a successful CLI exit alone is insufficient.

## Measured performance

Arithmetic mean of three trials, with the observed minimum–maximum in parentheses.
These ranges are not confidence intervals. Changes use the ratio of unrounded means.
Output tok/s unless the row explicitly says total tok/s.
All **936 timed requests** completed with the requested output-token counts and zero failures.

| Workload | Previous preset | Latency preset | Throughput preset |
| --- | ---: | ---: | ---: |
| Random c1 | 33.6 (33.0–34.0) | 65.0 (52.4–76.8) | 39.5 (39.2–39.9) |
| Random c8 | 91.0 (88.6–93.8) | 114.4 (107.6–120.7) | 113.2 (111.4–114.9) |
| Random c32 | 115.0 (112.8–116.6) | 135.0 (126.1–142.9) | 178.2 (177.8–178.8) |
| 8K prefill (total tok/s) | 2429.8 (2418.4–2435.7) | 2587.5 (2571.0–2596.3) | 3008.4 (2987.0–3020.1) |
| Interactive c1 | 39.8 (39.7–39.8) | 71.4 (70.2–72.1) | 46.0 (46.0–46.0) |
| Interactive c8 | 110.7 (110.1–111.3) | 155.9 (154.9–156.6) | 139.7 (138.5–140.8) |

| Workload | Latency change vs previous | Throughput change vs previous |
| --- | ---: | ---: |
| Random c1 | +93.3% | +17.6% |
| Random c8 | +25.7% | +24.4% |
| Random c32 | +17.4% | +55.0% |
| 8K prefill (total tok/s) | +6.5% | +23.8% |
| Interactive c1 | +79.4% | +15.6% |
| Interactive c8 | +40.9% | +26.3% |

Mean of the three trial medians, in milliseconds. The c32 baseline allows only
16 active sequences and includes queueing; the throughput preset allows 32.

| Workload | Previous TTFT / TPOT | Latency TTFT / TPOT | Throughput TTFT / TPOT |
| --- | ---: | ---: | ---: |
| Random c1 | 688.3 / 24.26 | 647.8 / 9.68 | 538.3 / 21.10 |
| Random c8 | 2398.3 / 68.86 | 1715.2 / 46.97 | 1923.4 / 55.71 |
| Random c32 | 19545.0 / 124.41 | 15183.0 / 84.20 | 2733.2 / 157.15 |
| 8K prefill (total tok/s) | 6727.5 / 0.00 | 5938.9 / 0.00 | 5432.4 / 0.00 |
| Interactive c1 | 288.3 / 24.10 | 228.7 / 12.89 | 234.7 / 20.97 |
| Interactive c8 | 1187.0 / 67.92 | 1035.5 / 43.32 | 935.4 / 53.81 |

Wider batching raises aggregate throughput and reduces initial queueing, while
c32 token latency is higher than in the old preset. Choose the latency preset
for interactive generation and the throughput preset for concurrent batch work.
The new gains must not be multiplied by the earlier graph/eager speedup.

[Full-precision comparison](2026-09-14-v41-optimization/comparison.json) ·
[Previous traces](2026-09-14-v41-optimization/baseline/bench/) ·
[Latency traces](2026-09-14-v41-optimization/latency/bench/) ·
[Throughput traces](2026-09-14-v41-optimization/throughput/bench/).

## Sanity results

| Preset | Expected-word continuations | Passage perplexity | Chat answer | 32K and 8×8K cache probes |
| --- | ---: | ---: | --- | --- |
| baseline | 5/6 | 2.737 | $8 | Not repeated in this control run |
| latency | 6/6 | 2.978 | $8 | 9/9 finite and correct code |
| throughput | 5/6 | 2.527 | $8 | 9/9 finite and correct code |

The installed 0.2.0 wheel passed **9 GPU tests**; the benchmark and planner
passed **4 CPU unit tests**. [Validation record](2026-09-14-v41-optimization/validation.json) ·
[Versions and checkpoint header hashes](2026-09-14-v41-optimization/environment.json).

Five additional requests with the identical passage (78 scored tokens) on the
unchanged final server produced perplexities **2.695–2.966**, all finite,
compared with 2.980 in its startup probe. Each used a fresh cache salt,
`temperature=0`, `max_tokens=1`, `prompt_logprobs=0` and `echo=true`, with
`PPL_TEXT` from `deploy/verify.py`. [Per-token log probabilities](2026-09-14-v41-optimization/latency/ppl-repeat.json)
are retained. The cause of this variation is unresolved. These measurements
support basic execution sanity, not numerical or task-quality parity; the
single baseline/latency perplexity difference should not be treated as proof
of either equivalence or a measured task-quality regression.

## Host memory: remove allocator padding

The original pinned allocator rounds large host allocations to powers of two.
Engram has separate table and scale buffers: 16 table shards round to 16 GiB
and 16 scale shards round to 0.5 GiB, totaling 264 GiB. The actual payload is
188.83 GiB. A nominal 12 GiB/rank expert offload budget spills 31 whole
parameters totaling 12.3926 GiB, which round to 23.5 GiB/rank. Thus the model
accounts for 264 + 8 × 23.5 = 452 GiB before about 1 GiB of shared IPC.
The planner's earlier 29-buffer estimate mistakenly charged GPU-resident
scales against the expert offload budget; that accounting is corrected.

Plugin 0.2.0 registers page-rounded anonymous mappings with `cudaHostRegister`
and constructs tensors over that storage. The mapping survives while any
storage alias or vLLM CUDA view remains. The change is scoped to large,
contiguous persistent weight allocations in Engram, expert offload and
post-repack restoration. Activations, IPC, small/noncontiguous buffers and
temporary transfers retain their existing allocation paths. CUDA registration
supports pinned, mapped host memory; see the
[CUDA runtime memory API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html).

At the same 12 GiB/rank expert budget, observed shared memory falls from
approximately 453 to 289 GiB: about **164 GiB recovered**. This intermediate
configuration also enables static DSpark, so it is not an allocator-only
throughput ablation. DSpark does not explain the removed host-buffer padding.
The final latency and throughput configurations move more experts onto the
GPUs and use approximately 279 and 264 GiB shared memory respectively.
Shared memory is not total process memory or peak required RAM.

The latency configuration also completed startup, benchmarks, one 32,000-token
request and eight concurrent 8K requests inside a **384 GiB cgroup v2 memory
limit**, with `oom=0` and `oom_kill=0`. Model file cache eviction was requested
before startup. The cgroup reclaimed file cache and reached its limit; that
is expected and is distinct from OOM. This kernel does not expose
`memory.peak`, so no peak figure is asserted. This was a limit experiment on
a 503 GiB machine, not validation on a physical 384 GiB machine. See the
[memory snapshot](2026-09-14-v41-optimization/latency-capped/memory.json).

## DSpark and graph memory

Adaptive verification failed at startup ([VL-014](../../docs/05-fault-inventory.md)): `DeepseekV41IndexerBackend` does not
support the device-decided query lengths required on SM120. The implementation
has narrower architecture gates for its variable-length indexer paths. We kept
that support check intact and tested static verification instead.

Static DSpark drafts five tokens using the checkpoint's existing drafter.
Capturing no more than 96 query tokens covers 16 sequences × (5 drafts + 1).
The measured graph pool falls from about 0.39 to 0.30 GiB/rank; the profiler's
reservation estimate falls from 0.94 to 0.72 GiB. Together with 0.95 GPU
utilization, this permits lowering the requested offload budget from 12 to
11 GiB/rank while retaining a measured 0.86 GiB/rank KV cache. The throughput
configuration has 0.64 GiB/rank KV cache. Those capacities correspond to
vLLM estimates of 3.76 and 2.93 simultaneous full 32K contexts, respectively;
the active-sequence caps of 16/32 do not promise that many full-length contexts.
Those controls were changed together, so their individual speed contributions
are not isolated.

The earlier 12 GiB DSpark configuration delivered 65–66 output tokens/s on
the interactive c1 case, but its 8K prefill throughput was about 2,389 total
tokens/s. The smaller capture/budget configuration improves that tradeoff.
Acceptance varies with generated text, even across nominally greedy runs;
raw acceptance counters are included. Speculative streaming can deliver
multiple tokens per event, so ITL and TPOT are not interchangeable.

## NUMA: concurrent traffic changes the conclusion

The earlier one-GPU experiment read host memory at 51.3 GB/s locally and
51.1 GB/s remotely. It did not establish that eight simultaneous GPUs were
NUMA-independent. The new benchmark allocates one 256 MiB pinned buffer per
GPU, verifies page placement, and runs 64 read/store kernels per graph,
four replays per timed trial, three trials, with all eight workers released
from a common barrier.

| Host placement | Sum of per-GPU median bandwidth |
| --- | ---: |
| Local to each GPU | 260.3 GB/s |
| Across sockets | 197.4 GB/s |

Local placement gives **31.9% more aggregate read bandwidth** in this test.
The serving workers' existing CPU affinities already place approximately
34.69 GiB of each latency worker's mapped buffers on its local node, with
negligible remote placement. Therefore this microbenchmark is not claimed
as a 31.9% serving improvement or evidence that adding `--numa-bind` helps.
[Raw bandwidth trials](2026-09-14-v41-optimization/numa-bandwidth/results.json) ·
[Worker placement](2026-09-14-v41-optimization/latency-capped/placement.json).
The fresh stock-allocation baseline is also mostly local: ranks 0–3 have
about 0.9 GiB remote each out of 56.5 GiB of mapped host buffers; ranks 4–7
have negligible remote allocation. This does not isolate NUMA as the cause
of any end-to-end performance difference.

## Correctness checks and evidence limits

The allocator GPU tests cover storage/view lifetime, a sliced UVA view,
FP32/BF16/uint8/FP8, exact allocation accounting and fallback layouts. Another
regression test checks that disabling UVA still installs vLLM's per-forward
transfer wrapper; previously the plugin could spend the offload budget before
that wrapper was installed. Exact allocation explicitly rejects disabling its
required single-copy path.

Every primary server is checked with `deploy/verify.py` before timing. Its
six expected-word continuations, short-passage perplexity and one chat answer
are sanity checks, not model-quality parity. A plausible answer without the
expected word can fail one continuation. Target weights are unchanged, but
this work does not prove equivalent output distributions under stochastic
sampling or scores on a full task suite.

The context probe places a verification code at the **end** of the prompt.
It tests cache capacity, execution shapes and finite output log probabilities;
it does not test long-range retrieval. The 1M advertised context, vision with
these new presets, hours-long traffic, and other hardware remain untested.

Excluded from performance comparisons: an early stale model alias caused
all requests to return 404 despite a zero CLI exit; an interactive attempt
lacked the custom-loader dependency; adaptive DSpark failed at startup; an
orchestration snapshot failed on this kernel's missing `memory.peak`; a
benchmark begun before the recovered sanity check was stopped. These are
recorded as failures, not converted into throughput claims. A helper naming
collision and a process-wait API incompatibility were also corrected; their
partial runs are excluded.

## Reproduce and extend

Use [the deployment runbook](../../deploy/README.md), then the
[six-case repeated protocol](../README.md#repeated-optimization-measurements).
The evidence directory contains full-precision summaries, per-request lengths,
TTFT/ITL/end-to-end timings, launch/benchmark commands, memory snapshots and
sanity outputs. Detailed traces use compact JSON formatting. Bulk `generated_texts` are omitted from serving traces;
`verify.json` and context probes retain their outputs. Run-directory paths
and process IDs are normalized or omitted. Timestamps in vLLM result files
use the server's UTC+08 timezone. Source hashes identify the shipped plugin
and reproduction scripts. The final latency suite ran those public scripts;
intermediate experiments used helper wrappers with the saved CLI commands.

Further work with a concrete validation target:

- Investigate prompt-logprob repeatability on this stack, then score a wider
  task set and stochastic sampling before claiming quality parity.
- Adaptive SM120 DSpark needs an indexer path honoring device-side lengths,
  graph replay and padding contracts, followed by attention/output parity
  tests. Removing the support gate is insufficient.
- Tune draft length and scheduler policy on representative prompt mixes and
  longer runs. The current static five-token choice is workload dependent.
- Profile per-rank expert activity before testing `round_robin` placement.
  The current strategy is `linear`; equal numbers of stored experts do not
  guarantee balanced active work. No placement speedup is claimed here.
- Test more GPU-resident expert placement only with measured KV-cache
  headroom, concurrency and longer outputs; no untested budget is a preset.
- Evaluate TP4 × PP2 after fixing the cross-stage `input_ids` contract, and
  compare end-to-end communication and throughput on this NUMA topology.
- Lower-bit expert formats and CPU expert kernels need full-model quality
  and serving measurements. Microbenchmarks cannot settle an engine ranking.

These results improve this repository's prior deployment on one machine.
They do not establish a global state-of-the-art ranking against other engines.
