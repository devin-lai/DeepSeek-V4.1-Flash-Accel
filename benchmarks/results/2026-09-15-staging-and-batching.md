# Exact expert-weight staging and serving optimization on 8× RTX 5090

This campaign extends the previous latency and throughput presets on the
same eight RTX 5090 D GPUs, two Xeon Gold 6530 sockets, 503 GiB RAM, and
PCIe-only interconnect. It preserves the checkpoint, MXFP4 expert weights,
FP8 dense weights and BF16 expert activations. It compares against the
already patched, CUDA-graph-enabled presets, not upstream stock vLLM.

## Faster short requests with a fixed KV budget

The `v41-flash-fast` preset requests 9.75 GiB/rank of expert offload, keeps
static DSpark-5, caps active sequences at eight and graph captures at 48 query
tokens, and explicitly reserves **256 MiB/rank for KV cache**. Staging is
disabled in this preset. Whole-tensor offload granularity moves approximately
1.055 GiB/rank of expert weights back onto GPU relative to the previous
11 GiB/rank latency preset. The checkpoint and arithmetic remain unchanged.

| Metric | Previous latency preset | Fast preset | Change |
| --- | ---: | ---: | ---: |
| Interactive c1 output tok/s | 69.86 (68.29–71.08) | 74.75 (72.52–76.06) | **+7.0%** |
| Interactive c8 output tok/s | 153.05 (150.42–156.62) | 168.49 (164.77–171.59) | **+10.1%** |
| 8K prefill c2, total tok/s | 2,581.45 | 2,780.41 | +7.7% |
| c1 median TPOT, ms | 13.59 | 12.76 | −6.1% |
| c1 median TTFT, ms | 223.97 | 267.08 | **+19.2%; regression** |
| c8 median TPOT, ms | 43.51 | 39.63 | −8.9% |
| c8 median TTFT, ms | 1,108.93 | 1,096.00 | −1.2% |
| 8K prefill median TTFT, ms | 5,951.84 | 5,524.77 | −7.2% |

Three trials per case, client workload seeds 0/1/2 and fresh cache salts.
All 60 timed candidate requests completed at the requested output length.
DSpark acceptance varies between runs; the recorded acceptance lengths are
part of the evidence. This is a whole-preset comparison, not an isolated
estimate of the value of each setting.

The fixed KV reservation **takes precedence over utilization-based memory
profiling**. Reported capacity is 35,852 tokens / 1.09 full 32K requests,
so eight active sequences does not mean eight full contexts fit at once.
The 32,000-token request and all eight concurrent 8K probes completed with
the expected trailing code and finite log probabilities. The preemption
counter remained zero; admission can still wait for cache space. GSM8K
remained 126/128, with no errors or truncations.

Use this preset for short interactive requests. Its first token arrives about
43 ms later in the c1 test, despite faster overall output. For long prompts,
the staging configuration below has higher prefill throughput and a larger
cache budget. Fixed KV bytes need new capacity checks on a different GPU,
stack, context cap or workload.

[Preset](../../deploy/presets/v41-flash-fast.env) ·
[Configuration](2026-09-15-staging-and-batching/latency-cache/configuration.json) ·
[Raw trials](2026-09-15-staging-and-batching/latency-cache/bench/) ·
[GSM8K outputs](2026-09-15-staging-and-batching/latency-cache/gsm8k.json) ·
[Context checks](2026-09-15-staging-and-batching/latency-cache/context/long-context.json).

## Isolated staging result

Enable `DSV41_MARLIN_STAGE_MIN_TOKENS=256` on the unchanged
`v41-flash-latency` configuration: 11 GiB/rank requested offload, static
DSpark-5, 16 active sequences, 2,048-token prefill chunks, utilization 0.95,
and graph captures through 96 query tokens. Three trials per case, with
seeds 0/1/2 and fresh cache salts:

| Workload | Previous latency preset | Same preset + staging | Change |
| --- | ---: | ---: | ---: |
| Interactive c1, output tok/s | 69.86 (68.29–71.08) | 70.45 (69.97–70.84) | +0.8%; ranges overlap |
| Interactive c8, output tok/s | 153.05 (150.42–156.62) | 153.45 (149.78–158.78) | +0.3%; ranges overlap |
| 8K prefill c2, total tok/s | 2,581.45 (2,567.42–2,591.71) | 3,524.84 (3,523.26–3,526.05) | **+36.5%** |
| 8K prefill median TTFT, ms | 5,951.84 | 4,340.73 | **−27.1%** |

Throughput entries are arithmetic means with observed minimum–maximum,
not confidence intervals. TTFT is the mean of the three trial medians.
Interactive cases use the existing eight code/prose prompts, 256 output
tokens, and concurrency 1 or 8. Prefill uses four requests, 8,192 input
tokens and one output token, with concurrency 2. All **120 timed requests**
across these two configurations completed with the requested output lengths.

[Baseline traces](2026-09-15-staging-and-batching/baseline-latency/bench/) ·
[Staging first trial](2026-09-15-staging-and-batching/stage-latency/bench/) ·
[Staging trials 2–3](2026-09-15-staging-and-batching/stage-latency/repeated/).
The latter directory numbers its repeats 1/2 but uses seeds 1/2; together
with the first trial's seed 0 these are three distinct trials.

## Wider batches with staging

The second candidate adds staging and raises `MAX_SEQS` from 32 to 64 on
`v41-flash-throughput`. It retains 9 GiB/rank requested offload, utilization
0.95, 2,048-token prefill chunks, no speculation and explicit graph sizes
1/2/4/8/16/24/32/40/48/56/64. It is a complete configuration comparison;
the contributions of staging and the wider scheduler are not isolated here.

| Workload / metric | Previous throughput preset | 64 sequences + staging | Change |
| --- | ---: | ---: | ---: |
| c32 output tok/s | 176.73 (174.12–178.96) | 203.06 (198.08–206.10) | +14.9% |
| c64 output tok/s | 176.96 (172.71–179.30) | 217.55 (214.48–219.70) | +22.9% |
| 8K prefill c2, total tok/s | 3,004.34 | 4,077.99 | +35.7% |
| c32 median TTFT, ms | 2,742.62 | 1,982.83 | −27.7% |
| c32 median TPOT, ms | 158.77 | 142.66 | −10.1% |
| c32 p99 request latency, ms | 31,880.49 | 26,671.84 | −16.3% |
| c64 median TTFT, ms | 24,143.70 | 4,470.29 | −81.5% |
| c64 median request latency, ms | 45,301.13 | 33,504.71 | −26.0% |
| c64 median TPOT, ms | 166.03 | 234.05 | **+41.0%; regression** |
| c64 p99 request latency, ms | 55,587.71 | 64,492.19 | **+16.0%; regression** |
| 8K prefill median TTFT, ms | 5,437.14 | 3,992.29 | −26.6% |

All latency columns average the corresponding statistic across three trials.
The c32 and c64 cases use 64 and 128 requests per trial, respectively, each
with 1,024 input and 128 output tokens. All **1,176 timed requests** across
these two configurations completed with the requested output lengths.

A larger active batch reduces queueing for the median request and raises
aggregate output, but divides service across more active streams. At c64,
per-token latency and p99 completion latency worsen. This candidate is for
aggregate batch throughput; it is not a universal latency improvement.

[Original throughput traces](2026-09-15-staging-and-batching/baseline-throughput/bench/) ·
[64-sequence traces](2026-09-15-staging-and-batching/batch64/bench/).

## Why staging helps

The original Marlin path reads packed expert matrices directly from mapped
host memory. A matrix used by multiple token tiles can cross PCIe repeatedly.
The new hook copies **one packed projection matrix** into temporary GPU
memory, then invokes the same Marlin GEMM. Later token tiles read that copy
from GPU memory. The down projection and next layer can reuse the allocator's
storage on the same CUDA stream.

Only marked UVA-offloaded weights are eligible. The threshold counts query
tokens before top-k expansion. Small decode batches and CUDA graph capture
retain direct UVA access, because copying all 48 local experts is expensive
when only a few are selected. An allocation failure falls back to the
original UVA GEMM; the first fallback is logged per worker. That path keeps
the computation valid but does not promise a speedup under memory pressure.

This revises the earlier blanket dismissal of staging in the
[PCIe profile](../../docs/08-pcie-bound-serving.md): measuring the raw link
bandwidth establishes a transfer ceiling, but does not establish how often
a GEMM rereads its weights. Copying once can reduce total traffic even when
it cannot increase the link's bandwidth.

### Kernel evidence

One NUMA-local GPU, synthetic packed weights at the V4.1 dimensions,
48 local / 384 global experts, top-6 routing, and BF16 activations. L2 is
flushed before each timed call; the flush is excluded from CUDA-event timing.
Each number is the median of seven measurements after warmup.

| Projection / query tokens | Direct UVA, ms | Staged, ms | Direct / staged |
| --- | ---: | ---: | ---: |
| Gate/up / 6 | 1.845 | 10.114 | 0.18× |
| Gate/up / 256 | 11.265 | 10.423 | 1.08× |
| Gate/up / 2,048 | 11.271 | 10.609 | 1.06× |
| Gate/up / 4,096 | 15.610 | 10.955 | 1.42× |
| Gate/up / 8,192 | 31.423 | 11.717 | 2.68× |
| Down / 8,192 | 16.322 | 5.911 | 2.76× |

All 48 projection/batch/placement combinations produced exactly equal BF16
values to the GPU-resident reference (zero relative and absolute tolerance).
A separate rerun compared the underlying 16-bit storage, including the sign
bit of zero: all 48 combinations were bitwise equal. That rerun used one
timing trial per combination and is used only for the equality check.
This is kernel parity at fixed inputs, not bitwise full-model generation
parity. The single-GPU synthetic routing
does not reproduce the serving model's routing skew or eight-GPU memory
contention; its speed ratios must not be substituted for serving results.

[Raw kernel trials](2026-09-15-staging-and-batching/marlin-staging.json) ·
[Bitwise checks](2026-09-15-staging-and-batching/marlin-bitwise.json) ·
[Reproduction script](../../scripts/bench/marlin_staging.py) ·
[Implementation](../../vllm_dsv41_opt/vllm_dsv41_opt/staging.py).

## Precision and validation scope

The original throughput preset and the 64-sequence staging candidate both
scored **126/128** on the same seeded GSM8K subset. They agreed on correctness
for every selected question, including the same two wrong answers; neither
run had request failures or 512-token truncations. The protocol is zero-shot,
thinking disabled, temperature zero, seed 20260915 and concurrency eight.
This is a paired subset regression check, not a full benchmark score or
a proof of general model equivalence.

[Dataset source](https://github.com/openai/grade-school-math) ·
[Harness and protocol](../../scripts/bench/gsm8k_compare.py) ·
[Baseline answers](2026-09-15-staging-and-batching/baseline-throughput/gsm8k.json) ·
[64-sequence answers](2026-09-15-staging-and-batching/batch64/gsm8k.json).

The 64-sequence candidate also completed a 32,000-token request and eight
simultaneous 8,192-token requests. All nine returned the code at the end of
the prompt with finite output log probabilities. The final cache-preemption
counter was zero, and no staging-allocation fallback was logged. Its startup
KV budget was 0.57 GiB / 85,192 reported tokens, compared with 0.64 GiB /
95,942 tokens for the original throughput preset. Sixty-four active sequences
therefore does not mean sixty-four simultaneous 32K contexts.

[Context probes](2026-09-15-staging-and-batching/batch64/context/long-context.json) ·
[Candidate configuration](2026-09-15-staging-and-batching/batch64/configuration.json).

The original and staged latency configurations both scored **22/24** on the
[small regression screen](../../scripts/bench/v41_quality.py). In the
baseline, the two failures gave correct numerical answers but included
explanations despite the requested number-only format. The screen uses exact
answer/format matching and is not a general reasoning evaluation.

Repeated teacher-forced probes scored 480 and 396 tokens per passage.
Baseline mean negative log-likelihood ranges were 0.5414–0.5517 and
0.5866–0.6110; staged ranges were 0.5574–0.5672 and 0.5835–0.5959.
All log probabilities were finite. These small probes do not resolve the
pre-existing repeatability issue or establish full-model quality equivalence.

[Baseline outputs](2026-09-15-staging-and-batching/baseline-latency/quality.json) ·
[Staged outputs](2026-09-15-staging-and-batching/stage-latency/quality.json).

## Rejected configurations and observed limits

- **64 sequences, staging, utilization 0.96:** increasing the batch preset's
  KV budget to 131,955 reported tokens left too little staging headroom.
  Every worker logged a fallback for a 566,231,040-byte packed matrix on a
  2,048-query batch. Across three trials, c32 output fell from 203.06 to
  185.81 tok/s and 8K prefill from 4,077.99 to 3,271.77 total tok/s. C64 output
  was essentially unchanged (217.55 versus 218.13), while median TPOT rose
  from 234.05 to 265.02 ms. The configuration is rejected. The fallback kept
  all 588 timed requests complete and maintained 126/128 on GSM8K, with no
  request errors or truncations. It preserves execution, not the staged
  performance gain. [Trials](2026-09-15-staging-and-batching/batch64-cache/bench/) ·
  [Quality check](2026-09-15-staging-and-batching/batch64-cache/gsm8k.json).
- **Routing-based layer placement:** the diagnostic counter recorded 618
  six-query, top-6 verification steps in every one of the 40 layers, after
  resetting startup counters. Calibration used eight independent code/prose
  prompts, including the CLI's initial request. The sum of per-layer mean
  busiest-rank expert counts was 67.477 for the current offloaded layers
  20–33 and 65.312 for the cheapest 14 layers: only **3.2% lower**. The latter
  selection is 6/7/10/16/20/24/25/26/27/28/29/30/31/33. This is a traffic
  proxy, not a measured latency improvement; the existing placement is kept.
  [Counters](2026-09-15-staging-and-batching/routing-profile/calibrated.json) ·
  [Calibration prompts](../workloads/offload-calibration.jsonl) ·
  [Diagnostic protocol](../../vllm_dsv41_opt/README.md#routing-diagnostics).
- **Selective copying for decode:** a Triton prototype marks the local
  experts referenced by the aligned Marlin routing, copies only their packed
  matrices, then invokes the same GEMM. It was **1.5–3.9% slower** than direct
  UVA in all eight projection/batch comparisons (6/12/24/48 queries, three
  timing trials each, one NUMA-local GPU). All 24 variant outputs were
  bitwise equal to the resident reference. This SM-driven copy is excluded
  from serving; it does not establish a limit for every possible transfer
  implementation. [Prototype](../../scripts/bench/marlin_selective.py) ·
  [Raw trials](2026-09-15-staging-and-batching/marlin-selective.json).
- **DSpark-3, 10.25 GiB/rank, utilization 0.96, 16 sequences, staging:**
  short requests worked, but the concurrent 8K case exhausted memory while
  allocating a 540 MiB staging tensor. The allocator reported 676.79 MiB
  reserved but unallocated and only 429.56 MiB device-free. All four timed
  prefill requests failed; this configuration is excluded from improvements.
  The allocation fallback was added after this failure.
- **DSpark-5, 9 GiB/rank, utilization 0.96, 8 sequences, 512-token chunks:**
  startup reported −0.38 GiB available for KV cache and failed. Smaller
  prefill chunks did not release enough memory to support that placement.
- **Dynamic draft schedules:** this pinned vLLM build forces the current
  model runner to piecewise graphs; full graph support for this option lives
  in the V2 runner. The support gate was retained.
- **Round-robin expert placement:** this build requires multiple expert
  groups and falls back to linear placement for this model. Merely setting
  the option would not change placement.

The measurements are short trials on one machine. They do not establish a
cross-engine ranking, full-context concurrency, long-duration reliability,
vision performance, or the advertised 1M context capability.

## Validation and reproduction

The GPU test suite passed **15 tests**, including mapped-storage lifetime,
staging eligibility, allocation fallback, CUDA-graph replay and snapshot/reset
of inference tensors. Five CPU tests passed for benchmark validation, answer
extraction and the memory planner. The standalone selective-copy script also
passed six bitwise checks through its public entry point.

[Validation record](2026-09-15-staging-and-batching/validation.json) ·
[Benchmark commands](../README.md#exact-staging-and-further-serving-tuning-2026-09-15) ·
[Launch and preset guide](../../deploy/README.md).

Stop the current server before switching presets. With the pinned stack and
plugin 0.3.0 installed, use `PRESET=v41-flash-fast`,
`PRESET=v41-flash-balanced`, or `PRESET=v41-flash-batch` with `deploy/serve.sh`.
The older presets remain available for matched baselines. Keep the diagnostic
routing profiler and development RPC disabled for speed measurements.

### Deployment check

After the experiments, plugin 0.3.0 and the public `v41-flash-fast` preset
were launched on the reference server's original `0.0.0.0:8000` listener.
At 2026-09-15 08:21 UTC, health returned 200, all six continuation probes
matched, and the development RPC endpoint was absent. An additional eight
sequential 256-token requests completed at **74.41 output tok/s**. This is
a deployment smoke test and is excluded from the three-trial comparison.

[Deployment record](2026-09-15-staging-and-batching/final-fast/deployment.json) ·
[Output verification](2026-09-15-staging-and-batching/final-fast/verify.json) ·
[Smoke trace](2026-09-15-staging-and-batching/final-fast/smoke/).
