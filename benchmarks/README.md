# Benchmark evidence and reproduction

[Project overview](../README.md) · [Contributing](../CONTRIBUTING.md)

## Published V4.1 comparison

| Artifact | What it contains |
| --- | --- |
| [CUDA graph results](results/2026-09-14-v41/kit-v41-graphs-bench.json) | Rounded throughput and median latency summaries from `vllm bench serve` |
| [Eager baseline](results/2026-09-14-v41/kit-v41-preset-bench.json) | The corresponding eager-mode summaries |
| [Graph-enabled sanity probes](results/2026-09-14-v41/kit-v41-graphs-verify.json) | Generated continuations, short-passage perplexity, and chat answer |
| [Graph investigation](results/2026-09-14-v41-cudagraphs.md) | Experiment configurations and recorded outcomes |
| [Initial serving report](results/2026-09-14-v41-first-serve.md) | Earlier eager-mode deployment and offload results |

The 2026-09-14 comparison uses one 8× RTX 5090 D machine, 503 GiB host RAM,
TP8 + expert parallelism, Marlin, Engram on CPU, and 12 GiB/rank offloaded.
The graph-enabled run restricts offload to decoder layers 20–39. See the
[engineering report](../docs/07-engineering-report.md#2-what-v41-flash-costs-today)
and [text preset](../deploy/presets/v41-flash.env) for configuration details.

These committed files contain summaries and selected probe outputs, not a
complete per-request trace or all original diagnostic logs. The original
1/8/32-concurrency cases used 4/16/64 requests, with random 1,024-token inputs,
128-token outputs, and `ignore_eos`. The 8K prefill case used four requests,
two concurrent requests, and one output token. Repeated-trial uncertainty is
not available for the headline comparison.

## Reproduce against a running server

Start the reference server using the [quickstart](../README.md#quickstart).
In another shell, from the repository root:

```bash
source /data/venvs/vllm-dsv41/bin/activate
export MODEL=/data/models/DeepSeek-V4.1-Flash
RUN_DIR=$(mktemp -d /data/v41-bench-XXXXXX)
python deploy/verify.py --json "$RUN_DIR/verify.json"

# Original single-request case. Inspect verify.json before measuring speed.
vllm bench serve \
  --backend vllm --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions --model deepseek-v4.1-flash \
  --tokenizer "$MODEL" --trust-remote-code \
  --dataset-name random --random-input-len 1024 --random-output-len 128 \
  --num-prompts 4 --max-concurrency 1 --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el --save-result \
  --result-dir "$RUN_DIR" --result-filename c1_1k_128.json
```

For the other original cases, use `(input, output, prompts, concurrency)` =
`(1024, 128, 16, 8)`, `(1024, 128, 64, 32)`, and `(8192, 1, 4, 2)`;
use a distinct result filename for each. Client concurrency can exceed the
preset's `MAX_SEQS=16`; those measurements include queueing.

For an eager comparison, copy `deploy/presets/v41-flash.env` to a local preset
named `v41-flash-eager.env`, change only `ENFORCE_EAGER=0` to
`ENFORCE_EAGER=1`, stop the graph-enabled server, and restart with
`PRESET=v41-flash-eager`. The preset assigns this value itself, so setting
`ENFORCE_EAGER=1` in the caller's environment does not override it. Validate
outputs again and save to a separate run directory.

For a new performance claim, use enough requests to reach stable behavior,
repeat each case at least three times, record warm-up and cache policy, and
publish the distribution. Treat the short original cases as a reproduction
target rather than a sufficient production capacity test.

## What every new report should include

- Model ID, revision, weight format, tokenizer revision, and changed model files.
- Project commit; vLLM, FlashInfer, PyTorch, CUDA, and driver versions; patches.
- GPU model/count/VRAM, CPU, host RAM, PCIe/NVLink/P2P topology, and memory use.
- Exact launch and benchmark commands, environment settings, context cap,
  server sequence cap, offload placement/budget, and graph mode.
- Dataset or prompt generator, seed if available, request count, input/output
  lengths, concurrency, streaming, EOS behavior, warm-up, and cache state.
- All request successes/failures, output throughput, TTFT, TPOT, latency
  percentiles, and individual run files. Explain any discarded samples.
- Generated-output checks and task-quality results appropriate to the claim.
  The small `verify.py` probes alone cannot establish quality parity.

## Comparison rules

Compare the same model revision, quantization, hardware, workload, and harness.
Change one variable for an ablation, or enumerate every difference. State
whether a baseline is upstream stock, locally patched, or a prior preset.

Output tokens/s counts generated tokens across the workload; total tokens/s
also includes prompt tokens. Neither is interchangeable with per-user decode
speed. Keep V4-Flash measurements separate from V4.1-Flash measurements.
The custom `deploy/bench.py` and `vllm bench serve` results must not be merged
into one comparison without reconciling their measurement behavior.

For reports of lower-bit expert formats, publish full-model quality and actual
kernel/runtime measurements before presenting them as a deployable speedup.
Expert-output error and calculated memory fit are narrower research results.
