# DeepSeek V4.1 Flash Accel

| **1.8× interactive output** | **55% higher concurrent throughput** | **42% less shared host RAM** |
| :--- | :--- | :--- |
| **39.8 → 71.4 tok/s** · interactive c1 | **115.0 → 178.2 tok/s** · random c32 | **453 → 264 GiB** · throughput preset |
| Static DSpark latency preset | Wider non-speculative batch | Exact pinned weight allocation |

**Measured on 8× RTX 5090, against this project's previous patched CUDA-graph
preset.** Three trials per workload, same prompts/seeds, unchanged checkpoint
and quantization. The latency preset uses approximately 279 GiB shared
RAM and also passed a 384 GiB cgroup limit experiment. These are short tests
on one 503 GiB host. [Full results, ranges and limits](benchmarks/results/2026-09-14-v41-optimization.md).

**Deploy and accelerate DeepSeek-V4.1-Flash with vLLM on consumer Blackwell GPUs.**
Deployment presets, CUDA graph and attention fixes, CPU offload tools, and
reproducible experiments for inference research. Text and image serving use an
OpenAI-compatible API.

TTFT is time to first token; TPOT is time per output token after the first.
Throughput is measured across all requests in each workload.

[Quickstart](#quickstart) · [How it works](#how-the-optimizations-work) · [Benchmarks](#measured-performance) · [Documentation](docs/README.md) · [简体中文](README.zh-CN.md)

An independent community project for [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash),
built on [vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer), and
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM). Experimental; not affiliated
with or endorsed by DeepSeek.

## Supported configuration

| Resource | Reference deployment |
| --- | --- |
| GPU | **8× RTX 5090, 32 GB each**, SM120, PCIe Gen5 ×16, no NVLink |
| Host RAM | **503 GiB installed**; approximately **279 GiB shared** with the latency preset or **264 GiB** with throughput. Latency also tested under a 384 GiB cgroup cap. |
| Checkpoint | Approximately **476 GiB / 510 GB**, plus space for the environment and caches |
| Placement | Engram on CPU; 11 GiB/rank decoder offload for latency, 9 GiB/rank for throughput; TP8 + expert parallelism |
| Context | **32,768 tokens** in the serving presets; the model's advertised 1M context is untested here |
| Text / images | Text benchmarked; simple image probe recorded; broader quality evaluation remains open |
| Stack | vLLM `8c1d1c2974ee42757ee2e93cc898932edfd9d265` + repository patches, FlashInfer `0.6.18.post1`, PyTorch 2.13 + cu130, CUDA 13.2 toolkit, driver 595.71.05 |

This is the measured configuration, not an established minimum. Other GPUs
and memory sizes need validation. See [hardware details](docs/02-hardware-topology.md)
and the [memory planner](tools/plan_memory.py) before allocating a machine.

## Quickstart

On a matching Linux NVIDIA host, have Python 3, `pip` or `uv`, `aria2c`, `curl`,
and the CUDA toolkit available, with `nvcc` on `PATH`. Setup creates a Python
3.11 environment. The examples use writable `/data` directories; change the
model, environment, and log paths if your machine uses another layout.

```bash
git clone https://github.com/devin-lai/DeepSeek-V4.1-Flash-Accel.git
cd DeepSeek-V4.1-Flash-Accel

# Download the checkpoint through ModelScope / hf-mirror; resumable.
MODEL_DIR=/data/models/DeepSeek-V4.1-Flash bash scripts/download/download_model.sh
python3 scripts/download/verify_shards.py /data/models/DeepSeek-V4.1-Flash

# Install the pinned stack, offload plugin, and patches; run preflight.
PYPI=https://pypi.org/simple \
  VENV=/data/venvs/vllm-dsv41 MODEL=/data/models/DeepSeek-V4.1-Flash \
  bash scripts/env/setup.sh
source /data/venvs/vllm-dsv41/bin/activate

# Start text serving in the foreground, listening locally.
HOST=127.0.0.1 MODEL=/data/models/DeepSeek-V4.1-Flash \
  PRESET=v41-flash-latency deploy/serve.sh
```

For mainland China, set `PYPI=https://pypi.tuna.tsinghua.edu.cn/simple` during
setup; the model downloader already supports ModelScope and hf-mirror.

Once the server reports ready, open a second shell in the repository root:

```bash
source /data/venvs/vllm-dsv41/bin/activate
deploy/healthcheck.sh
mkdir -p benchmarks/local
python deploy/verify.py --json benchmarks/local/verify.json

curl -sS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","prompt":"The capital of France is","max_tokens":8,"temperature":0}'
```

Inspect the probe output and saved JSON before benchmarking. These are sanity
checks; a passing verdict alone does not establish model quality.

Use `PRESET=v41-flash-throughput` for concurrent batch work; its wider batch
increases aggregate throughput while trading per-request token latency. The
previous `v41-flash` preset remains available as a reproduction baseline.
For image inputs, stop the text server and restart with
`PRESET=v41-flash-vision`. Download and first compilation add time; the recorded
eager startup took about 285 seconds with a warm OS page cache. Shard checks
validate file structure; add `--sha256` for comparison with ModelScope hashes.

[Custom paths, presets, and systemd](deploy/README.md) ·
[Patch inspection and rollback](upstream/README.md) ·
[19 known failure modes](docs/05-fault-inventory.md)

## How the optimizations work

### 1. Make CUDA graphs usable

CUDA graph replay reduces repeated CPU launch overhead. Two fixes address
incorrect outputs after capture: vLLM clears the null KV-cache blocks, and
FlashInfer uses a finite row for masked sparse-attention gathers. The pinned
stack also needs the dispatch and cache-layout fixes in this repository.

![Eager execution launches each GPU kernel separately; graph replay reuses a captured sequence. The fixes clear null KV blocks after capture and use finite rows for masked gathers.](docs/assets/cuda-graphs.webp)

**Recorded effect:** 33.7 vs 6.0 output tok/s at one concurrent request;
median time per output token (TPOT) falls from 159.52 to 24.04 ms.
[Patch details](upstream/README.md) ·
[Capture investigation](benchmarks/results/2026-09-14-v41-cudagraphs.md)

### 2. Offload decoder experts; keep encoder experts resident

V4.1's causal encoder-decoder structure makes placement matter. Restricting
expert offload to layers 20–39 keeps encoder expert weights on the GPUs during
prefill. Selected decoder expert weights and the Engram table reside in CPU
RAM; expert computation remains on the GPUs, with host memory accessed over
PCIe. Engram lookups still use host memory.

![Eight RTX 5090 GPUs keep encoder experts in layers 0–19 resident. Selected decoder experts from layers 20–39 and Engram live in CPU RAM; decoder computation reads host-resident weights over PCIe/UVA.](docs/assets/expert-placement.webp)

**Recorded effect:** 9.5% higher 8K prefill throughput and 8.7% lower time to
first token (TTFT) in a separate eager-mode placement experiment.
[Placement measurements](benchmarks/results/2026-09-14-v41-first-serve.md#where-the-offloaded-experts-should-live)

Two further changes help the checkpoint fit:

| Optimization | What changes |
| --- | --- |
| Expert parallelism | Whole experts avoid tensor-parallel padding: **33.6 vs 44.8 GiB/rank of expert weights** in the reference layout calculation, 25% less. This is not a 25% reduction in total runtime VRAM. [Memory breakdown](docs/03-deployment-design.md#parallelism-expert-parallelism-not-pipeline-parallelism) |
| Single-copy host offload | Allocate the pinned CPU buffer directly and copy once, avoiding a temporary pageable host copy. Enabled by the [offload plugin](vllm_dsv41_opt/README.md). |

The illustrations explain the mechanisms; the tables and linked result files
are the source for quantitative claims.

### 3. Remove host padding and tune the decode batch

The new exact allocator replaces power-of-two rounding with page-rounded
CUDA host registration for persistent weights. At the same expert budget,
shared host memory falls from about 453 to 289 GiB. The latency preset also
uses static DSpark-5 and smaller graph captures, keeping more experts on GPU
and reaching about 279 GiB shared RAM. The throughput preset retains still
more experts on GPU and permits 32 active sequences, using about 264 GiB.

The checkpoint and its MXFP4 quantization stay unchanged. Adaptive DSpark
verification remains unsupported by this SM120 indexer backend; the preset
uses static verification. [Allocator implementation and GPU tests](vllm_dsv41_opt/README.md) ·
[Measurements and tradeoffs](benchmarks/results/2026-09-14-v41-optimization.md).

## Measured performance

**New text presets, three-trial means.** Output tok/s unless marked total.
The control is the previous `v41-flash` graph-enabled preset; all runs use
the same six-case harness. See the report for individual trials and latencies.

| Workload | Previous preset | Latency preset | Throughput preset |
| --- | ---: | ---: | ---: |
| Random 1K / 128, c1 | 33.6 | 65.0 | 39.5 |
| Random 1K / 128, c8 | 91.0 | 114.4 | 113.2 |
| Random 1K / 128, c32 | 115.0 | 135.0 | 178.2 |
| 8K prefill, c2 (total tok/s) | 2429.8 | 2587.5 | 3008.4 |
| Interactive / 256, c1 | 39.8 | 71.4 | 46.0 |
| Interactive / 256, c8 | 110.7 | 155.9 | 139.7 |

Choose `v41-flash-latency` for interactive generation and
`v41-flash-throughput` for concurrent batch work. Wider batching reduces
queueing but increases per-request token latency at c32. Both passed one
32K and eight concurrent 8K cache probes; full task-quality parity and
production capacity remain unmeasured.

[Repeated trials and raw request timings](benchmarks/results/2026-09-14-v41-optimization.md) ·
[Reproduction commands](benchmarks/README.md#repeated-optimization-measurements).

### Earlier graph and offload-placement experiments

These were recorded separately with earlier harness defaults. They are not
the numerical control for the new preset comparison above.

**DeepSeek-V4.1-Flash, text only, 8× RTX 5090.** TP8 + expert parallelism,
Marlin, CPU Engram, and 12 GiB/rank of decoder expert offload.
`vllm bench serve`, random 1,024-token inputs / 128-token outputs, `ignore_eos`.

| Concurrent requests | Output tok/s, graphs | Output tok/s, eager | Speedup | Median TTFT, graphs | Median TPOT, graphs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **33.7** | 6.0 | **5.6×** | 692 ms | **24.04 ms** |
| 8 | **92.4** | 38.2 | **2.4×** | 2,167 ms | 72.28 ms |
| 32 | **115.8** | 73.7 | **1.6×** | 19,688 ms | 128.91 ms |

The baseline is the **same patched deployment in eager mode**, not stock vLLM
or another engine. The cases used only 4 / 16 / 64 requests. The server allows
16 sequences, so the 32-request case includes queueing. Output throughput
covers the request lifecycle and is not the reciprocal of TPOT.

[Graph results](benchmarks/results/2026-09-14-v41/kit-v41-graphs-bench.json) ·
[Eager baseline](benchmarks/results/2026-09-14-v41/kit-v41-preset-bench.json) ·
[Reproduce the comparison](benchmarks/README.md)

**Separate offload-placement experiment:** eager mode, 8,192-token inputs /
1-token outputs, concurrency 2, four requests. Change only the permitted
expert-offload layers:

| Placement | Prefill total tok/s | Median TTFT |
| --- | ---: | ---: |
| Stock layer order | 2,424.0 | 6,289.2 ms |
| Decoder layers 20–39 | **2,654.5 (+9.5%)** | **5,741.0 ms (−8.7%)** |

[Saved placement results](benchmarks/results/2026-09-14-v41/ladder2-offload-placement.json).
These gains must not be multiplied with the CUDA graph speedup.

**Evidence limits for the earlier comparison:** one machine, short synthetic
workloads and no repeated trials. The newer preset comparison above reports
three trials per workload; production workloads remain untested. The graph-enabled run
matched 5/6 greedy continuation probes, scored perplexity 2.662 on one short
English passage, and answered the word problem with `$8`
([saved sanity probes](benchmarks/results/2026-09-14-v41/kit-v41-graphs-verify.json)).
The missed continuation was plausible but lacked the expected word. Outputs
have varied across identical runs; these checks do not establish quality parity.
Broader vision and long-context quality evaluation remain open; the new text
presets have separate 32K/8×8K cache and finite-output sanity checks. Earlier V4-Flash results are
kept in a [separate tuning matrix](docs/07-engineering-report.md#5-the-v4-flash-tuning-matrix).

## Research and contributions

**Got it running? [Share your hardware and results](https://github.com/devin-lai/DeepSeek-V4.1-Flash-Accel/issues/new?template=measurement.md).**
Successful and failed reproductions both help extend hardware coverage.
Issues and pull requests are welcome in English or Chinese.

| Work on | Start with |
| --- | --- |
| Deployment, memory fit, or a failed launch | [Runbook](deploy/README.md), [memory planner](tools/plan_memory.py), [log scanner](tools/faultscan.py) |
| Sparse attention or cache correctness | [Reference and shape probes](tools/sm120_sparse_mla/README.md), [tiny model](tools/tiny/README.md), [patch reports](upstream/README.md) |
| Quantization and offload research | [Expert-level study](docs/06-expert-quantization.md); smaller formats remain experimental |
| Vision quality, adaptive DSpark, or other GPUs | [Benchmark protocol](benchmarks/README.md) and [contribution guide](CONTRIBUTING.md) |

Include exact revisions, hardware, flags, workload, baseline, and output-quality
checks with performance claims. For research citations, use [CITATION.cff](CITATION.cff)
and the exact commit. Credit the model and relevant upstream projects.
The patch reports have not been submitted or accepted upstream.

[Repository checks](https://github.com/devin-lai/DeepSeek-V4.1-Flash-Accel/actions/workflows/check.yml)
cover public files, syntax, and documentation links. GPU performance and output
quality require separate runs on the target hardware.

## License

[MIT](LICENSE), with component exceptions in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The offload plugin uses [Apache-2.0](vllm_dsv41_opt/LICENSE). Model weights are
downloaded separately and retain their [own license](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/LICENSE).
