# DeepSeek V4.1 Flash Accel

**Deploy and accelerate DeepSeek-V4.1-Flash with vLLM on NVIDIA RTX 5090 GPUs.**

[简体中文](README.zh-CN.md) · [Quickstart](#quickstart) · [Benchmarks](#measured-performance) · [Documentation](docs/README.md) · [Report a deployment](https://github.com/devin-lai/DeepSeek-V4.1-Flash-Accel/issues/new?template=measurement.md)

Run the released checkpoint on **8× RTX 5090 D with 503 GiB host RAM**, using
CPU offload, expert parallelism, and patched CUDA graph support. This repository
provides deployment presets, vLLM and FlashInfer patches, memory planning,
failure diagnosis, and tools for inference research.

**Recorded result: 33.7 output tokens/s at one concurrent request, 5.6× the
patched eager baseline.** The same graph-enabled setup reaches 115.8 aggregate
output tokens/s at 32 concurrent requests. These are short synthetic benchmarks
on one machine; [configuration, baseline, and saved results](#measured-performance)
are below.

Independent community project for [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).
Not affiliated with or endorsed by DeepSeek. Experimental, with one measured hardware configuration.
Built on [vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer), and
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM).

## What you can do

| Goal | Start here |
| --- | --- |
| Serve text or images through an OpenAI-compatible API | [Quickstart](#quickstart) and [deployment runbook](deploy/README.md) |
| Estimate GPU and host memory before loading the checkpoint | [Memory planner](docs/07-engineering-report.md#4-the-memory-model) |
| Diagnose a failed or unhealthy deployment | [18 documented failure modes](docs/05-fault-inventory.md) and [log scanner](tools/faultscan.py) |
| Understand the fixes and performance tradeoffs | [Engineering report](docs/07-engineering-report.md) and [patch reports](upstream/README.md) |
| Reproduce a kernel problem without the full checkpoint | [Sparse-MLA reference and shape probes](tools/sm120_sparse_mla/README.md) or [tiny model](tools/tiny/README.md) |
| Contribute measurements or extend hardware coverage | [Benchmark protocol](benchmarks/README.md) and [contribution guide](CONTRIBUTING.md) |

## Supported configuration

Status reflects the saved results from **2026-09-14**.

| Area | Current evidence |
| --- | --- |
| DeepSeek-V4.1-Flash, text | Serves with `v41-flash`; saved generation and perplexity sanity probes |
| CUDA graphs | Enabled with the VL-013 and FI-004 patches; benchmarked below |
| Images | `v41-flash-vision` serves a simple image probe; broader vision evaluation remains open |
| Hardware | Measured on 8× RTX 5090 D, SM120, PCIe Gen5 ×16, no NVLink; other GPUs need validation |
| Host memory | 503 GiB installed; approximately 452 GiB pinned during the recorded deployment |
| Checkpoint | Approximately 476 GiB / 510 GB on disk; Engram on CPU and 12 GiB of decoder experts offloaded per GPU rank |
| Context | V4.1 presets configure **32,768 tokens**; the model's advertised 1M context is not validated by this deployment |
| Software | vLLM commit `8c1d1c2974ee42757ee2e93cc898932edfd9d265` plus local patches; FlashInfer `0.6.18.post1`; PyTorch 2.13 + cu130; CUDA 13.2 toolkit; driver 595.71.05 |

See the [reference hardware](docs/02-hardware-topology.md) for CPU, storage, and
interconnect details. These are measured requirements for this configuration;
the repository does not yet establish a minimum hardware configuration.

## Quickstart

Use a Linux NVIDIA host matching the reference configuration. Have Python 3,
`pip` or `uv`, `aria2c`, `curl`, and the CUDA toolkit (`nvcc` on `PATH`) available.
Allow space for the checkpoint plus the environment and build caches. Start
from the root of your clone or extracted repository.

```bash
git clone https://github.com/devin-lai/DeepSeek-V4.1-Flash-Accel.git
cd DeepSeek-V4.1-Flash-Accel

# 1. Download the checkpoint. Uses ModelScope and hf-mirror; resumable.
MODEL_DIR=/data/models/DeepSeek-V4.1-Flash bash scripts/download/download_model.sh
python3 scripts/download/verify_shards.py /data/models/DeepSeek-V4.1-Flash

# 2. Install the pinned stack, plugin, and patches; run preflight.
# PYPI can be changed to a preferred package index.
PYPI=https://pypi.org/simple \
  VENV=/data/venvs/vllm-dsv41 MODEL=/data/models/DeepSeek-V4.1-Flash \
  bash scripts/env/setup.sh
source /data/venvs/vllm-dsv41/bin/activate

# 3. Start text serving. This command stays in the foreground.
MODEL=/data/models/DeepSeek-V4.1-Flash PRESET=v41-flash deploy/serve.sh
```

After the server reports ready, open a second shell in the repository root:

```bash
source /data/venvs/vllm-dsv41/bin/activate
deploy/healthcheck.sh
python deploy/verify.py

curl -sS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","prompt":"The capital of France is","max_tokens":8,"temperature":0}'
```

The recorded startup took approximately 285 seconds with the OS page cache
warm; download and first-time compilation add time. The shard verifier checks
file structure by default; `--sha256` additionally compares hashes with
ModelScope's listing.

To serve images, stop the text server and launch with
`PRESET=v41-flash-vision` instead. For custom paths, patch inspection and
rollback, see the [runbook](deploy/README.md) and [patch guide](upstream/README.md).

## Measured performance

**DeepSeek-V4.1-Flash on 8× RTX 5090 D**, TP8 + expert parallelism, Marlin,
Engram on CPU, 12 GiB/rank of decoder experts offloaded, text only.
`vllm bench serve`, random 1,024-token inputs / 128-token outputs, `ignore_eos`.
Client concurrency is shown below; the server preset allows 16 sequences, so
the 32-request case includes queueing.

| Concurrent requests | Output tok/s, CUDA graphs | Output tok/s, eager | Median TTFT, graphs | Median TPOT, graphs |
| ---: | ---: | ---: | ---: | ---: |
| 1 | **33.7** | 6.0 | 692 ms | **24.04 ms** |
| 8 | **92.4** | 38.2 | 2,167 ms | 72.28 ms |
| 32 | **115.8** | 73.7 | 19,688 ms | 128.91 ms |

The **5.6×** figure is `33.7 / 6.0`: single-request output throughput versus
the same kit running in eager mode. It is not a comparison against current
unmodified vLLM or another inference engine. Median TPOT fell from 159.52 to
24.04 ms. Output throughput includes the benchmark's request lifecycle; it is
not the reciprocal of TPOT. TTFT means time to first token; TPOT means time
per output token after the first.

Saved [graph-enabled results](benchmarks/results/2026-09-14-v41/kit-v41-graphs-bench.json),
[eager results](benchmarks/results/2026-09-14-v41/kit-v41-preset-bench.json),
and [experiment notes](benchmarks/results/2026-09-14-v41-cudagraphs.md) support
this table. The original cases used 4, 16, and 64 requests respectively.
Repeated-trial distributions and production workload evaluations remain open;
see the [protocol](benchmarks/README.md) to contribute them.

**Quality evidence is limited to sanity probes.** The graph-enabled run
matched 5/6 greedy continuation probes, scored perplexity 2.662 on one short
English passage, and answered the word problem with `$8`
([saved output](benchmarks/results/2026-09-14-v41/kit-v41-graphs-verify.json)).
The missed continuation was plausible but did not contain the probe's expected
word. These checks detect obvious failures; they do not establish model quality
parity. Identical configurations have produced different greedy continuations
and perplexities across runs.

DeepSeek-V4-Flash is also covered in the
[separate V4 tuning matrix](docs/07-engineering-report.md#5-the-v4-flash-tuning-matrix).
Its results are for a different model.

## Research and contributions

Useful next contributions include independent reproduction, other GPU and
memory configurations, longer-context and vision evaluation, DSpark speculative
decoding measurements, and pipeline-parallel support. The
[quantization study](docs/06-expert-quantization.md) explores expert-level
tradeoffs; its smaller formats are research candidates, not validated serving
presets. If you get a deployment running, [share your hardware and results](https://github.com/devin-lai/DeepSeek-V4.1-Flash-Accel/issues/new?template=measurement.md)
so others can reproduce it.

Start with [CONTRIBUTING.md](CONTRIBUTING.md). Include the exact model and code
revisions, launch flags, hardware, quality checks, and saved measurements when
reporting a speedup. For research citations, use [CITATION.cff](CITATION.cff)
and include the exact commit, alongside the model and relevant upstream projects.

The [patch reports](upstream/README.md) document the fixes and evidence.
They have not been submitted or accepted upstream.

## License

Repository materials use [MIT](LICENSE), with component exceptions listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The offload plugin uses
[Apache-2.0](vllm_dsv41_opt/LICENSE). Model weights are downloaded separately
and retain their [own license](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/LICENSE).
