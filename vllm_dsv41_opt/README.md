# vllm-dsv41-opt

A vLLM general plugin for CPU weight offload, distributed as part of
[DeepSeek V4.1 Flash Accel](../README.md). It registers at runtime and provides:

- Direct allocation of the pinned host buffer to avoid a temporary second copy.
- Control over **which layers** `--cpu-offload-gb` may spill to host memory.
- Page-rounded pinned weight allocation, enabled with `DSV41_EXACT_PINNED=1`.
- Exact packed-weight staging for large eager Marlin batches, enabled with
  `DSV41_MARLIN_STAGE_MIN_TOKENS=256` (plugin 0.3.0).

```bash
pip install ./vllm_dsv41_opt
DSV41_OFFLOAD_LAYERS=20-39 vllm serve ... --cpu-offload-gb 12 --cpu-offload-params w13_weight w2_weight
```

The layer filter controls placement. Both layers 0–19 and 20–39 execute
during prefill on the measured vLLM path, so decoder-only placement still
incurs PCIe reads during prefill. See the [kernel profile](../docs/08-pcie-bound-serving.md).

Set `DSV41_SINGLE_COPY=0` to disable the single-copy change for an ablation.
Also unset `DSV41_EXACT_PINNED` for that ablation; exact allocation requires
the single-copy path.
Leave `DSV41_OFFLOAD_LAYERS` unset to retain stock layer ordering.

## Exact pinned weight allocation

PyTorch's pinned host allocator rounds each large buffer to a power of two.
On this model that inflates Engram tables and scales from 188.83 to 264 GiB;
expert buffers incur their own rounding. `DSV41_EXACT_PINNED=1` instead
registers page-sized anonymous mappings with CUDA. Tensor storage owns each
mapping, and vLLM's CUDA views retain that storage until the last view dies.
Weights and their quantization are unchanged.

At the same 12 GiB/rank offload budget, observed shared host memory fell from
about 453 to 289 GiB. The tuned DSpark preset uses about 279 GiB and also
completed startup and request probes under a 384 GiB cgroup memory limit.
This is a limit experiment on a 503 GiB host, not validation of a physical
384 GiB machine. [Measurements and scope](../benchmarks/results/2026-09-14-v41-optimization.md).

```bash
DSV41_EXACT_PINNED=1 DSV41_OFFLOAD_LAYERS=20-39 \
  vllm serve ... --cpu-offload-gb 12 --cpu-offload-params w13_weight w2_weight
```

The allocator is scoped to Engram weight creation, the plugin's expert
offloader, and vLLM's post-repack weight restoration. Activations, request
buffers, IPC, and the global `torch` module keep their normal allocators.
Small allocations and noncontiguous layouts also keep the original allocator.
This path is for persistent weights with synchronous copies; it does not
implement the caching allocator's lifetime tracking for temporary asynchronous
transfer buffers. Linux with CUDA is required.

## Marlin weight staging

For an offloaded expert matrix, a large Marlin GEMM can read the same weights
from host memory for multiple token tiles. Staging copies the packed matrix
into a temporary GPU tensor and calls the same GEMM with that tensor. It
preserves the checkpoint bytes, scales, BF16 activations and FP32 reduction
settings. It stages one projection at a time; PyTorch reuses the temporary
allocation on the current stream after the GEMM completes.

```bash
DSV41_MARLIN_STAGE_MIN_TOKENS=256 PRESET=v41-flash-latency deploy/serve.sh
```

The threshold counts query tokens before top-k expansion. The hook applies
only to weights marked by vLLM as UVA-offloaded. Small batches and CUDA graph
capture retain direct UVA access. This avoids copying all local experts when
a decode batch uses only a few. Startup logs report both installation and the
first staged matrix, so an enabled flag can be distinguished from actual use.
If the temporary allocation fails, the hook calls the original UVA GEMM and
logs the first fallback per worker. This preserves execution when allocator
fragmentation or a larger request shape leaves insufficient staging space;
it does not guarantee the staged speedup under that memory pressure.

Staging consumes temporary GPU memory and must be present during vLLM's startup
memory profile. Keep it disabled on unmeasured models or kernels. The 256-token
threshold was tested on this repository's pinned V4.1/SM120 stack; it is not
a universal crossover.

## Routing diagnostics

`DSV41_ROUTING_PROFILE=/data/routing.json` enables counters for the measured
TP8 + EP, 384-expert, linear-placement, DSpark-5 configuration. It counts the
distinct experts selected by six verification queries on each GPU, per layer,
including the busiest GPU. The counters live in persistent GPU storage and
accumulate during CUDA-graph replay. Other batch shapes are excluded.

Start the diagnostic server on loopback with its worker extension and vLLM's
development RPC route:

```bash
HOST=127.0.0.1 VLLM_SERVER_DEV_MODE=1 DSV41_ROUTING_PROFILE=/data/routing.json \
  EXTRA='--worker-extension-cls vllm_dsv41_opt.routing_profile.RoutingProfileWorkerExtension' \
  PRESET=v41-flash-latency deploy/serve.sh
# In another shell, while serving is idle:
curl -fsS http://127.0.0.1:8000/collective_rpc \
  -H 'Content-Type: application/json' -d '{"method":"snapshot_dsv41_routing"}'
```

Snapshot once after startup to save and reset warmup counters. Run a
calibration workload, then snapshot while idle again to save its counters.
Each JSON snapshot includes a generation number. Keep calibration separate
from evaluation and disable the profiler, worker extension and development
mode for production and speed measurements: the counters add GPU work.

## Validation

Run GPU checks with the serving process stopped, in the serving environment
(install `pytest` there if needed):

```bash
python -m pytest vllm_dsv41_opt/tests -q
```

These cover CUDA-view lifetime, FP8/BF16 storage, exact allocation sizes,
noncontiguous fallback, the transfer wrapper when UVA is disabled, staging
eligibility, exact copies and allocation-failure fallback, CUDA-graph replay
without staged allocations, and routing counts during eager and graph execution.

The [kernel benchmark](../scripts/bench/marlin_staging.py) checks bitwise output
agreement for UVA, staged and resident gate/up and down GEMMs at V4.1 dimensions:

```bash
numactl --cpunodebind=0 --membind=0 python scripts/bench/marlin_staging.py \
  --device 0 --out /data/marlin-staging.json
```

The plugin hooks vLLM's UVA offloader and selected weight allocation modules. Layer indices come
from submodule prefixes (`...layers.N...`). It is validated only with the
[recorded stack](../README.md#supported-configuration); other versions and
models need testing. V4.1 serving also requires the separate
[attention and cache patches](../upstream/README.md).

License: [Apache-2.0](LICENSE).
