# vllm-dsv41-opt

A vLLM general plugin for CPU weight offload, distributed as part of
[DeepSeek V4.1 Flash Accel](../README.md). It registers at runtime and provides:

- Direct allocation of the pinned host buffer to avoid a temporary second copy.
- Control over **which layers** `--cpu-offload-gb` may spill to host memory.
- Page-rounded pinned weight allocation, enabled with `DSV41_EXACT_PINNED=1`.

```bash
pip install ./vllm_dsv41_opt
DSV41_OFFLOAD_LAYERS=20-39 vllm serve ... --cpu-offload-gb 12 --cpu-offload-params w13_weight w2_weight
```

Why: DeepSeek-V4.1-Flash is a causal encoder-decoder. Prefill only executes
layers 0-19, so offloaded experts in those layers are streamed over PCIe for
every prompt token, while offloaded experts in layers 20-39 are only touched
during decode. Same GPU memory saved, prefill untaxed.

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

Run the GPU checks in the serving environment (install `pytest` there if needed):

```bash
python -m pytest vllm_dsv41_opt/tests -q
```

These cover CUDA-view lifetime, FP8/BF16 storage, exact allocation sizes,
noncontiguous fallback, and the transfer wrapper when UVA is disabled.

The plugin hooks vLLM's UVA offloader and selected weight allocation modules. Layer indices come
from submodule prefixes (`...layers.N...`). It is validated only with the
[recorded stack](../README.md#supported-configuration); other versions and
models need testing. V4.1 serving also requires the separate
[attention and cache patches](../upstream/README.md).

License: [Apache-2.0](LICENSE).
