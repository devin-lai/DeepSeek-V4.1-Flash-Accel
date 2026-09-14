# vllm-dsv41-opt

A vLLM general plugin for CPU weight offload, distributed as part of
[DeepSeek V4.1 Flash Accel](../README.md). It registers at runtime and provides:

- Direct allocation of the pinned host buffer to avoid a temporary second copy.
- Control over **which layers** `--cpu-offload-gb` may spill to host memory.

```bash
pip install ./vllm_dsv41_opt
DSV41_OFFLOAD_LAYERS=20-39 vllm serve ... --cpu-offload-gb 12 --cpu-offload-params w13_weight w2_weight
```

Why: DeepSeek-V4.1-Flash is a causal encoder-decoder. Prefill only executes
layers 0-19, so offloaded experts in those layers are streamed over PCIe for
every prompt token, while offloaded experts in layers 20-39 are only touched
during decode. Same GPU memory saved, prefill untaxed.

Set `DSV41_SINGLE_COPY=0` to disable the single-copy change for an ablation.
Leave `DSV41_OFFLOAD_LAYERS` unset to retain stock layer ordering.

The plugin hooks vLLM's `make_layers` and UVA offloader. Layer indices come
from submodule prefixes (`...layers.N...`). It is validated only with the
[recorded stack](../README.md#supported-configuration); other versions and
models need testing. V4.1 serving also requires the separate
[attention and cache patches](../upstream/README.md).

License: [Apache-2.0](LICENSE).
