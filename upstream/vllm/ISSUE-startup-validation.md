# DeepSeek-V4.1 on SM120 validates a FlashInfer shape the model never requests

**Component:** `vllm/models/deepseek_v4_1/nvidia/flashinfer_sparse.py`
**Version:** vLLM main @ `8c1d1c297`, FlashInfer 0.6.18.post1
**Hardware:** 8x RTX 5090 (sm_120), DeepSeek-V4.1-Flash, TP8 + EP

## Summary

`DeepseekV4FlashInferMLAAttentionSM120.__init__` has a guard whose whole job is
to fail fast when FlashInfer lacks the kernel the model needs. It checks the
wrong quantity, so it passes on a configuration that cannot decode, and the real
failure lands about six minutes later — after 476 GiB of weights are loaded and
pinned — inside CUDA-graph capture.

## What happens

```python
required_topk = _required_sm120_sparse_topk(vllm_config, self.window_size)
if not has_flashinfer_sparse_mla_sm120_config(self.padded_heads, required_topk):
    raise RuntimeError(
        "FLASHINFER_MLA_SPARSE_DSV4 on SM120 requires a FlashInfer "
        f"DSV4 sparse MLA decode specialization for "
        f"(num_q_heads={self.padded_heads}, top_k={required_topk}). ..."
    )
```

and

```python
def _required_sm120_sparse_topk(vllm_config: VllmConfig, window_size: int) -> int:
    if not vllm_config.attention_config.use_non_causal:
        return window_size
    speculative_config = vllm_config.speculative_config
    if speculative_config is None:
        return window_size
    return get_dspark_swa_index_width(window_size, speculative_config.num_speculative_tokens)
```

With speculative decoding off, `required_topk` is the bare `sliding_window`,
128. `(8, 128)` is in FlashInfer's table, so the guard passes.

At runtime the same layer calls

```python
flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
    ..., sparse_indices=swa_indices_chunk, ...)
```

and `sparse_indices` is `swa_metadata.prefill_swa_indices`, whose width is the
*index-buffer* width — `2 * index_topk + window` = `2*512 + 128` = **1152** for
DeepSeek-V4.1-Flash, sized for the widest CSA2 layer and shared across layers,
with `swa_lens` carrying the per-token true length. FlashInfer has no
specialisation at 1152:

```
ValueError: SM120 sparse-MLA has no decode kernel for this shape:
num_tokens=8, num_heads=8, topk=1152, d_qk=512, page_block_size=32,
model_type=1, extra_topk=0
```

So the validated shape (the *window*) and the requested shape (the *index
buffer*) are different quantities that happen to share a name.

The same mismatch exists on the page-block axis: the guard does not look at page
geometry at all, and the SWA cache's page block size is 32, while FlashInfer's
DSv4 decode accepts only 64.

## Why it is worth fixing even though FlashInfer is also at fault

The kernel gap is real and has its own report (patch attached there; after it,
`topk=1152, pbs=32` dispatches and matches a reference implementation). But the
guard's value is that it costs six minutes to learn this the current way, on a
machine where each attempt pins 452 GiB of host RAM. A guard that checks the
shape the model will actually pass turns that into an immediate error naming the
number to look for.

## Suggested fix

Compute the index-buffer width the metadata builder will produce, and validate
that — it is derivable at construction time from `index_topk`, `compress_ratio`
and the SWA window, which are all config fields. Include the page block size of
the SWA cache in the query.

A related cleanup: `has_flashinfer_sparse_mla_sm120_config` reaches into
FlashInfer's private `_DECODE_DSV4_DISPATCH` frozenset and does a tuple
membership test, with a comment acknowledging it is a stand-in "until it exposes
a public capability query". That coupling is what makes the check hard to state
correctly. A published predicate on the FlashInfer side —
`sparse_mla_sm120_decode_supported(num_heads, topk, page_block_size)` — would
let this guard ask the real question.

## Third-order issue found in the same area

vLLM gates all of FlashInfer on `shutil.which("nvcc")` (`has_flashinfer()` →
`has_flashinfer_cubin() or nvcc on PATH`). On a box where CUDA is installed at
`/usr/local/cuda` but `/usr/local/cuda/bin` is not on a non-interactive `PATH`,
the DSv4.1 SM120 layer reports

```
RuntimeError: FLASHINFER_MLA_SPARSE_DSV4 on SM120 requires a FlashInfer DSV4
sparse MLA decode specialization for (num_q_heads=8, top_k=128).
Install a FlashInfer build containing flashinfer-ai/flashinfer#4380.
```

which sends the reader to rebuild FlashInfer when the actual fix is
`export PATH=/usr/local/cuda/bin:$PATH`. Distinguishing "FlashInfer is
unavailable in this environment" from "FlashInfer is present but lacks this
kernel" in the message would make that a ten-second fix.
