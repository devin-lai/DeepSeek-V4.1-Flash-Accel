# DeepSeek-V4.x: text-only serving still sizes the SWA prefill index for images, and the SWA cache asks for a page size no sm_120 kernel implements

**Component:** `vllm/v1/attention/backends/mla/sparse_swa.py`,
`vllm/models/deepseek_v4_1/attention.py`
**Version:** vLLM main @ `8c1d1c297`
**Hardware:** 8x RTX 5090 (sm_120), DeepSeek-V4.1-Flash, TP8 + EP

Two small, independent defects in the DeepSeek-V4 sliding-window path. Neither
is sm_120-specific in nature, but on sm_120 they are the difference between a
model that serves and one that cannot dispatch a prefill kernel at all.

---

## 1. `--language-model-only` does not narrow the SWA prefill index

`DeepseekSparseSWAMetadataBuilder.__init__`:

```python
# Vision variant: image spans (up to vision_max_n_token tokens) are
# visible bidirectionally, so prefill index rows widen from
# window_size to window_size + max_image_tokens.
self.max_image_tokens = (
    getattr(hf_config, "vision_max_n_token", 0)
    if getattr(hf_config, "vision_n_layers", 0) > 0
    else 0
)
self.prefill_index_width = self.window_size + self.max_image_tokens
```

`vision_n_layers > 0` says the *checkpoint* has a vision tower. It does not say
this engine will ever be handed an image. Both `--language-model-only` and
`--limit-mm-per-prompt '{"image": 0}'` put vLLM in text-only mode — the
registry even logs it —

```
INFO [registry.py:146] All limits of multimodal modalities supported by the
model are set to 0, running in text-only mode.
```

— and neither touches the hf_config attribute. Measured on V4.1-Flash with
`--language-model-only`:

```
ModelConfig.hf_config -> vision_n_layers=32 vision_max_n_token=1024
prefill_index_width   -> 128 + 1024 = 1152
```

So every prefill builds 1152-wide index rows per token, and the Triton kernel
fills all 1152 columns, to describe bidirectional visibility inside image spans
that cannot occur. At `max_num_batched_tokens = 8192` that is 36 MiB of index
buffer and 9x the necessary column writes, on every platform.

### Suggested fix

Key the widening on whether images can arrive, not on whether the checkpoint
could accept them:

```python
def images_can_arrive(vllm_config) -> bool:
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None or not getattr(model_config, "is_multimodal_model", False):
        return False
    mm_config = getattr(model_config, "multimodal_config", None)
    if mm_config is None:
        return False
    try:
        return mm_config.get_limit_per_prompt("image") > 0
    except Exception:
        return True
```

`DeepseekV4MLAAttention.__init__` computes the same quantity for its JIT warmup
key (`_COMPUTE_SWA_INDICES_AND_LENS_KERNEL.register_warmup(...,
max_image_tokens=...)`) and should use the same gate, so the warmed
specialisation is the one the kernel is actually called with.

### Why it is load-bearing on sm_120

FlashInfer's sm_120 sparse-MLA prefill dispatch instantiates `TOPK` from a
fixed list — `{128, 192, 256, 512, 1024, 2048}` for the single-cache path, and
**only `topk == 128`** for the dual-cache path that every compressed layer
uses. `topk` here *is* `prefill_index_width`. At 1152 nothing dispatches:

```
tvm.error.InternalError: Check failed: (ok) is false:
Unsupported sparse-MLA prefill configuration: model=DSV4 num_heads=8
topk=1152 page_block_size=32 topk_extra=0 extra_page_block_size=0
```

With the gate above, a text-only engine asks for 128, which is the one value
the dual-cache path implements, and DeepSeek-V4.1-Flash serves correctly on
consumer Blackwell.

---

## 2. The SWA cache asks for 32-token pages; every sm_120 sparse-MLA kernel is written for 64

`DeepseekV4MLAAttention.__init__`:

```python
self.swa_cache_layer = DeepseekV4SWACache(
    ...,
    block_size=32,
)
```

That literal is the `page_block_size` every sparse-MLA entry point receives for
the sliding-window cache. FlashInfer's sm_120 kernels assume 64 throughout:

- `sparse_mla_sm120_prefill.cu` bakes `PAGE_BLOCK_SIZE = 64` into all 30 of its
  DSV4 single-cache launches (5 head counts × 6 topks) and returns `false` for
  anything else;
- `sparse_mla_sm120_decode_dsv4.cu` had `if (mt != ModelType::DSV4 ||
  page_block_size != 64) return false;`, which is the second half of the "no
  decode kernel for this shape: ... page_block_size=32" report.

The SWA backend already declares `MultipleOf(32)`, so 64 is a legal request. It
also makes the 128-token window span two pages instead of four. Changing the
literal to 64 removes the page-block axis from both failures with no kernel
change.

If 32 is deliberate for another architecture, it should be selected per
platform rather than fixed, and the sm_120 sparse-MLA backends should declare
what they can actually consume instead of failing inside the kernel launch.

---

## Reproducing

```bash
vllm serve /path/to/DeepSeek-V4.1-Flash \
  --tokenizer-mode deepseek_v41 --tensor-parallel-size 8 --enable-expert-parallel \
  --language-model-only --block-size 64 --enforce-eager
```

Without either fix: `Unsupported sparse-MLA prefill configuration ... topk=1152
page_block_size=32`. With both: the server answers, and a greedy
`"The capital of France is"` returns `" Paris. The Eiffel Tower is"`.
