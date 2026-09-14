# DeepSeek-V4.1 has no working KV block size on sm_100/sm_120 — because the block size is uniform across layers and it should not be

**Component:** `vllm/models/deepseek_v4_1/attention.py`,
`vllm/v1/attention/backends/mla/indexer.py`,
`vllm/models/deepseek_v4_1/{sparse_mla.py,nvidia/flashinfer_sparse.py}`,
`vllm/v1/worker/utils.py`
**Version:** vLLM main @ `8c1d1c297`, DeepGEMM vendored
**Hardware:** 8x RTX 5090 (sm_120), DeepSeek-V4.1-Flash, TP8 + EP
**Status:** fixed locally; patch in `upstream/vllm/apply_patch.py`, and
DeepSeek-V4.1-Flash now serves on 8x RTX 5090 with it.

## Summary

Three components in one KV cache group disagree about the block size, and on
sm_100/sm_120 the intersection is empty. Every value fails, each in a different
place:

| `--block-size` | what happens |
| --- | --- |
| 128 (the only value offered) | CUDA-graph capture aborts: `Assertion error (csrc/apis/attention.hpp:262): block_kv == 32 or block_kv == 64` |
| 64 | `ValueError: No common block size for 64.` — the DSv4.1 indexer backend declares `[128]` |

sm_90 is unaffected only because both declarations happen to branch to 64 there.

## Why no `--block-size` works

The DSA indexer's states-per-block is `block_size // tokens_per_state`, and
`tokens_per_state` is the layer's compression ratio. DeepSeek-V4.1-Flash uses
more than one. From its `config.json`:

```
compress_ratios = [0, 0,  2 x18 (layers 2-19),  1 x20 (layers 20-39),  0, 0, 0]
```

The CED encoder half compresses two tokens per indexer state, the decoder half
one. Both run in the same engine, so a single `--block-size` produces two
different states-per-block:

| `--block-size` | ratio-2 layers | ratio-1 layers | result |
| ---: | ---: | ---: | --- |
| 128 | 64 | **128** | ratio-1 fails `:262` (needs 32 or 64) |
| 64 | **32** | 64 | ratio-2 fails `:320` (sm_120 + FP8 needs exactly 64) |

Both were run; both fail exactly there. DeepGEMM wants 64 states per block from
every layer, and no single *token* block size delivers it for two different
compression ratios.

## The fix: stop requiring one block size for layers that compress differently

The framing above contains its own answer. What the kernel constrains is
**states per block**, not tokens per block. Scale each layer's block size by
its own compression ratio and every layer stores the same number of states:

```python
def states_per_block_to_tokens(block_size: int, compress_ratio: int) -> int:
    return block_size * compress_ratio if compress_ratio > 1 else block_size
```

applied where the specs are built — `DeepseekV4MLAAttention.get_kv_cache_spec`
and `DeepseekV4IndexerCache.get_kv_cache_spec`. With `--block-size 64`:

| layer half | tokens/block | states/block | page bytes (MLA / indexer) |
| --- | ---: | ---: | ---: |
| ratio 2 (encoder, layers 2-19) | 128 | **64** | 37 440 / 8 640 |
| ratio 1 (decoder, layers 20-39) | 64 | **64** | 37 440 / 8 640 |

Both satisfy DeepGEMM, and the two halves land in separate cache groups on
their own: `UniformTypeKVCacheSpecs.is_uniform_type` returns False as soon as
block sizes differ. Nothing in the allocator has to change, and this model
already runs four different block sizes — the SWA cache asks for 32, the
compressor for 8.

Bytes per token are unchanged: a ratio-1 layer stores twice the states per
token either way.

It also generalises rather than special-casing V4.1. `DeepseekV4IndexerBackend`
already declares `[256]` with the comment *"Block sizes count uncompressed
tokens: C4 indexer pages hold 64 rows"* — which is exactly `64 x
compress_ratio`. This makes that relationship the rule instead of a constant
that happens to be right for one ratio.

Measured, after the change:

```
gid=14 manager_bs=128 layers=6 [FLASHINFER_MLA_SPARSE_DSV41, DEEPSEEK_V41_INDEXER]
   layers 2/8/14 .attn + .indexer.k_cache   bs=128 tps=2 states=64
gid=16 manager_bs=64  layers=2 [FLASHINFER_MLA_SPARSE_DSV41, DEEPSEEK_V41_INDEXER]
   layer 20 .attn + .indexer.k_cache        bs=64  tps=1 states=64
```

## The route this report first proposed does not work

An earlier version of this report argued for per-group **kernel** block sizes:
let the ratio-2 group run at kernel block 128 inside 128-token manager blocks
and the ratio-1 group at 64, using machinery that is already half-built —
`select_common_block_size` returns a kernel block size per cache group, the
indexer builder already rescales its block table by `spec.block_size //
kernel_block_size`, and `AttentionSpec.get_num_kernel_states` takes exactly the
right argument and is never called with it.

That was implemented and it cannot work for this model, for a reason worth
recording:

```python
padded_page_size = getattr(spec, "page_size_padded", None)
if padded_page_size is not None:
    assert kernel_block_size is None or kernel_block_size == spec.block_size, (
        "Padded KV pages do not support kernel block splitting."
    )
```

V4.1's pages are always padded — 576-byte alignment against a 584-byte MLA
state and a 132-byte indexer row — so kernel block splitting is unavailable to
it by construction.

It also would not have helped on its own: the ratio-1 and ratio-2 layers are in
the *same* KV cache group (`UniformTypeKVCacheSpecs` buckets by spec type, not
by `tokens_per_state`), and the kernel block size is negotiated per group, so a
single value would still have had to serve both. The four attention groups
*inside* that cache group already separate by `tokens_per_state`; the
constraint is per attention group and the negotiation is per cache group. That
mismatch is the structural gap, and giving the halves different manager block
sizes closes it from the other side.

## Two more things that would have saved time

**1. `num_states` is derived from the manager block size in a builder that has
already rescaled its block table for a different one.** In
`DeepseekV32IndexerMetadataBuilder.build`:

```python
if (kernel_block_size is not None
        and self.kv_cache_spec.block_size != kernel_block_size
        and self.kv_cache_spec.block_size % kernel_block_size == 0):
    factor = self.kv_cache_spec.block_size // kernel_block_size
    compressed = block_table[:, ::factor] // factor
...
metadata = get_paged_mqa_logits_metadata(
    seq_lens, self.kv_cache_spec.num_states, self.num_sms, indices=decode_indices)
```

The table describes kernel blocks; `num_states` describes manager blocks. This
is latent today because `create_metadata_builders` hands the builder a spec
whose `block_size` has already been replaced by the kernel block size, so the
two can never differ — but that also makes the rescale above dead code, and
either the rescale or the accessor is wrong. `get_num_kernel_states` exists for
exactly this and should be used.

**2. The negotiation error names no backend.**

```
ValueError: No common block size for 64.
```

is the whole message, with every backend in the group declaring a different
set. Three lines turn it into a diagnosis:

```python
detail = "; ".join(
    f"{b.get_name()} supports {b.get_supported_kernel_block_sizes()}"
    for b in backends
)
raise ValueError(
    f"No common block size for {kv_manager_block_size}. "
    f"Candidates tried: {sorted(all_int_supported_sizes, reverse=True)}. {detail}"
)
```

Better still, let the negotiation see the constraint that actually decides the
outcome. `prepare_kernel_block_sizes` has the spec in hand; passing an
acceptance predicate into `select_common_block_size` lets a DSA-indexer group
refuse a size whose states count DeepGEMM cannot take, at startup, naming the
constraint — instead of aborting in a C++ assert during graph capture six
minutes in. The patch in this repo does that too.

## Suggested upstream change

1. Scale the V4.x MLA and indexer specs' block size by the layer's compression
   ratio, as above.
2. Offer both 64 and 128 in `DeepseekV4SparseMLABackend`,
   `DeepseekV4FlashInferMLASparseBackend` and `DeepseekV41IndexerBackend` on
   every architecture rather than just sm_90.
3. Pass `get_num_kernel_states(kernel_block_size)` where the indexer builder
   currently passes `num_states`, so the value matches the table it rescales.
4. Validate the DeepGEMM states constraint during block-size negotiation, where
   it can name the flag.

`upstream/vllm/apply_patch.py` in this repo applies 1, 2 and 4, and the model
serves with them.

## Also: the FP4 branch exists and vLLM refuses to use it

DeepGEMM has an sm_120 FP4 path that accepts `block_kv` of 32. vLLM will not
select it:

```python
use_fp4 = kv_dtype == "mxfp4"
if use_fp4 and not current_platform.is_device_capability_family(100):
    raise ValueError(
        "indexer_kv_dtype='mxfp4' requires Blackwell datacenter GPUs "
        "(sm_10x, e.g. B200/GB200); sm_120 (consumer Blackwell) and "
        "earlier architectures are not supported.")
```

If that reflects a gap elsewhere in the MXFP4 indexer path, the message should
say where. If it is simply older than DeepGEMM's sm_120 FP4 branch, lifting it
would give this model a second working configuration.
