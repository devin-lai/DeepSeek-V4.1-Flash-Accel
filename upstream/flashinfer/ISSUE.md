# SM120 sparse-MLA DSv4 decode: no kernel for DeepSeek-V4.1-Flash's shapes (topk=1152, page_block_size=32)

**Component:** `csrc/sparse_mla_sm120_decode_dsv4.cu`, `flashinfer/mla/_sparse_mla_sm120.py`
**Version:** flashinfer 0.6.18.post1 (`flashinfer-jit-cache` wheel), CUDA 13.2, driver 595.71.05
**Hardware:** 8x NVIDIA GeForce RTX 5090 D (sm_120), PCIe Gen5 x16, no NVLink
**Consumer:** vLLM main @ `8c1d1c297`, `FLASHINFER_MLA_SPARSE_DSV41` backend, DeepSeek-V4.1-Flash TP8

## Summary

`launch_sparse_mla_decode_dsv4` dispatches through a compile-time grid of 25
specialisations (`NUM_HEADS` x `TOPK`) and rejects any `page_block_size != 64`
outright. DeepSeek-V4.1-Flash on sm_120 requests `(num_heads=8, topk=1152,
page_block_size=32)`, which is outside that grid on both axes, so the model
cannot decode at all — there is no alternative backend, since FlashMLA
advertises sm_90 and sm_100 only.

Neither parameter is structural inside the kernel. Making both runtime
arguments **shrinks** the instantiation grid from 25 kernels to 5 and makes
every `(topk, page_block_size)` dispatchable. A patch and a validation harness
are attached; 141/141 probed shapes match a PyTorch reference.

## Reproduction

No checkpoint required — the harness builds the packed DSv4 KV layout directly:

```bash
python tools/sm120_sparse_mla/probe_sm120.py --oracle --shape 8,1152,32
```

Against stock 0.6.18.post1:

```
== oracle check: shapes FlashInfer already ships ==
  h=8    topk=128   pbs=64    PASS  rel=2.68e-02 cos=0.999702
  h=8    topk=512   pbs=64    PASS  rel=2.84e-02 cos=0.999721
  h=16   topk=256   pbs=64    PASS  rel=2.91e-02 cos=0.999699
  h=8    topk=1024  pbs=64    PASS  rel=2.34e-02 cos=0.999712
== oracle check with an extra segment (dual cache) ==
  h=8    topk=128   pbs=64   extra=128    PASS  rel=2.70e-02 cos=0.999707

  h=8    topk=1152  pbs=32    DISPATCH-FAIL
  ValueError: SM120 sparse-MLA has no decode kernel for this shape:
  num_tokens=8, num_heads=8, topk=1152, d_qk=512, page_block_size=32,
  model_type=1, extra_topk=0.
```

The first five rows establish that the reference implementation
(`tools/sm120_sparse_mla/dsv4_ref.py`) reproduces the shipped kernels; the
residual `rel ~ 2.7e-2` at `cos ~ 0.9997` is the kernel's internal FP8 Q
quantisation, and is the same for every shape.

In situ, the failure surfaces about six minutes into an 8-GPU launch, during
CUDA-graph capture, after 476 GiB of weights have been loaded:

```
File ".../vllm/models/deepseek_v4_1/nvidia/flashinfer_sparse.py", line 895, in _forward_prefill
  flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
File ".../flashinfer/mla/_sparse_mla_sm120.py", line 394, in _paged_attention
  raise ValueError(
ValueError: SM120 sparse-MLA has no decode kernel for this shape: ...
```

## Where the requested shape comes from

`topk = 1152` is the width of the sparse-index buffer V4.1 hands to the SWA
segment: `2 * index_topk + window = 2*512 + 128`, sized for the widest
(ratio-2 CSA2) layer and shared by every layer, with `topk_length` carrying the
per-token true length. `page_block_size = 32` is the SWA cache's own page
geometry; the `--block-size 128` that vLLM's DSv4.1 backend requires is the
*logical* block size and is a different quantity.

## Why this is a small change

In `decode_dsv4_kernel.cuh`, `TOPK` appears exactly three times:

```c
int topk_len = topk_length_ptr ? __ldg(topk_length_ptr + t_idx) : TOPK;
topk_len = topk_len < 0 ? 0 : (topk_len > TOPK ? TOPK : topk_len);
const int32_t* idx_base = indices + (size_t)t_idx * TOPK;
```

— a default, a clamp, and an index-row stride. `PAGE_BLOCK_SIZE` appears once,
as `constexpr int pbs = PAGE_BLOCK_SIZE;`, and is consumed only as

```c
const int block_idx_g = idx / section_pbs;
const int local_idx_g = idx - block_idx_g * section_pbs;
... + (size_t)section_pbs * IO_STRIDE + ...
```

Neither sizes shared memory (the smem block is a function of `HPB`, `DSV4_BI`
and the `KVCacheTraits`), bounds a `#pragma unroll`, or reaches an MMA tile
shape. The KV loop is already chunked by the runtime `num_splits` /
`chunks_per_block`, not by `TOPK`.

The same file already takes the *extra* cache's page block size as a runtime
`pbs_extra`, and says why that is fine:

```c
// Page block size of THIS section. Main is compile-time constexpr (typ.
// 64); extra is runtime (DSv4 C128A passes 2). The 8-cycle runtime div
// is dwarfed by the cp.async.bulk that follows.
```

The patch applies that same reasoning to the main cache.

## Patch

`upstream/flashinfer/apply_patch.py` (idempotent, `--revert`-able, `--check`
reports state). It:

1. drops `TOPK` and `PAGE_BLOCK_SIZE` from `sparse_mla_decode_dsv4_kernel`'s
   template parameters and passes them as trailing `int topk, int pbs`;
2. collapses the 25-entry `DSV4_DISPATCH(H, K)` macro list to a 5-entry
   `DSV4_DISPATCH(H)`, and replaces `page_block_size != 64 → return false` with
   `topk <= 0 || page_block_size <= 0 → return false`;
3. relaxes `_decode_dsv4_dispatchable` in `_sparse_mla_sm120.py` accordingly,
   keeping `_DECODE_DSV4_DISPATCH` present because downstream callers probe it
   (vLLM's `has_flashinfer_sparse_mla_sm120_config` does exactly that).

Point 3 is the part worth designing rather than copying: a capability that is
now "any top-k" is badly expressed as a set of pairs. A published predicate —
`sparse_mla_sm120_decode_supported(num_heads, topk, page_block_size)` — would
let consumers ask the question directly instead of reaching into a private
frozenset.

## Validation

`probe_sm120.py --sweep` over `page_block_size` in {16, 32, 64, 128} x
`num_heads` in {8, 16, 32, 64, 128} x `topk` in {128, 192, 256, 512, 1024,
1152, 2048}:

```
141 ok, 0 failed of 141
worst rel: h=64 topk=128 pbs=128 rel=3.672e-02
worst cos: h=16 topk=128 pbs=128 cos=0.999685
numerical mismatches: 0
```

Every shape now dispatches, and every one agrees with the reference to the
accuracy the shipped shapes already achieve. The previously shipped shapes are
unchanged, bit for bit against the same reference.

## A documentation gap found the same way

`out_lse` comes back in **base 2**, not natural log. Nothing says so: the
parameter is named `out_lse`, the docstring calls it "LSE", and the
`_decode_scratch_views` error message calls `mid_lse` "fp32 LSE scratch". The
offset is easy to mistake for an accuracy problem because it grows with `topk`:

| topk | `kernel_lse - ln(sum exp)` | `ln(topk) x (1/ln2 - 1)` |
| ---: | ---: | ---: |
| 128 | 2.18 | 2.15 |
| 256 | 2.42 | 2.46 |
| 512 | 2.78 | 2.76 |
| 1024 | 3.07 | 3.07 |

It is self-consistent — the split-K merge kernel uses the same base throughout —
so nothing is wrong today, and vLLM's sm_120 path passes `return_lse=False` and
never sees it. But any caller that merges these values with its own natural-log
LSE (chunked prefill, ring attention, a speculative-decode verifier) would be
silently wrong. One sentence in the docstring, or a rename to `out_lse2`, closes
it.

## A second, unrelated snag found while testing this

Editing `flashinfer/data/csrc/*.cu` has no effect while the `flashinfer-jit-cache`
wheel is installed: `JitSpec.build_and_load()` resolves to

```
site-packages/flashinfer_jit_cache/jit_cache/<module>/<module>.so
```

before consulting the source tree, and clearing
`~/.cache/flashinfer/<version>/<arch>/cached_ops/<module>` does not change that.
The stale kernel then runs silently. Moving the module's directory out of
`jit_cache/` forces a real build. A one-line note in the contributor docs, or an
`FLASHINFER_FORCE_JIT=1` escape hatch, would save the next person an hour.
