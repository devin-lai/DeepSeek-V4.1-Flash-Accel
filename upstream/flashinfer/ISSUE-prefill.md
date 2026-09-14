# SM120 sparse-MLA prefill: `TOPK` and `PAGE_BLOCK_SIZE` are compile-time, and DeepSeek-V4.1 lands outside the grid on both

**Component:** `csrc/sparse_mla_sm120_prefill.cu`,
`include/flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh`
**Version:** flashinfer 0.6.18.post1
**Hardware:** 8x RTX 5090 D (sm_120), DeepSeek-V4.1-Flash under vLLM main

This is the prefill sibling of the decode-dispatch problem in
[`ISSUE.md`](ISSUE.md). Same shape, same cause, different file — and a second
restriction underneath it that the decode side does not have.

## Symptom

```
tvm.error.InternalError: Check failed: (ok) is false:
Unsupported sparse-MLA prefill configuration: model=DSV4 num_heads=8
topk=1152 page_block_size=32 topk_extra=0 extra_page_block_size=0
```

## The grid

`dispatch_dsv4_single` instantiates

```
NUM_HEADS       in {8, 16, 32, 64, 128}
TOPK            in {128, 192, 256, 512, 1024, 2048}   (BF16 below 512, FP8 above)
PAGE_BLOCK_SIZE == 64                                  (literal in all launches)
```

and `dispatch_dsv4_dual` — the path every compressed layer takes — narrows
`TOPK` to a single value:

```c
if (topk != 128) return false;
if (extra_page_block_size == 64) { DISPATCH_BY_NH_PBSX(64); }
else if (extra_page_block_size == 2) { DISPATCH_BY_NH_PBSX(2); }
return false;
```

DeepSeek-V4.1 asks for `topk = 1152` (vLLM's SWA prefill index width,
`sliding_window + vision_max_n_token` = 128 + 1024) and `page_block_size = 32`
(the SWA cache's block size).

## `TOPK` is not a work bound

Inside `prefill_kernel.cuh`, the tile count comes from the **runtime** length,
not from `TOPK`:

```c
int topk_len = cold.topk_length ? __ldg(cold.topk_length + s_i) : TOPK;
topk_len = topk_len < 0 ? 0 : (topk_len > TOPK ? TOPK : topk_len);
const int actual_ni = ASSUME_FULL_TILES ? NI : ((topk_len + BI - 1) / BI);
...
for (int ti = 0; ti < actual_ni; ti++) { ... }
```

`TOPK` appears only as

- the index row stride, `idx_base = indices + (size_t)s_i * TOPK`,
- the clamp ceiling for `topk_len`,
- `NI = TOPK / BI`, the main/extra split point in the dual-cache loop.

It does not size shared memory (`SmemLayoutMG<MT, CM>::TOTAL` does not mention
it), bound a `#pragma unroll`, or reach a tensor-core tile shape. The file's own
header comment — "Iterates over ALL NI = TOPK/BI tiles (no split)" — describes
`ASSUME_FULL_TILES`, not the general path.

The practical consequence is that a caller can already dodge the single-cache
restriction from outside the kernel: allocate the index rows at the next
instantiated `TOPK` and pass `topk_length`. The padding is allocated and never
read, and with `BI = 64` any width that is a multiple of 64 (1152 is 18 tiles)
keeps every tile inside the real data. That is what this repo does, and the
single-cache layers then dispatch and are numerically correct.

There is no such escape for the dual-cache path, because its restriction is
`topk == 128` exactly, not "one of a list".

## `PAGE_BLOCK_SIZE` is div/mod

```c
template <ModelType MT, int PAGE_BLOCK_SIZE>
... prefill_kv_entry_base(...) {
    const int bi = idx / PAGE_BLOCK_SIZE;
    const int li = idx % PAGE_BLOCK_SIZE;
```

plus the same value threaded into `io_gather_scales`, `io_bulk_gather_tile` and
`xv_rope_mma`. As on the decode side, nothing structural depends on it; the
kernel already takes the *extra* cache's page block size as a runtime argument
in the dual-cache variants.

## Suggested fix

Mirror what [`ISSUE.md`](ISSUE.md) proposes for decode:

1. Make `TOPK` and `PAGE_BLOCK_SIZE` runtime arguments of the prefill kernels.
   The single-cache DSV4 grid collapses from 30 instantiations (5 head counts x
   6 topks) to 5, and every `(topk, page_block_size)` becomes dispatchable.
2. Failing that, at minimum instantiate the dual-cache path at the same `TOPK`
   list as the single-cache path. One value is not a dispatch table; it is a
   hard-coded model assumption, and it is the only thing keeping DeepSeek-V4.1
   from serving images on consumer Blackwell.
3. Independently: the error is raised after the launch decision, so it cannot
   say *which* axis failed. `num_heads=8 topk=1152 page_block_size=32` is three
   numbers and a reader has to find the grid in the source to learn that only
   the last two are wrong, and that one of them has a list while the other has
   a single legal value.

## What this repo did instead

No kernel change. vLLM was patched to (a) round the prefill index width up to
an instantiated `TOPK`, and (b) ask the SWA cache for 64-token pages, which is
what these kernels have always assumed. With those, DeepSeek-V4.1-Flash serves
text correctly on 8x RTX 5090. Image serving still needs item 2 above.
