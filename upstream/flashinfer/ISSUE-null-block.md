# SM120 sparse-MLA gathers row 0 of block 0 for every masked index, so the null block must be finite

**Component:** `include/flashinfer/attention/sparse_mla_sm120/common/kv_cache_io.cuh`
(`io_bulk_gather_tile`, `io_gather_scales`),
`.../prefill_kernel.cuh` (`prefill_kv_entry_base` and the two inline clamps),
`.../decode_dsv4_kernel.cuh` (`issue_gather`)
**Version:** flashinfer 0.6.18.post1
**Hardware:** 8x RTX 5090 D (sm_120), DeepSeek-V4.1-Flash under vLLM main
**Fix:** attached (`apply_patch.py`, edit group FI-004), verified on the model

## Symptom, as seen from vLLM

With CUDA graphs enabled, DeepSeek-V4.1 answers every request with non-finite
logits. The forward that produces them is a plain eager prefill; it does not
replay a graph. It fails at the *first* attention layer, and it fails for a
very specific set of query tokens:

```
prompt: 132 tokens, single request, positions 0..131, layer 0 (sliding window)
raw attention output finite for tokens: [63, 127, 128, 129, 130, 131]
non-finite for the other 126
```

Tokens 63 and 127 are the last token of a 64-token block; 128..131 have a full
128-entry window. Every one of them has a sliding-window index row that fills
whole 64-entry tiles. Every other token has a partial last tile, padded with
`-1`.

## What the kernel does with a `-1`

```c
// kv_cache_io.cuh, io_bulk_gather_tile (and identically in io_gather_scales)
int idx = indices[bi];
idx = (idx >= 0) ? idx : 0;
...
src = kv_ptr + (size_t)block_idx * stride_kv_block + (size_t)local_idx * IO::IO_STRIDE;
cp_async_bulk_g2s(dst + bi * SMEM_STRIDE, src, COPY_BYTES, mbar);
```

`prefill_kv_entry_base` and the decode kernel's `issue_gather` do the same
(`const int idx = (idx_raw >= 0) ? idx_raw : 0;`). A masked entry is clamped to
index 0 and its row -- row 0 of block 0 -- is copied into shared memory like any
other. The score of that entry is then forced to `-1e30`, so its softmax weight
is exactly zero, and the row is expected to fall out of the P·V product.

That only works if the row is finite. `0 * NaN` is `NaN`, and one NaN byte in
the 512-byte FP8 payload (or in the bf16 RoPE tail, which the decode kernel
does not scale) poisons the whole output row for that query. Tokens whose rows
have no `-1` never touch block 0, which is the pattern above.

## Why block 0 is not finite under vLLM

vLLM never hands block 0 to a request; it is the null block that padded and
dummy work is allowed to hit. Its plain KV inserts skip `PAD_SLOT_ID`, but
DeepSeek-V4.1's compressed-KV insert, indexer-K store and fp32 compressor state
ring derive their slots from the *block table*, and the dummy forwards run for
CUDA-graph capture use an all-zero block table. So capture writes fp8 and fp32
bytes into block 0 of caches that share those bytes with layer 0's
sliding-window cache (hybrid groups overlay one buffer). Read back after
capture on this box:

```
layer-0 SWA cache, block 0: 8143 non-zero bytes over rows 0..7
row 0: 581/584 bytes non-zero, 2 of the 512 FP8 payload bytes are NaN
```

Before capture the buffer is `torch.zeros`, which is why `--enforce-eager` is
correct: nothing ever writes block 0 in that mode.

The vLLM half of this -- dummy runs that write through block-table-derived
slots -- is reported separately (`upstream/vllm/ISSUE-v41-cudagraphs.md`). It
is still worth fixing here, because a kernel that reads a row it has been told
to ignore has a contract nobody else can be expected to keep: any runtime that
ever writes anything into block 0 will trip it.

## Fix

Gather a row that is known to be finite instead of row 0. Index rows are
compacted (valid entries first; the DSv4 producers in vLLM build them that
way), so the first entry of any tile the kernel iterates is a real row:

```c
int idx = indices[bi];
if (idx < 0) idx = indices[0];   // masked entry: any finite row will do
idx = (idx >= 0) ? idx : 0;
```

and in the decode kernel's gather, `section_idx_base[g_start]` for the chunk.
The clamp to 0 stays as a last resort for a fully-invalid row, which no caller
produces. No arithmetic changes for valid entries; masked entries still get
weight zero -- they now multiply a finite row by it.

Five sites, all the same two lines: `io_bulk_gather_tile`, `io_gather_scales`,
`prefill_kv_entry_base`'s callers (three inline clamps in `prefill_kernel.cuh`)
and `issue_gather` in `decode_dsv4_kernel.cuh`. `apply_patch.py` in this
directory carries them as edit group FI-004.

## Reproducing without vLLM

Take any DSv4 sparse-MLA case with `topk_length` shorter than the row, write a
NaN byte into row 0 of block 0 of the KV pool, and compare the output for a
query whose row has a masked entry against the same query with a zeroed row 0.
The patched kernel is unaffected by the contents of row 0.
