# Runtime page size in the SM120 sparse-MLA MG prefill kernel

Experimental verification of a review comment on FlashInfer PR
[#5204](https://github.com/flashinfer-ai/flashinfer/pull/5204) ("support runtime
sparse decode page sizes"), where a maintainer noted:

> Runtime page size has large impacts on MG prefill performance. Most can be
> mitigated by reassigning registers but not all of them.

The related refactor ([#5197](https://github.com/flashinfer-ai/flashinfer/pull/5197))
keeps compile-time page divisors and reports, from its own earlier measurements,
that rejected runtime-page variants cost "up to +30% with generic division and
+4–7% with pow2 shift/mask on the heaviest single-cache shape".

This report reproduces that experiment on an RTX 5090 (SM120) with the kernel
sources that this repository deploys (FlashInfer 0.6.18.post1 plus the
`upstream/flashinfer` patches), and adds two variants: a register split
change and a fast-divmod (multiply-high) decomposition that supports
non-power-of-two page sizes without integer division.

## What the page size does inside the kernel

`sparse_mla_prefill_mg_kernel<MT, CM, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE, MG_N_HG>`
is warp-specialised: 4 IO warps shrink to 32 registers
(`setmaxnreg.dec 32`) and 8 math warps grow to 232 (`setmaxnreg.inc 232`).
`PAGE_BLOCK_SIZE` is used only to turn a global slot index into a
`(page, offset)` pair:

| site | warps | per tile of 64 entries |
| --- | --- | --- |
| `io_bulk_gather_tile` | IO (32 regs) | 64 div/mod for the `cp.async.bulk` sources |
| `io_gather_scales` | IO (32 regs) | 64 div/mod plus the footer offset `pbs * 576` |
| `prefill_kv_entry_base` | math | 1 div/mod per thread (rope prefetch) |
| `xv_rope_mma_mg` | math | 4 div/mod per thread per 16-entry step |

With a compile-time power of two these are shifts and masks. With a runtime
divisor they become either a ~20-instruction integer division sequence
(generic), a shift by a runtime amount (pow2 only), or one `mul.hi` plus a
shift (fast divmod, any divisor).

## Method

`tools/sm120_sparse_mla/pbs_bench/` (this repository) contains:

- `make_variants.py` — copies the installed `sparse_mla_sm120` header tree and
  rewrites the MG kernel so the main-cache page size is carried at run time in
  `PrefillColdParams` (a `__grid_constant__` struct). `SMLA_PBS_MODE` selects
  compile-time (0, the control), generic division (1), pow2 shift/mask (2) or
  fast divmod (3). `SMLA_IO_MAXNREG` / `SMLA_MATH_MAXNREG` change the
  `setmaxnreg` split.
- `bench_mg.cu` — a standalone harness that builds a DSv4 packed KV pool
  (448 B FP8 nope + 128 B BF16 rope per token, 8 B UE8M0 scale footer per token
  at the end of each page), sliding-window or random index rows, launches the
  kernel exactly as `csrc/sparse_mla_sm120_prefill.cu` does, times 20 launches
  with CUDA events and hashes the outputs.
- `build.sh`, `run_matrix.py`, `summarize.py`, `parse_ptxas.py`.

Control: the unmodified headers (`pristine`). The patched tree compiled in
mode 0 must reproduce the control bit for bit; every runtime variant must
match the compile-time kernel at the same page size bit for bit (the
arithmetic on the data is unchanged; only address computation differs).

Shapes: the deployment's TP8 shape (8 heads, top-k 128, BF16 compute, single
and dual cache) up to the heaviest single-cache shape in the dispatch grid
(128 heads, top-k 2048, FP8 compute). Page sizes 32 and 64 have compile-time
controls; 16 and 128 are runtime-only.

## Results

RTX 5090 D, CUDA 13.2, `-gencode=arch=compute_120f,code=sm_120f -O3`
(FlashInfer's own flags), idle GPU, median of 20 launches. Every variant
reproduced the control's output and LSE hashes bit for bit at every page size
(the `*` marker never appeared), and the patched tree compiled in mode 0 is
indistinguishable from the pristine headers.

Kernel time relative to the compile-time kernel at the same page size
(page 64 shown; page 32 gave the same ratios). Full table with absolute
microseconds and the page-16/128 rows: [`summary.txt`](2026-09-15-sm120-prefill-page-size/summary.txt);
raw runs: [`results.jsonl`](2026-09-15-sm120-prefill-page-size/results.jsonl).

| shape (queries) | control µs | generic div | generic div, IO 40 regs | generic div, IO 48 regs | pow2 shift/mask | fast divmod |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16, 8 heads, top-k 128, window (4096) | 224 | 1.13 | 1.15 | 1.15 | 1.00 | 1.01 |
| BF16, 16 heads, top-k 128, window (4096) | 318 | 1.07 | 1.08 | 1.08 | 1.00 | 1.00 |
| FP8, 32 heads, top-k 512, random (4096) | 1,323 | 1.22 | 1.21 | 1.20 | 1.00 | 1.04 |
| FP8, 64 heads, top-k 2048, random (2048) | 3,830 | 1.34 | 1.34 | 1.32 | 1.02 | 1.06 |
| FP8, 128 heads, top-k 2048, random (2048) | 7,455 | 1.35 | 1.34 | 1.32 | 1.03 | 1.06 |
| dual cache, BF16, 8 heads, 128 + 512 extra (4096) | 795 | 1.08 | 1.08 | 1.08 | 1.01 | 1.00 |
| dual cache, BF16, 32 heads, 128 + 512 extra (4096) | 1,398 | 1.06 | 1.06 | 1.07 | 1.01 | 1.01 |
| dual cache, 8 heads, *extra* page size also runtime | 795 | 1.36 | — | — | — | 1.04 |
| dual cache, 32 heads, *extra* page size also runtime | 1,398 | 1.22 | — | — | — | 1.04 |

"IO 40/48 regs" moves the `setmaxnreg` split from 32/232 to 40/224 and
48/224 registers for the IO and math warps.

Findings, in the order of the review comment:

1. **The impact is real and large for generic division.** Passing the page
   size as a plain runtime divisor costs +13% on the 8-head sliding-window
   shape this deployment runs, +7% at 16 heads, +22% at 32 heads / top-k 512,
   and **+34–35% on the heaviest single-cache shapes** (64 and 128 heads,
   top-k 2048, FP8). That matches the "+30%" the maintainer reported and
   confirms the comment on the kernel this repository ships.
2. **The cost is the arithmetic, not the page size.** A runtime page size of
   16 or 128 runs at exactly the speed of 64 in every runtime variant, and the
   compile-time kernel at page 32 is as fast as at page 64. There is no
   inherent penalty for smaller or larger pages in this kernel; the penalty
   comes from how `(page, offset)` is computed.
3. **Most of it can be removed, but not all.** Restricting runtime sizes to
   powers of two and shifting instead of dividing brings the overhead down to
   0–3%; a CUTLASS-style fast divmod (one multiply-high plus a shift, any
   divisor) costs 1–6%. The residual on the FP8 top-k 2048 shapes stays
   measurable in both. The maintainer's "+4–7% with pow2 shift/mask" is the
   same picture; the shift variant here lands slightly lower.
4. **Register reassignment did not rescue generic division here.** Giving
   the IO warps 40 or 48 registers (and the math warps 224) changed the
   generic-division penalty by at most three points on the heaviest shapes
   (1.34 → 1.32) and made the small BF16 shapes slightly slower (1.13 →
   1.15). In this kernel the divisor arithmetic is not limited by the
   32-register IO budget; the extra instructions in the gather loop and in
   the math warps' rope-index decomposition are the cost. Whatever
   reassignment the maintainer applied to the newer `prefill_mg_kernel.cuh`
   on `main` (a different kernel from the one measured here) does not
   transfer to this one; the arithmetic change does.
5. **Making the dual-cache *extra* page size runtime is far more expensive.**
   With generic division on both caches the dual-cache kernel loses 22–36%
   (the extra cache carries 512 of the 640 entries), while fast divmod holds
   it to 4%. PR #5204 only makes the main-cache page size runtime, and its
   decode kernel, which is not measured here, has a different structure
   (split-K over top-k chunks, no `setmaxnreg` split); the prefill result
   supports keeping the extra page size compile-time as #5197 does, or
   using a multiply-high decomposition if it must be runtime.

The IO warps run under `setmaxnreg.dec 32`, so any extra live values for the
divisor arithmetic compete with the 64-entry gather loop; the math warps pay
again in `xv_rope_mma_mg`, where every 16-entry step decomposes four indices.
`ptxas -v` reports the launch-bound cap of 168 registers for every variant
and does not expose the per-warp-group allocation, so timing, not the
register report, is the evidence.

## Reproducing

```bash
FI=$(python -c "import flashinfer,os;print(os.path.dirname(flashinfer.__file__))")
cd tools/sm120_sparse_mla/pbs_bench
bash build.sh "$FI/data/include/flashinfer/attention/sparse_mla_sm120" out \
  pristine mode0 mode1 mode2 mode3 mode1_io40_math224 mode1_io48_math224 \
  mode2_io40_math224 mode3_io40_math224 mode1_xrt mode3_xrt
CUDA_VISIBLE_DEVICES=0 python run_matrix.py out pristine mode0 mode1 mode2 mode3 \
  mode1_io40_math224 mode1_io48_math224 mode2_io40_math224 mode3_io40_math224 mode1_xrt mode3_xrt
python summarize.py out/results.jsonl
python parse_ptxas.py out/ptxas/*.log > out/ptxas.jsonl && python ptxas_table.py out/ptxas.jsonl
```

The GPU must be idle and have a few GB free: the first attempt here ran next
to the serving engine and every large shape failed with out-of-memory. The
whole matrix takes about 40 minutes on one RTX 5090, mostly the 128-head
shapes.

## Relevance to this deployment

DeepSeek-V4.1-Flash on 8× RTX 5090 spends about 1.7% of an 8K-token prefill
in `sparse_mla_prefill_mg_dual_kernel` (torch profiler, rank 0, 2026-09-15);
the same profile is dominated by reading offloaded decoder experts over PCIe
and by NCCL all-reduces. A page-size change in this kernel therefore cannot
move end-to-end prefill throughput by more than a fraction of a percent here.
The measurements above matter for the FlashInfer dispatch design decision,
not for this repository's presets.
