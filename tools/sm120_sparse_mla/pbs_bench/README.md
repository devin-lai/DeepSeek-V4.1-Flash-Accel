# MG prefill page-size micro-benchmark

Measures what a *runtime* main-cache page block size costs in FlashInfer's
SM120 sparse-MLA MG prefill kernel, versus the stock compile-time
`PAGE_BLOCK_SIZE`. Written to verify the review comment on FlashInfer
[PR #5204](https://github.com/flashinfer-ai/flashinfer/pull/5204); results and
discussion are in
[`benchmarks/results/2026-09-15-sm120-prefill-page-size.md`](../../../benchmarks/results/2026-09-15-sm120-prefill-page-size.md).

| file | role |
| --- | --- |
| `make_variants.py` | copies the installed `sparse_mla_sm120` headers and rewrites the MG kernel so the page size can be a kernel argument (`SMLA_PBS_MODE` 0 compile-time, 1 generic division, 2 pow2 shift/mask, 3 fast divmod; `SMLA_IO_MAXNREG` / `SMLA_MATH_MAXNREG` change the `setmaxnreg` split) |
| `bench_mg.cu` | standalone harness: packed DSv4 KV pool, index rows, launch, CUDA-event timing, output hashes |
| `build.sh` | builds `pristine` (stock headers) plus any list of variants, keeping `ptxas -v` logs |
| `run_matrix.py`, `summarize.py` | shape × page-size sweep and the comparison table |
| `parse_ptxas.py`, `ptxas_table.py` | registers / spills per kernel instantiation |

```bash
FI=$(python -c "import flashinfer,os;print(os.path.dirname(flashinfer.__file__))")
SRC=$FI/data/include/flashinfer/attention/sparse_mla_sm120
bash build.sh "$SRC" out pristine mode0 mode1 mode2 mode3 mode1_io40_math224
CUDA_VISIBLE_DEVICES=0 python run_matrix.py out pristine mode0 mode1 mode2 mode3 mode1_io40_math224
python summarize.py out/results.jsonl
```

Needs an idle SM120 GPU with a few GB free (the largest shape holds 2048 × 128
heads of Q and output) and the CUDA 13 toolkit. Every variant must reproduce
the `pristine` output hash at the same page size; a `*` in the table marks a
mismatch.
