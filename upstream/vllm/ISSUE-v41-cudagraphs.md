# CUDA-graph capture writes DeepSeek-V4.1 state into the null block, and the sm_120 sparse-MLA kernels read it back

**Component:** `vllm/v1/worker/gpu/model_runner.py` (`capture_model`),
`vllm/v1/attention/backends/mla/compressor_utils.py`,
`vllm/models/deepseek_v4_1/compressor.py`
**Version:** vLLM main @ `8c1d1c297`, FlashInfer 0.6.18.post1, torch 2.13+cu130
**Hardware:** 8× RTX 5090 (sm_120), DeepSeek-V4.1-Flash, TP8 + EP
**Status:** root cause established; patch attached (`apply_patch.py`, edit
`v1/worker/gpu/model_runner.py`); the kernel half is
`upstream/flashinfer/ISSUE-null-block.md`.

## Symptom

With CUDA graphs enabled the server starts normally, captures without
complaint, and answers every request with non-finite logits: the same token
id forever, `logprobs` rejected as "Out of range float values are not JSON
compliant", and an illegal memory access on longer prompts. `--enforce-eager`
on the identical build is correct. The first request of a fresh server is
already wrong, and it is an eager prefill (132 tokens, above every captured
size), so graph *replay* is not involved.

## Bisection

Each row is one boot of the real checkpoint, with a plugin that switches one
step off and records per-layer finiteness for every forward:

| configuration | first request |
| --- | --- |
| eager | correct |
| graphs, `profile_cudagraph_memory` skipped | non-finite |
| graphs, `capture_model` skipped entirely | correct |
| graphs, `capture_model` runs only its dummy **warm-up** forwards, records no graph, every real batch still eager | **non-finite** |
| graphs, full capture, then block 0 of every KV cache zeroed | correct (see below) |

So capture's dummy forwards corrupt state that a later eager forward reads.
Not the graphs; the forwards run to warm them up.

## Where the first NaN is

Layer 0 -- a pure sliding-window layer with no compressor and no indexer. Its
q, kv, weights, RoPE cache and embeddings are finite; the raw output of the
FlashInfer sparse-MLA prefill kernel is not, for 126 of 132 tokens. The six
finite tokens are 63, 127 and 128..131: exactly the queries whose
sliding-window index row fills whole 64-entry tiles. Every other row ends in a
partial tile padded with `-1`.

The kernel clamps a `-1` index to 0 and gathers row 0 of block 0 for it,
relying on a zero softmax weight to cancel the row. `0 * NaN` is `NaN`. Read
back after capture, block 0 of layer 0's SWA cache holds 8143 non-zero bytes
in rows 0..7 (it was `torch.zeros` at allocation) and row 0 has two NaN bytes
in its FP8 payload.

## Who wrote block 0

`capture_model`'s dummy batches use an all-zero block table and a slot mapping
of `PAD_SLOT_ID`. The plain KV inserts honour the pad and write nothing. Three
V4.1 writers do not look at the slot mapping at all:

- `DeepseekV4SparseMLAMetadataBuilder.build` /
  `DeepseekV32IndexerMetadataBuilder.build` derive the compressed-KV and
  indexer-K slots with `get_compressed_slot_mapping(...)` from the block table
  -> block 0;
- `CompressorMetadataBuilder.build` derives the fp32 compressor state ring's
  slot from `block_table[req, 0]` -> block 0.

Those caches share their bytes with the sliding-window caches (hybrid KV
groups overlay one buffer), so the fp32 state ring lands on top of layer 0's
block 0. Every forward after that -- eager or not -- gathers that row for every
masked index.

DeepSeek-V4 has none of these writers in the same shape, which is why it is
correct with graphs on the same build.

## Fixes

1. **This repo (attached).** After `capture_model` finishes, zero block 0 of
   every KV cache view. Block 0 is never allocated to a request, so this is
   free and exact; the real allocation started as zeros. Verified: graphs on,
   first request correct, 6/6 continuations, perplexity in the eager range.
2. **Kernel side (`upstream/flashinfer`, FI-004).** Gather the tile's first
   entry instead of row 0 for masked indices. Makes the kernels indifferent to
   the null block's contents, which is the contract every other consumer of
   `-1` assumes.
3. **Upstream-proper.** Give the block-table-derived slot mappings the same
   pad semantics as the token slot mapping: when `slot_mapping[t] < 0`, the
   compressed / indexer / state-ring slot for token `t` should be `-1` too.
   Then dummy runs write nothing anywhere, which is what the null block's
   design assumes.

Two things would have made this a five-minute bug instead of a two-day one:

- a correctness assertion after capture (one greedy decode, finite logits);
- a debug mode that fills the null block with NaN before capture and checks it
  after, so any writer that reaches block 0 is named at startup.

## Reproducing

```bash
vllm serve /path/to/DeepSeek-V4.1-Flash \
  --tokenizer-mode deepseek_v41 --tensor-parallel-size 8 --enable-expert-parallel \
  --engram-config '{"cpu_offload": true}' --cpu-offload-gb 12 \
  --cpu-offload-params w13_weight w2_weight \
  --block-size 64 --language-model-only --max-model-len 32768
curl -s localhost:8000/v1/completions -d '{"model":"...","prompt":"<130+ tokens>",
  "max_tokens":1,"temperature":0,"logprobs":1}'   # -> non-finite
```

Requires the other patches in this repository to start the model on sm_120 at
all; see `upstream/README.md`.
