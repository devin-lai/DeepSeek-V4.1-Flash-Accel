# 2026-09-14 — DeepSeek-V4.1-Flash serves on 8× RTX 5090

First working configuration. Raw artefacts in
[`2026-09-14-v41/`](2026-09-14-v41/); each row below names the run directory it
came from on the reference machine.

Stack: vLLM main @`8c1d1c297` + `upstream/vllm/apply_patch.py`, FlashInfer
0.6.18.post1 + `upstream/flashinfer/apply_patch.py`, torch 2.13+cu130, driver
595.71.05. 8× RTX 5090 D, 2× Xeon Gold 6530, 503 GiB DDR5.

Common flags: TP-8, `--enable-expert-parallel`, `--engram-config
'{"cpu_offload": true}'`, `--cpu-offload-params w13_weight w2_weight`,
`--block-size 64`, `--language-model-only`, `--max-model-len 32768`,
`--max-num-seqs 16`, `--gpu-memory-utilization 0.93`, Marlin MoE backend,
FlashInfer autotune off, no `--numa-bind`.

## Correctness

`deploy/verify.py`, eager, 12 GiB/rank offloaded:

| probe | result |
| --- | --- |
| greedy continuations | **6 / 6** |
| teacher-forced perplexity, 78 tokens of held-out English | **2.98** (`ladder/eager_off12-111435`), **2.59** (`G-nan-105018`) |
| "A shop sells pens at 3 for $2. How much do 12 pens cost?" | **$8** |

Control, same patched build, CUDA graphs **on**, DeepSeek-V4-Flash-NVFP4: 6/6,
perplexity 2.61. That is what makes VL-013 a V4.1 problem rather than a
regression from these patches.

### Greedy decoding is not run-to-run reproducible here

Two runs of the identical configuration produced different greedy
continuations — *"Celsius. Water boils at 100 degrees"* against *"Celsius.\\nWater
freezes at zero degrees"* — and perplexities 15 % apart. Both are sane; neither
is a fluke. The likely mechanism is that MXFP4 expert arithmetic is not
associative and the expert-parallel all-to-all does not fix a reduction order,
so a near-tie in the logits can flip. Worth knowing before treating a single
quality number from this stack as exact.

## Throughput, eager

`vllm bench serve`, random 1024-in / 128-out (`ladder/eager_off12-111435`):

| case | concurrency | out tok/s | total tok/s | median TTFT | median TPOT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 → 128 | 1 | 5.5 | 49.3 | 704 ms | 183.0 ms |
| 1024 → 128 | 8 | 40.9 | 368.3 | 2 487 ms | 174.5 ms |
| 1024 → 128 | 32 | 72.4 | 652.0 | 31 372 ms | 212.8 ms |
| 8192 → 1 | 2 | — | 2 411.7 | 6 329 ms | — |

Startup 285 s with the page cache warm (48 shards in 43 s). Host MemAvailable
low-water 12.7 GiB of 503.

**TPOT is flat across a 32× range of concurrency.** That is a fixed per-step
cost, not compute or bandwidth — 40 layers launched one kernel at a time
because CUDA graphs corrupt this model (VL-013).

## The offload floor

Same configuration, varying `--cpu-offload-gb` (`ladder/`):

| offload per rank | outcome |
| ---: | --- |
| 12 GiB | serves |
| 8 GiB | `ValueError: No available memory for the cache blocks` |
| 4 GiB | CUDA OOM during weight load |

`tools/plan_memory.py --tp 8 --ep` reports 12.0 for this configuration. It only
gets there by counting the vision tower as replicated rather than sharded, and
by adding a measured 1.6 GiB/rank for the MoE backend's repacked layout and
workspaces — bytes that are not in the checkpoint headers. Measured resident
weights are 24.09 GiB/rank alongside 12.39 GiB offloaded; the header arithmetic
alone predicts 34.9 against the 36.5 observed.

## CUDA graphs (VL-013)

Same configuration, varying `cudagraph_mode` (`gfx-*`):

| mode | boots | correct |
| --- | --- | --- |
| off (`--enforce-eager`) | yes, 285 s | **yes** |
| `PIECEWISE` | yes, 295 s | no — same token id repeated, non-finite logprobs |
| `FULL` | see `gfx-FULL-114641` | no |

`VLLM_USE_BREAKABLE_CUDAGRAPH=0` behaves identically to the default, so the
breakable-graph path is not implicated. A per-layer finiteness probe inside the
decoder loop reports nothing in eager mode, which is where the model is
correct.

## Where the offloaded experts should live

Three placements of the same 12 GiB/rank budget, everything else identical
(`ladder2/`, raw in [`2026-09-14-v41/ladder2-offload-placement.json`](2026-09-14-v41/ladder2-offload-placement.json)).
All three verified correct first.

| placement | out tok/s @1 | @8 | @32 | 8K prefill total tok/s | 8K TTFT | TTFT @8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stock walk order | 5.7 | 37.3 | 73.4 | 2 424.0 | 6 289 ms | 2 383 ms |
| **`DSV41_OFFLOAD_LAYERS=20-39`** (decoder) | 5.4 | **42.0** | **78.8** | **2 654.5** | **5 741 ms** | **2 245 ms** |
| `DSV41_OFFLOAD_LAYERS=0-19` (encoder) | 5.6 | 38.3 | 73.3 | 2 424.7 | 6 279 ms | 2 371 ms |

Two things fall out of this, and the first is the control that makes the second
believable:

**Stock walk order is encoder offload.** 2 424.0 against 2 424.7 tok/s and
6 289 against 6 279 ms — the same numbers. vLLM's offloader walks layers in
order and spends the budget on the first ~14, which on this model is the CED
encoder half. That is what the plugin was written to change, and it confirms
the mechanism rather than just the outcome.

**Offloading the decoder half is worth ~9 % of prefill.** +9.5 % prefill
throughput and −8.7 % TTFT on 8K prompts, −5.8 % TTFT at 8 concurrent streams.
Under CED, prefill runs only the encoder, so keeping those layers resident
takes PCIe out of the prefill path entirely; the decoder layers are paid for
during decode either way.

Decode moves less and less cleanly: +12.6 % at 8 streams and +7.4 % at 32, but
−5 % at 1. Given that two runs of an identical configuration differ by more
than that at low concurrency (above), the single-stream number should not be
read as a regression. The concurrent gains are consistent with chunked prefill
sharing steps with decode: a faster prefill leaves more of each step for it.

**Conclusion: `OFFLOAD_LAYERS=20-39` is right for this model**, and it is in
`deploy/presets/v41-flash.env`. It is worth nothing on a plain decoder — on
V4-Flash the same restriction measured 55.7 against 56.1 tok/s — so it is a
CED-specific lever, not a general one.
