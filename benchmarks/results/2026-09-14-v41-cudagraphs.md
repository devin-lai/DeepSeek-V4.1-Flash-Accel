# 2026-09-14 — VL-013: CUDA graphs on DeepSeek-V4.1-Flash, sm_120

Every row is one fresh boot of the real checkpoint on the 8× RTX 5090 box
(TP8 + EP, Engram on host, 12 GiB/rank offloaded, `--block-size 64`,
text-only unless stated). A throwaway vLLM general plugin (`dsv41diag`,
kept out of the repo) switched one startup step off per run and recorded
per-layer finiteness for every forward into `stats/rank<N>.jsonl`; run
directories are `/data/runs/diag/R*/` on the box.

"First request" is a 132-token eager prefill (the Antikythera paragraph),
sent before any other request, so no captured graph can apply to it.
`verify.py` = 6 greedy continuations, teacher-forced perplexity, a word problem.

| run | configuration | first request | verify |
| --- | --- | --- | --- |
| R5 / R8 | `--enforce-eager` | correct | SANE |
| R2 | graphs; `profile_cudagraph_memory` skipped | non-finite | NOT SANE |
| R3 | graphs; `capture_model` skipped entirely | correct | SANE |
| R4 / R7 | graphs, stock | non-finite | NOT SANE |
| R6 | graphs; `capture_model` runs only its dummy **warm-up** forwards, records no graph, every real batch eager | **non-finite** | NOT SANE |
| R9 | graphs; block 0 of every KV cache zeroed after capture (the VL-013 vLLM patch, as a plugin) | correct | SANE |
| R10 | graphs; FlashInfer FI-004 gather fix only, kernels rebuilt | correct | SANE |
| R11 | graphs; both patches | correct | SANE |
| R12 | graphs; both patches; vision tower loaded, `--limit-mm-per-prompt image=1`; a 224×224 image request (231 prompt tokens) | correct; image request dispatched (FI-003 dual-cache at topk 2048) | SANE |

## Where the first non-finite value is (R4, R7)

First post-capture forward, rank 0, layer 0 (pure sliding-window layer):

| tensor | non-finite |
| --- | --- |
| embeddings, attention input, `fused_wqa_wkv` output, q, kv | 0 |
| layer-0 weights, `attn_sink`, RoPE cos/sin cache | 0 |
| raw sparse-MLA prefill output | **126 of 132 tokens** — finite tokens are 63, 127, 128, 129, 130, 131 |
| attention output after `o_proj` | 126 of 132 tokens |

Tokens 63 and 127 are the last of a 64-token block; 128..131 have a full
128-entry window. Every finite token's index row fills whole 64-entry tiles;
every non-finite token's row ends in a partial tile padded with `-1`.

## Block 0 of layer 0's SWA cache (dumps `L0_rank0_fwd*.pt`)

| | non-zero bytes in block 0 | NaN FP8 bytes in row 0 |
| --- | ---: | ---: |
| R8, eager | 0 | 0 |
| R7, after capture | 8143 (rows 0..7) | 2 |

The KV buffer is `torch.zeros` at allocation. The 8 KB pattern is the
compressor's fp32 state ring, written through the all-zero dummy block table
of capture's warm-up forwards; the kernels gather row 0 for every masked
index and multiply it by a zero weight.

## With graphs on: the kit, as shipped

`PRESET=v41-flash deploy/serve.sh` (graphs on, both patches), `vllm bench
serve`, random 1024-in / 128-out, same cases as the eager table in the README:

| | 1 stream | 8 streams | 32 streams | 8K prefill (2 streams) |
| --- | ---: | ---: | ---: | ---: |
| output tok/s | 33.7 | 92.4 | 115.8 | — |
| total tok/s | 303.0 | 831.3 | 1 042.6 | 2 650.2 |
| median TTFT | 692.1 ms | 2 167.1 ms | 19 688.4 ms | 5 739.2 ms |
| median TPOT | 24.04 ms | 72.28 ms | 128.91 ms | — |

Eager, same kit, earlier the same day (`2026-09-14-v41-first-serve.md`): 6.0 /
38.2 / 73.7 output tok/s, TPOT 159.5 / 194.6 / 194.1 ms, prefill 2 658 tok/s.
Capture: 6 s and 0.09 GiB per rank; `init engine` 60.6 s. `verify.py` on the
graphs server: 5/6 continuations, perplexity 2.662, "$8". Raw JSON:
`2026-09-14-v41/kit-v41-graphs-bench.json`, `kit-v41-graphs-verify.json`.
