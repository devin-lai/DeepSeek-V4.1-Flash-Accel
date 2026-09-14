# DeepSeek-V4.1-Flash anatomy, from a deployer's point of view

This is the part of the tech report and checkpoint that decides how the model
can be served. Numbers come from `config.json`, the safetensors headers
(`tools/plan_memory.py`) and the technical report shipped with the weights.

## The shape of the network

| Property | Value |
| --- | --- |
| Layers | 40 (20-layer **causal encoder** + 20-layer **decoder**), plus 3 DSpark drafter blocks |
| Hidden size | 5120, with mHC keeping 4 residual streams (`hc_mult = 4`) |
| Experts | 384 routed + 1 shared per layer, top-6, `moe_intermediate_size = 2304` |
| Attention | 64 heads, one 512-dim KV latent (MLA style), 64-dim RoPE part, sliding window 128 |
| Sparse attention | CSA2: Top-512 selected entries per query, indexer with 32 heads × 128 dim |
| Context | 1,048,576 tokens (YaRN ×16 over a 65,536 base) |
| Engram | 2 modules (layers 1 and 14), 384M rows × 256 dims each, FP8 |
| Vision | 32-layer DeepSeek-ViT (1024 wide), 3×3 pixel-unshuffle, ≤1024 image tokens |
| Params | 552B backbone + 196B Engram; **8B active per prefill token, 16B per decode token** |

### Why prefill is cheaper than decode (CED)

The decoder's global KV is projected from the *encoder's* final hidden state,
not from each decoder layer. During prefill only the 20 encoder layers run
over the prompt; the decoder only replays the last 128 tokens (Decoder SWA
Bounded Replay) to rebuild its sliding-window KV. That is why the recipe's
8B/16B split exists and why prefill throughput on this model is unusually high
relative to decode.

### Why the KV cache is tiny (CSA2 + FP4 KV)

Only four layers (`kv_source_layer_ids = [2, 8, 14, 20]`) write global KV.
Eight layers (`index_source_layer_ids`) run an indexer; all others reuse the
Top-K indices of the most recent indexing layer. Main KV is stored as E2M1 with
one E4M3 scale per 16 channels. Result: **≈ 890 bytes per token** of global KV.
A 1M-token context costs well under 6 GiB across all ranks; KV capacity is not
the constraint on this hardware — weights are.

The decoder's first Full-mode layer (layer 20) also builds a 2048-block × 8
candidate pool that later Reindex layers search, so per-token indexer cost is
bounded regardless of context length.

### The two compression ratios, and why they reach the launch flags

`compress_ratios` in `config.json` is per layer:

```
[0, 0,  2 ×18 (layers 2-19),  1 ×20 (layers 20-39),  0, 0, 0]
```

0 is a pure sliding-window layer. **2 means the CED encoder half compresses two
tokens into one indexer state; 1 means the decoder half compresses none.** That
is an anatomy fact with a direct deployment consequence: the DSA indexer kernel
constrains *states* per block, so one token-block size gives the two halves
different state counts and at most one of them can be legal. It is the whole of
[VL-009](05-fault-inventory.md), and it is why `--block-size` on this model
means indexer states per block rather than tokens.

If you are porting this to another DeepSeek-V4-family checkpoint, read
`compress_ratios` first: DeepSeek-V4-Flash alternates 4 and 128, which vLLM
handles by giving each ratio its own layer *type*, and V4.1 is the first to mix
ratios that share one.

### DSpark (speculative decoding)

Three extra transformer blocks with a 128-token window draft **5 positions in
one pass**, a Markov head models dependencies between them and a confidence
head predicts per-position acceptance. vLLM uses the confidence head for
*adaptive verification*: it profiles step cost at startup and admits only the
draft slots that are worth verifying at the current load. For a memory-bound
multi-GPU box this should be the single largest decode speed-up available — it
is **unmeasured here** — CUDA graphs only became usable on sm_120 once
[VL-013](05-fault-inventory.md) was fixed, and DSpark has not been measured
since — and it is
free in quality terms (draft tokens are verified against the target).

## Where the bytes are (510 GB checkpoint)

From `tools/plan_memory.py` on the released checkpoint (GiB = 2^30):

| Component | GiB | Share | Where it goes on 8× 5090 |
| --- | ---: | ---: | --- |
| Routed experts, MXFP4 + ue8m0 block scales (incl. DSpark experts) | 275.7 | 58 % | GPU, **~1/3 spilled to pinned host RAM** |
| Engram tables + projections, FP8 | 189.1 | 40 % | pinned host RAM (UVA lookups) |
| Attention / dense / routers / norms, FP8 | 5.8 | 1.2 % | GPU |
| Token embeddings + LM head, BF16 | 2.5 | 0.5 % | GPU |
| Shared experts, FP8 | 1.4 | 0.3 % | GPU |
| Vision encoder + projector | 0.8 | 0.2 % | GPU (or skipped with `--language-model-only`) |
| **Total** | **475.2** | | |

Two facts follow directly:

1. **Engram must be on the host.** 189 GiB of tables that are read by hash
   lookup (48 rows × 256 bytes per token) belong in DRAM; the tech report
   itself prefetches them from host memory. vLLM's `--engram-config
   '{"cpu_offload": true}'` keeps the TP-sharded table in pinned memory and
   gathers rows over PCIe with a Triton kernel.
2. **Even without Engram the GPU part (286 GiB) does not fit in 251 GiB of
   GPU memory.** Per TP-8 rank that is 35.8 GiB of weights against ~25 GiB usable
   after KV cache, CUDA graphs and workspace. About 11 GiB per rank (a third of
   the routed experts) has to be served from host memory as well.

## Token-level cost model for decode

Per decode token, per layer: 6 routed experts × 3 matrices × 2304 × 5120 ×
0.53 bytes ≈ 18.8 MB of MXFP4 expert weights; × 40 layers ≈ **750 MB per
token**. Across 8 ranks that is ~94 MB per rank per token from GPU memory at 1.8 TB/s
(≈ 0.05 ms) — the model is *not* weight-bandwidth-bound on a 5090 at batch 1.
What dominates is:

- ~100 small all-reduces per step over PCIe/SHM (≈ 8 ms on this box),
- kernel launch count per layer (vLLM runs CUDA graphs for decode; keep them),
- the fraction of expert bytes that live behind PCIe: at 33 % offload and
  ~20 GB/s effective UVA bandwidth this adds ≈ 1.5 ms per token per rank at
  batch 1, growing with batch as more unique experts are touched.

Prefill is compute-bound (MXFP4 Marlin GEMMs + FP8 attention on the encoder
half only) and scales well with chunk size.

## Checkpoint layout notes

- 48 shards; shards 47 and 48 (~100 GB each) hold the two Engram tables.
- `quantization_config`: FP8 dense with 32×32 block scales (`ue8m0`), FP4
  experts (`expert_dtype: fp4`, OCP MXFP4).
- No Jinja chat template: prompts are built by the `encoding/` reference or by
  vLLM's `deepseek_v41` tokenizer mode / parsers.
- Tokenizer: 129,280 vocab; Engram hashes n-grams over a normalized
  99,092-entry compressed vocabulary (case/accent/whitespace folded).
