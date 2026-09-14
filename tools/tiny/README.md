# A DeepSeek-V4.1 that boots in a minute

Every V4.1 bug in [`faults/`](../../faults/inventory.toml) took five to seven
minutes per attempt to reproduce: 476 GiB off NVMe, 264 GiB pinned for Engram,
a Marlin repack of the expert bank, then memory profiling. Most of those bugs
are in *shape and configuration* logic — KV cache grouping, block-size
negotiation, index widths, dispatch tables — which does not care how big the
weights are.

`make_tiny_model.py` writes a config that keeps the structure and throws away
the bytes:

```bash
python tools/tiny/make_tiny_model.py \
    --src /data/models/DeepSeek-V4.1-Flash \
    --dst /data/models/dsv41-tiny --layers 8

vllm serve /data/models/dsv41-tiny --load-format dummy \
    --tokenizer-mode deepseek_v41 --trust-remote-code \
    --block-size 64 --max-model-len 2048 --enforce-eager
```

## What it keeps

The parts that decide which code path runs:

- **the CED split with two compression ratios** — `compress_ratios` becomes
  `[0, 0, 2…2, 1…1, 0, 0, 0]`, with `kv_source_layer_ids` and
  `index_source_layer_ids` placed one per half. This is what makes a layer's
  KV cache group, its states-per-block, and its DeepGEMM constraint (VL-009).
- **the sparse-MLA head geometry** — `head_dim` 512, `index_head_dim` 128,
  `index_n_heads` 32, `sliding_window` 128. These reach the kernel dispatch
  tables (FI-001, FI-003) unchanged.
- Engram, the MoE gate, the vision tower: present, tiny.

## What it shrinks

`hidden_size`, `moe_intermediate_size`, `n_routed_experts`, `vocab_size`,
`engram_num_embeddings`, the vision tower, and the layer count. With
`--load-format dummy` there is no checkpoint to read at all.

## What it cannot tell you

The weights are random, so **only structural outcomes are meaningful**: does
the engine come up, which KV cache groups form, which kernel gets dispatched,
does a call fail an assert or an index bound.

**Numerical health is not one of them.** Dummy weights overflow this
architecture on their own — a tiny model in eager mode, where the real
checkpoint is correct, still returns non-finite logits and the same token id
forever. Anything about output quality, finiteness, throughput or accuracy has
to be measured on the real checkpoint.

## Two things it will trip over

Both are relationships in the config that a naive shrink breaks, and both are
worth knowing if you adapt this for another model:

- **`engram_vocab_size` and `engram_num_embeddings` must shrink together.** The
  table is partitioned into `(max_ngram_size - 1) * n_heads` prime-sized
  buckets, each drawn just above `engram_vocab_size`, and
  `ParallelEngramEmbedding` asserts their sum fits. That is why the real
  config's 384M rows pair with a 16M hash vocabulary.
- **`vocab_size` cannot shrink at all** while the real tokenizer is used: the
  first prompt indexes past the embedding table and the gather trips a device
  assert. At `hidden_size` 1024 the full table costs 265 MB, which is not what
  makes the real model large.
