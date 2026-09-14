#!/usr/bin/env python3
"""Build a tiny DeepSeek-V4.1 config that keeps the *structure* that matters --
the CED encoder/decoder split with two indexer compression ratios, the sparse
MLA head geometry, Engram, MoE -- and shrinks everything that only costs bytes.

Boots in seconds with --load-format dummy, so the KV-cache-group and block-size
negotiation can be exercised without the 476 GiB checkpoint.
"""
import argparse, json, os, shutil, sys

def build(src, dst, layers=8):
    cfg = json.load(open(os.path.join(src, "config.json")))
    t = cfg["text_config"]

    # Keep the geometry the attention/indexer kernels see.
    #   head_dim 512, index_head_dim 128, index_n_heads 32, sliding_window 128
    # Shrink the rest.
    t["num_hidden_layers"] = layers
    t["hidden_size"] = 1024
    t["num_attention_heads"] = 8
    t["q_lora_rank"] = 256
    t["o_lora_rank"] = 256
    t["moe_intermediate_size"] = 256
    t["n_routed_experts"] = 8
    t["num_experts_per_tok"] = 2
    # vocab_size stays at the real value: the tokenizer is the real one, so a
    # smaller embedding table indexes out of bounds on the first prompt. At
    # hidden_size 1024 it costs 265 MB, which is not what makes this big.
    t["max_position_embeddings"] = 8192
    t["index_topk"] = 64

    # CED split: layers 0-1 sliding-window (cr 0), then a ratio-2 encoder half
    # and a ratio-1 decoder half -- the mix that has no common block size.
    half = (layers - 2) // 2
    ratios = [0, 0] + [2] * half + [1] * (layers - 2 - half)
    # MTP/dspark layers appended as in the real config (cr 0)
    t["compress_ratios"] = ratios + [0, 0, 0]
    kv_src = [2, 2 + half]                    # one per half
    t["kv_source_layer_ids"] = kv_src
    t["index_source_layer_ids"] = sorted({2, 2 + half, layers - 1})
    t["candidate_source_layer_id"] = 2 + half

    # Engram: one small table instead of two 384M-row ones.
    #
    # The table is partitioned into `(max_ngram_size - 1) * n_heads` = 24
    # prime-sized buckets, each drawn just above `engram_vocab_size`, and
    # `ParallelEngramEmbedding` asserts their sum fits in `num_embeddings`.
    # So the two have to shrink together: 24 x 65 536 is about 1.57M rows, and
    # 2^21 leaves room. (That relationship is why the real config's 384M rows
    # pair with a 16M hash vocabulary.) The head geometry is left alone so the
    # gather kernel sees the shapes it sees in production.
    t["engram_layer_ids"] = [1]
    t["engram_vocab_size"] = 65536
    t["engram_num_embeddings"] = [1 << 21]

    t["num_nextn_predict_layers"] = 0
    t["dspark_target_layer_ids"] = []
    t["dspark_n_routed_experts"] = 4
    cfg["text_config"] = t

    v = cfg["vision_config"]
    v["num_hidden_layers"] = 2
    v["hidden_size"] = 128
    v["num_attention_heads"] = 2
    v["intermediate_size"] = 256
    cfg["vision_config"] = v

    os.makedirs(dst, exist_ok=True)
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=1)
    for name in os.listdir(src):
        if name == "config.json" or name.endswith((".safetensors", ".index.json", ".bin")):
            continue
        s = os.path.join(src, name)
        if os.path.isfile(s) and os.path.getsize(s) < 64 * 1024 * 1024:
            shutil.copy2(s, os.path.join(dst, name))
    print(f"wrote {dst}")
    print("  compress_ratios:", t["compress_ratios"])
    print("  kv_source_layer_ids:", t["kv_source_layer_ids"])
    print("  index_source_layer_ids:", t["index_source_layer_ids"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data/models/DeepSeek-V4.1-Flash")
    ap.add_argument("--dst", default="/data/models/dsv41-tiny")
    ap.add_argument("--layers", type=int, default=8)
    a = ap.parse_args()
    build(a.src, a.dst, a.layers)
