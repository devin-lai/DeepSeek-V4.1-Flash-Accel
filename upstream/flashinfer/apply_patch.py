#!/usr/bin/env python3
"""Turn SM120 DSv4 sparse-MLA decode's ``TOPK`` and ``PAGE_BLOCK_SIZE`` template
parameters into runtime arguments.

Why this is safe: inside the kernel, ``TOPK`` appears only as the index-row
stride and the default/clamp for ``topk_len``, and ``PAGE_BLOCK_SIZE`` only as
the divisor that splits a global token index into (page, offset) plus the
constant scale-footer offset.  Neither sizes shared memory, bounds a
``#pragma unroll``, or reaches a tensor-core tile shape.  FlashInfer already
takes the *extra* cache's page block size as a runtime ``pbs_extra`` and says so
in a comment ("The 8-cycle runtime div is dwarfed by the cp.async.bulk that
follows"); this change applies the same reasoning to the main cache.

Effect: the instantiation grid collapses from 25 kernels (5 head counts x 5
top-k) to 5 (head counts only), and every ``(topk, page_block_size)`` becomes
dispatchable -- including ``(1152, 32)``, which DeepSeek-V4.1-Flash requests on
sm_120 and which no shipped specialisation covers.

Usage:  python apply_patch.py <flashinfer-package-root> [--revert]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

CUH = "data/include/flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh"
CU = "data/csrc/sparse_mla_sm120_decode_dsv4.cu"
PY = "mla/_sparse_mla_sm120.py"

EDITS: dict[str, list[tuple[str, str]]] = {}

# ---------------------------------------------------------------- kernel .cuh
EDITS[CUH] = [
    (
        "template <ModelType MT, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE>\n"
        "__global__ void __launch_bounds__(DSV4_BLOCK_THREADS) sparse_mla_decode_dsv4_kernel(",
        "template <ModelType MT, int NUM_HEADS>\n"
        "__global__ void __launch_bounds__(DSV4_BLOCK_THREADS) sparse_mla_decode_dsv4_kernel(",
    ),
    (
        "    size_t stride_extra_kv_block, int num_tokens, int num_splits, int chunks_per_block,\n"
        "    float sm_scale, size_t stride_kv_block) {",
        "    size_t stride_extra_kv_block, int num_tokens, int num_splits, int chunks_per_block,\n"
        "    float sm_scale, size_t stride_kv_block,\n"
        "    int topk,   // index-row stride + default/clamp for topk_len (was TOPK)\n"
        "    int pbs) {  // main-cache page block size (was PAGE_BLOCK_SIZE)",
    ),
    (
        "  constexpr int IO_STRIDE = D_NOPE + D_ROPE_C * 2;                  // 576\n"
        "  constexpr int pbs = PAGE_BLOCK_SIZE;\n",
        "  constexpr int IO_STRIDE = D_NOPE + D_ROPE_C * 2;                  // 576\n",
    ),
    (
        "  int topk_len = topk_length_ptr ? __ldg(topk_length_ptr + t_idx) : TOPK;\n"
        "  topk_len = topk_len < 0 ? 0 : (topk_len > TOPK ? TOPK : topk_len);",
        "  int topk_len = topk_length_ptr ? __ldg(topk_length_ptr + t_idx) : topk;\n"
        "  topk_len = topk_len < 0 ? 0 : (topk_len > topk ? topk : topk_len);",
    ),
    (
        "  const int32_t* idx_base = indices + (size_t)t_idx * TOPK;",
        "  const int32_t* idx_base = indices + (size_t)t_idx * topk;",
    ),
    (
        "    // Page block size of THIS section. Main is compile-time constexpr (typ.\n"
        "    // 64); extra is runtime (DSv4 C128A passes 2). The 8-cycle runtime div\n"
        "    // is dwarfed by the cp.async.bulk that follows.",
        "    // Page block size of THIS section. Both main and extra are runtime; the\n"
        "    // 8-cycle runtime div is dwarfed by the cp.async.bulk that follows.",
    ),
]

# ------------------------------------------------------------- launcher .cu
EDITS[CU] = [
    (
        "template <ModelType MT, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE>\n"
        "static bool launch_decode_dsv4_impl(",
        "template <ModelType MT, int NUM_HEADS>\n"
        "static bool launch_decode_dsv4_impl(",
    ),
    (
        "                                    int chunks_per_block_override, float sm_scale,\n"
        "                                    size_t stride_kv_block, cudaStream_t stream) {",
        "                                    int chunks_per_block_override, float sm_scale,\n"
        "                                    size_t stride_kv_block, int topk, int pbs,\n"
        "                                    cudaStream_t stream) {",
    ),
    (
        "  auto kernel = sparse_mla_decode_dsv4_kernel<MT, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE>;",
        "  auto kernel = sparse_mla_decode_dsv4_kernel<MT, NUM_HEADS>;",
    ),
    (
        "      extra_topk_length, extra_topk, pbs_extra, stride_extra_kv_block, num_tokens, "
        "num_splits,\n"
        "      chunks_per_block, sm_scale, stride_kv_block);",
        "      extra_topk_length, extra_topk, pbs_extra, stride_extra_kv_block, num_tokens, "
        "num_splits,\n"
        "      chunks_per_block, sm_scale, stride_kv_block, topk, pbs);",
    ),
]

_OLD_DISPATCH_HEAD = """  if (mt != ModelType::DSV4 || page_block_size != 64) return false;
  if (num_splits <= 0) return false;
#define DSV4_DISPATCH(H, K)                                                                 \\
  if (num_heads == (H) && topk == (K)) {                                                    \\
    return launch_decode_dsv4_impl<ModelType::DSV4, (H), (K), 64>(                          \\
        Q, KV_cache, indices, mid_out, mid_lse, topk_length, output, out_lse, attn_sink,    \\
        extra_KV_cache, extra_indices, extra_topk_length, extra_topk, pbs_extra,            \\
        stride_extra_kv_block, num_tokens, num_splits, chunks_per_block_override, sm_scale, \\
        stride_kv_block, stream);                                                           \\
  }
"""

_NEW_DISPATCH_HEAD = """  if (mt != ModelType::DSV4) return false;
  if (num_splits <= 0 || topk <= 0 || page_block_size <= 0) return false;
#define DSV4_DISPATCH(H)                                                                    \\
  if (num_heads == (H)) {                                                                   \\
    return launch_decode_dsv4_impl<ModelType::DSV4, (H)>(                                   \\
        Q, KV_cache, indices, mid_out, mid_lse, topk_length, output, out_lse, attn_sink,    \\
        extra_KV_cache, extra_indices, extra_topk_length, extra_topk, pbs_extra,            \\
        stride_extra_kv_block, num_tokens, num_splits, chunks_per_block_override, sm_scale, \\
        stride_kv_block, topk, page_block_size, stream);                                    \\
  }
"""

_OLD_DISPATCH_BODY = "\n".join(
    f"  DSV4_DISPATCH({h}, {k})"
    for h in (8, 16, 32, 64, 128)
    for k in (128, 192, 256, 512, 1024)
) + "\n"

_NEW_DISPATCH_BODY = "\n".join(f"  DSV4_DISPATCH({h})" for h in (8, 16, 32, 64, 128)) + "\n"

EDITS[CU] += [
    (_OLD_DISPATCH_HEAD, _NEW_DISPATCH_HEAD),
    (_OLD_DISPATCH_BODY, _NEW_DISPATCH_BODY),
    (
        "// Public surface — explicit instantiation switch over the PR-body bench grid.\n"
        "// DSV4 only, page_block_size=64 only. NUM_HEADS ∈ {8, 16, 32, 64, 128},\n"
        "// TOPK ∈ {128, 192, 256, 512, 1024}. TOPK=192 covers the padded\n"
        "// DeepSeek-V4-Flash-0731 DSpark K=5 shape (128 SWA + 5 active draft entries),\n"
        "// while TOPK=256 covers wider DSpark configurations.",
        "// Public surface — explicit instantiation switch on NUM_HEADS only.\n"
        "// DSV4 only. NUM_HEADS ∈ {8, 16, 32, 64, 128}; topk and page_block_size are\n"
        "// runtime, so every (topk, page_block_size) dispatches. This covers the\n"
        "// shapes DeepSeek-V4.1-Flash asks for — notably topk=1152 (2*index_topk +\n"
        "// the 128-token SWA window) at page_block_size=32 — which no compile-time\n"
        "// grid enumerated.",
    ),
]

# ------------------------------------------------------------------ python
# --------------------------------------------------- prefill launcher .cu
# FI-003, second half: the dual-cache prefill dispatch (the path every
# compressed layer takes) was instantiated at TOPK == 128 only -- the DSv4
# sliding window with no image tokens.  DeepSeek-V4.1's vision variant widens
# the SWA index row to sliding_window + vision_max_n_token = 1152, which vLLM
# pads to the next instantiated width, 2048.  Inside prefill_kernel.cuh TOPK is
# the index-row stride and the clamp for the runtime topk_length, not a work
# bound, so instantiating the dual kernel at 2048 costs compile time and
# nothing else.  FP8 compute mode is what the single-cache grid already uses
# above topk 256.
PREFILL_CU = "data/csrc/sparse_mla_sm120_prefill.cu"
EDITS[PREFILL_CU] = [
    (
        "  if (topk != 128) return false;\n"
        "  if (extra_page_block_size == 64) {\n"
        "    DISPATCH_BY_NH_PBSX(64);\n"
        "  } else if (extra_page_block_size == 2) {\n"
        "    DISPATCH_BY_NH_PBSX(2);\n"
        "  }\n"
        "  return false;\n"
        "#undef DISPATCH_BY_NH_PBSX\n"
        "#undef DISPATCH_DUAL_MG_CM\n",
        "// Wider SWA index rows (DeepSeek-V4.1 with image tokens: 128 + 1024,\n"
        "// padded by the caller to 2048).  Same kernel, same NHG split; FP8\n"
        "// compute mode as the single-cache grid uses above topk 256.\n"
        "#define DISPATCH_BY_NH_CM_TK_PBSX(CM, TK, PBSX)      \\\n"
        "  do {                                                \\\n"
        "    switch (num_heads) {                              \\\n"
        "      case 8:                                         \\\n"
        "        DISPATCH_DUAL_MG_CM(CM, 8, TK, PBSX, 1);      \\\n"
        "        return true;                                  \\\n"
        "      case 16:                                        \\\n"
        "        DISPATCH_DUAL_MG_CM(CM, 16, TK, PBSX, 1);     \\\n"
        "        return true;                                  \\\n"
        "      case 32:                                        \\\n"
        "        DISPATCH_DUAL_MG_CM(CM, 32, TK, PBSX, 2);     \\\n"
        "        return true;                                  \\\n"
        "      case 64:                                        \\\n"
        "        DISPATCH_DUAL_MG_CM(CM, 64, TK, PBSX, 2);     \\\n"
        "        return true;                                  \\\n"
        "      case 128:                                       \\\n"
        "        DISPATCH_DUAL_MG_CM(CM, 128, TK, PBSX, 2);    \\\n"
        "        return true;                                  \\\n"
        "      default:                                        \\\n"
        "        return false;                                 \\\n"
        "    }                                                 \\\n"
        "  } while (0)\n"
        "\n"
        "  if (topk == 128) {\n"
        "    if (extra_page_block_size == 64) {\n"
        "      DISPATCH_BY_NH_PBSX(64);\n"
        "    } else if (extra_page_block_size == 2) {\n"
        "      DISPATCH_BY_NH_PBSX(2);\n"
        "    }\n"
        "    return false;\n"
        "  }\n"
        "  if (topk == 2048) {\n"
        "    if (extra_page_block_size == 64) {\n"
        "      DISPATCH_BY_NH_CM_TK_PBSX(FP8, 2048, 64);\n"
        "    } else if (extra_page_block_size == 2) {\n"
        "      DISPATCH_BY_NH_CM_TK_PBSX(FP8, 2048, 2);\n"
        "    }\n"
        "    return false;\n"
        "  }\n"
        "  return false;\n"
        "#undef DISPATCH_BY_NH_CM_TK_PBSX\n"
        "#undef DISPATCH_BY_NH_PBSX\n"
        "#undef DISPATCH_DUAL_MG_CM\n",
    ),
]

# ------------------------------------------------ FI-004: masked-index gathers
# Every sparse-MLA kernel here clamps a masked index (-1, "no token") to 0 and
# gathers row 0 of block 0 for it.  The score of that entry is forced to -1e30
# so its softmax weight is exactly 0 -- but the row still enters the P.V
# product, and 0 * NaN is NaN.  Row 0 of block 0 is vLLM's null block, which is
# never handed to a request and which the capture-time dummy forwards of
# DeepSeek-V4.1 fill with compressed-KV / indexer / fp32 compressor-state bytes
# (their slot mappings derive from all-zero block tables).  Once a NaN bit
# pattern lands there, every query whose index row is a partial 64-entry tile
# comes back NaN.  Gather the tile's first entry instead: index rows are
# compacted (valid entries first), so it is a real, finite row whenever the
# tile is iterated at all.  Pure robustness; no arithmetic changes for valid
# entries.
IO_CUH = "data/include/flashinfer/attention/sparse_mla_sm120/common/kv_cache_io.cuh"
PREFILL_CUH = "data/include/flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh"
_IO_OLD = (
    "    int idx = indices[bi];\n"
    "    idx = (idx >= 0) ? idx : 0;\n"
)
_IO_NEW = (
    "    int idx = indices[bi];\n"
    "    // Masked (-1) entry: weight 0 in P.V, but the row must still be finite.\n"
    "    // Block 0 is the caller's null block and may hold anything; gather the\n"
    "    // tile's first (valid) entry instead.\n"
    "    if (idx < 0) idx = indices[0];\n"
    "    idx = (idx >= 0) ? idx : 0;\n"
)
EDITS[IO_CUH] = [(_IO_OLD, _IO_NEW), (_IO_OLD, _IO_NEW)]  # gather tile, gather scales
EDITS[PREFILL_CUH] = [
    (
        "          int idx = ib[qk_nb + e];\n"
        "          idx = (idx >= 0) ? idx : 0;\n",
        "          int idx = ib[qk_nb + e];\n"
        "          if (idx < 0) idx = ib[0];  // masked entry: any finite row (FI-004)\n"
        "          idx = (idx >= 0) ? idx : 0;\n",
    ),
    (
        "        int idx = ib[qk_nb + gid];\n"
        "        idx = (idx >= 0) ? idx : 0;\n",
        "        int idx = ib[qk_nb + gid];\n"
        "        if (idx < 0) idx = ib[0];  // masked entry: any finite row (FI-004)\n"
        "        idx = (idx >= 0) ? idx : 0;\n",
    ),
    (
        "        const int idx = ib[qk_nb + gid];\n"
        "        if constexpr (DUAL_CACHE) {\n",
        "        int idx = ib[qk_nb + gid];\n"
        "        if (idx < 0) idx = ib[0];  // masked entry: any finite row (FI-004)\n"
        "        if constexpr (DUAL_CACHE) {\n",
    ),
]
EDITS[CUH] += [
    (
        "      const int idx_raw = (cand_pos < g_end) ? section_idx_base[cand_pos] : -1;\n"
        "      const int idx = (idx_raw >= 0) ? idx_raw : 0;\n",
        "      const int idx_raw = (cand_pos < g_end) ? section_idx_base[cand_pos] : -1;\n"
        "      // Masked entry: weight 0 in P.V, but the gathered row must be finite;\n"
        "      // block 0 is the caller's null block.  Use the chunk's first entry (FI-004).\n"
        "      int idx = (idx_raw >= 0) ? idx_raw : section_idx_base[g_start];\n"
        "      idx = (idx >= 0) ? idx : 0;\n",
    ),
]

EDITS[PY] = [
    (
        """# decode-dsv4 instantiation set. NH=8 is the small-TP corner case; the kernel
# pads the head tile to HPB=16 with zero-Q rows and gates writes by NUM_HEADS.
_DECODE_DSV4_DISPATCH = frozenset(
    {
        (8, 128),
        (8, 192),
        (8, 256),
        (8, 512),
        (8, 1024),
        (16, 128),
        (16, 192),
        (16, 256),
        (16, 512),
        (16, 1024),
        (32, 128),
        (32, 192),
        (32, 256),
        (32, 512),
        (32, 1024),
        (64, 128),
        (64, 192),
        (64, 256),
        (64, 512),
        (64, 1024),
        (128, 128),
        (128, 192),
        (128, 256),
        (128, 512),
        (128, 1024),
    }
)
_DECODE_DSV4_PAGE_BLOCK_SIZE = 64
""",
        '''# decode-dsv4 instantiation set. NH=8 is the small-TP corner case; the kernel
# pads the head tile to HPB=16 with zero-Q rows and gates writes by NUM_HEADS.
# top-k and page_block_size are runtime kernel arguments, so only the head count
# is specialised.
_DECODE_DSV4_HEADS = frozenset({8, 16, 32, 64, 128})


class _AnyTopkDispatch(frozenset):
    """``(num_heads, topk)`` membership for a kernel that takes topk at runtime.

    Kept as a ``frozenset`` subclass because callers -- vLLM's
    ``has_flashinfer_sparse_mla_sm120_config`` among them -- probe this object
    with ``in`` to decide whether a model configuration is servable. Iterating
    it still yields a representative grid.
    """

    def __contains__(self, item) -> bool:  # type: ignore[override]
        try:
            num_heads, topk = item
        except (TypeError, ValueError):
            return False
        return int(num_heads) in _DECODE_DSV4_HEADS and int(topk) > 0


_DECODE_DSV4_DISPATCH = _AnyTopkDispatch(
    (h, k)
    for h in sorted(_DECODE_DSV4_HEADS)
    for k in (128, 192, 256, 512, 1024, 1152, 2048)
)
# Retained for callers that import it; the kernel now accepts any page block
# size, so it is no longer a dispatch precondition.
_DECODE_DSV4_PAGE_BLOCK_SIZE = 64
''',
    ),
    (
        """    return (
        num_tokens <= _DECODE_MAX_TOKENS
        and d_qk == 512
        and page_block_size == _DECODE_DSV4_PAGE_BLOCK_SIZE
        and (num_heads, topk) in _DECODE_DSV4_DISPATCH
    )""",
        """    return (
        num_tokens <= _DECODE_MAX_TOKENS
        and d_qk == 512
        and page_block_size > 0
        and topk > 0
        and num_heads in _DECODE_DSV4_HEADS
    )""",
    ),
    (
        """        if (
            model_type == _MODEL_TYPE_DSV4
            and kv_pbs == _DECODE_DSV4_PAGE_BLOCK_SIZE
            and _decode_dsv4_dispatchable(
                num_tokens, num_heads, topk, d_qk, kv_pbs, extra_topk
            )
        ):""",
        """        if model_type == _MODEL_TYPE_DSV4 and _decode_dsv4_dispatchable(
            num_tokens, num_heads, topk, d_qk, kv_pbs, extra_topk
        ):""",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="flashinfer package root (…/site-packages/flashinfer)")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true", help="report state, change nothing")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    rc = 0
    for rel, edits in EDITS.items():
        path = root / rel
        if not path.exists():
            print(f"MISSING  {rel}")
            rc = 1
            continue
        text = path.read_text()
        pairs = [(b, a) for a, b in edits] if args.revert else edits
        out, applied, already = text, 0, 0
        for old, new in pairs:
            if old in out:
                out = out.replace(old, new, 1)
                applied += 1
            elif new in out:
                already += 1
            else:
                print(f"FAILED   {rel}: anchor not found:\n    {old.splitlines()[0][:96]}")
                rc = 1
        state = f"{applied} applied, {already} already"
        if args.check:
            print(f"CHECK    {rel}: {state}")
            continue
        if out != text:
            backup = path.with_suffix(path.suffix + ".orig")
            if not backup.exists():
                backup.write_text(text)
            path.write_text(out)
        print(f"{'REVERT' if args.revert else 'PATCH '}   {rel}: {state}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
