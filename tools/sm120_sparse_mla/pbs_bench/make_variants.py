#!/usr/bin/env python3
"""Derive a page-size-runtime variant of FlashInfer's SM120 sparse-MLA prefill tree.

Copies ``<flashinfer>/data/include/flashinfer/attention/sparse_mla_sm120`` to
``<out>/flashinfer/attention/sparse_mla_sm120`` and rewrites the MG prefill kernel
so that the main-cache (and, optionally, extra-cache) page block size can be a
runtime value carried in ``PrefillColdParams``.  Behaviour is selected at
compile time with preprocessor macros so one source tree serves every variant:

  -DSMLA_PBS_MODE=0   compile-time PAGE_BLOCK_SIZE (the stock kernel; control)
  -DSMLA_PBS_MODE=1   runtime page size, generic integer division / modulo
  -DSMLA_PBS_MODE=2   runtime page size, power-of-two shift / mask (log2 passed)
  -DSMLA_PBS_MODE=3   runtime page size, CUTLASS-style fast divmod (mulhi+shift)
  -DSMLA_EXTRA_RUNTIME=1  also make the dual-cache *extra* page size runtime
  -DSMLA_IO_MAXNREG=32    setmaxnreg.dec value for the 4 IO warps   (stock 32)
  -DSMLA_MATH_MAXNREG=232 setmaxnreg.inc value for the 8 math warps (stock 232)

Only text that the stock 0.6.18.post1 tree contains is rewritten; every
substitution is checked and the script fails loudly if an anchor is missing.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import sys

GEOM_DEFS = r'''
// ---------------------------------------------------------------------------
// Runtime page geometry (benchmark variant; see make_variants.py).
// ---------------------------------------------------------------------------
#ifndef SMLA_PBS_MODE
#define SMLA_PBS_MODE 0
#endif
#ifndef SMLA_EXTRA_RUNTIME
#define SMLA_EXTRA_RUNTIME 0
#endif
#ifndef SMLA_IO_MAXNREG
#define SMLA_IO_MAXNREG 32
#endif
#ifndef SMLA_MATH_MAXNREG
#define SMLA_MATH_MAXNREG 232
#endif

struct PageGeom {
  int pbs;         // page block size in tokens
  int log2_pbs;    // valid when pbs is a power of two
  uint32_t magic;  // fast-divmod multiplier (CUTLASS find_divisor)
  uint32_t shift;  // fast-divmod shift
};

// Host helper: fill a PageGeom for divisor `pbs` (>= 2).
__host__ __device__ inline PageGeom make_page_geom(int pbs) {
  PageGeom g;
  g.pbs = pbs;
  int l = 0;
  while ((1 << l) < pbs) ++l;  // ceil(log2(pbs))
  g.log2_pbs = l;
  const unsigned p = 31u + (unsigned)l;
  const unsigned long long m = ((1ull << p) + (unsigned long long)pbs - 1ull) / (unsigned long long)pbs;
  g.magic = (uint32_t)m;
  g.shift = p - 32u;
  return g;
}

// (block, local) decomposition of a global slot index.  RT=false keeps the
// stock compile-time arithmetic regardless of SMLA_PBS_MODE.
template <int PBS_T, bool RT>
__device__ __forceinline__ void page_divmod(int idx, const PageGeom& g, int& bi, int& li) {
  if constexpr (!RT) {
    bi = idx / PBS_T;
    li = idx % PBS_T;
  } else {
#if SMLA_PBS_MODE == 1
    bi = idx / g.pbs;
    li = idx - bi * g.pbs;
#elif SMLA_PBS_MODE == 2
    bi = idx >> g.log2_pbs;
    li = idx & (g.pbs - 1);
#elif SMLA_PBS_MODE == 3
    bi = (int)(__umulhi((unsigned)idx, g.magic) >> g.shift);
    li = idx - bi * g.pbs;
#else
    bi = idx / PBS_T;
    li = idx % PBS_T;
#endif
  }
}

template <int PBS_T, bool RT>
__device__ __forceinline__ int page_size_of(const PageGeom& g) {
  if constexpr (RT) return g.pbs;
  return PBS_T;
}
'''


def must_replace(text: str, old: str, new: str, count: int | None = None, what: str = "") -> str:
    n = text.count(old)
    if n == 0:
        raise SystemExit(f"anchor not found ({what}):\n{old}")
    if count is not None and n != count:
        raise SystemExit(f"anchor count {n} != {count} ({what}):\n{old}")
    return text.replace(old, new)


def patch_calls(text: str, head_old: str, head_new: str, extra_arg: str) -> tuple[str, int]:
    """Replace every call `head_old(...)` by `head_new(..., extra_arg)`.

    `head_old` must end with '(' ; the matching ')' is found by paren counting.
    """
    assert head_old.endswith("(") and head_new.endswith("(")
    out = []
    i = 0
    n = 0
    while True:
        j = text.find(head_old, i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        k = j + len(head_old)
        depth = 1
        while depth:
            c = text[k]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            k += 1
        # k is just past the matching ')'
        inner = text[j + len(head_old):k - 1]
        out.append(head_new + inner + ", " + extra_arg + ")")
        i = k
        n += 1
    return "".join(out), n


def patch_tree(src: pathlib.Path, out: pathlib.Path) -> None:
    dst = out / "flashinfer" / "attention" / "sparse_mla_sm120"
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.orig"))

    # ---------------------------------------------------------------- traits
    p = dst / "model" / "kv_cache_traits.cuh"
    t = p.read_text()
    # The headers live in the global namespace; append the geometry helpers.
    t = t.rstrip("\n") + "\n" + GEOM_DEFS + "\n"
    p.write_text(t)

    # ------------------------------------------------------------ kv_cache_io
    p = dst / "common" / "kv_cache_io.cuh"
    t = p.read_text()
    t = must_replace(
        t,
        "template <ModelType MT, int PAGE_BLOCK_SIZE, bool USE_L2_HINT = false>\n"
        "__device__ __forceinline__ void io_bulk_gather_tile(uint8_t* dst, const int32_t* indices,\n"
        "                                                    const uint8_t* __restrict__ kv_ptr,\n"
        "                                                    uint64_t* mbar, int io_tid,\n"
        "                                                    size_t stride_kv_block,\n"
        "                                                    uint64_t cache_policy = 0) {",
        "template <ModelType MT, int PAGE_BLOCK_SIZE, bool USE_L2_HINT = false, bool RT = false>\n"
        "__device__ __forceinline__ void io_bulk_gather_tile(uint8_t* dst, const int32_t* indices,\n"
        "                                                    const uint8_t* __restrict__ kv_ptr,\n"
        "                                                    uint64_t* mbar, int io_tid,\n"
        "                                                    size_t stride_kv_block,\n"
        "                                                    uint64_t cache_policy = 0,\n"
        "                                                    PageGeom pg = PageGeom{}) {",
        1, "io_bulk_gather_tile signature",
    )
    t = must_replace(
        t,
        "      constexpr int pbs = PAGE_BLOCK_SIZE;\n"
        "      int block_idx = idx / pbs;\n"
        "      int local_idx = idx % pbs;\n"
        "      src = kv_ptr + (size_t)block_idx * stride_kv_block + (size_t)local_idx * IO::IO_STRIDE;",
        "      int block_idx, local_idx;\n"
        "      page_divmod<PAGE_BLOCK_SIZE, RT>(idx, pg, block_idx, local_idx);\n"
        "      src = kv_ptr + (size_t)block_idx * stride_kv_block + (size_t)local_idx * IO::IO_STRIDE;",
        1, "io_bulk_gather_tile body",
    )
    t = must_replace(
        t,
        "template <ModelType MT, int PAGE_BLOCK_SIZE>\n"
        "__device__ __forceinline__ void io_gather_scales(uint8_t* scale_dst, const int32_t* indices,\n"
        "                                                 const uint8_t* __restrict__ kv_ptr, int io_tid,\n"
        "                                                 size_t stride_kv_block) {",
        "template <ModelType MT, int PAGE_BLOCK_SIZE, bool RT = false>\n"
        "__device__ __forceinline__ void io_gather_scales(uint8_t* scale_dst, const int32_t* indices,\n"
        "                                                 const uint8_t* __restrict__ kv_ptr, int io_tid,\n"
        "                                                 size_t stride_kv_block,\n"
        "                                                 PageGeom pg = PageGeom{}) {",
        1, "io_gather_scales signature",
    )
    t = must_replace(
        t,
        "  constexpr int pbs = PAGE_BLOCK_SIZE;\n"
        "  constexpr int SCALE_BYTES = KV::SCALE_BYTES_PER_TOKEN;\n",
        "  constexpr int SCALE_BYTES = KV::SCALE_BYTES_PER_TOKEN;\n",
        1, "io_gather_scales pbs constexpr",
    )
    t = must_replace(
        t,
        "    int block_idx = idx / pbs;\n"
        "    int local_idx = idx % pbs;\n"
        "    const uint8_t* src = kv_ptr + (size_t)block_idx * stride_kv_block +\n"
        "                         (size_t)pbs * IO::IO_STRIDE + (size_t)local_idx * SCALE_BYTES;",
        "    int block_idx, local_idx;\n"
        "    page_divmod<PAGE_BLOCK_SIZE, RT>(idx, pg, block_idx, local_idx);\n"
        "    const uint8_t* src = kv_ptr + (size_t)block_idx * stride_kv_block +\n"
        "                         (size_t)page_size_of<PAGE_BLOCK_SIZE, RT>(pg) * IO::IO_STRIDE +\n"
        "                         (size_t)local_idx * SCALE_BYTES;",
        1, "io_gather_scales body",
    )
    p.write_text(t)

    # ------------------------------------------------------------ xv_rope_mma
    p = dst / "common" / "xv_rope_mma.cuh"
    t = p.read_text()
    head, sep, tail = t.partition("template <ModelType MT, int PAGE_BLOCK_SIZE, int N_HG>\n__device__ __forceinline__ void xv_rope_mma_mg(")
    if not sep:
        raise SystemExit("xv_rope_mma_mg head not found")
    tail = must_replace(
        tail,
        "                                               int lane, size_t stride_kv_block,\n"
        "                                               bf16* weight_smem) {",
        "                                               int lane, size_t stride_kv_block,\n"
        "                                               bf16* weight_smem,\n"
        "                                               PageGeom pg = PageGeom{}) {",
        1, "xv_rope_mma_mg signature",
    )
    tail = must_replace(
        tail,
        "        constexpr int pbs = PAGE_BLOCK_SIZE;\n"
        "        int bi = idx / pbs;\n"
        "        int li = idx % pbs;\n",
        "        int bi, li;\n"
        "        page_divmod<PAGE_BLOCK_SIZE, RT>(idx, pg, bi, li);\n",
        1, "xv_rope_mma_mg body",
    )
    t = head + "template <ModelType MT, int PAGE_BLOCK_SIZE, int N_HG, bool RT = false>\n__device__ __forceinline__ void xv_rope_mma_mg(" + tail
    p.write_text(t)

    # --------------------------------------------------------- prefill_kernel
    p = dst / "prefill_kernel.cuh"
    t = p.read_text()
    # Runtime flags visible to the whole file.
    t = must_replace(
        t,
        "struct PrefillColdParams {\n  float sm_scale;\n",
        "static constexpr bool RT_MAIN = (SMLA_PBS_MODE != 0);\n"
        "static constexpr bool RT_EXTRA = (SMLA_PBS_MODE != 0) && (SMLA_EXTRA_RUNTIME != 0);\n\n"
        "struct PrefillColdParams {\n  float sm_scale;\n",
        1, "cold params head",
    )
    t = must_replace(
        t,
        "  const int*\n"
        "      topk_length_extra;  // [num_tokens] int32, dual-cache only. nullptr = uniform topk_extra.\n"
        "};",
        "  const int*\n"
        "      topk_length_extra;  // [num_tokens] int32, dual-cache only. nullptr = uniform topk_extra.\n"
        "  PageGeom geom;        // main-cache page geometry (runtime variants)\n"
        "  PageGeom geom_extra;  // extra-cache page geometry (runtime variants)\n"
        "};",
        1, "cold params tail",
    )
    # Register split knobs.
    t = must_replace(
        t,
        'asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\\n" ::"n"(32));',
        'asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\\n" ::"n"(SMLA_IO_MAXNREG));',
        None, "setmaxnreg.dec",
    )
    t = must_replace(
        t,
        'asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\\n" ::"n"(232));',
        'asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\\n" ::"n"(SMLA_MATH_MAXNREG));',
        None, "setmaxnreg.inc",
    )
    # prefill_kv_entry_base definition.
    t = must_replace(
        t,
        "template <ModelType MT, int PAGE_BLOCK_SIZE>\n"
        "__device__ __forceinline__ const uint8_t* prefill_kv_entry_base(\n"
        "    const uint8_t* __restrict__ kv_global, int idx, size_t stride_kv_block) {",
        "template <ModelType MT, int PAGE_BLOCK_SIZE, bool RT = false>\n"
        "__device__ __forceinline__ const uint8_t* prefill_kv_entry_base(\n"
        "    const uint8_t* __restrict__ kv_global, int idx, size_t stride_kv_block,\n"
        "    PageGeom pg = PageGeom{}) {",
        1, "prefill_kv_entry_base signature",
    )
    t = must_replace(
        t,
        "    const int bi = idx / PAGE_BLOCK_SIZE;\n"
        "    const int li = idx % PAGE_BLOCK_SIZE;\n"
        "    return kv_global + (size_t)bi * stride_kv_block + (size_t)li * IO::IO_STRIDE;",
        "    int bi, li;\n"
        "    page_divmod<PAGE_BLOCK_SIZE, RT>(idx, pg, bi, li);\n"
        "    return kv_global + (size_t)bi * stride_kv_block + (size_t)li * IO::IO_STRIDE;",
        1, "prefill_kv_entry_base body",
    )
    # Call sites: split the file at the MG implementation so the SG kernel
    # (which we do not benchmark) keeps stock arithmetic.
    marker = "// Shared MG implementation for single-cache and dual-cache prefill."
    sg_part, sep, mg_part = t.partition(marker)
    if not sep:
        raise SystemExit("MG marker not found")
    total = 0
    for head_old, head_new, arg in [
        ("io_gather_scales<MT, PAGE_BLOCK_SIZE>(", "io_gather_scales<MT, PAGE_BLOCK_SIZE, RT_MAIN>(", "cold.geom"),
        ("io_gather_scales<MT, PAGE_BLOCK_SIZE_EXTRA>(", "io_gather_scales<MT, PAGE_BLOCK_SIZE_EXTRA, RT_EXTRA>(", "cold.geom_extra"),
        ("io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE, true>(", "io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE, true, RT_MAIN>(", "cold.geom"),
        ("io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE_EXTRA, true>(", "io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE_EXTRA, true, RT_EXTRA>(", "cold.geom_extra"),
        ("prefill_kv_entry_base<MT, PAGE_BLOCK_SIZE>(", "prefill_kv_entry_base<MT, PAGE_BLOCK_SIZE, RT_MAIN>(", "cold.geom"),
        ("prefill_kv_entry_base<MT, PAGE_BLOCK_SIZE_EXTRA>(", "prefill_kv_entry_base<MT, PAGE_BLOCK_SIZE_EXTRA, RT_EXTRA>(", "cold.geom_extra"),
        ("xv_rope_mma_mg<MT, PAGE_BLOCK_SIZE, MG_N_HG>(", "xv_rope_mma_mg<MT, PAGE_BLOCK_SIZE, MG_N_HG, RT_MAIN>(", "cold.geom"),
        ("xv_rope_mma_mg<MT, PAGE_BLOCK_SIZE_EXTRA, MG_N_HG>(", "xv_rope_mma_mg<MT, PAGE_BLOCK_SIZE_EXTRA, MG_N_HG, RT_EXTRA>(", "cold.geom_extra"),
    ]:
        mg_part, n = patch_calls(mg_part, head_old, head_new, arg)
        if n == 0:
            raise SystemExit(f"no call sites for {head_old}")
        print(f"  {n:2d} x {head_old}")
        total += n
    # Sanity: every page-size-dependent helper call in the MG part is rewritten.
    leftovers = re.findall(r"(io_gather_scales|io_bulk_gather_tile|prefill_kv_entry_base|xv_rope_mma_mg)<MT, PAGE_BLOCK_SIZE(_EXTRA)?(, true)?(, MG_N_HG)?>\(", mg_part)
    if leftovers:
        raise SystemExit(f"unpatched call sites remain: {leftovers}")
    t = sg_part + marker + mg_part
    p.write_text(t)
    print(f"patched {total} MG call sites -> {dst}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=pathlib.Path, help=".../flashinfer/attention/sparse_mla_sm120 (installed tree)")
    ap.add_argument("out", type=pathlib.Path, help="output include root")
    a = ap.parse_args()
    patch_tree(a.src, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
