#!/usr/bin/env python3
"""Local vLLM fixes for DeepSeek-V4.1-Flash on sm_120.

Each edit is small, reversible, and paired with an entry in
`faults/inventory.toml`.  They are kept here rather than as a fork so the stack
stays a stock wheel plus a readable diff.

    python upstream/vllm/apply_patch.py <vllm-package-root> [--revert|--check]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

FI_SPARSE = "models/deepseek_v4_1/nvidia/flashinfer_sparse.py"
SPARSE_MLA = "models/deepseek_v4_1/sparse_mla.py"
INDEXER = "v1/attention/backends/mla/indexer.py"
WORKER_UTILS = "v1/worker/utils.py"
V41_ATTENTION = "models/deepseek_v4_1/attention.py"
SPARSE_SWA = "v1/attention/backends/mla/sparse_swa.py"

EDITS: dict[str, list[tuple[str, str]]] = {
    # VL-009: the DSv4.1 FlashInfer backend pins the kernel block size to 128,
    # but the DSA indexer's decode path goes through DeepGEMM's paged MQA
    # logits, which asserts `block_kv == 32 or block_kv == 64` on
    # `kv_cache_spec.num_states == block_size // tokens_per_state`.  At 128 the
    # two constraints have no common value and CUDA-graph capture aborts.
    # sm_90 already runs this model at 64 (see sparse_mla.py), and FlashInfer's
    # sm_120 sparse-MLA decode takes the page block size at runtime once
    # upstream/flashinfer/apply_patch.py is applied -- so 64 satisfies both.
    FI_SPARSE: [
        (
            """    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [128]
""",
            """    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # 64 first: DeepGEMM's paged MQA logits (the DSA indexer decode path)
        # asserts block_kv in {32, 64}, so the indexer cannot be built at 128.
        # sm_90 already serves this model at 64.
        return [64, 128]
""",
        ),
    ],
    SPARSE_MLA: [
        (
            """    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64 if current_platform.is_device_capability_family(90) else 128]
""",
            """    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # 128 is unreachable wherever the DSA indexer runs on DeepGEMM, which
        # asserts block_kv in {32, 64}; offer 64 everywhere and let the
        # negotiation pick it.
        return [64, 128]
""",
        ),
    ],
    # Same edit, same reason, at the other end of the constraint: the DSv4.1
    # indexer and the DSv4.1 MLA backend share a KV cache group, so they must
    # agree on one kernel block size.  The indexer offered only 128 off sm_90,
    # and 128 is exactly the value DeepGEMM rejects.
    INDEXER: [
        (
            """        return [64 if current_platform.is_device_capability_family(90) else 128]


@dataclass
class DeepseekV32IndexerPrefillChunkMetadata:""",
            """        # Offer 64 everywhere, not just on sm_90.  `num_states` --
        # block_size // tokens_per_state -- is what reaches DeepGEMM's paged
        # MQA logits, which asserts it is 32 or 64; at block_size 128 with
        # tokens_per_state 1 it is 128 and CUDA-graph capture aborts.
        return [64, 128]


@dataclass
class DeepseekV32IndexerPrefillChunkMetadata:""",
        ),
    ],
    # VL-010: DeepSeek-V4.1 mixes two indexer compression ratios in one engine,
    # so one manager block size yields two different states-per-block and
    # DeepGEMM wants 64 from both.  The kernel block size is already negotiated
    # *per cache group* -- it is simply never allowed to differ from the manager
    # block size here, because the negotiation cannot see the spec that turns a
    # block size into a states count.  Give it the spec, and the ratio-2 group
    # runs at kernel block 128 while the ratio-1 group runs at 64; both land on
    # the 64 states DeepGEMM asserts.
    WORKER_UTILS: [
        (
            """from collections.abc import Iterable, Mapping, Sequence
""",
            """from collections.abc import Callable, Iterable, Mapping, Sequence
""",
        ),
        (
            """def select_common_block_size(
    kv_manager_block_size: int,
    backends: list[type[AttentionBackend]],
) -> int:
""",
            """# Backends whose decode path goes through DeepGEMM's paged MQA logits.  The
# value that kernel asserts on is the cache's *states* per block, which is
# `kernel_block_size // tokens_per_state` -- not the block size itself.
DSA_INDEXER_BACKEND_NAMES = (
    "DEEPSEEK_V32_INDEXER",
    "DEEPSEEK_V4_INDEXER",
    "DEEPSEEK_V41_INDEXER",
)


def deepgemm_indexer_states(is_fp4: bool = False) -> tuple[int, ...]:
    \"\"\"States per block that DeepGEMM's paged MQA logits accepts, preferred
    first.

    Two predicates in `csrc/apis/attention.hpp` have to hold at once: the
    metadata builder (`:262`) accepts 32 or 64, and the kernel itself (`:320`)
    narrows that per architecture -- on sm_120 an FP8 indexer cache must be
    exactly 64, while the FP4 branch there accepts 32 as well.
    \"\"\"
    if current_platform.is_device_capability_family(120) and not is_fp4:
        return (64,)
    return (64, 32)


def dsa_indexer_state_filter(
    kv_cache_spec: KVCacheSpec,
    backends: list[type[AttentionBackend]],
) -> Callable[[int], bool] | None:
    \"\"\"Accept only kernel block sizes whose states count DeepGEMM supports.

    Returns None for groups that never reach that kernel, leaving the ordinary
    negotiation untouched.
    \"\"\"
    if not any(b.get_name() in DSA_INDEXER_BACKEND_NAMES for b in backends):
        return None
    if not isinstance(kv_cache_spec, AttentionSpec):
        return None
    if kv_cache_spec.tokens_per_state <= 0:
        return None
    is_fp4 = getattr(kv_cache_spec, "cache_dtype_str", None) == "mxfp4"
    legal = deepgemm_indexer_states(is_fp4)
    return lambda block_size: kv_cache_spec.get_num_kernel_states(block_size) in legal


def select_common_block_size(
    kv_manager_block_size: int,
    backends: list[type[AttentionBackend]],
    accept: Callable[[int], bool] | None = None,
) -> int:
""",
        ),
        (
            """    # Case 1: if the block_size of kv cache manager is supported by all backends,
    # return it directly.
    if block_size_is_supported(backends, kv_manager_block_size):
        return kv_manager_block_size
""",
            """    # Case 1: if the block_size of kv cache manager is supported by all backends,
    # return it directly.
    if block_size_is_supported(backends, kv_manager_block_size) and (
        accept is None or accept(kv_manager_block_size)
    ):
        return kv_manager_block_size
""",
        ),
        (
            """    for supported_size in sorted(all_int_supported_sizes, reverse=True):
        if kv_manager_block_size % supported_size != 0:
            continue
        if block_size_is_supported(backends, supported_size):
            return supported_size
    raise ValueError(f"No common block size for {kv_manager_block_size}. ")
""",
            """    for supported_size in sorted(all_int_supported_sizes, reverse=True):
        if kv_manager_block_size % supported_size != 0:
            continue
        if not block_size_is_supported(backends, supported_size):
            continue
        if accept is not None and not accept(supported_size):
            continue
        return supported_size
    detail = "; ".join(
        f"{b.get_name()} supports {b.get_supported_kernel_block_sizes()}"
        for b in backends
    )
    raise ValueError(
        f"No common block size for {kv_manager_block_size}. "
        f"Candidates tried: {sorted(all_int_supported_sizes, reverse=True)}"
        f"{' (further constrained by the cache spec)' if accept else ''}. "
        f"{detail}"
    )
""",
        ),
        (
            """            kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
            group_backends = [g.backend for g in attn_groups[kv_cache_gid]]
            selected_kernel_size = select_common_block_size(
                kv_manager_block_size, group_backends
            )
            kernel_block_sizes.append(selected_kernel_size)
""",
            """            kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
            group_backends = [g.backend for g in attn_groups[kv_cache_gid]]
            selected_kernel_size = select_common_block_size(
                kv_manager_block_size,
                group_backends,
                accept=dsa_indexer_state_filter(kv_cache_spec, group_backends),
            )
            if selected_kernel_size != kv_manager_block_size:
                logger.info(
                    "KV cache group %d (tokens_per_state=%s) runs at kernel "
                    "block size %d inside manager blocks of %d: %d states per "
                    "kernel block.",
                    kv_cache_gid,
                    getattr(kv_cache_spec, "tokens_per_state", "?"),
                    selected_kernel_size,
                    kv_manager_block_size,
                    kv_cache_spec.get_num_kernel_states(selected_kernel_size),
                )
            kernel_block_sizes.append(selected_kernel_size)
""",
        ),
    ],
    # VL-011: the real fix for the two-compression-ratios problem.  What the
    # DSA indexer kernel constrains is the number of *states* in a block, not
    # the number of tokens -- `states = block_size // compress_ratio`, and
    # DeepGEMM wants 64.  V4.1 runs compress_ratio 2 (CED encoder) and 1
    # (decoder) in one engine, so one token block size cannot give both 64.
    #
    # Scale each layer's block size by its own ratio instead.  Every layer then
    # stores the same number of states per block, which is the relationship
    # upstream already documents for V4 ("C4 indexer pages hold 64 rows" at
    # block size 256 = 64 x 4).  Layers whose block sizes differ land in
    # different KV cache groups automatically -- `UniformTypeKVCacheSpecs.
    # is_uniform_type` returns False on differing block sizes -- and vLLM
    # already runs this model's groups at four different block sizes (the SWA
    # cache asks for 32, the compressor for 8), so nothing new is required of
    # the allocator.  `--block-size` becomes "indexer states per block"; pass
    # 64 on Blackwell.
    #
    # Bytes per token are unchanged: a ratio-1 layer stores twice the states of
    # a ratio-2 layer per token either way.
    V41_ATTENTION: [
        (
            """        uses_fp8_ds_mla_layout = self.kv_cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
""",
            """        uses_fp8_ds_mla_layout = self.kv_cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=states_per_block_to_tokens(
                vllm_config.cache_config.block_size, self.compress_ratio
            ),
""",
        ),
        (
            """        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
""",
            """        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=states_per_block_to_tokens(
                self.cache_config.block_size, self.compress_ratio
            ),
""",
        ),
        (
            """class DeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
""",
            """def states_per_block_to_tokens(block_size: int, compress_ratio: int) -> int:
    \"\"\"Token width of a block that holds `block_size` indexer states.

    DeepSeek-V4.1 mixes compression ratios across its layers -- 2 in the CED
    encoder half, 1 in the decoder half -- and the DSA indexer kernel
    constrains states per block, not tokens per block.  Giving every layer the
    same token width therefore gives them different states counts and at most
    one of them can be legal.  Scaling by the layer's own ratio keeps the
    states count uniform and puts the two halves in separate KV cache groups.

    compress_ratio 0 marks a pure sliding-window layer, which owns no
    compressed cache; it is passed through unchanged.
    \"\"\"
    return block_size * compress_ratio if compress_ratio > 1 else block_size


class DeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase):
    # Its block size comes from states_per_block_to_tokens above (VL-009).
    def __init__(
""",
        ),

        (
            """            backend_cls=self.swa_backend_cls,
            block_size=32,
        )
""",
            """            backend_cls=self.swa_backend_cls,
            # 64, not 32: every sm_120 sparse-MLA kernel is written for a
            # 64-token page.  The prefill dispatch bakes PAGE_BLOCK_SIZE=64
            # into each launch and rejects the call otherwise, and the decode
            # dispatch did the same before the FlashInfer patch made it a
            # runtime argument.  The SWA backend declares MultipleOf(32), so
            # this is a legal request, and a 128-token window now spans two
            # pages instead of four.
            block_size=64,
        )
""",
        ),
        (
            """        self.max_image_tokens = (
            getattr(config, "vision_max_n_token", 0)
            if getattr(config, "vision_n_layers", 0) > 0
            else 0
        )
""",
            """        from vllm.v1.attention.backends.mla.sparse_swa import images_can_arrive

        self.max_image_tokens = (
            getattr(config, "vision_max_n_token", 0)
            if getattr(config, "vision_n_layers", 0) > 0
            and images_can_arrive(vllm_config)
            else 0
        )
""",
        ),
    ],
    # VL-012 / FI-003: the sm_120 sparse-MLA *prefill* kernel is a second,
    # separate dispatch grid from the decode one, and V4.1 misses it on both
    # axes at once:
    #
    #   Unsupported sparse-MLA prefill configuration: model=DSV4 num_heads=8
    #   topk=1152 page_block_size=32 topk_extra=0 extra_page_block_size=0
    #
    # `topk` is the width of the SWA index rows, `window_size + max_image_tokens`
    # = 128 + 1024 for the vision variant, and the kernel only instantiates
    # {128, 192, 256, 512, 1024, 2048}.  Widening the buffer to the next
    # instantiated value costs nothing at run time: the kernel derives its tile
    # count from the *runtime* `topk_length` (`actual_ni = ceil(topk_len / BI)`,
    # BI = 64) and iterates only that many, so the padding is allocated and
    # never read.  1152 is a multiple of BI, so no tile straddles the boundary.
    #
    # `page_block_size` is the SWA cache's own block size, which vLLM hard-codes
    # to 32 while every sm_120 sparse-MLA kernel is written for 64 -- the prefill
    # dispatch bakes 64 into every launch and the unpatched decode dispatch
    # rejected anything else outright.  The SWA backend already declares
    # MultipleOf(32), so 64 is a legal request; it also makes the window's
    # 128 tokens span two pages instead of four.
    SPARSE_SWA: [
        (
            """def _layer_type_for(compress_ratio: int) -> str:
    if compress_ratio <= 1:
""",
            """def images_can_arrive(vllm_config) -> bool:
    \"\"\"Whether this engine can actually receive image tokens.

    `vision_n_layers > 0` says the *checkpoint* has a vision tower; it does not
    say the engine will ever be handed an image.  `--language-model-only` and
    `--limit-mm-per-prompt '{"image": 0}'` both put vLLM in text-only mode
    while leaving the architecture flag alone, so keying image-visibility
    buffers off the architecture widens them for inputs that cannot occur.
    \"\"\"
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None or not getattr(model_config, "is_multimodal_model", False):
        return False
    mm_config = getattr(model_config, "multimodal_config", None)
    if mm_config is None:
        return False
    try:
        return mm_config.get_limit_per_prompt("image") > 0
    except Exception:  # noqa: BLE001 -- unknown modality, assume enabled
        return True


# TOPK values the sm_120 sparse-MLA prefill kernel instantiates
# (`sparse_mla_sm120_prefill.cu`, `dispatch_dsv4_single`).
_SM120_PREFILL_TOPKS = (128, 192, 256, 512, 1024, 2048)


def _sm120_prefill_index_width(width: int) -> int:
    \"\"\"Round a prefill index width up to one the sm_120 kernel dispatches.

    The kernel takes the per-token length at run time and iterates
    `ceil(topk_length / BI)` tiles, so the only cost of a wider row is the
    buffer itself -- the padding is never read.  Off sm_120 the width is
    returned unchanged.
    \"\"\"
    if not current_platform.is_device_capability_family(120):
        return width
    for supported in _SM120_PREFILL_TOPKS:
        if supported >= width:
            return supported
    return width


def _layer_type_for(compress_ratio: int) -> str:
    # See _sm120_prefill_index_width above: this file now also owns the width
    # of the sm_120 prefill index rows (FI-003, VL-011).
    if compress_ratio <= 1:
""",
        ),
        (
            """        self.max_image_tokens = (
            getattr(hf_config, "vision_max_n_token", 0)
            if getattr(hf_config, "vision_n_layers", 0) > 0
            else 0
        )
        self.prefill_index_width = self.window_size + self.max_image_tokens
""",
            """        self.max_image_tokens = (
            getattr(hf_config, "vision_max_n_token", 0)
            if getattr(hf_config, "vision_n_layers", 0) > 0
            and images_can_arrive(self.vllm_config)
            else 0
        )
        self.prefill_index_width = _sm120_prefill_index_width(
            self.window_size + self.max_image_tokens
        )
""",
        ),
    ],
}


# VL-013: CUDA-graph capture corrupts every later forward on sm_120.
#
# capture_model()'s dummy forwards run with all-zero block tables and PAD slot
# mappings.  Plain KV inserts skip PAD slots, but DeepSeek-V4.1's compressed-KV,
# indexer-K and fp32 compressor-state writes derive their slots from the block
# table, so they land in block 0 -- the null block, which the scheduler never
# hands to a request and which starts out zero.  FlashInfer's sm_120 sparse-MLA
# kernels gather row 0 of block 0 for every masked (-1) index and rely on a
# zero softmax weight to cancel it; a NaN bit pattern there is 0 * NaN = NaN
# for every query whose index row is a partial tile.  Scrub block 0 after
# capture so the null block is finite again.  (upstream/flashinfer FI-004 makes
# the kernels stop reading block 0 at all; this edit keeps the runner honest
# about its own contract regardless.)
MODEL_RUNNER = "v1/worker/gpu/model_runner.py"
EDITS[MODEL_RUNNER] = [
    (
        "            end_free_gpu_memory = torch.accelerator.get_memory_info()[0]\n"
        "\n"
        "        if not profile_only:\n"
        "            # Lock workspace to prevent resizing during execution. A resize after\n"
        "            # capture frees the static cuda graph buffer.\n"
        "            lock_workspace()\n",
        "            end_free_gpu_memory = torch.accelerator.get_memory_info()[0]\n"
        "\n"
        "        # VL-013: the dummy forwards above wrote into block 0 (the null block)\n"
        "        # through block-table-derived slot mappings; sparse-attention kernels\n"
        "        # gather that row for masked indices and need it finite.\n"
        "        torch.accelerator.synchronize()\n"
        "        for kv in self.kv_caches:\n"
        "            for t in (kv if isinstance(kv, list) else [kv]):\n"
        "                if isinstance(t, torch.Tensor) and t.numel() > 0:\n"
        "                    t[0].zero_()\n"
        "\n"
        "        if not profile_only:\n"
        "            # Lock workspace to prevent resizing during execution. A resize after\n"
        "            # capture frees the static cuda graph buffer.\n"
        "            lock_workspace()\n",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="vLLM package root (…/site-packages/vllm)")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true")
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
            # `new` first: an edit that *inserts* text keeps its anchor inside
            # the replacement, so testing `old` first would re-apply it and
            # duplicate the insertion. Every edit here is written so that `new`
            # is not a substring of `old`, which makes this test decisive.
            if new in out:
                already += 1
            elif old in out:
                out = out.replace(old, new, 1)
                applied += 1
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
