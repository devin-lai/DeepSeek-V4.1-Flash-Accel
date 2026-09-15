"""Stage immutable UVA expert weights for large, eager Marlin batches."""

from __future__ import annotations

from functools import wraps

import torch


def staged_marlin_gemm(
    original, min_tokens: int, on_first_stage=None, on_first_fallback=None
):
    """Copy one packed matrix at a time; leave decode and graph capture alone.

    The temporary allocation belongs to the current CUDA stream. PyTorch can
    reuse it after the GEMM, including for the down projection or next layer.
    Both the packed weights and the Marlin arithmetic are unchanged.
    """
    if min_tokens < 1:
        raise ValueError("min_tokens must be positive")
    reported = False
    fallback_reported = False

    @wraps(original)
    def gemm(*args, **kwargs):
        nonlocal reported, fallback_reported
        weight = args[2] if len(args) > 2 else kwargs["b_qweight"]
        routing = args[12] if len(args) > 12 else kwargs["topk_weights"]
        if (
            getattr(weight, "_vllm_is_uva_offloaded", False)
            and routing.shape[0] >= min_tokens
            and not torch.cuda.is_current_stream_capturing()
        ):
            try:
                staged = weight.clone(memory_format=torch.contiguous_format)
            except torch.OutOfMemoryError:
                # Allocation failed before GEMM. Its original UVA arguments
                # remain valid, including work already queued on other streams.
                if not fallback_reported and on_first_fallback is not None:
                    on_first_fallback(weight, routing.shape[0])
                    fallback_reported = True
                return original(*args, **kwargs)
            if not reported and on_first_stage is not None:
                on_first_stage(weight, routing.shape[0])
                reported = True
            if len(args) > 2:
                args = (*args[:2], staged, *args[3:])
            else:
                kwargs["b_qweight"] = staged
        return original(*args, **kwargs)

    return gemm


def install_marlin_staging(min_tokens: int) -> None:
    from vllm import _custom_ops as ops
    from vllm.logger import init_logger

    logger = init_logger("vllm.dsv41_opt")

    def report(weight, tokens):
        logger.info(
            "Marlin staging active: %s packed bytes, %d query tokens",
            weight.numel() * weight.element_size(),
            tokens,
        )

    def report_fallback(weight, tokens):
        logger.warning(
            "Marlin staging allocation did not fit (%d bytes, %d query tokens); "
            "using original UVA GEMM. First allocation fallback in this worker.",
            weight.numel() * weight.element_size(),
            tokens,
        )

    ops.moe_wna16_marlin_gemm = staged_marlin_gemm(
        ops.moe_wna16_marlin_gemm, min_tokens, report, report_fallback
    )
    logger.info(
        "Exact Marlin weight staging enabled for eager batches >= %d tokens",
        min_tokens,
    )
