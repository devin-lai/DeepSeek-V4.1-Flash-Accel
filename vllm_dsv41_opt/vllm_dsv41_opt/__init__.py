"""Runtime weight-offload optimizations for DeepSeek-V4.1-Flash.

Registered through vllm.general_plugins in every vLLM process:

* DSV41_SINGLE_COPY=1 (default): allocate pinned host weight buffers directly.
* DSV41_OFFLOAD_LAYERS=20-39: restrict expert offload to selected layers.
* DSV41_EXACT_PINNED=1 (opt-in): replace power-of-two host allocation rounding
  with page-rounded CUDA registration for persistent weights, including Engram.
* DSV41_MARLIN_STAGE_MIN_TOKENS=256 (opt-in): stage immutable packed expert
  matrices for large eager batches while keeping small decode batches on UVA.
* DSV41_ROUTING_PROFILE=/data/routing.json (diagnostic): count selected experts;
  snapshot/reset through the optional worker extension RPC, as in the README.

No installed vLLM source files are changed. The hooks require the recorded
vLLM version; see the package README for supported paths and GPU tests.
"""

from __future__ import annotations

import os
import re

__all__ = ["register", "parse_layer_set", "layer_index_of"]

_LAYER_RE = re.compile(r"\.layers\.(\d+)(?:\.|$)")


def parse_layer_set(spec: str) -> set[int]:
    """``"20-39"`` or ``"1,4,20-25"`` -> a set of layer indices."""
    out: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def layer_index_of(module) -> int | None:
    """Absolute layer index of a decoder layer, read from any submodule prefix."""
    for m in module.modules():
        for attr in ("prefix", "layer_name", "_prefix"):
            val = getattr(m, attr, None)
            if isinstance(val, str):
                hit = _LAYER_RE.search(val)
                if hit:
                    return int(hit.group(1))
    return None


def register() -> None:
    import torch

    from vllm.logger import init_logger
    from vllm.model_executor.offloader import uva
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    # Log under the "vllm." namespace: vLLM only attaches handlers there.
    logger = init_logger("vllm.dsv41_opt")

    if profile_path := os.environ.get("DSV41_ROUTING_PROFILE"):
        from .routing_profile import install_routing_profile

        install_routing_profile(profile_path)

    staging_tokens = int(os.environ.get("DSV41_MARLIN_STAGE_MIN_TOKENS", "0"))
    if staging_tokens:
        from .staging import install_marlin_staging

        install_marlin_staging(staging_tokens)

    spec = os.environ.get("DSV41_OFFLOAD_LAYERS", "").strip()
    allowed = parse_layer_set(spec) if spec else None
    single_copy = os.environ.get("DSV41_SINGLE_COPY", "1") not in (
        "0",
        "false",
        "False",
    )
    exact_pinned = os.environ.get("DSV41_EXACT_PINNED", "0") == "1"
    weight_torch = torch
    if exact_pinned:
        if not single_copy:
            raise ValueError("DSV41_EXACT_PINNED=1 requires DSV41_SINGLE_COPY=1")
        from .pinned import WeightTorchProxy, install_weight_allocator

        install_weight_allocator()
        weight_torch = WeightTorchProxy()
        logger.info("Exact pinned weight allocation enabled (Engram and UVA experts)")
    if allowed is None and not single_copy:
        return

    original = uva.UVAOffloader._maybe_offload_to_cpu

    def _offload_params_single_copy(self, module, prefix: str) -> bool:
        """Offload this module's eligible params using one host buffer each."""
        did = False
        for name, p in module.named_parameters():
            if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
                break
            if p.device.type == "cpu" or getattr(p, "_vllm_is_uva_offloaded", False):
                continue
            if self.cpu_offload_params and not any(
                f".{seg}." in f".{prefix}{name}." for seg in self.cpu_offload_params
            ):
                continue
            # One allocation, pinned up front; copy straight from device.
            cpu_data = weight_torch.empty(
                p.data.shape,
                dtype=p.data.dtype,
                device="cpu",
                pin_memory=self.pin_memory,
            )
            cpu_data.copy_(p.data)
            p.data = get_accelerator_view_from_cpu_tensor(cpu_data)
            p._vllm_is_uva_offloaded = True
            self.cpu_offload_bytes += cpu_data.numel() * cpu_data.element_size()
            did = True
        return did

    def _maybe_offload_to_cpu(self, module, prefix: str = ""):
        if allowed is not None:
            idx = layer_index_of(module)
            # Modules that are not decoder layers (towers, heads) keep stock behaviour.
            if idx is not None and idx not in allowed:
                return module
        if not single_copy or not self.uva_offloading:
            # Stock must install its functional_call wrapper before the budget
            # is spent. Moving weights first would leave an unwrapped CPU layer.
            return original(self, module, prefix)

        if (params := next(module.parameters(), None)) is None:
            return module
        if params.device == torch.device("cpu"):
            return module
        if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
            return module
        if prefix:
            prefix = prefix if prefix.endswith(".") else f"{prefix}."

        _offload_params_single_copy(self, module, prefix)
        return module

    uva.UVAOffloader._maybe_offload_to_cpu = _maybe_offload_to_cpu
    logger.info(
        "vllm_dsv41_opt active: single_copy_offload=%s, offload_layers=%s",
        single_copy,
        sorted(allowed) if allowed is not None else "all",
    )
