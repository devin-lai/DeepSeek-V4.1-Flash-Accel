"""vLLM general plugin for serving huge MoE checkpoints on small GPUs.

Two independent fixes to vLLM's UVA weight offloader, both needed to fit
DeepSeek-V4.1-Flash (552B backbone + 196B Engram) on 8x 32 GB consumer cards.
Neither patches vLLM itself: the package registers under the
``vllm.general_plugins`` entry-point group and is picked up by every vLLM
process automatically.

1. **Single-copy offload (always on).** Stock ``UVAOffloader`` does::

       cpu_data = p.data.to(device="cpu")   # pageable host copy
       cpu_data = cpu_data.pin_memory()     # second, pinned host copy

   Both buffers are alive at once and PyTorch's host allocator caches the
   pageable one, so a budget of N GiB per rank costs up to 2N GiB of host RAM.
   On this box, ``--cpu-offload-gb 24`` at TP8 drove ``Shmem`` past 460 GiB of
   503 GiB and the run died before the engine came up. This plugin allocates
   the pinned buffer directly and copies into it once: N GiB of budget costs
   N GiB of host RAM.

2. **Layer-range restriction (opt-in via ``DSV41_OFFLOAD_LAYERS``).** vLLM
   walks the decoder layers in order and offloads until the budget is spent,
   so the *first* layers always go to host memory. DeepSeek-V4.1-Flash is a
   causal encoder-decoder: prefill executes only layers 0-19, while layers
   20-39 run during decode (plus a 128-token replay). Offloading the decoder
   half costs the same GPU memory but keeps prefill entirely on-device.
   Set ``DSV41_OFFLOAD_LAYERS=20-39`` (ranges and lists both work, e.g.
   ``20-29,35-39``). Unset means stock ordering.

Set ``DSV41_SINGLE_COPY=0`` to disable fix 1 (for A/B measurement).
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

    spec = os.environ.get("DSV41_OFFLOAD_LAYERS", "").strip()
    allowed = parse_layer_set(spec) if spec else None
    single_copy = os.environ.get("DSV41_SINGLE_COPY", "1") not in ("0", "false", "False")
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
            cpu_data = torch.empty(
                p.data.shape, dtype=p.data.dtype, device="cpu", pin_memory=self.pin_memory
            )
            cpu_data.copy_(p.data)
            if self.uva_offloading:
                p.data = get_accelerator_view_from_cpu_tensor(cpu_data)
                p._vllm_is_uva_offloaded = True
            else:
                p.data = cpu_data
            self.cpu_offload_bytes += cpu_data.numel() * cpu_data.element_size()
            did = True
        return did

    def _maybe_offload_to_cpu(self, module, prefix: str = ""):
        if allowed is not None:
            idx = layer_index_of(module)
            # Modules that are not decoder layers (towers, heads) keep stock behaviour.
            if idx is not None and idx not in allowed:
                return module
        if not single_copy:
            return original(self, module, prefix)

        if (params := next(module.parameters(), None)) is None:
            return module
        if params.device == torch.device("cpu"):
            return module
        if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
            return module
        if prefix:
            prefix = prefix if prefix.endswith(".") else f"{prefix}."

        did = _offload_params_single_copy(self, module, prefix)
        if did and not self.uva_offloading:
            # Without UVA, stock vLLM installs a functional_call wrapper that
            # moves the module onto the device for each forward. Fall back to
            # it rather than reimplementing that path.
            return original(self, module, prefix)
        return module

    uva.UVAOffloader._maybe_offload_to_cpu = _maybe_offload_to_cpu
    logger.info(
        "vllm_dsv41_opt active: single_copy_offload=%s, offload_layers=%s",
        single_copy, sorted(allowed) if allowed is not None else "all",
    )
