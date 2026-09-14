"""Page-sized pinned allocations for long-lived, immutable CPU weights.

PyTorch's host caching allocator rounds these large allocations to powers of
two. Register an anonymous mapping instead, and let tensor storage own its
lifetime. This is deliberately scoped to weight creation: it is not a general
replacement for the caching allocator's asynchronous transfer tracking.
"""

from __future__ import annotations

import ctypes
import math
import mmap
import weakref

import torch

MIN_BYTES = 64 * 1024**2
_live_bytes = 0
_peak_bytes = 0
_failed_releases = []


def allocation_stats() -> dict[str, int]:
    return {"live_bytes": _live_bytes, "peak_bytes": _peak_bytes}


def _release(runtime, pointer: int, mapping, nbytes: int) -> None:
    global _live_bytes
    # CUDA views retain the CPU tensor, which retains the buffer. Only the last
    # storage reference reaches this point. Weight copies are synchronous.
    error = runtime.cudaHostUnregister(pointer)
    if int(error) != 0:
        # Never unmap memory that CUDA may still own.
        import warnings

        warnings.warn(f"cudaHostUnregister failed: {error}", ResourceWarning)
        _failed_releases.append(mapping)
        return
    mapping.close()
    _live_bytes -= nbytes


def empty_pinned(shape, *, dtype: torch.dtype) -> torch.Tensor:
    """Allocate contiguous CPU weight storage, without power-of-two rounding.

    Keep the returned tensor (or a storage alias) alive until all GPU work using
    it finishes. Do not use short-lived buffers for asynchronous H2D copies.
    """
    global _live_bytes, _peak_bytes
    shape = tuple(shape)
    if any(d < 0 for d in shape):
        raise ValueError(f"negative dimension in {shape}")
    itemsize = torch.empty((), device="cpu", dtype=dtype).element_size()
    nbytes = math.prod(shape) * itemsize
    if nbytes == 0:
        return torch.empty(shape, device="cpu", dtype=dtype, pin_memory=True)
    mapped_bytes = math.ceil(nbytes / mmap.PAGESIZE) * mmap.PAGESIZE
    mapping = mmap.mmap(-1, mapped_bytes)
    pointer = ctypes.addressof(ctypes.c_ubyte.from_buffer(mapping))
    runtime = torch.cuda.cudart()
    # Portable and mapped: all devices in this process may read this storage.
    error = runtime.cudaHostRegister(pointer, mapped_bytes, 3)
    if int(error) != 0:
        mapping.close()
        raise RuntimeError(f"cudaHostRegister({mapped_bytes} bytes) failed: {error}")
    buffer = (ctypes.c_ubyte * nbytes).from_address(pointer)
    finalizer = weakref.finalize(
        buffer, _release, runtime, pointer, mapping, mapped_bytes
    )
    # At interpreter shutdown CUDA and Torch may already be unloaded; the OS
    # releases process mappings. Normal storage destruction still frees them.
    finalizer.atexit = False
    _live_bytes += mapped_bytes
    _peak_bytes = max(_peak_bytes, _live_bytes)
    return torch.frombuffer(buffer, dtype=dtype).reshape(shape)


class WeightTorchProxy:
    """Override only large pinned weight allocations in selected vLLM modules."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def _eligible(shape, kwargs, dtype) -> bool:
        if not kwargs.get("pin_memory") or str(kwargs.get("device")) != "cpu":
            return False
        if kwargs.keys() - {"device", "dtype", "pin_memory", "requires_grad"}:
            return False
        size = (
            math.prod(shape) * torch.empty((), device="cpu", dtype=dtype).element_size()
        )
        return size >= MIN_BYTES

    def empty(self, *size, **kwargs):
        shape = (
            size[0]
            if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size))
            else size
        )
        dtype = kwargs.get("dtype", torch.get_default_dtype())
        if not self._eligible(shape, kwargs, dtype):
            return torch.empty(*size, **kwargs)
        return empty_pinned(shape, dtype=dtype).requires_grad_(
            kwargs.get("requires_grad", False)
        )

    def empty_like(self, tensor, **kwargs):
        dtype = kwargs.get("dtype", tensor.dtype)
        if not tensor.is_contiguous() or not self._eligible(
            tensor.shape, kwargs, dtype
        ):
            return torch.empty_like(tensor, **kwargs)
        return empty_pinned(tensor.shape, dtype=dtype).requires_grad_(
            kwargs.get("requires_grad", False)
        )


def install_weight_allocator() -> None:
    from vllm.model_executor.model_loader import utils as loader_utils
    from vllm.models.deepseek_v4_1.common import engram

    # No global torch monkey patch: activation buffers, IPC, and asynchronous
    # request transfers continue to use PyTorch's normal allocator.
    loader_utils.torch = WeightTorchProxy()
    engram.torch = WeightTorchProxy()
