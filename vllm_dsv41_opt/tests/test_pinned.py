"""Storage lifetime and exact allocation checks; run on a CUDA Linux host."""

import gc
import math
import mmap

import pytest
import torch

from vllm_dsv41_opt.pinned import WeightTorchProxy, allocation_stats, empty_pinned

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.uint8, torch.float8_e4m3fn]
)
def test_uva_alias_retains_storage_until_last_view(dtype):
    """A CUDA view must retain registered memory after the CPU tensor dies."""
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    before = allocation_stats()["live_bytes"]
    cpu = empty_pinned((4097,), dtype=dtype)
    cpu.fill_(2)
    assert cpu.is_pinned()
    alias = get_accelerator_view_from_cpu_tensor(cpu)[1:]
    del cpu
    gc.collect()
    assert allocation_stats()["live_bytes"] > before
    torch.testing.assert_close(alias.float().cpu(), torch.full((4096,), 2.0))
    torch.cuda.synchronize()
    del alias
    gc.collect()
    assert allocation_stats()["live_bytes"] == before


def test_large_weight_is_page_rounded_and_global_torch_is_unchanged():
    before = allocation_stats()["live_bytes"]
    original_empty = torch.empty
    proxy = WeightTorchProxy()
    size = 72 * 1024**2 + 17
    weight = proxy.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)
    assert weight.is_pinned() and weight.numel() == size
    assert (
        allocation_stats()["live_bytes"] - before
        == math.ceil(size / mmap.PAGESIZE) * mmap.PAGESIZE
    )
    assert torch.empty is original_empty
    del weight
    gc.collect()
    assert allocation_stats()["live_bytes"] == before


def test_noncontiguous_empty_like_preserves_strides():
    source = torch.empty(4096, 4096, device="cpu").T
    result = WeightTorchProxy().empty_like(source, device="cpu", pin_memory=True)
    assert result.stride() == source.stride()
    assert result.is_pinned()


def test_empty_and_invalid_shape():
    assert empty_pinned((0, 7), dtype=torch.uint8).shape == (0, 7)
    with pytest.raises(ValueError, match="negative dimension"):
        empty_pinned((-1,), dtype=torch.uint8)
