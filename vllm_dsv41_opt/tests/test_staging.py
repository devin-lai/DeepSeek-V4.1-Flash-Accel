"""Staging must preserve bytes and must not capture large weight allocations."""

import pytest
import torch
from vllm_dsv41_opt.staging import staged_marlin_gemm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_staging_is_exact_and_only_applies_to_large_offloaded_batches():
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    cpu = torch.arange(4096, dtype=torch.int32).pin_memory()
    weight = get_accelerator_view_from_cpu_tensor(cpu)
    weight._vllm_is_uva_offloaded = True
    output = torch.empty_like(weight)
    pointers = []

    def gemm(*, b_qweight, topk_weights):
        pointers.append(b_qweight.data_ptr())
        output.copy_(b_qweight)
        return output

    staged = staged_marlin_gemm(gemm, 256)
    for count, offloaded, expect_copy in [
        (255, True, False),
        (256, True, True),
        (256, False, False),
    ]:
        weight._vllm_is_uva_offloaded = offloaded
        staged(b_qweight=weight, topk_weights=torch.empty(count, 6))
        torch.testing.assert_close(output.cpu(), cpu, rtol=0, atol=0)
        assert (pointers[-1] != weight.data_ptr()) == expect_copy


def test_graph_capture_keeps_uva_weights_and_replay_reads_current_storage():
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    cpu = torch.ones(4096, dtype=torch.int32, pin_memory=True)
    weight = get_accelerator_view_from_cpu_tensor(cpu)
    weight._vllm_is_uva_offloaded = True
    output = torch.empty_like(weight)
    pointers = []

    def gemm(*args):
        pointers.append(args[2].data_ptr())
        return output.copy_(args[2])

    staged = staged_marlin_gemm(gemm, 256)
    args = (None, None, weight, *([None] * 9), torch.empty(256, 6))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        staged(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        staged(*args)
    assert pointers[-1] == weight.data_ptr()
    torch.cuda.synchronize()
    cpu.fill_(7)
    graph.replay()
    torch.testing.assert_close(output.cpu(), cpu, rtol=0, atol=0)


def test_failed_staging_allocation_falls_back_to_original_gemm(monkeypatch):
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    cpu = torch.arange(4096, dtype=torch.int32).pin_memory()
    weight = get_accelerator_view_from_cpu_tensor(cpu)
    weight._vllm_is_uva_offloaded = True
    output = torch.empty_like(weight)
    reports = []

    def fail_clone(self, **kwargs):
        raise torch.OutOfMemoryError("staging allocation could not fit")

    def gemm(*, b_qweight, topk_weights):
        assert b_qweight is weight
        return output.copy_(b_qweight)

    staged = staged_marlin_gemm(
        gemm, 256, on_first_fallback=lambda w, n: reports.append((w.data_ptr(), n))
    )
    monkeypatch.setattr(torch.Tensor, "clone", fail_clone)
    for _ in range(2):
        staged(b_qweight=weight, topk_weights=torch.empty(256, 6))
        assert torch.equal(output.cpu(), cpu)
    assert reports == [(weight.data_ptr(), 256)]
