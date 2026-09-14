"""Fallback offloading must still install vLLM's per-forward transfer wrapper."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_exact_pinned_rejects_disabling_its_single_copy_path(monkeypatch):
    from vllm_dsv41_opt import register

    monkeypatch.setenv("DSV41_EXACT_PINNED", "1")
    monkeypatch.setenv("DSV41_SINGLE_COPY", "0")
    with pytest.raises(ValueError, match="requires DSV41_SINGLE_COPY"):
        register()


def test_disabled_uva_runs_cpu_weights_on_gpu(monkeypatch):
    from vllm.model_executor.offloader.uva import UVAOffloader
    from vllm_dsv41_opt import register

    monkeypatch.setenv("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", "1")
    monkeypatch.setenv("DSV41_SINGLE_COPY", "1")
    monkeypatch.setenv("DSV41_EXACT_PINNED", "0")
    monkeypatch.delenv("DSV41_OFFLOAD_LAYERS", raising=False)
    monkeypatch.setattr(
        UVAOffloader, "_maybe_offload_to_cpu", UVAOffloader._maybe_offload_to_cpu
    )
    register()
    layer = torch.nn.Linear(8, 4, bias=False, device="cuda")
    inputs = torch.ones(2, 8, device="cuda")
    expected = layer(inputs).detach()
    offloader = UVAOffloader(cpu_offload_max_bytes=1)
    wrapped = offloader.wrap_modules(iter([layer]))[0]
    assert wrapped.weight.device.type == "cpu"
    for _ in range(2):
        torch.testing.assert_close(wrapped(inputs), expected)
