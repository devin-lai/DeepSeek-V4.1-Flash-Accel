"""A routing profile must count unique experts, including the busiest rank."""

import json
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("use_graph", [False, True])
def test_unique_counts_and_histogram_include_repeated_and_boundary_experts(use_graph):
    from vllm_dsv41_opt.routing_profile import _count_routes

    ids = torch.tensor(
        [
            [0, 0, 47, 48, 95, 96],
            [96, 97, 98, 99, 100, 101],
            [143, 144, 191, 192, 239, 240],
            [240, 241, 242, 243, 244, 245],
            [287, 288, 335, 336, 382, 383],
            [336, 337, 338, 339, 340, 341],
        ],
        dtype=torch.int32,
    )
    unique = set(ids.flatten().tolist())
    expected = [
        sum(rank * 48 <= value < (rank + 1) * 48 for value in unique)
        for rank in range(8)
    ]
    device_ids = ids.cuda()
    counters = torch.zeros(59, dtype=torch.int64, device="cuda")
    if use_graph:
        # Compile eagerly; replay must accumulate into persistent counters.
        _count_routes[(1,)](device_ids, counters, num_warps=8)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _count_routes[(1,)](device_ids, counters, num_warps=8)
        counters.zero_()
    for _ in range(3):
        if use_graph:
            graph.replay()
        else:
            _count_routes[(1,)](device_ids, counters, num_warps=8)
    actual = counters.cpu().tolist()
    assert actual[0] == 3
    assert actual[1:9] == [3 * value for value in expected]
    assert actual[9] == 3 * max(expected)
    assert actual[10 + max(expected)] == 3
    assert sum(actual[10:]) == 3


def test_worker_rpc_resets_counters_created_in_inference_mode(monkeypatch, tmp_path):
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

    from vllm_dsv41_opt import routing_profile

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(Mxfp4MoEMethod, "apply", lambda *args: args[2])
    monkeypatch.setattr(routing_profile, "_snapshot_callback", None)
    destination = tmp_path / "routing.json"
    routing_profile.install_routing_profile(str(destination))
    layer = SimpleNamespace(
        global_num_experts=384,
        w13_weight=torch.empty(48, 1),
        layer_name="model.layers.0.experts",
    )
    with torch.inference_mode():
        ids = torch.arange(36, device="cuda", dtype=torch.int32).reshape(6, 6)
        x = torch.zeros(6, 1, device="cuda")
        Mxfp4MoEMethod.apply(None, layer, x, x, ids, None, None)
    worker = routing_profile.RoutingProfileWorkerExtension()
    worker.snapshot_dsv41_routing()
    first = json.loads(destination.read_text())
    assert first["generation"] == 1
    assert first["rows"][0]["steps"] == 1
    assert first["rows"][0]["mean_max_rank_unique"] == 36
    worker.snapshot_dsv41_routing()
    reset = json.loads(destination.read_text())
    assert reset["generation"] == 2
    assert reset["rows"][0]["steps"] == 0
    assert reset["rows"][0]["mean_max_rank_unique"] is None
