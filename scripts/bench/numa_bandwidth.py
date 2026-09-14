#!/usr/bin/env python3
"""Measure simultaneous GPU reads from local versus remote NUMA host memory.

Requires Linux, numactl, CUDA, Triton, vLLM and vllm-dsv41-opt. Run while the
inference server is idle. The default topology is the reference dual-socket,
eight-GPU host; pass --gpu-nodes for another dual-socket layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import triton
import triton.language as tl


def worker(args):
    import torch
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    from vllm_dsv41_opt.pinned import empty_pinned

    @triton.jit
    def read_host(src, dst, size: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(src + offsets, offsets < size, other=0)
        tl.store(dst + offsets, values, offsets < size)

    torch.cuda.set_device(args.worker)
    size = args.mib * 1024**2
    host = empty_pinned((size,), dtype=torch.uint8)
    host.fill_(7)
    source = get_accelerator_view_from_cpu_tensor(host)
    output = torch.empty(size, dtype=torch.uint8, device="cuda")
    numa_line = next(
        (
            line
            for line in Path("/proc/self/numa_maps").read_text().splitlines()
            if line.startswith(f"{host.data_ptr():x} ")
        ),
        "",
    )
    pages = dict(
        (int(node), int(count))
        for node, count in re.findall(r"N(\d+)=(\d+)", numa_line)
    )

    def launch():
        read_host[(triton.cdiv(size, 4096),)](source, output, size, 4096)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        launch()
        launch()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(args.iterations):
            launch()
    torch.cuda.synchronize()
    (args.dir / f"ready-{args.worker}").write_text("ready")
    deadline = time.monotonic() + 120
    while not (args.dir / "go").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("parent did not release the start barrier")
        time.sleep(0.01)
    rates = []
    for _ in range(3):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(4):
            graph.replay()
        end.record()
        end.synchronize()
        rates.append(
            size * args.iterations * 4 / (start.elapsed_time(end) / 1000) / 1e9
        )
    assert output[0].item() == 7 and output[-1].item() == 7
    result = {
        "gpu": args.worker,
        "numa_pages": pages,
        "bytes_per_kernel": size,
        "GBps_trials": rates,
        "median_GBps": statistics.median(rates),
        "verified": True,
    }
    (args.dir / f"gpu-{args.worker}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--gpu-nodes", default="0,0,0,0,1,1,1,1")
    ap.add_argument("--mib", type=int, default=256)
    ap.add_argument("--iterations", type=int, default=64)
    ap.add_argument("--worker", type=int)
    args = ap.parse_args()
    if args.worker is not None:
        worker(args)
        return
    nodes = [int(x) for x in args.gpu_nodes.split(",")]
    if set(nodes) != {0, 1} or args.mib <= 0 or args.iterations <= 0:
        ap.error(
            "provide a two-socket node map (0 and 1), positive size and iterations"
        )
    args.dir.mkdir(parents=True, exist_ok=False)
    results = []
    for placement in ("local", "remote"):
        folder = args.dir / placement
        folder.mkdir()
        procs = []
        try:
            for gpu, local_node in enumerate(nodes):
                node = local_node if placement == "local" else 1 - local_node
                cmd = [
                    "numactl",
                    f"--cpunodebind={node}",
                    f"--membind={node}",
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    str(gpu),
                    "--dir",
                    str(folder),
                    "--mib",
                    str(args.mib),
                    "--iterations",
                    str(args.iterations),
                ]
                with (folder / f"gpu-{gpu}.log").open("w") as log:
                    procs.append(
                        subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                    )
            deadline = time.monotonic() + 120
            while len(list(folder.glob("ready-*"))) != len(nodes):
                if any(p.poll() is not None for p in procs):
                    raise RuntimeError(
                        f"{placement}: worker failed; inspect worker logs"
                    )
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{placement}: workers did not become ready")
                time.sleep(0.1)
            (folder / "go").write_text("go")
            for proc in procs:
                if proc.wait(timeout=60) != 0:
                    raise RuntimeError(
                        f"{placement}: worker failed; inspect worker logs"
                    )
            rows = [
                json.loads((folder / f"gpu-{gpu}.json").read_text())
                for gpu in range(len(nodes))
            ]
            result = {
                "placement": placement,
                "gpu_nodes": nodes,
                "ranks": rows,
                "aggregate_GBps": sum(row["median_GBps"] for row in rows),
                "slowest_gpu_GBps": min(row["median_GBps"] for row in rows),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.terminate()
            for proc in procs:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
    (args.dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
