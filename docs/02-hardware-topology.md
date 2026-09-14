# Hardware: what an 8× RTX 5090 box really is

Measurements below describe the reference machine used for this repository.
The deployment implications combine those measurements with estimates of the
model's workload. Other hardware needs its own topology and bandwidth checks.

## Reference machine

| Item | Value |
| --- | --- |
| GPUs | 8× NVIDIA GeForce RTX 5090, 32 607 MiB each (31.4 GiB usable), sm_120 |
| Driver / CUDA | 595.71.05 / CUDA 13.2 toolkit (torch 2.13 + cu130 wheels) |
| PCIe | Gen5 x16 per GPU, **no NVLink** |
| GPU P2P | **disabled** (GeForce driver policy) — NCCL falls back to `SHM/direct/direct` |
| CPU | 2× Intel Xeon Gold 6530 (32 cores each, 128 threads), AMX |
| NUMA | node 0 → GPU 0-3 + NIC, node 1 → GPU 4-7; cross-socket traffic over UPI |
| RAM | 503 GiB DDR5, no swap |
| Storage | 880 GB NVMe (OS) + 3.5 TB NVMe (`/data`, models live here) |
| External network | ~12 MB/s egress cap (a 510 GB checkpoint takes ~12 hours) |

`nvidia-smi topo -m` prints `NODE` inside a socket and `SYS` across sockets
for every GPU pair; there is no `NV#` anywhere.

## Interconnect numbers (NCCL 2.29.7, nccl-tests, 20 timed iterations)

| All-reduce group | 512 KiB latency | 256 MiB bus bandwidth |
| --- | ---: | ---: |
| GPU 0,1 (same socket) | 35.8 µs | 32.8 GB/s |
| GPU 0,4 (cross socket) | 38.0 µs | 32.1 GB/s |
| GPU 0-3 (one socket) | 55.7 µs | 38.2 GB/s |
| GPU 0,1,4,5 (split) | 67.4 µs | 37.8 GB/s |
| GPU 0-7 | 80.5 µs | 40.0 GB/s |

A single-GPU UVA read measured 51.3 GB/s locally and 51.1 GB/s remotely.
That does not predict eight-GPU contention: the concurrent read benchmark
measured **260.3 GB/s aggregate local vs 197.4 GB/s remote**. The exact-pinned
serving workers already place almost all host buffers on their GPU's local
node. Presets leave strict `--numa-bind` off because the original rounded
allocations could exhaust one socket; this is not a claim that NUMA is free.
See the [concurrent experiment](../benchmarks/results/2026-09-14-v41-optimization.md).

## What this means for a 552B MoE

1. **Weights do not fit.** 8 × 31.4 GiB = 251 GiB of GPU memory. The non-Engram
   part of DeepSeek-V4.1-Flash is ~292 GiB. Something has to live in host RAM.
2. **Decode all-reduces are cheap enough.** A TP-8 decode step issues on the
   order of 100 small all-reduces (10 KB each at batch 1). At ~80 µs each that
   is ≈ 8 ms per step — noticeable, not fatal. Large-message bandwidth
   (40 GB/s) only matters for prefill.
3. **PCIe is the budget for offloaded experts.** The UVA expert-read
   benchmark reached approximately 51 GB/s for one GPU, or 32.5 GB/s per GPU
   with all eight reading local memory simultaneously. Each
   decode token touches 6 experts × 40 layers ≈ 4.3 GB of MXFP4 expert bytes
   across all ranks; the fraction that is offloaded is what you pay for.
4. **Two NUMA islands.** TP-4 inside a socket plus PP-2 across the UPI link is
   a candidate layout: all-reduces stay on one socket, and only one activation
   tensor crosses the UPI per layer boundary. V4.1 pipeline parallelism remains
   blocked by the `input_ids` handoff issue in the current stack (VL-002).

## Quick topology checks for your own box

```bash
nvidia-smi topo -m
nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv
lscpu | grep -E "NUMA|Model name"
numactl -H | head -20
free -g
```

If `nvidia-smi topo -m` shows `NV#`, record those NVLink connections in your
benchmark report. Interconnect differences require their own measurements;
the results here come from a machine without NVLink.
