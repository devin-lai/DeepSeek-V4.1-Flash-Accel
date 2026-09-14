# Documentation

Start with the [quickstart](../README.md#quickstart) and
[deployment runbook](../deploy/README.md). The reference configuration is
8× RTX 5090 with 503 GiB host RAM; other hardware needs validation.

| Guide | Purpose |
| --- | --- |
| [New text presets](../benchmarks/results/2026-09-14-v41-optimization.md) | Exact host allocation, static DSpark, wider batching, repeated trials and raw timings |
| [Benchmarks](../benchmarks/README.md) | Saved evidence, comparison scope, and reproduction commands |
| [Hardware](02-hardware-topology.md) | Reference GPU, CPU, RAM, and interconnect measurements |
| [Troubleshooting](05-fault-inventory.md) | Known failure signatures and fixes for the pinned stack |
| [Patches](../upstream/README.md) | Apply, inspect, and revert the vLLM and FlashInfer fixes |
| [Model anatomy](01-model-anatomy.md) | Architecture and memory costs relevant to deployment |
| [Deployment design](03-deployment-design.md) | Reasons for the reference placement and parallelism choices |
| [Optimization guide](04-optimization-guide.md) | Offload, memory, and kernel tradeoffs; distinguish estimates from measurements |
| [Engineering report](07-engineering-report.md) | Detailed investigation and separate V4/V4.1 measurements |
| [Expert quantization](06-expert-quantization.md) | Exploratory expert-level measurements; not validated serving formats |

For contributions, see [CONTRIBUTING.md](../CONTRIBUTING.md). Measurements
describe the recorded stack and workloads; they do not certify current upstream
versions or production reliability.
