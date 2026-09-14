---
name: Hardware or benchmark report
about: Share a reproduction, another GPU setup, or an optimization result
title: "[Measurement] "
---

## Result and scope

What worked or failed? Is this a reproduction, an ablation, or a new setup?

## Configuration

- GPU count/model/VRAM, CPU, host RAM, topology, and peak memory:
- Model ID, revision, tokenizer, and quantization:
- Project commit, dependency versions, and patches:
- Exact launch command, preset, graph mode, and offload settings:

## Workload and baseline

Exact benchmark command, dataset, input/output lengths, request count,
client concurrency, server sequence cap, warm-up/cache policy, and EOS behavior.
Identify the baseline and every configuration difference.

## Measurements

| Configuration | Output tok/s | TTFT | TPOT | Failures | Repetitions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | | | | | |
| Changed | | | | | |

Specify units, median/percentile, and variability. Attach individual result
files and relevant logs using the protocol in `benchmarks/README.md`.

## Output quality

Attach generated-output checks and any task evaluations. State what remains
untested, including vision or long context. Sanity probes alone do not prove
full-model quality parity.
