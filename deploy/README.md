# Deployment runbook

[Quickstart](../README.md#quickstart) · [Benchmarks](../benchmarks/README.md) · [Patches](../upstream/README.md)

The reference V4.1 setup uses Linux, 8× RTX 5090, 503 GiB host RAM,
approximately 476 GiB of weights, and the pinned stack installed by
`scripts/env/setup.sh`. The presets configure 32,768 tokens of context.

## Install and launch

Follow the [quickstart](../README.md#quickstart) for download and installation.
Run from the repository root:

```bash
source /data/venvs/vllm-dsv41/bin/activate
HOST=127.0.0.1 MODEL=/data/models/DeepSeek-V4.1-Flash \
  PRESET=v41-flash deploy/serve.sh
```

The launcher runs preflight, starts vLLM in the foreground, and saves logs
under `/data/logs`. A preflight exit code of 1 blocks startup; 2 reports
warnings. Download and first compilation can take substantial time; the
recorded eager startup with a warm OS page cache took approximately 285 s.

In another shell, check generated output before benchmarking:

```bash
source /data/venvs/vllm-dsv41/bin/activate
deploy/healthcheck.sh
python deploy/verify.py --json /data/verify.json
```

Read the probe JSON as well as the verdict. These are small sanity checks,
not a quality evaluation. The current verdict can pass without a perplexity
result and does not grade the chat answer. A `/health` response alone cannot
identify the incorrect-output faults documented in the inventory.

## Choose a preset

| Preset | Model and purpose |
| --- | --- |
| `v41-flash` | V4.1-Flash text, CUDA graphs, decoder expert offload, 32K context cap |
| `v41-flash-vision` | V4.1-Flash images and text, same context cap; simple image probe recorded |
| `default`, `latency`, `throughput`, `long-context` | Earlier V4-Flash experiments; these are a different model |

For images, stop the text server and restart with `PRESET=v41-flash-vision`.
The FI-003 FlashInfer patch is required. Vision throughput and broader quality
have not been evaluated.

V4.1 uses TP8 + expert parallelism, Marlin, CPU Engram, and 12 GiB of expert
weights offloaded per rank from decoder layers 20–39. CUDA graphs require the
VL-013 and FI-004 fixes; the other dispatch and cache patches are also required.
See the [patch guide](../upstream/README.md) to inspect or revert them.

The launcher defaults to the earlier V4 `default` preset for compatibility.
Always specify `PRESET=v41-flash` for this project's main model.

## Local configuration

| Setting | Default | Purpose |
| --- | --- | --- |
| `MODEL` | `/data/models/DeepSeek-V4.1-Flash` with a V4.1 preset | Checkpoint directory |
| `VENV` | `/data/venvs/vllm-dsv41` | Installed Python environment |
| `HOST`, `PORT` | `0.0.0.0`, `8000` | Listening address and port |
| `LOG_DIR` | `/data/logs` | Server logs |
| `CUDA_HOME` | Detected under `/usr/local/cuda*` | CUDA toolkit |

Use `HOST=127.0.0.1` for local access. Binding all interfaces exposes an
unauthenticated API unless you configure authentication and network access.

Presets assign tuning values directly, so caller environment variables do not
override settings such as `ENFORCE_EAGER` or `MAX_LEN`. To change these, copy a
preset to an ignored local file, edit it, and use its filename without `.env`:

```bash
cp deploy/presets/v41-flash.env deploy/presets/v41-flash-eager.env
# Edit v41-flash-eager.env: change ENFORCE_EAGER=0 to ENFORCE_EAGER=1.
HOST=127.0.0.1 MODEL=/data/models/DeepSeek-V4.1-Flash \
  PRESET=v41-flash-eager deploy/serve.sh
```

Stop the previous server before restarting on the same GPUs. See the
[benchmark protocol](../benchmarks/README.md) for comparable graph/eager runs.

## Run with systemd

First verify foreground serving. Place the checkout at
`/opt/DeepSeek-V4.1-Flash-Accel`, or edit the unit's paths to its actual location.
The environment example contains plain assignments for systemd; shell presets
are read separately by the launcher.

```bash
sudo cp deploy/systemd/dsv41-flash.env.example /etc/dsv41-flash.env
# Edit /etc/dsv41-flash.env for your model and environment paths.
sudo cp deploy/systemd/dsv41-flash.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dsv41-flash
sudo journalctl -u dsv41-flash -f
```

The unit permits pinned memory and limits repeated restarts. It does not bind
memory to one NUMA node. The example listens on loopback by default; change
`HOST` only for the access pattern you intend. This service wrapper has not
been validated on the reference GPU host.

## Diagnose a failure

```bash
# Inspect a failed launch or incorrect output.
python tools/faultscan.py /data/logs/serve-example.log

# Check hardware, installed patches, and checkpoint before launching.
python deploy/preflight.py --model /data/models/DeepSeek-V4.1-Flash \
  --tp 8 --expert-parallel --offload-gb 12 --engram-gib 264 \
  --block-size 64 --v41 --text-only

# Estimate memory from checkpoint headers.
python tools/plan_memory.py /data/models/DeepSeek-V4.1-Flash --tp 8 --ep
```

The [fault inventory](../docs/05-fault-inventory.md) explains recognized
failures. For a new one, [open a deployment report](https://github.com/devin-lai/DeepSeek-V4.1-Flash-Accel/issues/new?template=deployment.md)
with the exact versions, launch command, and a sanitized log excerpt.
