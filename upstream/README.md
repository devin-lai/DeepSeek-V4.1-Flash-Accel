# Patch guide and technical reports

Patches and eight technical reports for DeepSeek-V4.1-Flash on 8× RTX 5090.
The reports describe the pinned stack, failure mechanisms, reproductions, and
available evidence. They have not been submitted or accepted upstream.

| report | component | status |
| --- | --- | --- |
| [`flashinfer/ISSUE.md`](flashinfer/ISSUE.md) | FlashInfer | **patch attached and verified** (141/141 shapes) — sm_120 sparse-MLA *decode* dispatch |
| [`flashinfer/ISSUE-prefill.md`](flashinfer/ISSUE-prefill.md) | FlashInfer | **patch attached** — the *prefill* dispatch grid; the single-cache half is worked around in vLLM, the dual-cache path is instantiated at `topk` 2048 so images can be served |
| [`flashinfer/ISSUE-null-block.md`](flashinfer/ISSUE-null-block.md) | FlashInfer | **patch attached and verified** — every masked index gathers row 0 of block 0 and relies on `0 * row == 0` (FI-004) |
| [`vllm/ISSUE-block-size.md`](vllm/ISSUE-block-size.md) | vLLM + DeepGEMM | **patch attached** — block size per compression ratio |
| [`vllm/ISSUE-prefill-width.md`](vllm/ISSUE-prefill-width.md) | vLLM | **patch attached** — text-only index width, and the SWA page size |
| [`vllm/ISSUE-v41-cudagraphs.md`](vllm/ISSUE-v41-cudagraphs.md) | vLLM | **root cause established, patch attached and verified** — capture's dummy forwards write V4.1 state into the null block, which the sparse-MLA kernels gather for every masked index (VL-013) |
| [`vllm/ISSUE-startup-validation.md`](vllm/ISSUE-startup-validation.md) | vLLM | diagnosis, suggested fix |
| [`vllm/ISSUE-pp-input-ids.md`](vllm/ISSUE-pp-input-ids.md) | vLLM | diagnosis, three candidate fixes |

`faults/inventory.toml` carries the log signature of each, so
`tools/faultscan.py` can name them from a server log.

## Applying the patches

`scripts/env/setup.sh` does all of this. By hand:

```bash
PKG() { python -c "import $1, os; print(os.path.dirname($1.__file__))"; }

# FlashInfer: (FI-001) make TOPK and PAGE_BLOCK_SIZE runtime arguments of the
# sm_120 DSv4 sparse-MLA decode kernel -- required for V4.1 to decode at all;
# (FI-004) gather a finite row instead of block 0 for masked indices -- required
# with CUDA graphs; (FI-003) instantiate the dual-cache prefill at topk 2048 --
# required to serve images.
python upstream/flashinfer/apply_patch.py "$(PKG flashinfer)"

# ... and move the prebuilt .so aside, or the CUDA edit is silently ignored:
mv "$(PKG flashinfer_jit_cache)/jit_cache/sparse_mla_sm120" /var/tmp/

# vLLM: six edits across five files -- block size scaled by each layer's
# compression ratio, 64-token SWA pages, a prefill index width the kernel
# implements, no image widening when images cannot arrive, a block-size
# negotiation that can refuse an illegal size by name, and the null block
# scrubbed after CUDA-graph capture.
python upstream/vllm/apply_patch.py "$(PKG vllm)"
```

Both scripts are idempotent, keep a `.orig` beside each file they touch, and
support `--check` and `--revert`. `deploy/preflight.py --v41` checks every one
of the markers they introduce, so a partly applied stack is reported as such
rather than failing four minutes into a launch.

Verify the FlashInfer patch before trusting it — about two minutes, most of it
nvcc:

```bash
python tools/sm120_sparse_mla/probe_sm120.py --oracle --sweep
```

The `--oracle` rows check the PyTorch reference against shapes FlashInfer
already ships; only if those pass does the sweep mean anything.

## What the vLLM patch changes

| file | edit | fault |
| --- | --- | --- |
| `models/deepseek_v4_1/attention.py` | block size × compression ratio, in both spec builders | VL-009 |
| `models/deepseek_v4_1/attention.py` | SWA cache page 32 → 64 | VL-012 |
| `models/deepseek_v4_1/attention.py` | image widening gated on the multimodal config | VL-011 |
| `v1/attention/backends/mla/sparse_swa.py` | prefill index width rounded to an instantiated `TOPK` | FI-003 |
| `v1/attention/backends/mla/sparse_swa.py` | the same image gate, where it decides the width | VL-011 |
| `v1/worker/utils.py` | block-size negotiation can refuse a size by the constraint that rejects it | VL-009 |
| `v1/worker/gpu/model_runner.py` | block 0 of every KV cache zeroed at the end of `capture_model` | VL-013 |
| `models/deepseek_v4_1/{sparse_mla,nvidia/flashinfer_sparse}.py`, `v1/attention/backends/mla/indexer.py` | offer 64 alongside 128 on every architecture, not just sm_90 | VL-009 |

## Reverting

```bash
python upstream/flashinfer/apply_patch.py "$(PKG flashinfer)" --revert
python upstream/vllm/apply_patch.py "$(PKG vllm)" --revert
mv /var/tmp/sparse_mla_sm120 "$(PKG flashinfer_jit_cache)/jit_cache/"
rm -rf "${FLASHINFER_WORKSPACE_BASE:-$HOME}"/.cache/flashinfer/*/*/cached_ops/sparse_mla_sm120*
```

After applying or reverting the FlashInfer edits, clear that JIT cache
directory before the next launch (it lives under `FLASHINFER_WORKSPACE_BASE`
when that is set, else under `$HOME`); ninja does track header changes and
rebuilt on its own here, but a clean rebuild removes any doubt about which
kernel is running.
