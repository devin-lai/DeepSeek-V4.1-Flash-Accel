#!/usr/bin/env python3
"""Check a box before serving a large MoE, and lint the launch flags.

Every check here corresponds to something that has actually gone wrong on this
hardware; `faults/inventory.toml` carries the long form.  Run it before the
first start on a new machine, and from the systemd unit's `ExecStartPre` after
that -- most of these are cheap, and the expensive failures they prevent cost
six minutes of weight loading each.

    python deploy/preflight.py --model /data/models/DeepSeek-V4-Flash-NVFP4
    python deploy/preflight.py --model ... --tp 8 --expert-parallel --offload-gb 12

Exit status: 0 all clear, 1 a blocking problem, 2 warnings only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys

GIB = 1024**3
OK, WARN, FAIL = "ok", "warn", "fail"
_RESULTS: list[tuple[str, str, str]] = []


def record(level: str, name: str, detail: str) -> None:
    _RESULTS.append((level, name, detail))


def sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


# --------------------------------------------------------------------- host


def check_gpus(want_gpus: int) -> list[float]:
    out = sh("nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap", "--format=csv,noheader,nounits")
    if not out:
        record(FAIL, "gpus", "nvidia-smi produced nothing; is the driver loaded?")
        return []
    rows = [r.split(", ") for r in out.splitlines()]
    gib = [int(r[2]) / 1024 for r in rows]
    caps = {r[3] for r in rows}
    names = {r[1] for r in rows}
    detail = f"{len(rows)}x {', '.join(sorted(names))}, {gib[0]:.1f} GiB each, cc {', '.join(sorted(caps))}"
    if len(rows) < want_gpus:
        record(FAIL, "gpus", f"{detail} -- need {want_gpus}")
    elif len(caps) > 1:
        record(WARN, "gpus", f"{detail} -- mixed compute capability")
    else:
        record(OK, "gpus", detail)

    used = [int(x) for x in sh("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits").split()]
    busy = [i for i, m in enumerate(used) if m > 512]
    if busy:
        record(FAIL, "gpu free", f"GPUs {busy} already hold memory; another server is running")
    else:
        record(OK, "gpu free", "all GPUs idle")
    return gib


def check_nvcc() -> None:
    # vLLM gates ALL of FlashInfer on shutil.which("nvcc") unless the
    # flashinfer-cubin package is installed, and reports the absence as a
    # missing kernel specialisation rather than a missing compiler.
    if shutil.which("nvcc"):
        record(OK, "nvcc", shutil.which("nvcc"))
        return
    guess = glob.glob("/usr/local/cuda*/bin/nvcc")
    if guess:
        record(FAIL, "nvcc",
               f"not on PATH but present at {guess[0]} -- "
               f"export PATH={os.path.dirname(guess[0])}:$PATH")
    else:
        record(FAIL, "nvcc", "not found; FlashInfer cannot JIT-compile")


# Host bytes an offload budget actually pins, measured at EP8 on 8x RTX 5090:
# 12.3 GiB/rank of expert weights across 29 buffers occupied 22.0 GiB.
# `tools/plan_memory.py` models the same thing buffer by buffer.
OFFLOAD_PIN_FACTOR = 1.78


def check_host_ram(engram_gib: float, offload_gb: float, tp: int) -> None:
    try:
        with open("/proc/meminfo") as fh:
            meminfo = dict(
                (k.strip(), v) for k, v in (line.split(":", 1) for line in fh)
            )
    except OSError:
        # Not Linux: the numbers below are the whole point of this check, so
        # say so rather than guessing at them.
        record(WARN, "host ram", "no /proc/meminfo; run this on the serving host")
        return
    total = int(meminfo["MemTotal"].split()[0]) / (1024 * 1024)
    avail = int(meminfo["MemAvailable"].split()[0]) / (1024 * 1024)
    swap = int(meminfo["SwapTotal"].split()[0]) / (1024 * 1024)

    # Pinned allocations round up to the next power of two and the knobs that
    # look like they control it do not (PT-001). The Engram figure passed in
    # already accounts for that; the offload budget does not, because the
    # offloader pins one buffer per parameter and expert matrices are nowhere
    # near powers of two. Measured at EP8: 12.3 GiB/rank of weights occupy
    # 22.0. Summing the budget instead under-predicts by ~100 GiB on this box.
    need = engram_gib + offload_gb * tp * OFFLOAD_PIN_FACTOR
    detail = f"{total:.0f} GiB total, {avail:.0f} GiB available, {swap:.0f} GiB swap"
    if need:
        detail += (f"; this config needs about {need:.0f} GiB pinned"
                   f" ({offload_gb:.0f} GiB/rank offload x {OFFLOAD_PIN_FACTOR} "
                   f"for pinned rounding)" if offload_gb else
                   f"; this config needs about {need:.0f} GiB pinned")
    if need and need > avail:
        record(FAIL, "host ram", detail + " -- will not fit")
    elif need and need > avail * 0.92:
        record(WARN, "host ram", detail + " -- under 8 % headroom")
    else:
        record(OK, "host ram", detail)
    if swap > 0:
        record(WARN, "swap", f"{swap:.0f} GiB of swap is on; pinned pages cannot swap, "
                             "but everything else thrashing around them will")


def check_numa() -> None:
    nodes = sorted(glob.glob("/sys/devices/system/node/node[0-9]*"))
    if len(nodes) <= 1:
        record(OK, "numa", "single node")
        return
    sizes = []
    for n in nodes:
        try:
            for line in open(f"{n}/meminfo"):
                if "MemTotal" in line:
                    sizes.append(int(line.split()[3]) / (1024 * 1024))
                    break
        except OSError:
            pass
    record(OK, "numa", f"{len(nodes)} nodes, {', '.join(f'{s:.0f} GiB' for s in sizes)} "
                       "-- do NOT pass --numa-bind with host-offloaded weights (VL-004)")


# -------------------------------------------------------------------- model


def check_model(model_dir: str) -> dict:
    path = pathlib.Path(model_dir)
    if not path.is_dir():
        record(FAIL, "model", f"{model_dir} is not a directory")
        return {}
    cfg_path = path / "config.json"
    if not cfg_path.exists():
        record(FAIL, "model", f"no config.json in {model_dir}")
        return {}
    cfg = json.loads(cfg_path.read_text())

    index = path / "model.safetensors.index.json"
    shards = sorted(path.glob("*.safetensors"))
    if index.exists():
        want = set(json.loads(index.read_text())["weight_map"].values())
        missing = sorted(w for w in want if not (path / w).exists())
        if missing:
            record(FAIL, "shards", f"{len(missing)} of {len(want)} shards missing, "
                                   f"first: {missing[0]}")
        else:
            record(OK, "shards", f"{len(want)} shards present")
    elif shards:
        record(WARN, "shards", f"{len(shards)} safetensors files, no index to check against")
    else:
        record(FAIL, "shards", "no safetensors found")

    # Cheap truncation check: the header length prefix must fit in the file.
    bad = []
    for s in shards[:4] + shards[-4:]:
        try:
            with open(s, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
            if 8 + n > s.stat().st_size:
                bad.append(s.name)
        except (OSError, struct.error):
            bad.append(s.name)
    if bad:
        record(FAIL, "shard headers", f"truncated or unreadable: {', '.join(bad)}")
    else:
        record(OK, "shard headers", f"sampled {min(8, len(shards))} shards, headers intact")

    total = sum(s.stat().st_size for s in shards) / GIB
    text = cfg.get("text_config", cfg)
    record(OK, "model", f"{text.get('model_type', '?')}, {text.get('num_hidden_layers', '?')} layers, "
                        f"{text.get('n_routed_experts', '?')} experts, {total:.1f} GiB on disk")
    return cfg


def check_stack() -> None:
    try:
        import torch
        record(OK, "torch", f"{torch.__version__}, cuda {torch.version.cuda}")
    except ImportError:
        record(FAIL, "torch", "not importable")
        return
    try:
        import vllm
        record(OK, "vllm", vllm.__version__)
    except ImportError:
        record(FAIL, "vllm", "not importable")
    try:
        import flashinfer
        record(OK, "flashinfer", flashinfer.__version__)
    except ImportError:
        record(WARN, "flashinfer", "not importable")
    try:
        import vllm_dsv41_opt  # noqa: F401
        record(OK, "vllm_dsv41_opt", "installed (single-copy offload, layer ranges)")
    except ImportError:
        record(WARN, "vllm_dsv41_opt", "not installed; stock two-copy offloader will be used (VL-008)")


# ------------------------------------------------------- V4.1-specific stack


_DUAL_PREFILL_WIDE = False


def _module_dir(name: str):
    try:
        mod = __import__(name)
    except Exception:  # noqa: BLE001
        return None
    return pathlib.Path(mod.__file__).parent


def check_v41_patches() -> None:
    """The three patch sets DeepSeek-V4.1 needs on sm_120, by their markers.

    Each is checked for the symbol or string the patch introduces, so a partly
    applied stack is reported as such instead of failing four minutes into a
    launch.
    """
    vllm_dir = _module_dir("vllm")
    if vllm_dir is None:
        record(FAIL, "v41 patches", "vllm not importable")
        return

    wanted = [
        ("block size per compression ratio (VL-009)",
         "models/deepseek_v4_1/attention.py", "states_per_block_to_tokens"),
        ("SWA 64-token pages (VL-012)",
         "models/deepseek_v4_1/attention.py", "block_size=64,"),
        ("prefill index width (FI-003)",
         "v1/attention/backends/mla/sparse_swa.py", "_sm120_prefill_index_width"),
        ("text-only index width (VL-011)",
         "v1/attention/backends/mla/sparse_swa.py", "images_can_arrive"),
        ("states-aware block negotiation (VL-009)",
         "v1/worker/utils.py", "dsa_indexer_state_filter"),
        ("null block scrubbed after CUDA-graph capture (VL-013)",
         "v1/worker/gpu/model_runner.py", "VL-013"),
    ]
    missing = []
    for label, rel, marker in wanted:
        f = vllm_dir / rel
        if not f.exists() or marker not in f.read_text():
            missing.append(label)
    if missing:
        record(FAIL, "vllm patches",
               f"{len(missing)} of {len(wanted)} missing: {'; '.join(missing)}. "
               "Run upstream/vllm/apply_patch.py against "
               f"{vllm_dir}")
    else:
        record(OK, "vllm patches", f"all {len(wanted)} applied")

    fi_dir = _module_dir("flashinfer")
    if fi_dir is None:
        record(WARN, "flashinfer patch", "flashinfer not importable")
    else:
        cu = fi_dir / "data/csrc/sparse_mla_sm120_decode_dsv4.cu"
        if not cu.exists():
            record(WARN, "flashinfer patch", f"{cu} not found; cannot verify")
        elif "page_block_size != 64" in cu.read_text():
            record(FAIL, "flashinfer patch",
                   "sm_120 decode still rejects page_block_size != 64 (FI-001). "
                   "Run upstream/flashinfer/apply_patch.py")
        else:
            record(OK, "flashinfer patch", "sm_120 decode dispatch is runtime (FI-001)")
        io = fi_dir / "data/include/flashinfer/attention/sparse_mla_sm120/common/kv_cache_io.cuh"
        if io.exists() and "if (idx < 0) idx = indices[0];" not in io.read_text():
            record(FAIL, "flashinfer patch",
                   "sm_120 sparse-MLA still gathers row 0 of block 0 for masked "
                   "indices (FI-004); with CUDA graphs on, V4.1 returns NaN. "
                   "Run upstream/flashinfer/apply_patch.py and clear "
                   "${FLASHINFER_WORKSPACE_BASE:-~}/.cache/flashinfer/*/*/cached_ops/sparse_mla_sm120*")
        elif io.exists():
            record(OK, "flashinfer patch", "masked-index gathers are null-block safe (FI-004)")
        pf = fi_dir / "data/csrc/sparse_mla_sm120_prefill.cu"
        global _DUAL_PREFILL_WIDE
        _DUAL_PREFILL_WIDE = pf.exists() and "DISPATCH_BY_NH_CM_TK_PBSX" in pf.read_text()
        record(OK if _DUAL_PREFILL_WIDE else WARN, "flashinfer patch",
               "dual-cache prefill instantiated at topk 2048: images can be served (FI-003)"
               if _DUAL_PREFILL_WIDE else
               "dual-cache prefill only at topk 128: text-only serving (FI-003). "
               "Run upstream/flashinfer/apply_patch.py to serve images")

    # FI-002: a prebuilt .so in flashinfer-jit-cache shadows any csrc edit.
    jit = _module_dir("flashinfer_jit_cache")
    if jit is not None and (jit / "jit_cache/sparse_mla_sm120").exists():
        record(FAIL, "flashinfer jit cache",
               "flashinfer_jit_cache ships a prebuilt sparse_mla_sm120 module that "
               "shadows the patched csrc (FI-002). Move it aside: "
               f"mv {jit}/jit_cache/sparse_mla_sm120 /var/tmp/")
    elif jit is not None:
        record(OK, "flashinfer jit cache", "no sparse_mla_sm120 shadow (FI-002)")


def lint_v41_flags(args) -> None:
    """Flags V4.1 needs on sm_120 that nothing else will tell you about."""
    if args.block_size != 64:
        record(FAIL, "flags",
               f"--block-size {args.block_size or '(unset)'}: V4.1 needs 64. It counts "
               "indexer STATES per block and is scaled by each layer's compression "
               "ratio, and 64 is the only value DeepGEMM accepts on sm_120 with an "
               "FP8 indexer cache (VL-009)")
    if not args.text_only and not _DUAL_PREFILL_WIDE:
        record(FAIL, "flags",
               "V4.1 on sm_120 must serve text-only (--language-model-only) unless "
               "the FI-003 FlashInfer patch is applied: the stock dual-cache "
               "sparse-MLA prefill kernel is instantiated only at topk=128, the SWA "
               "window with no image tokens")
    if args.enforce_eager:
        record(WARN, "flags",
               "--enforce-eager is no longer needed for V4.1 on sm_120 (VL-013 and "
               "FI-004 are patched) and costs most of the decode throughput; drop it")
    if args.engram_gib <= 0:
        record(WARN, "flags",
               "no Engram host offload: its two tables are 189 GiB of FP8 that are "
               "indexed, not multiplied, and pin ~264 GiB of host RAM when offloaded "
               "(PT-001). Pass --engram-config '{\"cpu_offload\": true}'")


# ---------------------------------------------------------------- flag lint


def lint_flags(args) -> None:
    if args.numa_bind and args.offload_gb > 0:
        record(FAIL, "flags", "--numa-bind with host-offloaded weights OOM-kills a worker "
                              "and buys no bandwidth (VL-004)")
    elif args.numa_bind:
        record(WARN, "flags", "--numa-bind measured within noise here (118.0 vs 117.9 tok/s); "
                              "it is only a liability (VL-004)")
    if not args.expert_parallel and args.tp > 1:
        record(WARN, "flags", "no --enable-expert-parallel: MXFP4 pads the per-rank intermediate "
                              "size, costing up to 1.33x the expert bytes, and EP measured "
                              "+79 % single-stream here (VL-006)")
    if args.autotune:
        record(WARN, "flags", "FlashInfer autotune inside the startup collective has hung "
                              "multi-GPU launches for 40 min; "
                              "--kernel-config '{\"enable_flashinfer_autotune\": false}' (VL-003)")
    if args.block_size and not args.v41:
        record(WARN, "flags", f"--block-size {args.block_size} is a backend constraint, not a "
                              "tuning knob, and does not port between checkpoints (VL-005)")
    if args.offload_gb > 0:
        record(WARN, "flags", f"{args.offload_gb} GiB/rank offloaded: expect roughly linear "
                              "decode loss (6 GiB/rank halved it here) (VL-007)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--expert-parallel", action="store_true")
    ap.add_argument("--offload-gb", type=float, default=0.0)
    ap.add_argument("--engram-gib", type=float, default=0.0,
                    help="pinned host GiB for Engram tables (V4.1: 264)")
    ap.add_argument("--numa-bind", action="store_true")
    ap.add_argument("--autotune", action="store_true")
    ap.add_argument("--block-size", type=int, default=0)
    ap.add_argument("--v41", action="store_true",
                    help="apply the DeepSeek-V4.1 sm_120 checks (patches and flags)")
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()

    check_gpus(args.gpus)
    check_nvcc()
    check_host_ram(args.engram_gib, args.offload_gb, args.tp)
    check_numa()
    check_model(args.model)
    check_stack()
    lint_flags(args)
    if args.v41:
        check_v41_patches()
        lint_v41_flags(args)

    width = max(len(n) for _, n, _ in _RESULTS)
    print()
    for level, name, detail in _RESULTS:
        mark = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[level]
        print(f"[{mark}] {name.ljust(width)}  {detail}")
    fails = sum(1 for lvl, _, _ in _RESULTS if lvl == FAIL)
    warns = sum(1 for lvl, _, _ in _RESULTS if lvl == WARN)
    print(f"\n{len(_RESULTS)} checks: {fails} blocking, {warns} warnings")
    if fails:
        print("Fix the blocking items before launching. `python tools/faultscan.py --list` "
              "explains the codes.")
    return 1 if fails else (2 if warns else 0)


if __name__ == "__main__":
    sys.exit(main())
