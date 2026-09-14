#!/usr/bin/env bash
# Reference launcher for DeepSeek-V4-Flash / V4.1-Flash on 8x RTX 5090.
#
#   deploy/serve.sh                         # uses deploy/presets/default.env
#   PRESET=throughput deploy/serve.sh
#   deploy/serve.sh --preset long-context
#
# Every default here is a measured one; `docs/05-fault-inventory.md` says why.
# The launcher refuses to start if deploy/preflight.py finds a blocking problem
# (override with SKIP_PREFLIGHT=1).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"

case "${1:-}" in
  --preset) PRESET="${2:?--preset needs a name}"; shift 2 ;;
esac
PRESET="${PRESET:-default}"
PRESET_FILE="$HERE/presets/$PRESET.env"
[ -f "$PRESET_FILE" ] || { echo "no such preset: $PRESET (have: $(ls "$HERE/presets" | sed 's/\.env//' | tr '\n' ' '))" >&2; exit 64; }
# shellcheck disable=SC1090
source "$PRESET_FILE"

MODEL="${MODEL:?set MODEL, or put it in the preset}"
VENV="${VENV:-/data/venvs/vllm-dsv41}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
SERVED_NAME="${SERVED_NAME:-$(basename "$MODEL")}"
TP="${TP:-8}"
EXPERT_PARALLEL="${EXPERT_PARALLEL:-1}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_LEN="${MAX_LEN:-65536}"
MAX_SEQS="${MAX_SEQS:-32}"
OFFLOAD_GB="${OFFLOAD_GB:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
TEXT_ONLY="${TEXT_ONLY:-0}"
ENGRAM_OFFLOAD="${ENGRAM_OFFLOAD:-0}"
BLOCK_SIZE="${BLOCK_SIZE:-}"
TOKENIZER_MODE="${TOKENIZER_MODE:-auto}"
LOG_DIR="${LOG_DIR:-/data/logs}"
mkdir -p "$LOG_DIR"

# --- environment -----------------------------------------------------------
# vLLM gates every FlashInfer backend on `shutil.which("nvcc")` unless the
# flashinfer-cubin package is installed, and reports its absence as a missing
# kernel specialisation. Put the toolkit on PATH before anything else.
CUDA_HOME="${CUDA_HOME:-$(ls -d /usr/local/cuda* 2>/dev/null | tail -1)}"
[ -n "$CUDA_HOME" ] && export PATH="$CUDA_HOME/bin:$PATH"

export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/data/cache/flashinfer}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/data/cache/inductor}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/data/cache/vllm}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
[ -n "${OFFLOAD_LAYERS:-}" ] && export DSV41_OFFLOAD_LAYERS="$OFFLOAD_LAYERS"

# shellcheck disable=SC1091
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

# --- preflight -------------------------------------------------------------
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  PF=( --model "$MODEL" --gpus "$TP" --tp "$TP" --offload-gb "$OFFLOAD_GB" )
  [ "$EXPERT_PARALLEL" = "1" ] && PF+=( --expert-parallel )
  [ -n "$BLOCK_SIZE" ] && PF+=( --block-size "$BLOCK_SIZE" )
  [ "$ENGRAM_OFFLOAD" = "1" ] && PF+=( --engram-gib "${ENGRAM_GIB:-264}" )
  [ "$TEXT_ONLY" = "1" ] && PF+=( --text-only )
  [ "$ENFORCE_EAGER" = "1" ] && PF+=( --enforce-eager )
  # V4.1 needs patches and flags nothing else will warn about; the tokenizer
  # mode is the cheapest reliable signal that this is that model.
  [ "$TOKENIZER_MODE" = "deepseek_v41" ] && PF+=( --v41 )
  # Exit 1 is a blocking problem and stops the launch; exit 2 is a warning about
  # a flag choice and must not, so the status is captured rather than inherited.
  rc=0
  "${PYTHON:-python3}" "$REPO/deploy/preflight.py" "${PF[@]}" || rc=$?
  if [ "$rc" -eq 1 ]; then
    echo "preflight found a blocking problem; SKIP_PREFLIGHT=1 to override" >&2
    exit 1
  fi
fi

# --- launch ----------------------------------------------------------------
ARGS=(
  "$MODEL"
  --served-model-name "$SERVED_NAME"
  --host "$HOST" --port "$PORT"
  --trust-remote-code
  --tensor-parallel-size "$TP"
  --gpu-memory-utilization "$GPU_UTIL"
  --max-model-len "$MAX_LEN"
  --max-num-seqs "$MAX_SEQS"
  --enable-chunked-prefill
  # VL-003: autotune JIT-compiles inside a collective; ranks finish minutes
  # apart, a peer drops, and the launch dies ~40 min in.
  --kernel-config "{\"enable_flashinfer_autotune\": false${MOE_BACKEND:+, \"moe_backend\": \"$MOE_BACKEND\"}}"
)
[ "$TOKENIZER_MODE" != "auto" ] && ARGS+=( --tokenizer-mode "$TOKENIZER_MODE" )
# VL-006: whole experts are never split, so MXFP4 pads nothing. Largest single
# win measured here: +79 % single stream, +94 % at 8 streams.
[ "$EXPERT_PARALLEL" = "1" ] && ARGS+=( --enable-expert-parallel )
# VL-005: a backend constraint, not a tuning knob. Leave unset unless required.
[ -n "$BLOCK_SIZE" ] && ARGS+=( --block-size "$BLOCK_SIZE" )
[ "$ENGRAM_OFFLOAD" = "1" ] && ARGS+=( --engram-config '{"cpu_offload": true}' )
# VL-007: every offloaded GiB is paid on every decode step. Offload the least
# that makes the model fit; `tools/plan_memory.py` reports that number.
if [ "${OFFLOAD_GB%.*}" != "0" ]; then
  ARGS+=( --cpu-offload-gb "$OFFLOAD_GB" --cpu-offload-params w13_weight w2_weight )
fi
# NOTE: no --numa-bind. VL-004: it OOM-kills a worker once weights are
# host-resident, and UVA reads are NUMA-insensitive here (51.3 vs 51.1 GB/s).
[ -n "${REASONING_PARSER:-}" ] && ARGS+=( --reasoning-parser "$REASONING_PARSER" )
[ -n "${TOOL_PARSER:-}" ] && ARGS+=( --tool-call-parser "$TOOL_PARSER" --enable-auto-tool-choice )
[ "${TEXT_ONLY:-0}" = "1" ] && ARGS+=( --language-model-only )
# VL-013 / FI-004: the patched V4.1 presets enable CUDA graphs.
# Eager remains available for diagnosis; dispatch and cache-layout fixes still apply.
[ "${ENFORCE_EAGER:-0}" = "1" ] && ARGS+=( --enforce-eager )
# shellcheck disable=SC2206
ARGS+=( ${EXTRA:-} )

STAMP="$(date +%Y%m%d-%H%M%S)"
echo "preset=$PRESET model=$MODEL tp=$TP ep=$EXPERT_PARALLEL offload=${OFFLOAD_GB}GiB"
echo "vllm serve ${ARGS[*]}"
exec vllm serve "${ARGS[@]}" 2>&1 | tee "$LOG_DIR/serve-$SERVED_NAME-$STAMP.log"
