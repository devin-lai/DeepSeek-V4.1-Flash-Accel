#!/usr/bin/env bash
# One command from a bare venv to a verified DeepSeek-V4.1-Flash stack.
#
#   VENV=/data/venvs/vllm-dsv41 MODEL=/data/models/DeepSeek-V4.1-Flash \
#     bash scripts/env/setup.sh
#
# Steps, each skippable and each idempotent:
#   1. install vLLM + FlashInfer into $VENV        (scripts/env/install_vllm.sh)
#   2. install the offload plugin                  (vllm_dsv41_opt)
#   3. apply the upstream patches                  (upstream/*/apply_patch.py)
#   4. move the flashinfer-jit-cache shadow aside  (FI-002)
#   5. run preflight                               (deploy/preflight.py)
#
# Nothing here touches the model download; see scripts/download/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV="${VENV:-/data/venvs/vllm-dsv41}"
MODEL="${MODEL:-/data/models/DeepSeek-V4.1-Flash}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
PY="$VENV/bin/python"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

if [ "$SKIP_INSTALL" != "1" ]; then
  say "1/5  installing vLLM into $VENV"
  VENV="$VENV" bash "$REPO/scripts/env/install_vllm.sh"
else
  say "1/5  skipping install (SKIP_INSTALL=1)"
fi
[ -x "$PY" ] || { echo "no python at $PY" >&2; exit 1; }

say "2/5  installing the offload plugin"
"$VENV/bin/pip" install -q "$REPO/vllm_dsv41_opt"

pkgdir() { "$PY" -c "import $1, os; print(os.path.dirname($1.__file__))" 2>/dev/null || true; }

say "3/5  applying upstream patches"
FI="$(pkgdir flashinfer)"
VL="$(pkgdir vllm)"
[ -n "$FI" ] && "$PY" "$REPO/upstream/flashinfer/apply_patch.py" "$FI" || echo "flashinfer not importable, skipped"
# The sm_120 sparse-MLA module is JIT-built from the patched csrc; drop any
# cached build so the next launch compiles the edited kernels (FI-001/003/004).
rm -rf "${FLASHINFER_WORKSPACE_BASE:-$HOME}"/.cache/flashinfer/*/*/cached_ops/sparse_mla_sm120* 2>/dev/null || true
[ -n "$VL" ] && "$PY" "$REPO/upstream/vllm/apply_patch.py" "$VL" || { echo "vllm not importable" >&2; exit 1; }

say "4/5  clearing the flashinfer-jit-cache shadow (FI-002)"
JIT="$(pkgdir flashinfer_jit_cache)"
if [ -n "$JIT" ] && [ -d "$JIT/jit_cache/sparse_mla_sm120" ]; then
  mv "$JIT/jit_cache/sparse_mla_sm120" "/var/tmp/sparse_mla_sm120.$(date +%s)"
  echo "moved aside; the patched csrc will now actually be compiled"
else
  echo "no shadow present"
fi

say "5/5  preflight"
"$PY" "$REPO/deploy/preflight.py" --model "$MODEL" --gpus 8 --tp 8 \
  --expert-parallel --offload-gb 12 --engram-gib 264 --block-size 64 \
  --v41 --text-only --enforce-eager || rc=$?
rc=${rc:-0}

cat <<MSG

Next:
  MODEL=$MODEL PRESET=v41-flash $REPO/deploy/serve.sh
  $REPO/deploy/healthcheck.sh && $PY $REPO/deploy/verify.py

MSG
# preflight exits 2 for warnings only; that is not a setup failure.
[ "$rc" -eq 1 ] && exit 1
exit 0
