#!/usr/bin/env bash
# Build a vLLM environment that can serve DeepSeek-V4.1-Flash on RTX 5090 (sm_120).
#
# DeepSeek-V4.1 support landed on vLLM main on 2026-09-10 and is NOT in the
# 0.29.0 release, so we install the per-commit wheel that vLLM CI publishes
# for every main commit (CUDA 13.0 build, torch 2.13.0). No compilation.
#
#   VENV=/data/venvs/vllm-dsv41 bash scripts/env/install_vllm.sh
#
# Mirrors: PyPI via Tsinghua TUNA (fast in mainland China), vLLM wheels from
# wheels.vllm.ai, FlashInfer from flashinfer.ai. pypi.org itself does not need
# to be reachable.
set -euo pipefail

VENV="${VENV:-/data/venvs/vllm-dsv41}"
VLLM_COMMIT="${VLLM_COMMIT:-8c1d1c2974ee42757ee2e93cc898932edfd9d265}"   # main @ 2026-09-11
VLLM_VERSION="${VLLM_VERSION:-0.1.1.dev7+g8c1d1c297}"                        # version string of that wheel
FLASHINFER_VERSION="${FLASHINFER_VERSION:-0.6.18.post1}"                     # pinned by vllm requirements/cuda.txt
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PYPI="${PYPI:-https://pypi.tuna.tsinghua.edu.cn/simple}"

export UV_DEFAULT_INDEX="$PYPI"
export UV_INDEX_STRATEGY=unsafe-best-match
export UV_HTTP_TIMEOUT=600
export UV_CACHE_DIR="${UV_CACHE_DIR:-$(dirname "$VENV")/../cache/uv}"

if ! command -v uv >/dev/null; then
  python3 -m pip install -i "$PYPI" uv
fi

uv venv "$VENV" --python "$PYTHON_VERSION"
uv pip install --python "$VENV/bin/python" \
  --index "https://wheels.vllm.ai/${VLLM_COMMIT}/" \
  --index https://flashinfer.ai/whl/cu130/ \
  "vllm==${VLLM_VERSION}"

# prebuilt FlashInfer kernels (cubin + JIT cache) so first start does not spend
# an hour in nvcc; modelscope for downloads
uv pip install --python "$VENV/bin/python" \
  --index https://flashinfer.ai/whl/cu130/ \
  "flashinfer-cubin==${FLASHINFER_VERSION}" "flashinfer-jit-cache==${FLASHINFER_VERSION}" \
  modelscope hf_transfer pip

"$VENV/bin/python" - <<'PY'
import torch, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda, "vllm", vllm.__version__)
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
assert "sm_120" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
PY
echo "OK: activate with  source $VENV/bin/activate"
