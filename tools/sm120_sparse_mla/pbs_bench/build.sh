#!/usr/bin/env bash
# Build the MG-prefill page-size benchmark variants.
#   build.sh <flashinfer sparse_mla_sm120 include dir> <out dir> [variant ...]
# A variant is  mode<M>[_io<IO>_math<MATH>][_xrt]  e.g. mode0, mode1, mode2_io40_math224, mode3_xrt
set -euo pipefail
SRC=${1:?installed sparse_mla_sm120 dir}
OUT=${2:?out dir}
shift 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
ARCH=${ARCH:--gencode=arch=compute_120f,code=sm_120f}
mkdir -p "$OUT/include" "$OUT/bin" "$OUT/ptxas"
python3 "$HERE/make_variants.py" "$SRC" "$OUT/include" > "$OUT/patch.log"
# Pristine copy for the reference binary (stock headers, untouched).
mkdir -p "$OUT/pristine/flashinfer/attention"
rm -rf "$OUT/pristine/flashinfer/attention/sparse_mla_sm120"
cp -r "$SRC" "$OUT/pristine/flashinfer/attention/sparse_mla_sm120"
find "$OUT/pristine" -name '*.orig' -delete

build_one() {
  local v=$1 inc=$2 defs=$3
  local bin="$OUT/bin/bench_mg_$v"
  if [ -x "$bin" ] && [ -z "${FORCE:-}" ]; then echo "have $v"; return; fi
  echo "building $v ($defs)"
  "$NVCC" $ARCH -O3 -std=c++17 --expt-relaxed-constexpr -use_fast_math -static-global-template-stub=false \
      -Xptxas -v -lineinfo $defs -I"$inc" "$HERE/bench_mg.cu" -o "$bin" 2> "$OUT/ptxas/$v.log" \
      || { echo "BUILD FAILED $v"; tail -30 "$OUT/ptxas/$v.log"; return 1; }
  echo "built $v"
}

for v in "$@"; do
  if [ "$v" = "pristine" ]; then
    # Stock headers: PrefillColdParams has no geom fields, so compile the harness
    # with a shim that defines them away.  We only need mode-0 semantics.
    build_one pristine "$OUT/pristine" "-DSMLA_PBS_MODE=0 -DSMLA_PRISTINE=1" || true
    continue
  fi
  mode=$(echo "$v" | sed -E 's/^mode([0-9]).*/\1/')
  io=$(echo "$v" | sed -nE 's/.*_io([0-9]+).*/\1/p'); io=${io:-32}
  math=$(echo "$v" | sed -nE 's/.*_math([0-9]+).*/\1/p'); math=${math:-232}
  xrt=0; case "$v" in *_xrt*) xrt=1;; esac
  build_one "$v" "$OUT/include" "-DSMLA_PBS_MODE=$mode -DSMLA_IO_MAXNREG=$io -DSMLA_MATH_MAXNREG=$math -DSMLA_EXTRA_RUNTIME=$xrt" || true
done
