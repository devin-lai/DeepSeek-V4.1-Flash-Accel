#!/usr/bin/env bash
# Download / resume DeepSeek-V4.1-Flash with aria2 from ModelScope and hf-mirror
# at the same time (each shard is split across both mirrors, 16 connections
# each). Safe to re-run: complete shards are skipped, partial ones resume.
#
#   MODEL_DIR=/data/models/DeepSeek-V4.1-Flash bash scripts/download/download_model.sh
#
# Verify afterwards with:  python scripts/download/verify_shards.py "$MODEL_DIR"
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/data/models/DeepSeek-V4.1-Flash}"
REPO="deepseek-ai/DeepSeek-V4.1-Flash"
MS="https://www.modelscope.cn/models/${REPO}/resolve/master"
HF="https://hf-mirror.com/${REPO}/resolve/main"
CONN="${CONN:-16}"

mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

fetch() {  # fetch <relative path>
  local f="$1"
  mkdir -p "$(dirname "$f")"
  until aria2c -c -x"$CONN" -s"$CONN" -k4M --file-allocation=none \
        --max-tries=0 --retry-wait=10 --timeout=60 \
        --console-log-level=warn --summary-interval=60 \
        -d "$MODEL_DIR" -o "$f" "$MS/$f" "$HF/$f"; do
    echo "[$(date)] retry $f"; sleep 15
  done
}

# small files first (config, tokenizer, index, reference code)
for f in config.json configuration.json tokenizer.json tokenizer_config.json \
         model.safetensors.index.json README.md LICENSE DeepSeek_V41_Tech_Report.pdf; do
  [ -s "$f" ] || fetch "$f"
done

# ModelScope's own partial files are contiguous prefixes: let aria2 continue them
for f in *.incomplete; do [ -e "$f" ] && mv -n "$f" "${f%.incomplete}"; done

# shards, in index order, skipping the ones that already verify
python3 - "$MODEL_DIR" <<'PY' > /tmp/dsv41_missing.txt
import json, os, struct, sys
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
for f in sorted(set(idx["weight_map"].values())):
    p = os.path.join(d, f); ok = False
    if os.path.exists(p) and not os.path.exists(p + ".aria2"):
        try:
            with open(p, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
            end = max(v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__")
            ok = (8 + n + end) == os.path.getsize(p)
        except Exception:
            ok = False
    if not ok: print(f)
PY
echo "shards to fetch: $(wc -l < /tmp/dsv41_missing.txt)"
while read -r f; do
  echo "[$(date)] $f"; fetch "$f"
done < /tmp/dsv41_missing.txt
echo "[$(date)] all shards present; run verify_shards.py"
