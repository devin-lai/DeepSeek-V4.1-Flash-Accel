#!/usr/bin/env bash
# Superseded by deploy/serve.sh.
#
# This script used to launch V4.1 with --numa-bind and without
# --enable-expert-parallel. Both were later measured to be wrong on this box:
# --numa-bind OOM-kills a worker once weights are host-resident and buys no
# bandwidth (VL-004), and expert parallelism is the single largest throughput
# win there is (VL-006). Rather than leave a launcher that encodes the old
# understanding, it forwards to the one that encodes the measurements.
#
#   deploy/serve.sh --preset v41-flash
#
# See deploy/README.md for the runbook and deploy/presets/ for the rest.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$(dirname "$HERE")")"

echo "scripts/serve/serve_tp8.sh is superseded; running deploy/serve.sh --preset v41-flash" >&2
exec "$REPO/deploy/serve.sh" --preset v41-flash "$@"
