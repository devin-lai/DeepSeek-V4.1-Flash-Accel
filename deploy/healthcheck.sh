#!/usr/bin/env bash
# Liveness + correctness + latency probe. Suitable for a systemd watchdog, a
# load-balancer health endpoint, or cron.
#
#   deploy/healthcheck.sh                      # localhost:8000
#   ENDPOINT=http://host:8000 MODEL=dsv4 deploy/healthcheck.sh
#   MAX_LATENCY_MS=20000 deploy/healthcheck.sh # fail if the probe is slow
#
# Exit: 0 healthy, 1 unreachable, 2 wrong answer, 3 too slow.
set -uo pipefail
ENDPOINT="${ENDPOINT:-http://127.0.0.1:8000}"
TIMEOUT="${TIMEOUT:-60}"
MAX_LATENCY_MS="${MAX_LATENCY_MS:-0}"
# Generous by default: these are reasoning models, and the answer arrives
# after a chain of thought rather than as the first token.
MAX_TOKENS="${MAX_TOKENS:-256}"

if ! curl -sf -m 5 "$ENDPOINT/health" >/dev/null; then
  echo "UNHEALTHY: $ENDPOINT/health did not answer"; exit 1
fi

# Parse it rather than grepping: /v1/models nests a permission object that also
# has an "id", and a greedy regex picks that one.
MODEL="${MODEL:-$(curl -s -m 5 "$ENDPOINT/v1/models" | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin)["data"][0]["id"])
except Exception:
    pass')}"
[ -n "$MODEL" ] || { echo "UNHEALTHY: no model advertised"; exit 1; }

# A prompt with exactly one defensible answer, so a degraded server (wrong
# expert routing, a corrupt shard, a stale KV cache) shows up as a wrong
# answer rather than as fluent noise.
START=$(python3 -c 'import time;print(int(time.time()*1000))')
BODY=$(curl -s -m "$TIMEOUT" "$ENDPOINT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 6 multiplied by 7? End your reply with just the number.\"}],\"max_tokens\":$MAX_TOKENS,\"temperature\":0}")
END=$(python3 -c 'import time;print(int(time.time()*1000))')
ELAPSED=$((END-START))

# Take the visible content, falling back to reasoning_content: a reasoning model
# that ran out of budget mid-thought has produced something, and "it answered
# the wrong thing" and "it never finished" are different failures.
ANSWER=$(printf '%s' "$BODY" | python3 -c 'import json,sys
try:
    m = json.load(sys.stdin)["choices"][0]["message"]
    print((m.get("content") or m.get("reasoning_content") or "").strip()[-400:])
except Exception:
    print("")' 2>/dev/null)

if [ -z "$ANSWER" ]; then
  echo "UNHEALTHY: no completion returned in ${ELAPSED}ms"
  printf '%s\n' "$BODY" | head -c 400; echo; exit 2
fi
case "$ANSWER" in
  *42*) : ;;
  *) echo "DEGRADED: model=$MODEL did not answer 42 in $MAX_TOKENS tokens (${ELAPSED}ms)"
     echo "  tail of reply: ...${ANSWER}"
     echo "  (raise MAX_TOKENS if this model reasons at length before answering)"
     exit 2 ;;
esac
if [ "$MAX_LATENCY_MS" -gt 0 ] && [ "$ELAPSED" -gt "$MAX_LATENCY_MS" ]; then
  echo "SLOW: ${ELAPSED}ms > ${MAX_LATENCY_MS}ms threshold (answer was correct)"; exit 3
fi
echo "HEALTHY: model=$MODEL answered correctly in ${ELAPSED}ms"
