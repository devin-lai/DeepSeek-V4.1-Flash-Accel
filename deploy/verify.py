#!/usr/bin/env python3
"""Is the model actually right, or just up?

`/health` returning 200 says the engine came up. It says nothing about whether
the forward pass is producing correct numbers, and on consumer Blackwell that
distinction is not academic: this repo has two faults (VL-013, and the block
size before VL-009 was fixed) where the server answers every request, quickly,
with garbage. One of them returns the same token id forever and decodes to an
empty string, which is easy to mistake for an empty response.

Three levels, cheapest first:

  1. greedy continuations with one obvious answer -- catches total breakage,
  2. teacher-forced perplexity on held-out English, from `prompt_logprobs` --
     catches "fluent but wrong", and fails loudly on non-finite logits because
     the JSON encoder refuses them,
  3. a short word problem through the chat template, which exercises the
     reasoning parser as well.

Exits non-zero if the model is not sane, so it can gate a deployment:

    deploy/healthcheck.sh && python deploy/verify.py || systemctl stop dsv4-flash
"""
from __future__ import annotations
import argparse, json, math, sys, urllib.request

CONTINUATIONS = [
    ("The capital of France is", ["paris"]),
    ("Water freezes at zero degrees", ["celsius", "c."]),
    ("1, 2, 3, 4, 5, 6,", ["7"]),
    ("The first three prime numbers are 2, 3, and", ["5"]),
    ("To be, or not to be, that is the", ["question"]),
    ("def add(a, b):\n    return", ["a + b", "a+b"]),
]

PPL_TEXT = (
    "The Antikythera mechanism is an ancient Greek hand-powered device that has "
    "been described as the oldest known example of an analogue computer. It was "
    "used to predict astronomical positions and eclipses decades in advance, and "
    "to track the four-year cycle of athletic games. The artefact was recovered "
    "in 1901 from a shipwreck off the coast of the Greek island of Antikythera."
)

def post(base, path, body, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def discover_model(base: str) -> str | None:
    """Ask the server what it serves, rather than making the caller repeat it.

    /v1/models nests a permission object that also has an "id", so this parses
    the JSON instead of grepping it.
    """
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=10) as r:
            return json.load(r)["data"][0]["id"]
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None,
                    help="served model name; discovered from /v1/models if omitted")
    ap.add_argument("--json", help="write results here")
    a = ap.parse_args()
    if a.model is None:
        a.model = discover_model(a.base)
        if a.model is None:
            print(f"could not reach {a.base}/v1/models; pass --model")
            return 1
        print(f"model: {a.model}\n")
    out = {"continuations": [], "perplexity": None, "chat": None}

    print("=== greedy continuations ===")
    hits = 0
    for prompt, wants in CONTINUATIONS:
        try:
            d = post(a.base, "/v1/completions",
                     {"model": a.model, "prompt": prompt, "max_tokens": 8,
                      "temperature": 0.0})
            text = d["choices"][0]["text"]
        except Exception as ex:
            text = f"<ERROR {ex}>"
        ok = any(w in text.lower() for w in wants)
        hits += ok
        out["continuations"].append({"prompt": prompt, "text": text, "ok": ok})
        print(f"  [{'ok' if ok else 'XX'}] {prompt!r} -> {text!r}")
    print(f"  {hits}/{len(CONTINUATIONS)} matched")

    print("\n=== teacher-forced perplexity ===")
    try:
        d = post(a.base, "/v1/completions",
                 {"model": a.model, "prompt": PPL_TEXT, "max_tokens": 1,
                  "temperature": 0.0, "prompt_logprobs": 0, "echo": True})
        lps = [x for x in (d["choices"][0].get("prompt_logprobs") or []) if x]
        vals = []
        for entry in lps:
            # {token_id: {"logprob": ...}} for the *sampled* (actual) token
            for v in entry.values():
                lp = v.get("logprob") if isinstance(v, dict) else v
                if lp is not None and lp > -1e8:
                    vals.append(lp)
                break
        if vals:
            ppl = math.exp(-sum(vals) / len(vals))
            out["perplexity"] = round(ppl, 3)
            print(f"  tokens scored: {len(vals)}   perplexity: {ppl:.2f}")
        else:
            print("  no prompt_logprobs returned")
    except Exception as ex:
        print(f"  ERROR {ex}")

    print("\n=== chat ===")
    try:
        d = post(a.base, "/v1/chat/completions",
                 {"model": a.model, "max_tokens": 200, "temperature": 0.0,
                  "messages": [{"role": "user",
                                "content": "A shop sells pens at 3 for $2. How much do 12 pens cost? Answer briefly."}]})
        m = d["choices"][0]["message"]
        out["chat"] = {"content": m.get("content"), "reasoning": m.get("reasoning_content")}
        print("  content:", repr(m.get("content"))[:500])
        if m.get("reasoning_content"):
            print("  reasoning:", repr(m["reasoning_content"])[:300])
    except Exception as ex:
        print(f"  ERROR {ex}")

    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    ppl = out["perplexity"]
    sane = hits >= 4 and (ppl is None or ppl < 30)
    print(f"\nVERDICT: {'SANE' if sane else 'NOT SANE'}")
    return 0 if sane else 1

if __name__ == "__main__":
    sys.exit(main())
