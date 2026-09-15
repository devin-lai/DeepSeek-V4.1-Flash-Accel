#!/usr/bin/env python3
"""Small scored smoke suite and repeated teacher-forced logprob probes.

This is a regression screen, not a general model-quality benchmark. Each
request uses a fresh cache salt. Retain raw responses to compare configurations
against the reference's own run-to-run variability.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import urllib.request
import uuid
from pathlib import Path

TASKS = [
    ("What is 37 * 19? Reply with the number only.", "703"),
    ("What is 1024 / 16? Reply with the number only.", "64"),
    ("What is 2**10 + 3**4? Reply with the integer only.", "1105"),
    ("What is 1001 modulo 13? Reply with the integer only.", "0"),
    (
        "A train travels 180 km in 2.5 hours. Give its average speed in km/h as a number only.",
        "72",
    ),
    (
        "A price of 80 increases by 25%, then decreases by 20%. Reply with the final price only.",
        "80",
    ),
    (
        "Sort 17, -3, 8, 0, 8 in ascending order. Output a JSON array only.",
        "[-3,0,8,8,17]",
    ),
    (
        "Give the unique values from [4,2,4,1,2] in order of first occurrence. JSON array only.",
        "[4,2,1]",
    ),
    ("Reverse the string 'algorithm'. Output the string only.", "mhtirogla"),
    ("What does Python len(set('abracadabra')) return? Integer only.", "5"),
    ("What does Python list(range(2, 11, 3)) return? JSON array only.", "[2,5,8]"),
    ("What does Python sum(x*x for x in range(5)) return? Integer only.", "30"),
    (
        "What is the time complexity of binary search in a sorted array? Reply as O(...), nothing else.",
        "o(logn)",
    ),
    (
        "All daxes are blue. No blue thing is round. Can a dax be round? Answer yes or no only.",
        "no",
    ),
    (
        "Alice is taller than Bob. Bob is taller than Cara. Who is shortest? Name only.",
        "cara",
    ),
    ("Convert binary 101101 to decimal. Integer only.", "45"),
    ("Convert hexadecimal 2F to decimal. Integer only.", "47"),
    ("How many minutes are there in 3 hours and 17 minutes? Integer only.", "197"),
    ("一盒有12支笔，买7盒送3支，一共有多少支？只回答数字。", "87"),
    ("把数字9、2、6按从小到大排列，只输出JSON数组。", "[2,6,9]"),
    (
        "Extract the code from: 'Order code: ZX-731. Quantity: 19.' Output only the code.",
        "zx-731",
    ),
    ("If today is Tuesday, what day is it 10 days later? Day name only.", "friday"),
    ("A rectangle has perimeter 30 and width 5. What is its area? Number only.", "50"),
    ("What is the next number: 3, 6, 12, 24? Number only.", "48"),
]

PASSAGES = [
    (
        "A compiler translates a source program into another representation. "
        "The parser builds a syntax tree, and later passes resolve names and check "
        "types. An optimizer must preserve observable program behavior. A faster "
        "implementation is useful only when its results remain correct. Memory "
        "allocation and data movement can cost more than arithmetic. Measurements "
        "therefore need realistic inputs, warmup, repeated trials, and a clear "
        "description of the machine and software used. "
    ),
    (
        "The town library opens at nine in the morning. Visitors return books at "
        "the front desk and collect reserved titles from a shelf near the door. "
        "On Wednesday afternoons, volunteers help children with mathematics. The "
        "reading room is quiet, but the workshop next door encourages discussion. "
        "A notebook on the desk records attendance without storing private details. "
    ),
]


def post(base, endpoint, body):
    request = urllib.request.Request(
        base + endpoint,
        data=json.dumps({**body, "cache_salt": str(uuid.uuid4())}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def normalize(text):
    return re.sub(r"\s+", "", text.strip().strip("`").lower()).rstrip(".")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("use a fresh output path")
    result = {"tasks": [], "logprobs": []}

    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    for prompt, expected in TASKS:
        response = post(
            args.base,
            "/v1/chat/completions",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 64,
                "chat_template_kwargs": {"thinking": False},
            },
        )
        actual = response["choices"][0]["message"]["content"]
        result["tasks"].append(
            {
                "prompt": prompt,
                "expected": expected,
                "actual": actual,
                "correct": normalize(actual) == expected,
                "response": response,
            }
        )
        save()
    for index, passage in enumerate(PASSAGES):
        # Exercise prefill staging with over 256 prompt tokens.
        prompt = passage * 6
        for repeat in range(3):
            response = post(
                args.base,
                "/v1/completions",
                {
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": 1,
                    "temperature": 0,
                    "prompt_logprobs": 0,
                    "echo": True,
                },
            )
            entries = response["choices"][0]["prompt_logprobs"]
            values = [
                next(iter(entry.values()))["logprob"] for entry in entries if entry
            ]
            if not values or not all(math.isfinite(x) for x in values):
                raise ValueError("missing or nonfinite prompt log probabilities")
            result["logprobs"].append(
                {
                    "passage": index,
                    "repeat": repeat,
                    "prompt": prompt,
                    "tokens_scored": len(values),
                    "mean_nll": -sum(values) / len(values),
                    "response": response,
                }
            )
            save()
    result["correct"] = sum(item["correct"] for item in result["tasks"])
    result["total"] = len(TASKS)
    save()
    print(json.dumps({k: result[k] for k in ("correct", "total")}))


if __name__ == "__main__":
    main()
