#!/usr/bin/env python3
"""Run a reproducible GSM8K subset as a paired serving regression check.

Download the original test.jsonl from openai/grade-school-math separately.
This is a zero-shot subset with thinking disabled, not the standard full-set
leaderboard protocol. The output records source hash and selected row IDs;
ground-truth solutions are never included in requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

SOURCE = "https://github.com/openai/grade-school-math"
SOURCE_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
INSTRUCTION = "\nSolve the problem briefly. End your answer with '#### <number>'."
ANSWER = re.compile(r"####\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)")


def extract(text):
    matches = ANSWER.findall(text)
    return str(Decimal(matches[-1].replace(",", ""))) if matches else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("use a fresh output path")
    data = args.dataset.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SOURCE_SHA256:
        parser.error("dataset differs from the recorded original GSM8K test file")
    questions = [json.loads(line) for line in data.splitlines()]
    if not 1 <= args.count <= len(questions) or args.concurrency < 1:
        parser.error("invalid count or concurrency")
    indices = sorted(random.Random(args.seed).sample(range(len(questions)), args.count))
    result = {
        "source": SOURCE,
        "source_sha256": digest,
        "protocol": "zero-shot, thinking disabled; paired regression subset",
        "instruction": INSTRUCTION,
        "indices": indices,
        "temperature": 0,
        "max_tokens": 512,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "rows": [],
    }

    def run(index):
        item = questions[index]
        expected = extract(item["answer"])
        if expected is None:
            raise ValueError(f"unparseable ground truth at row {index}")
        start = time.monotonic()
        row = {
            "index": index,
            "expected": expected,
            "question_sha256": hashlib.sha256(item["question"].encode()).hexdigest(),
        }
        request = urllib.request.Request(
            args.base + "/v1/chat/completions",
            data=json.dumps(
                {
                    "model": args.model,
                    "messages": [
                        {"role": "user", "content": item["question"] + INSTRUCTION}
                    ],
                    "temperature": 0,
                    "max_tokens": 512,
                    "seed": args.seed,
                    "chat_template_kwargs": {"thinking": False},
                    "cache_salt": str(uuid.uuid4()),
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                payload = json.load(response)
            choice = payload["choices"][0]
            actual = choice["message"]["content"] or ""
            predicted = extract(actual)
            row.update(
                actual=actual,
                predicted=predicted,
                correct=predicted is not None
                and Decimal(predicted) == Decimal(expected),
                finish_reason=choice["finish_reason"],
                usage=payload["usage"],
            )
        except (OSError, ValueError, LookupError, TypeError, ArithmeticError) as error:
            row.update(error=str(error), correct=False)
        row["elapsed_s"] = time.monotonic() - start
        return row

    def save():
        result["rows"].sort(key=lambda row: row["index"])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run, index) for index in indices]
        for future in as_completed(futures):
            result["rows"].append(future.result())
            save()
            if len(result["rows"]) % 16 == 0:
                print(f"completed {len(result['rows'])}/{args.count}", flush=True)
    result.update(
        correct=sum(row["correct"] for row in result["rows"]),
        total=args.count,
        failed=sum("error" in row for row in result["rows"]),
        truncated=sum(row.get("finish_reason") == "length" for row in result["rows"]),
        elapsed_s=time.monotonic() - start,
    )
    save()
    print(
        json.dumps(
            {key: result[key] for key in ("correct", "total", "failed", "truncated")}
        )
    )
    return int(result["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
