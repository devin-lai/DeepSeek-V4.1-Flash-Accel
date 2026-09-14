#!/usr/bin/env python3
"""Repeated serving benchmarks that reject failed or incomplete requests."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import uuid

CASES = {
    "c1_1k_128": (1024, 128, 4, 1),
    "c8_1k_128": (1024, 128, 16, 8),
    "c32_1k_128": (1024, 128, 64, 32),
    "prefill_c2_8k_1": (8192, 1, 4, 2),
}
DEFAULT_CASES = tuple(CASES)
CASES.update(
    {
        "interactive_c1_256": (0, 256, 8, 1),
        "interactive_c8_256": (0, 256, 8, 8),
    }
)


def validate_result(data: dict, prompts: int, output_len: int) -> None:
    """vLLM's CLI can exit zero with failed requests; check the evidence."""
    if data.get("completed") != prompts or data.get("failed", 0) != 0:
        raise ValueError(
            f"expected {prompts} successes, got completed={data.get('completed')} "
            f"failed={data.get('failed')}"
        )
    if data.get("total_output_tokens") != prompts * output_len:
        raise ValueError(
            "output token count differs from requested ignore_eos workload"
        )
    for key in ("output_throughput", "total_token_throughput", "median_ttft_ms"):
        value = data.get(key)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"invalid {key}: {value}")
    if output_len > 1:
        value = data.get("median_tpot_ms")
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"invalid median_tpot_ms: {value}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--port", default="8000")
    ap.add_argument("--served-model", default="deepseek-v4.1-flash")
    ap.add_argument("--case", choices=CASES, action="append", dest="cases")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--request-multiplier", type=int, default=1)
    ap.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "benchmarks/workloads/interactive.jsonl",
    )
    args = ap.parse_args()
    if args.repeat < 1 or args.request_multiplier < 1:
        ap.error("repeat and request-multiplier must be positive")
    args.dir.mkdir(parents=True, exist_ok=True)
    if (args.dir / "bench.json").exists():
        ap.error("use a fresh result directory; bench.json already exists")
    rows = []
    manifest = []
    for repeat in range(args.repeat):
        for name in args.cases or DEFAULT_CASES:
            input_len, output_len, prompts, concurrency = CASES[name]
            prompts *= args.request_multiplier
            stem = f"r{repeat + 1}-{name}" if args.repeat > 1 else name
            result_path = args.dir / f"{stem}.json"
            if result_path.exists():
                raise FileExistsError(result_path)
            seed = args.seed + repeat
            cmd = [
                os.environ.get("VLLM", "vllm"),
                "bench",
                "serve",
                "--backend",
                "vllm",
                "--base-url",
                f"http://127.0.0.1:{args.port}",
                "--endpoint",
                "/v1/completions",
                "--model",
                args.served_model,
                "--tokenizer",
                os.environ.get("MODEL", "/data/models/DeepSeek-V4.1-Flash"),
                "--trust-remote-code",
                "--dataset-name",
                "random",
                "--random-input-len",
                str(input_len),
                "--random-output-len",
                str(output_len),
                "--num-prompts",
                str(prompts),
                "--max-concurrency",
                str(concurrency),
                "--ignore-eos",
                "--temperature",
                str(args.temperature),
                "--seed",
                str(seed),
                "--extra-body",
                json.dumps({"cache_salt": str(uuid.uuid4())}),
                "--percentile-metrics",
                "ttft,tpot,itl,e2el",
                "--save-result",
                "--save-detailed",
                "--result-dir",
                str(args.dir),
                "--result-filename",
                result_path.name,
            ]
            if name.startswith("interactive_"):
                cmd[cmd.index("--dataset-name") + 1] = "custom"
                cmd += [
                    "--dataset-path",
                    str(args.dataset_path),
                    "--custom-output-len",
                    str(output_len),
                    "--tokenizer-mode",
                    "deepseek_v41",
                    "--chat-template-kwargs",
                    '{"thinking":false}',
                ]
            manifest.append(
                {"name": name, "repeat": repeat + 1, "seed": seed, "command": cmd}
            )
            (args.dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            row = {
                "name": name,
                "repeat": repeat + 1,
                "seed": seed,
                "in": input_len or "variable",
                "out": output_len,
                "prompts": prompts,
                "conc": concurrency,
            }
            try:
                with (args.dir / f"bench-{stem}.log").open("w") as log:
                    subprocess.run(
                        cmd,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=2400,
                        check=True,
                    )
                data = json.loads(result_path.read_text())
                validate_result(data, prompts, output_len)
                row.update(
                    out_tok_s=data["output_throughput"],
                    total_tok_s=data["total_token_throughput"],
                    ttft_ms=data["median_ttft_ms"],
                    tpot_ms=data["median_tpot_ms"],
                    completed=data["completed"],
                    failed=data.get("failed", 0),
                )
                if "spec_decode_acceptance_length" in data:
                    row["acceptance_length"] = data["spec_decode_acceptance_length"]
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                row["error"] = str(error)
            rows.append(row)
            (args.dir / "bench.json").write_text(
                json.dumps(rows, indent=2, allow_nan=False) + "\n"
            )
            print(json.dumps(row), flush=True)
            if "error" in row:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
