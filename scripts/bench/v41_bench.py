#!/usr/bin/env python3
"""Benchmark a running V4.1 endpoint at several concurrencies."""
import argparse, json, os, subprocess
VLLM = os.environ.get("VLLM", "vllm")
MODEL = os.environ.get("MODEL", "/data/models/DeepSeek-V4.1-Flash")
CASES = [("c1_1k_128", 1024, 128, 4, 1),
         ("c8_1k_128", 1024, 128, 16, 8),
         ("c32_1k_128", 1024, 128, 64, 32),
         ("prefill_c2_8k_1", 8192, 1, 4, 2)]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True)
    ap.add_argument("--port", default="8000")
    ap.add_argument("--served-model", default="deepseek-v4.1-flash")
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    rows = []
    for name, i, o, n, c in CASES:
        cmd = [VLLM,"bench","serve","--backend","vllm","--base-url",f"http://127.0.0.1:{a.port}",
               "--endpoint","/v1/completions","--model",a.served_model,"--tokenizer",MODEL,
               "--trust-remote-code","--dataset-name","random","--random-input-len",str(i),
               "--random-output-len",str(o),"--num-prompts",str(n),"--max-concurrency",str(c),
               "--ignore-eos","--percentile-metrics","ttft,tpot,itl,e2el","--save-result",
               "--result-dir",a.dir,"--result-filename",f"{name}.json"]
        with open(f"{a.dir}/bench-{name}.log","ab") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=2400).returncode
        p = f"{a.dir}/{name}.json"
        if rc == 0 and os.path.exists(p):
            d = json.load(open(p))
            rows.append({"name":name,"in":i,"out":o,"conc":c,
                         "out_tok_s":round(d.get("output_throughput",0),1),
                         "total_tok_s":round(d.get("total_token_throughput",0),1),
                         "ttft_ms":round(d.get("median_ttft_ms",0),1),
                         "tpot_ms":round(d.get("median_tpot_ms",0),2)})
        else:
            rows.append({"name":name,"error":f"rc={rc}"})
        json.dump(rows, open(f"{a.dir}/bench.json","w"), indent=1)
    print(json.dumps(rows, indent=1))
    raise SystemExit(1 if any("error" in row for row in rows) else 0)
if __name__ == "__main__":
    main()
