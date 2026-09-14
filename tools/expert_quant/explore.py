#!/usr/bin/env python3
"""Which expert quantisation makes DeepSeek-V4.1-Flash fit on 8x 32 GB, and what does it cost?

Reads real expert tensors out of the checkpoint, requantises them with each
candidate scheme, and reports three things per scheme:

  * **bytes per rank** under EP8 -- does the offload disappear, or not;
  * **weight error** -- assumption-free, but only loosely related to quality;
  * **output error** -- ``||(W - Ŵ)x|| / ||Wx||`` on activations drawn the way
    the model actually produces them.

The activation model is the part worth arguing with, so it is explicit. The
input to `w1`/`w3` is the output of `ffn_norm`, which is unit-RMS per token by
construction, so it is modelled as unit-RMS noise scaled by the *real*
`ffn_norm.weight` read from the checkpoint. The input to `w2` is then not
modelled at all: it is computed, `silu(w1 x) * (w3 x)`, from those same
activations and the real weights.

One caveat that changes how every number here should be read: the shipped
checkpoint is **already MXFP4**, so there is no FP32 original to compare
against. Every error below is a *requantisation* error measured from the
shipped grid -- the right question for a deployment ("what do I give up by
shrinking what DeepSeek published?") and the wrong one for judging the schemes
as quantisers in the abstract.

    python tools/expert_quant/explore.py /data/models/DeepSeek-V4.1-Flash
    python tools/expert_quant/explore.py MODEL --layers 6 20 39 --experts 4 --markdown
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schemes import SCHEMES, gptq  # noqa: E402

GIB = 1024**3
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


# --------------------------------------------------------------- checkpoint


class Checkpoint:
    """Minimal safetensors reader: headers only until a tensor is asked for."""

    def __init__(self, model_dir: str):
        self.dir = model_dir
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as fh:
            self.weight_map = json.load(fh)["weight_map"]
        self._headers: dict[str, tuple[dict, int]] = {}

    def _header(self, shard: str):
        if shard not in self._headers:
            with open(os.path.join(self.dir, shard), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self._headers[shard] = (json.loads(fh.read(n)), 8 + n)
        return self._headers[shard]

    def get(self, name: str, device="cpu") -> torch.Tensor:
        shard = self.weight_map[name]
        hdr, base = self._header(shard)
        meta = hdr[name]
        start, end = meta["data_offsets"]
        with open(os.path.join(self.dir, shard), "rb") as fh:
            fh.seek(base + start)
            raw = fh.read(end - start)
        dtype = {
            "I8": torch.int8, "U8": torch.uint8, "F8_E8M0": torch.uint8,
            "F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16,
            "F16": torch.float16, "F32": torch.float32,
        }[meta["dtype"]]
        t = torch.frombuffer(bytearray(raw), dtype=dtype).view(meta["shape"])
        return t.to(device)


def dequant_mxfp4(packed: torch.Tensor, scale_e8m0: torch.Tensor, group: int = 32) -> torch.Tensor:
    """[out, in/2] int8 nibbles + [out, in/group] UE8M0 -> [out, in] float32.

    Low nibble is the even-indexed weight, matching vLLM's MXFP4 loader.
    """
    b = packed.view(torch.uint8)  # reinterpret the I8 bits, do not convert values
    lo = E2M1.to(b.device)[(b & 0x0F).long()]
    hi = E2M1.to(b.device)[(b >> 4).long()]
    w = torch.stack([lo, hi], dim=-1).flatten(-2)  # [out, in]
    scale = torch.exp2(scale_e8m0.to(torch.int32).float() - 127)
    return (w.unflatten(-1, (scale.shape[-1], group)) * scale.unsqueeze(-1)).flatten(-2)


# ------------------------------------------------------------- measurement


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def measure_layer(ck: Checkpoint, layer: int, experts: list[int], tokens: int,
                  device: str, schemes: dict) -> dict[str, dict]:
    g = ck.get(f"layers.{layer}.ffn_norm.weight", device).float()
    hidden = g.numel()
    torch.manual_seed(0xC0FFEE + layer)
    # RMSNorm output: unit RMS per token, then the learned per-channel gain.
    u = torch.randn(tokens, hidden, device=device)
    x = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-30) * (hidden**0.5) * g

    acc: dict[str, dict] = {}
    for e in experts:
        ref = {}
        for proj in ("w1", "w2", "w3"):
            p = ck.get(f"layers.{layer}.ffn.experts.{e}.{proj}.weight", device)
            s = ck.get(f"layers.{layer}.ffn.experts.{e}.{proj}.scale", device)
            ref[proj] = dequant_mxfp4(p, s)

        h_ref = torch.nn.functional.silu(x @ ref["w1"].T) * (x @ ref["w3"].T)
        y_ref = h_ref @ ref["w2"].T

        # GPTQ needs the second moment of each projection's *own* input:
        # the normed hidden state for w1/w3, the SwiGLU product for w2.
        hess = {"w1": x.T @ x, "w3": x.T @ x, "w2": h_ref.T @ h_ref}

        # Control: the checkpoint is already MXFP4, so re-encoding it must be a
        # no-op.  If this is not ~0 the encoder disagrees with the one that
        # produced the checkpoint, and every other row is measured against the
        # wrong reference.
        if "mxfp4" in schemes:
            from schemes import mxfp4 as _mxfp4  # noqa: PLC0415
            drift = rel_err(_mxfp4(ref["w1"]), ref["w1"])
            if drift > 1e-6:
                print(f"    WARNING: MXFP4 round trip is not exact "
                      f"(rel={drift:.2e}); the reference is suspect")

        for name, spec in schemes.items():
            fn = spec["fn"]
            # A scheme describes either one recipe for all three projections, or
            # one per projection (mixed precision). The latter is a dict keyed
            # by projection name -- which a GPTQ keyword set is not, so check
            # the keys rather than the type.
            per_proj = isinstance(fn, dict) and "w1" in fn
            if spec.get("gptq"):
                q = {p: gptq(ref[p], hess[p], **(fn[p] if per_proj else fn))
                     for p in ("w1", "w2", "w3")}
            else:
                q = {p: (fn[p] if per_proj else fn)(ref[p])
                     for p in ("w1", "w2", "w3")}
            h_q = torch.nn.functional.silu(x @ q["w1"].T) * (x @ q["w3"].T)
            y_q = h_q @ q["w2"].T
            row = acc.setdefault(name, {"w_err": [], "y_err": [], "h_err": [],
                                        "w13_err": [], "w2_err": []})
            row["w13_err"].append((rel_err(q["w1"], ref["w1"]) + rel_err(q["w3"], ref["w3"])) / 2)
            row["w2_err"].append(rel_err(q["w2"], ref["w2"]))
            row["w_err"].append(
                sum(rel_err(q[p], ref[p]) for p in ("w1", "w2", "w3")) / 3
            )
            row["h_err"].append(rel_err(h_q, h_ref))
            row["y_err"].append(rel_err(y_q, y_ref))
    return acc


# ------------------------------------------------------------ byte budget


def budget(cfg: dict, bpw: float, ranks: int) -> float:
    """GiB of routed-expert weight per rank under EP8, at `bpw` bits/weight."""
    t = cfg.get("text_config", cfg)
    layers = t["num_hidden_layers"]
    experts = t["n_routed_experts"]
    inter = t["moe_intermediate_size"]
    hidden = t["hidden_size"]
    weights_per_expert = 3 * inter * hidden  # w1, w3: inter x hidden; w2: hidden x inter
    total_bits = layers * experts * weights_per_expert * bpw
    return total_bits / 8 / ranks / GIB


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 20, 39])
    ap.add_argument("--experts", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--ranks", type=int, default=8)
    ap.add_argument("--gpu-gib", type=float, default=31.4)
    ap.add_argument("--reserve-gib", type=float, default=11.0,
                    help="per rank for KV cache, dense weights, activations, graphs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", nargs="+", help="restrict to these scheme names")
    ap.add_argument("--json")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    ck = Checkpoint(args.model_dir)
    with open(os.path.join(args.model_dir, "config.json")) as fh:
        cfg = json.load(fh)
    schemes = {k: v for k, v in SCHEMES.items() if not args.only or k in args.only}

    print(f"model: {args.model_dir}")
    print(f"layers {args.layers}, {args.experts} experts each, "
          f"{args.tokens} tokens, device {args.device}\n")

    per_layer: dict[str, dict] = {}
    for layer in args.layers:
        t0 = time.time()
        acc = measure_layer(ck, layer, list(range(args.experts)), args.tokens,
                            args.device, schemes)
        print(f"  layer {layer}: {time.time() - t0:.1f}s")
        for name, row in acc.items():
            agg = per_layer.setdefault(
                name, {"w_err": [], "h_err": [], "y_err": [], "w13_err": [], "w2_err": []})
            for k in agg:
                agg[k].extend(row[k])

    fits = args.gpu_gib - args.reserve_gib
    rows = []
    for name, spec in schemes.items():
        m = per_layer[name]
        gib = budget(cfg, spec["bpw"], args.ranks)
        rows.append({
            "scheme": name,
            "bpw": spec["bpw"],
            "gib_per_rank": gib,
            "offload_gib": max(0.0, gib - fits),
            "w_err": sum(m["w_err"]) / len(m["w_err"]),
            "w13_err": sum(m["w13_err"]) / len(m["w13_err"]),
            "w2_err": sum(m["w2_err"]) / len(m["w2_err"]),
            "h_err": sum(m["h_err"]) / len(m["h_err"]),
            "y_err": sum(m["y_err"]) / len(m["y_err"]),
            "note": spec["note"],
        })
    rows.sort(key=lambda r: r["bpw"], reverse=True)

    print(f"\nEP8 on {args.ranks} ranks; {args.gpu_gib:.1f} GiB/card minus "
          f"{args.reserve_gib:.1f} GiB reserved leaves {fits:.1f} GiB for experts.\n")
    hdr = f"{'scheme':<16} {'bpw':>5} {'GiB/rank':>9} {'offload':>8} " \
          f"{'w err':>8} {'swiglu':>8} {'out err':>8}  note"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        off = "none" if r["offload_gib"] == 0 else f"{r['offload_gib']:.1f} GiB"
        print(f"{r['scheme']:<16} {r['bpw']:>5.2f} {r['gib_per_rank']:>9.1f} {off:>8} "
              f"{r['w_err']:>8.4f} {r['h_err']:>8.4f} {r['y_err']:>8.4f}  {r['note']}")

    if args.markdown:
        print("\n| scheme | bits/weight | GiB/rank (EP8) | must offload | weight err | SwiGLU err | expert-output err |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in rows:
            off = "**none**" if r["offload_gib"] == 0 else f"{r['offload_gib']:.1f} GiB"
            print(f"| `{r['scheme']}` | {r['bpw']:.2f} | {r['gib_per_rank']:.1f} | {off} "
                  f"| {r['w_err']:.4f} | {r['h_err']:.4f} | {r['y_err']:.4f} |")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"model": args.model_dir, "layers": args.layers,
                       "experts": args.experts, "tokens": args.tokens,
                       "ranks": args.ranks, "gpu_gib": args.gpu_gib,
                       "reserve_gib": args.reserve_gib, "rows": rows}, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
