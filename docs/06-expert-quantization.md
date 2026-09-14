# Expert quantisation: what actually makes V4.1 fit, and what it costs

DeepSeek-V4.1-Flash ships its 384 routed experts per layer in MXFP4. Under EP8
on 8x RTX 5090 that is **33.6 GiB per rank** against roughly 20.4 GiB of room
once the KV cache, dense weights, activations and CUDA graphs are paid for — so
about 13 GiB per rank has to live in host RAM and be read over PCIe on every
decode step, which costs roughly 70 % of decode throughput.

The obvious escape is to store the experts in fewer bits. This page measures
what that is worth, using the real checkpoint.

## Method, and the control that makes it trustworthy

`tools/expert_quant/explore.py` reads real expert tensors, requantises them with
each candidate scheme, and reports three numbers: bytes per rank (does the
offload disappear), weight error, and **expert-output error** —
`||(W - Ŵ)x|| / ||Wx||` on activations drawn the way the model produces them.

The activation model is the assumption worth arguing with, so it is explicit.
The input to `w1`/`w3` is the output of `ffn_norm`, which is unit-RMS per token
by construction, so it is modelled as unit-RMS noise scaled by the **real**
`ffn_norm.weight` from the checkpoint. The input to `w2` is not modelled at all:
it is computed, `silu(w1 x) * (w3 x)`, from those activations and the real
weights.

The control is the MXFP4 row. The checkpoint is already MXFP4, so re-encoding it
must be a no-op, and it is — `0.0000` on all three metrics. That is what says
the encoder here matches the one that produced the checkpoint, and it is
load-bearing: an earlier version of this study used `floor(log2(amax/6))` for
the shared scale instead of the OCP rule `floor(log2(amax)) - 2`, clipped every
block maximum, and reported 0.0964 weight error for a transformation that should
have been exact. Every other row would have been measured against a corrupted
reference.

**Read every error below as a *requantisation* error from the shipped 4-bit
grid, not from the original weights.** DeepSeek did not publish BF16 weights, so
a 3-bit build made this way carries MXFP4's error *and* its own. A 3-bit
quantisation done from the original weights would be materially better than the
3-bit row here. That is not a caveat about precision of measurement; it is the
main practical finding, and it means the best available path to a smaller V4.1
runs through DeepSeek publishing higher-precision weights, not through anything
a deployer can do downstream.

## Results

40 layers, 384 experts, `moe_intermediate_size` 2304, `hidden_size` 5120; EP8
across 8 ranks; 31.4 GiB per card less 11.0 GiB reserved leaves 20.4 GiB for
experts. Sampled at layers 6, 20 and 39, three experts each, 512 tokens.

| scheme | bits/weight | GiB/rank | must offload | weight err | SwiGLU err | expert-output err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nvfp4` | 4.50 | 35.6 | 15.2 GiB | 0.0634 | 0.1087 | 0.1470 |
| **`mxfp4` (shipped)** | 4.25 | 33.6 | 13.2 GiB | **0.0000** | **0.0000** | **0.0000** |
| `int4-g128` | 4.25 | 33.6 | 13.2 GiB | 0.1161 | 0.1581 | 0.1951 |
| `vq4-g128` | 4.12 | 32.6 | 12.2 GiB | 0.0925 | 0.1255 | 0.1548 |
| `int3-g64` | 3.50 | 27.7 | 7.3 GiB | 0.1969 | 0.2673 | 0.3312 |
| `vq3/4-g128` | 3.46 | 27.4 | 7.0 GiB | 0.1529 | 0.2471 | 0.2620 |
| `int3-g128` | 3.25 | 25.7 | 5.3 GiB | 0.2146 | 0.2891 | 0.3553 |
| **`gptq-int3-g128`** | 3.25 | 25.7 | 5.3 GiB | 0.2860 | 0.0723 | **0.1080** |
| `vq3-g128` | 3.12 | 24.7 | 4.3 GiB | 0.1831 | 0.2471 | 0.3026 |
| `vq3-g128-norot` | 3.12 | 24.7 | 4.3 GiB | 0.2311 | 0.3108 | 0.3811 |
| **`gptq-vq3-g128`** | 3.12 | 24.7 | 4.3 GiB | 0.2425 | 0.0602 | **0.0873** |
| `int2-g64` | 2.50 | 19.8 | **none** | 0.4569 | 0.6863 | 0.9234 |
| `vq2/3-g128` | 2.46 | 19.4 | **none** | 0.2921 | 0.4604 | 0.4867 |
| **`gptq-vq2/3-g128`** | 2.46 | 19.4 | **none** | 0.3831 | 0.1350 | **0.1490** |
| `vq2-g128` | 2.12 | 16.8 | **none** | 0.3466 | 0.4604 | 0.5514 |
| `vq2-g128-norot` | 2.12 | 16.8 | **none** | 0.3574 | 0.4741 | 0.5691 |
| `gptq-vq2-g128` | 2.12 | 16.8 | **none** | 0.4636 | 0.1350 | 0.1958 |

`vq*` is the EXL3/QuIP# family without the trellis: a random-sign Hadamard
rotation along the input dimension, then a Lloyd-Max codebook fitted to the
rotated distribution, scaled per group of 128. The `-norot` rows are the same
codebook with the rotation removed, so the pair isolates what incoherence
processing is worth on its own. The `gptq-*` rows add error feedback: each
column's rounding error is pushed into the columns not yet reached, weighted by
the inverse Hessian of that projection's own activations.

### Five things the table says

**Error feedback is the largest lever here — larger than the format, larger
than the rotation, larger than a whole extra bit.** It cuts expert-output error
by 2.8x to 3.5x at every budget:

| | without | with GPTQ | |
| --- | ---: | ---: | ---: |
| `int3-g128` | 0.3553 | 0.1080 | 3.3x |
| `vq3-g128` | 0.3026 | 0.0873 | 3.5x |
| `vq2/3-g128` | 0.4867 | 0.1490 | 3.3x |
| `vq2-g128` | 0.5514 | 0.1958 | 2.8x |

Note that it *raises* weight error while cutting output error — `vq3-g128` goes
from 0.1831 to 0.2425 in weight space and from 0.3026 to 0.0873 in output space.
That is the correct signature: GPTQ deliberately spends weight fidelity to buy
output fidelity, and it is the reason weight error is a poor proxy for anything
that matters.

**The offload can be removed, and it now looks like a reasonable trade.**
`gptq-vq2/3-g128` fits entirely on the cards at 19.4 GiB per rank, with 0.149
expert-output error — the same magnitude as requantising to NVFP4 (0.147), which
costs 45 % *more* memory and still needs 15 GiB per rank in host RAM. Without
error feedback the best-fitting scheme was three times worse, which is why the
first version of this page concluded the opposite.

**The middle of the range is where this gets genuinely attractive.**
`gptq-vq3-g128` cuts the offload from 13.2 GiB per rank to 4.3 at 0.0873 output
error. On the V4-Flash measurements, going from 12 GiB per rank to 6 roughly
doubled decode throughput.

**Converting MXFP4 to NVFP4 is a pure loss.** NVFP4 spends *more* bits (4.50 vs
4.25), takes *more* memory (35.6 vs 33.6 GiB per rank), and still lands 0.147
away from the original. FP8 block scales cannot represent the power-of-two
scales MXFP4 already used, so the conversion adds error while costing space.

**Incoherence processing survives error feedback, and mixed precision does
too.** Without GPTQ the rotation is worth about a fifth of the error at 3 bits
(0.3026 vs 0.3811) and almost nothing at 2 (0.5514 vs 0.5691). With it,
`gptq-vq3-g128` at 3.12 bits beats `gptq-int3-g128` at 3.25 — better, at fewer
bits. And `gptq-vq2/3-g128` (2 bits on `w1`/`w3`, 3 on `w2`) beats
`gptq-vq2-g128` by 24 % for 0.34 extra bits, because `w2` reads a SwiGLU product
whose dynamic range is far wider than the normed activations `w1` and `w3` see,
and it is only a third of an expert's weights.

### A methodological note worth repeating

The first GPTQ run made every *rotated* scheme worse — `vq3-g128` went from
0.3026 to 0.4018 — while making the unrotated `int3-g128` three times better.
That asymmetry was the tell. Rotating the weights by `R` along the input
dimension means the columns now multiply `R^T x`, so the Hessian has to become
`R^T H R`; feeding the unrotated Hessian pushes each column's error along the
wrong axis. Without the unrotated control in the same table, "GPTQ hurts EXL3-
style schemes" would have looked like a finding instead of a bug.

### So which one would you actually build?

**`gptq-vq3-g128`, if you want the model to stay close to what DeepSeek
shipped**: 3.12 bits per weight, 24.7 GiB per rank, 4.3 GiB per rank offloaded
instead of 13.2, at 0.087 expert-output error. Most of the offload cost
disappears and the perturbation is the smallest of anything that shrinks the
model at all.

**`gptq-vq2/3-g128`, if you want the offload gone**: 2.46 bits per weight,
19.4 GiB per rank, nothing in host RAM, at 0.149 — which is where a 4.5-bit
NVFP4 requantisation lands while still needing 15 GiB per rank offloaded.

Both numbers are expert-output error, which is a proxy. Expert outputs are
scaled by routing weights and added to a much larger residual, so end-to-end
damage is smaller than these look — but nothing here measures that, and nothing
here should be read as claiming either model would still be good. What the table
does establish is the ordering and the shape of the trade, and that the
interesting lever is error feedback rather than the storage format.

## Was KTransformers the better idea instead?

KTransformers answers the same problem differently: leave the experts in host
RAM and have the **CPU** multiply by them, so only activations cross PCIe.
Both designs are bandwidth problems at decode and both touch identical bytes, so
whichever side has more bandwidth wins. `tools/expert_quant/cpu_experts.py`
measures the CPU side directly — a gathered expert GEMV at the model's real
shapes, with a fresh routing draw per iteration so the working set cannot sit in
cache.

On 2x Xeon Gold 6530, 64 threads, `numactl --interleave=all`:

| | |
| --- | ---: |
| host DRAM, 8 GiB sequential read | 215-267 GB/s |
| gathered expert GEMV, effective | **110 GB/s** |
| one PCIe Gen5 x16 link, measured UVA read | 51.3 GB/s |
| eight of them | **410 GB/s** |

which settles it:

| design | ms per decode step, routed experts only | ceiling |
| --- | ---: | ---: |
| CPU experts, all 384 on host, bf16 | 154.1 | 6.5 tok/s |
| ... same, extrapolated to a perfect 4.25-bit AMX kernel | 40.9 | 24.4 tok/s |
| GPU experts, 36 % offloaded (the deployed config) | 4.2 | 238 tok/s |
| GPU experts, 100 % offloaded | 11.0 | 91 tok/s |

**CPU experts win below about 2.1 GPUs on this box.** The crossover is where one
memory system stops beating N PCIe links — `gpus x 51.3 < 110` — and this
machine has eight. KTransformers is a good design for the hardware it targets,
one or two GPUs against a large host; with eight GPUs the aggregate PCIe
bandwidth is four times the achievable host bandwidth and the experts belong on
the cards.

Note the bf16 row is what torch can do today; the 4.25-bit row grants
KTransformers a perfect hand-written AMX kernel at the shipped precision and
assumes it stays bandwidth-bound. Even with that gift it is 9.7x slower than the
deployed GPU configuration. The `--dtype int8` path in the tool is *not* a
substitute for that kernel — torch has no fused int8 gathered expert GEMV on
CPU, so it materialises a bf16 copy per expert and measures the conversion; the
tool says so when you ask for it.

## Reproducing

```bash
python tools/expert_quant/explore.py /data/models/DeepSeek-V4.1-Flash \
    --layers 6 20 39 --experts 3 --tokens 512 --markdown

numactl --interleave=all python tools/expert_quant/cpu_experts.py \
    --threads 64 --offload-fraction 0.357
```

The explorer prints a warning if the MXFP4 control stops round-tripping, which
is the first thing to check if the numbers ever look different.

## What this study does not do

- **No end-to-end evaluation.** Expert-output error is a proxy. Perplexity or a
  benchmark suite on a rebuilt checkpoint is the real test, and it needs
  inference kernels for these formats, which is the bulk of what EXL3 and
  KTransformers actually are.
- **No trellis.** The `vq*` rows are EXL3's structure minus its trellis-coded
  quantiser, which buys roughly another 0.1-0.2 bits of effective precision.
  Read them as a conservative estimate of that family.
- **Synthetic activations.** Unit-RMS noise shaped by the real `ffn_norm` gain
  is a defensible model of a normed hidden state, but it is not a calibration
  set, and it has no outlier channels of the kind real text produces.
