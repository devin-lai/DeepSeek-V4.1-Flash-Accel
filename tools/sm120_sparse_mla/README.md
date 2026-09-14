# SM120 sparse-MLA: a reference implementation and a shape probe

`dsv4_ref.py` is a PyTorch implementation of FlashInfer's SM120 DSv4 sparse-MLA
decode, including the packed FP8 KV layout the kernel expects — which is not
documented anywhere outside the CUDA headers:

```
per page of `pbs` tokens:
  [0        : pbs*576)   per token: 448 B FP8-E4M3 nope, then 128 B bf16 rope
  [pbs*576  : pbs*584)   per token: 8 B scale footer (7 UE8M0 exponents + 1 pad)
```

Note the footer: a token costs 584 bytes but its block scales live at the *end*
of the page, at `pbs*576 + local_idx*8`, not beside its data. Getting that wrong
produces plausible-looking garbage rather than an error.

`probe_sm120.py` uses the reference two ways.

**As an oracle.** It first runs four shapes FlashInfer already ships plus one
dual-cache case, and compares them to the reference. Those rows are the
experiment's control: they establish that the byte layout, the UE8M0 convention
and the attention-sink handling are all right, so that a disagreement on a new
shape means something.

```
h=8    topk=128   pbs=64    PASS  rel=2.68e-02 cos=0.999702
h=8    topk=512   pbs=64    PASS  rel=2.84e-02 cos=0.999721
h=16   topk=256   pbs=64    PASS  rel=2.91e-02 cos=0.999699
h=8    topk=1024  pbs=64    PASS  rel=2.34e-02 cos=0.999712
h=8    topk=128   pbs=64  extra=128  PASS  rel=2.70e-02 cos=0.999707
```

The residual is not error in the reference: the kernel quantises Q to FP8
internally, so about 2.7 % relative on the largest element at `cos ≈ 0.9997` is
the floor for every shape, shipped or added.

**As a coverage sweep.** `--sweep` walks `page_block_size` x `num_heads` x
`topk` and reports which shapes dispatch and how far each lands from the
reference. Against stock FlashInfer 0.6.18.post1 only the 25 enumerated
`(heads, topk)` pairs at `pbs=64` dispatch; with `upstream/flashinfer/apply_patch.py`
applied:

```
141 ok, 0 failed of 141
worst rel: h=64 topk=128 pbs=128 rel=3.672e-02
worst cos: h=16 topk=128 pbs=128 cos=0.999685
numerical mismatches: 0
```

including `h=8 topk=1152 pbs=32`, which is what DeepSeek-V4.1-Flash asks for on
TP8 and which stock FlashInfer rejects outright.

## Running it

Needs an sm_120 card and FlashInfer; no checkpoint, no weights, seconds per
shape after the JIT build.

```bash
python probe_sm120.py --oracle                 # is the reference trustworthy?
python probe_sm120.py --shape 8,1152,32        # one shape
python probe_sm120.py --sweep --json sweep.json
```
