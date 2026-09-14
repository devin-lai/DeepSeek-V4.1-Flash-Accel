"""Candidate quantisation schemes for a MoE expert bank, and their cost.

Each scheme is a pair of functions -- one that returns bits per weight, one that
does a quantise/dequantise round trip -- so a scheme can be judged on the two
things that decide a deployment: how many bytes of GPU it costs, and how much
signal it destroys.

The reference point is `mxfp4`, because that is what DeepSeek-V4.1-Flash
actually ships.  A scheme is only interesting here if it is materially smaller
than MXFP4; "as accurate as FP16" is not the bar, "does it still answer
correctly at 2.5 bits" is.

Incoherence processing (the random-rotation step EXL3 and QuIP# rely on) is
implemented as a Hadamard transform with random sign flips, applied along the
input dimension in blocks.  It costs nothing in stored bits -- the sign seed is
one integer per tensor -- and it is what makes sub-3-bit quantisation viable, so
the paired `*-had` schemes isolate exactly how much it is worth.
"""

from __future__ import annotations

import math

import torch

FP4_E2M1_LEVELS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)
E8M0_BIAS = 127


# --------------------------------------------------------------- primitives


def _round_to_levels(x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Nearest-value rounding onto a signed symmetric level set."""
    sign = torch.sign(x)
    idx = torch.bucketize(x.abs().contiguous(), levels)
    lo = levels[(idx - 1).clamp_min(0)]
    hi = levels[idx.clamp_max(levels.numel() - 1)]
    pick = torch.where((x.abs() - lo) <= (hi - x.abs()), lo, hi)
    return sign * pick


def _blocks(w: torch.Tensor, group: int) -> tuple[torch.Tensor, int]:
    """Split the last dim into groups, zero-padding the tail."""
    n = w.shape[-1]
    pad = (-n) % group
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    return w.unflatten(-1, (w.shape[-1] // group, group)), pad


def hadamard(n: int, device, dtype=torch.float32) -> torch.Tensor:
    """Normalised Sylvester-Hadamard matrix; n must be a power of two."""
    assert n & (n - 1) == 0, f"Hadamard size must be a power of two, got {n}"
    h = torch.ones(1, 1, device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    return h / math.sqrt(n)


class Incoherence:
    """Random-sign Hadamard rotation applied blockwise along the input dim.

    Orthogonal, so it preserves the Frobenius norm and the layer's output under
    the matching inverse rotation of the activations -- which is why a real
    implementation folds the rotation into the preceding norm rather than paying
    for it at runtime.
    """

    def __init__(self, width: int, block: int, device, seed: int = 0):
        self.block = min(block, 1 << int(math.log2(width)))
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.signs = (torch.randint(0, 2, (width,), generator=g) * 2 - 1).to(
            device=device, dtype=torch.float32
        )
        self.h = hadamard(self.block, device)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        n = w.shape[-1]
        usable = (n // self.block) * self.block
        out = (w * self.signs[:n]).clone()
        head = out[..., :usable].unflatten(-1, (usable // self.block, self.block))
        out[..., :usable] = (head @ self.h).flatten(-2)
        return out

    def inverse(self, w: torch.Tensor) -> torch.Tensor:
        n = w.shape[-1]
        usable = (n // self.block) * self.block
        out = w.clone()
        head = out[..., :usable].unflatten(-1, (usable // self.block, self.block))
        out[..., :usable] = (head @ self.h.T).flatten(-2)
        return out * self.signs[:n]


# ------------------------------------------------------------------ schemes


def mxfp4(w: torch.Tensor, group: int = 32) -> torch.Tensor:
    """OCP MXFP4: E2M1 elements, one UE8M0 (power-of-two) scale per 32.

    This is the format the V4.1 checkpoint ships, so it is the baseline every
    other row is compared against rather than an alternative.  Round-tripping
    already-MXFP4 data through it is exact, which is the control that says the
    encoder here matches the one that produced the checkpoint.
    """
    blk, pad = _blocks(w, group)
    amax = blk.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    # OCP MX shared scale: 2**(floor(log2(amax)) - emax_elem), with emax_elem=2
    # for E2M1 (max normal 6 = 1.5 * 2**2).  Not `floor(log2(amax/6))`, which
    # picks a scale one binade too small and clips the block maximum -- and, in
    # particular, fails to round-trip data that is *already* MXFP4.
    exp = (torch.floor(torch.log2(amax)) - 2).clamp(-E8M0_BIAS, 255 - E8M0_BIAS)
    scale = torch.exp2(exp)
    q = _round_to_levels((blk / scale).clamp(-6, 6), FP4_E2M1_LEVELS.to(w.device))
    out = (q * scale).flatten(-2)
    return out[..., : w.shape[-1]] if pad else out


def nvfp4(w: torch.Tensor, group: int = 16) -> torch.Tensor:
    """NVFP4: E2M1 elements, an FP8-E4M3 scale per 16, one FP32 per tensor."""
    global_scale = w.abs().amax().clamp_min(1e-30) / (6.0 * 448.0)
    blk, pad = _blocks(w / global_scale, group)
    amax = blk.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    scale = (amax / 6.0).to(torch.float8_e4m3fn).float().clamp_min(1e-30)
    q = _round_to_levels((blk / scale).clamp(-6, 6), FP4_E2M1_LEVELS.to(w.device))
    out = (q * scale).flatten(-2) * global_scale
    return out[..., : w.shape[-1]] if pad else out


def int_group(w: torch.Tensor, bits: int, group: int, asymmetric: bool = True) -> torch.Tensor:
    """Group-wise integer quantisation -- the GPTQ/AWQ storage format."""
    blk, pad = _blocks(w, group)
    if asymmetric:
        lo = blk.amin(dim=-1, keepdim=True)
        hi = blk.amax(dim=-1, keepdim=True)
        scale = ((hi - lo) / (2**bits - 1)).clamp_min(1e-30)
        q = ((blk - lo) / scale).round().clamp(0, 2**bits - 1)
        out = (q * scale + lo).flatten(-2)
    else:
        amax = blk.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
        scale = amax / (2 ** (bits - 1) - 1)
        q = (blk / scale).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        out = (q * scale).flatten(-2)
    return out[..., : w.shape[-1]] if pad else out


def _lloyd_max(samples: torch.Tensor, k: int, iters: int = 40) -> torch.Tensor:
    """1-D Lloyd-Max codebook for the sample distribution (k levels)."""
    qs = torch.linspace(0.5 / k, 1 - 0.5 / k, k, device=samples.device)
    cb = torch.quantile(samples.float().flatten()[:: max(1, samples.numel() // 200_000)], qs)
    for _ in range(iters):
        idx = torch.bucketize(samples.flatten(), (cb[1:] + cb[:-1]) / 2)
        new = torch.zeros_like(cb)
        cnt = torch.zeros_like(cb)
        new.scatter_add_(0, idx, samples.flatten())
        cnt.scatter_add_(0, idx, torch.ones_like(samples.flatten()))
        cb = torch.where(cnt > 0, new / cnt.clamp_min(1), cb)
        cb, _ = torch.sort(cb)
    return cb


def vq_codebook(w: torch.Tensor, bits: int, group: int, seed: int = 0,
                incoherent: bool = True) -> torch.Tensor:
    """Rotation + a distribution-matched scalar codebook, scaled per group.

    This is the shape of EXL3 / QuIP# minus the trellis: rotate so the weights
    look Gaussian, then spend the bit budget on levels placed where the mass
    actually is, rather than uniformly.  The trellis in a real EXL3 build buys
    roughly another 0.1-0.2 bits of effective precision on top of this, so read
    these rows as a conservative estimate of that family.
    """
    rot = Incoherence(w.shape[-1], group, w.device, seed) if incoherent else None
    x = rot.forward(w) if rot else w
    blk, pad = _blocks(x, group)
    scale = blk.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    norm = blk / scale
    cb = _lloyd_max(norm, 2**bits)
    idx = torch.bucketize(norm.flatten(), (cb[1:] + cb[:-1]) / 2)
    out = (cb[idx].view_as(norm) * scale).flatten(-2)
    if pad:
        out = out[..., : x.shape[-1]]
    return rot.inverse(out) if rot else out


# ---------------------------------------------------------------- registry


def _bpw_group(bits: int, group: int, scale_bits: int, zero_bits: int = 0) -> float:
    return bits + (scale_bits + zero_bits) / group


SCHEMES: dict[str, dict] = {
    "mxfp4": dict(
        fn=lambda w: mxfp4(w, 32), bpw=_bpw_group(4, 32, 8),
        note="shipped format: E2M1 + UE8M0 per 32",
    ),
    "nvfp4": dict(
        fn=lambda w: nvfp4(w, 16), bpw=_bpw_group(4, 16, 8),
        note="E2M1 + FP8 scale per 16",
    ),
    "int4-g128": dict(
        fn=lambda w: int_group(w, 4, 128), bpw=_bpw_group(4, 128, 16, 16),
        note="GPTQ/AWQ storage, asymmetric",
    ),
    "int3-g128": dict(
        fn=lambda w: int_group(w, 3, 128), bpw=_bpw_group(3, 128, 16, 16),
        note="GPTQ/AWQ storage, asymmetric",
    ),
    "int2-g64": dict(
        fn=lambda w: int_group(w, 2, 64), bpw=_bpw_group(2, 64, 16, 16),
        note="GPTQ/AWQ storage, asymmetric",
    ),
    "vq4-g128": dict(
        fn=lambda w: vq_codebook(w, 4, 128), bpw=_bpw_group(4, 128, 16),
        note="EXL3-family: Hadamard + Lloyd-Max codebook",
    ),
    "vq3-g128": dict(
        fn=lambda w: vq_codebook(w, 3, 128), bpw=_bpw_group(3, 128, 16),
        note="EXL3-family: Hadamard + Lloyd-Max codebook",
    ),
    "vq2-g128": dict(
        fn=lambda w: vq_codebook(w, 2, 128), bpw=_bpw_group(2, 128, 16),
        note="EXL3-family: Hadamard + Lloyd-Max codebook",
    ),
    "vq3-g128-norot": dict(
        fn=lambda w: vq_codebook(w, 3, 128, incoherent=False), bpw=_bpw_group(3, 128, 16),
        note="control: same codebook, no rotation -- isolates incoherence",
    ),
    "vq2-g128-norot": dict(
        fn=lambda w: vq_codebook(w, 2, 128, incoherent=False), bpw=_bpw_group(2, 128, 16),
        note="control: same codebook, no rotation",
    ),
}


def _mixed_bpw(bits_w13: float, bits_w2: float, group: int, scale_bits: int) -> float:
    """`w1` and `w3` are two thirds of an expert's weights, `w2` the other third."""
    return (2 * bits_w13 + bits_w2) / 3 + scale_bits / group


# Mixed precision, because the three projections are not equally fragile: `w2`
# reads a SwiGLU product whose dynamic range is much wider than the normed
# activations `w1`/`w3` see, and it is the one that writes into the residual
# stream. Spending the extra bit there is the standard move and costs only a
# third of what spending it everywhere would.
SCHEMES["vq2/3-g128"] = dict(
    fn={"w1": lambda w: vq_codebook(w, 2, 128),
        "w3": lambda w: vq_codebook(w, 2, 128),
        "w2": lambda w: vq_codebook(w, 3, 128)},
    bpw=_mixed_bpw(2, 3, 128, 16),
    note="EXL3-family, 2 bits on w1/w3 and 3 on w2",
)
SCHEMES["vq3/4-g128"] = dict(
    fn={"w1": lambda w: vq_codebook(w, 3, 128),
        "w3": lambda w: vq_codebook(w, 3, 128),
        "w2": lambda w: vq_codebook(w, 4, 128)},
    bpw=_mixed_bpw(3, 4, 128, 16),
    note="EXL3-family, 3 bits on w1/w3 and 4 on w2",
)
SCHEMES["int3-g64"] = dict(
    fn=lambda w: int_group(w, 3, 64), bpw=_bpw_group(3, 64, 16, 16),
    note="GPTQ/AWQ storage, finer groups",
)


# --------------------------------------------------------------- GPTQ ------
# Everything above quantises each weight independently of what the layer does
# with it.  Real pipelines do not: GPTQ walks the input dimension column by
# column and pushes each column's rounding error into the columns it has not
# reached yet, weighted by the inverse Hessian of the layer's own activations.
# It is the difference between comparing storage formats and comparing methods,
# and it typically recovers a large part of the gap at 3 bits and below.


def _group_params_int(block: torch.Tensor, bits: int):
    lo = block.amin(dim=-1, keepdim=True)
    hi = block.amax(dim=-1, keepdim=True)
    scale = ((hi - lo) / (2**bits - 1)).clamp_min(1e-30)
    return scale, lo


def _quant_col_int(col: torch.Tensor, params, bits: int) -> torch.Tensor:
    scale, lo = params
    q = ((col.unsqueeze(-1) - lo) / scale).round().clamp(0, 2**bits - 1)
    return (q * scale + lo).squeeze(-1)


def _group_params_vq(block: torch.Tensor, bits: int):
    scale = block.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    cb = _lloyd_max(block / scale, 2**bits)
    return scale, cb


def _quant_col_vq(col: torch.Tensor, params, bits: int) -> torch.Tensor:
    scale, cb = params
    norm = (col.unsqueeze(-1) / scale).squeeze(-1)
    idx = torch.bucketize(norm.contiguous(), (cb[1:] + cb[:-1]) / 2)
    return cb[idx] * scale.squeeze(-1)


def gptq(
    w: torch.Tensor,
    hessian: torch.Tensor,
    *,
    bits: int,
    group: int = 128,
    kind: str = "vq",
    incoherent: bool = True,
    damp: float = 0.01,
    seed: int = 0,
) -> torch.Tensor:
    """GPTQ with a group-wise quantiser.  ``w`` is ``[out, in]``, ``hessian`` ``[in, in]``.

    Follows the published algorithm: damp the Hessian, take the upper Cholesky
    of its inverse, then sweep columns left to right, subtracting each column's
    scaled error from the remaining columns.  Group scales are re-derived at
    each group boundary from the *already compensated* weights, which is what
    makes the compensation compound correctly.
    """
    rot = Incoherence(w.shape[-1], group, w.device, seed) if incoherent else None
    x = (rot.forward(w) if rot else w).float().clone()
    n_in = x.shape[-1]

    h = hessian.float().clone()
    if rot is not None:
        # The rotation is a change of basis on the input dimension: with
        # W' = W R, the activations the columns of W' multiply are R^T x, so the
        # Hessian must become R^T H R.  Feeding the *unrotated* Hessian to the
        # sweep pushes each column's error along the wrong axis, and measures as
        # GPTQ making the rotated schemes worse rather than better.
        # `Incoherence.forward(M)` computes M R, so forward(forward(H)^T) is
        # R^T H R for symmetric H.
        h = rot.forward(rot.forward(h).T.contiguous())
    dead = torch.diag(h) == 0
    h[dead, dead] = 1.0
    x[:, dead] = 0.0
    h += torch.eye(n_in, device=h.device) * (damp * torch.diag(h).mean())
    try:
        hinv = torch.linalg.cholesky(
            torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True
        )
    except RuntimeError:
        # Not positive definite even after damping: fall back to no feedback.
        return vq_codebook(w, bits, group, seed, incoherent) if kind == "vq" \
            else int_group(w, bits, group)

    find = _group_params_vq if kind == "vq" else _group_params_int
    quant = _quant_col_vq if kind == "vq" else _quant_col_int

    out = torch.zeros_like(x)
    params = None
    for j in range(n_in):
        if j % group == 0:
            hi = min(j + group, n_in)
            params = find(x[:, j:hi], bits)
        q = quant(x[:, j], params, bits)
        out[:, j] = q
        err = (x[:, j] - q) / hinv[j, j]
        if j + 1 < n_in:
            x[:, j + 1:] -= err.unsqueeze(1) * hinv[j, j + 1:].unsqueeze(0)
    return rot.inverse(out) if rot else out


# The same formats again, this time with error feedback. Bit budgets are
# identical -- GPTQ changes which values the same number of bits encode, not
# how many there are.
SCHEMES["gptq-vq3-g128"] = dict(
    fn=dict(bits=3, group=128, kind="vq"), gptq=True, bpw=_bpw_group(3, 128, 16),
    note="vq3-g128 + GPTQ error feedback",
)
SCHEMES["gptq-vq2-g128"] = dict(
    fn=dict(bits=2, group=128, kind="vq"), gptq=True, bpw=_bpw_group(2, 128, 16),
    note="vq2-g128 + GPTQ error feedback",
)
SCHEMES["gptq-vq2/3-g128"] = dict(
    fn={"w1": dict(bits=2, group=128, kind="vq"),
        "w3": dict(bits=2, group=128, kind="vq"),
        "w2": dict(bits=3, group=128, kind="vq")},
    gptq=True, bpw=_mixed_bpw(2, 3, 128, 16),
    note="mixed 2/3 + GPTQ error feedback",
)
SCHEMES["gptq-int3-g128"] = dict(
    fn=dict(bits=3, group=128, kind="int", incoherent=False), gptq=True,
    bpw=_bpw_group(3, 128, 16, 16),
    note="int3-g128 + GPTQ error feedback (the classic GPTQ recipe)",
)
