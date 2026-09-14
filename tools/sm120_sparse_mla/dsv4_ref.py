"""Packing and a PyTorch reference for FlashInfer's SM120 DSv4 sparse MLA.

The kernel's on-device KV layout is not documented outside the CUDA headers, so
this module carries it explicitly.  Per page of ``page_block_size`` (pbs) tokens
the bytes are laid out as a **footer** layout, not as an array of structs:

    [0                : pbs*576)      per token: 448 B FP8 nope, then 128 B bf16 rope
    [pbs*576          : pbs*584)      per token: 8 B scale footer (7 UE8M0 + 1 pad)

so one token costs 584 B but its scale lives at the end of the page, at
``pbs*576 + local_idx*8``.  ``stride_kv_block`` is taken from
``kv_cache.stride(0)``, so a ``[num_pages, pbs*584]`` uint8 tensor is the
simplest container.

Constants mirror ``KVCacheTraits<ModelType::DSV4>`` in
``flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh``.
"""

from __future__ import annotations

import torch

D_NOPE = 448
D_ROPE = 64
D_QK = 512  # D_NOPE + D_ROPE
D_V = 512  # V carries the rope tail too (V_HAS_ROPE)
QUANT_TILE = 64
NUM_SCALES = 7  # D_NOPE // QUANT_TILE
SCALE_BYTES_PER_TOKEN = 8
IO_STRIDE = 576  # D_NOPE + D_ROPE*sizeof(bf16)
BYTES_PER_TOKEN = 584
E8M0_BIAS = 127
FP8_E4M3_MAX = 448.0


def _to_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Power-of-two float scale -> UE8M0 exponent byte."""
    exp = torch.log2(scale).round().to(torch.int32) + E8M0_BIAS
    return exp.clamp_(0, 255).to(torch.uint8)


def _from_e8m0(byte: torch.Tensor) -> torch.Tensor:
    return torch.exp2((byte.to(torch.int32) - E8M0_BIAS).to(torch.float32))


def quantize_nope(nope: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[..., 448] float -> (fp8_e4m3 bytes [..., 448], UE8M0 scale bytes [..., 7]).

    Block scales are rounded **up** to a power of two so the UE8M0 round-trip is
    exact, matching ``fp8_quant.cuh``.
    """
    tiles = nope.float().unflatten(-1, (NUM_SCALES, QUANT_TILE))
    amax = tiles.abs().amax(dim=-1).clamp_min(1e-30)
    exp = torch.ceil(torch.log2(amax / FP8_E4M3_MAX))
    scale = torch.exp2(exp)
    q = (tiles / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return q.flatten(-2).view(torch.uint8), _to_e8m0(scale)


def dequantize_nope(q_bytes: torch.Tensor, scale_bytes: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`quantize_nope`, in float32."""
    q = q_bytes.view(torch.float8_e4m3fn).float().unflatten(-1, (NUM_SCALES, QUANT_TILE))
    return (q * _from_e8m0(scale_bytes).unsqueeze(-1)).flatten(-2)


def pack_pages(
    nope: torch.Tensor, rope: torch.Tensor, pbs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack ``[num_tokens, 448] / [num_tokens, 64]`` into the paged byte layout.

    ``num_tokens`` must be a multiple of ``pbs``.  Returns the ``[num_pages,
    pbs*584]`` uint8 cache and the dequantised ``[num_tokens, 512]`` float32 K/V
    the kernel will actually see -- the reference must score against *that*, not
    against the pre-quantisation input.
    """
    num_tokens = nope.shape[0]
    assert num_tokens % pbs == 0, f"{num_tokens} tokens is not a multiple of pbs={pbs}"
    num_pages = num_tokens // pbs

    q_bytes, scale_bytes = quantize_nope(nope)
    rope_bf16 = rope.to(torch.bfloat16)

    cache = torch.zeros(num_pages, pbs * BYTES_PER_TOKEN, dtype=torch.uint8, device=nope.device)
    body = cache[:, : pbs * IO_STRIDE].view(num_pages, pbs, IO_STRIDE)
    body[:, :, :D_NOPE] = q_bytes.view(num_pages, pbs, D_NOPE)
    body[:, :, D_NOPE:] = rope_bf16.view(num_pages, pbs, D_ROPE).view(torch.uint8).view(
        num_pages, pbs, D_ROPE * 2
    )
    footer = cache[:, pbs * IO_STRIDE :].view(num_pages, pbs, SCALE_BYTES_PER_TOKEN)
    footer[:, :, :NUM_SCALES] = scale_bytes.view(num_pages, pbs, NUM_SCALES)

    effective = torch.cat(
        [dequantize_nope(q_bytes, scale_bytes), rope_bf16.float()], dim=-1
    )
    return cache, effective


def sparse_mla_dsv4_ref(
    q: torch.Tensor,
    kv_effective: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    extra_kv_effective: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference sparse MLA in float32.  Returns ``(out [T,H,512], lse [T,H])``.

    ``kv_effective`` is the post-quantisation K/V the kernel reads, i.e. the
    second return of :func:`pack_pages`.  A negative index, or a position at or
    past ``topk_length``, is masked out.  The extra segment is concatenated
    after the main one, exactly as the kernel chunks it.
    """
    num_tokens, num_heads, _ = q.shape
    qf = q.float()

    def gather(kv: torch.Tensor, idx: torch.Tensor, lens: torch.Tensor | None):
        width = idx.shape[-1]
        flat = idx.reshape(num_tokens, width)
        valid = flat >= 0
        if lens is not None:
            pos = torch.arange(width, device=idx.device).unsqueeze(0)
            valid &= pos < lens.reshape(num_tokens, 1)
        k = kv[flat.clamp_min(0).long()]  # [T, width, 512]
        return k, valid

    k_main, valid_main = gather(kv_effective, indices, topk_length)
    logits = torch.einsum("thd,tkd->thk", qf, k_main) * sm_scale
    mask = valid_main.unsqueeze(1).expand(-1, num_heads, -1)
    values = k_main

    if extra_indices is not None:
        assert extra_kv_effective is not None
        k_extra, valid_extra = gather(extra_kv_effective, extra_indices, extra_topk_length)
        logits = torch.cat(
            [logits, torch.einsum("thd,tkd->thk", qf, k_extra) * sm_scale], dim=-1
        )
        mask = torch.cat([mask, valid_extra.unsqueeze(1).expand(-1, num_heads, -1)], dim=-1)
        values = torch.cat([values, k_extra], dim=1)

    logits = logits.masked_fill(~mask, float("-inf"))
    if attn_sink is not None:
        sink = attn_sink.float().view(1, num_heads, 1).expand(num_tokens, -1, -1)
        logits = torch.cat([logits, sink], dim=-1)

    m = logits.amax(dim=-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    p = torch.exp(logits - m)
    denom = p.sum(dim=-1, keepdim=True)
    lse = (m + torch.log(denom.clamp_min(1e-30))).squeeze(-1)

    if attn_sink is not None:
        p = p[..., :-1]
    out = torch.einsum("thk,tkd->thd", p / denom.clamp_min(1e-30), values)
    return out, lse
