"""Correctness reference for the dispatch/combine round trip.

There is no fused GEMM in this port's scope. DeepEP's own non-expand
``combine`` does not re-apply gate weights or sum multiple local-expert
contributions itself -- that happens in the grouped-GEMM epilogue *before*
`combine` is called (see ``kernels/combine.py`` for the detailed reasoning).
``simulate_expert_compute`` stands in for that epilogue: identity "expert
compute" (the input row, unchanged) scaled by the sum of this rank's valid
top-k gate weights for that compact row (0 if this rank doesn't actually own
any of the token's experts, >1 term summed if it owns more than one). This
still exercises every routing/addressing/accumulation path in dispatch and
combine -- a misrouted or dropped/duplicated contribution changes the sum a
token's original weights should reconstruct to.
"""

from __future__ import annotations

import torch


def simulate_expert_compute(recv_x: torch.Tensor, recv_topk_idx: torch.Tensor, recv_topk_weights: torch.Tensor) -> torch.Tensor:
    """Identity-compute stand-in for the missing grouped-GEMM epilogue."""
    valid = recv_topk_idx >= 0
    weight_sum = torch.where(valid, recv_topk_weights, torch.zeros_like(recv_topk_weights)).sum(dim=-1, keepdim=True)
    return recv_x * weight_sum.to(recv_x.dtype)


def reference_combined(x: torch.Tensor, topk_weights: torch.Tensor, topk_idx: torch.Tensor | None = None) -> torch.Tensor:
    """Expected combine output: every token's top-k weight sum applied once.

    Entries with `topk_idx < 0` are unselected and contribute nothing, so a
    token with no selections at all comes back as zero.
    """
    w = topk_weights
    if topk_idx is not None:
        w = torch.where(topk_idx >= 0, w, torch.zeros_like(w))
    return x * w.sum(dim=-1, keepdim=True).to(x.dtype)


def make_topk(num_tokens: int, topk: int, num_experts: int, device, masked_ratio: float = 0.0):
    """Routing inputs, with `masked_ratio` of the selections marked unselected.

    `-1` is DeepEP's "no selection" marker and reaches dispatch's dedup and
    slot-claiming as a destination of -1, so it needs exercising rather than
    assuming.
    """
    idx = torch.randint(0, num_experts, (num_tokens, topk), device=device)
    weights = torch.rand(num_tokens, topk, device=device)
    if masked_ratio > 0:
        idx = idx.masked_fill(torch.rand_like(idx, dtype=torch.float) < masked_ratio, -1)
    return idx, weights


# ---------------------------------------------------------------------------
# FP8 dispatch
#
# The layout is DeepEP's `per_token_cast_to_fp8`: quantise per token over
# groups of `group` elements, keeping one fp32 scale per group. Values and
# scales are packed into one uint8 row -- payload bytes, then the per-group
# fp32 scales immediately after -- rather than returned as two separate
# tensors. Dispatch's scatter does one remote store per token-destination
# pair either way; keeping values and scales apart would mean a *second*
# store for the scale alone, and a store's fixed per-call cost (peer-address
# translation, warp sync) turns out to swamp a transfer that small --
# measured at ~74us out of ~567us end to end, for 224 bytes of scale against
# 7168 bytes of payload. Packing them contiguously here costs nothing extra:
# quantisation already has to write its output somewhere, and writing the
# scale right after the payload it belongs to is no more expensive than
# writing it elsewhere.
#
# The row is padded, at minimum to 16 bytes: `put_warp`'s bulk path
# reinterprets the row as `int4`, so a row whose *stride* is not 16-byte
# aligned is a `"misaligned address"` CUDA error outright (`hidden=128` packs
# to 132 raw bytes, not even a multiple of 16).
#
# 512 = 32 lanes * 16 bytes is a second, coarser boundary worth reaching when
# it is cheap. `cp_warp_impl`'s *drain* loop -- the tail past whole
# `UNROLL_FACTOR`-sized chunks -- is `for i = drain_start + lane; i < N_int4;
# i += 32`, so unless the leftover `int4` count is a multiple of 32 some lanes
# run one more iteration than the rest, which idle through it. At hidden=7168
# the fused row is 7392 bytes: a multiple of 16 but not of 512 (462 int4,
# remainder 78 -- 14 lanes take a 3rd drain iteration, 18 take 2).
#
# But reaching 512 costs the padding bytes, and those cross NVLink too. Which
# way that trades depends entirely on how much padding it takes, so
# `packed_row_bytes` only pays for 512 when it is nearly free. Measured on the
# real dispatch kernel (8xB200, 8192 tokens/rank, top-8, 256 experts, 64 SMs,
# min of 3-4 samples each, all taken in verified-idle windows):
#
#     hidden   16-aligned    512-aligned   padding    result
#       2048   2112 -> 211us  2560 -> 230us  +21.2%   512 is 8.9% slower
#       4096   4224 -> 325us  4608 -> 348us   +9.1%   512 is 6.8% slower
#       7168   7392 -> 535us  7680 -> 522us   +3.9%   512 is 2.4% faster
#
# So a uniform drain is worth roughly 6% gross, and `_ROW_ALIGN_MAX_PAD`
# takes it only when the padding costs less than that. Note this is invisible
# in GB/s -- an isolated `put_warp` roofline reads 594 GB/s at 7392 against
# 618 at 7680 and 512 looks like a clear win, but the wall time is identical
# (203.8 against 203.7us) because the extra bytes eat exactly what the
# uniformity buys. Compare times, not rates, whenever the byte count moves.
#
# `buffer.py`'s `row_bytes` calls this; `kernels/dispatch.py` is handed the
# result rather than recomputing it.
# ---------------------------------------------------------------------------

_FP8_MAX = 448.0

# `int4`, the unit `put_warp`'s bulk path moves -- a hard requirement.
_ROW_ALIGN = 16
# 32 lanes * one `int4` each: one full drain iteration, warp-uniform.
_ROW_ALIGN_WIDE = 512
# How much padding reaching `_ROW_ALIGN_WIDE` may cost before it stops paying
# for itself. See the measurements in the module docstring above.
_ROW_ALIGN_MAX_PAD = 0.05


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def align_up(x: int, align: int) -> int:
    return _cdiv(x, align) * align


def packed_row_bytes(hidden: int, group: int = 128) -> int:
    """FP8's packed row width in bytes: `hidden` payload bytes plus one fp32
    scale per `group`, padded to whichever of `_ROW_ALIGN`/`_ROW_ALIGN_WIDE`
    is faster at this size. See the module docstring above."""
    raw = hidden + (hidden // group) * 4
    narrow = align_up(raw, _ROW_ALIGN)
    wide = align_up(raw, _ROW_ALIGN_WIDE)
    return wide if wide <= narrow * (1 + _ROW_ALIGN_MAX_PAD) else narrow


def per_token_cast_to_fp8(x: torch.Tensor, group: int = 128) -> torch.Tensor:
    """`[m, n]` bf16 -> `[m, packed_row_bytes(n, group)]` uint8 (fp8 payload, then fp32 scales, then padding)."""
    assert x.dim() == 2 and x.shape[1] % group == 0, f"hidden {x.shape[1]} is not a multiple of {group}"
    m, n = x.shape
    grouped = x.view(m, -1, group)
    amax = grouped.abs().float().amax(dim=2).clamp(1e-4)
    q = (grouped.float() * (_FP8_MAX / amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n)
    scales = (amax / _FP8_MAX).contiguous()
    scale_bytes = scales.shape[1] * 4

    packed = torch.empty((m, packed_row_bytes(n, group)), dtype=torch.uint8, device=x.device)
    packed[:, :n] = q.view(torch.uint8)
    packed[:, n : n + scale_bytes].view(torch.float32).copy_(scales)
    return packed


def per_token_cast_back(packed: torch.Tensor, hidden: int, group: int = 128) -> torch.Tensor:
    """Inverse of `per_token_cast_to_fp8`, back to bf16."""
    m, _ = packed.shape
    scale_dim = hidden // group
    values = packed[:, :hidden].view(torch.float8_e4m3fn)
    scales = packed[:, hidden : hidden + scale_dim * 4].view(torch.float32)
    return (values.view(m, -1, group).float() * scales.view(m, -1, 1)).view(m, hidden).to(torch.bfloat16)
