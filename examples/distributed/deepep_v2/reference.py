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


def reference_combined(x: torch.Tensor, topk_weights: torch.Tensor,
                       topk_idx: torch.Tensor | None = None) -> torch.Tensor:
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
# groups of `group` elements, keeping one fp32 scale per group. Dispatch moves
# the quantised values and the scales; the caller casts back before the expert
# computation, which is why `dispatch` never sees a bf16 tensor on this path.
# ---------------------------------------------------------------------------

_FP8_MAX = 448.0


def per_token_cast_to_fp8(x: torch.Tensor, group: int = 128):
    """`[m, n]` bf16 -> `([m, n]` fp8`, [m, n // group]` fp32 scales`)`."""
    assert x.dim() == 2 and x.shape[1] % group == 0, f"hidden {x.shape[1]} is not a multiple of {group}"
    m, n = x.shape
    grouped = x.view(m, -1, group)
    amax = grouped.abs().float().amax(dim=2).clamp(1e-4)
    q = (grouped.float() * (_FP8_MAX / amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    return q.view(m, n).contiguous(), (amax / _FP8_MAX).contiguous()


def per_token_cast_back(x_fp8: torch.Tensor, x_scales: torch.Tensor, group: int = 128) -> torch.Tensor:
    """Inverse of `per_token_cast_to_fp8`, back to bf16."""
    m, n = x_fp8.shape
    return (x_fp8.view(m, -1, group).float() * x_scales.view(m, -1, 1)).view(m, n).to(torch.bfloat16)
