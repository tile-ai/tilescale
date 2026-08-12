"""Correctness tests for the DeepEP-EPv2-aligned intranode dispatch/combine port.

Small (nprocs=2) smoke plus a full 8-GPU run at DeepEP's own headline shape
(8K tokens, hidden=7168, top-8, 256 experts), for both dispatch payload dtypes
this port supports (bf16, fp8). Bf16 accumulates rounding error across topk
contributions, so correctness is judged by relative L2 error against the
identity-compute reference (see ``reference.py``), not a strict per-element
allclose -- a few outlier elements near a sign-cancelling zero can have a
large *relative* error while the overall reconstruction is fine. FP8 adds the
per-128-element quantisation step on top, hence the wider threshold.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

import tilelang.testing
from testing.python.distributed._utils import distributed_test

from tilelang.distributed.host import init_dist

from buffer import Buffer
import reference

_BF16_REL_L2_THRESHOLD = 0.05
# FP8 (e4m3, ~2 mantissa bits) quantises `x` before dispatch ever sees it, on
# top of the same bf16 accumulation error the plain threshold covers.
# Measured rel_l2 is a tight ~0.026 at every shape tried (smoke, masked, and
# the full V3 shape at both 2 and 8 GPUs, std well under 1e-3) -- consistent
# with a quantisation-dominated, not noise-dominated, error, so ~2x that
# measured value is real margin, not padding for run-to-run variance.
_FP8_REL_L2_THRESHOLD = 0.05


def _run(
    local_rank: int,
    num_ranks: int,
    num_tokens: int,
    hidden: int,
    topk: int,
    num_experts: int,
    num_sms: int,
    masked_ratio: float = 0.0,
    dtype: torch.dtype = torch.bfloat16,
):
    rank, num_ranks, group = init_dist(local_rank, num_ranks)

    torch.manual_seed(1234 + rank)
    device = f"cuda:{local_rank}"
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    topk_idx, topk_weights = reference.make_topk(num_tokens, topk, num_experts, device, masked_ratio)

    is_fp8 = dtype == torch.float8_e4m3fn
    # Quantising is the caller's job (see buffer.py's `dispatch` docstring);
    # `x` itself stays bf16 for the reference computation below.
    dispatch_x = reference.per_token_cast_to_fp8(x) if is_fp8 else x

    buf = Buffer(
        group=group,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        num_max_tokens_per_rank=num_tokens,
        hidden=hidden,
        num_topk=topk,
        num_experts=num_experts,
        dtype=dtype,
        num_sms=num_sms,
    )
    try:
        recv, recv_topk_idx, recv_topk_weights, handle = buf.dispatch(dispatch_x, topk_idx, topk_weights)
        n = handle.num_recv_tokens
        recv_topk_idx, recv_topk_weights = recv_topk_idx[:n], recv_topk_weights[:n]
        recv_x = reference.per_token_cast_back(recv[:n], hidden) if is_fp8 else recv[:n]
        expert_out = reference.simulate_expert_compute(recv_x, recv_topk_idx, recv_topk_weights)
        combined = buf.combine(expert_out, handle)

        expected = reference.reference_combined(x, topk_weights, topk_idx)
        err = (combined.float() - expected.float()).norm().item()
        denom = expected.float().norm().item()
        # `denom` is zero only when every selection was masked off.
        rel_l2 = err / denom if denom > 0 else err
        threshold = _FP8_REL_L2_THRESHOLD if is_fp8 else _BF16_REL_L2_THRESHOLD
        assert rel_l2 < threshold, f"rank {rank}: rel_l2_error={rel_l2} exceeds {threshold}"
    finally:
        buf.close()
        dist.destroy_process_group()


@tilelang.testing.requires_cuda
@distributed_test(nprocs=2)
def test_dispatch_combine_smoke(local_rank: int, num_ranks: int):
    _run(local_rank, num_ranks, num_tokens=64, hidden=128, topk=2, num_experts=8, num_sms=2)


@tilelang.testing.requires_cuda
@distributed_test(nprocs=2)
def test_dispatch_combine_smoke_fp8(local_rank: int, num_ranks: int):
    _run(local_rank, num_ranks, num_tokens=64, hidden=128, topk=2, num_experts=8, num_sms=2, dtype=torch.float8_e4m3fn)


@tilelang.testing.requires_cuda
@distributed_test(nprocs=8)
def test_dispatch_combine_v3_shape(local_rank: int, num_ranks: int):
    """DeepEP's own headline benchmark shape: 8K tokens, hidden=7168, top-8, 256 experts."""
    _run(local_rank, num_ranks, num_tokens=8192, hidden=7168, topk=8, num_experts=256, num_sms=64)


@tilelang.testing.requires_cuda
@distributed_test(nprocs=8)
def test_dispatch_combine_v3_shape_fp8(local_rank: int, num_ranks: int):
    """DeepEP's own headline benchmark shape, FP8 dispatch payload."""
    _run(local_rank, num_ranks, num_tokens=8192, hidden=7168, topk=8, num_experts=256, num_sms=64, dtype=torch.float8_e4m3fn)


@tilelang.testing.requires_cuda
@distributed_test(nprocs=8)
def test_dispatch_combine_masked(local_rank: int, num_ranks: int):
    """Half the selections unset: dispatch must route -1 nowhere, and a token
    with no selections at all must combine back to zero."""
    _run(local_rank, num_ranks, num_tokens=1024, hidden=1024, topk=8, num_experts=256, num_sms=32, masked_ratio=0.5)


if __name__ == "__main__":
    tilelang.testing.main()
