"""Correctness tests for the DeepEP-EPv2-aligned intranode dispatch/combine port.

Small (nprocs=2) smoke plus a full 8-GPU run at DeepEP's own headline shape
(8K tokens, hidden=7168, top-8, 256 experts). Bf16 accumulates rounding error
across topk contributions, so correctness is judged by relative L2 error
against the identity-compute reference (see ``reference.py``), not a strict
per-element allclose -- a few outlier elements near a sign-cancelling zero can
have a large *relative* error while the overall reconstruction is fine.
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


def _run(local_rank: int, num_ranks: int, num_tokens: int, hidden: int, topk: int, num_experts: int, num_sms: int):
    rank, num_ranks, group = init_dist(local_rank, num_ranks)

    torch.manual_seed(1234 + rank)
    device = f"cuda:{local_rank}"
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    topk_idx = torch.randint(0, num_experts, (num_tokens, topk), device=device)
    topk_weights = torch.rand(num_tokens, topk, device=device)

    buf = Buffer(
        group=group,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        num_max_tokens_per_rank=num_tokens,
        hidden=hidden,
        num_topk=topk,
        num_experts=num_experts,
        dtype=torch.bfloat16,
        num_sms=num_sms,
    )
    try:
        recv_x, recv_topk_idx, recv_topk_weights, handle = buf.dispatch(x, topk_idx, topk_weights)
        n = handle.num_recv_tokens
        recv_x, recv_topk_idx, recv_topk_weights = recv_x[:n], recv_topk_idx[:n], recv_topk_weights[:n]
        expert_out = reference.simulate_expert_compute(recv_x, recv_topk_idx, recv_topk_weights)
        combined = buf.combine(expert_out, handle)

        expected = reference.reference_combined(x, topk_weights)
        rel_l2 = (combined.float() - expected.float()).norm().item() / expected.float().norm().item()
        assert rel_l2 < _BF16_REL_L2_THRESHOLD, f"rank {rank}: rel_l2_error={rel_l2} exceeds {_BF16_REL_L2_THRESHOLD}"
    finally:
        buf.close()
        dist.destroy_process_group()


@tilelang.testing.requires_cuda
@distributed_test(nprocs=2)
def test_dispatch_combine_smoke(local_rank: int, num_ranks: int):
    _run(local_rank, num_ranks, num_tokens=64, hidden=128, topk=2, num_experts=8, num_sms=2)


@tilelang.testing.requires_cuda
@distributed_test(nprocs=8)
def test_dispatch_combine_v3_shape(local_rank: int, num_ranks: int):
    """DeepEP's own headline benchmark shape: 8K tokens, hidden=7168, top-8, 256 experts."""
    _run(local_rank, num_ranks, num_tokens=8192, hidden=7168, topk=8, num_experts=256, num_sms=64)


if __name__ == "__main__":
    tilelang.testing.main()
