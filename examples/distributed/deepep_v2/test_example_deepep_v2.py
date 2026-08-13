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

import pytest
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
    num_bias: int = 0,
    scatter_sms: int = 0,
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
        scatter_sms=scatter_sms,
    )
    try:
        recv, recv_topk_idx, recv_topk_weights, handle, _ = buf.dispatch(dispatch_x, topk_idx, topk_weights)
        n = handle.num_recv_tokens
        recv_topk_idx, recv_topk_weights = recv_topk_idx[:n], recv_topk_weights[:n]
        recv_x = reference.per_token_cast_back(recv[:n], hidden) if is_fp8 else recv[:n]
        expert_out = reference.simulate_expert_compute(recv_x, recv_topk_idx, recv_topk_weights)
        # Deliberately the same magnitude as the combined output, so a dropped
        # or double-applied bias moves rel_l2 well past the threshold.
        biases = [torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device) for _ in range(num_bias)]
        combined, _ = buf.combine(expert_out, handle, bias=biases or None)

        expected = reference.reference_combined(x, topk_weights, topk_idx)
        for b in biases:
            expected = expected + b
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


# One `_run` per test, like every other case here: `_run` owns the process
# group's whole lifetime, so looping over parameters inside one test tears it
# down and rebuilds it on the same port, which is intermittently refused.
@tilelang.testing.requires_cuda
@distributed_test(nprocs=2)
def test_combine_bias_single(local_rank: int, num_ranks: int):
    """DeepEP's `bias_0`, on its own."""
    _run(local_rank, num_ranks, num_tokens=64, hidden=128, topk=2, num_experts=8, num_sms=2, num_bias=1)


@tilelang.testing.requires_cuda
@distributed_test(nprocs=2)
def test_combine_bias_pair(local_rank: int, num_ranks: int):
    """DeepEP's `bias_0` and `bias_1` together."""
    _run(local_rank, num_ranks, num_tokens=64, hidden=128, topk=2, num_experts=8, num_sms=2, num_bias=2)


@tilelang.testing.requires_cuda
@distributed_test(nprocs=8)
def test_combine_bias_masked(local_rank: int, num_ranks: int):
    """Bias with half the selections masked off: a token with no contributions
    at all must still come back as its bias, not as zero."""
    _run(local_rank, num_ranks, num_tokens=1024, hidden=1024, topk=8, num_experts=256, num_sms=32, masked_ratio=0.5, num_bias=2)


@tilelang.testing.requires_cuda
@distributed_test(nprocs=8)
def test_wide_scatter_grid(local_rank: int, num_ranks: int):
    """Dispatch's scatter runs on its own, wider grid than its notify.

    The two phases are separate launches, so the scatter is not bound by the
    persistent grid the notify needs; `scatter_sms` has to stride the token
    loop, the padding loop and the stats scan by the grid it actually runs on,
    and getting any of those wrong drops or double-writes rows.
    """
    _run(local_rank, num_ranks, num_tokens=1024, hidden=1024, topk=8, num_experts=256, num_sms=16, scatter_sms=64)


@tilelang.testing.requires_cuda
@distributed_test(nprocs=8)
def test_dispatch_combine_masked(local_rank: int, num_ranks: int):
    """Half the selections unset: dispatch must route -1 nowhere, and a token
    with no selections at all must combine back to zero."""
    _run(local_rank, num_ranks, num_tokens=1024, hidden=1024, topk=8, num_experts=256, num_sms=32, masked_ratio=0.5)


@tilelang.testing.requires_cuda
@distributed_test(nprocs=8)
def test_cumulative_local_expert_recv_stats(local_rank: int, num_ranks: int):
    """Per-expert receive counts, against an all-gathered torch reference.

    Three dispatches into the same counter, and a quarter of the top-k entries
    masked off, so the test covers the accumulate-don't-overwrite contract and
    the -1 entries the scan has to skip.
    """
    rank, num_ranks, group = init_dist(local_rank, num_ranks)
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(1234 + rank)

    num_tokens, hidden, topk, num_experts, num_calls = 512, 512, 4, 32, 3
    experts_per_rank = num_experts // num_ranks
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    topk_idx, topk_weights = reference.make_topk(num_tokens, topk, num_experts, device, 0.25)

    buf = Buffer(
        group=group,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        num_max_tokens_per_rank=num_tokens,
        hidden=hidden,
        num_topk=topk,
        num_experts=num_experts,
        dtype=torch.bfloat16,
        num_sms=8,
    )
    try:
        stats = torch.zeros(experts_per_rank, dtype=torch.uint32, device=device)
        for _ in range(num_calls):
            buf.dispatch(x, topk_idx, topk_weights, cumulative_local_expert_recv_stats=stats)

        # What every rank sent to each expert, summed -- my counters are the
        # columns belonging to my own experts.
        sent = torch.zeros(num_experts, dtype=torch.int64, device=device)
        selected = topk_idx[topk_idx >= 0]
        sent.scatter_add_(0, selected.long(), torch.ones_like(selected, dtype=torch.int64))
        per_rank = [torch.zeros_like(sent) for _ in range(num_ranks)]
        dist.all_gather(per_rank, sent, group)
        expected = torch.stack(per_rank).sum(0)[rank * experts_per_rank : (rank + 1) * experts_per_rank] * num_calls

        got = stats.to(torch.int64)
        assert torch.equal(got, expected), f"rank {rank}: got {got.tolist()}, expected {expected.tolist()}"
    finally:
        buf.close()
        dist.destroy_process_group()


@tilelang.testing.requires_cuda
@distributed_test(nprocs=8)
def test_dispatch_expanded_layout(local_rank: int, num_ranks: int):
    """DeepEP's `do_expand`: one row per (token, expert), grouped by expert.

    Checked against `reference.expanded_layout` -- per-expert counts, the
    aligned segment offsets, the exact set of (src_rank, src_token) in each
    segment, the payload of every row, and that the alignment padding is
    zeroed and marked unoccupied.
    """
    rank, num_ranks, group = init_dist(local_rank, num_ranks)
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(1234 + rank)

    num_tokens, hidden, topk, num_experts, alignment = 128, 256, 4, 32, 8
    experts_per_rank = num_experts // num_ranks
    # Distinct experts per token, which is what DeepEP asserts and real top-k
    # routing produces; -1 marks an unselected slot.
    idx = torch.stack([torch.randperm(num_experts, device=device)[:topk] for _ in range(num_tokens)]).int()
    idx = idx.masked_fill(torch.rand_like(idx, dtype=torch.float) < 0.25, -1)
    weights = torch.rand(num_tokens, topk, device=device)
    # A token's identity, so a misrouted row is obvious rather than plausible.
    x = torch.arange(num_tokens, device=device, dtype=torch.bfloat16).view(-1, 1).repeat(1, hidden)
    x = x + rank * 1000

    buf = Buffer(
        group=group,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        num_max_tokens_per_rank=num_tokens,
        hidden=hidden,
        num_topk=topk,
        num_experts=num_experts,
        dtype=torch.bfloat16,
        num_sms=8,
        do_expand=True,
        expert_alignment=alignment,
        expand_factor=float(min(topk, experts_per_rank)),
    )
    try:
        all_idx = [torch.zeros_like(idx) for _ in range(num_ranks)]
        dist.all_gather(all_idx, idx, group)
        exp_rows, exp_counts, exp_offsets = reference.expanded_layout([t.cpu() for t in all_idx], num_experts, num_ranks, alignment)

        recv_x, _, _, handle, _ = buf.dispatch(x, idx, weights)
        assert handle.expand_overflow == 0, f"rank {rank}: overflowed by {handle.expand_overflow}"

        counts = handle.expert_count.cpu().tolist()
        offsets = handle.expert_offset.cpu().tolist()
        assert counts == exp_counts[rank], f"rank {rank}: counts {counts} != {exp_counts[rank]}"
        assert offsets == exp_offsets[rank], f"rank {rank}: offsets {offsets} != {exp_offsets[rank]}"

        src_rank = buf.recv_src_rank.cpu()
        src_token = buf.recv_src_token.cpu()
        rows = recv_x.cpu()
        for e in range(experts_per_rank):
            begin, end = offsets[e], offsets[e] + counts[e]
            got = sorted(zip(src_rank[begin:end].tolist(), src_token[begin:end].tolist()))
            assert got == sorted(exp_rows[rank][e]), f"rank {rank} expert {e}: {got} != {sorted(exp_rows[rank][e])}"
            # Every row carries its origin, so the payload pins down routing.
            for i in range(begin, end):
                want = float(src_token[i]) + float(src_rank[i]) * 1000
                assert torch.all(rows[i] == want), f"rank {rank} row {i}: payload {rows[i][0]} != {want}"
            # Alignment padding: zeroed, and owned by nobody.
            for i in range(end, offsets[e + 1]):
                assert torch.all(rows[i] == 0), f"rank {rank} pad row {i} not zeroed"
                assert src_rank[i] == -1 and src_token[i] == -1, f"rank {rank} pad row {i} marked occupied"
    finally:
        buf.close()
        dist.destroy_process_group()


@tilelang.testing.requires_cuda
@distributed_test(nprocs=8)
def test_async_finish_round_trip(local_rank: int, num_ranks: int):
    """The same round trip run asynchronously must give the same answer.

    Both collectives run with `async_finish=True` and real work is enqueued on
    the compute stream inside the `with event:` block, so the test fails if the
    event does not actually order the two streams -- reading `recv_x` before
    the dispatch landed gives whatever the previous iteration left.

    `allocate_on_comm_stream=True` on the dispatch exercises the path that
    keeps this call's temporaries alive through the event rather than
    `record_stream`, and `previous_event` chains the combine behind that same
    event instead of behind the whole compute stream.
    """
    rank, num_ranks, group = init_dist(local_rank, num_ranks)
    device = f"cuda:{local_rank}"
    torch.manual_seed(1234 + rank)

    num_tokens, hidden, topk, num_experts = 512, 512, 4, 32
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    topk_idx, topk_weights = reference.make_topk(num_tokens, topk, num_experts, device, 0.25)

    buf = Buffer(
        group=group,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        num_max_tokens_per_rank=num_tokens,
        hidden=hidden,
        num_topk=topk,
        num_experts=num_experts,
        dtype=torch.bfloat16,
        num_sms=8,
        # Otherwise every third dispatch blocks the host, which is not what a
        # caller driving its own overlap asked for. See `Buffer.pipeline_depth`.
        pipeline_depth=0,
    )
    try:
        recv, recv_topk_idx, recv_topk_weights, handle, dispatch_event = buf.dispatch(
            x, topk_idx, topk_weights, async_finish=True, allocate_on_comm_stream=True
        )
        # Something real on the compute stream, overlapping the dispatch.
        filler = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
        with dispatch_event:
            filler = filler @ filler

        n = handle.num_recv_tokens
        expert_out = reference.simulate_expert_compute(recv[:n], recv_topk_idx[:n], recv_topk_weights[:n])
        combined, combine_event = buf.combine(
            expert_out, handle, previous_event=dispatch_event, async_finish=True, allocate_on_comm_stream=True
        )
        combine_event.current_stream_wait()

        expected = reference.reference_combined(x, topk_weights, topk_idx)
        err = (combined.float() - expected.float()).norm().item()
        denom = expected.float().norm().item()
        rel_l2 = err / denom if denom > 0 else err
        assert rel_l2 < _BF16_REL_L2_THRESHOLD, f"rank {rank}: rel_l2_error={rel_l2} exceeds {_BF16_REL_L2_THRESHOLD}"
        assert filler.isfinite().all(), f"rank {rank}: overlapped work was corrupted"
    finally:
        buf.close()
        dist.destroy_process_group()


@tilelang.testing.requires_cuda
@distributed_test(nprocs=2)
def test_event_overlap_is_returned_when_synchronous(local_rank: int, num_ranks: int):
    """Synchronous calls still return an `EventOverlap`, wrapping `None`.

    That is what lets a caller write `with event:` without knowing which mode
    it asked for, so it is part of the contract rather than an accident.
    """
    rank, num_ranks, group = init_dist(local_rank, num_ranks)
    device = f"cuda:{local_rank}"
    torch.manual_seed(1234 + rank)

    num_tokens, hidden, topk, num_experts = 64, 128, 2, 8
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    topk_idx, topk_weights = reference.make_topk(num_tokens, topk, num_experts, device)

    buf = Buffer(
        group=group,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        num_max_tokens_per_rank=num_tokens,
        hidden=hidden,
        num_topk=topk,
        num_experts=num_experts,
        dtype=torch.bfloat16,
        num_sms=2,
    )
    try:
        recv, recv_topk_idx, recv_topk_weights, handle, event = buf.dispatch(x, topk_idx, topk_weights)
        assert event.event is None
        with event:  # a no-op, but it must not raise
            pass
        n = handle.num_recv_tokens
        expert_out = reference.simulate_expert_compute(recv[:n], recv_topk_idx[:n], recv_topk_weights[:n])
        _, combine_event = buf.combine(expert_out, handle)
        assert combine_event.event is None
        with pytest.raises(AssertionError):
            combine_event.current_stream_wait()
    finally:
        buf.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    tilelang.testing.main()
