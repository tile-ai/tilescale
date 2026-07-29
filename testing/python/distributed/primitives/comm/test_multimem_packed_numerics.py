"""Numerical coverage for the direct packed 16-bit multimem instructions.

The rest of the multimem suite covers these paths through codegen and rejection
asserts, and `test_multimem` itself runs in fp32, so the packed f16x2/bf16x2
specializations in src/tl_templates/cuda/distributed/multimem.h had no
device-level numerical check. These cases close that gap for
LdReduceV2/StV2 (two-shot all-reduce) and RedV2 (reduce-into-multicast).

Inputs are small integers so every partial sum is exactly representable in bf16
and fp16, which lets the assertions compare exactly instead of with a tolerance
that could mask a coherence failure.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T
import tilelang.testing

from testing.python.distributed._utils import distributed_test

_NUM_RANKS = 4
_SHARD = 256
_N = _NUM_RANKS * _SHARD
_TILE = 256
_THREADS = 128


def _two_shot_kernel(dtype):
    """Each rank load-reduces its own shard, then broadcasts it to every peer."""

    @T.prim_func
    def main(mcast_buf: T.Tensor((_N,), dtype)):
        with T.Kernel(_SHARD // _TILE, threads=_THREADS) as bx:
            rank = T.get_rank()
            offset = rank * _SHARD + bx * _TILE
            acc = T.alloc_fragment((_TILE,), dtype)
            T.multimem_ld_reduce(mcast_buf[offset : offset + _TILE], acc)
            T.multimem_st(acc, mcast_buf[offset : offset + _TILE])

    return main


def _red_kernel(dtype):
    """Every rank reduces its whole contribution into every peer's copy.

    multimem.red is an atomic reduction, so all ranks can target the same range
    concurrently and the buffer ends up holding the sum over ranks.
    """

    @T.prim_func
    def main(mcast_buf: T.Tensor((_N,), dtype), src: T.Tensor((_N,), dtype)):
        with T.Kernel(_N // _TILE, threads=_THREADS) as bx:
            offset = bx * _TILE
            acc = T.alloc_fragment((_TILE,), dtype)
            T.copy(src[offset : offset + _TILE], acc)
            T.multimem_red(acc, mcast_buf[offset : offset + _TILE])

    return main


def _exact_pattern(rank: int, torch_dtype, device):
    # Values 0..7 shifted by rank keep every partial sum <= 34, which bf16 and
    # fp16 both represent exactly.
    return ((torch.arange(_N, device=device) % 8) + rank).to(torch_dtype)


def _run(local_rank: int, num_ranks: int, torch_dtype, tl_dtype, mode: str):
    from tilelang.distributed.host import init_dist

    rank, _, group = init_dist(local_rank, num_ranks)
    allocator = None
    try:
        allocator = tilelang.get_allocator(
            size=2**22,
            device=f"cuda:{local_rank}",
            is_distributed=True,
            local_rank=local_rank,
            num_local_ranks=num_ranks,
            group=group,
            mcast_size=_N * torch.empty((), dtype=torch_dtype).element_size(),
        )

        factory = _two_shot_kernel if mode == "two_shot" else _red_kernel
        kernel = tilelang.compile(
            factory(tl_dtype),
            pass_configs={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True},
            compile_once=True,
            compile_group=group,
        )
        kernel.initialize(allocator=allocator)

        device = f"cuda:{local_rank}"
        local_data = _exact_pattern(rank, torch_dtype, device)
        mcast_buf, local_buf = allocator._allocate_mcast_tensor((_N,), torch_dtype)

        if mode == "two_shot":
            local_buf.copy_(local_data)
            args = (mcast_buf,)
        else:
            # Accumulate from zero so the result is exactly the rank sum.
            local_buf.zero_()
            src = tilelang.tensor((_N,), torch_dtype, allocator=allocator)
            src.copy_(local_data)
            args = (mcast_buf, src)

        torch.cuda.synchronize()
        dist.barrier(group)
        kernel(*args)
        torch.cuda.synchronize()
        dist.barrier(group)

        expected = local_data.to(torch.float32)
        dist.all_reduce(expected, op=dist.ReduceOp.SUM, group=group)
        actual = local_buf.to(torch.float32)

        wrong = int((actual != expected).sum())
        assert wrong == 0, (
            f"rank {rank} {mode}/{torch_dtype}: {wrong}/{_N} elements differ; "
            f"got {actual[actual != expected][:8].tolist()}, "
            f"expected {expected[actual != expected][:8].tolist()}"
        )
    finally:
        if allocator is not None:
            allocator.close()
        dist.destroy_process_group()


# distributed_test looks each worker up by (module, qualname) after re-importing
# this module in the spawned child, so every case must be a module-level
# function rather than a parametrized closure.


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@distributed_test(nprocs=_NUM_RANKS, require_multicast=True)
def test_multimem_packed_two_shot_bf16(local_rank: int, num_ranks: int):
    """LdReduceV2 + StV2 in bf16 must all-reduce exactly."""
    _run(local_rank, num_ranks, torch.bfloat16, T.bfloat16, "two_shot")


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@distributed_test(nprocs=_NUM_RANKS, require_multicast=True)
def test_multimem_packed_two_shot_fp16(local_rank: int, num_ranks: int):
    """LdReduceV2 + StV2 in fp16 must all-reduce exactly."""
    _run(local_rank, num_ranks, torch.float16, T.float16, "two_shot")


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@distributed_test(nprocs=_NUM_RANKS, require_multicast=True)
def test_multimem_packed_red_bf16(local_rank: int, num_ranks: int):
    """RedV2 in bf16 must accumulate exactly on every peer."""
    _run(local_rank, num_ranks, torch.bfloat16, T.bfloat16, "red")


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@distributed_test(nprocs=_NUM_RANKS, require_multicast=True)
def test_multimem_packed_red_fp16(local_rank: int, num_ranks: int):
    """RedV2 in fp16 must accumulate exactly on every peer."""
    _run(local_rank, num_ranks, torch.float16, T.float16, "red")


if __name__ == "__main__":
    tilelang.testing.main()
