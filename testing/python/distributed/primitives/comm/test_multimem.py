"""Small multimem correctness tests."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.contrib import nvcc
from testing.python.distributed._utils import distributed_test

os.environ.setdefault("NCCL_DEBUG", "WARN")

_N = 1024
_BLOCK_N = 256
_THREADS = 128
_TMA_NUM_RANKS = 2
_TMA_SHARD_N = 256
_TMA_N = _TMA_NUM_RANKS * _TMA_SHARD_N


def _has_cuda_toolkit_13_1() -> bool:
    try:
        return nvcc.get_cuda_version() >= (13, 1)
    except (OSError, RuntimeError, ValueError):
        return False


def _multimem_allreduce_kernel(N: int, block_N: int, threads: int):
    @T.prim_func
    def main(mcast_buf: T.Tensor((N,), T.float32), out: T.Tensor((N,), T.float32)):
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
            tmp = T.alloc_fragment((block_N,), T.float32)
            T.multimem_ld_reduce(
                mcast_buf[bx * block_N : (bx + 1) * block_N],
                tmp,
                reduce_op=T.MultimemReduceOp.ADD,
            )
            T.copy(tmp, out[bx * block_N : (bx + 1) * block_N])

    return main


def _multimem_tma_broadcast_kernel(shard_N: int, threads: int):
    @T.prim_func
    def main(mcast_buf: T.Tensor((_TMA_N,), T.float32)):
        with T.Kernel(1, threads=threads):
            rank = T.get_rank()
            tx = T.get_thread_binding()
            shard = T.alloc_shared((shard_N,), T.float32)
            for i in T.Parallel(shard_N):
                shard[i] = T.cast(rank * shard_N + i + 1, T.float32)
            T.fence_proxy_async()
            T.sync_threads()
            if tx == 0:
                T.multimem_tma_store(
                    shard,
                    mcast_buf[rank * shard_N : (rank + 1) * shard_N],
                )
                T.tma_store_arrive()
                T.tma_store_wait(0, False)

    return main


def _multimem_tma_add_kernel(N: int, threads: int):
    @T.prim_func
    def main(src: T.Tensor((N,), T.float32), mcast_buf: T.Tensor((N,), T.float32)):
        with T.Kernel(1, threads=threads):
            tx = T.get_thread_binding()
            shard = T.alloc_shared((N,), T.float32)
            T.copy(src, shard, disable_tma=True)
            T.fence_proxy_async()
            T.sync_threads()
            if tx == 0:
                T.multimem_tma_store(
                    shard,
                    mcast_buf,
                    reduce_op=T.MultimemReduceOp.ADD,
                )
                T.tma_store_arrive()
                T.tma_store_wait(0, False)

    return main


def _assert_multimem_tma_codegen(source: str, helper: str) -> None:
    ordered_calls = (
        "tl::fence_proxy_async();",
        "__syncthreads();",
        helper,
        "tl::tma_store_arrive();",
        "tl::tma_store_wait<0, false>();",
    )
    call_positions = [source.rfind(call) for call in ordered_calls]
    assert all(position >= 0 for position in call_positions)
    assert call_positions == sorted(call_positions)


def _synchronize_ranks(group) -> None:
    torch.cuda.synchronize()
    dist.barrier(group)


def _assert_all_ranks_equal(actual: torch.Tensor, expected: torch.Tensor, group, label: str) -> None:
    failure = None
    if not torch.equal(actual, expected):
        max_diff = (actual - expected).abs().max().item()
        failure = f"rank {dist.get_rank(group)} {label}: max_abs_diff={max_diff}"

    failures = [None] * dist.get_world_size(group)
    dist.all_gather_object(failures, failure, group=group)
    failures = [item for item in failures if item is not None]
    assert not failures, "; ".join(failures)


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@distributed_test(nprocs=None, require_multicast=True)
def test_multimem(local_rank: int, num_ranks: int):
    from tilelang.distributed.host import init_dist

    rank, _, group = init_dist(local_rank, num_ranks)
    allocator = tilelang.get_allocator(
        size=2**22,
        device=f"cuda:{local_rank}",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group,
        mcast_size=_N * torch.empty((), dtype=torch.float32).element_size(),
    )

    kernel = tilelang.compile(
        _multimem_allreduce_kernel(_N, _BLOCK_N, _THREADS),
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True},
        compile_once=True,
        compile_group=group,
    )
    kernel.initialize(allocator=allocator)

    torch.manual_seed(100 + rank)
    local_data = torch.randn(_N, dtype=torch.float32, device=f"cuda:{local_rank}")
    mcast_buf, local_buf = allocator._allocate_mcast_tensor((_N,), torch.float32)
    local_buf.copy_(local_data)
    out = tilelang.tensor((_N,), torch.float32, allocator=allocator).zero_()

    torch.cuda.synchronize()
    dist.barrier(group)
    kernel(mcast_buf, out)
    torch.cuda.synchronize()
    dist.barrier(group)

    expected = local_data.clone()
    dist.all_reduce(expected, op=dist.ReduceOp.SUM, group=group)
    assert torch.allclose(expected, out, atol=1e-5, rtol=1e-5)

    allocator.close()
    dist.destroy_process_group()


@pytest.mark.skipif(not _has_cuda_toolkit_13_1(), reason="Requires CUDA Toolkit 13.1+")
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
@distributed_test(nprocs=_TMA_NUM_RANKS, require_multicast=True)
def test_multimem_tma_store_plain_and_add(local_rank: int, num_ranks: int):
    """Exercise native multimem bulk broadcast and ADD on physical backings."""
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
            use_vmm=True,
            mcast_size=_TMA_N * torch.empty((), dtype=torch.float32).element_size(),
        )
        assert allocator._use_vmm
        assert allocator._use_multicast

        pass_configs = {
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        }
        broadcast = tilelang.compile(
            _multimem_tma_broadcast_kernel(_TMA_SHARD_N, _THREADS),
            pass_configs=pass_configs,
            compile_once=True,
            compile_group=group,
        )
        add = tilelang.compile(
            _multimem_tma_add_kernel(_TMA_N, _THREADS),
            pass_configs=pass_configs,
            compile_once=True,
            compile_group=group,
        )
        broadcast.initialize(allocator=allocator)
        add.initialize(allocator=allocator)
        _assert_multimem_tma_codegen(
            broadcast.get_kernel_source(),
            "tl::multimem::cp_async_bulk",
        )
        _assert_multimem_tma_codegen(
            add.get_kernel_source(),
            "tl::multimem::cp_reduce_async_bulk_add_f32",
        )

        mcast_buf, local_backing = allocator._allocate_mcast_tensor((_TMA_N,), torch.float32)
        local_backing.fill_(-1)
        _synchronize_ranks(group)
        broadcast(mcast_buf)
        _synchronize_ranks(group)

        expected_broadcast = torch.arange(
            1,
            _TMA_N + 1,
            dtype=torch.float32,
            device=f"cuda:{local_rank}",
        )
        _assert_all_ranks_equal(local_backing, expected_broadcast, group, "plain broadcast")

        index = torch.arange(_TMA_N, dtype=torch.float32, device=f"cuda:{local_rank}")
        src = index * 0.25 + float(rank + 1)
        expected_add = index * (0.25 * num_ranks) + float(num_ranks * (num_ranks + 1) // 2)

        sentinel = float(10 * (rank + 1))
        local_backing.fill_(sentinel)
        _synchronize_ranks(group)
        add(src, mcast_buf)
        _synchronize_ranks(group)
        _assert_all_ranks_equal(local_backing, expected_add + sentinel, group, "ADD with sentinel")

        local_backing.zero_()
        _synchronize_ranks(group)
        add(src, mcast_buf)
        _synchronize_ranks(group)
        _assert_all_ranks_equal(local_backing, expected_add, group, "ADD after zero fill")
    finally:
        try:
            if allocator is not None:
                allocator.close()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    import tilelang.testing

    tilelang.testing.main()
