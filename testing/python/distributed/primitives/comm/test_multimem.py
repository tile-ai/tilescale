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

_N = 1001
_BLOCK_N = 256
_THREADS = 128
_TMA_NUM_RANKS = 4
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


def _multimem_ld_codegen_kernel(
    N: int,
    block_N: int,
    threads: int,
    *,
    offset: int = 0,
    reduce_op: T.MultimemReduceOp = T.MultimemReduceOp.ADD,
):
    @T.prim_func
    def main(mcast_buf: T.Tensor((N + offset,), T.float32), out: T.Tensor((N,), T.float32)):
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
            tmp = T.alloc_fragment((block_N,), T.float32)
            T.multimem_ld_reduce(
                mcast_buf[offset + bx * block_N : offset + (bx + 1) * block_N],
                tmp,
                reduce_op=reduce_op,
            )
            T.copy(tmp, out[bx * block_N : (bx + 1) * block_N])

    return main


def _multimem_low_base_alignment_kernel():
    @T.prim_func
    def main(mcast_ptr: T.handle, out: T.Tensor((8,), T.float32)):
        mcast_buf = T.match_buffer(mcast_ptr, (8,), dtype=T.float32, align=4)
        with T.Kernel(1, threads=1):
            tmp = T.alloc_fragment((8,), T.float32)
            T.multimem_ld_reduce(mcast_buf, tmp)
            T.copy(tmp, out)

    return main


def _multimem_write_codegen_kernel(mode: str, N: int, block_N: int, threads: int):
    if mode == "st":

        @T.prim_func
        def main(mcast_buf: T.Tensor((N,), T.float32)):
            with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
                tmp = T.alloc_fragment((block_N,), T.float32)
                for i in T.Parallel(block_N):
                    tmp[i] = T.cast(i + 1, T.float32)
                T.multimem_st(tmp, mcast_buf[bx * block_N : (bx + 1) * block_N])

    else:

        @T.prim_func
        def main(mcast_buf: T.Tensor((N,), T.float32)):
            with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
                tmp = T.alloc_fragment((block_N,), T.float32)
                for i in T.Parallel(block_N):
                    tmp[i] = T.cast(i + 1, T.float32)
                T.multimem_red(
                    tmp,
                    mcast_buf[bx * block_N : (bx + 1) * block_N],
                    reduce_op=T.MultimemReduceOp.ADD,
                )

    return main


def _multimem_mismatched_extent_kernel():
    @T.prim_func
    def main(mcast_buf: T.Tensor((8,), T.float32)):
        with T.Kernel(1, threads=1):
            tmp = T.alloc_fragment((4,), T.float32)
            T.multimem_ld_reduce(mcast_buf, tmp)

    return main


def _multimem_mismatched_rank_kernel():
    @T.prim_func
    def main(mcast_buf: T.Tensor((2, 4), T.float32)):
        with T.Kernel(1, threads=1):
            tmp = T.alloc_fragment((8,), T.float32)
            T.multimem_ld_reduce(mcast_buf, tmp)

    return main


def _multimem_unsupported_reduce_kernel():
    @T.prim_func
    def main(mcast_buf: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=1):
            tmp = T.alloc_fragment((1,), T.float32)
            T.multimem_ld_reduce(
                mcast_buf,
                tmp,
                reduce_op=T.MultimemReduceOp.MIN,
            )

    return main


def _multimem_unsupported_dtype_kernel():
    @T.prim_func
    def main(mcast_buf: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=1):
            tmp = T.alloc_fragment((1,), T.int32)
            T.multimem_ld_reduce(mcast_buf, tmp)

    return main


def _multimem_vector_element_dtype_kernel():
    @T.prim_func
    def main(mcast_buf: T.Tensor((1,), T.float32x2)):
        with T.Kernel(1, threads=1):
            tmp = T.alloc_fragment((1,), T.float32x2)
            T.multimem_ld_reduce(mcast_buf, tmp)

    return main


def _multimem_singleton_runtime_offset_kernel():
    @T.prim_func
    def main(
        mcast_buf: T.Tensor((1,), T.float32),
        out: T.Tensor((1,), T.float32),
    ):
        with T.Kernel(1, threads=1):
            rank = T.get_rank()
            tmp = T.alloc_fragment((1,), T.float32)
            T.multimem_ld_reduce(mcast_buf[rank], tmp)
            T.copy(tmp, out)

    return main


def _multimem_packed_tail_kernel():
    @T.prim_func
    def main(mcast_buf: T.Tensor((257,), T.float16)):
        with T.Kernel(2, threads=128) as bx:
            tmp = T.alloc_fragment((256,), T.float16)
            T.multimem_st(tmp, mcast_buf[bx * 256 : (bx + 1) * 256])

    return main


def _multimem_packed_odd_offset_kernel():
    @T.prim_func
    def main(mcast_buf: T.Tensor((5,), T.float16)):
        with T.Kernel(1, threads=1):
            tmp = T.alloc_fragment((4,), T.float16)
            T.multimem_st(tmp, mcast_buf[1:5])

    return main


def _multimem_packed_local_slice_kernel():
    @T.prim_func
    def main(mcast_buf: T.Tensor((4,), T.float16)):
        with T.Kernel(1, threads=1):
            tmp = T.alloc_fragment((8,), T.float16)
            T.multimem_st(tmp[2:6], mcast_buf)

    return main


def _multimem_packed_partial_2d_local_kernel():
    @T.prim_func
    def main(mcast_buf: T.Tensor((2, 2), T.float16)):
        with T.Kernel(1, threads=1):
            tmp = T.alloc_fragment((2, 4), T.float16)
            T.multimem_st(tmp[0:2, 0:2], mcast_buf)

    return main


def _multimem_packed_odd_row_stride_kernel():
    @T.prim_func
    def main(mcast_buf: T.Tensor((2, 3), T.float16)):
        with T.Kernel(1, threads=1):
            tmp = T.alloc_fragment((2, 2), T.float16)
            T.multimem_st(tmp, mcast_buf[0:2, 0:2])

    return main


def _multimem_packed_runtime_offset_kernel():
    @T.prim_func
    def main(mcast_buf: T.Tensor((32,), T.bfloat16)):
        with T.Kernel(1, threads=4):
            rank = T.get_rank()
            tmp = T.alloc_fragment((8,), T.bfloat16)
            T.multimem_ld_reduce(mcast_buf[rank * 8 : (rank + 1) * 8], tmp)
            T.multimem_st(tmp, mcast_buf[rank * 8 : (rank + 1) * 8])

    return main


def _multimem_packed_partial_overlap_kernel():
    @T.prim_func
    def main(mcast_buf: T.Tensor((32,), T.bfloat16)):
        with T.Kernel(1, threads=4):
            rank = T.get_rank()
            tmp = T.alloc_fragment((8,), T.bfloat16)
            T.multimem_st(tmp, mcast_buf[rank * 10 : rank * 10 + 8])

    return main


def _multimem_packed_aligned_overlaunch_kernel():
    @T.prim_func
    def main(mcast_buf: T.Tensor((8,), T.bfloat16)):
        with T.Kernel(2, threads=4) as bx:
            tmp = T.alloc_fragment((8,), T.bfloat16)
            T.multimem_ld_reduce(mcast_buf[bx * 8 : (bx + 1) * 8], tmp)
            T.multimem_st(tmp, mcast_buf[bx * 8 : (bx + 1) * 8])

    return main


def _multimem_packed_incompatible_layout_kernel():
    @T.prim_func
    def main(
        mcast_buf: T.Tensor((8,), T.bfloat16),
        out: T.Tensor((8,), T.bfloat16),
    ):
        with T.Kernel(1, threads=4):
            tmp = T.alloc_fragment((8,), T.bfloat16)
            T.annotate_layout(
                {
                    tmp: T.Fragment(
                        (8,),
                        forward_fn=lambda i: (3 - i // 2, i % 2),
                    )
                }
            )
            T.multimem_ld_reduce(mcast_buf, tmp)
            T.copy(tmp, out)

    return main


def _multimem_packed_extra_replicate_layout_kernel():
    @T.prim_func
    def main(mcast_buf: T.Tensor((8,), T.bfloat16)):
        with T.Kernel(1, threads=4):
            tmp = T.alloc_fragment((8,), T.bfloat16)
            T.annotate_layout(
                {
                    tmp: T.Fragment(
                        (8,),
                        forward_thread_fn=lambda i, rep: i // 2,
                        forward_index_fn=lambda i: i % 2,
                        replicate=4,
                    )
                }
            )
            T.multimem_ld_reduce(mcast_buf, tmp)

    return main


def _multimem_packed_unbound_thread_range_layout_kernel():
    @T.prim_func
    def main(mcast_buf: T.Tensor((8,), T.bfloat16)):
        with T.Kernel(1, threads=4):
            tmp = T.alloc_fragment((8,), T.bfloat16)
            T.annotate_layout(
                {
                    tmp: T.Fragment(
                        (8,),
                        forward_thread_fn=lambda i, rep: i // 2,
                        forward_index_fn=lambda i: i % 2,
                        replicate=2,
                    )
                }
            )
            T.multimem_ld_reduce(mcast_buf, tmp)

    return main


def _multimem_packed_low_base_alignment_kernel():
    @T.prim_func
    def main(mcast_ptr: T.handle):
        mcast_buf = T.match_buffer(mcast_ptr, (8,), dtype=T.bfloat16, align=2)
        with T.Kernel(1, threads=4):
            tmp = T.alloc_fragment((8,), T.bfloat16)
            T.annotate_layout(
                {
                    tmp: T.Fragment(
                        (8,),
                        forward_thread_fn=lambda i, rep: i // 2,
                        forward_index_fn=lambda i: i % 2,
                        replicate=2,
                    )
                }
            )
            T.multimem_ld_reduce(mcast_buf, tmp)

    return main


def _compile_multimem_source(kernel) -> str:
    compiled = tilelang.compile(
        kernel,
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True},
    )
    return compiled.get_kernel_source()


@pytest.mark.parametrize("N", [1, 3, 5])
def test_multimem_scalar_width_codegen(N: int):
    source = _compile_multimem_source(_multimem_ld_codegen_kernel(N, N, 1))
    assert source.count("tl::multimem::LdReduceV1") == 1
    assert "tl::multimem::LdReduceV2" not in source
    assert "tl::multimem::LdReduceV4" not in source


def test_multimem_predicated_tail_codegen():
    ld_source = _compile_multimem_source(_multimem_ld_codegen_kernel(1001, 256, 128))
    st_source = _compile_multimem_source(_multimem_write_codegen_kernel("st", 257, 256, 128))
    red_source = _compile_multimem_source(_multimem_write_codegen_kernel("red", 257, 256, 128))

    assert "tl::multimem::LdReduceV2" in ld_source
    assert "tl::multimem::LdReduceV1" in ld_source
    assert "tl::multimem::StV2" in st_source
    assert "tl::multimem::StV1" in st_source
    assert "tl::multimem::RedV2" in red_source
    assert "tl::multimem::RedV1" in red_source
    assert "< 1001)" in ld_source
    assert "((int)blockIdx.x) < 1" in st_source
    assert "((int)blockIdx.x) < 1" in red_source


def test_multimem_width_and_alignment_codegen():
    wide_source = _compile_multimem_source(_multimem_ld_codegen_kernel(8, 8, 1))
    scalar_source = _compile_multimem_source(_multimem_ld_codegen_kernel(8, 8, 1, offset=1))
    low_base_alignment_source = _compile_multimem_source(_multimem_low_base_alignment_kernel())

    assert wide_source.count("tl::multimem::LdReduceV4") == 2
    assert "tl::multimem::LdReduceV1" not in wide_source
    assert "tl::multimem::LdReduceV2" not in wide_source
    assert scalar_source.count("tl::multimem::LdReduceV1") == 1
    assert "tl::multimem::LdReduceV2" not in scalar_source
    assert "tl::multimem::LdReduceV4" not in scalar_source
    assert low_base_alignment_source.count("tl::multimem::LdReduceV1") == 8
    assert "tl::multimem::LdReduceV2" not in low_base_alignment_source
    assert "tl::multimem::LdReduceV4" not in low_base_alignment_source


_INVALID_DIRECT_CASES = (
    pytest.param(
        _multimem_mismatched_extent_kernel,
        r"matching source and destination extents",
        id="extent",
    ),
    pytest.param(
        _multimem_mismatched_rank_kernel,
        r"regions with matching rank",
        id="rank",
    ),
    pytest.param(
        _multimem_unsupported_reduce_kernel,
        r"supports ADD only",
        id="reduce-op",
    ),
    pytest.param(
        _multimem_unsupported_dtype_kernel,
        r"scalar float32, float16, or bfloat16",
        id="dtype",
    ),
    pytest.param(
        _multimem_vector_element_dtype_kernel,
        r"scalar float32, float16, or bfloat16",
        id="vector-element-dtype",
    ),
)


@pytest.mark.parametrize("kernel_factory,error", _INVALID_DIRECT_CASES)
def test_multimem_rejects_invalid_direct_contract(kernel_factory, error: str):
    with pytest.raises(RuntimeError, match=error):
        _compile_multimem_source(kernel_factory())


def test_multimem_singleton_runtime_bounds_codegen():
    source = _compile_multimem_source(_multimem_singleton_runtime_offset_kernel())
    assert "tl::get_rank()" in source
    assert "tl::multimem::LdReduceV1" in source
    assert "} else {\n    tmp[0] = 0x0p+0f" in source


_INVALID_PACKED_REGION_CASES = (
    pytest.param(
        _multimem_packed_tail_kernel,
        r"packed multicast.*in bounds",
        id="partial-tail",
    ),
    pytest.param(
        _multimem_packed_odd_offset_kernel,
        r"4-byte-aligned multicast start address",
        id="start-alignment",
    ),
    pytest.param(
        _multimem_packed_local_slice_kernel,
        r"local regions to start at zero",
        id="local-start",
    ),
    pytest.param(
        _multimem_packed_partial_2d_local_kernel,
        r"local region to cover the entire fragment buffer",
        id="local-coverage",
    ),
    pytest.param(
        _multimem_packed_odd_row_stride_kernel,
        r"even physical stride",
        id="row-stride",
    ),
    pytest.param(
        _multimem_packed_partial_overlap_kernel,
        r"packed multicast.*(partial|bounds|partition)",
        id="dynamic-partial-overlap",
    ),
    pytest.param(
        _multimem_packed_low_base_alignment_kernel,
        r"packed multicast.*base alignment.*at least 4 bytes",
        id="base-alignment",
    ),
)


@pytest.mark.parametrize("kernel_factory,error", _INVALID_PACKED_REGION_CASES)
def test_multimem_rejects_invalid_packed_region(kernel_factory, error: str):
    with pytest.raises(RuntimeError, match=error):
        _compile_multimem_source(kernel_factory())


def test_multimem_packed_runtime_bounds_codegen():
    source = _compile_multimem_source(_multimem_packed_runtime_offset_kernel())
    assert "tl::get_rank()" in source
    assert "if ((0 <= rank) && (rank <= 3))" in source
    assert "tl::multimem::LdReduceV2" in source
    assert source.count("bfloat16_t(0x0p+0f") == 2


def test_multimem_packed_aligned_overlaunch_codegen():
    source = _compile_multimem_source(_multimem_packed_aligned_overlaunch_kernel())
    assert "if (((int)blockIdx.x) == 0)" in source
    assert "tl::multimem::LdReduceV2" in source
    assert source.count("bfloat16_t(0x0p+0f") == 2


_INVALID_PACKED_LAYOUT_CASES = (
    pytest.param(
        _multimem_packed_incompatible_layout_kernel,
        r"packed x2.*canonical pair ownership",
        id="pair-ownership",
    ),
    pytest.param(
        _multimem_packed_extra_replicate_layout_kernel,
        r"packed x2.*replicate",
        id="replicate-extent",
    ),
)


@pytest.mark.parametrize("kernel_factory,error", _INVALID_PACKED_LAYOUT_CASES)
def test_multimem_rejects_invalid_packed_layout(kernel_factory, error: str):
    with pytest.raises(RuntimeError, match=error):
        _compile_multimem_source(kernel_factory())


def test_multimem_accepts_canonical_unbound_thread_range_layout():
    source = _compile_multimem_source(_multimem_packed_unbound_thread_range_layout_kernel())
    assert "tl::multimem::LdReduceV2" in source


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


def _multimem_tma_reduce_kernel(
    N: int,
    threads: int,
    dtype,
    reduce_op: T.MultimemReduceOp,
):
    @T.prim_func
    def main(src: T.Tensor((N,), dtype), mcast_buf: T.Tensor((N,), dtype)):
        with T.Kernel(1, threads=threads):
            tx = T.get_thread_binding()
            shard = T.alloc_shared((N,), dtype)
            T.copy(src, shard, disable_tma=True)
            T.fence_proxy_async()
            T.sync_threads()
            if tx == 0:
                T.multimem_tma_store(shard, mcast_buf, reduce_op=reduce_op)
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
@distributed_test(nprocs=4, require_multicast=True)
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
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
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


@pytest.mark.skipif(not _has_cuda_toolkit_13_1(), reason="Requires CUDA Toolkit 13.1+")
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@distributed_test(nprocs=_TMA_NUM_RANKS, require_multicast=True)
def test_multimem_tma_reduce_dtype_matrix(local_rank: int, num_ranks: int):
    """Run every exposed floating-point bulk reduction on physical backings."""
    from tilelang.distributed.host import init_dist

    rank, _, group = init_dist(local_rank, num_ranks)
    allocator = None
    try:
        cases = (
            (T.float16, torch.float16, T.MultimemReduceOp.ADD, "add_f16"),
            (T.float16, torch.float16, T.MultimemReduceOp.MIN, "min_f16"),
            (T.float16, torch.float16, T.MultimemReduceOp.MAX, "max_f16"),
            (T.bfloat16, torch.bfloat16, T.MultimemReduceOp.ADD, "add_bf16"),
            (T.bfloat16, torch.bfloat16, T.MultimemReduceOp.MIN, "min_bf16"),
            (T.bfloat16, torch.bfloat16, T.MultimemReduceOp.MAX, "max_bf16"),
        )
        allocator = tilelang.get_allocator(
            size=2**22,
            device=f"cuda:{local_rank}",
            is_distributed=True,
            local_rank=local_rank,
            num_local_ranks=num_ranks,
            group=group,
            use_vmm=True,
            mcast_size=len(cases) * _TMA_N * 2 + 4096,
        )
        assert allocator._use_vmm
        assert allocator._use_multicast

        pass_configs = {
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        }
        index = torch.arange(_TMA_N, dtype=torch.float32, device=f"cuda:{local_rank}")
        base = index.remainder(8)

        for tl_dtype, torch_dtype, reduce_op, helper_suffix in cases:
            kernel = tilelang.compile(
                _multimem_tma_reduce_kernel(_TMA_N, _THREADS, tl_dtype, reduce_op),
                pass_configs=pass_configs,
                compile_once=True,
                compile_group=group,
            )
            kernel.initialize(allocator=allocator)
            assert f"tl::multimem::cp_reduce_async_bulk_{helper_suffix}" in kernel.get_kernel_source()

            mcast_buf, local_backing = allocator._allocate_mcast_tensor((_TMA_N,), torch_dtype)
            src = (base + float(rank * 2)).to(torch_dtype)
            if reduce_op == T.MultimemReduceOp.ADD:
                local_backing.zero_()
                expected = base * num_ranks + float(num_ranks * (num_ranks - 1))
            elif reduce_op == T.MultimemReduceOp.MIN:
                local_backing.fill_(100)
                expected = base
            else:
                local_backing.fill_(-100)
                expected = base + float((num_ranks - 1) * 2)

            _synchronize_ranks(group)
            kernel(src, mcast_buf)
            _synchronize_ranks(group)
            _assert_all_ranks_equal(
                local_backing,
                expected.to(torch_dtype),
                group,
                helper_suffix,
            )
    finally:
        try:
            if allocator is not None:
                allocator.close()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    import tilelang.testing

    tilelang.testing.main()
