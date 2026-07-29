"""Tests for distributed put/get primitives (block and warp granularity).

All four ops are tested in a single spawn session to avoid paying per-test
import / NCCL setup overhead (~10s per spawn).  pytest -k can select individual
ops via the test names reported by the worker.

CI configuration: 4 GPUs, compute >= 9.0.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T
import tilelang.testing
from testing.python.distributed._utils import distributed_test

os.environ.setdefault("NCCL_DEBUG", "WARN")

_M = 65536
_BLOCK_M = 4096
_THREADS = 128


# ---------------------------------------------------------------------------
# Kernel definitions
# ---------------------------------------------------------------------------


def _kernel_get_block(M, num_rank, block_M, threads):
    @T.prim_func
    def main(dst: T.Tensor((M), "float32"), src: T.Tensor((M), "float32")):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as (bx):
            rank = T.alloc_local([1], "uint64")
            rank[0] = T.get_rank()
            T.get_block(
                src=T.address_of(src[bx * block_M]),
                dst=T.address_of(dst[bx * block_M]),
                size=block_M,
                src_pe=rank[0] ^ 1,
            )
            T.fence_sys()

    return main


def _kernel_get_warp(M, num_rank, block_M, threads):
    @T.prim_func
    def main(dst: T.Tensor((M), "float32"), src: T.Tensor((M), "float32")):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as (bx):
            rank = T.alloc_local([1], "uint64")
            rank[0] = T.get_rank()
            warp_idx = T.get_thread_binding(0) // 32
            warp_copy_size = T.ceildiv(block_M, threads // 32)
            warp_start = bx * block_M + warp_copy_size * warp_idx
            T.get_warp(
                src=T.address_of(src[warp_start]),
                dst=T.address_of(dst[warp_start]),
                size=warp_copy_size,
                src_pe=rank[0] ^ 1,
                unroll_factor=4,
            )
            T.fence_sys()

    return main


def _kernel_put_block(M, num_rank, block_M, threads):
    @T.prim_func
    def main(dst: T.Tensor((M), "float32"), src: T.Tensor((M), "float32")):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as (bx):
            rank = T.alloc_local([1], "uint64")
            rank[0] = T.get_rank()
            T.put_block(
                src=T.address_of(src[bx * block_M]),
                dst=T.address_of(dst[bx * block_M]),
                size=block_M,
                dst_pe=rank[0] ^ 1,
            )

    return main


def _kernel_put_warp(M, num_rank, block_M, threads):
    @T.prim_func
    def main(dst: T.Tensor((M), "bfloat16"), src: T.Tensor((M), "bfloat16")):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as (bx):
            rank = T.alloc_local([1], "uint64")
            rank[0] = T.get_rank()
            warp_idx = T.get_thread_binding(0) // 32
            warp_copy_size = T.ceildiv(block_M, threads // 32)
            warp_start = bx * block_M + warp_copy_size * warp_idx
            T.put_warp(
                src=T.address_of(src[warp_start]),
                dst=T.address_of(dst[warp_start]),
                size=warp_copy_size,
                dst_pe=rank[0] ^ 1,
                unroll_factor=4,
            )

    return main


# Pre-compile all kernels at import time (warms disk cache for children)
_OP_NAMES = ["get_block", "get_warp", "put_block", "put_warp"]
_KERNEL_FNS = [_kernel_get_block, _kernel_get_warp, _kernel_put_block, _kernel_put_warp]

for _fn in _KERNEL_FNS:
    tilelang.compile(_fn(_M, 4, _BLOCK_M, _THREADS))


# ---------------------------------------------------------------------------
# Worker: runs all four kernel tests in one spawn session
# ---------------------------------------------------------------------------


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@distributed_test(nprocs=4)
def test_put_get(local_rank: int, num_ranks: int):
    from tilelang.distributed.host import init_dist

    rank, num_ranks, group = init_dist(local_rank, num_ranks)

    allocator = tilelang.get_allocator(
        size=2**25,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group,
    )

    for name, kernel_fn in zip(_OP_NAMES, _KERNEL_FNS):
        kernel = tilelang.compile(
            kernel_fn(_M, num_ranks, _BLOCK_M, _THREADS),
            compile_once=True,
            compile_group=group,
        )
        kernel.initialize(allocator=allocator)

        dtype = torch.bfloat16 if name == "put_warp" else torch.float32
        src = tilelang.tensor((_M,), dtype, allocator=allocator).normal_()
        dst = tilelang.tensor((_M,), dtype, allocator=allocator)

        torch.cuda.synchronize()
        dist.barrier(group)
        kernel(dst, src)
        torch.cuda.synchronize()
        dist.barrier(group)

        dst_refs = [torch.empty_like(src) for _ in range(num_ranks)]
        dist.all_gather(dst_refs, src, group)
        expected = dst_refs[local_rank ^ 1]

        assert torch.allclose(expected, dst, atol=1e-6, rtol=1e-6), f"rank {local_rank}: {name} mismatch"

    allocator.close()
    dist.destroy_process_group()


def _dynamic_size_put_kernel():
    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32"), n: T.int32):
        with T.Kernel(1, threads=32):
            rank = T.alloc_local((1,), "uint64")
            rank[0] = T.get_rank()
            T.put_warp(T.address_of(A[0]), T.address_of(B[0]), n, dst_pe=rank[0] ^ 1)

    return main


def _dynamic_size_get_kernel():
    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32"), n: T.int32):
        with T.Kernel(1, threads=32):
            rank = T.alloc_local((1,), "uint64")
            rank[0] = T.get_rank()
            T.get_warp(T.address_of(A[0]), T.address_of(B[0]), n, src_pe=rank[0] ^ 1)

    return main


def _constant_size_put_kernel():
    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32")):
        with T.Kernel(1, threads=32):
            rank = T.alloc_local((1,), "uint64")
            rank[0] = T.get_rank()
            T.put_warp(T.address_of(A[0]), T.address_of(B[0]), 128, dst_pe=rank[0] ^ 1)

    return main


@pytest.mark.parametrize("kernel_factory", [_dynamic_size_put_kernel, _dynamic_size_get_kernel])
def test_put_get_rejects_dynamic_size(kernel_factory):
    """The copy size becomes a template argument, so it must be a constant.

    A runtime size used to be emitted verbatim as `tl::cp_warp<size, ...>`, whose
    only diagnostic was an nvcc template error.
    """
    with pytest.raises(Exception, match="compile-time constant size"):
        tilelang.compile(kernel_factory())


def test_put_accepts_constant_size():
    source = tilelang.compile(_constant_size_put_kernel()).get_kernel_source()
    assert "tl::cp_warp<128," in source


def _peer_none_kernel():
    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32"), flag: T.Tensor((1,), "uint32")):
        with T.Kernel(1, threads=32):
            v = T.alloc_local((1,), "float32")
            T.ld(A[0], v[0], src_pe=None)
            T.st(B[0], v[0], dst_pe=None)
            T.put_warp(T.address_of(A[0]), T.address_of(B[0]), 64, dst_pe=None)
            T.get_warp(T.address_of(A[0]), T.address_of(B[0]), 64, src_pe=None)
            T.put_block(T.address_of(A[0]), T.address_of(B[0]), 64, dst_pe=None)
            T.get_block(T.address_of(A[0]), T.address_of(B[0]), 64, src_pe=None)
            T.wait_eq(flag[0], 1, None)

    return main


def test_peer_none_means_local():
    """The signatures advertise ``| None``, so None must behave like the -1 sentinel.

    It previously reached call_intrin unchanged and failed with an opaque error.
    """
    source = tilelang.compile(_peer_none_kernel()).get_kernel_source()
    assert "get_remote_base_ptr" not in source


if __name__ == "__main__":
    import tilelang.testing

    tilelang.testing.main()
