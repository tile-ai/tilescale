"""GIN put/signal lowering and single-node correctness.

Two levels of checking, because they fail for different reasons and the cheap one
should not need GPUs:

* ``test_gin_put_lowering`` compiles the kernel with no distributed runtime and
  asserts on the generated CUDA. It catches a lowering regression -- a wrong
  callee name, a missing header, a coop value where a type belongs -- without
  needing NCCL, a devcomm, or more than one process.
* ``test_gin_put_selfloop`` actually runs it. With one node the peer is the rank
  itself, so the transfer never crosses the fabric, but it still goes through
  ncclGin::put, the arena window, and the signal, which is everything except the
  NIC.

An inter-node run needs two hosts and lives in
``examples/distributed/nccl_gin_internode/``, not here.
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

_NUMEL = 4096
_BLOCK = 1024
_THREADS = 128
_SIGNAL_ID = 0


def _gin_put_kernel(numel: int, block: int, threads: int):
    num_blocks = (numel + block - 1) // block

    @T.prim_func
    def main(
        send: T.Tensor((numel,), "float32"),
        recv: T.Tensor((numel,), "float32"),
        peer: T.int32,
    ):
        # `numel` has to appear in the body, not just the annotations: the eager
        # builder only captures closure variables the body actually references, so
        # an annotation-only `numel` is not a closure cell and fails to evaluate.
        # The dtype is a literal here for the same reason.
        with T.Kernel(T.ceildiv(numel, block), threads=threads) as bx:
            T.nccl_gin.put_signal(
                src=send[bx * block],
                dst=recv[bx * block],
                size=block,
                peer=peer,
                signal_id=_SIGNAL_ID,
                scope="block",
            )
            # One increment arrives per remote block, so the cumulative total this
            # rank waits for is the block count, not one.
            T.nccl_gin.wait_signal(least=num_blocks, signal_id=_SIGNAL_ID, scope="block")

    return main


def test_gin_put_lowering():
    """The generated CUDA must call the GIN helpers and include their header."""
    kernel = tilelang.compile(_gin_put_kernel(_NUMEL, _BLOCK, _THREADS))
    source = kernel.get_kernel_source()

    assert "tl::gin::put_signal_addr<ncclCoopCta>" in source, source
    assert "tl::gin::wait_signal<ncclCoopCta>" in source, source
    # Without this include the tl::gin:: calls do not resolve, which is the
    # failure mode the codegen prefix table exists to prevent.
    assert "distributed/nccl_gin.h" in source, source
    # The size argument is bytes at the C++ boundary: 1024 float32 elements.
    assert str(_BLOCK * 4) in source, source


def test_gin_put_scope_validation():
    """An unknown scope fails in Python, not as an nvcc template error."""
    with pytest.raises(ValueError, match="scope must be one of"):
        T.nccl_gin.put_signal(src=None, dst=None, size=1, peer=0, scope="grid")


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@distributed_test(nprocs=2)
def test_gin_put_selfloop(local_rank: int, num_ranks: int):
    """Run the GIN path end to end with each rank putting to itself.

    Skips rather than fails when the Device API is absent: most environments ship
    NCCL 2.27.5, which has no GIN at all, and that is a missing dependency rather
    than a defect in this code.
    """
    from tilelang.distributed.host import init_dist
    from tilelang.distributed import nccl_window as _win

    if not _win.supports_device_api():
        pytest.skip(f"NCCL Device API unavailable: {_win.unavailable_reason()}")

    rank, num_ranks, group, node_info = init_dist(local_rank, num_ranks, return_node_info=True)

    allocator = tilelang.get_allocator(
        size=2**22,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group,
        node_info=node_info,
    )

    kernel = tilelang.compile(
        _gin_put_kernel(_NUMEL, _BLOCK, _THREADS),
        compile_once=True,
        compile_group=group,
    )
    kernel.initialize(allocator=allocator)

    send = tilelang.tensor((_NUMEL,), torch.float32, allocator=allocator)
    recv = tilelang.tensor((_NUMEL,), torch.float32, allocator=allocator)
    send.copy_(torch.arange(_NUMEL, device=send.device, dtype=send.dtype) + rank * 1000.0)
    recv.zero_()

    torch.cuda.synchronize()
    dist.barrier(group)

    # Self-put: the peer is this rank, so the expected data is this rank's own.
    kernel(send, recv, rank)
    torch.cuda.synchronize()

    assert torch.equal(recv, send)

    dist.barrier(group)
    allocator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    tilelang.testing.main()
