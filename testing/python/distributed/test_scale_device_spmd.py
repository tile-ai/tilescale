"""T.Scale("device", n): SPMD execution across a process group.

Every rank runs the device-scale body once with the scale var bound to its
own rank (tl::get_rank() from the peer table installed by
kernel.initialize(allocator)).

CI configuration: 4 GPUs, compute >= 9.0.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T
import tilelang.testing
from testing.python.distributed._utils import distributed_test

os.environ.setdefault("NCCL_DEBUG", "WARN")

_NPROCS = 4
_THREADS = 32


def _kernel_device_scale_rank():
    @T.prim_func
    def main(out: T.Tensor((2,), T.int32)):
        with T.Scale("device", _NPROCS) as di:
            with T.Scale("block", 1) as _bx:
                with T.Scale("thread", _THREADS) as tid:
                    if tid == 0:
                        out[0] = di
                        out[1] = di * 10

    return main


# Pre-compile at import time so children load from disk cache.
tilelang.compile(_kernel_device_scale_rank())


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@distributed_test(nprocs=_NPROCS)
def test_scale_device_spmd(local_rank: int, num_local_ranks: int):
    from tilelang.distributed.host import init_dist

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    allocator = tilelang.get_allocator(
        size=2**20,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_local_ranks,
        group=group,
    )

    kernel = tilelang.compile(_kernel_device_scale_rank(), compile_once=True, compile_group=group)
    kernel.initialize(allocator=allocator)

    out = tilelang.tensor((2,), T.int32, allocator=allocator)

    torch.cuda.synchronize()
    dist.barrier(group)
    kernel(out)
    torch.cuda.synchronize()
    dist.barrier(group)

    out_cpu = out.cpu()
    assert out_cpu[0].item() == local_rank, (
        f"rank {local_rank}: device-scale var = {out_cpu[0].item()}, expected {local_rank}")
    assert out_cpu[1].item() == local_rank * 10

    allocator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    import tilelang.testing

    tilelang.testing.main()
