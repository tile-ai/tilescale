# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
import torch.distributed as dist
import triton_dist.pynvshmem as pynvshmem
import tilelang
import tilelang.language as T
from tilelang.profiler import TensorSupplyType
from tilelang.distributed.utils import init_distributed, dtype_map

tilelang.disable_cache()


def allgather(PE_num, M, N, block_M, block_N, dtype="float16"):

    @T.prim_func
    def naive_a2a(
            A: T.Tensor((M, N), dtype), # type: ignore
            B: T.Tensor((M * PE_num, N), dtype), # type: ignore
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            mype = T.alloc_local([1], "int32")
            npes = T.alloc_local([1], "int32")
            peer = T.alloc_local([1], "int32")
            mype[0] = T.get_pe()
            npes[0] = T.get_pe_num()

            A_shared = T.alloc_shared((block_M, block_N), dtype)
            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(A_shared, B[mype[0] * M, bx * block_N])
            for k in T.serial(PE_num - 1):
                peer[0] = (mype[0] + 1 + k) % npes[0]
                T.putmem_nbi_block(
                    T.address_of(B[mype[0] * M, 0]), 
                    T.address_of(A[0, 0]), 
                    block_M * block_N * dtype_map[dtype].itemsize,
                    peer[0])

    return naive_a2a


if __name__ == '__main__':
    WORLD_SIZE, RANK, LOCAL_RANK, TP_GROUP = init_distributed(return_tp_group=True)

    M, N, block_M, block_N = 64, 32, 64, 32
    PE_num = WORLD_SIZE
    dtype = torch.float16
    nelems = M * PE_num * N

    func = allgather(PE_num, M, N, block_M, block_N)
    kernel = tilelang.compile(func, pass_configs={"tl.disable_tma_lower": True})

    # Get CUDA Source
    if RANK == 0:
        print(kernel.get_kernel_source())

    profiler = kernel.get_profiler(tensor_supply_type=TensorSupplyType.Randn)

    local_ref_tensor = torch.randn(M, N, dtype=dtype).cuda()

    ag_buffer = pynvshmem.nvshmem_create_tensor([M, N], dtype)
    ag_buffer.copy_(local_ref_tensor)
    print("ag_buffer:", ag_buffer)

    out = pynvshmem.nvshmem_create_tensor([M * PE_num, N], dtype)
    kernel(ag_buffer, out)
    print("out:", out)

    ref = torch.empty((M * PE_num, N), dtype=dtype).cuda()
    dist.all_gather_into_tensor(ref, local_ref_tensor, group=TP_GROUP)
    
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    print(f"rank {RANK} check passed.✅")
            
    dist.destroy_process_group()

