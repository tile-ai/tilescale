# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
import pynvshmem
import os
import tilelang
import tilelang.language as T
from tilelang.profiler import TensorSupplyType

PE_num = 8

def allgather(M, N, block_M, block_N, dtype="float16"):

    @T.prim_func
    def main(
            A: T.Buffer((M, N), dtype),
            B: T.Buffer((M * PE_num, N), dtype),
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
                    T.address_of(B[mype[0] * M, 0]), T.address_of(A[0, 0]), block_M * block_N * 2, peer[0])

    return main


M, N, block_M, block_N = 32, 32, 32, 32
dtype = torch.float16
nelems = M * PE_num * N

RANK = int(os.environ.get("RANK", 0))

func = allgather(M, N, block_M, block_N)
kernel = tilelang.compile(func, out_idx=-1)

# Get CUDA Source
if RANK == 0:   
    print(kernel.get_kernel_source())

profiler = kernel.get_profiler(tensor_supply_type=TensorSupplyType.Randn)

ref_tensor = torch.arange(nelems, dtype=dtype).cuda()
ref_tensor = ref_tensor.reshape(M * PE_num, N)

profiler.init_distributed()
ag_buffer = pynvshmem.nvshmem_create_tensor([M, N], dtype)
ag_buffer[:].copy_(ref_tensor[M * RANK:M * (RANK + 1), :])
print("ag_buffer:", ag_buffer)

out = kernel(ag_buffer)
print("out:", out)


ref_cpu = ref_tensor.cpu()
for i in range(PE_num):
    if i == RANK:
        out_cpu = out.cpu()
        assert torch.allclose(out_cpu, ref_cpu, atol=1e-3, rtol=1e-3)
        print(f"rank {i} check passed.")