# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
import torch.distributed as dist
import pynvshmem
import tilelang
import tilelang.language as T
from tilelang.distributed.utils import init_distributed, dtype_map, dsize_map
import math
import argparse

tilelang.disable_cache()


def cannon(MESH, M, N, K, block_M, block_N, block_K, dtype="float16"):

    M_local = T.ceildiv(M, MESH)
    N_local = T.ceildiv(N, MESH)
    K_local = T.ceildiv(K, MESH)
    K_local = T.ceildiv(N, MESH)
    accum_dtype = "float32"

    @T.prim_func
    def main(
            A: T.Tensor((2, M_local, K_local), dtype),
            B: T.Tensor((2, N_local, K_local), dtype),
            A_signal_to: T.Tensor((T.ceildiv(M, block_M),), "uint64"),
            A_signal_from: T.Tensor((T.ceildiv(M, block_M),), "uint64"),
            B_signal_to: T.Tensor((T.ceildiv(N, block_N),), "uint64"),
            B_signal_from: T.Tensor((T.ceildiv(N, block_N),), "uint64"),
            C: T.Tensor((M_local, N_local), dtype),
    ):
        with T.Kernel(
                T.ceildiv(M_local, block_M), T.ceildiv(N_local, block_N), threads=128) as (bx, by):
            mype = T.alloc_local([1], "int32")
            npes = T.alloc_local([1], "int32")
            a_peer_from = T.alloc_local([1], "int32")
            a_peer_to = T.alloc_local([1], "int32")
            b_peer_from = T.alloc_local([1], "int32")
            b_peer_to = T.alloc_local([1], "int32")
            mype[0] = T.get_pe()
            npes[0] = T.get_pe_num()

            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            tx = T.get_thread_binding(0)
            a_peer_from[0] = (mype[0] + 1) % MESH + MESH * (mype[0] // MESH)
            a_peer_to[0] = (mype[0] - 1 + MESH) % MESH + MESH * (mype[0] // MESH)
            b_peer_from[0] = (mype[0] + MESH) % npes[0]
            b_peer_to[0] = (mype[0] - MESH + npes[0]) % npes[0]
            T.clear(C_local)
            for ko in T.serial(MESH):
                if tx == 0:
                    T.signal_wait_until(
                        T.address_of(A_signal_from[bx]),
                        T.NVSHMEM_CMP_EQ,
                        T.ceildiv(N_local, block_N) * ko,
                    )
                    T.signal_wait_until(
                        T.address_of(B_signal_from[by]),
                        T.NVSHMEM_CMP_EQ,
                        T.ceildiv(M_local, block_M) * ko,
                    )

                if by == 0:
                    T.putmem_signal_nbi_block(
                        T.address_of(A[(ko + 1) % 2, bx * block_M, 0]),
                        T.address_of(A[ko % 2, bx * block_M,
                                       0]), block_M * K_local * dsize_map[dtype],
                        T.address_of(A_signal_to[bx]), ko + 1, T.NVSHMEM_SIGNAL_SET, a_peer_to[0])
                if bx == 0:
                    T.putmem_signal_nbi_block(
                        T.address_of(B[(ko + 1) % 2, by * block_N, 0]),
                        T.address_of(B[ko % 2, by * block_N,
                                       0]), block_N * K_local * dsize_map[dtype],
                        T.address_of(B_signal_to[by]), ko + 1, T.NVSHMEM_SIGNAL_SET, b_peer_to[0])

                for ki in T.Pipelined(T.ceildiv(K_local, block_K)):
                    T.copy(
                        A[ko % 2, bx * block_M:(bx + 1) * block_M, ki * block_K:(ki + 1) * block_K],
                        A_shared)
                    T.copy(
                        B[ko % 2, by * block_N:(by + 1) * block_N, ki * block_K:(ki + 1) * block_K],
                        B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                if tx == 0:
                    T.signal_wait_until(
                        T.address_of(A_signal_to[bx]),
                        T.NVSHMEM_CMP_EQ,
                        ko + 1,
                    )
                    T.signal_wait_until(
                        T.address_of(B_signal_to[by]),
                        T.NVSHMEM_CMP_EQ,
                        ko + 1,
                    )
                    T.signal_op(
                        T.address_of(A_signal_from[bx]),
                        1,
                        T.NVSHMEM_SIGNAL_ADD,
                        a_peer_from[0],
                    )
                    T.signal_op(
                        T.address_of(B_signal_from[by]),
                        1,
                        T.NVSHMEM_SIGNAL_ADD,
                        b_peer_from[0],
                    )
            T.copy(C_local, C[bx * block_M:(bx + 1) * block_M, by * block_N:(by + 1) * block_N])

    return main


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", default=256, type=int)
    parser.add_argument("--N", default=256, type=int)
    parser.add_argument("--K", default=256, type=int)
    parser.add_argument("--warmup", default=20, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="float16", type=str, help="data type")
    return parser.parse_args()


if __name__ == "__main__":
    # init
    args = parse_args()

    WORLD_SIZE, RANK, LOCAL_RANK = init_distributed()

    MESH = math.ceil(math.sqrt(WORLD_SIZE))
    assert MESH * MESH == WORLD_SIZE, "Mesh size must match world size"

    M, N, K = args.M, args.N, args.K
    block_M, block_N, block_K = 64, 64, 64
    dtype = dtype_map[args.dtype]

    M_local = math.ceil(M / MESH)
    N_local = math.ceil(N / MESH)
    K_local = math.ceil(K / MESH)

    func = cannon(MESH, M, N, K, block_M, block_N, block_K, args.dtype)
    kernel = tilelang.compile(
        func, pass_configs={
            "tl.disable_tma_lower": True,
            "tl.disable_warp_specialized": True
        })

    # Get CUDA Source
    if RANK == 0:
        print(kernel.get_kernel_source())

    device = torch.device(f"cuda:{RANK}")
    ref = torch.empty((M_local, N_local), dtype=dtype, device=device)
    A_ref = torch.empty((M_local, K_local), dtype=dtype, device=device)
    B_ref = torch.empty((N_local, K_local), dtype=dtype, device=device)

    if RANK == 0:
        A = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(N, K, dtype=dtype, device=device)
        C = A @ B.T

        c_scatter_list = []
        a_scatter_list = []
        b_scatter_list = []
        for r in range(WORLD_SIZE):
            rr, cc = divmod(r, MESH)
            c_tile = C[M_local * rr:M_local * (rr + 1), N_local * cc:N_local * (cc + 1)]
            a_tile = A[M_local * rr:M_local * (rr + 1),
                       K_local * ((cc + rr) % MESH):K_local * ((cc + rr) % MESH + 1)]
            b_tile = B[N_local * cc:N_local * (cc + 1),
                       K_local * ((cc + rr) % MESH):K_local * ((cc + rr) % MESH + 1)]

            c_scatter_list.append(c_tile.contiguous())
            a_scatter_list.append(a_tile.contiguous())
            b_scatter_list.append(b_tile.contiguous())
    else:
        c_scatter_list = None
        a_scatter_list = None
        b_scatter_list = None

    dist.scatter(tensor=ref, scatter_list=c_scatter_list, src=0)
    dist.scatter(tensor=A_ref, scatter_list=a_scatter_list, src=0)
    dist.scatter(tensor=B_ref, scatter_list=b_scatter_list, src=0)
    dist.barrier()

    A = pynvshmem.nvshmem_create_tensor([2, M_local, K_local], dtype)
    B = pynvshmem.nvshmem_create_tensor([2, N_local, K_local], dtype)
    A[0, :, :].copy_(A_ref)
    B[0, :, :].copy_(B_ref)
    A_signal_to = pynvshmem.nvshmem_create_tensor([math.ceil(M / block_M)], torch.uint64)
    A_signal_from = pynvshmem.nvshmem_create_tensor([math.ceil(M / block_M)], torch.uint64)
    B_signal_to = pynvshmem.nvshmem_create_tensor([math.ceil(N / block_N)], torch.uint64)
    B_signal_from = pynvshmem.nvshmem_create_tensor([math.ceil(N / block_N)], torch.uint64)
    A_signal_to.fill_(0)
    A_signal_from.fill_(0)
    B_signal_to.fill_(0)
    B_signal_from.fill_(0)
    C_tilelang = pynvshmem.nvshmem_create_tensor([M_local, N_local], dtype)

    kernel(A, B, A_signal_to, A_signal_from, B_signal_to, B_signal_from, C_tilelang)

    for r in range(WORLD_SIZE):
        dist.barrier()
        if r == RANK:
            if torch.allclose(C_tilelang, ref, rtol=1e-2, atol=1e-2):
                print('-' * 100)
                print(f"[Rank {RANK}] ✅ Tilelang and Torch match")
            else:
                print('-' * 100)
                print(f"[Rank {RANK}] ❌ Tilelang and Torch mismatch")
                print(f"[Rank {RANK}] ref:\n{ref}")
                print(f"[Rank {RANK}] tilelang:\n{C_tilelang}")
        dist.barrier()
