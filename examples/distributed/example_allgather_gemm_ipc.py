import os
import tilelang
import tilelang.language as T
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing
from tilelang.distributed.utils import init_dist
from cuda import cudart
from tilelang.distributed.utils import set_signal, wait_eq

tilelang.disable_cache()
os.environ['NCCL_DEBUG'] = 'WARN'  # silence NCCL log


def gemm_kernel(M,
                N,
                K,
                num_rank,
                block_M,
                block_N,
                block_K,
                threads,
                dtype="float16",
                accum_dtype="float"):

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N // num_rank), dtype),
            C: T.Tensor((M, N // num_rank), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def cp_engine_producer_all_gather_put(local_tensor, ag_buffer, signal_buffer, M_per_rank, N,
                                      signal_target, rank, local_world_size, world_size,
                                      intranode_ag_stream):
    local_rank = rank % local_world_size
    n_nodes = world_size // local_world_size
    node_rank = rank // local_world_size

    for i in range(1, local_world_size):
        segment = rank * M_per_rank * N
        local_dst_rank = (local_rank + local_world_size - i) % local_world_size
        src_ptr = ag_buffer[local_rank].data_ptr() + segment * local_tensor.element_size()
        dst_ptr = ag_buffer[local_dst_rank].data_ptr() + segment * local_tensor.element_size()
        # Using copy engine to perform intranode transmission
        # Sending rank-th local tensor to other ranks inside the node.
        (err,) = cudart.cudaMemcpyAsync(
            dst_ptr,
            src_ptr,
            M_per_rank * N * local_tensor.element_size(),
            cudart.cudaMemcpyKind.cudaMemcpyDefault,
            intranode_ag_stream.cuda_stream,
        )
        # Notify the peer that the transmission is done.
        set_signal(signal_buffer[local_dst_rank][rank], signal_target, intranode_ag_stream)

    for i in range(1, n_nodes):
        recv_rank = local_rank + (node_rank + n_nodes - i) % n_nodes * local_world_size
        recv_segment = recv_rank * M_per_rank * N
        # Waiting for the internode data ready
        wait_eq(signal_buffer[local_rank][recv_rank], signal_target, intranode_ag_stream)
        src_ptr = ag_buffer[local_rank].data_ptr() + recv_segment * local_tensor.element_size()
        for j in range(1, local_world_size):
            local_dst_rank = (local_rank + local_world_size - j) % local_world_size
            dst_ptr = ag_buffer[local_dst_rank].data_ptr(
            ) + recv_segment * local_tensor.element_size()
            # Sending (local_rank + j*local_world_size) % world_size -th local tensor to other ranks inside the node.
            (err,) = cudart.cudaMemcpyAsync(
                dst_ptr,
                src_ptr,
                M_per_rank * N * local_tensor.element_size(),
                cudart.cudaMemcpyKind.cudaMemcpyDefault,
                intranode_ag_stream.cuda_stream,
            )
            # Notify the peer that the transmission is done.
            set_signal(signal_buffer[local_dst_rank][recv_rank], signal_target, intranode_ag_stream)


def ag_gemm_op(A, B, C, ag_buffer, signal_buffer, M_per_rank, N, signal_target, rank, group,
               local_world_size, world_size, gemm_kernel, ag_stream):

    dist.barrier(group)

    # all_gather A to ag_buffer
    with torch.cuda.stream(ag_stream):
        cp_engine_producer_all_gather_put(A, ag_buffer, signal_buffer, M_per_rank, N, signal_target,
                                          rank, local_world_size, world_size, ag_stream)

    current_stream = torch.cuda.current_stream()
    current_stream.wait_stream(ag_stream)

    dist.barrier(group)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    torch.distributed.barrier(group)
    gemm_kernel(ag_buffer[rank], B, C)
    torch.cuda.synchronize()
    torch.distributed.barrier(group)

    return C


def torch_ag_gemm(
    pg: torch.distributed.ProcessGroup,
    local_input: torch.Tensor,
    local_weight: torch.Tensor,
    ag_out: torch.Tensor,
):
    torch.distributed.all_gather_into_tensor(ag_out, local_input, pg)
    ag_gemm_output = torch.matmul(ag_out, local_weight)
    return ag_gemm_output


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    dtype = torch.float16
    M = args.M if args else 8192
    N = args.N if args else 8192
    K = args.K if args else 8192
    M_per_rank = M // num_local_ranks
    N_per_rank = N // num_local_ranks

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    threads = 256

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    allocator = tilelang.get_allocator(
        size=2**30,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_local_ranks,
        group=group)
    kernel = tilelang.compile(gemm_kernel(M, N, K, num_ranks, BLOCK_M, BLOCK_N, BLOCK_K, threads))
    kernel.initialize(allocator=allocator)
    if local_rank == 0:
        print(kernel.get_kernel_source())

    A = tilelang.tensor((M_per_rank, K), dtype, allocator=allocator).normal_()
    B = tilelang.tensor((K, N_per_rank), dtype, allocator=allocator).normal_()
    C = tilelang.tensor((M, N_per_rank), dtype, allocator=allocator)
    ag_buffer = tilelang.tensor((M, K), dtype, allocator=allocator, return_peers=True)
    signal_buffer = tilelang.tensor((num_local_ranks,),
                                    torch.int32,
                                    allocator=allocator,
                                    return_peers=True)
    signal_buffer[rank].fill_(0)
    ag_buffer[rank][rank * M_per_rank:(rank + 1) * M_per_rank, :].copy_(A)

    dist.barrier(group)

    ag_stream = torch.cuda.Stream()
    signal_target = 1

    tilelang_C = ag_gemm_op(A, B, C, ag_buffer, signal_buffer, M_per_rank, K, signal_target, rank,
                            group, num_local_ranks, num_local_ranks, kernel, ag_stream)

    torch_ag_buffer = torch.empty([M, K], dtype=dtype, device="cuda")
    torch_C = torch_ag_gemm(group, A, B, torch_ag_buffer)

    if torch.allclose(torch_C, tilelang_C, atol=1e-6, rtol=1e-6):
        print(f"rank {local_rank} check passed.✅")
    else:
        print(f"rank {local_rank} check failed.❌")
        print(f"torch_C: {torch_C}, tilelang_C: {tilelang_C}")
        raise ValueError("Test failed")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-processes', type=int, default=2, help='Number of processes to spawn (default: 2)')
    parser.add_argument('--M', type=int, default=8192, help='M dimension')
    parser.add_argument('--N', type=int, default=8192, help='N dimension')
    parser.add_argument('--K', type=int, default=8192, help='K dimension')
    args = parser.parse_args()
    num_processes = args.num_processes

    torch.multiprocessing.spawn(main, args=(num_processes, args), nprocs=num_processes)
