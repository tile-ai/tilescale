import os
import tilelang
import tilelang.language as T
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing
from tilelang.distributed import init_dist
# from cuda.bindings import runtime as cudart
import importlib.metadata
cuda_python_version = importlib.metadata.version("cuda-python")
from packaging import version
if version.parse(cuda_python_version) >= version.parse("12.8.0"):
    from cuda.bindings import driver as cuda
    from cuda.bindings import runtime as cudart
else:
    from cuda import cuda, cudart
from tilelang.distributed import set_signal, wait_eq, perf_fn
from ag_utils import copy_and_barrier_all_intra_node_kernel
    
tilelang.disable_cache()
os.environ['NCCL_DEBUG'] = 'WARN'  # silence NCCL log


@tilelang.jit
def gemm_kernel(M,
                N,
                K,
                num_rank,
                local_rank,
                block_M,
                block_N,
                block_K,
                threads,
                dtype="float16",
                accum_dtype="float"):

    M_per_rank = T.ceildiv(M, num_rank)
    
    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N // num_rank), dtype),
            signal_buffer: T.Tensor((num_rank), "uint32"),
            C: T.Tensor((M, N // num_rank), dtype),
            clk_beg: T.Tensor((T.ceildiv(M, block_M), T.ceildiv(N // num_rank, block_N)), "uint64"),
            clk_end: T.Tensor((T.ceildiv(M, block_M), T.ceildiv(N // num_rank, block_N)), "uint64"),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N // num_rank, block_N), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)    
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            clk_beg_local = T.alloc_local((1), "uint64")
            clk_end_local = T.alloc_local((1), "uint64")

            tid = T.get_thread_binding(0)
            bx_t = (bx + T.ceildiv(local_rank * M_per_rank, block_M)) % T.ceildiv(M, block_M)
            T.clear(C_local)
            if tid == 0:
                clk_beg_local[0] = T.get_clock()
                T.wait_eq(signal_buffer[bx_t * block_M // M_per_rank], 1)
                clk_end_local[0] = T.get_clock()
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[bx_t * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, by * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[bx_t * block_M, by * block_N])
            if tid == 0:
                clk_beg[bx, by] = clk_beg_local[0]
                clk_end[bx, by] = clk_end_local[0]

    return main

def cp_engine_producer_all_gather_full_mesh_pull(
  local_tensor, ag_buffer, signal_buffer, M_per_rank, N,
  signal_target, rank, local_world_size, world_size,
  intranode_ag_stream,
):
    rank_orders = [(rank + i) % local_world_size for i in range(local_world_size)]

    with torch.cuda.stream(intranode_ag_stream):
        for src_rank in rank_orders:
            if src_rank == rank:
                continue
            dst = ag_buffer[rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank, :]
            src = ag_buffer[src_rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank, :]
            dst.copy_(src)

            (err, ) = cuda.cuStreamWriteValue32(
                intranode_ag_stream.cuda_stream,
                signal_buffer[rank][src_rank].data_ptr(),
                signal_target,
                cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
            )
            
def ag_gemm_op(A, B, C, ag_buffer, signal_buffer, sync_buffer, clk_beg, clk_end, lc_clk_beg, lc_clk_end, M_per_rank, N, signal_target, rank, group,
               local_world_size, world_size, local_copy_kernel, gemm_kernel, gemm_stream, ag_stream, iteration):
    
    with torch.cuda.stream(gemm_stream):
        local_copy_kernel(A, ag_buffer[rank], signal_buffer[rank], sync_buffer, lc_clk_beg, lc_clk_end, stream=gemm_stream.cuda_stream)
    
    # # current_stream = torch.cuda.current_stream()
    ag_stream.wait_stream(gemm_stream)

    cp_engine_producer_all_gather_full_mesh_pull(A, ag_buffer, signal_buffer, M_per_rank, N, signal_target,
                                      rank, local_world_size, world_size, ag_stream)

    with torch.cuda.stream(gemm_stream):
        gemm_kernel(ag_buffer[rank], B, signal_buffer[rank], C, clk_beg[iteration, :, :], clk_end[iteration, :, :], stream=gemm_stream.cuda_stream)
        # C = torch.matmul(ag_buffer[rank], B)

    gemm_stream.wait_stream(ag_stream)
    current_stream = torch.cuda.current_stream()
    current_stream.wait_stream(gemm_stream)
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
    BLOCK_N = 256
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
    kernel = gemm_kernel(M, N, K, num_ranks, rank, BLOCK_M, BLOCK_N, BLOCK_K, threads)
    local_copy_kernel = copy_and_barrier_all_intra_node_kernel(
        local_rank=local_rank,
        rank=local_rank,
        num_ranks=num_ranks,
        M=M,
        K=K,
        block_M=64,
        block_K=64,
        threads=128,
    )
    kernel.initialize(allocator=allocator)
    local_copy_kernel.initialize(allocator=allocator)
    if local_rank == 1:
        print(kernel.get_kernel_source())
        print(local_copy_kernel.get_kernel_source())

    A = tilelang.tensor((M_per_rank, K), dtype, allocator=allocator).normal_()
    B = tilelang.tensor((K, N_per_rank), dtype, allocator=allocator).normal_()
    C = tilelang.tensor((M, N_per_rank), dtype, allocator=allocator)
    ag_buffer = tilelang.tensor((M, K), dtype, allocator=allocator, return_peers=True)
    signal_buffer = tilelang.tensor((num_local_ranks,),
                                    torch.uint32,
                                    allocator=allocator,
                                    return_peers=True)
    signal_buffer[rank].fill_(0) # check if needed
    sync_buffer = tilelang.tensor((3 * num_ranks,),
                                    torch.uint32,
                                    allocator=allocator)
    clk_beg = tilelang.tensor((15, 64, 14), torch.uint64, allocator=allocator)
    clk_end = tilelang.tensor((15, 64, 14), torch.uint64, allocator=allocator)
    lc_clk_beg = tilelang.tensor((15, 132), torch.uint64, allocator=allocator)
    lc_clk_end = tilelang.tensor((15, 132), torch.uint64, allocator=allocator)
    # ag_buffer[rank][rank * M_per_rank:(rank + 1) * M_per_rank, :].copy_(A)

    # dist.barrier(group)

    gemm_stream = torch.cuda.Stream()
    ag_stream = torch.cuda.Stream(priority=-1)
    signal_target = 1

    tilelang_C = ag_gemm_op(A, B, C, ag_buffer, signal_buffer, sync_buffer, clk_beg, clk_end, lc_clk_beg, lc_clk_end, M_per_rank, K, signal_target, rank,
                            group, num_local_ranks, num_local_ranks, local_copy_kernel, kernel, gemm_stream, ag_stream, 0)

    torch_ag_buffer = torch.empty([M, K], dtype=dtype, device="cuda")
    torch_C = torch_ag_gemm(group, A, B, torch_ag_buffer)

    if torch.allclose(torch_C, tilelang_C, atol=1e-6, rtol=1e-6):
        print(f"rank {local_rank} check passed.✅")
    else:
        print(f"rank {local_rank} check failed.❌")
        print(f"torch_C: {torch_C}, tilelang_C: {tilelang_C}")
        # raise ValueError("Test failed")
    
    tl_out, tl_t = perf_fn(lambda i: ag_gemm_op(A, B, C, ag_buffer, signal_buffer, sync_buffer, clk_beg, clk_end, lc_clk_beg, lc_clk_end, M_per_rank, K, signal_target, rank,
                            group, num_local_ranks, num_local_ranks, local_copy_kernel, kernel, gemm_stream, ag_stream, i), warmup=5, rep=10)
    
    delta = clk_end.to(torch.int64) - clk_beg.to(torch.int64)
    lc_delta = lc_clk_end.to(torch.int64) - lc_clk_beg.to(torch.int64)
    
    torch.set_printoptions(threshold=float('inf'))
    print(f"rank {local_rank} wait_cycle: {delta[5]} cycles")
    print(f"rank {local_rank} lc_wait_cycle: {lc_delta[5]} cycles")
    print(f"rank {local_rank} tilelang ag_gemm time: {tl_t:.2f} ms, TFLOPS: {2*M*N*K/1e9/(tl_t)/num_local_ranks:.2f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-processes', type=int, default=2, help='Number of processes to spawn (default: 2)')
    parser.add_argument('--M', type=int, default=8192, help='M dimension')
    parser.add_argument('--N', type=int, default=28672, help='N dimension')
    parser.add_argument('--K', type=int, default=8192, help='K dimension')
    args = parser.parse_args()
    num_processes = args.num_processes

    torch.multiprocessing.spawn(main, args=(num_processes, args), nprocs=num_processes)
