import tilelang
import tilelang.language as T
from tilelang.distributed import init_dist
import torch
import torch.distributed as dist
import argparse


def alltoall(PE_num, M, N, block_M, block_N, threads):
    assert block_N == N

    @T.prim_func
    def main(
            src: T.Tensor((PE_num * M, N), "float16"),
            dst: T.Tensor((PE_num * M, N), "float16"),
            barrier: T.Tensor((PE_num), "int32"),
    ):
        # Currently not support tiled copy
        with T.Kernel(
                PE_num, T.ceildiv(M, block_M), T.ceildiv(N, block_N),
                threads=threads) as (bx, by, bz):
            rank = T.alloc_local([1], "int32")
            num_ranks = T.alloc_local([1], "int32")

            dst_rank = bx
            rank[0] = T.get_rank()
            num_ranks[0] = T.get_num_ranks()

            T.put_block(
                src=T.address_of(src[dst_rank * M + by * block_M, 0]),
                dst=T.address_of(dst[rank[0] * M + by * block_M, 0]),
                size=block_M * block_N,
                dst_pe=dst_rank,
            )
            T.fence_sys(sem=T.MemorySemantic.RELEASE)

    return main


def run_alltoall(local_rank, num_ranks, args):
    PE_num = args.PE_num
    M = args.M
    N = args.N
    block_M = 32
    block_N = N
    threads = 256

    local_rank, num_ranks, group_size = init_dist(local_rank, num_ranks)
    allocator = tilelang.get_allocator(
        size=2**35,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group_size,
    )
    kernel = tilelang.compile(alltoall(PE_num, M, N, block_M, block_N, threads))
    kernel.initialize(allocator=allocator)
    src = tilelang.tensor((PE_num * M, N), torch.float16, allocator=allocator).random_()
    dst = tilelang.tensor((PE_num * M, N), torch.float16, allocator=allocator).zero_()
    barrier = tilelang.tensor((PE_num), torch.int32, allocator=allocator).zero_()

    torch.cuda.synchronize()
    dist.barrier(group_size)

    # Warmup
    for _ in range(args.warmup):
        kernel(src, dst, barrier)
    torch.cuda.synchronize()
    dist.barrier(group_size)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iter):
        kernel(src, dst, barrier)
        torch.cuda.synchronize()
        dist.barrier(group_size)
    end.record()
    torch.cuda.synchronize()
    dist.barrier(group_size)
    elapsed_time = start.elapsed_time(end) / args.iter
    print(
        f"Rank {local_rank} Average Kernel execution time: {elapsed_time:.3f} ms, Bandwidth: {2 * PE_num * M * N / (elapsed_time * 1e6):.3f} GB/s"
    )

    # Torch Reference
    torch.cuda.synchronize()
    dst_ref = torch.zeros((PE_num * M, N), dtype=torch.float16, device="cuda")
    dist.all_to_all_single(dst_ref, src, group=group_size)
    torch.cuda.synchronize()

    # 对比结果
    if torch.allclose(dst, dst_ref, atol=1e-2, rtol=1e-2):
        print(f"Rank {local_rank} Verification Passed! ✅")
    else:
        max_diff = (dst - dst_ref).abs().max()
        print(f"Rank {local_rank} Verification Failed! ❌ Max diff: {max_diff}")
        print(f"dst: {dst}")
        print(f"dst_ref: {dst_ref}")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--PE_num", type=int, default=8)
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=7168)
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--iter", type=int, default=10, help="Number of benchmark iterations")

    args = parser.parse_args()
    torch.multiprocessing.spawn(run_alltoall, args=(args.PE_num, args), nprocs=args.PE_num)
