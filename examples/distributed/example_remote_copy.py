import tilelang
import tilelang.language as T
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing
from tilelang.distributed.utils import init_dist, create_dist_tensor, create_tensor

tilelang.disable_cache()


def kernel_(M, N, num_rank, block_M, threads):

    @T.prim_func
    def main(
            dst: T.Tensor((M, N), "float32", buffer_type="distributed"),
            src: T.Tensor((M, N), "float32"),
            rank: T.Tensor((1), "int32"),
            meta_data: T.Tensor((1, num_rank), "uint64", buffer_type="meta_data"),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as bid:
            #TODO(ipc): Add work partition among warps and vectorized copy
            T.remote_copy(
                src=T.address_of(src[bid * block_M, 0]),
                dst=T.address_of(dst[bid * block_M, 0]),
                size=block_M * N,
                dst_pe=rank[0] ^ 1,
                unroll_factor=4)

    return main


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    M, N = args.M, args.N
    BLOCK_M = 128
    threads = 128

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    src = torch.randn(M, N, device="cuda", dtype=torch.float32)
    dst = create_tensor([M, N], torch.float32)
    rank_tensor = torch.tensor([local_rank], device="cuda", dtype=torch.int32)
    buffer_ptrs_gpu = create_dist_tensor(local_rank, num_local_ranks, dst, rank,
                                         group).reshape(1, num_local_ranks)

    kernel = tilelang.compile(kernel_(M, N, num_ranks, BLOCK_M, threads))
    if local_rank == 0 and args.print_source:
        print(kernel.get_kernel_source())

    torch.cuda.synchronize()
    torch.distributed.barrier(group)
    kernel(dst, src, rank_tensor, buffer_ptrs_gpu)
    torch.cuda.synchronize()
    torch.distributed.barrier(group)

    dst_torchs = [torch.empty_like(src) for _ in range(num_local_ranks)]
    dist.all_gather(dst_torchs, src, group)
    dst_torch = dst_torchs[local_rank ^ 1]

    if torch.allclose(dst_torch, dst, atol=1e-6, rtol=1e-6):
        print(f"rank {local_rank} check passed.✅")
    else:
        print(f"rank {local_rank} check failed.❌")
        print(f"dst_torch: {dst_torch}, dst: {dst}")
        raise ValueError("Test failed")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-processes', type=int, default=2, help='Number of processes to spawn (default: 2)')
    parser.add_argument('--M', type=int, default=256, help='M dimension')
    parser.add_argument('--N', type=int, default=256, help='N dimension')
    parser.add_argument(
        '--print-source', action='store_true', help='Print the source code of the kernel')
    args = parser.parse_args()
    num_processes = args.num_processes

    torch.multiprocessing.spawn(main, args=(num_processes, args), nprocs=num_processes)
