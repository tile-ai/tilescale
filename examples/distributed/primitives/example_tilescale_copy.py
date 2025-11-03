import os
import tilelang
import tilelang.language as T
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing
from tilelang.distributed import init_dist

tilelang.disable_cache()
os.environ['NCCL_DEBUG'] = 'WARN'  # silence NCCL log


@tilelang.jit
def get_kernel(M, N, block_M, block_N, threads, kernel='simt_push_tile'):

    @T.prim_func
    def simt_push_buffer(
            dst: T.Tensor((M, N), "float32"),
            src: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel((1), threads=threads):
            rank = T.alloc_local([1], "uint64")
            rank[0] = T.get_rank()

            T.copy(
                src,
                dst,
                dst_pe=1 - rank[0],
                disable_tma=True  # Ensure testing SIMT remote copy
            )

    @T.prim_func
    def simt_push_tile(
            dst: T.Tensor((M, N), "float32"),
            src: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(M // block_M, N // block_N, threads=threads) as (bx, by):
            rank = T.alloc_local([1], "uint64")
            rank[0] = T.get_rank()

            smem = T.alloc_shared((block_M, block_N), "float32")
            T.annotate_layout({smem: tilelang.layout.make_swizzled_layout(smem)})

            T.copy(
                src[bx * block_M:(bx + 1) * block_M, by * block_N:(by + 1) * block_N],
                smem,
                disable_tma=True  # Ensure testing SIMT remote copy
            )

            T.copy(
                smem,
                dst[bx * block_M:(bx + 1) * block_M, by * block_N:(by + 1) * block_N],
                dst_pe=1 - rank[0],
                disable_tma=True  # Ensure testing SIMT remote copy
            )

    @T.prim_func
    def simt_pull_tile(
            dst: T.Tensor((M, N), "float32"),
            src: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(M // block_M, N // block_N, threads=threads) as (bx, by):
            rank = T.alloc_local([1], "uint64")
            rank[0] = T.get_rank()

            smem = T.alloc_shared((block_M, block_N), "float32")
            T.annotate_layout({smem: tilelang.layout.make_swizzled_layout(smem)})

            T.copy(
                src[bx * block_M:(bx + 1) * block_M, by * block_N:(by + 1) * block_N],
                smem,
                src_pe=1 - rank[0],
                disable_tma=True  # Ensure testing SIMT remote copy
            )

            T.copy(
                smem,
                dst[bx * block_M:(bx + 1) * block_M, by * block_N:(by + 1) * block_N],
                disable_tma=True  # Ensure testing SIMT remote copy
            )

    # TMA kernel requires run-time aware peer rank
    @T.prim_func
    def tma_load_tile(
            dst: T.Tensor((M, N), "float32"),
            src: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(M // block_M, N // block_N, threads=threads) as (bx, by):

            smem = T.alloc_shared((block_M, block_N), "float32")
            T.annotate_layout({smem: tilelang.layout.make_swizzled_layout(smem)})

            # TMA load
            T.copy(
                src[bx * block_M:(bx + 1) * block_M, by * block_N:(by + 1) * block_N],
                smem,
                src_pe=1 - T.get_rank(),
                # NOTE(wt): We cannot use rank[0] as above for TMA remote copy currently.
            )

            T.copy(
                smem,
                dst[bx * block_M:(bx + 1) * block_M, by * block_N:(by + 1) * block_N],
                disable_tma=True  # Ensure testing SIMT remote copy
            )

    @T.prim_func
    def tma_store_tile(
            dst: T.Tensor((M, N), "float32"),
            src: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(M // block_M, N // block_N, threads=threads) as (bx, by):

            smem = T.alloc_shared((block_M, block_N), "float32")
            T.annotate_layout({smem: tilelang.layout.make_swizzled_layout(smem)})

            T.copy(
                src[bx * block_M:(bx + 1) * block_M, by * block_N:(by + 1) * block_N],
                smem,
                disable_tma=True  # Ensure testing SIMT remote copy
            )

            # TMA store
            T.copy(
                smem,
                dst[bx * block_M:(bx + 1) * block_M, by * block_N:(by + 1) * block_N],
                dst_pe=1 - T.get_rank())

    return {
        'simt_push_buffer': simt_push_buffer,
        'simt_push_tile': simt_push_tile,
        'simt_pull_tile': simt_pull_tile,
        'tma_load_tile': tma_load_tile,
        'tma_store_tile': tma_store_tile
    }[kernel]


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    M = args.M
    N = args.N
    BLOCK_M = 64
    BLOCK_N = 128
    threads = 128
    assert num_local_ranks == 2, "this example only supports 2 ranks copying to each other"

    _, _, group = init_dist(local_rank, num_local_ranks)
    allocator = tilelang.get_allocator(
        size=2**25,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_local_ranks,
        group=group)

    kernel = get_kernel(
        M, N, BLOCK_M, BLOCK_N, threads, kernel=args.kernel)
    kernel.initialize(allocator=allocator)
    if local_rank == 0:
        print(kernel.get_kernel_source())

    src = tilelang.tensor((M, N), torch.float32, allocator=allocator).normal_()
    dst = tilelang.tensor((M, N), torch.float32, allocator=allocator)

    torch.cuda.synchronize()
    torch.distributed.barrier(group)
    kernel(dst, src)
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
    parser.add_argument('--M', type=int, default=1024, help='M dimension')
    parser.add_argument('--N', type=int, default=1024, help='N dimension')
    parser.add_argument('--kernel', type=str, default='simt_push_tile', help='kernel to use')
    args = parser.parse_args()
    num_processes = 2

    torch.multiprocessing.spawn(main, args=(num_processes, args), nprocs=num_processes)
