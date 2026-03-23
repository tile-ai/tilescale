import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing
import tilelang
from tilelang.distributed import init_dist, perf_fn
from reduce_scatter import reduce_scatter_2d_op, create_reduce_scater_2d_ctx


def torch_reduce_scatter(
    pg: torch.distributed.ProcessGroup,
    input: torch.Tensor,
    num_local_ranks: int,
) -> torch.Tensor:
    M, N = input.shape
    output = torch.empty((M // num_local_ranks, N), dtype=input.dtype, device=input.device)
    torch.distributed.reduce_scatter_tensor(output, input, group=pg)
    return output


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    dtype = torch.float16
    M = args.M
    N = args.N
    M_per_rank = M // num_local_ranks

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert rank == local_rank and num_ranks == num_local_ranks, "only support single node for now"

    allocator = tilelang.get_allocator(
        size=2**30, device="cuda", is_distributed=True, local_rank=local_rank, num_local_ranks=num_local_ranks, group=group
    )

    input_tensor = tilelang.tensor((M, N), dtype, allocator=allocator).normal_() / 10
    output_tensor = tilelang.tensor((M_per_rank, N), dtype, allocator=allocator)

    ctx = create_reduce_scater_2d_ctx(
        M, N, local_rank, num_local_ranks, num_local_ranks, dtype, allocator, overlap_with_gemm=False
    )

    dist.barrier()

    tilelang_out = reduce_scatter_2d_op(input_tensor, ctx, output_tensor)
    torch_out = torch_reduce_scatter(group, input_tensor, num_local_ranks)

    atol = 1e-2
    rtol = 1e-2
    if torch.allclose(torch_out, tilelang_out, atol=atol, rtol=rtol):
        print(f"rank {local_rank} check passed. ✅")
    else:
        print(f"rank {local_rank} check failed. ❌")
        print(f"max diff: {(torch_out - tilelang_out).abs().max()}")

    tl_t = perf_fn(lambda: reduce_scatter_2d_op(input_tensor, ctx, output_tensor), warmup=5, rep=10)
    input_bytes = M * N * torch.finfo(dtype).bits // 8
    algbw = input_bytes / tl_t / 1e6  # GB/s
    print(f"rank {local_rank} tilelang reduce_scatter time: {tl_t:.2f} ms, algbw: {algbw:.2f} GB/s")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=2, help="Number of processes to spawn (default: 2)")
    parser.add_argument("--M", type=int, default=8192, help="M dimension")
    parser.add_argument("--N", type=int, default=8192, help="N dimension")
    args = parser.parse_args()
    num_processes = args.num_processes

    torch.multiprocessing.spawn(main, args=(num_processes, args), nprocs=num_processes)
