import torch
import torch.distributed as dist
import argparse
import torch.multiprocessing
from tilelang.distributed.utils import init_dist, create_dist_tensor
from test_kernel.set_value import set_value_cuda


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    data = torch.randn(1, device=f'cuda:{local_rank}', dtype=torch.float32)
    buffer_ptrs_gpu = create_dist_tensor(local_rank, num_local_ranks, data, rank, group)

    print(f"Before set_value, rank {rank} data: {data}")
    torch.distributed.barrier(group)
    set_value_cuda(buffer_ptrs_gpu[rank ^ 1], 1, 99)
    torch.distributed.barrier(group)
    print(f"After set_value, rank {rank} data: {data}")

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-processes', type=int, default=2, help='Number of processes to spawn (default: 2)')
    args = parser.parse_args()
    num_processes = args.num_processes
    torch.multiprocessing.spawn(main, args=(num_processes, args), nprocs=num_processes)
