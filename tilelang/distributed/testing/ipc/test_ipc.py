# python -m torch.distributed.run test_ipc.py --num-processes 2

import torch
import torch.distributed as dist
import os
import inspect
import argparse
import torch.multiprocessing
import ctypes
from ipc_ext import create_ipc_handle, sync_ipc_handles
# build_and_use.py
from torch.utils.cpp_extension import load
from test_kernel.set_value import set_value_cuda

def init_dist(local_rank: int, num_local_ranks: int):
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))

    sig = inspect.signature(dist.init_process_group)
    params = {
        'backend': 'nccl',
        'init_method': f'tcp://{ip}:{port}',
        'world_size': num_nodes * num_local_ranks,
        'rank': node_rank * num_local_ranks + local_rank,
    }
    if 'device_id' in sig.parameters:
        # noinspection PyTypeChecker
        params['device_id'] = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    torch.cuda.set_device(local_rank)

    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))

def get_local_ipc_handle(data: torch.Tensor):
    p = ctypes.c_void_p(data.data_ptr())
    handle = create_ipc_handle(p.value)
    return handle

def create_dist_tensor(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    print(f"Creating dist tensor for rank {local_rank} with {num_local_ranks} local ranks")
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    # Synchronize device IDs
    device_ids = [None, ] * group.size()
    local_device_id = local_rank
    dist.all_gather_object(device_ids, local_device_id, group)
    # print(f"Device IDs: {device_ids}")

    data = torch.randn(8, device=f'cuda:{local_rank}', dtype=torch.float32)

    # Synchronize IPC handles
    ipc_handles = [None, ] * group.size()
    local_ipc_handle = get_local_ipc_handle(data)
    print(f"Local IPC handle: {local_ipc_handle}")
    dist.all_gather_object(ipc_handles, local_ipc_handle, group)

    buffer_ptrs_gpu = torch.empty(group.size(), dtype=torch.uint64, device="cuda")
    sync_ipc_handles(rank, device_ids, ctypes.c_void_p(buffer_ptrs_gpu.data_ptr()).value, ipc_handles, None)

    # print(f"Buffer pointers: {buffer_ptrs_gpu}")

    print(f"Before set_value, rank {rank} data: {data}")

    torch.distributed.barrier(group)
    set_value_cuda(buffer_ptrs_gpu[rank ^ 1], 1, 99)
    torch.distributed.barrier(group)

    print(f"After set_value, rank {rank} data: {data}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=8,
                       help='Number of processes to spawn (default: 8)')
    args = parser.parse_args()
    num_processes = args.num_processes
    torch.multiprocessing.spawn(create_dist_tensor, args=(num_processes, args), nprocs=num_processes)