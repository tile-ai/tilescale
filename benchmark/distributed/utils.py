# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
# TODO: remove this
from triton_dist import pynvshmem

def dtype_size_in_bytes(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()

class AllToAllContext:

    def __init__(
        self,
        max_m: int,
        hidden: int,
        rank: int,
        num_tot_experts: int,
        WORLD_SIZE: int,
        experts_per_rank: int,
        dtype=torch.bfloat16,
        scale_dtype=torch.float,
    ):
        """
        max_m: max number of tokens per rank
        """
        self.send_buf = pynvshmem.nvshmem_create_tensor([max_m, hidden], dtype)
        self.recv_buf = pynvshmem.nvshmem_create_tensor([WORLD_SIZE * max_m * 2, hidden], dtype)
        self.scale_send_buf = pynvshmem.nvshmem_create_tensor([max_m], scale_dtype)
        self.scale_recv_buf = pynvshmem.nvshmem_create_tensor([WORLD_SIZE * max_m * 2], scale_dtype)
        self.split_send_buf = pynvshmem.nvshmem_create_tensor([num_tot_experts], torch.int32)
        self.split_recv_buf = pynvshmem.nvshmem_create_tensor([num_tot_experts * 2], torch.int32)
        self.signal_buf = pynvshmem.nvshmem_create_tensor([WORLD_SIZE * 2], torch.uint64)

        self.max_m = max_m
        self.hidden = hidden
        self.dtype = dtype
        self.scale_dtype = scale_dtype
        self.ele_size = dtype_size_in_bytes(self.dtype)
        self.scale_ele_size = dtype_size_in_bytes(self.scale_dtype)

        self.num_tot_experts = num_tot_experts
        self.experts_per_rank = experts_per_rank

        self.WORLD_SIZE = WORLD_SIZE
        self.rank = rank

        # start from 1, becase the initial values of signal buffer is 0
        self.call_count = 1
        self.MOD_VALUE = 1000000


def create_all_to_all_context(
    max_m: int,
    hidden: int,
    rank: int,
    num_tot_experts: int,
    WORLD_SIZE: int,
    experts_per_rank: int,
    dtype=torch.bfloat16,
    scale_dtype=torch.float,
):
    return AllToAllContext(
        max_m,
        hidden,
        rank,
        num_tot_experts,
        WORLD_SIZE,
        experts_per_rank,
        dtype,
        scale_dtype,
    )
