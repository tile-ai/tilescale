# For intranode only

# Currently we focus on bf16 w/o cache
# TODO(wt): Support: cache mode, fp8 w/ scale ...

import os
import torch
import torch.distributed as dist
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
from typing import Optional, Tuple, Union
from ..utils import Config, create_moe_recv_counters

tilelang.disable_cache()

num_ranks: int = int(os.environ.get("WORLD_SIZE", 1))  # num_PE

# [QST] Is `num_ranks` the same as `group_size` in DeepEP?


# TODO(wt): Add async functionality
# deep_ep/buffer.py:Buffer::dispatch()
# csrc/deep_ep.cpp:Buffer::intranode_dispatch()
# Launches notify_dispatch and dispatch, or their cached variants
def intranode_dispatch(
        x: torch.Tensor,

        # Following 3 args are obtained by get_dispatch_layout()
        num_tokens_per_rank: torch.Tensor,  # Need in non-cached mode
        is_token_in_rank: torch.Tensor,  # Need in non-cached mode
        num_tokens_per_expert: torch.Tensor,  # Need in non-cached mode
        topk_idx: torch.Tensor,  # Not necessary for cached mode
        topk_weights: torch.Tensor,  # Not necessary for cached mode

        # TODO(wt): Add cached args
    expert_alignment: int = 1,  # 1 means no alignment
        num_worst_tokens: int = 0,
        config: Optional[Config] = None):
    """
    Dispatch tokens to different ranks, currently only intranode ranks are supported.
    Intranode kernels require all the ranks should be visible via NVLink.
    Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
        index should be visible via RDMA.

    Arguments:
        x: `torch.Tensor`, the shape must be `[num_tokens, hidden]`. (We don't support fp8 now).
        
        num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
        is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
        num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
        
        topk_idx: `[num_tokens, num_topk]` with `torch.int64`, the expert indices selected by each token,
            `-1` means no selections.
        topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
        
        expert_alignment: align the number of tokens received by each local expert to this variable.
        num_worst_tokens: the worst number of tokens to receive, if specified, there will be no CPU sync, and it
            will be CUDA-graph compatible. Please also notice that this flag is for intranode only.
        config: the performance tuning config.

    Returns:
        recv_x: received tokens, the same type and tuple as the input `x`, but the number of tokens equals to the
            received token count.
        recv_topk_idx: received expert indices.
        recv_topk_weights: received expert weights.
        num_recv_tokens_per_expert_list: Python list shaped `[num_local_experts]`, the received token count by
            each local expert, aligned to the input `expert_alignment`. If `num_worst_tokens` is specified, the list
            will be empty.
        handle: the returned communication handle.
    """

    config = Config.get_dispatch_config(num_ranks) if config is None else config

    # Non-cached mode
    assert num_tokens_per_rank is not None and is_token_in_rank is not None and num_tokens_per_expert is not None

    # One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
    assert config.num_sms % 2 == 0
    num_channels: int = config.num_sms // 2

    # Check inputs
    assert x.dim() == 2 and x.is_contiguous()
    assert x.size(1) * x.element_size() % (
        4 * torch.int32.itemsize) == 0  # Ensure x's last dim can be split into int4
    assert is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous()
    assert num_tokens_per_expert.dim() == 1 and num_tokens_per_expert.is_contiguous()
    assert num_tokens_per_expert.size(0) % num_ranks == 0
    # EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
    assert num_tokens_per_rank.shape == (num_ranks,) and num_tokens_per_rank.is_contiguous()

    num_tokens, hidden = x.shape
    num_experts = num_tokens_per_expert.size(0)  # Cached-mode: 0
    num_local_experts = num_experts // num_ranks

    num_topk = topk_idx.size(1)
    assert num_experts > 0
    assert topk_idx.dim() == 2 and topk_idx.is_contiguous() and topk_idx.size(0) == num_tokens
    assert topk_weights.dim() == 2 and topk_weights.is_contiguous() and topk_weights.shape == (
        num_tokens, num_topk) and topk_weights.dtype == torch.float32

    # Allocate tensors
    # TODO(wt): Wait on previous events and allocate on comm stream when adding async functionality

    # For non-cached mode, create handle
    num_memset_int = num_channels * num_ranks * 4
    rank_prefix_matrix = torch.empty([num_ranks, num_ranks], dtype=torch.int32, device='cuda')
    channel_prefix_matrix = torch.empty([num_ranks, num_channels], dtype=torch.int32, device='cuda')

    # Send sizes: `rank_prefix_matrix` and `channel_prefix_matrix`
    #  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
    #  - Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
    # NOTES: no more token dropping in this version
    global moe_recv_counter, moe_recv_expert_counter, moe_recv_rdma_counter
    for counter in [moe_recv_counter, moe_recv_expert_counter, moe_recv_rdma_counter]:
        counter.fill(-1)

    # TODO: launch notify_dispatch kernel

    assert num_worst_tokens == 0  # TODO(wt): Support `num_worst_tokens` later

    # TODO: Wait


moe_recv_counter, moe_recv_expert_counter, moe_recv_rdma_counter = create_moe_recv_counters(
    num_ranks)
