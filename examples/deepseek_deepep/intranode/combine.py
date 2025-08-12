# For intranode only

# Currently we focus on bf16 w/o cache
# TODO(wt): Support cache mode and fp8 w/ scale

import os
import torch
import torch.distributed as dist
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
from typing import Optional, Tuple, Union
from ..utils import Config, unpack_bias

tilelang.disable_cache()

group_size: int = int(os.environ.get("WORLD_SIZE", 1))  # num_PE


# TODO(wt): Add async functionality
def combine(
    x: torch.Tensor, 
    handle: Tuple,  # Must provide
    topk_weights: Optional[torch.Tensor] = None,
    bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
    config: Optional[Config] = None,
    num_ranks: int = group_size
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Combine (reduce) tokens (addition **without** weights) from different ranks, currently only support intranode settings.
    Intranode kernels require all the ranks should be visible via NVLink.
    Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
        index should be visible via RDMA.

    Arguments:
        x: `[num_tokens, hidden]` with `torch.bfloat16`, the tokens to send for reducing to its original ranks.
        handle: a must-set communication handle obtained from the dispatch function.
        topk_weights: `[num_tokens, num_topk]` with `torch.float`, the tokens' top-k weights for reducing to its original ranks.
        config: the performance tuning config. If not provided, will use the default settings defined in DeepEP.
        num_ranks: the number of ranks.
        
    Returns:
        recv_x: the reduced token from its dispatched ranks.
        recv_topk_weights: the reduced top-k weights from its dispatch ranks.
    """
    config = Config.get_combine_config(group_size) if config is not None else config
    
    # TODO: Implement internode combine here
    
    # NOTE: the second `_` is for the sending side, so we should use the third one
    rank_prefix_matrix, _, channel_prefix_matrix, src_idx, is_recv_token_in_rank, send_head = handle
    bias_0, bias_1 = unpack_bias(bias)
    
    # Check inputs
    assert x.dim() == 2 and x.is_contiguous()
    assert src_idx.dim() == 1 and src_idx.is_contiguous() and src_idx.dtype == torch.int32
    assert send_head.dim() == 2 and send_head.is_contiguous() and send_head.scalar_type() == torch.int32
    assert rank_prefix_matrix.dim() == 2 and rank_prefix_matrix.is_contiguous() and rank_prefix_matrix.scalar_type() == torch.int32
    assert channel_prefix_matrix.dim() == 2 and channel_prefix_matrix.is_contiguous() and channel_prefix_matrix.scalar_type() == torch.int32
    
    # NOTE: Each channel use 2 blocks, even-numbered for sending and odd-numbered for receiving.
    assert config.num_sms % 2 == 0
    num_channels: int = config.num_sms // 2
    
    num_tokens, hidden = x.shape
    num_recv_tokens = send_head.size(0)
    
    assert src_idx.size(0) == num_tokens
    assert send_head.size(1) == num_ranks
    assert rank_prefix_matrix.shape == (num_ranks, num_ranks)
    assert channel_prefix_matrix.shape == (num_ranks, num_channels)
    assert hidden * x.element_size() % (4 * torch.int32.itemsize()) == 0  # x's last dim can be split into int4
    
    # Allocate tensors
    # TODO(wt): Wait on previous events and allocate on comm stream when adding async functionality
    assert topk_weights is not None  # [QST] When will topk_weights be None?
    assert topk_weights.dim() == 2 and topk_weights.is_contiguous() and topk_weights.dtype == torch.float32 and topk_weights.size(0) == num_tokens
    num_topk = topk_weights.size(1)
    recv_topk_weights = torch.empty_like(topk_weights, [num_recv_tokens, num_topk])
    
    # Launch barrier and reset queue head and tail
    # TODO(wt): launch cache_notify_combine kernel
    
    assert bias_0.dim() == 2 and bias_0.is_contiguous() and bias_0.dtype == x.dtype and bias_0.shape == (num_recv_tokens, hidden)
    if bias_1 is not None:
        assert bias_1.dim() == 2 and bias_1.is_contiguous() and bias_1.dtype == x.dtype and bias_1.shape == (num_recv_tokens, hidden)
    
    recv_x = torch.empty_like(x, [num_recv_tokens, hidden])    
    # TODO(wt): launch combine kernel
        
    # TODO(wt): Wait stream and switch back to compute stream when adding async functionality
    return recv_x, recv_topk_weights