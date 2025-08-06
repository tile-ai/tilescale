import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
from einops import rearrange, einsum
import argparse
import torch.distributed as dist
from tilelang.distributed.utils import init_distributed, dtype_map, perf_fn

tilelang.disable_cache()

# int4* recv_x, float* recv_x_scales, int* recv_src_idx, int64_t* recv_topk_idx, float* recv_topk_weights, int* recv_channel_offset,
# int* send_head, const int4* x, const float* x_scales, const int64_t* topk_idx, const float* topk_weights,
# const bool* is_token_in_rank, const int* channel_prefix_matrix,
# int num_tokens, int num_worst_tokens, int hidden_int4, int num_topk, int num_experts, int num_scales,
# int scale_token_stride, int scale_hidden_stride,
# void** buffer_ptrs, int rank,
# int num_max_send_tokens, int num_recv_buffer_tokens


# Now only support non-cached mode,
# and requires num_worst_tokens > 0
def intranode_dispatch(
    num_tokens, 
    num_worst_tokens, 
    hidden_int4, 
    num_topk, 
    num_experts, 
    num_scales, 
    scale_token_stride, 
    scale_hidden_stride, 
    rank, 
    num_max_send_tokens, 
    num_recv_buffer_tokens,
    num_ranks=1,
    num_sms=20,
    threads=768,
):
    
    num_channels = num_sms // 2
    # TODO: support cached mode and num_worst_tokens == 0
    num_recv_tokens = num_worst_tokens

    @T.prim_func
    def kernel(
            recv_x: T.Tensor([num_recv_tokens, hidden_int4], "int4"),
            recv_x_scales: T.Tensor([num_recv_tokens, num_scales], "float32"),
            recv_src_idx: T.Tensor([num_recv_tokens], "int32"),
            recv_topk_idx: T.Tensor([num_recv_tokens, num_topk], "uint64"),
            recv_topk_weights: T.Tensor([num_recv_tokens, num_topk], "float32"),
            recv_channel_offset: T.Tensor([num_ranks, num_channels], "int32"),
            send_head: T.Tensor([num_tokens, num_ranks], "int32"),
            x: T.Tensor([num_tokens, hidden_int4], "int4"),
            x_scales: T.Tensor([num_tokens, num_scales], "float32"),
            topk_idx: T.Tensor([num_tokens, num_topk], "uint64"),
            topk_weights: T.Tensor([num_tokens, num_topk], "float32"),
            is_token_in_rank: T.Tensor([num_tokens, num_ranks], "bool"),
            channel_prefix_matrix: T.Tensor([num_ranks, num_channels], "int32"),
            # For now we use NVSHMEM to allocate buffer
            # instead of using CUDA IPC on the host side
            # buffer_ptrs: T.Tensor([...], "int32"),
            # channel metadatas (for local rank)
            channel_start_offset: T.Tensor([num_channels, num_ranks], "int32"),
            channel_end_offset: T.Tensor([num_channels, num_ranks], "int32"),
            channel_head_idx: T.Tensor([num_channels, num_ranks], "int32"),
            channel_tail_idx: T.Tensor([num_channels, num_ranks], "int32"),
            # channel buffers (for remote ranks)
            channel_x_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens, hidden_int4], "int4"),
            channel_src_idx_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens], "int32"),
            channel_topk_idx_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens, num_topk], "uint64"),
            channel_topk_weights_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens, num_topk], "float32"),
            channel_x_scales_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens, num_scales], "float32"),
    ):
        with T.Kernel(num_sms, threads=threads) as (bid):

            tx = T.get_thread_binding(0)
            lane_id = tx % 32
            is_sender = bid % 2 == 0
            num_threads_per_rank = threads // num_ranks
            num_channels = num_sms // 2
            responsible_rank = tx // num_threads_per_rank
            responsible_channel = bid // 2
            num_experts_per_rank = num_experts // num_ranks

            # TODO: Add some asserts here
            # EP_DEVICE_ASSERT(num_experts_per_rank > 0 or num_topk == 0);
            # EP_DEVICE_ASSERT(num_topk <= 32);
            # EP_DEVICE_ASSERT((topk_idx == nullptr)  == (topk_weights == nullptr));
            # EP_DEVICE_ASSERT((recv_topk_idx == nullptr) == (recv_topk_weights == nullptr));

            # int target_rank = is_sender ? rank : responsible_rank;
            # auto num_channels_total = num_channels * kNumRanks;
            # auto channel_rank_offset = responsible_channel * kNumRanks + target_rank; 
            target_rank = is_sender * rank + (1 - is_sender) * responsible_rank
            num_channels_total = num_channels * num_ranks
            channel_rank_offset = responsible_channel * num_ranks + target_rank

            if is_sender:
                num_send_warps = threads // 32
                num_send_warps_per_rank = num_send_warps // num_ranks
                send_thread_id = tx
                send_warp_id_in_rank = send_thread_id % num_threads_per_rank // 32

                if lane_id == 0 and send_warp_id_in_rank == 0:
                    if responsible_channel > 0:
                        channel_start_offset[responsible_channel, target_rank] = -channel_prefix_matrix[responsible_rank, responsible_channel - 1] - 1
                    else:
                        channel_start_offset[responsible_channel, target_rank] = 0
                    channel_end_offset[responsible_channel, target_rank] = -channel_prefix_matrix[responsible_rank, responsible_channel] - 1

                # TODO: __syncwarp();
                num_tokens_per_sm = T.ceildiv(num_tokens, num_channels);
                token_start_idx = T.min(num_tokens_per_sm * bid, num_tokens)
                token_end_idx = T.min(token_start_idx + num_tokens_per_sm, num_tokens)

    return kernel

def main(WORLD_SIZE, RANK, LOCAL_RANK, TP_GROUP):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tokens', type=int, default=132, help='number of tokens')
    parser.add_argument('--num_worst_tokens', type=int, default=128, help='number of worst tokens')
    parser.add_argument('--hidden_int4', type=int, default=512, help='hidden int4')
    parser.add_argument('--num_topk', type=int, default=1, help='number of topk')
    parser.add_argument('--num_experts', type=int, default=64, help='number of experts')
    parser.add_argument('--num_scales', type=int, default=1, help='number of scales')
    parser.add_argument('--scale_token_stride', type=int, default=1, help='scale token stride')
    parser.add_argument('--scale_hidden_stride', type=int, default=1, help='scale hidden stride')
    parser.add_argument('--rank', type=int, default=1, help='rank')
    parser.add_argument('--num_max_send_tokens', type=int, default=1, help='number of max send tokens')
    parser.add_argument('--num_recv_buffer_tokens', type=int, default=1, help='number of recv buffer tokens')
    parser.add_argument('--num_ranks', type=int, default=1, help='number of ranks')
    parser.add_argument('--num_sms', type=int, default=20, help='number of sms')
    parser.add_argument('--threads', type=int, default=768, help='number of threads')
    parser.add_argument('--print_source', action='store_true', help='print source')
    args = parser.parse_args()

    func = intranode_dispatch(
        args.num_tokens, 
        args.num_worst_tokens, 
        args.hidden_int4, 
        args.num_topk, 
        args.num_experts, 
        args.num_scales, 
        args.scale_token_stride, 
        args.scale_hidden_stride, 
        args.rank, 
        args.num_max_send_tokens, 
        args.num_recv_buffer_tokens, 
        args.num_ranks, 
        args.num_sms, 
        args.threads
    )
    
    kernel = tilelang.compile(func, pass_configs={"tl.disable_tma_lower": True})

    # Get CUDA Source
    if RANK == 0 and args.print_source:
        print(kernel.get_kernel_source())



if __name__ == "__main__":
    WORLD_SIZE, RANK, LOCAL_RANK, TP_GROUP = init_distributed(return_tp_group=True)
    main(WORLD_SIZE, RANK, LOCAL_RANK, TP_GROUP)
