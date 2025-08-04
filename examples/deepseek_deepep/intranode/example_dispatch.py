import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
from einops import rearrange, einsum
import argparse


# int4* recv_x, float* recv_x_scales, int* recv_src_idx, int64_t* recv_topk_idx, float* recv_topk_weights, int* recv_channel_offset,
# int* send_head, const int4* x, const float* x_scales, const int64_t* topk_idx, const float* topk_weights,
# const bool* is_token_in_rank, const int* channel_prefix_matrix,
# int num_tokens, int num_worst_tokens, int hidden_int4, int num_topk, int num_experts, int num_scales,
# int scale_token_stride, int scale_hidden_stride,
# void** buffer_ptrs, int rank,
# int num_max_send_tokens, int num_recv_buffer_tokens

@tilelang.jit(out_idx=[6])
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
    @T.prim_func
    def kernel(
            recv_x: T.Tensor([], "int4"),
            recv_x_scales: T.Tensor([], "float32"),
            recv_src_idx: T.Tensor([], "int32"),
            recv_topk_idx: T.Tensor([], "uint64"),
            recv_topk_weights: T.Tensor([], "float32"),
            recv_channel_offset: T.Tensor([], "int32"),
            send_head: T.Tensor([], "int32"),
            x: T.Tensor([], "int4"),
            x_scales: T.Tensor([], "float32"),
            topk_idx: T.Tensor([], "uint64"),
            topk_weights: T.Tensor([], "float32"),
            is_token_in_rank: T.Tensor([], "bool"),
            channel_prefix_matrix: T.Tensor([], "int32"),
            buffer_ptrs: T.Tensor([], "int32"),
    ):
        with T.Kernel(num_sms, threads=threads) as (bid):
            recv_x_shared = T.alloc_shared([num_tokens, hidden_int4], "int4")
            recv_x_scales_shared = T.alloc_shared([num_tokens], "float32")
            recv_src_idx_shared = T.alloc_shared([num_tokens], "int32")
            recv_topk_idx_shared = T.alloc_shared([num_tokens], "uint64")
            recv_topk_weights_shared = T.alloc_shared([num_tokens], "float32")
            recv_channel_offset_shared = T.alloc_shared([num_tokens], "int32")
            send_head_shared = T.alloc_shared([num_tokens], "int32")

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
                # constexpr int num_send_warps = kNumThreads / 32;
                # constexpr int num_send_warps_per_rank = num_send_warps / kNumRanks;
                # const auto send_thread_id = thread_id;
                # const auto send_warp_id_in_rank = send_thread_id % num_threads_per_rank / 32;

                num_send_warps = threads // 32
                num_send_warps_per_rank = num_send_warps // num_ranks
                send_thread_id = tx
                send_warp_id_in_rank = send_thread_id % num_threads_per_rank // 32

                # if (lane_id == 0 and send_warp_id_in_rank == 0) {
                #     int value = responsible_channel > 0 ? channel_prefix_matrix[responsible_rank * num_channels + responsible_channel - 1] : 0;
                #     st_relaxed_sys_global(channel_start_offset.buffer(), -value - 1);
                #     value = channel_prefix_matrix[responsible_rank * num_channels + responsible_channel];
                #     st_relaxed_sys_global(channel_end_offset.buffer(), -value - 1);
                # }
                # __syncwarp();

                if lane_id == 0 and send_warp_id_in_rank == 0:
                    value = responsible_channel > 0 ? channel_prefix_matrix[responsible_rank * num_channels + responsible_channel - 1] : 0
                    send_head_shared[tx] = -value - 1
                else:
                    send_head_shared[tx] = 0
                T.sync_threads()

                

    return kernel

def main():
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
    args = parser.parse_args()

    kernel = intranode_dispatch(**vars(args))


if __name__ == "__main__":
    main()
