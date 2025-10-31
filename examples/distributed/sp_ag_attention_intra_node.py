import torch
import tilelang
import tilelang.language as T
from typing import List
import math
from dataclasses import dataclass

tilelang.disable_cache()

@tilelang.jit
def barrier_all_blocks_sys_kernel(num_local_rank,):

    @T.prim_func
    def main(
        barrier: T.Tensor((num_local_rank), "int32"),
    ):
        with T.Kernel(1, threads=32) as (bid):
            T.barrier_all_blocks_sys(barrier)

    return main


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def flashattn(batch_size,
              groups,
              UQ,
              UKV,
              heads,
              dim,
              is_causal,
              block_M=64,
              block_N=64,
              num_stages=1,
              threads=128):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    q_shape = [UQ, heads, dim]
    kv_shape = [UKV, head_kv, dim]
    o_shape = [UQ, heads, dim]
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def main(
            Q_unpad: T.Tensor(q_shape, dtype),
            K_unpad: T.Tensor(kv_shape, dtype),
            V_unpad: T.Tensor(kv_shape, dtype),
            cu_seqlens_q: T.Tensor([batch_size + 1], "int32"),
            cu_seqlens_k: T.Tensor([batch_size + 1], "int32"),
            max_seqlen_q: T.int32,
            Output_unpad: T.Tensor(o_shape, dtype),
    ):
        with T.Kernel(
                T.ceildiv(max_seqlen_q, block_M), heads, batch_size,
                threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            batch_idx = bz
            head_idx = by
            kv_head_idx = head_idx // groups

            q_start_idx = cu_seqlens_q[batch_idx]
            k_start_idx = cu_seqlens_k[batch_idx]
            v_start_idx = cu_seqlens_k[batch_idx]
            q_end_idx = cu_seqlens_q[batch_idx + 1]
            k_end_idx = cu_seqlens_k[batch_idx + 1]
            v_end_idx = cu_seqlens_k[batch_idx + 1]

            q_current_seqlen = q_end_idx - q_start_idx
            k_current_seqlen = k_end_idx - k_start_idx
            v_current_seqlen = v_end_idx - v_start_idx

            T.copy(
                Q_unpad[q_start_idx + bx * block_M:q_start_idx + (bx + 1) * block_M, head_idx, :],
                Q_shared)
            for i, d in T.Parallel(block_M, dim):
                if bx * block_M + i >= q_current_seqlen:
                    Q_shared[i, d] = 0

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(k_current_seqlen, block_N), T.ceildiv(
                    (bx + 1) * block_M, block_N)) if is_causal else T.ceildiv(k_current_seqlen, block_N))

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(
                    K_unpad[k_start_idx + k * block_N:k_start_idx + (k + 1) * block_N,
                            kv_head_idx, :], K_shared)
                for i, d in T.Parallel(block_N, dim):
                    if k * block_N + i >= k_current_seqlen:
                        K_shared[i, d] = 0

                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else((bx * block_M + i < k * block_N + j) or
                                                     (bx * block_M + i >= q_current_seqlen or
                                                      k * block_N + j >= k_current_seqlen),
                                                     -1e9, 0)
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else((bx * block_M + i >= q_current_seqlen or
                                                      k * block_N + j >= k_current_seqlen),
                                                     -1e9, 0)

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(
                    V_unpad[v_start_idx + k * block_N:v_start_idx + (k + 1) * block_N,
                            kv_head_idx, :], V_shared)
                for i, d in T.Parallel(block_N, dim):
                    if k * block_N + i >= v_current_seqlen:
                        V_shared[i, d] = 0

                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)

            for i, d in T.Parallel(block_M, dim):
                if bx * block_M + i < q_current_seqlen:
                    Output_unpad[q_start_idx + bx * block_M + i, head_idx, d] = O_shared[i, d]

    return main

@dataclass
class SPAllGatherAttentionContextIntraNode:
    ag_k_buffers: List[torch.Tensor]
    ag_k_buffer: torch.Tensor
    # ag_k_buffers_ptr: torch.Tensor
    ag_v_buffers: List[torch.Tensor]
    ag_v_buffer: torch.Tensor
    # ag_v_buffers_ptr: torch.Tensor
    attn_output_buffer: torch.Tensor
    ag_stream: torch.cuda.Stream
    barrier: torch.Tensor


def create_sp_ag_attention_context_intra_node(
    batch_size,
    q_head,
    kv_head,
    max_seqlen_k,
    max_q_shard_len,
    head_dim,
    input_dtype,
    output_dtype,
    rank,
    world_size,
    device,
    allocator,
):
    ag_k_buffers = tilelang.tensor((batch_size * max_seqlen_k, kv_head, head_dim), dtype=input_dtype, allocator=allocator, return_peers=True)
    ag_k_buffer = ag_k_buffers[rank]

    ag_v_buffers = tilelang.tensor((batch_size * max_seqlen_k, kv_head, head_dim), dtype=input_dtype, allocator=allocator, return_peers=True)
    ag_v_buffer = ag_v_buffers[rank]

    attn_output_buffer = torch.empty(
        batch_size * max_q_shard_len,
        q_head,
        head_dim,
        dtype=output_dtype,
        device=device,
    )
    
    barrier = tilelang.tensor((world_size), dtype=torch.int32, allocator=allocator)

    # stream for copy
    ag_stream = torch.cuda.Stream()

    ctx = SPAllGatherAttentionContextIntraNode(ag_k_buffers=ag_k_buffers, ag_k_buffer=ag_k_buffer,
                                               ag_v_buffers=ag_v_buffers, ag_v_buffer=ag_v_buffer,
                                               attn_output_buffer=attn_output_buffer, ag_stream=ag_stream,
                                               barrier=barrier)

    return ctx


def barrier_all_on_stream(barrier: torch.Tensor, stream: torch.cuda.Stream, world_size: int):
    barrier_all_blocks_sys_func = barrier_all_blocks_sys_kernel(world_size)
    barrier_all_blocks_sys_func(barrier, stream=stream.cuda_stream)

def cp_engine_producer_kv_all_gather(
    k_shard: torch.Tensor,  # [total_kv_shard, kv_head, head_dim]
    v_shard: torch.Tensor,  # [total_kv_shard, kv_head, head_dim]
    k_buffer: torch.Tensor,  # [total_kv, kv_head, head_dim]
    v_buffer: torch.Tensor,  # [total_kv, kv_head, head_dim]
    k_buffers: List[torch.Tensor],
    v_buffers: List[torch.Tensor],
    cu_seqlens_k: torch.Tensor,  # kv_full_lens
    rank: int,
    world_size: int,
    ag_stream: torch.cuda.Stream,
    compute_stream: torch.cuda.Stream,
    barrier: torch.Tensor,
):
    assert k_buffer.is_contiguous()
    assert v_buffer.is_contiguous()
    assert k_shard.is_contiguous()
    assert v_shard.is_contiguous()

    total_kv_shard, kv_head, head_dim = k_shard.shape
    batch_size = cu_seqlens_k.shape[0] - 1

    byte_per_token = kv_head * head_dim * k_shard.dtype.itemsize

    def _cp_engine_copy_data(dst, src, stream):
        with torch.cuda.stream(stream):
            dst.copy_(src)

    # local copy in compute stream
    with torch.cuda.stream(compute_stream):
        for i in range(batch_size):
            cu_seqlens_k_start = cu_seqlens_k[i].item()
            cu_seqlens_k_end = cu_seqlens_k[i + 1].item()
            seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
            k_shard_len = seqlen_k // world_size
            byte_start = cu_seqlens_k_start * byte_per_token
            byte_per_rank = k_shard_len * byte_per_token
            cp_size = byte_per_rank

            k_dst = k_buffers[rank][cu_seqlens_k_start + rank * k_shard_len : cu_seqlens_k_start + (rank + 1) * k_shard_len, :, :]
            k_src = k_shard[cu_seqlens_k_start // world_size : cu_seqlens_k_start // world_size + k_shard_len, :, :]
            _cp_engine_copy_data(k_dst, k_src, compute_stream)

            v_dst = v_buffers[rank][cu_seqlens_k_start + rank * k_shard_len : cu_seqlens_k_start + (rank + 1) * k_shard_len, :, :]
            v_src = v_shard[cu_seqlens_k_start // world_size : cu_seqlens_k_start // world_size + k_shard_len, :, :]
            _cp_engine_copy_data(v_dst, v_src, compute_stream)

    barrier_all_on_stream(barrier, compute_stream, world_size)
    ag_stream.wait_stream(compute_stream)

    with torch.cuda.stream(ag_stream):
        for i in range(batch_size):
            cu_seqlens_k_start = cu_seqlens_k[i].item()
            cu_seqlens_k_end = cu_seqlens_k[i + 1].item()
            seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
            k_shard_len = seqlen_k // world_size
            byte_start = cu_seqlens_k_start * byte_per_token
            byte_per_rank = k_shard_len * byte_per_token
            cp_size = byte_per_rank
            for offset in range(1, world_size):
                src_rank = (rank + offset) % world_size

                k_src = k_buffers[src_rank][cu_seqlens_k_start + src_rank * k_shard_len : cu_seqlens_k_start + (src_rank + 1) * k_shard_len, :, :]
                k_dst = k_buffers[rank][cu_seqlens_k_start + src_rank * k_shard_len : cu_seqlens_k_start + (src_rank + 1) * k_shard_len, :, :]
                _cp_engine_copy_data(k_dst, k_src, ag_stream)

                v_src = v_buffers[src_rank][cu_seqlens_k_start + src_rank * k_shard_len : cu_seqlens_k_start + (src_rank + 1) * k_shard_len, :, :]
                v_dst = v_buffers[rank][cu_seqlens_k_start + src_rank * k_shard_len : cu_seqlens_k_start + (src_rank + 1) * k_shard_len, :, :]
                _cp_engine_copy_data(v_dst, v_src, ag_stream)

    barrier_all_on_stream(barrier, ag_stream, world_size)
    compute_stream.wait_stream(ag_stream)



def fused_sp_ag_attn_intra_node(
    ctx: SPAllGatherAttentionContextIntraNode,
    q_shard: torch.Tensor,  # [total_q_shard, q_head, head_dim]
    k_shard: torch.Tensor,  # [total_kv_shard, kv_head, head_dim]
    v_shard: torch.Tensor,  # [total_kv_shard, kv_head, head_dim]
    output: torch.Tensor,  # [total_q_shard, q_head, head_dim]
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    rank: int,
    world_size: int,
    is_causal: bool = True,
    enable_zig_zag: bool = True,
):
    
    BLOCK_M = 128
    BLOCK_N = 128
    threads = 256
    q_tokens = q_shard.shape[0]
    assert ctx.ag_k_buffers[rank].shape[0] == ctx.ag_v_buffers[rank].shape[0]
    kv_tokens = ctx.ag_k_buffers[rank].shape[0]
    q_head = q_shard.shape[1]
    kv_head = k_shard.shape[1]
    batch = cu_seqlens_q.shape[0] - 1

    compute_stream = torch.cuda.current_stream()
    ag_k = ctx.ag_k_buffers[rank]
    ag_v = ctx.ag_v_buffers[rank]

    ctx.ag_stream.wait_stream(compute_stream)
    # kv all gather
    cp_engine_producer_kv_all_gather(
        k_shard,
        v_shard,
        ag_k,
        ag_v,
        ctx.ag_k_buffers,
        ctx.ag_v_buffers,
        cu_seqlens_k,
        rank,
        world_size,
        ctx.ag_stream,
        compute_stream,
        ctx.barrier,
    )

    # flash attn
    stage = 3 if is_causal else 1
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = q_shard.shape[-1], k_shard.shape[-1]
    HEAD_DIM_V = v_shard.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    sm_scale = 1 / math.sqrt(HEAD_DIM_Q)

    with torch.cuda.stream(compute_stream):
        kernel = flashattn(
            batch,
            q_head // kv_head,
            q_tokens,
            kv_tokens,
            q_head,
            HEAD_DIM_Q,
            is_causal,
            block_M=BLOCK_M,
            block_N=BLOCK_N,
            num_stages=1,
            threads=threads)
        
        kernel(q_shard, ag_k, ag_v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, output, stream=compute_stream.cuda_stream)
    
    compute_stream.wait_stream(ctx.ag_stream)
