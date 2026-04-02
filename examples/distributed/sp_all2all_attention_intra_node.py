import torch
import tilelang
import tilelang.language as T
from typing import List
from dataclasses import dataclass
from cuda import cudart
from tilelang.distributed.utils import CUDA_CHECK


@tilelang.jit
def barrier_all_blocks_sys_kernel(num_local_rank):
    @T.prim_func
    def main(barrier: T.Tensor((num_local_rank), "int32")):
        with T.Kernel(1, threads=32):
            T.barrier_blocks(barrier)

    return main


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
    compile_flags=[
        "-O3",
        "-Wno-deprecated-declarations",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--ptxas-options=-v,--register-usage-level=10",
        "-DNDEBUG",
    ],
)
def flashattn_packed(
    batch_size,
    groups,
    UQ,
    UKV,
    heads,
    dim,
    is_causal,
    enable_zig_zag,
    rank,
    num_ranks,
    block_M=64,
    block_N=64,
    num_stages=1,
    threads=128,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    packed_heads = heads + 2 * head_kv
    packed_shape = [UQ, packed_heads, dim]
    o_shape = [UQ, heads, dim]
    dtype = "float16"
    accum_dtype = "float"

    @T.macro
    def inner_packed(
        Packed_unpad: T.Tensor(packed_shape, dtype),
        Output_unpad: T.Tensor(o_shape, dtype),
        Q_shared: T.SharedBuffer([block_M, dim], dtype),
        K_shared: T.SharedBuffer([block_N, dim], dtype),
        V_shared: T.SharedBuffer([block_N, dim], dtype),
        O_shared: T.SharedBuffer([block_M, dim], dtype),
        acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
        acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
        acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
        scores_max: T.FragmentBuffer([block_M], accum_dtype),
        scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
        scores_sum: T.FragmentBuffer([block_M], accum_dtype),
        logsum: T.FragmentBuffer([block_M], accum_dtype),
        q_load_start_idx: T.int32,
        q_write_start_idx: T.int32,
        k_start_idx: T.int32,
        q_current_seqlen: T.int32,
        k_current_seqlen: T.int32,
        bx: T.int32,
        head_idx: T.int32,
        kv_head_idx: T.int32,
        global_offset_q: T.int32,
        kv_len_per_sp_block: T.int32,
    ):
        q_head_offset = 0
        k_head_offset = heads
        v_head_offset = heads + head_kv
        q_token_offset = rank * q_current_seqlen

        T.copy(
            Packed_unpad[
                q_load_start_idx + q_token_offset + bx * block_M : q_load_start_idx + q_token_offset + (bx + 1) * block_M,
                q_head_offset + head_idx,
                :,
            ],
            Q_shared,
        )

        T.fill(acc_o, 0)
        T.fill(logsum, 0)
        T.fill(scores_max, -T.infinity(accum_dtype))

        prefix_len = k_current_seqlen - q_current_seqlen * num_ranks
        loop_range = (
            T.ceildiv(prefix_len + global_offset_q + (bx + 1) * block_M, block_N) if is_causal else T.ceildiv(k_current_seqlen, block_N)
        )

        for k in T.Pipelined(loop_range, num_stages=num_stages):
            sp_block_idx = (k * block_N) // kv_len_per_sp_block
            wait_rank = sp_block_idx if sp_block_idx < num_ranks else 2 * num_ranks - sp_block_idx - 1
            kv_load_offset = (
                (k * block_N) % kv_len_per_sp_block
                + sp_block_idx // num_ranks * kv_len_per_sp_block
                + wait_rank * (k_current_seqlen // num_ranks)
            )
            T.copy(
                Packed_unpad[k_start_idx + kv_load_offset : k_start_idx + kv_load_offset + block_N, k_head_offset + kv_head_idx, :],
                K_shared,
            )

            if is_causal:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(
                        (prefix_len + global_offset_q + bx * block_M + i < k * block_N + j)
                        or (bx * block_M + i >= q_current_seqlen or k * block_N + j >= k_current_seqlen),
                        -1e9,
                        0,
                    )
            else:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(
                        (bx * block_M + i >= q_current_seqlen or k * block_N + j >= k_current_seqlen),
                        -1e9,
                        0,
                    )

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
                Packed_unpad[k_start_idx + kv_load_offset : k_start_idx + kv_load_offset + block_N, v_head_offset + kv_head_idx, :],
                V_shared,
            )

            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] /= logsum[i]
        T.copy(acc_o, O_shared)

        for i, d in T.Parallel(block_M, dim):
            if bx * block_M + i < q_current_seqlen:
                Output_unpad[q_write_start_idx + bx * block_M + i, head_idx, d] = O_shared[i, d]

    @T.prim_func
    def main_packed(
        Packed_unpad: T.Tensor(packed_shape, dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], "int32"),
        cu_seqlens_k: T.Tensor([batch_size + 1], "int32"),
        max_seqlen_q: T.int32,
        Output_unpad: T.Tensor(o_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(max_seqlen_q, block_M), heads, batch_size, threads=threads) as (bx, by, bz):
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

            q_write_start_idx = cu_seqlens_q[batch_idx]
            q_load_start_idx = cu_seqlens_k[batch_idx]
            k_start_idx = cu_seqlens_k[batch_idx]
            q_end_idx = cu_seqlens_q[batch_idx + 1]
            k_end_idx = cu_seqlens_k[batch_idx + 1]

            q_current_seqlen = q_end_idx - q_write_start_idx
            k_current_seqlen = k_end_idx - k_start_idx

            global_offset_q = q_current_seqlen * rank
            kv_len_per_sp_block = k_current_seqlen // num_ranks

            inner_packed(
                Packed_unpad,
                Output_unpad,
                Q_shared,
                K_shared,
                V_shared,
                O_shared,
                acc_s,
                acc_s_cast,
                acc_o,
                scores_max,
                scores_max_prev,
                scores_scale,
                scores_sum,
                logsum,
                q_load_start_idx,
                q_write_start_idx,
                k_start_idx,
                q_current_seqlen,
                k_current_seqlen,
                bx,
                head_idx,
                kv_head_idx,
                global_offset_q,
                kv_len_per_sp_block,
            )

    @T.prim_func
    def main_packed_zigzag(
        Packed_unpad: T.Tensor(packed_shape, dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], "int32"),
        cu_seqlens_k: T.Tensor([batch_size + 1], "int32"),
        max_seqlen_q: T.int32,
        Output_unpad: T.Tensor(o_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(max_seqlen_q, block_M), heads, batch_size, threads=threads) as (bx, by, bz):
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

            q_write_start_idx = cu_seqlens_q[batch_idx]
            q_load_start_idx = cu_seqlens_k[batch_idx]
            k_start_idx = cu_seqlens_k[batch_idx]
            q_end_idx = cu_seqlens_q[batch_idx + 1]
            k_end_idx = cu_seqlens_k[batch_idx + 1]

            q_current_seqlen = q_end_idx - q_write_start_idx
            k_current_seqlen = k_end_idx - k_start_idx

            half_q_shard_len = q_current_seqlen // 2
            global_offset_q = (
                rank * half_q_shard_len if bx * block_M < half_q_shard_len else q_current_seqlen * num_ranks - (rank + 2) * half_q_shard_len
            )
            kv_len_per_sp_block = k_current_seqlen // (2 * num_ranks)

            inner_packed(
                Packed_unpad,
                Output_unpad,
                Q_shared,
                K_shared,
                V_shared,
                O_shared,
                acc_s,
                acc_s_cast,
                acc_o,
                scores_max,
                scores_max_prev,
                scores_scale,
                scores_sum,
                logsum,
                q_load_start_idx,
                q_write_start_idx,
                k_start_idx,
                q_current_seqlen,
                k_current_seqlen,
                bx,
                head_idx,
                kv_head_idx,
                global_offset_q,
                kv_len_per_sp_block,
            )

    return main_packed if not enable_zig_zag else main_packed_zigzag


# def packed_sp_all2all_attention(
#     packed_qkv: torch.Tensor,
#     output: torch.Tensor,
#     cu_seqlens_q: torch.Tensor,
#     cu_seqlens_k: torch.Tensor,
#     max_seqlen_q: int,
#     batch_size: int,
#     q_heads: int,
#     kv_heads: int,
#     groups: int,
#     rank: int,
#     num_ranks: int,
#     is_causal: bool = True,
#     enable_zig_zag: bool = True,
#     block_M: int = 128,
#     block_N: int = 128,
#     num_stages: int = 2,
#     threads: int = 256,
# ):
#     kernel = flashattn_packed(
#         batch_size,
#         groups,
#         packed_qkv.shape[0],
#         packed_qkv.shape[0],
#         q_heads,
#         packed_qkv.shape[-1],
#         is_causal,
#         enable_zig_zag,
#         rank,
#         num_ranks,
#         block_M=block_M,
#         block_N=block_N,
#         num_stages=num_stages,
#         threads=threads,
#     )
#     kernel(packed_qkv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, output)
#     return output


@dataclass
class SPAll2AllAttentionContextIntraNode:
    ag_packed_buffers: List[torch.Tensor]
    ag_packed_buffer: torch.Tensor
    attn_output_buffer: torch.Tensor
    ag_stream: torch.cuda.Stream
    barrier: torch.Tensor


def create_sp_all2all_attention_context_intra_node(
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
    packed_heads = q_head + 2 * kv_head
    ag_packed_buffers = tilelang.tensor(
        (batch_size * max_seqlen_k, packed_heads, head_dim),
        dtype=input_dtype,
        allocator=allocator,
        return_peers=True,
    )
    ag_packed_buffer = ag_packed_buffers[rank]

    attn_output_buffer = torch.empty(
        batch_size * max_seqlen_k,
        q_head,
        head_dim,
        dtype=output_dtype,
        device=device,
    )

    barrier = tilelang.tensor((world_size), dtype=torch.int32, allocator=allocator)
    ag_stream = torch.cuda.Stream()

    return SPAll2AllAttentionContextIntraNode(
        ag_packed_buffers=ag_packed_buffers,
        ag_packed_buffer=ag_packed_buffer,
        attn_output_buffer=attn_output_buffer,
        ag_stream=ag_stream,
        barrier=barrier,
    )


def barrier_all_on_stream(barrier: torch.Tensor, stream: torch.cuda.Stream, world_size: int):
    barrier_all_blocks_sys_func = barrier_all_blocks_sys_kernel(world_size)
    with torch.cuda.stream(stream):
        barrier_all_blocks_sys_func(barrier)


def cp_engine_producer_packed_all2all(
    packed_shards: list[torch.Tensor],
    packed_buffer: torch.Tensor,
    packed_buffers: list[torch.Tensor],
    cu_seqlens_k: torch.Tensor,
    rank: int,
    world_size: int,
    ag_stream: torch.cuda.Stream,
    compute_stream: torch.cuda.Stream,
    barrier: torch.Tensor,
):
    assert packed_buffer.is_contiguous()
    assert packed_shards[rank].is_contiguous()

    _, packed_heads_total, head_dim = packed_shards[rank].shape
    assert packed_heads_total % world_size == 0
    packed_heads_per_rank = packed_heads_total // world_size
    batch_size = cu_seqlens_k.shape[0] - 1
    dtype_itemsize = packed_shards[rank].dtype.itemsize
    src_token_bytes = packed_heads_total * head_dim * dtype_itemsize
    dst_token_bytes = packed_heads_per_rank * head_dim * dtype_itemsize

    def _cp_engine_copy_data(dst_ptr, src_ptr, cp_size, stream):
        (err,) = cudart.cudaMemcpyAsync(
            dst_ptr,
            src_ptr,
            cp_size,
            cudart.cudaMemcpyKind.cudaMemcpyDefault,
            stream.cuda_stream,
        )
        CUDA_CHECK(err)

    def _cp_engine_copy_2d(dst_ptr, src_ptr, width_bytes, height, src_pitch, dst_pitch, stream):
        (err,) = cudart.cudaMemcpy2DAsync(
            dst_ptr,
            dst_pitch,
            src_ptr,
            src_pitch,
            width_bytes,
            height,
            cudart.cudaMemcpyKind.cudaMemcpyDefault,
            stream.cuda_stream,
        )
        CUDA_CHECK(err)

    # self copy: src rank == local rank
    with torch.cuda.stream(compute_stream):
        for i in range(batch_size):
            cu_seqlens_k_start = cu_seqlens_k[i].item()
            cu_seqlens_k_end = cu_seqlens_k[i + 1].item()
            seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
            local_seq_len = seqlen_k // world_size
            src_token_start = cu_seqlens_k_start // world_size
            dst_token_start = cu_seqlens_k_start + rank * local_seq_len

            src_head_offset_bytes = rank * packed_heads_per_rank * head_dim * dtype_itemsize
            src_ptr = packed_shards[rank].data_ptr() + src_token_start * src_token_bytes + src_head_offset_bytes
            dst_ptr = packed_buffers[rank].data_ptr() + dst_token_start * dst_token_bytes
            _cp_engine_copy_2d(
                dst_ptr,
                src_ptr,
                dst_token_bytes,
                local_seq_len,
                src_token_bytes,
                dst_token_bytes,
                compute_stream,
            )

    # pull from remote src ranks into local destination sequence slots
    with torch.cuda.stream(ag_stream):
        for i in range(batch_size):
            cu_seqlens_k_start = cu_seqlens_k[i].item()
            cu_seqlens_k_end = cu_seqlens_k[i + 1].item()
            seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
            local_seq_len = seqlen_k // world_size
            src_token_start = cu_seqlens_k_start // world_size
            src_head_offset_bytes = rank * packed_heads_per_rank * head_dim * dtype_itemsize
            for offset in range(1, world_size):
                src_rank = (rank + offset) % world_size
                dst_token_start = cu_seqlens_k_start + src_rank * local_seq_len
                src_ptr = packed_shards[src_rank].data_ptr() + src_token_start * src_token_bytes + src_head_offset_bytes
                dst_ptr = packed_buffers[rank].data_ptr() + dst_token_start * dst_token_bytes
                _cp_engine_copy_2d(
                    dst_ptr,
                    src_ptr,
                    dst_token_bytes,
                    local_seq_len,
                    src_token_bytes,
                    dst_token_bytes,
                    ag_stream,
                )

    barrier_all_on_stream(barrier, ag_stream, world_size)
    compute_stream.wait_stream(ag_stream)


def fused_sp_all2all_attn_intra_node(
    ctx: SPAll2AllAttentionContextIntraNode,
    packed_qkv_shards: list[torch.Tensor],
    output: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    rank: int,
    world_size: int,
    q_head: int,
    kv_head: int,
    is_causal: bool = True,
    enable_zig_zag: bool = True,
    print_source: bool = False,
):
    block_M = 128
    block_N = 128
    num_stages = 2
    threads = 256

    compute_stream = torch.cuda.current_stream()
    ag_packed = ctx.ag_packed_buffers[rank]

    ctx.ag_stream.wait_stream(compute_stream)
    cp_engine_producer_packed_all2all(
        packed_qkv_shards,
        ag_packed,
        ctx.ag_packed_buffers,
        cu_seqlens_k,
        rank,
        world_size,
        ctx.ag_stream,
        compute_stream,
        ctx.barrier,
    )

    head_dim = packed_qkv_shards[rank].shape[-1]
    groups = q_head // kv_head
    batch = cu_seqlens_q.shape[0] - 1

    with torch.cuda.stream(compute_stream):
        kernel = flashattn_packed(
            batch,
            groups,
            ag_packed.shape[0],
            ag_packed.shape[0],
            q_head,
            head_dim,
            is_causal,
            enable_zig_zag,
            rank,
            world_size,
            block_M=block_M,
            block_N=block_N,
            num_stages=num_stages,
            threads=threads,
        )

        if rank == 0 and print_source:
            print(kernel.get_kernel_source())

        kernel(ag_packed, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, output)

    compute_stream.wait_stream(ctx.ag_stream)
    barrier_all_on_stream(ctx.barrier, compute_stream, world_size)
