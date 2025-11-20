# ruff: noqa
import torch
import tilelang
from tilelang import language as T
from tilelang.engine.callback import register_cuda_postproc_callback
import argparse
import math
from typing import Optional, Tuple
from tilelang.distributed.utils import dsize_map, dtype_map
from tilelang.carver.arch import driver
from sparse_mla_decode_utils import _quantize_k_cache_fp8
import flash_mla
from flash_mla import flash_mla_with_kvcache, get_mla_metadata

tilelang.disable_cache()

@tilelang.jit(
    compile_flags=[
        "-O3", "-Wno-deprecated-declarations", "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "--ptxas-options=-v,--register-usage-level=10", "-DNDEBUG"
    ],
)
def sparse_mla_fwd(
    b,
    s_q, 
    h_q,
    h_kv,
    num_blocks, 
    page_block_size, 
    nope_dim,
    rope_dim,
    max_num_blocks_per_seq,
    topk,
    dtype,
    cluster,
    threads=384,
):
    # assert dim == tilelang.math.next_power_of_2(
    #     dim), f"haven't check padding correctness yet, dim={dim}"
    # assert tail_dim == tilelang.math.next_power_of_2(
    #     tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
    # assert is_causal == True, 'non-casual is not supported'
    # assert topk % block_I == 0, 'otherwise will load some index=0 thus causing wrong kv to be loaded'
    # if sm_scale is None:
    #     sm_scale = (1.0 / (dim + tail_dim))**0.5 * 1.44269504  # log2(e)
    # else:
    #     sm_scale = sm_scale * 1.44269504  # log2(e)

    # head_kv = heads // kv_group
    # q_shape = [batch, seq_len, heads, dim + tail_dim]
    # kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    # o_shape = [batch, seq_len, heads, dim]
    # indices_shape = [batch, seq_len, kv_group, topk]
    # lse_shape = [batch, seq_len, heads]
    # indices_dtype = "int32"
    # dtype = "bfloat16"
    # accum_dtype = "float"

    # G = kv_group
    # H = head_kv
    # padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    # if padded_H != H:
    #     assert kv_group == 1, 'here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)'
    # BI = block_I
    # NI = tilelang.cdiv(topk, block_I)
    # assert NI % 2 == 0, 'NI should be a multiple of 2'
    # D = dim
    # D_tail = tail_dim
    # KV_stride = kv_stride
    # if head_kv > 64:
    #     assert head_kv % 64 == 0, 'head_kv should be a multiple of 64'
    #     REPLICATE_H = head_kv // 64
    # else:
    #     REPLICATE_H = 1

    # H_per_block = padded_H if REPLICATE_H == 1 else 64
    # num_m_block = T.ceildiv(q_head_per_hk, 2*BLOCK_M) * 2
    assert h_kv == 1, 'only support h_kv == 1 for now'
    sm_num = driver.get_num_sms()
    TOPK_BLOCK_SIZE = 64
    PAGE_BLOCK_SIZE = 64
    BLOCK_M = 64
    num_m_block = T.ceildiv(h_q // h_kv, 2 * BLOCK_M) * 2
    num_sm_parts = T.max((sm_num // 2) // h_kv // (T.ceildiv(h_q // h_kv, 2 * BLOCK_M) * s_q), 1)
    
    # @T.macro
    # def cvt_fp8x8_bf16x8(
    #     inputs: T.LocalBuffer([8,], "float8_e4m3"),
    #     inputs_float_local: T.LocalBuffer((8,), "float"),
    #     scale: T.LocalBuffer((1,), "float"),
    #     scale_bf16: T.LocalBuffer((1,), "bfloat16"),
    #     outputs: T.LocalBuffer([8], "bfloat16"),
    # ):
    #     scale_bf16[0] = scale[0]
    #     for i in T.vectorized(8):
    #         inputs_float_local[i] = T.cast(inputs[i], "float")
    #         # T.copy(inputs, inputs_float_local)
    #     for i in T.vectorized(8):
    #         outputs[i] = T.cast(inputs_float_local[i], "bfloat16")
    #     # T.copy(inputs_float_local, outputs)
    #     for i in T.vectorized(8):
    #         outputs[i] *= scale_bf16[0]
        

    @T.prim_func
    def main(
        q: T.Tensor([b, s_q, h_q, nope_dim + rope_dim], dtype),  # type: ignore
        kv_nope: T.Tensor([num_blocks, page_block_size, h_kv, nope_dim], "float8_e4m3"),    # [?, block_size, h_kv, d]
        kv_rope: T.Tensor([num_blocks, page_block_size, h_kv, rope_dim], "bfloat16"),    # [?, block_size, h_kv, d]
        kv_scale: T.Tensor([num_blocks, page_block_size, h_kv, 4], "float32"),  # type: ignore
        cache_seqlens: T.Tensor([b], "int32"),  # type: ignore
        block_table: T.Tensor([b, max_num_blocks_per_seq], "int32"),  # type: ignore
        indices: T.Tensor([b, s_q, topk], "int32"),  # type: ignore
        begin_idx: T.Tensor([num_sm_parts], "int32"),  # type: ignore
        sched_begin_block_idx: T.Tensor([num_sm_parts], "int32"),  # type: ignore
        end_idx: T.Tensor([num_sm_parts], "int32"),  # type: ignore
        sched_end_block_idx: T.Tensor([num_sm_parts], "int32"),  #
    ):
        with T.ScopeKernel(
                grid=(num_m_block, s_q, num_sm_parts),
                cluster=cluster,
                threads=threads):
            
            # Q_shared_l = T.alloc_shared([H_per_block, D // 2], dtype)
            # Q_shared_r = T.alloc_shared([H_per_block, D // 2], dtype)
            # Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            cur_fp8x16 = T.alloc_local([16], "float8_e4m3")
            cur_fp32x8 = T.alloc_local([8], "float32")
            cur_bf16x8 = T.alloc_local([8], "bfloat16")
            scale_bf16 = T.alloc_local((1,), "bfloat16")
            scale = T.alloc_local((1,), "float")
            K_shared_nope = T.alloc_shared([nope_dim, TOPK_BLOCK_SIZE], dtype)
            K_shared_rope = T.alloc_shared([rope_dim, TOPK_BLOCK_SIZE], dtype)
            indice_local = T.alloc_local([1], "int32")
            
            head_block_idx, s_q_idx, partition_idx = T.get_block_bindings()
            idx_in_cluster = head_block_idx % 2
            cx, cy, cz = T.get_cluster_bindings()
            tx = T.get_thread_binding(0)
            warpgroup_idx = tx // 128
            idx_in_warpgroup = tx % 128
            lane_idx = idx_in_warpgroup % 32
            warp_idx = tx // 32
            my_token_idx = warp_idx * 8 + lane_idx % 8
            
            bar_q = T.alloc_barrier(arrive_count=1)
            bar_k_local_ready = T.alloc_barrier(arrive_count=128)
            bar_k_remote_ready = T.alloc_barrier(arrive_count=1)
            bar_k_avail = T.alloc_barrier(arrive_count=4)
            
            # TODO: cute::cluster_arrive();
            
            # T.copy(q[b_i, s_i, H0:H1, 0:D // 2], Q_shared_l)
            # T.copy(q[b_i, s_i, H0:H1, D // 2:D], Q_shared_r)
            # T.copy(q[b_i, s_i, H0:H1, D:], Q_tail_shared)
            # T.barrier_arrive(bar_q)
            
            if tx >= 256:
                # producer
                T.set_max_nreg(80, 0)
                
                for batch_idx in T.serial(begin_idx[partition_idx], end_idx[partition_idx]):
                    # FIXME: check here
                    start_block_idx = 0
                    end_block_idx = T.ceildiv(topk, TOPK_BLOCK_SIZE)
                    for block_idx in T.serial(start_block_idx, end_block_idx):
                        if idx_in_warpgroup == 0:
                            # T.ptx_arrive_barrier_expect_tx(
                            #     bar_k_remote_ready[0], 
                            #     (TOPK_BLOCK_SIZE / 2) * (nope_dim + rope_dim) * dsize_map[dtype])
                            T.ptx_arrive_barrier_expect_tx(
                                bar_k_remote_ready[0], 
                                (TOPK_BLOCK_SIZE / 2) * (nope_dim) * dsize_map[dtype])
                        indice_local[0] = indices[batch_idx, s_q_idx, block_idx * TOPK_BLOCK_SIZE + cx * TOPK_BLOCK_SIZE // 2 + warp_idx * 8 + lane_idx % 8]
                        block_index = indice_local[0] // PAGE_BLOCK_SIZE
                        rel_idx_in_block = (indice_local[0] + PAGE_BLOCK_SIZE) % PAGE_BLOCK_SIZE
                        for dim_idx in T.serial(T.ceildiv(nope_dim, 64)):
                            # FIXME: fix index
                            for j in T.vectorized(16):
                                cur_fp8x16[j] = kv_nope[block_index, rel_idx_in_block, 0, dim_idx * 64 + (lane_idx // 8) * 16 + j]
                            # FIXME: check scale
                            scale_bf16[0] = scale[0]
                            for i in T.vectorized(8):
                                cur_fp32x8[i] = T.cast(cur_fp8x16[i], "float")
                            for i in T.vectorized(8):
                                cur_bf16x8[i] = T.cast(cur_fp32x8[i], "bfloat16")
                            for i in T.vectorized(8):
                                cur_bf16x8[i] *= scale_bf16[0]
                            for i in T.vectorized(8):
                                # plan.u.k[buf_idx].data() + (idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx)*8 + ((lane_idx/8)*16)*TOPK_BLOCK_SIZE;
                                K_shared_nope[(lane_idx // 8) * 16, (idx_in_cluster*(TOPK_BLOCK_SIZE//2) + my_token_idx) * 8]
                                K_shared_nope[(lane_idx/8)*16 + dim_idx*64 + 0, (idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx) * 8 + i] = cur_bf16x8[i]
                            T.put_thread(
                                src=T.address_of(cur_bf16x8[0]),
                                dst=T.address_of(K_shared_nope[((tx - 256) * 8) // block_N, ((tx - 256) * 8) % block_N]),
                                size=0,
                                mbar=T.address_of(bar_k_remote_ready),
                                dst_pe=(cx + 1) % cluster[0],
                                scope="cluster")
                            T.barrier_wait(bar_k_remote_ready, 0)


    return main


def sparse_mla_fwd_interface(q,
                             kv_nope, 
                             kv_rope, 
                             kv_scale,
                             block_table,
                             cache_seqlens,
                             indices,
                             begin_idx,
                             sched_begin_block_idx,
                             end_idx,
                             sched_end_block_idx,
                             topk,
                             dtype,
                             return_kernel=False,
                             print_kernel=False):
    assert q.is_contiguous() and kv_nope.is_contiguous() and kv_rope.is_contiguous() and kv_scale.is_contiguous()
    batch, seq_len_q, num_heads_q, dim = q.shape
    num_blocks, page_block_size, num_heads_k, _ = kv_nope.shape
    max_num_blocks_per_seq = block_table.shape[1]

    assert dim == 576, 'you should assign dim otherwise'
    nope_dim = 512
    rope_dim = dim - nope_dim
    assert kv_nope.shape[-1] == nope_dim
    assert kv_rope.shape[-1] == dim - nope_dim
    
    cluster = (2, 1, 1)
    threads = 384

    print(f"batch={batch}")
    print(f"seq_len_q={seq_len_q}")
    print(f"num_heads_q={num_heads_q}")
    print(f"num_heads_k={num_heads_k}")
    print(f"num_blocks={num_blocks}")
    print(f"page_block_size={page_block_size}")
    print(f"nope_dim={nope_dim}")
    print(f"rope_dim={rope_dim}")
    print(f"max_num_blocks_per_seq={max_num_blocks_per_seq}")
    print(f"topk={topk}")
    print(f"cluster={cluster}")
    print(f"threads={threads}")
    
    kernel = sparse_mla_fwd(
        batch, 
        seq_len_q, 
        num_heads_q, 
        num_heads_k, 
        num_blocks, 
        page_block_size, 
        nope_dim,
        rope_dim,
        max_num_blocks_per_seq,
        topk,
        dtype,
        cluster,
        threads=threads
    )
    
    print(kernel.get_kernel_source())
    kernel(q, kv_nope, kv_rope, kv_scale, cache_seqlens, block_table, indices, begin_idx, sched_begin_block_idx, end_idx, sched_end_block_idx)
    # if print_kernel:
    #     print(kernel.get_kernel_source())
    # out, lse = kernel(q, kv, cache_seqlens, block_table)
    # if return_kernel:
    #     return kernel
    # return out, lse


def reference_torch(
    cache_seqlens: torch.Tensor,    # [batch_size]
    block_table: torch.Tensor,      # [batch_size, ?]
    q: torch.Tensor,    # [batch_size, s_q, h_q, d]
    blocked_k: torch.Tensor,    # [?, block_size, h_kv, d]
    dv: int,
    is_causal: bool,
    indices: Optional[torch.Tensor] = None   # [batch_size, s_q, topk]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A reference implementation in PyTorch
    """
    def get_topk_attn_mask(s_q: int, s_k: int, indices: torch.Tensor):
        mask = torch.zeros(s_q, s_k, dtype=torch.bool)
        for i in range(s_q):
            cur_indices = indices[i]
            valid_indices = cur_indices[cur_indices != -1]
            mask[i, valid_indices] = True
        return mask

    def scaled_dot_product_attention(
        batch_idx: int,
        query: torch.Tensor,    # [h_q, s_q, d]
        kv: torch.Tensor,      # [h_kv, s_k, d]
        dv: int,
        is_causal,
        indices: Optional[torch.Tensor],  # [s_q, topk]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_q = query.size(0)
        h_kv = kv.size(0)
        s_q = query.shape[-2]
        s_k = kv.shape[-2]
        query = query.float()
        kv = kv.float()
        if h_kv != 1:
            kv = kv.repeat_interleave(h_q // h_kv, dim=0)
        kv[kv != kv] = 0.0
        attn_weight = query @ kv.transpose(-2, -1)  # [h_q, s_q, s_k]
        if (is_causal and query.size(1) > 1) or indices is not None:
            mask = torch.ones(s_q, s_k, dtype=torch.bool)
            if is_causal:
                assert indices is None
                mask = mask.tril(diagonal=s_k - s_q)
            if indices is not None:
                mask &= get_topk_attn_mask(s_q, s_k, indices)
            attn_bias = torch.zeros(s_q, s_k, dtype=torch.float)
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
            attn_weight += attn_bias.to(q.dtype)
        attn_weight /= math.sqrt(query.size(-1))
        lse = attn_weight.logsumexp(dim=-1)  # [h_q, s_q]
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
        output = attn_weight @ kv[..., :dv]    # [h_q, s_q, dv]
        # Correct for q tokens which has no attendable k
        lonely_q_mask = (lse == float("-inf"))
        output[lonely_q_mask.unsqueeze(-1).broadcast_to(h_q, s_q, dv)] = 0.0
        lse[lonely_q_mask] = float("+inf")

        return output, lse

    b, s_q, h_q, d = q.size()
    block_size = blocked_k.size(1)
    h_kv = blocked_k.size(2)
    cache_seqlens_cpu = cache_seqlens.cpu()
    out_ref = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
    lse_ref = torch.empty(b, h_q, s_q, dtype=torch.float32)
    for i in range(b):
        cur_len = cache_seqlens_cpu[i].item()
        cur_num_blocks = cdiv(cur_len, block_size)
        cur_block_indices = block_table[i][0: cur_num_blocks]
        cur_kv = blocked_k[cur_block_indices].view(-1, h_kv, d)[:cur_len, ...]
        cur_out, cur_lse = scaled_dot_product_attention(
            i,
            q[i].transpose(0, 1),
            cur_kv.transpose(0, 1),
            dv,
            is_causal,
            indices[i] if indices is not None else None
        )
        out_ref[i] = cur_out.transpose(0, 1)
        lse_ref[i] = cur_lse
    out_ref = out_ref.to(torch.bfloat16)
    return out_ref, lse_ref

def test_sparse_mla_fwd_pipelined(b=1,
                                  s_q=1,
                                  s_kv=8192,
                                  h_q=128,
                                  h_kv=1,
                                  d=576,
                                  dv=512,
                                  topk=2048,
                                  dtype="bfloat16",
                                  q_start_s_index=1024,
                                  check_correctness=True):
    torch.random.manual_seed(0)
    block_size = 64
    cache_seqlens = torch.tensor([s_kv + 2 * i for i in range(b)], dtype=torch.int32, device='cuda')
    total_seqlens = cache_seqlens.sum().item()
    mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = math.ceil(max_seqlen / 256) * 256
    
    q = torch.randn(b, s_q, h_q, d, dtype=dtype_map[dtype], device='cuda')
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32, device='cuda').view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d, dtype=dtype_map[dtype], device='cuda')


    # Generate indices_in_kvcache: [b, s_q, topk]
    indices_in_kvcache = torch.empty(b, s_q, topk, dtype=torch.int32, device=q.device)
    cache_seqlens_cpu = cache_seqlens.cpu()
    block_table_cpu = block_table.cpu()
    for i in range(b):
        cur_len = int(cache_seqlens_cpu[i].item())
        for j in range(s_q):
            if cur_len > 0:
                sel = torch.randperm(cur_len, device="cpu")[:topk]
            else:
                sel = torch.empty(0, dtype=torch.int64, device="cpu")
            if sel.numel() < topk:
                pad = torch.full((topk - sel.numel(),), -1, dtype=torch.int64, device="cpu")
                sel = torch.cat([sel, pad], dim=0)
            blk_idx = torch.where(sel >= 0, sel // block_size, torch.zeros_like(sel))
            off_idx = torch.where(sel >= 0, sel % block_size, torch.zeros_like(sel))
            phys_blk = block_table_cpu[i, blk_idx.clamp_min(0)]
            merged = (phys_blk * block_size + off_idx).to(torch.int32)
            merged[sel < 0] = -1
            indices_in_kvcache[i, j] = merged.to(q.device)

    # Quantize KV cache to FP8-with-scale format (1 head only)
    kv_nope, kv_rope, kv_scale = _quantize_k_cache_fp8(blocked_k, dv, 128)

    # Get schedule metadata for sparse + FP8
    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens,
        s_q * h_q // h_kv,
        h_kv,
        h_q,
        True,
        topk,
    )
    
    # print(f"tile_scheduler_metadata, {tile_scheduler_metadata}, {tile_scheduler_metadata.shape}")
    
    begin_idx = tile_scheduler_metadata[:, 0].contiguous()
    sched_begin_block_idx = tile_scheduler_metadata[:, 1].contiguous()
    end_idx = tile_scheduler_metadata[:, 2].contiguous()
    sched_end_block_idx = tile_scheduler_metadata[:, 3].contiguous()
    
    print(f"kv_nope.shape: {kv_nope.shape}, {kv_nope.dtype}")
    print(f"kv_rope.shape: {kv_rope.shape}, {kv_rope.dtype}")
    print(f"kv_scale.shape: {kv_scale.shape}, {kv_scale.dtype}")
    kernel = sparse_mla_fwd_interface(
        q, kv_nope, kv_rope, kv_scale, block_table, cache_seqlens, indices_in_kvcache, begin_idx, sched_begin_block_idx, end_idx, sched_end_block_idx,
        topk, dtype, return_kernel=False, print_kernel=True)

    # def fn():
    #     out, lse = kernel(q, kv, indices, q_start_s_index_t)
    #     if q_start_s_index == 0 and KV_stride > 1:
    #         out[:, :KV_stride - 1, :, :] = 0
    #     return out, lse

    # tl_out, tl_lse = fn()
    # ref_out = ref_sparse_mla_fwd_interface(q, kv, indices, q_start_s_index, KV_stride)
    # # print(f"tl_out: {tl_out}")
    # # print(f"ref_out: {ref_out}")

    # torch.testing.assert_close(tl_out, ref_out, rtol=1e-3, atol=1e-3)

    # from tilelang.profiler import do_bench
    # ms = do_bench(
    #     fn,
    #     rep=10,
    #     warmup=10,
    # )
    # print(f"Average time: {ms:.3f} ms")
    # print(f'fwd io bandwidth = ', (B * S * DQK * topk * 2) / (ms * 1e-3) / 1e12)
    # print(f'fwd tflops = ', (B * S * (DQK + DV) * topk * 2 * H) / (ms * 1e-3) / 1e12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_correctness", action="store_true")
    args = parser.parse_args()
    if args.test_correctness:
        b, s_q, s_kv, h_q, h_kv, d, dv, topk, dtype = 1, 1, 1024, 128, 1, 576, 512, 2048, torch.bfloat16
    else:
        b, s_q, s_kv, h_q, h_kv, d, dv, topk, dtype = 1, 1, 8192, 128, 1, 576, 512, 2048, torch.bfloat16
    test_sparse_mla_fwd_pipelined(
        b, s_q, s_kv, h_q, h_kv, d, dv, check_correctness=args.test_correctness)
