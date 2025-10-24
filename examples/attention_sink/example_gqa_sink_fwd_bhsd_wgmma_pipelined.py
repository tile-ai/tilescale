# Modified from tilelang/examples/flash_attention/example_gqa_fwd_bshd_wgmma_pipelined.py
# Optimized for Hopper architecture, with a benchmark to compare with official Triton impl

import torch
import tilelang
from tilelang.autotuner import autotune
from tilelang.profiler import do_bench
import tilelang.language as T
from tilelang.layout import make_swizzled_layout
import itertools
import argparse
from typing import Optional


def get_configs():
    iter_params = dict(block_M=[128], block_N=[128], num_stages=[0, 1, 2], threads=[128, 256])
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@autotune(
    configs=get_configs(),
    warmup=500,
    rep=100,
)
@tilelang.jit(
    out_idx=[3], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def flashattn(
    batch,
    heads,
    seq_q,
    seq_kv,
    dim,
    groups=1,
    window_size=None,  # None for full attention
    sm_scale=None,
    block_M=128,
    block_N=128,
    num_stages=2,
    threads=256,
    dtype: str = "float16",
):

    if window_size is not None:
        assert window_size % block_N == 0, "window_size must be divisible by block_N"

    if sm_scale is None:
        sm_scale = (1.0 / dim)**0.5
    scale = sm_scale * 1.44269504  # log2(e)

    head_kv = heads // groups
    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, head_kv, seq_kv, dim]
    accum_dtype = "float"

    past_len = seq_kv - seq_q
    assert past_len >= 0, "seq_kv must be greater than or equal to seq_q"

    @T.macro
    def MMA0(
        K: T.Tensor(kv_shape, dtype),
        Q_shared: T.SharedBuffer([block_M, dim], dtype),
        K_shared: T.SharedBuffer([block_N, dim], dtype),
        acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
        k: T.int32,
        bx: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(K[bz, by // groups, k * block_N:(k + 1) * block_N, :], K_shared)
        for i, j in T.Parallel(block_M, block_N):
            q_idx = bx * block_M + i + past_len
            k_idx = k * block_N + j
            if window_size is not None:
                acc_s[i, j] = T.if_then_else(q_idx >= k_idx and q_idx < k_idx + window_size, 0,
                                             -T.infinity(acc_s.dtype))
            else:
                acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def MMA1(
        V: T.Tensor(kv_shape, dtype),
        V_shared: T.SharedBuffer([block_M, dim], dtype),
        acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
        acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
        k: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(V[bz, by // groups, k * block_N:(k + 1) * block_N, :], V_shared)
        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def Softmax(
            acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
            acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
            scores_max: T.FragmentBuffer([block_M], accum_dtype),
            scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
            scores_scale: T.FragmentBuffer([block_M], accum_dtype),
            scores_sum: T.FragmentBuffer([block_M], accum_dtype),
            logsum: T.FragmentBuffer([block_M], accum_dtype),
    ):
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
        # To do causal softmax, we need to set the scores_max to 0 if it is -inf
        # This process is called Check_inf in FlashAttention3 code, and it only need to be done
        # NOTE(wt): check_inf is necessary for sliding window attention.
        for i in T.Parallel(block_M):
            if window_size is not None:
                scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0,
                                               scores_max[i])
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

        for i, j in T.Parallel(block_M, block_N):
            # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            # max * log_2(e)) This allows the compiler to use the ffma
            # instruction instead of fadd and fmul separately.
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
        T.reduce_sum(acc_s, scores_sum, dim=1)
        for i in T.Parallel(block_M):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
        T.copy(acc_s, acc_s_cast)

    @T.macro
    def Rescale(
            acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
            scores_scale: T.FragmentBuffer([block_M], accum_dtype),
    ):
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] *= scores_scale[i]

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            Output: T.Tensor(q_shape, dtype),
            Sinks: T.Tensor([heads], dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
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
            sinks = T.alloc_fragment([block_M], dtype)

            T.annotate_layout({
                Q_shared: make_swizzled_layout(Q_shared),
                K_shared: make_swizzled_layout(K_shared),
                V_shared: make_swizzled_layout(V_shared),
                O_shared: make_swizzled_layout(O_shared),
            })

            T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            for i in T.Parallel(block_M):
                sinks[i] = Sinks[by]

            end = T.min(
                T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M + past_len, block_N))

            start = T.alloc_local([1], 'int32')
            if window_size is not None:
                start[0] = T.max(0, (bx * block_M + past_len - window_size) // block_N)
            else:
                start[0] = 0

            for k in T.Pipelined(
                    start[0],
                    end,
                    num_stages=num_stages,
                    order=[-1, 0, 3, 1, -1, 2],
                    stage=[-1, 0, 0, 1, -1, 1],
                    group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10], [11], [12], [13]]):
                MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum,
                        logsum)
                Rescale(acc_o, scores_scale)
                MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
            for i in T.Parallel(block_M):
                logsum[i] += T.exp2(sinks[i] * 1.44269504 -
                                    scores_max[i] * scale)  # The only change for attention sink
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, by, bx * block_M:(bx + 1) * block_M, :])

    return main


# Following functions are adapted and optimized from
# https://github.com/openai/gpt-oss/blob/main/gpt_oss/triton/attention.py
def ref_program(query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                sinks: torch.Tensor,
                sliding_window: Optional[int] = None,
                dtype: torch.dtype = torch.float16) -> torch.Tensor:

    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    batch_size, num_keys, num_key_value_heads, head_dim = key.shape
    query = query.transpose(1, 2).contiguous()
    query = query.view(batch_size, query.shape[1], num_key_value_heads, -1, head_dim)
    batch_size, num_queries, num_key_value_heads, num_key_value_groups, head_dim = query.shape

    start_q = num_keys - num_queries
    sm_scale: float = 1.0 / head_dim**0.5

    sinks = sinks.view(1, num_key_value_heads, num_key_value_groups, 1, 1).float()
    key = key.unsqueeze(3)
    value = value.unsqueeze(3)

    pos_keys = torch.arange(num_keys, device=query.device)
    pos_queries = torch.arange(num_queries, device=query.device) + start_q
    mask = pos_keys[None, :] > pos_queries[:, None]
    mask = mask.float().masked_fill(mask, float("-inf"))

    if sliding_window:
        too_old = pos_keys[None, :] < (pos_queries[:, None] - sliding_window + 1)
        mask.masked_fill_(too_old, float("-inf"))

    logits = torch.einsum("bqhmd,bkhmd->bhmqk", query.float(), key.float()) * sm_scale
    logits = logits + mask[None, None, None, :, :]

    logits_max = torch.max(logits, dim=-1, keepdim=True).values
    logits_or_sinks_max = torch.maximum(sinks, logits_max)
    sinks = torch.exp(sinks - logits_or_sinks_max)
    unnormalized_scores = torch.exp(logits - logits_or_sinks_max)
    normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + sinks
    scores = unnormalized_scores / normalizer

    output = torch.einsum("bhmqk,bkhmd->bqhmd", scores, value.float())

    output = output.reshape(batch_size, num_queries, num_key_value_heads * num_key_value_groups,
                            head_dim).to(dtype)
    return output.transpose(1, 2).contiguous()


def gen_inputs(
        B,
        H,
        Sq,
        Skv,
        D,
        groups,
        dtype=torch.float16) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.randn([B, H, Sq, D], dtype=dtype, device='cuda')
    key = torch.randn([B, H // groups, Skv, D], dtype=dtype, device='cuda')
    value = torch.randn([B, H // groups, Skv, D], dtype=dtype, device='cuda')
    sinks = torch.randn([H], dtype=dtype, device='cuda')
    return query, key, value, sinks


def main(
    batch: int = 1,
    heads: int = 32,
    seq_q: int = 256,
    seq_kv: int = 256,
    dim: int = 128,
    groups: int = 8,
    window_size: int | None = None,
    dtype: str = "float16",
    tune: bool = False,
):
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    if window_size is not None:
        print('Using sliding window attention.')
        assert window_size <= seq_q
        flops_per_matmul = 2.0 * batch * heads * min(
            window_size, seq_kv // 2) * seq_q * dim  # just a rough estimation
    else:
        print('Using full attention.')
        flops_per_matmul = 2.0 * batch * heads * seq_q * seq_kv * dim * 0.5
    total_flops = 2 * flops_per_matmul

    if tune:
        kernel = flashattn(batch, heads, seq_q, seq_kv, dim, groups, window_size, dtype=dtype)
        print(f"Best latency: {kernel.latency}")
        print(f"Best TFlops: {total_flops / kernel.latency * 1e-9}")
        print(f"Best config: {kernel.config}")
    else:
        block_M = 128
        block_N = 128
        num_stages = 2
        threads = 256
        print(f"{block_M=}, {block_N=}, {num_stages=}, {threads=}")

        kernel = flashattn(
            batch,
            heads,
            seq_q,
            seq_kv,
            dim,
            groups,
            window_size,
            block_M=block_M,
            block_N=block_N,
            num_stages=num_stages,
            threads=threads,
            dtype=dtype)

        Q, K, V, sinks = gen_inputs(batch, heads, seq_q, seq_kv, dim, groups, dtype=torch_dtype)

        torch.testing.assert_close(
            kernel(Q, K, V, sinks),
            ref_program(Q, K, V, sinks, window_size, dtype=torch_dtype),
            rtol=1e-2,
            atol=1e-2)
        print("All checks passed.✅")

        # Benchmark tilelang
        latency_tilelang = do_bench(lambda: kernel(Q, K, V, sinks), warmup=500)
        print("Tilelang: {:.2f} ms".format(latency_tilelang))
        print("Tilelang: {:.2f} TFlops".format(total_flops / latency_tilelang * 1e-9))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--heads', type=int, default=64, help='heads')
    parser.add_argument('--seq_q', type=int, default=2048, help='sequence length of query')
    parser.add_argument('--seq_kv', type=int, default=2048, help='sequence length of key/value')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--groups', type=int, default=8, help='groups')
    parser.add_argument(
        '--window_size',
        type=int,
        default=None,
        help='window size (default: None, which means full attention)')
    parser.add_argument(
        '--dtype', type=str, default="float16", help="dtype, can be float16 or bfloat16")
    parser.add_argument('--tune', action='store_true', help='tune configs')
    args = parser.parse_args()
    main(args.batch, args.heads, args.seq_q, args.seq_kv, args.dim, args.groups, args.window_size,
         args.dtype, args.tune)
