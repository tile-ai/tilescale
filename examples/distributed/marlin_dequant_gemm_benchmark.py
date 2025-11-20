#!/usr/bin/env python3
"""
Profile vLLM custom op: ops.moe_wna16_marlin_gemm (Marlin MoE FP4 path).

This script:
- Synthesizes MXFP4-marlin formatted expert weights/scales (per expert).
- Builds random inputs (hidden states, topk ids/weights), aligns tokens
  to MoE block tiles, and calls the kernel twice (W1 then W2) like the
  real fused path.
- Measures time with CUDA events and reports per-call and combined TFLOPS.

Requirements:
- GPU with supported custom ops built for vLLM (Marlin GEMM).
- CUDA/ROCm available and torch.cuda.is_available() is True.

Usage examples:
  python scripts/profile_moe_wna16_marlin_gemm.py \
    --m 512 --k 4096 --n 14336 --experts 8 --topk 2 --dtype bf16 --iters 50

Parameters:
- m: number of tokens (per batch step)
- k: hidden size
- n: intermediate size per partition (per expert)
- experts: number of local experts (E)
- topk: experts-per-token (routing)
- dtype: compute dtype for activations (bf16|fp16)
- iters/warmup: repetitions for timing
"""

import argparse
import math
from typing import List

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    rand_marlin_weight_mxfp4_like,
)
from vllm.scalar_type import scalar_types


def pick_moe_block_size(m: int, topk: int, e: int) -> int:
    for b in [8, 16, 32, 48, 64]:
        if (m * topk) / e / b < 0.9:
            return b
    return 64


@torch.no_grad()
def make_mxfp4_marlin_weights(
    e: int, size_n: int, size_k: int, group_size: int = 32, device: torch.device = torch.device("cuda")
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create batched (E, size_n, size_k//2) marlin-repacked W and scales.

    Uses rand_marlin_weight_mxfp4_like to synthesize per-expert tensors.
    Returns:
      w_q (uint8): shape [E, size_n, size_k//2]
      w_scales (float8_e8m0fnu): shape [E, marlin_scale_cols]
    """
    wq_list: List[torch.Tensor] = []
    ws_list: List[torch.Tensor] = []
    base = torch.empty(size_n, size_k, dtype=torch.bfloat16, device=device)
    for _ in range(e):
        # The function ignores values distribution and returns packed weights
        _, marlin_qweight, marlin_scales = rand_marlin_weight_mxfp4_like(
            base, group_size
        )
        wq_list.append(marlin_qweight)
        ws_list.append(marlin_scales)
    w_q = torch.stack(wq_list, dim=0).contiguous()
    w_scales = torch.stack(ws_list, dim=0).contiguous()
    return w_q, w_scales


def to_dtype(name: str) -> torch.dtype:
    if name.lower() in ["bf16", "bfloat16", "bfloat"]:
        return torch.bfloat16
    if name.lower() in ["fp16", "float16", "half"]:
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def profile_kernel(
    m: int,
    k: int,
    n: int,
    e: int,
    topk: int,
    act_dtype: torch.dtype,
    iters: int,
    warmup: int,
    device: torch.device,
):
    assert torch.cuda.is_available(), "CUDA/ROCm device not available"

    # Inputs
    hidden_states = torch.randn(m, k, dtype=act_dtype, device=device)
    # Top-k routing: ids in [0, e), weights sum to 1 (optional)
    topk_ids = torch.randint(low=0, high=e, size=(m, topk), dtype=torch.int32, device=device)
    # Make ids unique per token to avoid duplicate expert dispatch (optional)
    # Simple trick: sort unique then pad if needed
    for i in range(m):
        uniq = torch.unique(topk_ids[i])
        if uniq.numel() < topk:
            need = topk - uniq.numel()
            extra = torch.randint(low=0, high=e, size=(need,), device=device, dtype=torch.int32)
            topk_ids[i, :uniq.numel()] = uniq
            topk_ids[i, uniq.numel():] = extra

    topk_weights = torch.rand(m, topk, dtype=torch.float32, device=device)
    topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True) + 1e-6)

    # Block size selection and routing alignment
    block_size_m = pick_moe_block_size(m, topk, e)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, e
    )

    # Workspace and kernel flags
    workspace = marlin_make_workspace_new(device, 4)
    major_cc = torch.cuda.get_device_capability(device)[0]
    use_atomic_add = act_dtype == torch.float16 or major_cc >= 9

    # Create Marlin-packed W1/W2 and corresponding scales
    # W1: (E, 2N, K//2)
    w1_q, w1_scales = make_mxfp4_marlin_weights(e, 2 * n, k, group_size=32, device=device)
    # W2: (E, K, N//2)
    w2_q, w2_scales = make_mxfp4_marlin_weights(e, k, n, group_size=32, device=device)

    # Bias per expert per channel; marlin MoE kernel expects bias dtype
    # to match activation dtype (bf16/fp16)
    w1_bias = torch.zeros(e, 2 * n, dtype=act_dtype, device=device)
    w2_bias = torch.zeros(e, k, dtype=act_dtype, device=device)

    # Pre-allocate outputs and activation buffer (post-GELU/SiLU split)
    out1 = torch.empty(m * topk, 2 * n, dtype=act_dtype, device=device)
    act_out = torch.empty(m * topk, n, dtype=act_dtype, device=device)
    out2 = torch.empty(m * topk, k, dtype=act_dtype, device=device)

    quant_type = scalar_types.float4_e2m1f

    def call_w1():
        return ops.moe_wna16_marlin_gemm(
            hidden_states,
            out1,
            w1_q,
            w1_bias,
            w1_scales,
            None,  # global_scale1 (NVFP4 only)
            None,  # w1_zeros
            None,  # g_idx1 (act_order)
            None,  # sort_indices1
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size=block_size_m,
            top_k=topk,
            mul_topk_weights=False,
            is_ep=False,
            b_q_type=quant_type,
            size_m=m,
            size_n=2 * n,
            size_k=k,
            is_k_full=True,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

    def apply_activation():
        # Match fused path: apply SiLU and elementwise multiply to halve dim
        # out1 is [M*topk, 2N] -> act_out is [M*topk, N]
        torch.ops._C.silu_and_mul(act_out, out1.view(-1, 2 * n))

    def call_w2():
        return ops.moe_wna16_marlin_gemm(
            act_out,  # A after activation: [M*topk, N]
            out2,  # C
            w2_q,
            w2_bias,
            w2_scales,
            None,
            None,
            None,
            None,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size=block_size_m,
            top_k=1,  # matches fused path (weights applied here if needed)
            mul_topk_weights=True,
            is_ep=False,
            b_q_type=quant_type,
            size_m=m * topk,
            size_n=k,
            size_k=n,
            is_k_full=True,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

    # Warmup
    for _ in range(warmup):
        call_w1()
        apply_activation()
        call_w2()
    torch.cuda.synchronize()

    # Timings
    def time_call(fn, repeat: int) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        total_ms = 0.0
        for _ in range(repeat):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            total_ms += start.elapsed_time(end)
        return total_ms / repeat

    t1_ms = time_call(call_w1, iters)
    # Ensure act_out is computed for W2 timing
    apply_activation()
    t2_ms = time_call(call_w2, iters)
    t_total_ms = t1_ms + t2_ms

    # FLOPs (two estimates: effective after padding, and theoretical without pad)
    m_eff = int(num_tokens_post_padded.item())
    flops_w1_eff = 2.0 * m_eff * k * (2 * n)
    flops_w2_eff = 2.0 * m_eff * n * k
    tflops_w1_eff = flops_w1_eff / (t1_ms / 1e3) / 1e12
    tflops_w2_eff = flops_w2_eff / (t2_ms / 1e3) / 1e12

    flops_w1_theo = 2.0 * m * topk * k * (2 * n)
    flops_w2_theo = 2.0 * m * topk * n * k
    tflops_w1_theo = flops_w1_theo / (t1_ms / 1e3) / 1e12
    tflops_w2_theo = flops_w2_theo / (t2_ms / 1e3) / 1e12

    print("==== moe_wna16_marlin_gemm profile ====")
    print(f"device        : {torch.cuda.get_device_name(device)}")
    print(f"dtype         : {act_dtype}")
    print(f"M={m}, K={k}, N={n}, E={e}, topk={topk}, block_m={block_size_m}")
    print(f"num_tokens_padded (effective M*topk): {m_eff}")
    print("")
    print(f"W1 ms         : {t1_ms:.3f} ms  | TFLOPS eff {tflops_w1_eff:.2f}, theo {tflops_w1_theo:.2f}")
    print(f"W2 ms         : {t2_ms:.3f} ms  | TFLOPS eff {tflops_w2_eff:.2f}, theo {tflops_w2_theo:.2f}")
    print(f"Total ms      : {t_total_ms:.3f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=512, help="Num tokens")
    parser.add_argument("--k", type=int, default=4096, help="Hidden size")
    parser.add_argument("--n", type=int, default=14336, help="Intermediate size per partition")
    parser.add_argument("--experts", type=int, default=8, help="Num local experts (E)")
    parser.add_argument("--topk", type=int, default=2, help="Experts per token")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="Activation dtype")
    parser.add_argument("--iters", type=int, default=50, help="Timing iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")

    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)

    profile_kernel(
        m=args.m,
        k=args.k,
        n=args.n,
        e=args.experts,
        topk=args.topk,
        act_dtype=to_dtype(args.dtype),
        iters=args.iters,
        warmup=args.warmup,
        device=device,
    )


if __name__ == "__main__":
    main()
