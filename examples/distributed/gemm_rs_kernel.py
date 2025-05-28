# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
import pynvshmem
import os
import tilelang
import tilelang.language as T
from gemm_rs_utils import GEMMReduceScatterTensorParallelContext

tilelang.disable_cache()


def gemm_rs_kernel(world_size, M, N, local_K, dtype="float16"):

    local_chunk = T.ceildiv(M, world_size)

    @T.prim_func
    def main(
            input: T.Tensor((M, local_K), dtype),
            weight: T.Tensor((N, local_K), dtype),
            gemm_out: T.Tensor((M, N), dtype),
            scatter_signal: T.Tensor((world_size,), "uint64"),
            workspace: T.Tensor((world_size,), "int32"),
    ):
        with T.Kernel(256, threads=128) as (bx):
            input[bx, 0] = 1

    return main


def gemm_rs_op(input, weight, ctx: GEMMReduceScatterTensorParallelContext, persistent: bool = True):
    world_size = ctx.rs_ctx.world_size
    local_world_size = ctx.rs_ctx.local_world_size
    rs_stream = ctx.rs_stream
    output_dtype = ctx.output_dtype
    num_gemm_sms = ctx.num_gemm_sms

    orig_M = input.shape[0]
    orig_M_per_rank = orig_M // world_size
    M, local_K = input.shape
    N = weight.shape[0]
    assert N == ctx.rs_ctx.N

    assert M % world_size == 0
    assert weight.shape[1] == local_K
    local_M = M // world_size
    current_stream = torch.cuda.current_stream()
    rs_stream.wait_stream(current_stream)

    output = torch.empty((local_M, N), dtype=output_dtype, device=input.device)
    workspace = torch.zeros((world_size,), dtype=torch.int32, device=input.device)
    gemm_out = ctx.get_gemm_out_buf(input)
    scatter_signal = ctx.rs_ctx.scatter_signal_buf

    if persistent:
        # TODO: implement this
        pass
        # triton_config = triton.Config(
        #     {
        #         "BLOCK_SIZE_M": ctx.BLOCK_M, "BLOCK_SIZE_N": ctx.BLOCK_N, "BLOCK_SIZE_K": ctx.BLOCK_K, "GROUP_SIZE_M":
        #         ctx.GROUP_M, "NUM_SMS": num_gemm_sms, "EPILOGUE_SUBTILE": False
        #     }, num_stages=ctx.stages, num_warps=8)
        # gemm_rs_producer_persistent(input, weight, gemm_out, scatter_signal, workspace, world_size, local_world_size,
        #                             num_gemm_sms, current_stream, triton_config)
    else:
        # triton_config = triton.Config(
        #     {
        #         "BLOCK_SIZE_M": ctx.BLOCK_M, "BLOCK_SIZE_N": ctx.BLOCK_N, "BLOCK_SIZE_K": ctx.BLOCK_K, "GROUP_SIZE_M":
        #         ctx.GROUP_M
        #     }, num_stages=ctx.stages, num_warps=8)
        # triton_config = update_triton_config(M, N, local_K, input.dtype, world_size, local_world_size, triton_config)
        # gemm_rs_producer_non_persistent(input, weight, gemm_out, scatter_signal, workspace, world_size,
        #                                 local_world_size, current_stream, triton_config)
        print(f"M: {M}, N: {N}, local_K: {local_K}")
        print(
            f"input.shape: {input.shape}, weight.shape: {weight.shape}, gemm_out.shape: {gemm_out.shape}"
        )
        program = gemm_rs_kernel(world_size, M, N, local_K)
        kernel = tilelang.compile(program, pass_configs={"tl.disable_tma_lower": True})
        kernel(input, weight, gemm_out, scatter_signal, workspace)

    # with torch.cuda.stream(rs_stream):
    #     output = reduce_scatter_2d_op(gemm_out, ctx.rs_ctx)
    # current_stream.wait_stream(rs_stream)

    # return output[:orig_M_per_rank]
    return None


def gemm_rs(a, b, ctx, persistent=True):
    """GEMM Reduce-Scatter for Multi-Node

    computes local GEMM (a x b) to generate partial results, followed by `reduce_scatter` to produce c

    Args:
        a (torch.Tensor<bfloat16/float16>): local matmul A matrix. shape: [M, local_K]
        b (torch.Tensor<bfloat16/float16>): local matmul B matrix. shape: [N, local_K]
        ctx(GEMMReduceScatterTensorParallelContext): context

    Returns:
        c (torch.Tensor<bfloat16/float16>): local matmul C matrix. shape: [M // world_size, N]
    """
    c = gemm_rs_op(a, b, ctx, persistent)
    return c
