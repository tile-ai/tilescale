"""TileScale-facing FP8/FP4 MegaMoE API.

This module mirrors DeepGEMM's public MegaMoE API and buffer layout.  The
SM100 path uses TileLang single-launch megakernels for the supported local and
intranode distributed SM100 shapes. Unsupported devices, shapes, and explicit
all-weight reference calls fall back to the PyTorch reference while preserving
the same preprocessed tensor contract.
"""

import math
import os
import types
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.distributed.allocator import get_allocator
from tilelang.distributed.shared_memory import tensor_from_ptr
from tilelang.distributed.tensor import tensor as distributed_tensor

from reference import (
    MegaMoEInputs,
    align,
    inverse_transpose_sf_for_utccp,
    inverse_transform_weights_for_mega_moe,
    mega_moe_reference,
)


TOKEN_ALIGNMENT = 384
GRAN_K = 32
PACKED_UE8M0_ONE = 0x7F7F7F7F
DEEPGEMM_MEGAMOE_MAX_BLOCK_M = 192
MEGAMOE_LOCAL_FUSED_MAP_MAX_ENTRIES = 32768
MEGAMOE_DISTRIBUTED_FUSED_MAP_MAX_ENTRIES = 32768
MEGAMOE_MAX_INTRANODE_RANKS = 8
MEGAMOE_WORKSPACE_BARRIER_ROWS = 2
MEGAMOE_WORKSPACE_BARRIER_SLOTS = MEGAMOE_WORKSPACE_BARRIER_ROWS * MEGAMOE_MAX_INTRANODE_RANKS
MEGAMOE_WORKSPACE_COMBINE_BARRIER_ROW = MEGAMOE_MAX_INTRANODE_RANKS


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True},
)
def _make_local_dispatch_l1_l2_combine_sm100_megakernel_prim_func(
    pool_rows: int,
    num_experts: int,
    num_topk: int,
    max_tokens: int,
    y_rows: int,
    max_blocks: int,
    hidden: int,
    padded_l1_sf_pool_rows: int,
    block_M: int,
    block_N: int,
    block_K: int,
    accum_dtype,
    num_stages: int,
    activation_clamp: float,
    use_activation_clamp: bool,
    sf_granularity_k: int = GRAN_K,
):
    assert block_M == 128
    assert block_N == 256
    assert block_K == 128
    assert max_blocks > 0
    assert hidden in (128, 256)
    assert sf_granularity_k == 32

    intermediate_hidden = 128
    l1_N = 256
    l2_N = hidden
    l1_k_iters = hidden // block_K
    l1_sf_k_groups = hidden // 128
    l2_sf_k_groups = 1
    sm_num = driver.get_num_sms()
    waves = _ceil_div(max_blocks, sm_num)
    x_work = pool_rows * hidden
    sf_work = pool_rows * l1_sf_k_groups
    stage_work = x_work + sf_work + pool_rows + y_rows * hidden

    @T.prim_func
    def main(
        x: T.Tensor((max_tokens, hidden), T.float8_e4m3fn),
        x_sf: T.Tensor((max_tokens, l1_sf_k_groups), T.int32),
        topk_idx: T.Tensor((max_tokens, num_topk), T.int64),
        topk_weights: T.Tensor((max_tokens, num_topk), T.float32),
        l1_A: T.Tensor((pool_rows, hidden), T.float8_e4m3fn),
        l1_B: T.Tensor((num_experts, l1_N, hidden), T.float4_e2m1fn),
        l1_SFA: T.Tensor((l1_sf_k_groups * padded_l1_sf_pool_rows,), T.uint32),
        l1_SFB: T.Tensor((l1_sf_k_groups * num_experts * l1_N,), T.uint32),
        l2_B: T.Tensor((num_experts, l2_N, intermediate_hidden), T.float4_e2m1fn),
        l2_SFB: T.Tensor((l2_sf_k_groups * num_experts * l2_N,), T.uint32),
        expert_counts: T.Tensor((num_experts,), T.int32),
        expert_offsets: T.Tensor((num_experts + 1,), T.int32),
        write_cursors: T.Tensor((num_experts,), T.int32),
        token_indices: T.Tensor((pool_rows,), T.int32),
        topk_slots: T.Tensor((pool_rows,), T.int32),
        local_expert_indices: T.Tensor((pool_rows,), T.int32),
        l1_topk_weights: T.Tensor((pool_rows,), T.float32),
        y: T.Tensor((y_rows, hidden), T.bfloat16),
        gpu_barrier: T.Tensor((1,), T.uint32),
        local_expert_start: T.int32,
        num_tokens: T.int32,
    ):
        with T.Kernel(sm_num, threads=128) as block_id:
            A_shared = T.alloc_shared((num_stages, block_M, block_K), T.float8_e4m3fn)
            B_shared = T.alloc_shared((num_stages, block_N, block_K), T.float4_e2m1_unpacked)
            SFA_shared = T.alloc_shared((num_stages, block_M), "uint32")
            SFB_shared = T.alloc_shared((num_stages, block_N), "uint32")

            C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            SFA_tmem = T.alloc_tmem([block_M, 4], "uint32")
            SFB_tmem = T.alloc_tmem([block_M, block_N // 128 * 4], "uint32")
            SFB_l2_tmem = T.alloc_tmem([block_M, l2_N // 128 * 4], "uint32")

            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), T.bfloat16)
            exp_shared = T.alloc_shared((block_M, 4), T.int32)
            scale_shared = T.alloc_shared((block_M, 4), T.float32)

            l1_loaded = T.alloc_barrier([32] * num_stages)
            l1_with_sf_full = T.alloc_barrier([32] * num_stages)
            l1_consumed = T.alloc_barrier([1] * num_stages)
            l1_tmem_full = T.alloc_barrier([1])
            l2_loaded = T.alloc_barrier([32])
            l2_with_sf_full = T.alloc_barrier([32])
            l2_consumed = T.alloc_barrier([1])
            l2_tmem_full = T.alloc_barrier([1])

            tx = T.get_thread_binding()
            T.use_swizzle(8)

            T.init_barrier_gpu(gpu_barrier[0], sm_num)

            if T.Cast("int32", block_id) == 0:
                for expert_init_i in T.serial(T.ceildiv(num_experts, 128)):
                    expert_i = expert_init_i * 128 + tx
                    if expert_i < num_experts:
                        expert_counts[expert_i] = 0
                        expert_offsets[expert_i] = 0
                        write_cursors[expert_i] = 0
                if tx == 0:
                    expert_offsets[num_experts] = 0
                for pool_init_i in T.serial(T.ceildiv(pool_rows, 128)):
                    pool_idx = pool_init_i * 128 + tx
                    if pool_idx < pool_rows:
                        token_indices[pool_idx] = -1
                        topk_slots[pool_idx] = -1
                        local_expert_indices[pool_idx] = -1
                T.sync_threads()

                for expert_i in T.serial(T.ceildiv(num_experts, 128)):
                    local_expert = expert_i * 128 + tx
                    if local_expert < num_experts:
                        global_expert = local_expert_start + local_expert
                        count = T.alloc_var(T.int32, init=0)
                        for token in T.serial(max_tokens):
                            if token < num_tokens:
                                for slot in T.serial(num_topk):
                                    if T.Cast("int32", topk_idx[token, slot]) == global_expert:
                                        count = count + T.int32(1)
                        expert_counts[local_expert] = count
                T.sync_threads()

                if tx == 0:
                    running = T.alloc_var(T.int32, init=0)
                    block_m_i = T.int32(block_M)
                    for expert_i in T.serial(num_experts):
                        start = ((running + block_m_i - T.int32(1)) // block_m_i) * block_m_i
                        expert_offsets[expert_i] = start
                        write_cursors[expert_i] = start
                        running = start + expert_counts[expert_i]
                    expert_offsets[num_experts] = running
                T.sync_threads()

                for expert_i in T.serial(T.ceildiv(num_experts, 128)):
                    local_expert = expert_i * 128 + tx
                    if local_expert < num_experts:
                        global_expert = local_expert_start + local_expert
                        pool_cursor = T.alloc_var(T.int32, init=write_cursors[local_expert])
                        for token in T.serial(max_tokens):
                            if token < num_tokens:
                                for slot in T.serial(num_topk):
                                    if T.Cast("int32", topk_idx[token, slot]) == global_expert:
                                        if pool_cursor < pool_rows:
                                            token_indices[pool_cursor] = token
                                            topk_slots[pool_cursor] = slot
                                            local_expert_indices[pool_cursor] = local_expert
                                        pool_cursor = pool_cursor + T.int32(1)
                        write_cursors[local_expert] = pool_cursor
                T.sync_threads()
                for work_i in T.serial(T.ceildiv(stage_work, 128)):
                    linear = work_i * 128 + tx
                    if linear < x_work:
                        pool_idx = linear // hidden
                        h = linear - pool_idx * hidden
                        token = token_indices[pool_idx]
                        if token >= 0 and token < num_tokens:
                            l1_A[pool_idx, h] = x[token, h]
                        else:
                            l1_A[pool_idx, h] = T.cast(0.0, T.float8_e4m3fn)
                    else:
                        tail = linear - x_work
                        if tail < sf_work:
                            sf_pool_idx = tail // l1_sf_k_groups
                            sf_col = tail - sf_pool_idx * l1_sf_k_groups
                            sf_token = token_indices[sf_pool_idx]
                            if sf_token >= 0 and sf_token < num_tokens:
                                l1_SFA[sf_col * padded_l1_sf_pool_rows + sf_pool_idx] = T.Cast(
                                    "uint32", x_sf[sf_token, sf_col]
                                )
                            else:
                                l1_SFA[sf_col * padded_l1_sf_pool_rows + sf_pool_idx] = T.uint32(PACKED_UE8M0_ONE)
                        else:
                            tail2 = tail - sf_work
                            if tail2 < pool_rows:
                                weight_pool_idx = tail2
                                weight_token = token_indices[weight_pool_idx]
                                slot = topk_slots[weight_pool_idx]
                                if weight_token >= 0 and weight_token < num_tokens and slot >= 0 and slot < num_topk:
                                    l1_topk_weights[weight_pool_idx] = topk_weights[weight_token, slot]
                                else:
                                    l1_topk_weights[weight_pool_idx] = 0.0
                            else:
                                y_linear = tail2 - pool_rows
                                if y_linear < y_rows * hidden:
                                    y_token = y_linear // hidden
                                    col = y_linear - y_token * hidden
                                    y[y_token, col] = T.Cast("bfloat16", 0.0)
                T.fence_sys()

            T.sync_barrier_gpu(gpu_barrier[0])

            for iter_i in T.serial(waves):
                schedule_idx = T.Cast("int32", block_id) + T.Cast("int32", iter_i) * T.int32(sm_num)
                remaining = T.alloc_var(T.int32, init=schedule_idx)
                expert = T.alloc_var(T.int32, init=-1)
                token_start = T.alloc_var(T.int32, init=0)
                for candidate in T.serial(num_experts):
                    candidate_blocks = T.ceildiv(expert_counts[candidate], block_M)
                    if expert < 0:
                        if remaining < candidate_blocks:
                            expert = candidate
                            token_start = remaining * T.int32(block_M)
                        else:
                            remaining = remaining - candidate_blocks
                valid_block = expert >= 0
                tile_m = T.Select(valid_block, expert_offsets[expert] + token_start, T.int32(0))
                actual_rows = T.Select(
                    valid_block,
                    T.min(T.int32(block_M), expert_counts[expert] - token_start),
                    T.int32(0),
                )

                if tx < 32:
                    for k in T.serial(l1_k_iters):
                        stage = k % num_stages
                        phase = ((k // num_stages) + iter_i) & 1
                        T.mbarrier_wait_parity(l1_consumed[stage], phase ^ 1)
                        T.tma_copy(
                            l1_A[tile_m : tile_m + block_M, k * block_K : (k + 1) * block_K],
                            A_shared[stage, :, :],
                            barrier=l1_loaded[stage],
                        )
                        if valid_block:
                            T.tma_copy(
                                l1_B[expert, :, k * block_K : (k + 1) * block_K],
                                B_shared[stage, :, :],
                                barrier=l1_loaded[stage],
                            )
                        T.tma_copy(
                            l1_SFA[k * padded_l1_sf_pool_rows + tile_m : k * padded_l1_sf_pool_rows + tile_m + block_M],
                            SFA_shared[stage, :],
                            barrier=l1_loaded[stage],
                        )
                        if valid_block:
                            T.tma_copy(
                                l1_SFB[
                                    k * num_experts * l1_N
                                    + expert * l1_N : k * num_experts * l1_N
                                    + expert * l1_N
                                    + l1_N
                                ],
                                SFB_shared[stage, :],
                                barrier=l1_loaded[stage],
                            )
                        T.mbarrier_arrive(l1_loaded[stage])

                elif tx < 64:
                    for k in T.serial(l1_k_iters):
                        stage = k % num_stages
                        phase = ((k // num_stages) + iter_i) & 1
                        T.mbarrier_wait_parity(l1_loaded[stage], phase)
                        T.mbarrier_wait_parity(l1_with_sf_full[stage], phase)
                        T.tcgen05_cp_warpx4(SFA_shared[stage, :], SFA_tmem)
                        T.tcgen05_cp_warpx4(SFB_shared[stage, :], SFB_tmem)
                        T.tcgen05_gemm_blockscaled(
                            A_shared[stage, :, :],
                            B_shared[stage, :, :],
                            C_tmem,
                            SFA_tmem,
                            SFB_tmem,
                            transpose_B=True,
                            mbar=l1_consumed[stage],
                            clear_accum=k == 0,
                            k_start=k * block_K,
                            sf_a_granularity_k=sf_granularity_k,
                            sf_b_granularity_k=sf_granularity_k,
                        )
                    T.tcgen05_mma_arrive(l1_tmem_full)

                elif tx < 96:
                    for k in T.serial(l1_k_iters):
                        stage = k % num_stages
                        phase = ((k // num_stages) + iter_i) & 1
                        T.mbarrier_wait_parity(l1_loaded[stage], phase)
                        T.tcgen05_sf_warp_transpose(SFA_shared[stage, :])
                        T.tcgen05_sf_warp_transpose(SFB_shared[stage, :])
                        T.fence_proxy_async()
                        T.mbarrier_arrive(l1_with_sf_full[stage])

                T.mbarrier_wait_parity(l1_tmem_full, iter_i & 1)
                T.sync_threads()
                T.copy(C_tmem, C_local)
                T.copy(C_local, C_shared)
                T.sync_threads()

                for row, sf_group in T.Parallel(block_M, 4):
                    valid = row < actual_rows
                    amax = T.alloc_var(T.float32, init=1.0e-4)
                    for j in T.serial(32):
                        i = sf_group * 32 + j
                        chunk = i // 8
                        in_chunk = i - chunk * 8
                        gate_col = chunk * 16 + in_chunk
                        up_col = chunk * 16 + 8 + in_chunk
                        act = T.alloc_var(T.float32, init=0.0)
                        if valid:
                            gate = T.alloc_var(T.float32)
                            up = T.alloc_var(T.float32)
                            gate = T.Cast("float32", C_shared[row, gate_col])
                            up = T.Cast("float32", C_shared[row, up_col])
                            if use_activation_clamp:
                                gate = T.min(gate, activation_clamp)
                                up = T.max(T.min(up, activation_clamp), -activation_clamp)
                            silu = gate / (1.0 + T.exp(-gate))
                            act = silu * up * l1_topk_weights[tile_m + row]
                        amax = T.max(amax, T.abs(act))
                    exp = T.alloc_var(T.int32)
                    exp = _ceil_to_ue8m0_exp_i32(amax / 448.0)
                    exp_shared[row, sf_group] = exp
                    scale_shared[row, sf_group] = _fast_pow2_i32(exp - 127)
                T.sync_threads()

                for row, i in T.Parallel(block_M, block_K):
                    act = T.alloc_var(T.float32, init=0.0)
                    if row < actual_rows:
                        chunk = i // 8
                        in_chunk = i - chunk * 8
                        gate_col = chunk * 16 + in_chunk
                        up_col = chunk * 16 + 8 + in_chunk
                        gate = T.alloc_var(T.float32)
                        up = T.alloc_var(T.float32)
                        gate = T.Cast("float32", C_shared[row, gate_col])
                        up = T.Cast("float32", C_shared[row, up_col])
                        if use_activation_clamp:
                            gate = T.min(gate, activation_clamp)
                            up = T.max(T.min(up, activation_clamp), -activation_clamp)
                        silu = gate / (1.0 + T.exp(-gate))
                        act = silu * up * l1_topk_weights[tile_m + row]
                        scale = scale_shared[row, i // 32]
                        A_shared[0, row, i] = T.cast(act / scale, T.float8_e4m3fn)
                    else:
                        A_shared[0, row, i] = T.cast(0.0, T.float8_e4m3fn)

                for row in T.Parallel(block_M):
                    packed = T.alloc_var(T.uint32)
                    if row < actual_rows:
                        packed = (
                            T.Cast("uint32", exp_shared[row, 0])
                            | T.shift_left(T.Cast("uint32", exp_shared[row, 1]), 8)
                            | T.shift_left(T.Cast("uint32", exp_shared[row, 2]), 16)
                            | T.shift_left(T.Cast("uint32", exp_shared[row, 3]), 24)
                        )
                        SFA_shared[0, row] = packed
                    else:
                        SFA_shared[0, row] = T.uint32(PACKED_UE8M0_ONE)
                T.sync_threads()

                if tx < 32:
                    T.mbarrier_wait_parity(l2_consumed[0], (iter_i & 1) ^ 1)
                    if valid_block:
                        T.tma_copy(l2_B[expert, :, :], B_shared[0, 0:l2_N, :], barrier=l2_loaded)
                        T.tma_copy(
                            l2_SFB[expert * l2_N : expert * l2_N + l2_N],
                            SFB_shared[0, 0:l2_N],
                            barrier=l2_loaded,
                        )
                    T.mbarrier_arrive(l2_loaded)

                elif tx < 64:
                    T.mbarrier_wait_parity(l2_loaded, iter_i & 1)
                    T.mbarrier_wait_parity(l2_with_sf_full, iter_i & 1)
                    T.tcgen05_cp_warpx4(SFA_shared[0, :], SFA_tmem)
                    T.tcgen05_cp_warpx4(SFB_shared[0, 0:l2_N], SFB_l2_tmem)
                    T.tcgen05_gemm_blockscaled(
                        A_shared[0, :, :],
                        B_shared[0, 0:l2_N, :],
                        C_tmem[:, 0:l2_N],
                        SFA_tmem,
                        SFB_l2_tmem,
                        transpose_B=True,
                        mbar=l2_consumed[0],
                        clear_accum=True,
                        k_start=0,
                        sf_a_granularity_k=sf_granularity_k,
                        sf_b_granularity_k=sf_granularity_k,
                    )
                    T.tcgen05_mma_arrive(l2_tmem_full)

                elif tx < 96:
                    T.mbarrier_wait_parity(l2_loaded, iter_i & 1)
                    T.tcgen05_sf_warp_transpose(SFA_shared[0, :])
                    T.tcgen05_sf_warp_transpose(SFB_shared[0, 0:l2_N])
                    T.fence_proxy_async()
                    T.mbarrier_arrive(l2_with_sf_full)

                T.mbarrier_wait_parity(l2_tmem_full, iter_i & 1)
                T.sync_threads()
                T.copy(C_tmem, C_local)
                T.copy(C_local, C_shared)
                T.sync_threads()

                for row, col in T.Parallel(block_M, l2_N):
                    if row < actual_rows:
                        pool_idx = tile_m + row
                        token = token_indices[pool_idx]
                        if token >= 0 and token < y_rows:
                            slot = topk_slots[pool_idx]
                            if slot >= 0 and slot < num_topk:
                                if num_topk != 1:
                                    if col % 2 == 0:
                                        T.atomic_addx2(y[token, col], C_shared[row, col])
                                else:
                                    if slot == 0:
                                        y[token, col] = C_shared[row, col]

    return main


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True},
)
def _make_distributed_dispatch_l1_l2_remote_combine_sm100_megakernel_prim_func(
    pool_rows: int,
    num_experts: int,
    num_topk: int,
    max_tokens: int,
    y_rows: int,
    max_blocks: int,
    num_ranks: int,
    hidden: int,
    padded_l1_sf_pool_rows: int,
    block_M: int,
    block_N: int,
    block_K: int,
    accum_dtype,
    num_stages: int,
    activation_clamp: float,
    use_activation_clamp: bool,
    sf_granularity_k: int = GRAN_K,
):
    assert block_M == 128
    assert block_N == 256
    assert block_K == 128
    assert max_blocks > 0
    assert hidden in (128, 256)
    assert sf_granularity_k == 32

    intermediate_hidden = 128
    l1_N = 256
    l2_N = hidden
    l1_k_iters = hidden // block_K
    l1_sf_k_groups = hidden // 128
    l2_sf_k_groups = 1
    combine_chunk_N = 128
    combine_chunks = hidden // combine_chunk_N
    sm_num = driver.get_num_sms()
    waves = _ceil_div(max_blocks, sm_num)

    x_work = pool_rows * hidden
    sf_work = pool_rows * l1_sf_k_groups
    stage_work = x_work + sf_work + pool_rows

    @T.prim_func
    def main(
        x: T.Tensor((max_tokens, hidden), T.float8_e4m3fn),
        x_sf: T.Tensor((max_tokens, l1_sf_k_groups), T.int32),
        topk_idx: T.Tensor((max_tokens, num_topk), T.int64),
        topk_weights: T.Tensor((max_tokens, num_topk), T.float32),
        l1_A: T.Tensor((pool_rows, hidden), T.float8_e4m3fn),
        l1_B: T.Tensor((num_experts, l1_N, hidden), T.float4_e2m1fn),
        l1_SFA: T.Tensor((l1_sf_k_groups * padded_l1_sf_pool_rows,), T.uint32),
        l1_SFB: T.Tensor((l1_sf_k_groups * num_experts * l1_N,), T.uint32),
        l2_B: T.Tensor((num_experts, l2_N, intermediate_hidden), T.float4_e2m1fn),
        l2_SFB: T.Tensor((l2_sf_k_groups * num_experts * l2_N,), T.uint32),
        expert_counts: T.Tensor((num_experts,), T.int32),
        expert_offsets: T.Tensor((num_experts + 1,), T.int32),
        write_cursors: T.Tensor((num_experts,), T.int32),
        token_indices: T.Tensor((pool_rows,), T.int32),
        topk_slots: T.Tensor((pool_rows,), T.int32),
        local_expert_indices: T.Tensor((pool_rows,), T.int32),
        source_ranks: T.Tensor((pool_rows,), T.int32),
        l1_topk_weights: T.Tensor((pool_rows,), T.float32),
        combine_acts: T.Tensor((num_topk, max_tokens, hidden), T.bfloat16),
        y: T.Tensor((y_rows, hidden), T.bfloat16),
        gpu_barrier: T.Tensor((2,), T.uint32),
        workspace_barrier: T.Tensor((MEGAMOE_WORKSPACE_BARRIER_SLOTS,), T.int32),
        local_expert_start: T.int32,
        num_tokens: T.int32,
    ):
        with T.Kernel(sm_num, threads=128) as block_id:
            A_shared = T.alloc_shared((num_stages, block_M, block_K), T.float8_e4m3fn)
            B_shared = T.alloc_shared((num_stages, block_N, block_K), T.float4_e2m1_unpacked)
            SFA_shared = T.alloc_shared((num_stages, block_M), "uint32")
            SFB_shared = T.alloc_shared((num_stages, block_N), "uint32")

            C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            SFA_tmem = T.alloc_tmem([block_M, 4], "uint32")
            SFB_tmem = T.alloc_tmem([block_M, block_N // 128 * 4], "uint32")
            SFB_l2_tmem = T.alloc_tmem([block_M, l2_N // 128 * 4], "uint32")

            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), T.bfloat16)
            combine_shared = T.alloc_shared((2, combine_chunk_N), T.bfloat16)
            exp_shared = T.alloc_shared((block_M, 4), T.int32)
            scale_shared = T.alloc_shared((block_M, 4), T.float32)
            remote_expert = T.alloc_shared((1,), T.int64)

            l1_loaded = T.alloc_barrier([32] * num_stages)
            l1_with_sf_full = T.alloc_barrier([32] * num_stages)
            l1_consumed = T.alloc_barrier([1] * num_stages)
            l1_tmem_full = T.alloc_barrier([1])
            l2_loaded = T.alloc_barrier([32])
            l2_with_sf_full = T.alloc_barrier([32])
            l2_consumed = T.alloc_barrier([1])
            l2_tmem_full = T.alloc_barrier([1])

            tx = T.get_thread_binding()
            T.use_swizzle(8)

            T.init_barrier_gpu(gpu_barrier[0], sm_num)
            T.init_barrier_gpu(gpu_barrier[1], sm_num)

            if T.Cast("int32", block_id) == 0:
                for expert_init_i in T.serial(T.ceildiv(num_experts, 128)):
                    expert_i = expert_init_i * 128 + tx
                    if expert_i < num_experts:
                        expert_counts[expert_i] = 0
                        expert_offsets[expert_i] = 0
                        write_cursors[expert_i] = 0
                if tx == 0:
                    expert_offsets[num_experts] = 0
                for pool_init_i in T.serial(T.ceildiv(pool_rows, 128)):
                    pool_idx = pool_init_i * 128 + tx
                    if pool_idx < pool_rows:
                        token_indices[pool_idx] = -1
                        topk_slots[pool_idx] = -1
                        local_expert_indices[pool_idx] = -1
                        source_ranks[pool_idx] = -1
                T.sync_threads()

                for expert_i in T.serial(T.ceildiv(num_experts, 128)):
                    local_expert = expert_i * 128 + tx
                    if local_expert < num_experts:
                        global_expert = local_expert_start + local_expert
                        count = T.alloc_var(T.int32, init=0)
                        for src_rank in T.serial(num_ranks):
                            for token in T.serial(max_tokens):
                                if token < num_tokens:
                                    for slot in T.serial(num_topk):
                                        T.get_block(
                                            src=T.address_of(topk_idx[token, slot]),
                                            dst=T.address_of(remote_expert[0]),
                                            size=1,
                                            src_pe=T.Cast("uint64", src_rank),
                                        )
                                        T.fence_sys()
                                        if T.Cast("int32", remote_expert[0]) == global_expert:
                                            count = count + T.int32(1)
                        expert_counts[local_expert] = count
                T.sync_threads()

                if tx == 0:
                    running = T.alloc_var(T.int32, init=0)
                    block_m_i = T.int32(block_M)
                    for expert_i in T.serial(num_experts):
                        start = ((running + block_m_i - T.int32(1)) // block_m_i) * block_m_i
                        expert_offsets[expert_i] = start
                        write_cursors[expert_i] = start
                        running = start + expert_counts[expert_i]
                    expert_offsets[num_experts] = running
                T.sync_threads()

                for expert_i in T.serial(T.ceildiv(num_experts, 128)):
                    local_expert = expert_i * 128 + tx
                    if local_expert < num_experts:
                        global_expert = local_expert_start + local_expert
                        pool_cursor = T.alloc_var(T.int32, init=write_cursors[local_expert])
                        for src_rank in T.serial(num_ranks):
                            for token in T.serial(max_tokens):
                                if token < num_tokens:
                                    for slot in T.serial(num_topk):
                                        T.get_block(
                                            src=T.address_of(topk_idx[token, slot]),
                                            dst=T.address_of(remote_expert[0]),
                                            size=1,
                                            src_pe=T.Cast("uint64", src_rank),
                                        )
                                        T.fence_sys()
                                        if T.Cast("int32", remote_expert[0]) == global_expert:
                                            if pool_cursor < pool_rows:
                                                token_indices[pool_cursor] = token
                                                topk_slots[pool_cursor] = slot
                                                local_expert_indices[pool_cursor] = local_expert
                                                source_ranks[pool_cursor] = T.Cast("int32", src_rank)
                                            pool_cursor = pool_cursor + T.int32(1)
                        write_cursors[local_expert] = pool_cursor
                T.sync_threads()

                for pool_idx in T.serial(pool_rows):
                    token = token_indices[pool_idx]
                    slot = topk_slots[pool_idx]
                    src_rank = source_ranks[pool_idx]
                    if token >= 0 and token < num_tokens and src_rank >= 0 and src_rank < num_ranks:
                        T.get_block(
                            src=T.address_of(x[token, 0]),
                            dst=T.address_of(l1_A[pool_idx, 0]),
                            size=hidden,
                            src_pe=T.Cast("uint64", src_rank),
                        )
                        for sf_col in T.serial(l1_sf_k_groups):
                            T.get_block(
                                src=T.address_of(x_sf[token, sf_col]),
                                dst=T.address_of(l1_SFA[sf_col * padded_l1_sf_pool_rows + pool_idx]),
                                size=1,
                                src_pe=T.Cast("uint64", src_rank),
                            )
                        if slot >= 0 and slot < num_topk:
                            T.get_block(
                                src=T.address_of(topk_weights[token, slot]),
                                dst=T.address_of(l1_topk_weights[pool_idx]),
                                size=1,
                                src_pe=T.Cast("uint64", src_rank),
                            )
                        else:
                            if tx == 0:
                                l1_topk_weights[pool_idx] = 0.0
                    else:
                        for h_i in T.serial(T.ceildiv(hidden, 128)):
                            h = h_i * 128 + tx
                            if h < hidden:
                                l1_A[pool_idx, h] = T.cast(0.0, T.float8_e4m3fn)
                        for sf_i in T.serial(T.ceildiv(l1_sf_k_groups, 128)):
                            sf_col = sf_i * 128 + tx
                            if sf_col < l1_sf_k_groups:
                                l1_SFA[sf_col * padded_l1_sf_pool_rows + pool_idx] = T.uint32(PACKED_UE8M0_ONE)
                        if tx == 0:
                            l1_topk_weights[pool_idx] = 0.0
                T.fence_sys()
                rank = T.Cast("int32", T.get_rank())
                if tx < num_ranks:
                    T.st(
                        workspace_barrier[rank],
                        T.int32(1),
                        scope="sys",
                        sem="release",
                        dst_pe=T.Cast("uint64", tx),
                    )
                T.sync_threads()

                for peer_rank in T.serial(num_ranks):
                    if tx == 0:
                        T.wait_eq(
                            workspace_barrier[peer_rank],
                            T.int32(1),
                            scope="sys",
                            semantics="acquire",
                        )
                T.sync_threads()

            T.sync_barrier_gpu(gpu_barrier[0])

            for iter_i in T.serial(waves):
                schedule_idx = T.Cast("int32", block_id) + T.Cast("int32", iter_i) * T.int32(sm_num)
                remaining = T.alloc_var(T.int32, init=schedule_idx)
                expert = T.alloc_var(T.int32, init=-1)
                token_start = T.alloc_var(T.int32, init=0)
                for candidate in T.serial(num_experts):
                    candidate_blocks = T.ceildiv(expert_counts[candidate], block_M)
                    if expert < 0:
                        if remaining < candidate_blocks:
                            expert = candidate
                            token_start = remaining * T.int32(block_M)
                        else:
                            remaining = remaining - candidate_blocks
                valid_block = expert >= 0
                tile_m = T.Select(valid_block, expert_offsets[expert] + token_start, T.int32(0))
                actual_rows = T.Select(
                    valid_block,
                    T.min(T.int32(block_M), expert_counts[expert] - token_start),
                    T.int32(0),
                )

                if tx < 32:
                    for k in T.serial(l1_k_iters):
                        stage = k % num_stages
                        phase = ((k // num_stages) + iter_i) & 1
                        T.mbarrier_wait_parity(l1_consumed[stage], phase ^ 1)
                        T.tma_copy(
                            l1_A[tile_m : tile_m + block_M, k * block_K : (k + 1) * block_K],
                            A_shared[stage, :, :],
                            barrier=l1_loaded[stage],
                        )
                        if valid_block:
                            T.tma_copy(
                                l1_B[expert, :, k * block_K : (k + 1) * block_K],
                                B_shared[stage, :, :],
                                barrier=l1_loaded[stage],
                            )
                        T.tma_copy(
                            l1_SFA[k * padded_l1_sf_pool_rows + tile_m : k * padded_l1_sf_pool_rows + tile_m + block_M],
                            SFA_shared[stage, :],
                            barrier=l1_loaded[stage],
                        )
                        if valid_block:
                            T.tma_copy(
                                l1_SFB[
                                    k * num_experts * l1_N
                                    + expert * l1_N : k * num_experts * l1_N
                                    + expert * l1_N
                                    + l1_N
                                ],
                                SFB_shared[stage, :],
                                barrier=l1_loaded[stage],
                            )
                        T.mbarrier_arrive(l1_loaded[stage])

                elif tx < 64:
                    for k in T.serial(l1_k_iters):
                        stage = k % num_stages
                        phase = ((k // num_stages) + iter_i) & 1
                        T.mbarrier_wait_parity(l1_loaded[stage], phase)
                        T.mbarrier_wait_parity(l1_with_sf_full[stage], phase)
                        T.tcgen05_cp_warpx4(SFA_shared[stage, :], SFA_tmem)
                        T.tcgen05_cp_warpx4(SFB_shared[stage, :], SFB_tmem)
                        T.tcgen05_gemm_blockscaled(
                            A_shared[stage, :, :],
                            B_shared[stage, :, :],
                            C_tmem,
                            SFA_tmem,
                            SFB_tmem,
                            transpose_B=True,
                            mbar=l1_consumed[stage],
                            clear_accum=k == 0,
                            k_start=k * block_K,
                            sf_a_granularity_k=sf_granularity_k,
                            sf_b_granularity_k=sf_granularity_k,
                        )
                    T.tcgen05_mma_arrive(l1_tmem_full)

                elif tx < 96:
                    for k in T.serial(l1_k_iters):
                        stage = k % num_stages
                        phase = ((k // num_stages) + iter_i) & 1
                        T.mbarrier_wait_parity(l1_loaded[stage], phase)
                        T.tcgen05_sf_warp_transpose(SFA_shared[stage, :])
                        T.tcgen05_sf_warp_transpose(SFB_shared[stage, :])
                        T.fence_proxy_async()
                        T.mbarrier_arrive(l1_with_sf_full[stage])

                T.mbarrier_wait_parity(l1_tmem_full, iter_i & 1)
                T.sync_threads()
                T.copy(C_tmem, C_local)
                T.copy(C_local, C_shared)
                T.sync_threads()

                for row, sf_group in T.Parallel(block_M, 4):
                    valid = row < actual_rows
                    amax = T.alloc_var(T.float32, init=1.0e-4)
                    for j in T.serial(32):
                        i = sf_group * 32 + j
                        chunk = i // 8
                        in_chunk = i - chunk * 8
                        gate_col = chunk * 16 + in_chunk
                        up_col = chunk * 16 + 8 + in_chunk
                        act = T.alloc_var(T.float32, init=0.0)
                        if valid:
                            gate = T.alloc_var(T.float32)
                            up = T.alloc_var(T.float32)
                            gate = T.Cast("float32", C_shared[row, gate_col])
                            up = T.Cast("float32", C_shared[row, up_col])
                            if use_activation_clamp:
                                gate = T.min(gate, activation_clamp)
                                up = T.max(T.min(up, activation_clamp), -activation_clamp)
                            silu = gate / (1.0 + T.exp(-gate))
                            act = silu * up * l1_topk_weights[tile_m + row]
                        amax = T.max(amax, T.abs(act))
                    exp = T.alloc_var(T.int32)
                    exp = _ceil_to_ue8m0_exp_i32(amax / 448.0)
                    exp_shared[row, sf_group] = exp
                    scale_shared[row, sf_group] = _fast_pow2_i32(exp - 127)
                T.sync_threads()

                for row, i in T.Parallel(block_M, block_K):
                    act = T.alloc_var(T.float32, init=0.0)
                    if row < actual_rows:
                        chunk = i // 8
                        in_chunk = i - chunk * 8
                        gate_col = chunk * 16 + in_chunk
                        up_col = chunk * 16 + 8 + in_chunk
                        gate = T.alloc_var(T.float32)
                        up = T.alloc_var(T.float32)
                        gate = T.Cast("float32", C_shared[row, gate_col])
                        up = T.Cast("float32", C_shared[row, up_col])
                        if use_activation_clamp:
                            gate = T.min(gate, activation_clamp)
                            up = T.max(T.min(up, activation_clamp), -activation_clamp)
                        silu = gate / (1.0 + T.exp(-gate))
                        act = silu * up * l1_topk_weights[tile_m + row]
                        scale = scale_shared[row, i // 32]
                        A_shared[0, row, i] = T.cast(act / scale, T.float8_e4m3fn)
                    else:
                        A_shared[0, row, i] = T.cast(0.0, T.float8_e4m3fn)

                for row in T.Parallel(block_M):
                    packed = T.alloc_var(T.uint32)
                    if row < actual_rows:
                        packed = (
                            T.Cast("uint32", exp_shared[row, 0])
                            | T.shift_left(T.Cast("uint32", exp_shared[row, 1]), 8)
                            | T.shift_left(T.Cast("uint32", exp_shared[row, 2]), 16)
                            | T.shift_left(T.Cast("uint32", exp_shared[row, 3]), 24)
                        )
                        SFA_shared[0, row] = packed
                    else:
                        SFA_shared[0, row] = T.uint32(PACKED_UE8M0_ONE)
                T.sync_threads()

                if tx < 32:
                    T.mbarrier_wait_parity(l2_consumed[0], (iter_i & 1) ^ 1)
                    if valid_block:
                        T.tma_copy(l2_B[expert, :, :], B_shared[0, 0:l2_N, :], barrier=l2_loaded)
                        T.tma_copy(
                            l2_SFB[expert * l2_N : expert * l2_N + l2_N],
                            SFB_shared[0, 0:l2_N],
                            barrier=l2_loaded,
                        )
                    T.mbarrier_arrive(l2_loaded)

                elif tx < 64:
                    T.mbarrier_wait_parity(l2_loaded, iter_i & 1)
                    T.mbarrier_wait_parity(l2_with_sf_full, iter_i & 1)
                    T.tcgen05_cp_warpx4(SFA_shared[0, :], SFA_tmem)
                    T.tcgen05_cp_warpx4(SFB_shared[0, 0:l2_N], SFB_l2_tmem)
                    T.tcgen05_gemm_blockscaled(
                        A_shared[0, :, :],
                        B_shared[0, 0:l2_N, :],
                        C_tmem[:, 0:l2_N],
                        SFA_tmem,
                        SFB_l2_tmem,
                        transpose_B=True,
                        mbar=l2_consumed[0],
                        clear_accum=True,
                        k_start=0,
                        sf_a_granularity_k=sf_granularity_k,
                        sf_b_granularity_k=sf_granularity_k,
                    )
                    T.tcgen05_mma_arrive(l2_tmem_full)

                elif tx < 96:
                    T.mbarrier_wait_parity(l2_loaded, iter_i & 1)
                    T.tcgen05_sf_warp_transpose(SFA_shared[0, :])
                    T.tcgen05_sf_warp_transpose(SFB_shared[0, 0:l2_N])
                    T.fence_proxy_async()
                    T.mbarrier_arrive(l2_with_sf_full)

                T.mbarrier_wait_parity(l2_tmem_full, iter_i & 1)
                T.sync_threads()
                T.copy(C_tmem, C_local)
                T.copy(C_local, C_shared)
                T.sync_threads()

                for row in T.serial(block_M):
                    if row < actual_rows:
                        pool_idx = tile_m + row
                        token = token_indices[pool_idx]
                        if token >= 0 and token < max_tokens:
                            slot = topk_slots[pool_idx]
                            dst_rank = source_ranks[pool_idx]
                            if slot >= 0 and slot < num_topk and dst_rank >= 0 and dst_rank < num_ranks:
                                T.put_block(
                                    src=T.address_of(C_shared[row, 0]),
                                    dst=T.address_of(combine_acts[slot, token, 0]),
                                    size=l2_N,
                                    dst_pe=T.Cast("uint64", dst_rank),
                                )
                T.fence_sys()

            T.sync_barrier_gpu(gpu_barrier[1])

            if T.Cast("int32", block_id) == 0:
                rank = T.Cast("int32", T.get_rank())
                if tx < num_ranks:
                    T.st(
                        workspace_barrier[MEGAMOE_WORKSPACE_COMBINE_BARRIER_ROW + rank],
                        T.int32(1),
                        scope="sys",
                        sem="release",
                        dst_pe=T.Cast("uint64", tx),
                    )
            T.sync_threads()

            for peer_rank in T.serial(num_ranks):
                if tx == 0:
                    T.wait_eq(
                        workspace_barrier[MEGAMOE_WORKSPACE_COMBINE_BARRIER_ROW + peer_rank],
                        T.int32(1),
                        scope="sys",
                        semantics="acquire",
                    )
            T.sync_threads()

            T.sync_barrier_gpu(gpu_barrier[1])
            for token_i in T.serial(T.ceildiv(y_rows, sm_num)):
                token = token_i * T.int32(sm_num) + T.Cast("int32", block_id)
                if token < y_rows:
                    for chunk in T.serial(combine_chunks):
                        chunk_start = chunk * combine_chunk_N
                        if num_topk == 1:
                            if T.Cast("int32", topk_idx[token, 0]) >= 0:
                                T.copy(
                                    combine_acts[0, token, chunk_start : chunk_start + combine_chunk_N],
                                    combine_shared[0, 0:combine_chunk_N],
                                )
                            else:
                                for col in T.Parallel(combine_chunk_N):
                                    combine_shared[0, col] = T.Cast("bfloat16", 0.0)
                            T.sync_threads()
                            T.copy(combine_shared[0, 0:combine_chunk_N], y[token, chunk_start : chunk_start + combine_chunk_N])
                        else:
                            for col in T.Parallel(combine_chunk_N):
                                combine_shared[0, col] = T.Cast("bfloat16", 0.0)
                            T.sync_threads()
                            for slot in T.serial(num_topk):
                                if T.Cast("int32", topk_idx[token, slot]) >= 0:
                                    T.copy(
                                        combine_acts[slot, token, chunk_start : chunk_start + combine_chunk_N],
                                        combine_shared[1, 0:combine_chunk_N],
                                    )
                                    T.sync_threads()
                                    for col in T.Parallel(combine_chunk_N):
                                        acc = T.Cast("float32", combine_shared[0, col]) + T.Cast(
                                            "float32", combine_shared[1, col]
                                        )
                                        combine_shared[0, col] = T.Cast("bfloat16", acc)
                                    T.sync_threads()
                            T.copy(combine_shared[0, 0:combine_chunk_N], y[token, chunk_start : chunk_start + combine_chunk_N])

    return main


def _fast_pow2_i32(x):
    bits = (x + 127) << 23
    return T.reinterpret(bits, "float32")


def _ceil_to_ue8m0_exp_i32(x):
    bits = T.reinterpret(x, "uint32")
    exp = T.Cast("int32", T.shift_right(bits, 23) & T.uint32(0xFF))
    mantissa = bits & T.uint32(0x7FFFFF)
    exp = exp + T.Select(mantissa != T.uint32(0), T.int32(1), T.int32(0))
    return T.max(T.min(exp, T.int32(254)), T.int32(1))


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _as_uint32_storage(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype is torch.uint32:
        return tensor.contiguous()
    if tensor.dtype is torch.int32:
        return tensor.contiguous().view(torch.uint32)
    raise ValueError("packed scale factors must be torch.int32 or torch.uint32")


def _packed_weight_sf_group_major(sf: torch.Tensor) -> torch.Tensor:
    if sf.dim() != 3:
        raise ValueError("packed weight scale-factor tensor must be rank-3 [experts, n, packed_k_groups]")
    return sf.permute(2, 0, 1).contiguous().reshape(-1)


def _num_max_pool_tokens(
    num_ranks: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    num_experts_per_rank: int,
) -> int:
    num_max_recv_tokens = num_ranks * num_max_tokens_per_rank
    num_max_experts_per_token = min(num_topk, num_experts_per_rank)
    return align(
        num_max_recv_tokens * num_max_experts_per_token
        + num_experts_per_rank * (DEEPGEMM_MEGAMOE_MAX_BLOCK_M - 1),
        TOKEN_ALIGNMENT,
    )


def _num_padded_sf_pool_tokens(num_max_pool_tokens: int) -> int:
    candidates = (8, 16, 32, 64, 96, 128, 192)
    return max((num_max_pool_tokens // block_m) * align(block_m, 128) for block_m in candidates)


def _require_cuda_dtype(dtype: torch.dtype) -> str:
    if dtype is torch.float8_e4m3fn:
        return "uint8"
    return str(dtype).split(".")[-1]


def _is_sm100_cuda_device(device: torch.device | str | None = None) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        cuda_device = torch.device("cuda" if device is None else device)
        if cuda_device.type != "cuda":
            return False
        device_index = cuda_device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        return torch.cuda.get_device_capability(device_index) == (10, 0)
    except Exception:
        return False


def _uses_distributed_symmetric_path(sym_buffer: "SymmBuffer") -> bool:
    return (
        sym_buffer.num_ranks > 1
        and sym_buffer.group is not None
        and dist.is_available()
        and dist.is_initialized()
    )


def _tensor_from_raw_buffer(
    buffer: torch.Tensor,
    offset: int,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    strides: Tuple[int, ...] | None = None,
) -> torch.Tensor:
    if strides is None:
        storage_numel = math.prod(shape)
    else:
        storage_numel = 1
        for dim, stride in zip(shape, strides):
            storage_numel += (dim - 1) * stride

    if buffer.is_cuda:
        flat = tensor_from_ptr(
            buffer.data_ptr() + offset,
            [storage_numel],
            _require_cuda_dtype(dtype),
            buffer.device.index or 0,
            False,
        ).view(dtype)
        if strides is not None:
            return flat.as_strided(shape, strides)
        return flat.view(shape)

    elem_size = torch.empty((), dtype=dtype).element_size()
    byte_view = buffer[offset : offset + storage_numel * elem_size]
    flat = byte_view.view(dtype)
    if strides is not None:
        return flat.as_strided(shape, strides)
    return flat.view(shape)


@dataclass(frozen=True)
class SymmBufferLayout:
    num_bytes: int
    num_max_pool_tokens: int
    num_padded_sf_pool_tokens: int
    workspace_bytes: int


@dataclass(frozen=True)
class LocalDispatchMap:
    expert_counts: torch.Tensor
    expert_offsets: torch.Tensor
    token_indices: torch.Tensor
    topk_slots: torch.Tensor
    local_expert_indices: torch.Tensor


@dataclass(frozen=True)
class DistributedDispatchMap:
    local_map: LocalDispatchMap
    source_ranks: torch.Tensor


def get_symm_buffer_layout_for_mega_moe(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> SymmBufferLayout:
    assert num_experts % num_ranks == 0
    assert num_ranks <= MEGAMOE_MAX_INTRANODE_RANKS
    assert hidden % 128 == 0 and intermediate_hidden % 128 == 0
    num_experts_per_rank = num_experts // num_ranks
    num_max_pool_tokens = _num_max_pool_tokens(
        num_ranks,
        num_max_tokens_per_rank,
        num_topk,
        num_experts_per_rank,
    )
    num_padded_sf_pool_tokens = _num_padded_sf_pool_tokens(num_max_pool_tokens)

    workspace_bytes = align(MEGAMOE_WORKSPACE_BARRIER_SLOTS * 4, 16)

    offset = workspace_bytes
    offset += num_max_tokens_per_rank * hidden
    offset += num_max_tokens_per_rank * (hidden // 128) * 4
    offset += num_max_tokens_per_rank * num_topk * 8
    offset += num_max_tokens_per_rank * num_topk * 4
    offset += num_max_pool_tokens * hidden
    offset += num_padded_sf_pool_tokens * (hidden // 128) * 4
    offset += num_max_pool_tokens * 4
    offset += num_topk * num_max_tokens_per_rank * hidden * 2
    return SymmBufferLayout(
        offset,
        num_max_pool_tokens,
        num_padded_sf_pool_tokens,
        workspace_bytes,
    )


class SymmBuffer:
    def __init__(
        self,
        group,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        *,
        use_fp8_dispatch: bool = True,
        activation: str = "swiglu",
        device: str | torch.device | None = None,
        use_vmm: bool | None = None,
    ) -> None:
        if activation != "swiglu":
            raise ValueError("Only swiglu activation is supported")
        if not use_fp8_dispatch:
            raise ValueError("MegaMoE requires FP8 dispatch")
        self.group = group
        self.num_experts = int(num_experts)
        self.num_max_tokens_per_rank = align(int(num_max_tokens_per_rank), TOKEN_ALIGNMENT)
        self.num_topk = int(num_topk)
        self.hidden = int(hidden)
        self.intermediate_hidden = int(intermediate_hidden)
        if group is not None and dist.is_available() and dist.is_initialized():
            self.num_ranks = dist.get_world_size(group)
            self.rank = dist.get_rank(group)
        else:
            self.num_ranks = 1
            self.rank = 0
        self.layout = get_symm_buffer_layout_for_mega_moe(
            self.num_ranks,
            self.num_experts,
            self.num_max_tokens_per_rank,
            self.num_topk,
            self.hidden,
            self.intermediate_hidden,
        )
        self.allocation_bytes = align(self.layout.num_bytes, 256)
        allocator_bytes = self.allocation_bytes
        self.allocator = None
        self._peer_tensors: dict[str, list[torch.Tensor]] = {}
        if device is None:
            device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if self.device.type == "cuda" and self.num_ranks > 1:
            allocator_bytes = align(self.layout.num_bytes + 16 * 4096, 256)
            self.allocator = get_allocator(
                allocator_bytes,
                device=str(self.device),
                is_distributed=True,
                local_rank=self.rank,
                num_local_ranks=self.num_ranks,
                group=group,
                use_vmm=use_vmm,
            )
            self.buffer = torch.empty((0,), dtype=torch.int8, device=self.device)
            self.handle = types.SimpleNamespace(
                buffer_ptrs=[int(x) for x in self.allocator._buffer_ptrs.cpu().tolist()]
            )
        else:
            self.buffer = torch.empty((self.allocation_bytes,), dtype=torch.int8, device=self.device)
            buffer_ptrs = [self.buffer.data_ptr()] if self.device.type == "cuda" else [0]
            self.handle = types.SimpleNamespace(buffer_ptrs=buffer_ptrs)

        self.buffer.zero_()
        self._buffer_specs: dict[str, tuple[int, tuple[int, ...], torch.dtype, tuple[int, ...] | None]] = {}
        if self.allocator is not None:
            self._allocate_distributed_input_buffers()
        else:
            self._slice_input_buffers()
        if self.group is not None and dist.is_available() and dist.is_initialized():
            dist.barrier(self.group)

    def _allocate_distributed_tensor(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        storage_dtype = torch.uint8 if dtype is torch.float8_e4m3fn else dtype
        peers = distributed_tensor(shape, storage_dtype, allocator=self.allocator, return_peers=True)
        if storage_dtype is not dtype:
            peers = [peer.view(dtype) for peer in peers]
        self._peer_tensors[name] = peers
        return peers[self.rank]

    def _allocate_distributed_input_buffers(self) -> None:
        hidden = self.hidden
        max_tokens = self.num_max_tokens_per_rank
        max_pool = self.layout.num_max_pool_tokens
        padded_sf_pool = self.layout.num_padded_sf_pool_tokens

        self.workspace = self._allocate_distributed_tensor(
            "workspace",
            (self.layout.workspace_bytes,),
            torch.uint8,
        )
        self._bind_workspace_views()
        self.x = self._allocate_distributed_tensor("x", (max_tokens, hidden), torch.float8_e4m3fn)
        self.x_sf = self._allocate_distributed_tensor("x_sf", (max_tokens, hidden // 128), torch.int32)
        self.topk_idx = self._allocate_distributed_tensor("topk_idx", (max_tokens, self.num_topk), torch.int64)
        self.topk_weights = self._allocate_distributed_tensor("topk_weights", (max_tokens, self.num_topk), torch.float32)
        self.l1_acts = self._allocate_distributed_tensor("l1_acts", (max_pool, hidden), torch.float8_e4m3fn)
        self.l1_acts_sf_storage = self._allocate_distributed_tensor(
            "l1_acts_sf_storage",
            (padded_sf_pool * (hidden // 128),),
            torch.int32,
        )
        self.l1_acts_sf = self.l1_acts_sf_storage.as_strided(
            (padded_sf_pool, hidden // 128),
            (1, padded_sf_pool),
        )
        self.l1_topk_weights = self._allocate_distributed_tensor("l1_topk_weights", (max_pool,), torch.float32)
        self.combine_acts = self._allocate_distributed_tensor(
            "combine_acts",
            (self.num_topk, max_tokens, hidden),
            torch.bfloat16,
        )

    def _bind_workspace_views(self) -> None:
        barrier_bytes = MEGAMOE_WORKSPACE_BARRIER_SLOTS * 4
        self.workspace_barrier = self.workspace[:barrier_bytes]

    def _view_from_buffer(
        self,
        buffer: torch.Tensor,
        offset: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        strides: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        return _tensor_from_raw_buffer(buffer, offset, shape, dtype, strides)

    def _record_buffer_spec(
        self,
        name: str,
        offset: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        strides: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        self._buffer_specs[name] = (offset, shape, dtype, strides)
        return self._view_from_buffer(self.buffer, offset, shape, dtype, strides)

    def _slice_input_buffers(self) -> None:
        hidden = self.hidden
        max_tokens = self.num_max_tokens_per_rank
        max_pool = self.layout.num_max_pool_tokens
        padded_sf_pool = self.layout.num_padded_sf_pool_tokens
        self.workspace = self._record_buffer_spec(
            "workspace",
            0,
            (self.layout.workspace_bytes,),
            torch.uint8,
        )
        self._bind_workspace_views()
        offset = self.layout.num_bytes
        # Recompute offsets from the end backwards to avoid duplicating the
        # workspace size expression in slicing code.
        combine_bytes = self.num_topk * max_tokens * hidden * 2
        offset -= combine_bytes
        self.combine_acts = self._record_buffer_spec(
            "combine_acts",
            offset,
            (self.num_topk, max_tokens, hidden),
            torch.bfloat16,
        )
        offset -= max_pool * 4
        self.l1_topk_weights = self._record_buffer_spec("l1_topk_weights", offset, (max_pool,), torch.float32)
        offset -= padded_sf_pool * (hidden // 128) * 4
        self.l1_acts_sf = self._record_buffer_spec(
            "l1_acts_sf",
            offset,
            (padded_sf_pool, hidden // 128),
            torch.int32,
            strides=(1, padded_sf_pool),
        )
        offset -= max_pool * hidden
        self.l1_acts = self._record_buffer_spec("l1_acts", offset, (max_pool, hidden), torch.float8_e4m3fn)
        offset -= max_tokens * self.num_topk * 4
        self.topk_weights = self._record_buffer_spec(
            "topk_weights",
            offset,
            (max_tokens, self.num_topk),
            torch.float32,
        )
        offset -= max_tokens * self.num_topk * 8
        self.topk_idx = self._record_buffer_spec("topk_idx", offset, (max_tokens, self.num_topk), torch.int64)
        offset -= max_tokens * (hidden // 128) * 4
        self.x_sf = self._record_buffer_spec("x_sf", offset, (max_tokens, hidden // 128), torch.int32)
        offset -= max_tokens * hidden
        self.x = self._record_buffer_spec("x", offset, (max_tokens, hidden), torch.float8_e4m3fn)

    def peer_tensor(self, rank: int, name: str) -> torch.Tensor:
        if rank < 0 or rank >= self.num_ranks:
            raise ValueError("rank is outside the symmetric buffer rank range")
        if name in self._peer_tensors:
            return self._peer_tensors[name][rank]
        if name == "l1_acts_sf" and "l1_acts_sf_storage" in self._peer_tensors:
            storage = self._peer_tensors["l1_acts_sf_storage"][rank]
            return storage.as_strided(self.l1_acts_sf.shape, self.l1_acts_sf.stride())
        if name not in self._buffer_specs:
            raise ValueError(f"unknown symmetric buffer tensor {name!r}")
        if self.device.type != "cuda":
            if rank != self.rank:
                raise ValueError("CPU symmetric buffer only has a local rank view")
            return getattr(self, name)
        offset, shape, dtype, strides = self._buffer_specs[name]
        base_ptr = int(self.handle.buffer_ptrs[rank])
        return _tensor_from_raw_buffer(
            tensor_from_ptr(base_ptr, [self.allocation_bytes], "int8", self.device.index or 0, False),
            offset,
            shape,
            dtype,
            strides,
        )

    def destroy(self) -> None:
        if self.group is not None and dist.is_available() and dist.is_initialized():
            try:
                dist.barrier(self.group)
            except Exception:
                pass
        if self.allocator is not None:
            self.allocator.close()
        self.handle = None
        self.buffer = None
        self.group = None


def get_symm_buffer_for_mega_moe(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    use_fp8_dispatch: bool = True,
    activation: str = "swiglu",
    *,
    device: str | torch.device | None = None,
    use_vmm: bool | None = None,
) -> SymmBuffer:
    return SymmBuffer(
        group,
        num_experts,
        align(num_max_tokens_per_rank, TOKEN_ALIGNMENT),
        num_topk,
        hidden,
        intermediate_hidden,
        use_fp8_dispatch=use_fp8_dispatch,
        activation=activation,
        device=device,
        use_vmm=use_vmm,
    )


def _copy_inputs_to_buffer(sym_buffer: SymmBuffer, inputs: MegaMoEInputs, num_tokens: int | None = None) -> None:
    if num_tokens is None:
        num_tokens = inputs.x_fp8.size(0)
    sym_buffer.x[:num_tokens].copy_(inputs.x_fp8[:num_tokens])
    sym_buffer.x_sf[:num_tokens].copy_(inputs.x_sf[:num_tokens])
    sym_buffer.topk_idx[:num_tokens].copy_(inputs.topk_idx[:num_tokens])
    sym_buffer.topk_weights[:num_tokens].copy_(inputs.topk_weights[:num_tokens])


def _all_gather_weight_tuple(
    weights: Tuple[torch.Tensor, torch.Tensor],
    group,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gathered = []
    for tensor in weights:
        parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
        dist.all_gather(parts, tensor.contiguous(), group=group)
        gathered.append(torch.cat(parts, dim=0).contiguous())
    return gathered[0], gathered[1]


def _build_local_dispatch_map_python(
    topk_idx: torch.Tensor,
    num_experts_per_rank: int,
    local_expert_start: int,
) -> LocalDispatchMap:
    capacity = topk_idx.numel()
    device = topk_idx.device
    expert_counts = torch.zeros((num_experts_per_rank,), dtype=torch.int32, device=device)
    expert_offsets = torch.zeros((num_experts_per_rank + 1,), dtype=torch.int32, device=device)
    token_indices = torch.full((capacity,), -1, dtype=torch.int32, device=device)
    topk_slots = torch.full((capacity,), -1, dtype=torch.int32, device=device)
    local_expert_indices = torch.full((capacity,), -1, dtype=torch.int32, device=device)

    cursor = 0
    for local_expert in range(num_experts_per_rank):
        expert = local_expert_start + local_expert
        positions = (topk_idx == expert).nonzero(as_tuple=False)
        count = positions.size(0)
        expert_counts[local_expert] = count
        expert_offsets[local_expert] = cursor
        if count:
            end = cursor + count
            token_indices[cursor:end] = positions[:, 0].to(torch.int32)
            topk_slots[cursor:end] = positions[:, 1].to(torch.int32)
            local_expert_indices[cursor:end] = local_expert
            cursor = end
    expert_offsets[num_experts_per_rank] = cursor
    return LocalDispatchMap(
        expert_counts=expert_counts,
        expert_offsets=expert_offsets,
        token_indices=token_indices,
        topk_slots=topk_slots,
        local_expert_indices=local_expert_indices,
    )


def _aligned_dispatch_capacity_upper_bound(total_entries: int, num_experts_per_rank: int, block_m: int) -> int:
    if block_m <= 0:
        raise ValueError("block_m must be positive")
    return align(total_entries + num_experts_per_rank * (block_m - 1), block_m)


def build_local_dispatch_map(
    topk_idx: torch.Tensor,
    num_experts_per_rank: int,
    local_expert_start: int = 0,
    *,
    num_tokens: int | None = None,
) -> LocalDispatchMap:
    """Group valid top-k entries by local expert for the MegaMoE dispatch path."""
    if topk_idx.dim() != 2:
        raise ValueError("topk_idx must be [num_tokens, num_topk]")
    if num_experts_per_rank <= 0:
        raise ValueError("num_experts_per_rank must be positive")
    if num_tokens is None:
        num_tokens = topk_idx.size(0)
    return _build_local_dispatch_map_python(
        topk_idx[:num_tokens],
        num_experts_per_rank,
        local_expert_start,
    )


def _flat_pool_sf_storage(sf_tensor: torch.Tensor) -> torch.Tensor:
    storage_numel = 1
    for dim, stride in zip(sf_tensor.shape, sf_tensor.stride()):
        storage_numel += (dim - 1) * stride
    return sf_tensor.as_strided((storage_numel,), (1,))


def _run_local_dispatch_l1_l2_combine_sm100_megakernel(
    y: torch.Tensor,
    sym_buffer: SymmBuffer,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    transformed_weights: bool,
    activation_clamp: Optional[float],
    *,
    num_tokens: int,
    num_experts_per_rank: int,
    local_expert_start: int,
) -> LocalDispatchMap | None:
    if not transformed_weights:
        return None
    if (
        sym_buffer.device.type != "cuda"
        or sym_buffer.hidden not in (128, 256)
        or sym_buffer.intermediate_hidden != 128
        or y.shape != (num_tokens, sym_buffer.hidden)
        or y.dtype is not torch.bfloat16
        or not y.is_cuda
        or not y.is_contiguous()
        or sym_buffer.num_topk < 1
        or num_tokens < 0
        or num_tokens > sym_buffer.num_max_tokens_per_rank
    ):
        return None
    active_entries = num_tokens * sym_buffer.num_topk
    if active_entries > MEGAMOE_LOCAL_FUSED_MAP_MAX_ENTRIES:
        return None
    required_capacity = _aligned_dispatch_capacity_upper_bound(active_entries, num_experts_per_rank, 128)
    if required_capacity > sym_buffer.l1_acts.shape[0]:
        return None
    capacity = required_capacity

    l1_w_fp4, l1_w_sf = l1_weights
    l2_w_fp4, l2_w_sf = l2_weights
    l1_w_fp4 = l1_w_fp4.contiguous()
    l2_w_fp4 = l2_w_fp4.contiguous()
    l1_w_sf_packed = inverse_transpose_sf_for_utccp(l1_w_sf)
    l2_w_sf_packed = inverse_transpose_sf_for_utccp(l2_w_sf)
    if (
        l1_w_fp4.dtype is not torch.int8
        or l1_w_sf_packed.dtype is not torch.int32
        or l2_w_fp4.dtype is not torch.int8
        or l2_w_sf_packed.dtype is not torch.int32
        or not l1_w_fp4.is_cuda
        or not l1_w_sf_packed.is_cuda
        or not l2_w_fp4.is_cuda
        or not l2_w_sf_packed.is_cuda
        or not l1_w_fp4.is_contiguous()
        or not l1_w_sf_packed.is_contiguous()
        or not l2_w_fp4.is_contiguous()
        or not l2_w_sf_packed.is_contiguous()
        or not sym_buffer.x.is_cuda
        or not sym_buffer.x_sf.is_cuda
        or not sym_buffer.topk_idx.is_cuda
        or not sym_buffer.topk_weights.is_cuda
        or not sym_buffer.x.is_contiguous()
        or not sym_buffer.x_sf.is_contiguous()
        or not sym_buffer.topk_idx.is_contiguous()
        or not sym_buffer.topk_weights.is_contiguous()
    ):
        return None
    if l1_w_fp4.shape != (num_experts_per_rank, 256, sym_buffer.hidden // 2):
        return None
    if l2_w_fp4.shape != (num_experts_per_rank, sym_buffer.hidden, 64):
        return None

    max_blocks = _ceil_div(capacity, 128)
    if max_blocks == 0:
        y.zero_()
        return LocalDispatchMap(
            expert_counts=torch.empty((num_experts_per_rank,), dtype=torch.int32, device=sym_buffer.device),
            expert_offsets=torch.empty((num_experts_per_rank + 1,), dtype=torch.int32, device=sym_buffer.device),
            token_indices=torch.empty((capacity,), dtype=torch.int32, device=sym_buffer.device),
            topk_slots=torch.empty((capacity,), dtype=torch.int32, device=sym_buffer.device),
            local_expert_indices=torch.empty((capacity,), dtype=torch.int32, device=sym_buffer.device),
        )
    expert_counts = torch.empty((num_experts_per_rank,), dtype=torch.int32, device=sym_buffer.device)
    expert_offsets = torch.empty((num_experts_per_rank + 1,), dtype=torch.int32, device=sym_buffer.device)
    write_cursors = torch.empty((num_experts_per_rank,), dtype=torch.int32, device=sym_buffer.device)
    token_indices = torch.empty((capacity,), dtype=torch.int32, device=sym_buffer.device)
    topk_slots = torch.empty((capacity,), dtype=torch.int32, device=sym_buffer.device)
    local_expert_indices = torch.empty((capacity,), dtype=torch.int32, device=sym_buffer.device)
    gpu_barrier = torch.empty((1,), dtype=torch.uint32, device=sym_buffer.device)
    kernel = _make_local_dispatch_l1_l2_combine_sm100_megakernel_prim_func(
        capacity,
        num_experts_per_rank,
        sym_buffer.num_topk,
        sym_buffer.num_max_tokens_per_rank,
        num_tokens,
        max_blocks,
        sym_buffer.hidden,
        sym_buffer.l1_acts_sf.shape[0],
        128,
        256,
        128,
        T.float32,
        2,
        0.0 if activation_clamp is None else float(activation_clamp),
        activation_clamp is not None,
        sf_granularity_k=GRAN_K,
    )
    kernel(
        sym_buffer.x,
        sym_buffer.x_sf,
        sym_buffer.topk_idx,
        sym_buffer.topk_weights,
        sym_buffer.l1_acts[:capacity],
        l1_w_fp4,
        _as_uint32_storage(_flat_pool_sf_storage(sym_buffer.l1_acts_sf)),
        _as_uint32_storage(_packed_weight_sf_group_major(l1_w_sf_packed)),
        l2_w_fp4,
        _as_uint32_storage(_packed_weight_sf_group_major(l2_w_sf_packed)),
        expert_counts,
        expert_offsets,
        write_cursors,
        token_indices,
        topk_slots,
        local_expert_indices,
        sym_buffer.l1_topk_weights[:capacity],
        y,
        gpu_barrier,
        int(local_expert_start),
        int(num_tokens),
    )
    return LocalDispatchMap(
        expert_counts=expert_counts,
        expert_offsets=expert_offsets,
        token_indices=token_indices,
        topk_slots=topk_slots,
        local_expert_indices=local_expert_indices,
    )


def _run_distributed_dispatch_l1_l2_remote_combine_sm100_megakernel(
    y: torch.Tensor,
    sym_buffer: SymmBuffer,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    transformed_weights: bool,
    activation_clamp: Optional[float],
    *,
    num_tokens: int,
) -> DistributedDispatchMap | None:
    if os.environ.get("TILESCALE_DISABLE_MEGAMOE_DIST_MEGAKERNEL") in {"1", "true", "True", "TRUE"}:
        return None
    if not transformed_weights:
        return None
    if (
        sym_buffer.allocator is None
        or sym_buffer.device.type != "cuda"
        or sym_buffer.num_ranks <= 1
        or sym_buffer.hidden not in (128, 256)
        or sym_buffer.intermediate_hidden != 128
        or y.shape != (num_tokens, sym_buffer.hidden)
        or y.dtype is not torch.bfloat16
        or not y.is_cuda
        or not y.is_contiguous()
        or not sym_buffer.combine_acts.is_contiguous()
        or not sym_buffer.workspace_barrier.is_contiguous()
        or sym_buffer.num_topk < 1
        or num_tokens < 0
        or num_tokens > sym_buffer.num_max_tokens_per_rank
    ):
        return None

    num_experts_per_rank = sym_buffer.num_experts // sym_buffer.num_ranks
    active_entries = num_tokens * sym_buffer.num_topk * sym_buffer.num_ranks
    if active_entries > MEGAMOE_DISTRIBUTED_FUSED_MAP_MAX_ENTRIES:
        return None
    required_capacity = _aligned_dispatch_capacity_upper_bound(active_entries, num_experts_per_rank, 128)
    if required_capacity > sym_buffer.l1_acts.shape[0]:
        return None
    capacity = required_capacity

    l1_w_fp4, l1_w_sf = l1_weights
    l2_w_fp4, l2_w_sf = l2_weights
    l1_w_fp4 = l1_w_fp4.contiguous()
    l2_w_fp4 = l2_w_fp4.contiguous()
    l1_w_sf_packed = inverse_transpose_sf_for_utccp(l1_w_sf)
    l2_w_sf_packed = inverse_transpose_sf_for_utccp(l2_w_sf)
    if (
        l1_w_fp4.dtype is not torch.int8
        or l1_w_sf_packed.dtype is not torch.int32
        or l2_w_fp4.dtype is not torch.int8
        or l2_w_sf_packed.dtype is not torch.int32
        or not l1_w_fp4.is_cuda
        or not l1_w_sf_packed.is_cuda
        or not l2_w_fp4.is_cuda
        or not l2_w_sf_packed.is_cuda
        or not l1_w_fp4.is_contiguous()
        or not l1_w_sf_packed.is_contiguous()
        or not l2_w_fp4.is_contiguous()
        or not l2_w_sf_packed.is_contiguous()
        or not sym_buffer.x.is_cuda
        or not sym_buffer.x_sf.is_cuda
        or not sym_buffer.topk_idx.is_cuda
        or not sym_buffer.topk_weights.is_cuda
        or not sym_buffer.x.is_contiguous()
        or not sym_buffer.x_sf.is_contiguous()
        or not sym_buffer.topk_idx.is_contiguous()
        or not sym_buffer.topk_weights.is_contiguous()
    ):
        return None
    if l1_w_fp4.shape != (num_experts_per_rank, 256, sym_buffer.hidden // 2):
        return None
    if l2_w_fp4.shape != (num_experts_per_rank, sym_buffer.hidden, 64):
        return None

    max_blocks = _ceil_div(capacity, 128)
    if max_blocks == 0:
        y.zero_()
        empty_map = LocalDispatchMap(
            expert_counts=torch.empty((num_experts_per_rank,), dtype=torch.int32, device=sym_buffer.device),
            expert_offsets=torch.empty((num_experts_per_rank + 1,), dtype=torch.int32, device=sym_buffer.device),
            token_indices=torch.empty((capacity,), dtype=torch.int32, device=sym_buffer.device),
            topk_slots=torch.empty((capacity,), dtype=torch.int32, device=sym_buffer.device),
            local_expert_indices=torch.empty((capacity,), dtype=torch.int32, device=sym_buffer.device),
        )
        return DistributedDispatchMap(
            local_map=empty_map,
            source_ranks=torch.empty((capacity,), dtype=torch.int32, device=sym_buffer.device),
        )
    local_expert_start = sym_buffer.rank * num_experts_per_rank
    expert_counts = torch.empty((num_experts_per_rank,), dtype=torch.int32, device=sym_buffer.device)
    expert_offsets = torch.empty((num_experts_per_rank + 1,), dtype=torch.int32, device=sym_buffer.device)
    write_cursors = torch.empty((num_experts_per_rank,), dtype=torch.int32, device=sym_buffer.device)
    token_indices = torch.empty((capacity,), dtype=torch.int32, device=sym_buffer.device)
    topk_slots = torch.empty((capacity,), dtype=torch.int32, device=sym_buffer.device)
    local_expert_indices = torch.empty((capacity,), dtype=torch.int32, device=sym_buffer.device)
    source_ranks = torch.empty((capacity,), dtype=torch.int32, device=sym_buffer.device)
    gpu_barrier = distributed_tensor((2,), torch.uint32, allocator=sym_buffer.allocator)
    sym_buffer.workspace_barrier.view(torch.int32)[:MEGAMOE_WORKSPACE_BARRIER_SLOTS].zero_()
    torch.cuda.synchronize(sym_buffer.device)
    if sym_buffer.group is not None and dist.is_available() and dist.is_initialized():
        dist.barrier(sym_buffer.group)

    kernel = _make_distributed_dispatch_l1_l2_remote_combine_sm100_megakernel_prim_func(
        capacity,
        num_experts_per_rank,
        sym_buffer.num_topk,
        sym_buffer.num_max_tokens_per_rank,
        num_tokens,
        max_blocks,
        sym_buffer.num_ranks,
        sym_buffer.hidden,
        sym_buffer.l1_acts_sf.shape[0],
        128,
        256,
        128,
        T.float32,
        2,
        0.0 if activation_clamp is None else float(activation_clamp),
        activation_clamp is not None,
        sf_granularity_k=GRAN_K,
    )
    kernel.initialize(allocator=sym_buffer.allocator)
    kernel(
        sym_buffer.x,
        sym_buffer.x_sf,
        sym_buffer.topk_idx,
        sym_buffer.topk_weights,
        sym_buffer.l1_acts[:capacity],
        l1_w_fp4,
        _as_uint32_storage(_flat_pool_sf_storage(sym_buffer.l1_acts_sf)),
        _as_uint32_storage(_packed_weight_sf_group_major(l1_w_sf_packed)),
        l2_w_fp4,
        _as_uint32_storage(_packed_weight_sf_group_major(l2_w_sf_packed)),
        expert_counts,
        expert_offsets,
        write_cursors,
        token_indices,
        topk_slots,
        local_expert_indices,
        source_ranks,
        sym_buffer.l1_topk_weights[:capacity],
        sym_buffer.combine_acts,
        y,
        gpu_barrier,
        sym_buffer.workspace_barrier.view(torch.int32)[:MEGAMOE_WORKSPACE_BARRIER_SLOTS],
        int(local_expert_start),
        int(num_tokens),
    )
    torch.cuda.synchronize(sym_buffer.device)
    return DistributedDispatchMap(
        local_map=LocalDispatchMap(
            expert_counts=expert_counts,
            expert_offsets=expert_offsets,
            token_indices=token_indices,
            topk_slots=topk_slots,
            local_expert_indices=local_expert_indices,
        ),
        source_ranks=source_ranks,
    )


def _update_cumulative_local_expert_recv_stats_from_counts(
    local_map: LocalDispatchMap,
    cumulative_local_expert_recv_stats: torch.Tensor | None,
) -> None:
    if cumulative_local_expert_recv_stats is None:
        return
    counts = local_map.expert_counts
    if cumulative_local_expert_recv_stats.shape != counts.shape:
        raise ValueError("cumulative_local_expert_recv_stats must match local expert count shape")
    cumulative_local_expert_recv_stats.add_(counts.to(cumulative_local_expert_recv_stats.device))


def fp8_fp4_mega_moe(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: SymmBuffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    *,
    transformed_weights: bool = True,
    all_l1_weights: Tuple[torch.Tensor, torch.Tensor] | None = None,
    all_l2_weights: Tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute MegaMoE output into ``y``.

    The signature mirrors DeepGEMM.  ``fast_math`` is accepted for API
    compatibility; the functional backend uses PyTorch math.
    """
    del fast_math
    if recipe != (1, 1, 32):
        raise ValueError(f"Unsupported recipe {recipe}; expected (1, 1, 32)")
    if activation != "swiglu":
        raise ValueError("Only swiglu activation is supported")
    if y.dim() != 2:
        raise ValueError("y must be [num_tokens, hidden]")
    num_tokens, hidden = y.shape
    if hidden != sym_buffer.hidden:
        raise ValueError(f"y hidden={hidden} does not match sym_buffer.hidden={sym_buffer.hidden}")

    num_experts_per_rank = sym_buffer.num_experts // sym_buffer.num_ranks
    local_start = sym_buffer.rank * num_experts_per_rank
    if all_l1_weights is None and all_l2_weights is None and transformed_weights and sym_buffer.device.type == "cuda":
        if _uses_distributed_symmetric_path(sym_buffer):
            megakernel_map = _run_distributed_dispatch_l1_l2_remote_combine_sm100_megakernel(
                y,
                sym_buffer,
                l1_weights,
                l2_weights,
                transformed_weights,
                activation_clamp,
                num_tokens=num_tokens,
            )
            if megakernel_map is not None:
                torch.cuda.synchronize(sym_buffer.device)
                _update_cumulative_local_expert_recv_stats_from_counts(
                    megakernel_map.local_map,
                    cumulative_local_expert_recv_stats,
                )
                return y
        elif _is_sm100_cuda_device(sym_buffer.device):
            megakernel_map = _run_local_dispatch_l1_l2_combine_sm100_megakernel(
                y,
                sym_buffer,
                l1_weights,
                l2_weights,
                transformed_weights,
                activation_clamp,
                num_tokens=num_tokens,
                num_experts_per_rank=num_experts_per_rank,
                local_expert_start=local_start,
            )
            if megakernel_map is not None:
                _update_cumulative_local_expert_recv_stats_from_counts(
                    megakernel_map,
                    cumulative_local_expert_recv_stats,
                )
                return y

    if all_l1_weights is not None or all_l2_weights is not None:
        if all_l1_weights is None or all_l2_weights is None:
            raise ValueError("all_l1_weights and all_l2_weights must be provided together")
        if transformed_weights:
            plain_l1, plain_l2 = inverse_transform_weights_for_mega_moe(all_l1_weights, all_l2_weights)
        else:
            plain_l1, plain_l2 = all_l1_weights, all_l2_weights
        reference_num_experts = plain_l1[0].size(0)
    elif sym_buffer.num_ranks > 1 and sym_buffer.group is not None and dist.is_available() and dist.is_initialized():
        gathered_l1 = _all_gather_weight_tuple(l1_weights, sym_buffer.group)
        gathered_l2 = _all_gather_weight_tuple(l2_weights, sym_buffer.group)
        if transformed_weights:
            plain_l1, plain_l2 = inverse_transform_weights_for_mega_moe(gathered_l1, gathered_l2)
        else:
            plain_l1, plain_l2 = gathered_l1, gathered_l2
        reference_num_experts = plain_l1[0].size(0)
    elif transformed_weights:
        plain_l1, plain_l2 = inverse_transform_weights_for_mega_moe(l1_weights, l2_weights)
        reference_num_experts = plain_l1[0].size(0)
    else:
        plain_l1, plain_l2 = l1_weights, l2_weights
        reference_num_experts = plain_l1[0].size(0)

    result = mega_moe_reference(
        sym_buffer.x[:num_tokens],
        sym_buffer.x_sf[:num_tokens],
        sym_buffer.topk_idx[:num_tokens],
        sym_buffer.topk_weights[:num_tokens],
        plain_l1[0],
        plain_l1[1],
        plain_l2[0],
        plain_l2[1],
        sym_buffer.hidden,
        sym_buffer.intermediate_hidden,
        reference_num_experts,
        gran_k=GRAN_K,
        activation_clamp=activation_clamp,
    )
    y.copy_(result)
    if cumulative_local_expert_recv_stats is not None:
        local_dispatch_map = build_local_dispatch_map(
            sym_buffer.topk_idx,
            num_experts_per_rank,
            local_start,
            num_tokens=num_tokens,
        )
        _update_cumulative_local_expert_recv_stats_from_counts(
            local_dispatch_map,
            cumulative_local_expert_recv_stats,
        )
    return y
