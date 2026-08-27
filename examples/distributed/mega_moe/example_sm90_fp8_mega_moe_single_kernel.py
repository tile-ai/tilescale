"""Experimental single-kernel SM90 FP8 Mega MoE using TileScale.

Dedicated dispatch, GEMM, and combine warps form a cross-wave pipeline. L1
epilogues publish per-(expert, M-block) readiness for L2, while rank-wave
completion flags let combine consume wave N as the GEMM warps enter wave N+1.
The established two-kernel example remains the stable baseline.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing

import tilelang
import tilelang.language as T
from tilelang.distributed.allocator import get_allocator
from tilelang.distributed.bench import do_bench
from tilelang.distributed.host import init_dist

import example_sm90_fp8_mega_moe as two_kernel


os.environ.setdefault("NCCL_DEBUG", "ERROR")

MODEL_CONFIGS = two_kernel.MODEL_CONFIGS
FP8_MAX = two_kernel.FP8_MAX
SCALE_GRANULARITY = two_kernel.SCALE_GRANULARITY


def select_single_kernel_config(
    hidden: int,
    intermediate_hidden: int,
    num_tokens: int,
    num_topk: int,
    num_experts_per_rank: int,
    num_sms: int,
) -> Tuple[str, dict[str, int]]:
    """Collapse the validated L1/L2 schedules into one compatible schedule."""
    family, l1, l2 = two_kernel.select_manual_warp_configs(
        hidden,
        intermediate_hidden,
        num_tokens,
        num_topk,
        num_experts_per_rank,
        num_sms,
    )
    for key in ("block_m", "block_n", "block_k", "threads"):
        assert l1[key] == l2[key]
    preferred_wave_size = 16 if family == "compact" else 24
    preferred_wave_size = min(preferred_wave_size, num_experts_per_rank)
    wave_size = next(
        size
        for size in range(preferred_wave_size, num_experts_per_rank + 1)
        if num_experts_per_rank % size == 0
    )
    return family, {
        "block_m": l1["block_m"],
        "block_n": l1["block_n"],
        "block_k": l1["block_k"],
        "threads": 512,
        # The fused allocation includes both epilogues. Four stages keeps its
        # shared-memory footprint below the SM90 per-CTA limit.
        "pipeline_stages": min(max(l1["pipeline_stages"], l2["pipeline_stages"]), 4),
        "num_experts_per_wave": wave_size,
    }


def fused_single_kernel(
    num_tokens: int,
    hidden: int,
    intermediate_hidden: int,
    num_experts: int,
    num_topk: int,
    num_ranks: int,
    capacity: int,
    num_sms: int,
    activation_clamp: float = 10.0,
    block_m: int = 64,
    block_n: int = 256,
    block_k: int = 128,
    threads: int = 512,
    pipeline_stages: int = 3,
    num_experts_per_wave: int | None = None,
):
    num_experts_per_rank = num_experts // num_ranks
    num_experts_per_wave = num_experts_per_wave or num_experts_per_rank
    assert num_experts_per_rank % num_experts_per_wave == 0
    assert block_m == 64 and block_n == 256 and block_k == 128
    assert threads == 512 and pipeline_stages <= 4
    assert hidden % block_n == 0 and intermediate_hidden % block_k == 0

    l1_n = 2 * intermediate_hidden
    num_scale_groups = hidden // SCALE_GRANULARITY
    num_routes = num_tokens * num_topk
    num_m_blocks = T.ceildiv(capacity, block_m)
    l1_num_n_blocks = l1_n // block_n
    l2_num_n_blocks = hidden // block_n
    l1_num_k_blocks = hidden // block_k
    l2_num_k_blocks = intermediate_hidden // block_k
    num_expert_waves = num_experts_per_rank // num_experts_per_wave
    # Task cursors flatten (expert, M block, N block) so L1 and L2 can advance
    # independently once their block-level readiness checks are enabled.
    l1_total_tasks = num_experts_per_rank * num_m_blocks * l1_num_n_blocks
    l2_total_tasks = num_experts_per_rank * num_m_blocks * l2_num_n_blocks
    l1_total_rounds = T.ceildiv(l1_total_tasks, num_sms)
    l2_total_rounds = T.ceildiv(l2_total_tasks, num_sms)

    num_output_scale_groups = block_n // (2 * SCALE_GRANULARITY)
    num_l1_scale_groups = l1_n // (2 * SCALE_GRANULARITY)
    combine_block_n = 128
    num_combine_n_blocks = hidden // combine_block_n
    # Eight chunks amortize each wave wait while keeping enough tasks to fill
    # the four combine warps on all SMs.
    combine_n_blocks_per_task = 8
    num_combine_groups = T.ceildiv(
        num_combine_n_blocks, combine_n_blocks_per_task
    )
    combine_values_per_lane = combine_block_n // 32

    warp_size = 32
    dispatch_threads = 64
    producer_begin = dispatch_threads
    producer_threads = 64
    producer_end = producer_begin + producer_threads
    math_begin = producer_end
    num_math_threads = 256
    math_end = math_begin + num_math_threads
    # Keep the validated producer/math warp IDs unchanged and append combine.
    combine_begin = math_end
    combine_threads = 128
    combine_end = combine_begin + combine_threads
    num_combine_warps = combine_threads // warp_size
    math_warps = num_math_threads // warp_size
    rows_per_math_warp = block_m // math_warps
    route_threads = 256

    @T.prim_func
    def main(
        x: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        x_sf: T.Tensor((num_tokens, num_scale_groups), T.float32),
        topk_idx: T.Tensor((num_tokens, num_topk), T.int32),
        topk_weights: T.Tensor((num_tokens, num_topk), T.float32),
        route_counts: T.Tensor((num_ranks, num_experts), T.int32),
        recv_counts: T.Tensor((num_experts_per_rank,), T.int32),
        route_slots: T.Tensor((num_tokens, num_topk), T.int32),
        dispatch_arrivals: T.Tensor(
            (num_experts_per_rank, T.ceildiv(capacity, block_m)), T.uint32
        ),
        l2_arrivals: T.Tensor(
            (num_experts_per_rank, T.ceildiv(capacity, block_m)), T.uint32
        ),
        l2_task_ready: T.Tensor(
            (num_experts_per_rank, T.ceildiv(capacity, block_m), l2_num_n_blocks), T.uint32
        ),
        recv_x: T.Tensor(
            (num_experts_per_rank, capacity, hidden), T.float8_e4m3fn
        ),
        recv_x_sf: T.Tensor(
            (num_experts_per_rank, capacity, num_scale_groups), T.float32
        ),
        recv_weights: T.Tensor((num_experts_per_rank, capacity), T.float32),
        src_ranks: T.Tensor((num_experts_per_rank, capacity), T.int32),
        src_tokens: T.Tensor((num_experts_per_rank, capacity), T.int32),
        src_topk: T.Tensor((num_experts_per_rank, capacity), T.int32),
        l1_weight: T.Tensor(
            (num_experts_per_rank, 2 * intermediate_hidden, hidden),
            T.float8_e4m3fn,
        ),
        l1_weight_sf: T.Tensor(
            (
                num_experts_per_rank,
                2 * intermediate_hidden // SCALE_GRANULARITY,
                hidden // SCALE_GRANULARITY,
            ),
            T.float32,
        ),
        l2_weight: T.Tensor(
            (num_experts_per_rank, hidden, intermediate_hidden),
            T.float8_e4m3fn,
        ),
        l2_weight_sf: T.Tensor(
            (
                num_experts_per_rank,
                hidden // SCALE_GRANULARITY,
                intermediate_hidden // SCALE_GRANULARITY,
            ),
            T.float32,
        ),
        l2_x: T.Tensor(
            (num_experts_per_rank, capacity, intermediate_hidden),
            T.float8_e4m3fn,
        ),
        l2_x_sf: T.Tensor(
            (
                num_experts_per_rank,
                capacity,
                intermediate_hidden // SCALE_GRANULARITY,
            ),
            T.float32,
        ),
        combine: T.Tensor((num_tokens, num_topk, hidden), T.bfloat16),
        barrier: T.Tensor((num_ranks,), T.int32),
        out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        T.annotate_pass_configs({tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
        with T.Kernel(num_sms, threads=threads) as bid:
            tx = T.get_thread_binding()
            src_rank = T.alloc_local((1,), T.int32)
            src_rank[0] = T.get_rank()

            a_shared = T.alloc_shared(
                (pipeline_stages, block_m, block_k), T.float8_e4m3fn
            )
            b_shared = T.alloc_shared(
                (pipeline_stages, block_n, block_k), T.float8_e4m3fn
            )
            l2_a_sf_shared = T.alloc_shared(
                (pipeline_stages, block_m), T.float32
            )
            l1_out_shared = T.alloc_shared(
                (block_m, block_n // 2), T.float8_e4m3fn
            )
            l2_out_shared = T.alloc_shared((block_m, block_n), T.bfloat16)
            stage_barriers = T.alloc_barrier(
                [producer_threads] * pipeline_stages
                + [num_math_threads] * pipeline_stages
            )

            if bid == 0:
                if tx < route_threads:
                    for reset_wave in T.serial(T.ceildiv(num_experts, route_threads)):
                        reset_expert = tx + reset_wave * route_threads
                        if reset_expert < num_experts:
                            route_counts[src_rank[0], reset_expert] = 0
                    T.sync_threads(7, route_threads)

                    for assign_wave in T.serial(T.ceildiv(num_routes, route_threads)):
                        assign_route = tx + assign_wave * route_threads
                        if assign_route < num_routes:
                            assign_token = assign_route // num_topk
                            assign_topk = assign_route % num_topk
                            assign_expert = topk_idx[assign_token, assign_topk]
                            if assign_expert >= 0 and assign_expert < num_experts:
                                route_slots[assign_token, assign_topk] = T.atomic_add(
                                    route_counts[src_rank[0], assign_expert],
                                    1,
                                    memory_order="relaxed",
                                    return_prev=True,
                                )
                            else:
                                route_slots[assign_token, assign_topk] = -1
                    T.sync_threads(7, route_threads)

                    for publish_wave in T.serial(
                        T.ceildiv(num_experts * num_ranks, route_threads)
                    ):
                        publish_idx = tx + publish_wave * route_threads
                        if publish_idx < num_experts * num_ranks:
                            publish_rank = publish_idx // num_experts
                            publish_expert = publish_idx % num_experts
                            if publish_rank != src_rank[0]:
                                T.st(
                                    route_counts[src_rank[0], publish_expert],
                                    route_counts[src_rank[0], publish_expert],
                                    dst_pe=publish_rank,
                                )

                T.barrier_blocks(barrier[0])

                if tx < route_threads:
                    for count_wave in T.serial(
                        T.ceildiv(num_experts_per_rank, route_threads)
                    ):
                        local_expert = tx + count_wave * route_threads
                        if local_expert < num_experts_per_rank:
                            recv_count = T.alloc_var(T.int32, init=0)
                            recv_expert = (
                                src_rank[0] * num_experts_per_rank + local_expert
                            )
                            for count_rank in T.serial(num_ranks):
                                recv_count += route_counts[count_rank, recv_expert]
                            recv_counts[local_expert] = recv_count

                    for prefix_wave in T.serial(T.ceildiv(num_routes, route_threads)):
                        prefix_route = tx + prefix_wave * route_threads
                        if prefix_route < num_routes:
                            prefix_token = prefix_route // num_topk
                            prefix_topk = prefix_route % num_topk
                            prefix_expert = topk_idx[prefix_token, prefix_topk]
                            prefix_slot = T.alloc_var(
                                T.int32,
                                init=route_slots[prefix_token, prefix_topk],
                            )
                            if (
                                prefix_expert >= 0
                                and prefix_expert < num_experts
                                and prefix_slot >= 0
                            ):
                                for prefix_rank in T.serial(num_ranks):
                                    if prefix_rank < src_rank[0]:
                                        prefix_slot += route_counts[
                                            prefix_rank, prefix_expert
                                        ]
                                route_slots[prefix_token, prefix_topk] = prefix_slot

            T.sync_grid()

            if tx < math_begin or tx >= math_end:
                T.dec_max_nreg(48)
            else:
                T.inc_max_nreg(208)

            dispatch_warp = tx // warp_size
            dispatch_lane = tx % warp_size
            if tx < dispatch_threads:
                for metadata_wave in T.serial(
                    T.ceildiv(num_routes, num_sms * dispatch_threads)
                ):
                    metadata_route = (
                        bid * dispatch_threads
                        + tx
                        + metadata_wave * num_sms * dispatch_threads
                    )
                    if metadata_route < num_routes:
                        metadata_token = metadata_route // num_topk
                        metadata_topk = metadata_route % num_topk
                        metadata_expert = topk_idx[metadata_token, metadata_topk]
                        metadata_slot = route_slots[metadata_token, metadata_topk]
                        if (
                            metadata_expert >= 0
                            and metadata_slot >= 0
                            and metadata_slot < capacity
                        ):
                            metadata_rank = (
                                metadata_expert // num_experts_per_rank
                            )
                            metadata_local_expert = (
                                metadata_expert % num_experts_per_rank
                            )
                            T.st(
                                recv_weights[
                                    metadata_local_expert, metadata_slot
                                ],
                                topk_weights[metadata_token, metadata_topk],
                                dst_pe=metadata_rank,
                            )
                            T.st(
                                src_tokens[metadata_local_expert, metadata_slot],
                                metadata_token,
                                dst_pe=metadata_rank,
                            )
                            T.st(
                                src_topk[metadata_local_expert, metadata_slot],
                                metadata_topk,
                                dst_pe=metadata_rank,
                            )
                            T.st(
                                src_ranks[metadata_local_expert, metadata_slot],
                                src_rank[0],
                                scope="sys",
                                sem="release",
                                dst_pe=metadata_rank,
                            )

                for pull_wave in T.serial(
                    T.ceildiv(
                        num_experts_per_rank * capacity,
                        num_sms * (dispatch_threads // warp_size),
                    )
                ):
                    pull_idx = (
                        bid * (dispatch_threads // warp_size)
                        + dispatch_warp
                        + pull_wave
                        * num_sms
                        * (dispatch_threads // warp_size)
                    )
                    pull_expert = pull_idx // capacity
                    pull_slot = pull_idx % capacity
                    if (
                        pull_expert < num_experts_per_rank
                        and pull_slot < recv_counts[pull_expert]
                    ):
                        if dispatch_lane == 0:
                            T.wait_ge(
                                src_ranks[pull_expert, pull_slot],
                                0,
                                scope=T.WaitScope.SYS,
                                semantics=T.WaitSemantics.ACQUIRE,
                            )
                        T.sync_warp()
                        pull_rank = src_ranks[pull_expert, pull_slot]
                        pull_token = src_tokens[pull_expert, pull_slot]
                        T.get_warp(
                            T.address_of(x[pull_token, 0]),
                            T.address_of(recv_x[pull_expert, pull_slot, 0]),
                            hidden,
                            src_pe=pull_rank,
                            unroll_factor=8,
                        )
                        T.get_warp(
                            T.address_of(x_sf[pull_token, 0]),
                            T.address_of(recv_x_sf[pull_expert, pull_slot, 0]),
                            num_scale_groups,
                            src_pe=pull_rank,
                            unroll_factor=8,
                        )
                        T.sync_warp()
                        if dispatch_lane == 0:
                            T.atom_add(
                                dispatch_arrivals[
                                    pull_expert, pull_slot // block_m
                                ],
                                1,
                                scope="gpu",
                                sem="release",
                            )

            if tx >= combine_begin and tx < combine_end:
                combine_warp = (tx - combine_begin) // warp_size
                combine_lane = (tx - combine_begin) % warp_size
                combine_accum = T.alloc_local(
                    (
                        combine_n_blocks_per_task,
                        combine_values_per_lane,
                    ),
                    T.float32,
                )
                num_combine_tasks = num_tokens * num_combine_groups
                for combine_round in T.serial(
                    T.ceildiv(
                        num_combine_tasks,
                        num_sms * num_combine_warps,
                    )
                ):
                    combine_task = (
                        bid * num_combine_warps
                        + combine_warp
                        + combine_round * num_sms * num_combine_warps
                    )
                    if combine_task < num_combine_tasks:
                        combine_token = combine_task // num_combine_groups
                        combine_group = combine_task % num_combine_groups
                        for group_block in T.unroll(
                            combine_n_blocks_per_task
                        ):
                            for value_idx in T.unroll(
                                combine_values_per_lane
                            ):
                                combine_accum[group_block, value_idx] = 0.0

                        for expert_wave in T.serial(num_expert_waves):
                            wave_begin = expert_wave * num_experts_per_wave
                            wave_end = wave_begin + num_experts_per_wave
                            for topk_slot in T.serial(num_topk):
                                route_expert = topk_idx[
                                    combine_token, topk_slot
                                ]
                                route_slot = route_slots[
                                    combine_token, topk_slot
                                ]
                                route_local_expert = (
                                    route_expert % num_experts_per_rank
                                )
                                route_rank = (
                                    route_expert // num_experts_per_rank
                                )
                                if (
                                    route_expert >= 0
                                    and route_slot >= 0
                                    and route_slot < capacity
                                    and route_local_expert >= wave_begin
                                    and route_local_expert < wave_end
                                ):
                                    for group_block in T.unroll(
                                        combine_n_blocks_per_task
                                    ):
                                        combine_n_block = (
                                            combine_group
                                            * combine_n_blocks_per_task
                                            + group_block
                                        )
                                        if combine_n_block < num_combine_n_blocks:
                                            if combine_lane == 0:
                                                T.wait_ge(
                                                    l2_task_ready[
                                                        route_local_expert,
                                                        route_slot // block_m,
                                                        combine_n_block
                                                        // (block_n // combine_block_n),
                                                    ],
                                                    1,
                                                    scope=T.WaitScope.SYS,
                                                    semantics=T.WaitSemantics.ACQUIRE,
                                                )
                                            T.sync_warp()
                                            for value_idx in T.unroll(
                                                combine_values_per_lane
                                            ):
                                                combine_col = (
                                                    combine_n_block
                                                    * combine_block_n
                                                    + value_idx * warp_size
                                                    + combine_lane
                                                )
                                                combine_accum[
                                                    group_block, value_idx
                                                ] += T.cast(
                                                    combine[
                                                        combine_token,
                                                        topk_slot,
                                                        combine_col,
                                                    ],
                                                    T.float32,
                                                )

                        for group_block in T.unroll(
                            combine_n_blocks_per_task
                        ):
                            combine_n_block = (
                                combine_group * combine_n_blocks_per_task
                                + group_block
                            )
                            if combine_n_block < num_combine_n_blocks:
                                for value_idx in T.unroll(
                                    combine_values_per_lane
                                ):
                                    combine_col = (
                                        combine_n_block * combine_block_n
                                        + value_idx * warp_size
                                        + combine_lane
                                    )
                                    out[combine_token, combine_col] = T.cast(
                                        combine_accum[
                                            group_block, value_idx
                                        ],
                                        T.bfloat16,
                                    )

            if tx >= producer_begin and tx < producer_end:
                producer_step = T.alloc_var(T.int32, init=0)
                l1_task = T.alloc_var(T.int32, init=bid)
                l2_task = T.alloc_var(T.int32, init=bid)

                # SM90 keeps one producer warpgroup and advances each phase's
                # flattened task cursor independently. The L2 cursor waits on
                # the per-expert/M-block L1 readiness counter below.
                for l1_round in T.serial(l1_total_rounds):
                    if l1_task < l1_total_tasks:
                        l1_task_offset = T.alloc_var(T.int32, init=l1_task)
                        l1_expert = l1_task_offset // (num_m_blocks * l1_num_n_blocks)
                        l1_task_offset -= l1_expert * num_m_blocks * l1_num_n_blocks
                        l1_m_block = l1_task_offset // l1_num_n_blocks
                        l1_n_block = l1_task_offset % l1_num_n_blocks
                        l1_valid_m_blocks = T.ceildiv(
                            T.min(recv_counts[l1_expert], capacity), block_m
                        )
                        if l1_m_block < l1_valid_m_blocks:
                            valid_rows = T.min(
                                block_m,
                                recv_counts[l1_expert] - l1_m_block * block_m,
                            )
                            if tx == producer_begin:
                                T.wait_ge(
                                    dispatch_arrivals[l1_expert, l1_m_block],
                                    valid_rows,
                                    scope=T.WaitScope.GPU,
                                    semantics=T.WaitSemantics.ACQUIRE,
                                )
                            T.sync_threads(5, producer_threads)
                            for k_block in T.serial(l1_num_k_blocks):
                                stage = (producer_step + k_block) % pipeline_stages
                                phase = ((producer_step + k_block) // pipeline_stages) & 1
                                T.mbarrier_wait_parity(
                                    stage_barriers[pipeline_stages + stage], phase ^ 1
                                )
                                T.tma_copy(
                                    recv_x[
                                        l1_expert,
                                        l1_m_block * block_m : (l1_m_block + 1) * block_m,
                                        k_block * block_k : (k_block + 1) * block_k,
                                    ],
                                    a_shared[stage, :, :],
                                    barrier=stage_barriers[stage],
                                )
                                T.tma_copy(
                                    l1_weight[
                                        l1_expert,
                                        l1_n_block * block_n : (l1_n_block + 1) * block_n,
                                        k_block * block_k : (k_block + 1) * block_k,
                                    ],
                                    b_shared[stage, :, :],
                                    barrier=stage_barriers[stage],
                                )
                                T.mbarrier_arrive(stage_barriers[stage])
                            producer_step += l1_num_k_blocks
                        l1_task += num_sms

                for l2_round in T.serial(l2_total_rounds):
                    if l2_task < l2_total_tasks:
                        l2_task_offset = T.alloc_var(T.int32, init=l2_task)
                        l2_expert = l2_task_offset // (num_m_blocks * l2_num_n_blocks)
                        l2_task_offset -= l2_expert * num_m_blocks * l2_num_n_blocks
                        l2_m_block = l2_task_offset // l2_num_n_blocks
                        l2_n_block = l2_task_offset % l2_num_n_blocks
                        l2_valid_m_blocks = T.ceildiv(
                            T.min(recv_counts[l2_expert], capacity), block_m
                        )
                        if l2_m_block < l2_valid_m_blocks:
                            if tx == producer_begin:
                                T.wait_ge(
                                    l2_arrivals[l2_expert, l2_m_block],
                                    l1_num_n_blocks,
                                    scope=T.WaitScope.GPU,
                                    semantics=T.WaitSemantics.ACQUIRE,
                                )
                            T.sync_threads(6, producer_threads)
                            for k_block in T.serial(l2_num_k_blocks):
                                stage = (producer_step + k_block) % pipeline_stages
                                phase = ((producer_step + k_block) // pipeline_stages) & 1
                                T.mbarrier_wait_parity(
                                    stage_barriers[pipeline_stages + stage], phase ^ 1
                                )
                                T.tma_copy(
                                    l2_x[
                                        l2_expert,
                                        l2_m_block * block_m : (l2_m_block + 1) * block_m,
                                        k_block * block_k : (k_block + 1) * block_k,
                                    ],
                                    a_shared[stage, :, :],
                                    barrier=stage_barriers[stage],
                                )
                                T.tma_copy(
                                    l2_weight[
                                        l2_expert,
                                        l2_n_block * block_n : (l2_n_block + 1) * block_n,
                                        k_block * block_k : (k_block + 1) * block_k,
                                    ],
                                    b_shared[stage, :, :],
                                    barrier=stage_barriers[stage],
                                )
                                l2_a_sf_shared[stage, tx - producer_begin] = l2_x_sf[
                                    l2_expert,
                                    l2_m_block * block_m + tx - producer_begin,
                                    k_block,
                                ]
                                T.mbarrier_arrive(stage_barriers[stage])
                            producer_step += l2_num_k_blocks
                        l2_task += num_sms

            elif tx >= math_begin and tx < math_end:
                partial = T.alloc_fragment((block_m, block_n), T.float32)
                accum = T.alloc_fragment((block_m, block_n), T.bfloat16)
                gate = T.alloc_fragment((block_m, block_n // 2), T.float32)
                gate_grouped = T.reshape(
                    gate,
                    (
                        block_m,
                        num_output_scale_groups,
                        SCALE_GRANULARITY,
                    ),
                )
                up = T.alloc_fragment((block_m, block_n // 2), T.float32)
                amax = T.alloc_fragment(
                    (block_m, num_output_scale_groups), T.float32
                )
                scale = T.alloc_fragment(
                    (block_m, num_output_scale_groups), T.float32
                )
                quant_fp8 = T.alloc_fragment(
                    (block_m, block_n // 2), T.float8_e4m3fn
                )
                act_scale = T.alloc_fragment((block_m,), T.float32)
                weight_scale = T.alloc_local(
                    (2 * num_output_scale_groups,), T.float32
                )
                consumer_step = T.alloc_var(T.int32, init=0)
                l1_task = T.alloc_var(T.int32, init=bid)
                l2_task = T.alloc_var(T.int32, init=bid)
                scatter_dst_rank = T.alloc_var(T.int32, init=0)
                scatter_dst_token = T.alloc_var(T.int32, init=0)
                scatter_dst_topk = T.alloc_var(T.int32, init=0)

                # Keep one surrounding scope while task-level completion is
                # refined to block-level flags.
                for expert_wave in T.serial(1):

                    for l1_round in T.serial(l1_total_rounds):
                        if l1_task < l1_total_tasks:
                            tile_offset = T.alloc_var(T.int32, init=l1_task)
                            expert = tile_offset // (num_m_blocks * l1_num_n_blocks)
                            tile_offset -= expert * num_m_blocks * l1_num_n_blocks
                            m_block = tile_offset // l1_num_n_blocks
                            n_block = tile_offset % l1_num_n_blocks
                            l1_valid_m_blocks = T.ceildiv(
                                T.min(recv_counts[expert], capacity), block_m
                            )
                            if m_block < l1_valid_m_blocks and (
                                expert >= 0
                                and l2_num_k_blocks * block_k
                                == intermediate_hidden
                            ):
                                T.clear(partial)
                                T.clear(accum)
                                for k_block in T.serial(l1_num_k_blocks):
                                    stage = (consumer_step + k_block) % pipeline_stages
                                    phase = (
                                        (consumer_step + k_block)
                                        // pipeline_stages
                                    ) & 1
                                    T.mbarrier_wait_parity(
                                        stage_barriers[stage], phase
                                    )
                                    T.gemm(
                                        a_shared[stage, :, :],
                                        b_shared[stage, :, :],
                                        partial,
                                        transpose_B=True,
                                    )
                                    for i in T.Parallel(block_m):
                                        act_scale[i] = recv_x_sf[
                                            expert,
                                            m_block * block_m + i,
                                            k_block,
                                        ]
                                    for scale_group in T.serial(
                                        num_output_scale_groups
                                    ):
                                        weight_scale[2 * scale_group] = (
                                            l1_weight_sf[
                                                expert,
                                                n_block
                                                * num_output_scale_groups
                                                + scale_group,
                                                k_block,
                                            ]
                                        )
                                        weight_scale[2 * scale_group + 1] = (
                                            l1_weight_sf[
                                                expert,
                                                num_l1_scale_groups
                                                + n_block
                                                * num_output_scale_groups
                                                + scale_group,
                                                k_block,
                                            ]
                                        )
                                    for i, j in T.Parallel(block_m, block_n):
                                        accum[i, j] = (
                                            T.cast(partial[i, j], T.bfloat16)
                                            * T.cast(
                                                act_scale[i]
                                                * weight_scale[
                                                    2
                                                    * (
                                                        j
                                                        // (
                                                            2
                                                            * SCALE_GRANULARITY
                                                        )
                                                    )
                                                    + (j % 16) // 8
                                                ],
                                                T.bfloat16,
                                            )
                                            + accum[i, j]
                                        )
                                    T.clear(partial)
                                    T.mbarrier_arrive(
                                        stage_barriers[
                                            pipeline_stages + stage
                                        ]
                                    )
                                consumer_step += l1_num_k_blocks

                                for i, j in T.Parallel(
                                    block_m, block_n // 2
                                ):
                                    gate[i, j] = accum[
                                        i, (j // 8) * 16 + j % 8
                                    ]
                                for i, j in T.Parallel(
                                    block_m, block_n // 2
                                ):
                                    up[i, j] = accum[
                                        i, (j // 8) * 16 + j % 8 + 8
                                    ]
                                for i, j in T.Parallel(
                                    block_m, block_n // 2
                                ):
                                    clamped_gate = T.min(
                                        gate[i, j], activation_clamp
                                    )
                                    gate[i, j] = (
                                        clamped_gate
                                        * T.sigmoid(clamped_gate)
                                        * T.max(
                                            T.min(up[i, j], activation_clamp),
                                            -activation_clamp,
                                        )
                                        * recv_weights[
                                            expert, m_block * block_m + i
                                        ]
                                    )
                                T.reduce_absmax(gate_grouped, amax, dim=2)
                                for i, scale_group in T.Parallel(
                                    block_m, num_output_scale_groups
                                ):
                                    scale[i, scale_group] = (
                                        T.max(amax[i, scale_group], 1e-4)
                                        / FP8_MAX
                                    )
                                    l2_x_sf[
                                        expert,
                                        m_block * block_m + i,
                                        n_block
                                        * num_output_scale_groups
                                        + scale_group,
                                    ] = scale[i, scale_group]
                                for i, j in T.Parallel(
                                    block_m, block_n // 2
                                ):
                                    gate[i, j] = T.clamp(
                                        gate[i, j]
                                        / scale[
                                            i, j // SCALE_GRANULARITY
                                        ],
                                        -FP8_MAX,
                                        FP8_MAX,
                                    )
                                T.copy(gate, quant_fp8)
                                T.copy(quant_fp8, l1_out_shared)
                                T.copy(
                                    l1_out_shared,
                                    l2_x[
                                        expert,
                                        m_block
                                        * block_m : (m_block + 1)
                                        * block_m,
                                        n_block
                                        * (block_n // 2) : (n_block + 1)
                                        * (block_n // 2),
                                    ],
                                )
                                if tx == math_begin:
                                    T.atom_add(
                                        l2_arrivals[expert, m_block],
                                        1,
                                        scope="gpu",
                                        sem="release",
                                    )
                        l1_task += num_sms

                    for l2_round in T.serial(l2_total_rounds):
                        if l2_task < l2_total_tasks:
                            tile_offset = T.alloc_var(T.int32, init=l2_task)
                            expert = tile_offset // (num_m_blocks * l2_num_n_blocks)
                            tile_offset -= expert * num_m_blocks * l2_num_n_blocks
                            m_block = tile_offset // l2_num_n_blocks
                            n_block = tile_offset % l2_num_n_blocks
                            l2_valid_m_blocks = T.ceildiv(
                                T.min(recv_counts[expert], capacity), block_m
                            )
                            if m_block < l2_valid_m_blocks and expert >= 0:
                                T.clear(partial)
                                T.clear(accum)
                                for k_block in T.serial(l2_num_k_blocks):
                                    stage = (consumer_step + k_block) % pipeline_stages
                                    phase = (
                                        (consumer_step + k_block)
                                        // pipeline_stages
                                    ) & 1
                                    T.mbarrier_wait_parity(
                                        stage_barriers[stage], phase
                                    )
                                    T.gemm(
                                        a_shared[stage, :, :],
                                        b_shared[stage, :, :],
                                        partial,
                                        transpose_B=True,
                                    )
                                    for i in T.Parallel(block_m):
                                        act_scale[i] = l2_a_sf_shared[
                                            stage, i
                                        ]
                                    weight_scale[0] = l2_weight_sf[
                                        expert, n_block * 2, k_block
                                    ]
                                    weight_scale[1] = l2_weight_sf[
                                        expert, n_block * 2 + 1, k_block
                                    ]
                                    for i, j in T.Parallel(block_m, block_n):
                                        accum[i, j] = (
                                            T.cast(partial[i, j], T.bfloat16)
                                            * T.cast(
                                                act_scale[i]
                                                * weight_scale[j // 128],
                                                T.bfloat16,
                                            )
                                            + accum[i, j]
                                        )
                                    T.clear(partial)
                                    T.mbarrier_arrive(
                                        stage_barriers[
                                            pipeline_stages + stage
                                        ]
                                    )
                                consumer_step += l2_num_k_blocks
                                T.copy(accum, l2_out_shared)

                                scatter_warp = (
                                    tx - math_begin
                                ) // warp_size
                                for row_in_warp in T.serial(rows_per_math_warp):
                                    row = (
                                        scatter_warp * rows_per_math_warp
                                        + row_in_warp
                                    )
                                    pool_row = m_block * block_m + row
                                    if pool_row < recv_counts[expert]:
                                        if tx % warp_size == 0:
                                            scatter_dst_rank = src_ranks[
                                                expert, pool_row
                                            ]
                                            scatter_dst_token = src_tokens[
                                                expert, pool_row
                                            ]
                                            scatter_dst_topk = src_topk[
                                                expert, pool_row
                                            ]
                                        dst_rank = T.shfl_sync(
                                            scatter_dst_rank, 0
                                        )
                                        dst_token = T.shfl_sync(
                                            scatter_dst_token, 0
                                        )
                                        dst_topk = T.shfl_sync(
                                            scatter_dst_topk, 0
                                        )
                                        if (
                                            dst_rank >= 0
                                            and dst_rank < num_ranks
                                            and dst_token >= 0
                                            and dst_token < num_tokens
                                            and dst_topk >= 0
                                            and dst_topk < num_topk
                                        ):
                                            T.put_warp(
                                                T.address_of(
                                                    l2_out_shared[row, 0]
                                                ),
                                                T.address_of(
                                                    combine[
                                                        dst_token,
                                                        dst_topk,
                                                        n_block * block_n,
                                                    ]
                                                ),
                                                block_n,
                                                dst_pe=dst_rank,
                                                unroll_factor=1,
                                            )
                            T.fence_sys()
                            T.sync_threads(4, num_math_threads)
                            if tx == math_begin:
                                for ready_rank in T.serial(num_ranks):
                                    T.st(
                                        l2_task_ready[expert, m_block, n_block],
                                        1,
                                        scope="sys",
                                        sem="release",
                                        dst_pe=ready_rank,
                                    )
                            T.sync_threads(4, num_math_threads)
                        l2_task += num_sms


            T.fence_sys()
            T.sync_grid()

    return main


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    model_name, model = two_kernel.resolve_model_config(args)
    hidden = model["hidden"]
    intermediate_hidden = model["intermediate_hidden"]
    num_experts = model["num_experts"]
    num_topk = model["num_topk"]
    num_tokens = args.num_tokens
    activation_clamp = args.activation_clamp

    assert num_tokens > 0
    assert hidden >= 512 and hidden % 256 == 0
    assert intermediate_hidden > 0 and intermediate_hidden % 128 == 0
    assert num_experts > 0 and num_experts % num_local_ranks == 0
    assert 0 < num_topk <= min(32, num_experts)
    num_experts_per_rank = num_experts // num_local_ranks
    average_recv = (
        num_tokens * num_local_ranks * num_topk + num_experts - 1
    ) // num_experts
    capacity = (
        args.capacity
        if args.capacity is not None
        else (max(average_recv * 2, 64) + 63) // 64 * 64
    )
    assert capacity >= 64 and capacity % 64 == 0

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert rank == local_rank and num_ranks == num_local_ranks
    num_sms = torch.cuda.get_device_properties(local_rank).multi_processor_count
    allocator = get_allocator(
        size=two_kernel._allocator_size_bytes(
            num_tokens,
            hidden,
            intermediate_hidden,
            num_experts_per_rank,
            num_topk,
            capacity,
        ),
        device=f"cuda:{local_rank}",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_local_ranks,
        group=group,
        use_vmm=True,
    )

    shape_family, config = select_single_kernel_config(
        hidden,
        intermediate_hidden,
        num_tokens,
        num_topk,
        num_experts_per_rank,
        num_sms,
    )
    if args.experts_per_wave is not None:
        assert args.experts_per_wave > 0
        assert num_experts_per_rank % args.experts_per_wave == 0
        config["num_experts_per_wave"] = args.experts_per_wave
    if args.pipeline_stages is not None:
        assert 2 <= args.pipeline_stages <= 4
        config["pipeline_stages"] = args.pipeline_stages
    num_expert_waves = (
        num_experts_per_rank // config["num_experts_per_wave"]
    )

    spec = fused_single_kernel(
        num_tokens,
        hidden,
        intermediate_hidden,
        num_experts,
        num_topk,
        num_ranks,
        capacity,
        num_sms,
        activation_clamp=activation_clamp,
        **config,
    )
    kernel = tilelang.compile(spec, compile_once=True, compile_group=group)
    kernel.initialize(allocator=allocator)
    if local_rank == 0 and args.print_source:
        print(kernel.get_kernel_source())

    torch.manual_seed(args.seed + local_rank)
    x_bf16 = torch.randn(
        (num_tokens, hidden), dtype=torch.bfloat16, device="cuda"
    )
    x_fp8_src, x_sf_src = two_kernel.per_token_cast_to_fp8(x_bf16)
    scores = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    )
    topk_weights_src, topk_idx_src = torch.topk(
        scores, num_topk, dim=-1, sorted=False
    )
    topk_idx_src = topk_idx_src.to(torch.int32)

    l1_bf16 = (
        torch.randn(
            (num_experts_per_rank, 2 * intermediate_hidden, hidden),
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )
    l2_bf16 = (
        torch.randn(
            (num_experts_per_rank, hidden, intermediate_hidden),
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )
    l1_fp8_src, l1_sf_src = two_kernel.block_cast_to_fp8(l1_bf16)
    l2_fp8_src, l2_sf_src = two_kernel.block_cast_to_fp8(l2_bf16)
    del scores, l1_bf16, l2_bf16

    tensor = two_kernel.allocator_tensor
    barrier = tensor((num_ranks,), torch.int32, allocator=allocator)
    x = tensor(x_fp8_src.shape, x_fp8_src.dtype, allocator=allocator).copy_(
        x_fp8_src
    )
    x_sf = tensor(x_sf_src.shape, x_sf_src.dtype, allocator=allocator).copy_(
        x_sf_src
    )
    topk_idx = tensor(
        topk_idx_src.shape, topk_idx_src.dtype, allocator=allocator
    ).copy_(topk_idx_src)
    topk_weights = tensor(
        topk_weights_src.shape, topk_weights_src.dtype, allocator=allocator
    ).copy_(topk_weights_src)
    l1_fp8_kernel = two_kernel.interleave_gate_up_weights(l1_fp8_src)
    l1_fp8 = tensor(
        l1_fp8_kernel.shape, l1_fp8_kernel.dtype, allocator=allocator
    ).copy_(l1_fp8_kernel)
    del l1_fp8_kernel
    l1_sf = tensor(l1_sf_src.shape, l1_sf_src.dtype, allocator=allocator).copy_(
        l1_sf_src
    )
    l2_fp8 = tensor(
        l2_fp8_src.shape, l2_fp8_src.dtype, allocator=allocator
    ).copy_(l2_fp8_src)
    l2_sf = tensor(l2_sf_src.shape, l2_sf_src.dtype, allocator=allocator).copy_(
        l2_sf_src
    )

    route_counts = tensor(
        (num_ranks, num_experts), torch.int32, allocator=allocator
    )
    recv_counts = tensor(
        (num_experts_per_rank,), torch.int32, allocator=allocator
    )
    recv_x = tensor(
        (num_experts_per_rank, capacity, hidden),
        torch.float8_e4m3fn,
        allocator=allocator,
    )
    recv_x_sf = tensor(
        (num_experts_per_rank, capacity, hidden // SCALE_GRANULARITY),
        torch.float32,
        allocator=allocator,
    )
    recv_weights = tensor(
        (num_experts_per_rank, capacity), torch.float32, allocator=allocator
    )
    src_ranks = tensor(
        (num_experts_per_rank, capacity), torch.int32, allocator=allocator
    )
    src_tokens = tensor(
        (num_experts_per_rank, capacity), torch.int32, allocator=allocator
    )
    src_topk = tensor(
        (num_experts_per_rank, capacity), torch.int32, allocator=allocator
    )
    route_slots = tensor(
        (num_tokens, num_topk), torch.int32, allocator=allocator
    )
    dispatch_arrivals = tensor(
        (num_experts_per_rank, capacity // 64),
        torch.uint32,
        allocator=allocator,
    )
    l2_arrivals = tensor(
        (num_experts_per_rank, capacity // 64),
        torch.uint32,
        allocator=allocator,
    )
    l2_task_ready = tensor(
        (
            num_experts_per_rank,
            capacity // config["block_m"],
            hidden // config["block_n"],
        ),
        torch.uint32,
        allocator=allocator,
    )
    l2_x = tensor(
        (num_experts_per_rank, capacity, intermediate_hidden),
        torch.float8_e4m3fn,
        allocator=allocator,
    )
    l2_x_sf = tensor(
        (
            num_experts_per_rank,
            capacity,
            intermediate_hidden // SCALE_GRANULARITY,
        ),
        torch.float32,
        allocator=allocator,
    )
    combine = tensor(
        (num_tokens, num_topk, hidden), torch.bfloat16, allocator=allocator
    )
    out = tensor((num_tokens, hidden), torch.bfloat16, allocator=allocator)

    def reset_state():
        route_counts.zero_()
        barrier.zero_()
        recv_counts.zero_()
        dispatch_arrivals.zero_()
        l2_arrivals.zero_()
        l2_task_ready.zero_()
        recv_x.zero_()
        recv_x_sf.zero_()
        recv_weights.zero_()
        src_ranks.fill_(-1)
        combine.zero_()
        torch.cuda.synchronize()
        dist.barrier(group=group)

    def run_pipeline(check_capacity: bool = False):
        kernel(
            x,
            x_sf,
            topk_idx,
            topk_weights,
            route_counts,
            recv_counts,
            route_slots,
            dispatch_arrivals,
            l2_arrivals,
            l2_task_ready,
            recv_x,
            recv_x_sf,
            recv_weights,
            src_ranks,
            src_tokens,
            src_topk,
            l1_fp8,
            l1_sf,
            l2_fp8,
            l2_sf,
            l2_x,
            l2_x_sf,
            combine,
            barrier,
            out,
        )
        if check_capacity:
            local_max = recv_counts.max()
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=group)
            assert local_max.item() <= capacity
        return out

    reset_state()
    actual = run_pipeline(check_capacity=True)
    torch.cuda.synchronize()
    dist.barrier(group=group)

    if args.check:
        expected = two_kernel.torch_reference(
            x_fp8_src,
            x_sf_src,
            topk_idx_src,
            topk_weights_src,
            l1_fp8_src,
            l1_sf_src,
            l2_fp8_src,
            l2_sf_src,
            group,
            activation_clamp,
        )
        diff = two_kernel.calc_diff(actual, expected)
        assert diff < args.diff_tol, (
            f"rank {local_rank}: diff={diff} exceeds {args.diff_tol}"
        )
        print(f"rank {local_rank} check passed, diff={diff:.6f}")

    if args.rep > 0:
        reset_state()
        for _ in range(args.warmup):
            run_pipeline()
            reset_state()
        latency = do_bench(
            run_pipeline,
            warmup=0,
            rep=args.rep,
            post_fn=reset_state,
            group=group,
        )
        if local_rank == 0:
            print(
                "tilescale sm90 fp8 mega moe single kernel: "
                f"model={model_name} family={shape_family} M={num_tokens} "
                f"H={hidden} IH={intermediate_hidden} E={num_experts} "
                f"topk={num_topk} capacity={capacity} "
                f"epw={config['num_experts_per_wave']} "
                f"stages={config['pipeline_stages']} "
                f"latency={latency * 1000:.1f} us"
            )

    allocator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument(
        "--model-config", choices=tuple(MODEL_CONFIGS), default="smoke"
    )
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--intermediate-hidden", type=int, default=None)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--num-topk", type=int, default=None)
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=None)
    parser.add_argument("--experts-per-wave", type=int, default=None)
    parser.add_argument("--pipeline-stages", type=int, default=None)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diff-tol", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-source", action="store_true")
    args = parser.parse_args()
    torch.multiprocessing.spawn(
        main,
        args=(args.num_processes, args),
        nprocs=args.num_processes,
    )
