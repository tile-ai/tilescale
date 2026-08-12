"""Multi-GPU FP8 Mega MoE for SM90 using TileScale distributed primitives.

The Flash configuration uses two persistent kernels: one for routing, dispatch,
L1 GEMM, and SwiGLU quantization, and one for L2 GEMM, scatter, and reduction.
The Pro configuration retains the multi-kernel path until its four-warpgroup L1
kernel is ready.
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

os.environ.setdefault("NCCL_DEBUG", "ERROR")


MODEL_CONFIGS = {
    "smoke": {"hidden": 512, "intermediate_hidden": 512, "num_experts": 8, "num_topk": 2},
    "flash": {"hidden": 4096, "intermediate_hidden": 2048, "num_experts": 256, "num_topk": 6},
    "pro": {"hidden": 7168, "intermediate_hidden": 3072, "num_experts": 384, "num_topk": 6},
}

FP8_MAX = 448.0
SCALE_GRANULARITY = 128


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align_up(x: int, alignment: int) -> int:
    return ceil_div(x, alignment) * alignment


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    x_view = x.float().view(m, k // SCALE_GRANULARITY, SCALE_GRANULARITY)
    amax = x_view.abs().amax(dim=-1).clamp(1e-4)
    scale = amax / FP8_MAX
    x_fp8 = (x_view / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return x_fp8.view(m, k).contiguous(), scale.contiguous()


def block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    groups, n, k = x.shape
    x_view = x.float().view(
        groups,
        n // SCALE_GRANULARITY,
        SCALE_GRANULARITY,
        k // SCALE_GRANULARITY,
        SCALE_GRANULARITY,
    )
    amax = x_view.abs().amax(dim=(-1, -3)).clamp(1e-4)
    scale = amax / FP8_MAX
    x_fp8 = (x_view / scale.unsqueeze(-1).unsqueeze(-3)).to(torch.float8_e4m3fn)
    return x_fp8.view(groups, n, k).contiguous(), scale.contiguous()


def interleave_gate_up_weights(weight: torch.Tensor, granularity: int = 8) -> torch.Tensor:
    groups, n, k = weight.shape
    half = n // 2
    gate = weight[:, :half].view(groups, half // granularity, granularity, k)
    up = weight[:, half:].view(groups, half // granularity, granularity, k)
    return torch.stack((gate, up), dim=2).reshape(groups, n, k).contiguous()


def dequantize_per_token(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    return (x.float().view(m, k // SCALE_GRANULARITY, SCALE_GRANULARITY) * scale.unsqueeze(-1)).view(m, k)


def dequantize_block(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    groups, n, k = x.shape
    x_view = x.float().view(
        groups,
        n // SCALE_GRANULARITY,
        SCALE_GRANULARITY,
        k // SCALE_GRANULARITY,
        SCALE_GRANULARITY,
    )
    return (x_view * scale.unsqueeze(-1).unsqueeze(-3)).view(groups, n, k)


def assign_local_routes_kernel(
    num_tokens: int,
    num_experts: int,
    num_topk: int,
    num_ranks: int,
    threads: int = 128,
):
    @T.prim_func
    def main(
        topk_idx: T.Tensor((num_tokens, num_topk), T.int32),
        route_counts: T.Tensor((num_ranks, num_experts), T.int32),
        route_slots: T.Tensor((num_tokens, num_topk), T.int32),
    ):
        with T.Kernel(T.ceildiv(num_tokens * num_topk, threads), threads=threads) as bx:
            src_rank = T.alloc_local((1,), T.int32)
            src_rank[0] = T.get_rank()
            route_idx = bx * threads + T.get_thread_binding()
            if src_rank[0] < num_ranks and route_idx < num_tokens * num_topk:
                token_idx = route_idx // num_topk
                topk_slot = route_idx % num_topk
                expert_idx = topk_idx[token_idx, topk_slot]
                if expert_idx >= 0 and expert_idx < num_experts:
                    route_slots[token_idx, topk_slot] = T.atomic_add(
                        route_counts[src_rank[0], expert_idx],
                        1,
                        memory_order="relaxed",
                        return_prev=True,
                    )
                else:
                    route_slots[token_idx, topk_slot] = -1

    return main


def reset_route_counts_kernel(num_experts: int, num_ranks: int, threads: int = 128):
    @T.prim_func
    def main(route_counts: T.Tensor((num_ranks, num_experts), T.int32)):
        with T.Kernel(T.ceildiv(num_experts, threads), threads=threads) as bx:
            src_rank = T.alloc_local((1,), T.int32)
            src_rank[0] = T.get_rank()
            expert_idx = bx * threads + T.get_thread_binding()
            if src_rank[0] < num_ranks and expert_idx < num_experts:
                route_counts[src_rank[0], expert_idx] = 0

    return main


def publish_route_counts_kernel(num_experts: int, num_ranks: int, threads: int = 128):
    @T.prim_func
    def main(route_counts: T.Tensor((num_ranks, num_experts), T.int32)):
        with T.Kernel(T.ceildiv(num_experts * num_ranks, threads), threads=threads) as bx:
            src_rank = T.alloc_local((1,), T.int32)
            src_rank[0] = T.get_rank()
            idx = bx * threads + T.get_thread_binding()
            if idx < num_experts * num_ranks:
                dst_rank = idx // num_experts
                expert_idx = idx % num_experts
                if dst_rank != src_rank[0]:
                    T.st(
                        route_counts[src_rank[0], expert_idx],
                        route_counts[src_rank[0], expert_idx],
                        dst_pe=dst_rank,
                    )

    return main


def device_barrier_kernel(num_ranks: int):
    @T.prim_func
    def main(barrier: T.Tensor((num_ranks,), T.int32)):
        with T.Kernel(1, threads=32):
            rank = T.alloc_local((1,), T.int32)
            rank[0] = T.get_rank()
            if rank[0] < num_ranks:
                T.barrier_blocks(barrier[0])
                T.fence_sys()

    return main


def finalize_routes_kernel(
    num_tokens: int,
    num_experts: int,
    num_topk: int,
    num_ranks: int,
    threads: int = 128,
):
    num_experts_per_rank = num_experts // num_ranks
    num_routes = num_tokens * num_topk
    num_work_items = max(num_routes, num_experts_per_rank)

    @T.prim_func
    def main(
        topk_idx: T.Tensor((num_tokens, num_topk), T.int32),
        route_counts: T.Tensor((num_ranks, num_experts), T.int32),
        recv_counts: T.Tensor((num_experts_per_rank,), T.int32),
        route_slots: T.Tensor((num_tokens, num_topk), T.int32),
    ):
        with T.Kernel(T.ceildiv(num_work_items, threads), threads=threads) as bx:
            src_rank = T.alloc_local((1,), T.int32)
            src_rank[0] = T.get_rank()
            work_idx = bx * threads + T.get_thread_binding()

            if work_idx < num_experts_per_rank:
                count = T.alloc_var(T.int32, init=0)
                expert_idx = src_rank[0] * num_experts_per_rank + work_idx
                for peer_rank in T.serial(num_ranks):
                    count += route_counts[peer_rank, expert_idx]
                recv_counts[work_idx] = count

            if work_idx < num_routes:
                token_idx = work_idx // num_topk
                topk_slot = work_idx % num_topk
                expert_idx = topk_idx[token_idx, topk_slot]
                slot = T.alloc_var(T.int32, init=route_slots[token_idx, topk_slot])
                if token_idx < num_tokens and expert_idx >= 0 and expert_idx < num_experts and slot >= 0:
                    for peer_rank in T.serial(num_ranks):
                        if peer_rank < src_rank[0]:
                            slot += route_counts[peer_rank, expert_idx]
                    route_slots[token_idx, topk_slot] = slot

    return main


def dispatch_tokens_kernel(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    num_ranks: int,
    capacity: int,
    block_h: int = 256,
    threads: int = 128,
):
    num_experts_per_rank = num_experts // num_ranks
    num_scale_groups = hidden // SCALE_GRANULARITY

    @T.prim_func
    def main(
        x: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        x_sf: T.Tensor((num_tokens, num_scale_groups), T.float32),
        topk_idx: T.Tensor((num_tokens, num_topk), T.int32),
        topk_weights: T.Tensor((num_tokens, num_topk), T.float32),
        route_slots: T.Tensor((num_tokens, num_topk), T.int32),
        recv_x: T.Tensor((num_experts_per_rank, capacity, hidden), T.float8_e4m3fn),
        recv_x_sf: T.Tensor(
            (num_experts_per_rank, capacity, num_scale_groups),
            T.float32,
        ),
        recv_weights: T.Tensor((num_experts_per_rank, capacity), T.float32),
        src_ranks: T.Tensor((num_experts_per_rank, capacity), T.int32),
        src_tokens: T.Tensor((num_experts_per_rank, capacity), T.int32),
        src_topk: T.Tensor((num_experts_per_rank, capacity), T.int32),
    ):
        with T.Kernel(T.ceildiv(hidden, block_h), num_tokens * num_topk, threads=threads) as (bx, by):
            src_rank = T.alloc_local((1,), T.int32)
            src_rank[0] = T.get_rank()
            tx = T.get_thread_binding()
            token_idx = by // num_topk
            topk_slot = by % num_topk
            expert_idx = topk_idx[token_idx, topk_slot]
            slot = route_slots[token_idx, topk_slot]
            if expert_idx >= 0 and slot >= 0 and slot < capacity:
                dst_rank = expert_idx // num_experts_per_rank
                local_expert = expert_idx % num_experts_per_rank
                T.copy(
                    x[token_idx, bx * block_h : (bx + 1) * block_h],
                    recv_x[local_expert, slot, bx * block_h : (bx + 1) * block_h],
                    dst_pe=dst_rank,
                    disable_tma=True,
                )
                if bx == 0 and num_scale_groups > 0:
                    T.copy(
                        x_sf[token_idx, :],
                        recv_x_sf[local_expert, slot, :],
                        dst_pe=dst_rank,
                        disable_tma=True,
                    )
                    if tx == 0 and src_rank[0] < num_ranks:
                        T.st(recv_weights[local_expert, slot], topk_weights[token_idx, topk_slot], dst_pe=dst_rank)
                        T.st(src_ranks[local_expert, slot], src_rank[0], dst_pe=dst_rank)
                        T.st(src_tokens[local_expert, slot], token_idx, dst_pe=dst_rank)
                        T.st(src_topk[local_expert, slot], topk_slot, dst_pe=dst_rank)

    return main


def fused_l1_swiglu_manual_warp_kernel(
    num_tokens: int,
    hidden: int,
    l1_n: int,
    num_experts: int,
    num_topk: int,
    num_ranks: int,
    capacity: int,
    num_sms: int,
    activation_clamp: float = 10.0,
    block_m: int = 64,
    block_n: int = 256,
    block_k: int = 128,
    threads: int = 384,
    pipeline_stages: int = 5,
):
    num_experts_per_rank = num_experts // num_ranks
    num_scale_groups = hidden // SCALE_GRANULARITY
    num_routes = num_tokens * num_topk
    num_m_blocks = ceil_div(capacity, block_m)
    num_n_blocks = ceil_div(l1_n, block_n)
    num_compute_tiles = num_experts_per_rank * num_m_blocks * num_n_blocks
    num_k_blocks = hidden // block_k
    dispatch_thread = 0
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
        arrivals: T.Tensor((num_experts_per_rank, num_m_blocks), T.uint32),
        recv_x: T.Tensor((num_experts_per_rank, capacity, hidden), T.float8_e4m3fn),
        recv_x_sf: T.Tensor((num_experts_per_rank, capacity, num_scale_groups), T.float32),
        recv_weights: T.Tensor((num_experts_per_rank, capacity), T.float32),
        src_ranks: T.Tensor((num_experts_per_rank, capacity), T.int32),
        src_tokens: T.Tensor((num_experts_per_rank, capacity), T.int32),
        src_topk: T.Tensor((num_experts_per_rank, capacity), T.int32),
        l1_weight: T.Tensor((num_experts_per_rank, l1_n, hidden), T.float8_e4m3fn),
        l1_weight_sf: T.Tensor(
            (num_experts_per_rank, l1_n // SCALE_GRANULARITY, hidden // SCALE_GRANULARITY),
            T.float32,
        ),
        l2_x: T.Tensor((num_experts_per_rank, capacity, l1_n // 2), T.float8_e4m3fn),
        l2_x_sf: T.Tensor(
            (num_experts_per_rank, capacity, l1_n // (2 * SCALE_GRANULARITY)),
            T.float32,
        ),
        barrier: T.Tensor((num_ranks,), T.int32),
    ):
        T.annotate_pass_configs({tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
        with T.Kernel(num_sms, threads=threads) as bid:
            tx = T.get_thread_binding()
            src_rank = T.alloc_local((1,), T.int32)
            src_rank[0] = T.get_rank()

            a_shared = T.alloc_shared(
                (pipeline_stages, block_m, block_k),
                T.float8_e4m3fn,
            )
            b_shared = T.alloc_shared(
                (pipeline_stages, block_n, block_k),
                T.float8_e4m3fn,
            )
            out_shared = T.alloc_shared((block_m, block_n // 2), T.float8_e4m3fn)
            stage_barriers = T.alloc_barrier([64] * pipeline_stages + [256] * pipeline_stages)

            if bid == 0:
                if tx < route_threads:
                    for reset_wave in T.serial(ceil_div(num_experts, route_threads)):
                        reset_expert = tx + reset_wave * route_threads
                        if reset_expert < num_experts:
                            route_counts[src_rank[0], reset_expert] = 0
                    T.sync_threads(7, route_threads)

                    for assign_wave in T.serial(ceil_div(num_routes, route_threads)):
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

                    for publish_wave in T.serial(ceil_div(num_experts * num_ranks, route_threads)):
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
                    if tx < num_experts_per_rank:
                        recv_count = T.alloc_var(T.int32, init=0)
                        recv_expert = src_rank[0] * num_experts_per_rank + tx
                        for count_rank in T.serial(num_ranks):
                            recv_count += route_counts[count_rank, recv_expert]
                        recv_counts[tx] = recv_count

                    for prefix_wave in T.serial(ceil_div(num_routes, route_threads)):
                        prefix_route = tx + prefix_wave * route_threads
                        if prefix_route < num_routes:
                            prefix_token = prefix_route // num_topk
                            prefix_topk = prefix_route % num_topk
                            prefix_expert = topk_idx[prefix_token, prefix_topk]
                            prefix_slot = T.alloc_var(
                                T.int32,
                                init=route_slots[prefix_token, prefix_topk],
                            )
                            if prefix_token < num_tokens and prefix_expert >= 0 and prefix_expert < num_experts and prefix_slot >= 0:
                                for prefix_rank in T.serial(num_ranks):
                                    if prefix_rank < src_rank[0]:
                                        prefix_slot += route_counts[prefix_rank, prefix_expert]
                                route_slots[prefix_token, prefix_topk] = prefix_slot

            T.sync_grid()

            if tx < 128:
                T.dec_max_nreg(48)
            else:
                T.inc_max_nreg(208)

            dispatch_warp = tx // 32
            dispatch_lane = tx % 32
            if tx < 64:
                for metadata_wave in T.serial(ceil_div(num_routes, num_sms * 64)):
                    metadata_route = bid * 64 + tx + metadata_wave * num_sms * 64
                    if metadata_route < num_routes:
                        metadata_token = metadata_route // num_topk
                        metadata_topk = metadata_route % num_topk
                        metadata_expert = topk_idx[metadata_token, metadata_topk]
                        metadata_slot = route_slots[metadata_token, metadata_topk]
                        if metadata_expert >= 0 and metadata_slot >= 0 and metadata_slot < capacity:
                            metadata_rank = metadata_expert // num_experts_per_rank
                            metadata_local_expert = metadata_expert % num_experts_per_rank
                            T.st(
                                recv_weights[metadata_local_expert, metadata_slot],
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

            if tx < 64:
                for pull_wave in T.serial(ceil_div(num_experts_per_rank * capacity, num_sms * 2)):
                    pull_idx = bid * 2 + dispatch_warp + pull_wave * num_sms * 2
                    pull_expert = pull_idx // capacity
                    pull_slot = pull_idx % capacity
                    if pull_expert < num_experts_per_rank and pull_slot < recv_counts[pull_expert]:
                        if dispatch_lane == dispatch_thread:
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
                        if dispatch_lane == dispatch_thread:
                            T.atom_add(
                                arrivals[pull_expert, pull_slot // block_m],
                                1,
                                scope="gpu",
                                sem="release",
                            )

            if tx >= 64 and tx < 128:
                producer_step = T.alloc_var(T.int32, init=0)
                for producer_wave in T.serial(ceil_div(num_compute_tiles, num_sms)):
                    producer_tile = bid + producer_wave * num_sms
                    if producer_tile < num_compute_tiles:
                        producer_n = producer_tile % num_n_blocks
                        producer_m = (producer_tile // num_n_blocks) % num_m_blocks
                        producer_expert = producer_tile // (num_n_blocks * num_m_blocks)
                        if producer_n * block_n < l1_n and producer_m * block_m < recv_counts[producer_expert]:
                            producer_arrivals = T.min(
                                block_m,
                                recv_counts[producer_expert] - producer_m * block_m,
                            )
                            if tx == 64:
                                T.wait_ge(
                                    arrivals[producer_expert, producer_m],
                                    producer_arrivals,
                                    scope=T.WaitScope.GPU,
                                    semantics=T.WaitSemantics.ACQUIRE,
                                )
                            T.sync_threads(5, 64)
                            for producer_k in T.serial(num_k_blocks):
                                producer_stage = (producer_step + producer_k) % pipeline_stages
                                producer_phase = ((producer_step + producer_k) // pipeline_stages) & 1
                                T.mbarrier_wait_parity(
                                    stage_barriers[pipeline_stages + producer_stage],
                                    producer_phase ^ 1,
                                )
                                T.tma_copy(
                                    recv_x[
                                        producer_expert,
                                        producer_m * block_m : (producer_m + 1) * block_m,
                                        producer_k * block_k : (producer_k + 1) * block_k,
                                    ],
                                    a_shared[producer_stage, :, :],
                                    barrier=stage_barriers[producer_stage],
                                )
                                T.tma_copy(
                                    l1_weight[
                                        producer_expert,
                                        producer_n * block_n : (producer_n + 1) * block_n,
                                        producer_k * block_k : (producer_k + 1) * block_k,
                                    ],
                                    b_shared[producer_stage, :, :],
                                    barrier=stage_barriers[producer_stage],
                                )
                                T.mbarrier_arrive(stage_barriers[producer_stage])
                            producer_step += num_k_blocks

            elif tx >= 128:
                partial = T.alloc_fragment((block_m, block_n), T.float32)
                accum = T.alloc_fragment((block_m, block_n), T.bfloat16)
                gate = T.alloc_fragment((block_m, block_n // 2), T.float32)
                up = T.alloc_fragment((block_m, block_n // 2), T.float32)
                amax = T.alloc_fragment((block_m,), T.float32)
                scale = T.alloc_fragment((block_m,), T.float32)
                quant_fp8 = T.alloc_fragment((block_m, block_n // 2), T.float8_e4m3fn)
                act_scale = T.alloc_fragment((block_m,), T.float32)
                weight_scale = T.alloc_local((2,), T.float32)
                consumer_step = T.alloc_var(T.int32, init=0)

                for consumer_wave in T.serial(ceil_div(num_compute_tiles, num_sms)):
                    consumer_tile = bid + consumer_wave * num_sms
                    if consumer_tile < num_compute_tiles:
                        consumer_n = consumer_tile % num_n_blocks
                        consumer_m = (consumer_tile // num_n_blocks) % num_m_blocks
                        consumer_expert = consumer_tile // (num_n_blocks * num_m_blocks)
                        if consumer_n * block_n < l1_n and consumer_m * block_m < recv_counts[consumer_expert]:
                            T.clear(partial)
                            T.clear(accum)
                            for consumer_k in T.serial(num_k_blocks):
                                consumer_stage = (consumer_step + consumer_k) % pipeline_stages
                                consumer_phase = ((consumer_step + consumer_k) // pipeline_stages) & 1
                                T.mbarrier_wait_parity(
                                    stage_barriers[consumer_stage],
                                    consumer_phase,
                                )
                                T.gemm(
                                    a_shared[consumer_stage, :, :],
                                    b_shared[consumer_stage, :, :],
                                    partial,
                                    transpose_B=True,
                                )
                                for i in T.Parallel(block_m):
                                    act_scale[i] = recv_x_sf[
                                        consumer_expert,
                                        consumer_m * block_m + i,
                                        consumer_k,
                                    ]
                                weight_scale[0] = l1_weight_sf[consumer_expert, consumer_n, consumer_k]
                                weight_scale[1] = l1_weight_sf[
                                    consumer_expert,
                                    consumer_n + num_n_blocks,
                                    consumer_k,
                                ]
                                for i, j in T.Parallel(block_m, block_n):
                                    accum[i, j] = (
                                        T.cast(partial[i, j], T.bfloat16)
                                        * T.cast(
                                            act_scale[i] * weight_scale[(j % 16) // 8],
                                            T.bfloat16,
                                        )
                                        + accum[i, j]
                                    )
                                T.clear(partial)
                                T.mbarrier_arrive(stage_barriers[pipeline_stages + consumer_stage])
                            consumer_step += num_k_blocks
                            for i, j in T.Parallel(block_m, block_n // 2):
                                gate[i, j] = accum[i, (j // 8) * 16 + j % 8]
                            for i, j in T.Parallel(block_m, block_n // 2):
                                up[i, j] = accum[i, (j // 8) * 16 + j % 8 + 8]
                            for i, j in T.Parallel(block_m, block_n // 2):
                                gate[i, j] = (
                                    T.min(gate[i, j], activation_clamp)
                                    * T.sigmoid(T.min(gate[i, j], activation_clamp))
                                    * T.max(
                                        T.min(up[i, j], activation_clamp),
                                        -activation_clamp,
                                    )
                                    * recv_weights[
                                        consumer_expert,
                                        consumer_m * block_m + i,
                                    ]
                                )
                            T.reduce_absmax(gate, amax, dim=1)
                            for i in T.Parallel(block_m):
                                scale[i] = T.max(amax[i], 1e-4) / FP8_MAX
                                l2_x_sf[
                                    consumer_expert,
                                    consumer_m * block_m + i,
                                    consumer_n,
                                ] = scale[i]
                            for i, j in T.Parallel(block_m, block_n // 2):
                                gate[i, j] = T.clamp(
                                    gate[i, j] / scale[i],
                                    -FP8_MAX,
                                    FP8_MAX,
                                )
                            T.copy(gate, quant_fp8)
                            T.copy(quant_fp8, out_shared)
                            T.copy(
                                out_shared,
                                l2_x[
                                    consumer_expert,
                                    consumer_m * block_m,
                                    consumer_n * (block_n // 2),
                                ],
                            )

    return main


def fp8_grouped_gemm_kernel(
    num_experts_per_rank: int,
    capacity: int,
    n: int,
    k: int,
    block_m: int = 64,
    block_n: int = 128,
    block_k: int = 128,
    threads: int = 128,
    pipeline_stages: int = 4,
):
    @T.prim_func
    def main(
        a: T.Tensor((num_experts_per_rank, capacity, k), T.float8_e4m3fn),
        b: T.Tensor((num_experts_per_rank, n, k), T.float8_e4m3fn),
        a_sf: T.Tensor((num_experts_per_rank, capacity, k // SCALE_GRANULARITY), T.float32),
        b_sf: T.Tensor(
            (num_experts_per_rank, n // SCALE_GRANULARITY, k // SCALE_GRANULARITY),
            T.float32,
        ),
        recv_counts: T.Tensor((num_experts_per_rank,), T.int32),
        out: T.Tensor((num_experts_per_rank, capacity, n), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(capacity, block_m), num_experts_per_rank, threads=threads) as (
            bx,
            by,
            bz,
        ):
            a_shared = T.alloc_shared((block_m, block_k), T.float8_e4m3fn)
            b_shared = T.alloc_shared((block_n, block_k), T.float8_e4m3fn)
            out_shared = T.alloc_shared((block_m, block_n), T.bfloat16)
            partial = T.alloc_fragment((block_m, block_n), T.float32)
            accum = T.alloc_fragment((block_m, block_n), T.float32)

            if by * block_m < recv_counts[bz]:
                T.clear(partial)
                T.clear(accum)
                for ko in T.Pipelined(k // block_k, num_stages=pipeline_stages):
                    T.copy(a[bz, by * block_m, ko * block_k], a_shared)
                    T.copy(b[bz, bx * block_n, ko * block_k], b_shared)
                    T.gemm(a_shared, b_shared, partial, transpose_B=True)
                    b_scale = b_sf[bz, bx, ko]
                    for i, j in T.Parallel(block_m, block_n):
                        accum[i, j] += partial[i, j] * (a_sf[bz, by * block_m + i, ko] * b_scale)
                    T.clear(partial)
                T.copy(accum, out_shared)
                T.copy(out_shared, out[bz, by * block_m, bx * block_n])

    return main


def fused_l2_scatter_reduce_manual_warp_kernel(
    num_tokens: int,
    hidden: int,
    intermediate_hidden: int,
    num_experts_per_rank: int,
    num_topk: int,
    num_ranks: int,
    capacity: int,
    num_sms: int,
    block_m: int = 64,
    block_n: int = 256,
    block_k: int = 128,
    reduce_block_m: int = 8,
    reduce_block_h: int = 128,
    threads: int = 384,
    pipeline_stages: int = 3,
):
    num_m_blocks = ceil_div(capacity, block_m)
    num_n_blocks = ceil_div(hidden, block_n)
    num_compute_tiles = num_experts_per_rank * num_m_blocks * num_n_blocks
    num_k_blocks = intermediate_hidden // block_k
    num_reduce_n_blocks = ceil_div(hidden, reduce_block_h)
    num_reduce_m_blocks = ceil_div(num_tokens, reduce_block_m)
    num_reduce_tiles = num_reduce_n_blocks * num_reduce_m_blocks

    @T.prim_func
    def main(
        a: T.Tensor(
            (num_experts_per_rank, capacity, intermediate_hidden),
            T.float8_e4m3fn,
        ),
        b: T.Tensor(
            (num_experts_per_rank, hidden, intermediate_hidden),
            T.float8_e4m3fn,
        ),
        a_sf: T.Tensor(
            (
                num_experts_per_rank,
                capacity,
                intermediate_hidden // SCALE_GRANULARITY,
            ),
            T.float32,
        ),
        b_sf: T.Tensor(
            (
                num_experts_per_rank,
                hidden // SCALE_GRANULARITY,
                intermediate_hidden // SCALE_GRANULARITY,
            ),
            T.float32,
        ),
        recv_counts: T.Tensor((num_experts_per_rank,), T.int32),
        src_ranks: T.Tensor((num_experts_per_rank, capacity), T.int32),
        src_tokens: T.Tensor((num_experts_per_rank, capacity), T.int32),
        src_topk: T.Tensor((num_experts_per_rank, capacity), T.int32),
        combine: T.Tensor((num_tokens, num_topk, hidden), T.bfloat16),
        barrier: T.Tensor((num_ranks,), T.int32),
        out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        with T.Kernel(num_sms, threads=threads) as bid:
            tx = T.get_thread_binding()
            a_shared = T.alloc_shared(
                (pipeline_stages, block_m, block_k),
                T.float8_e4m3fn,
            )
            b_shared = T.alloc_shared(
                (pipeline_stages, block_n, block_k),
                T.float8_e4m3fn,
            )
            out_shared = T.alloc_shared((block_m, block_n), T.bfloat16)
            reduce_shared = T.alloc_shared((reduce_block_m, reduce_block_h), T.bfloat16)
            stage_barriers = T.alloc_barrier([64] * pipeline_stages + [256] * pipeline_stages)

            if tx < 128:
                T.dec_max_nreg(48)
            else:
                T.inc_max_nreg(208)

            if tx >= 64 and tx < 128:
                producer_step = T.alloc_var(T.int32, init=0)
                for producer_wave in T.serial(ceil_div(num_compute_tiles, num_sms)):
                    producer_tile = bid + producer_wave * num_sms
                    if producer_tile < num_compute_tiles:
                        producer_n = producer_tile % num_n_blocks
                        producer_m = (producer_tile // num_n_blocks) % num_m_blocks
                        producer_expert = producer_tile // (num_n_blocks * num_m_blocks)
                        if (
                            producer_expert < num_experts_per_rank
                            and producer_n * block_n < hidden
                            and producer_m * block_m < capacity
                            and producer_m * block_m < recv_counts[producer_expert]
                            and num_k_blocks * block_k == intermediate_hidden
                        ):
                            for producer_k in T.serial(num_k_blocks):
                                producer_stage = (producer_step + producer_k) % pipeline_stages
                                producer_phase = ((producer_step + producer_k) // pipeline_stages) & 1
                                T.mbarrier_wait_parity(
                                    stage_barriers[pipeline_stages + producer_stage],
                                    producer_phase ^ 1,
                                )
                                T.tma_copy(
                                    a[
                                        producer_expert,
                                        producer_m * block_m : (producer_m + 1) * block_m,
                                        producer_k * block_k : (producer_k + 1) * block_k,
                                    ],
                                    a_shared[producer_stage, :, :],
                                    barrier=stage_barriers[producer_stage],
                                )
                                T.tma_copy(
                                    b[
                                        producer_expert,
                                        producer_n * block_n : (producer_n + 1) * block_n,
                                        producer_k * block_k : (producer_k + 1) * block_k,
                                    ],
                                    b_shared[producer_stage, :, :],
                                    barrier=stage_barriers[producer_stage],
                                )
                                T.mbarrier_arrive(stage_barriers[producer_stage])
                            producer_step += num_k_blocks

            elif tx >= 128:
                partial = T.alloc_fragment((block_m, block_n), T.float32)
                accum = T.alloc_fragment((block_m, block_n), T.bfloat16)
                act_scale = T.alloc_fragment((block_m,), T.float32)
                weight_scale = T.alloc_local((2,), T.float32)
                consumer_step = T.alloc_var(T.int32, init=0)

                for consumer_wave in T.serial(ceil_div(num_compute_tiles, num_sms)):
                    consumer_tile = bid + consumer_wave * num_sms
                    if consumer_tile < num_compute_tiles:
                        consumer_n = consumer_tile % num_n_blocks
                        consumer_m = (consumer_tile // num_n_blocks) % num_m_blocks
                        consumer_expert = consumer_tile // (num_n_blocks * num_m_blocks)
                        if (
                            consumer_expert < num_experts_per_rank
                            and consumer_n * block_n < hidden
                            and consumer_m * block_m < capacity
                            and consumer_m * block_m < recv_counts[consumer_expert]
                            and num_k_blocks * block_k == intermediate_hidden
                        ):
                            T.clear(partial)
                            T.clear(accum)
                            for consumer_k in T.serial(num_k_blocks):
                                consumer_stage = (consumer_step + consumer_k) % pipeline_stages
                                consumer_phase = ((consumer_step + consumer_k) // pipeline_stages) & 1
                                T.mbarrier_wait_parity(
                                    stage_barriers[consumer_stage],
                                    consumer_phase,
                                )
                                T.gemm(
                                    a_shared[consumer_stage, :, :],
                                    b_shared[consumer_stage, :, :],
                                    partial,
                                    transpose_B=True,
                                )
                                for i in T.Parallel(block_m):
                                    act_scale[i] = a_sf[
                                        consumer_expert,
                                        consumer_m * block_m + i,
                                        consumer_k,
                                    ]
                                weight_scale[0] = b_sf[
                                    consumer_expert,
                                    consumer_n * 2,
                                    consumer_k,
                                ]
                                weight_scale[1] = b_sf[
                                    consumer_expert,
                                    consumer_n * 2 + 1,
                                    consumer_k,
                                ]
                                for i, j in T.Parallel(block_m, block_n):
                                    accum[i, j] = (
                                        T.cast(partial[i, j], T.bfloat16)
                                        * T.cast(
                                            act_scale[i] * weight_scale[j // 128],
                                            T.bfloat16,
                                        )
                                        + accum[i, j]
                                    )
                                T.clear(partial)
                                T.mbarrier_arrive(stage_barriers[pipeline_stages + consumer_stage])
                            consumer_step += num_k_blocks
                            T.copy(accum, out_shared)
                            scatter_warp = (tx - 128) // 32
                            for row_in_warp in T.serial(block_m // 8):
                                row = scatter_warp * (block_m // 8) + row_in_warp
                                pool_row = consumer_m * block_m + row
                                if pool_row < recv_counts[consumer_expert]:
                                    dst_rank = src_ranks[consumer_expert, pool_row]
                                    dst_token = src_tokens[consumer_expert, pool_row]
                                    dst_topk = src_topk[consumer_expert, pool_row]
                                    if (
                                        dst_rank >= 0
                                        and dst_rank < num_ranks
                                        and dst_token >= 0
                                        and dst_token < num_tokens
                                        and dst_topk >= 0
                                        and dst_topk < num_topk
                                    ):
                                        T.put_warp(
                                            T.address_of(out_shared[row, 0]),
                                            T.address_of(
                                                combine[
                                                    dst_token,
                                                    dst_topk,
                                                    consumer_n * block_n,
                                                ]
                                            ),
                                            block_n,
                                            dst_pe=dst_rank,
                                            unroll_factor=1,
                                        )

            T.fence_sys()
            T.sync_grid()
            if bid == 0:
                T.barrier_blocks(barrier[0])
            T.sync_grid()

            if tx < 128:
                reduce_accum = T.alloc_fragment((reduce_block_m, reduce_block_h), T.float32)
                for reduce_wave in T.serial(ceil_div(num_reduce_tiles, num_sms)):
                    reduce_tile = bid + reduce_wave * num_sms
                    if reduce_tile < num_reduce_tiles:
                        reduce_n = reduce_tile % num_reduce_n_blocks
                        reduce_m = reduce_tile // num_reduce_n_blocks
                        T.clear(reduce_accum)
                        for topk_slot in T.serial(num_topk):
                            for i, j in T.Parallel(reduce_block_m, reduce_block_h):
                                if reduce_m * reduce_block_m + i < num_tokens:
                                    reduce_accum[i, j] += combine[
                                        reduce_m * reduce_block_m + i,
                                        topk_slot,
                                        reduce_n * reduce_block_h + j,
                                    ]
                        T.copy(reduce_accum, reduce_shared)
                        T.copy(
                            reduce_shared,
                            out[
                                reduce_m * reduce_block_m,
                                reduce_n * reduce_block_h,
                            ],
                        )

    return main


def swiglu_quant_kernel(
    num_experts_per_rank: int,
    capacity: int,
    intermediate_hidden: int,
    block_m: int = 8,
    block_n: int = 128,
    threads: int = 128,
    activation_clamp: float = 10.0,
):
    @T.prim_func
    def main(
        gate_up: T.Tensor((num_experts_per_rank, capacity, 2 * intermediate_hidden), T.bfloat16),
        route_weights: T.Tensor((num_experts_per_rank, capacity), T.float32),
        recv_counts: T.Tensor((num_experts_per_rank,), T.int32),
        out: T.Tensor((num_experts_per_rank, capacity, intermediate_hidden), T.float8_e4m3fn),
        out_sf: T.Tensor(
            (num_experts_per_rank, capacity, intermediate_hidden // SCALE_GRANULARITY),
            T.float32,
        ),
    ):
        with T.Kernel(
            intermediate_hidden // block_n,
            T.ceildiv(capacity, block_m),
            num_experts_per_rank,
            threads=threads,
        ) as (bx, by, bz):
            gate = T.alloc_fragment((block_m, block_n), T.float32)
            up = T.alloc_fragment((block_m, block_n), T.float32)
            activated = T.alloc_fragment((block_m, block_n), T.float32)
            amax = T.alloc_fragment((block_m,), T.float32)
            scale = T.alloc_fragment((block_m,), T.float32)
            quant = T.alloc_fragment((block_m, block_n), T.float32)
            quant_fp8 = T.alloc_fragment((block_m, block_n), T.float8_e4m3fn)

            if by * block_m < recv_counts[bz]:
                T.copy(gate_up[bz, by * block_m, bx * block_n], gate)
                T.copy(gate_up[bz, by * block_m, intermediate_hidden + bx * block_n], up)
                for i, j in T.Parallel(block_m, block_n):
                    gate[i, j] = T.min(gate[i, j], activation_clamp)
                    up[i, j] = T.max(T.min(up[i, j], activation_clamp), -activation_clamp)
                    activated[i, j] = gate[i, j] * T.sigmoid(gate[i, j]) * up[i, j] * route_weights[bz, by * block_m + i]
                T.reduce_absmax(activated, amax, dim=1)
                for i in T.Parallel(block_m):
                    scale[i] = T.max(amax[i], 1e-4) / FP8_MAX
                    out_sf[bz, by * block_m + i, bx] = scale[i]
                for i, j in T.Parallel(block_m, block_n):
                    quant[i, j] = T.clamp(activated[i, j] / scale[i], -FP8_MAX, FP8_MAX)
                T.copy(quant, quant_fp8)
                T.copy(quant_fp8, out[bz, by * block_m, bx * block_n])

    return main


def scatter_outputs_kernel(
    num_experts_per_rank: int,
    capacity: int,
    num_tokens: int,
    num_topk: int,
    hidden: int,
    block_h: int = 256,
    threads: int = 128,
):
    @T.prim_func
    def main(
        local_out: T.Tensor((num_experts_per_rank, capacity, hidden), T.bfloat16),
        recv_counts: T.Tensor((num_experts_per_rank,), T.int32),
        src_ranks: T.Tensor((num_experts_per_rank, capacity), T.int32),
        src_tokens: T.Tensor((num_experts_per_rank, capacity), T.int32),
        src_topk: T.Tensor((num_experts_per_rank, capacity), T.int32),
        combine: T.Tensor((num_tokens, num_topk, hidden), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(hidden, block_h), capacity, num_experts_per_rank, threads=threads) as (bx, by, bz):
            if by < recv_counts[bz]:
                dst_rank = src_ranks[bz, by]
                token_idx = src_tokens[bz, by]
                topk_slot = src_topk[bz, by]
                if dst_rank >= 0 and token_idx < num_tokens and topk_slot < num_topk:
                    T.copy(
                        local_out[bz, by, bx * block_h : (bx + 1) * block_h],
                        combine[token_idx, topk_slot, bx * block_h : (bx + 1) * block_h],
                        dst_pe=dst_rank,
                        disable_tma=True,
                    )

    return main


def reduce_topk_kernel(
    num_tokens: int,
    num_topk: int,
    hidden: int,
    block_m: int = 8,
    block_h: int = 128,
    threads: int = 128,
):
    @T.prim_func
    def main(
        combine: T.Tensor((num_tokens, num_topk, hidden), T.bfloat16),
        out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(hidden, block_h), T.ceildiv(num_tokens, block_m), threads=threads) as (bx, by):
            accum = T.alloc_fragment((block_m, block_h), T.float32)
            out_shared = T.alloc_shared((block_m, block_h), T.bfloat16)
            T.clear(accum)
            for topk_slot in T.serial(num_topk):
                for i, j in T.Parallel(block_m, block_h):
                    if by * block_m + i < num_tokens:
                        accum[i, j] += combine[by * block_m + i, topk_slot, bx * block_h + j]
            T.copy(accum, out_shared)
            T.copy(out_shared, out[by * block_m, bx * block_h])

    return main


def _allocator_size_bytes(
    num_tokens: int,
    hidden: int,
    intermediate_hidden: int,
    num_experts_per_rank: int,
    num_topk: int,
    capacity: int,
) -> int:
    fp8 = 1
    bf16 = 2
    fp32 = 4
    i32 = 4
    weight_bytes = num_experts_per_rank * (2 * intermediate_hidden * hidden * fp8 + hidden * intermediate_hidden * fp8)
    weight_scale_bytes = (
        num_experts_per_rank * ((2 * intermediate_hidden // 128) * (hidden // 128) + (hidden // 128) * (intermediate_hidden // 128)) * fp32
    )
    pool_bytes = (
        num_experts_per_rank
        * capacity
        * (
            hidden * fp8
            + (hidden // 128) * fp32
            + 4 * i32
            + 2 * intermediate_hidden * bf16
            + intermediate_hidden * fp8
            + (intermediate_hidden // 128) * fp32
            + hidden * bf16
        )
    )
    input_bytes = num_tokens * (hidden * fp8 + (hidden // 128) * fp32 + num_topk * (3 * i32 + fp32) + num_topk * hidden * bf16)
    return align_up(weight_bytes + weight_scale_bytes + pool_bytes + input_bytes + 2**27, 2**20)


def _gather_cat(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, tensor, group=group)
    return torch.cat(gathered, dim=0)


def torch_reference(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    l1_fp8: torch.Tensor,
    l1_sf: torch.Tensor,
    l2_fp8: torch.Tensor,
    l2_sf: torch.Tensor,
    group: dist.ProcessGroup,
    activation_clamp: float,
) -> torch.Tensor:
    l1_all = _gather_cat(l1_fp8, group)
    l1_sf_all = _gather_cat(l1_sf, group)
    l2_all = _gather_cat(l2_fp8, group)
    l2_sf_all = _gather_cat(l2_sf, group)
    x = dequantize_per_token(x_fp8, x_sf)
    result = torch.zeros((x.size(0), l2_all.size(1)), dtype=torch.float32, device=x.device)

    for expert_idx in range(l1_all.size(0)):
        positions = (topk_idx == expert_idx).nonzero(as_tuple=False)
        if positions.numel() == 0:
            continue
        token_indices = positions[:, 0]
        topk_slots = positions[:, 1]
        l1_weight = dequantize_block(l1_all[expert_idx : expert_idx + 1], l1_sf_all[expert_idx : expert_idx + 1])[0]
        gate_up = x[token_indices] @ l1_weight.T
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=activation_clamp)
        up = up.clamp(min=-activation_clamp, max=activation_clamp)
        activated = torch.nn.functional.silu(gate) * up
        activated *= topk_weights[token_indices, topk_slots].unsqueeze(-1)
        activated_fp8, activated_sf = per_token_cast_to_fp8(activated)
        activated_dequant = dequantize_per_token(activated_fp8, activated_sf)
        l2_weight = dequantize_block(l2_all[expert_idx : expert_idx + 1], l2_sf_all[expert_idx : expert_idx + 1])[0]
        contribution = (activated_dequant @ l2_weight.T).to(torch.bfloat16).float()
        result.index_add_(0, token_indices, contribution)

    return result.to(torch.bfloat16)


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    return (1 - 2 * (x * y).sum() / (x.square() + y.square()).sum()).item()


def allocator_tensor(shape, dtype, allocator):
    if dtype == torch.float8_e4m3fn:
        return tilelang.tensor(shape, torch.uint8, allocator=allocator).view(dtype)
    return tilelang.tensor(shape, dtype, allocator=allocator)


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    model = MODEL_CONFIGS[args.model_config]
    hidden = model["hidden"]
    intermediate_hidden = model["intermediate_hidden"]
    num_experts = model["num_experts"]
    num_topk = model["num_topk"]
    num_tokens = args.num_tokens
    activation_clamp = args.activation_clamp
    use_fused = args.model_config != "pro"

    assert num_experts % num_local_ranks == 0
    assert hidden % 256 == 0 and intermediate_hidden % 128 == 0
    num_experts_per_rank = num_experts // num_local_ranks
    average_recv = ceil_div(num_tokens * num_local_ranks * num_topk, num_experts)
    capacity = args.capacity or align_up(max(average_recv * 2, 64), 64)

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert rank == local_rank and num_ranks == num_local_ranks
    num_sms = torch.cuda.get_device_properties(local_rank).multi_processor_count
    allocator = get_allocator(
        size=_allocator_size_bytes(
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

    if use_fused:
        kernel_specs = [
            fused_l1_swiglu_manual_warp_kernel(
                num_tokens,
                hidden,
                2 * intermediate_hidden,
                num_experts,
                num_topk,
                num_ranks,
                capacity,
                num_sms,
                activation_clamp=activation_clamp,
            ),
            fused_l2_scatter_reduce_manual_warp_kernel(
                num_tokens,
                hidden,
                intermediate_hidden,
                num_experts_per_rank,
                num_topk,
                num_ranks,
                capacity,
                num_sms,
            ),
        ]
    else:
        kernel_specs = [
            reset_route_counts_kernel(num_experts, num_ranks),
            assign_local_routes_kernel(num_tokens, num_experts, num_topk, num_ranks),
            publish_route_counts_kernel(num_experts, num_ranks),
            device_barrier_kernel(num_ranks),
            finalize_routes_kernel(num_tokens, num_experts, num_topk, num_ranks),
            dispatch_tokens_kernel(num_tokens, hidden, num_experts, num_topk, num_ranks, capacity),
            fp8_grouped_gemm_kernel(num_experts_per_rank, capacity, 2 * intermediate_hidden, hidden),
            swiglu_quant_kernel(
                num_experts_per_rank,
                capacity,
                intermediate_hidden,
                activation_clamp=activation_clamp,
            ),
            fp8_grouped_gemm_kernel(num_experts_per_rank, capacity, hidden, intermediate_hidden),
            scatter_outputs_kernel(num_experts_per_rank, capacity, num_tokens, num_topk, hidden),
            reduce_topk_kernel(num_tokens, num_topk, hidden),
        ]
    kernels = [tilelang.compile(spec, compile_once=True, compile_group=group) for spec in kernel_specs]
    for kernel in kernels:
        kernel.initialize(allocator=allocator)
    if use_fused:
        fused_l1, fused_l2 = kernels
    else:
        (
            reset_route_counts,
            assign_local_routes,
            publish_route_counts,
            device_barrier,
            finalize_routes,
            dispatch_tokens,
            l1_gemm,
            swiglu_quant,
            l2_gemm,
            scatter_outputs,
            reduce_topk,
        ) = kernels

    if local_rank == 0 and args.print_source:
        for kernel in kernels:
            print(kernel.get_kernel_source())

    torch.manual_seed(args.seed + local_rank)
    x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    x_fp8_src, x_sf_src = per_token_cast_to_fp8(x_bf16)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda")
    topk_weights_src, topk_idx_src = torch.topk(scores, num_topk, dim=-1, sorted=False)
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
    l1_fp8_src, l1_sf_src = block_cast_to_fp8(l1_bf16)
    l2_fp8_src, l2_sf_src = block_cast_to_fp8(l2_bf16)
    del scores, l1_bf16, l2_bf16

    # barrier_blocks currently lowers its byte offset as int32, so keep this
    # allocation before the multi-gigabyte Pro-model weights.
    barrier = allocator_tensor((num_ranks,), torch.int32, allocator=allocator)
    x = allocator_tensor(x_fp8_src.shape, x_fp8_src.dtype, allocator=allocator).copy_(x_fp8_src)
    x_sf = allocator_tensor(x_sf_src.shape, x_sf_src.dtype, allocator=allocator).copy_(x_sf_src)
    topk_idx = allocator_tensor(topk_idx_src.shape, topk_idx_src.dtype, allocator=allocator).copy_(topk_idx_src)
    topk_weights = allocator_tensor(topk_weights_src.shape, topk_weights_src.dtype, allocator=allocator).copy_(topk_weights_src)
    l1_fp8_kernel = interleave_gate_up_weights(l1_fp8_src) if use_fused else l1_fp8_src
    l1_fp8 = allocator_tensor(l1_fp8_kernel.shape, l1_fp8_kernel.dtype, allocator=allocator).copy_(l1_fp8_kernel)
    del l1_fp8_kernel
    l1_sf = allocator_tensor(l1_sf_src.shape, l1_sf_src.dtype, allocator=allocator).copy_(l1_sf_src)
    l2_fp8 = allocator_tensor(l2_fp8_src.shape, l2_fp8_src.dtype, allocator=allocator).copy_(l2_fp8_src)
    l2_sf = allocator_tensor(l2_sf_src.shape, l2_sf_src.dtype, allocator=allocator).copy_(l2_sf_src)

    route_counts = allocator_tensor((num_ranks, num_experts), torch.int32, allocator=allocator)
    recv_counts = allocator_tensor((num_experts_per_rank,), torch.int32, allocator=allocator)
    recv_x = allocator_tensor((num_experts_per_rank, capacity, hidden), torch.float8_e4m3fn, allocator=allocator)
    recv_x_sf = allocator_tensor(
        (num_experts_per_rank, capacity, hidden // SCALE_GRANULARITY),
        torch.float32,
        allocator=allocator,
    )
    recv_weights = allocator_tensor((num_experts_per_rank, capacity), torch.float32, allocator=allocator)
    src_ranks = allocator_tensor((num_experts_per_rank, capacity), torch.int32, allocator=allocator)
    src_tokens = allocator_tensor((num_experts_per_rank, capacity), torch.int32, allocator=allocator)
    src_topk = allocator_tensor((num_experts_per_rank, capacity), torch.int32, allocator=allocator)
    route_slots = allocator_tensor((num_tokens, num_topk), torch.int32, allocator=allocator)
    if use_fused:
        arrivals = allocator_tensor(
            (num_experts_per_rank, ceil_div(capacity, 64)),
            torch.uint32,
            allocator=allocator,
        )
    if not use_fused:
        l1_out = allocator_tensor(
            (num_experts_per_rank, capacity, 2 * intermediate_hidden),
            torch.bfloat16,
            allocator=allocator,
        )
    l2_x = allocator_tensor((num_experts_per_rank, capacity, intermediate_hidden), torch.float8_e4m3fn, allocator=allocator)
    l2_x_sf = allocator_tensor(
        (num_experts_per_rank, capacity, intermediate_hidden // SCALE_GRANULARITY),
        torch.float32,
        allocator=allocator,
    )
    if not use_fused:
        l2_out = allocator_tensor(
            (num_experts_per_rank, capacity, hidden),
            torch.bfloat16,
            allocator=allocator,
        )
    combine = allocator_tensor((num_tokens, num_topk, hidden), torch.bfloat16, allocator=allocator)
    out = allocator_tensor((num_tokens, hidden), torch.bfloat16, allocator=allocator)

    def reset_state():
        route_counts.zero_()
        barrier.zero_()
        recv_counts.zero_()
        if use_fused:
            arrivals.zero_()
        recv_x.zero_()
        recv_x_sf.zero_()
        recv_weights.zero_()
        src_ranks.fill_(-1)
        combine.zero_()
        torch.cuda.synchronize()
        dist.barrier(group=group)

    def run_pipeline(check_capacity: bool = False):
        if use_fused:
            fused_l1(
                x,
                x_sf,
                topk_idx,
                topk_weights,
                route_counts,
                recv_counts,
                route_slots,
                arrivals,
                recv_x,
                recv_x_sf,
                recv_weights,
                src_ranks,
                src_tokens,
                src_topk,
                l1_fp8,
                l1_sf,
                l2_x,
                l2_x_sf,
                barrier,
            )
        else:
            reset_route_counts(route_counts)
            assign_local_routes(topk_idx, route_counts, route_slots)
            publish_route_counts(route_counts)
            device_barrier(barrier)
            finalize_routes(topk_idx, route_counts, recv_counts, route_slots)
            dispatch_tokens(
                x,
                x_sf,
                topk_idx,
                topk_weights,
                route_slots,
                recv_x,
                recv_x_sf,
                recv_weights,
                src_ranks,
                src_tokens,
                src_topk,
            )
            device_barrier(barrier)
        if check_capacity:
            local_max = recv_counts.max()
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=group)
            assert local_max.item() <= capacity, f"expert capacity {capacity} is smaller than received routes {local_max.item()}"
        if use_fused:
            fused_l2(
                l2_x,
                l2_fp8,
                l2_x_sf,
                l2_sf,
                recv_counts,
                src_ranks,
                src_tokens,
                src_topk,
                combine,
                barrier,
                out,
            )
        else:
            l1_gemm(recv_x, l1_fp8, recv_x_sf, l1_sf, recv_counts, l1_out)
            swiglu_quant(l1_out, recv_weights, recv_counts, l2_x, l2_x_sf)
            l2_gemm(l2_x, l2_fp8, l2_x_sf, l2_sf, recv_counts, l2_out)
            scatter_outputs(l2_out, recv_counts, src_ranks, src_tokens, src_topk, combine)
            device_barrier(barrier)
            reduce_topk(combine, out)
        return out

    reset_state()
    actual = run_pipeline(check_capacity=True)
    torch.cuda.synchronize()
    dist.barrier(group=group)

    if args.check:
        expected = torch_reference(
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
        diff = calc_diff(actual, expected)
        assert diff < args.diff_tol, f"rank {local_rank}: diff={diff} exceeds {args.diff_tol}"
        print(f"rank {local_rank} check passed, diff={diff:.6f}")

    if args.rep > 0:
        reset_state()
        # Stateful synchronization counters must be reset between warmup iterations.
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
                f"tilescale sm90 fp8 mega moe: model={args.model_config} M={num_tokens} "
                f"implementation={'fused' if use_fused else 'multi-kernel'} "
                f"capacity={capacity} latency={latency * 1000:.1f} us"
            )

    allocator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--model-config", choices=tuple(MODEL_CONFIGS), default="smoke")
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=None)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diff-tol", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-source", action="store_true")
    args = parser.parse_args()
    torch.multiprocessing.spawn(main, args=(args.num_processes, args), nprocs=args.num_processes)
