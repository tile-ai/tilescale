"""Multi-GPU FP8 Mega MoE for SM90 using TileScale distributed primitives.

The fused implementation uses two persistent kernels: one for routing,
dispatch, L1 GEMM, and SwiGLU quantization, and one for L2 GEMM, scatter, and
reduction. Model presets and aligned custom shapes use manually selected warp
counts and shape/load-based pipeline depths. Custom hidden sizes must be at
least 512 and divisible by 256; intermediate sizes must be divisible by 128.
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


def resolve_model_config(args: argparse.Namespace) -> Tuple[str, dict[str, int]]:
    model = MODEL_CONFIGS[args.model_config].copy()
    overrides = {
        "hidden": getattr(args, "hidden", None),
        "intermediate_hidden": getattr(args, "intermediate_hidden", None),
        "num_experts": getattr(args, "num_experts", None),
        "num_topk": getattr(args, "num_topk", None),
    }
    is_custom = any(value is not None for value in overrides.values())
    model.update({key: value for key, value in overrides.items() if value is not None})
    return ("custom" if is_custom else args.model_config), model


def normalize_experts_per_wave(num_experts: int, requested: int) -> int:
    requested = min(max(requested, 1), num_experts)
    for candidate in range(requested, num_experts + 1):
        if num_experts % candidate == 0:
            return candidate
    return num_experts


def select_manual_warp_configs(
    hidden: int,
    intermediate_hidden: int,
    num_tokens: int,
    num_topk: int,
    num_experts_per_rank: int,
    num_sms: int,
) -> Tuple[str, dict[str, int], dict[str, int]]:
    """Select the TileScale counterpart of DeepGEMM SM90 schedule families."""
    if 3072 <= hidden < 5120 and 1536 <= intermediate_hidden < 2560:
        shape_family = "compact"
    elif 5120 <= hidden <= 8192 and 2560 <= intermediate_hidden <= 4096:
        shape_family = "wide"
    else:
        shape_family = "generic"

    routed_tokens = num_tokens * num_topk
    high_sm = num_sms >= 100

    # Measured on Flash (4x H200): three stages ties five at M<=512 and wins
    # 0.6%/2.1% at M=2048/8192, so the deeper default is not worth its
    # shared memory.
    l1_stages = 3
    l2_stages = 3
    generic_experts_per_wave = num_experts_per_rank
    if num_experts_per_rank <= routed_tokens <= 4 * num_experts_per_rank:
        expected_tokens = (routed_tokens + num_experts_per_rank - 1) // num_experts_per_rank
        num_m_blocks = (expected_tokens + 63) // 64
        blocks_per_expert = num_m_blocks * (2 * intermediate_hidden // 256)
        requested = min(num_experts_per_rank, (2 * num_sms + blocks_per_expert - 1) // blocks_per_expert)
        if blocks_per_expert < num_sms:
            max_candidate = min(num_experts_per_rank, 2 * requested)
            requested = max(
                range(requested, max_candidate + 1),
                key=lambda candidate: 1.0 if num_experts_per_rank % candidate == 0 else (num_experts_per_rank % candidate) / candidate,
            )
        generic_experts_per_wave = normalize_experts_per_wave(num_experts_per_rank, requested)
    l1_experts_per_wave = l2_experts_per_wave = generic_experts_per_wave
    if high_sm and shape_family == "compact":
        if routed_tokens <= 32 * num_experts_per_rank:
            l1_stages = l2_stages = 3
            l1_experts_per_wave = l2_experts_per_wave = normalize_experts_per_wave(num_experts_per_rank, 4)
        elif 128 * num_experts_per_rank < routed_tokens <= 256 * num_experts_per_rank or routed_tokens > 1024 * num_experts_per_rank:
            l1_stages = l2_stages = 4
            l1_experts_per_wave = l2_experts_per_wave = normalize_experts_per_wave(num_experts_per_rank, 32)
    elif high_sm and shape_family == "wide":
        # BN512/BK256 are profitable in the CUDA kernel, but the manually
        # tuned TileScale BN256/BK128 path is faster for the current WGMMA
        # lowering and remains the generic Wide schedule.
        l1_stages = 4
        if routed_tokens <= 24 * num_experts_per_rank:
            # CUDA selects 16 experts here, while TileScale's direct TIR
            # scheduler is faster with a shorter four-expert scan on H200.
            l1_experts_per_wave = l2_experts_per_wave = normalize_experts_per_wave(num_experts_per_rank, 4)
        elif 24 * num_experts_per_rank < routed_tokens <= 48 * num_experts_per_rank:
            l1_experts_per_wave = normalize_experts_per_wave(num_experts_per_rank, 8)
            l2_experts_per_wave = normalize_experts_per_wave(num_experts_per_rank, 48)
        elif routed_tokens > 48 * num_experts_per_rank:
            l1_experts_per_wave = l2_experts_per_wave = normalize_experts_per_wave(num_experts_per_rank, 16)

    common = {"block_m": 64, "block_n": 256, "block_k": 128, "threads": 384}
    return (
        shape_family,
        {**common, "pipeline_stages": l1_stages, "num_experts_per_wave": l1_experts_per_wave},
        {**common, "pipeline_stages": l2_stages, "num_experts_per_wave": l2_experts_per_wave},
    )


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    x_view = x.float().view(m, k // SCALE_GRANULARITY, SCALE_GRANULARITY)
    amax = x_view.abs().amax(dim=-1).clamp(1e-4)
    scale = amax / FP8_MAX
    x_fp8 = (x_view / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return x_fp8.view(m, k).contiguous(), scale.contiguous()


def block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    groups, n, k = x.shape
    x_view = x.float().view(groups, n // SCALE_GRANULARITY, SCALE_GRANULARITY, k // SCALE_GRANULARITY, SCALE_GRANULARITY)
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
    x_view = x.float().view(groups, n // SCALE_GRANULARITY, SCALE_GRANULARITY, k // SCALE_GRANULARITY, SCALE_GRANULARITY)
    return (x_view * scale.unsqueeze(-1).unsqueeze(-3)).view(groups, n, k)

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
    num_experts_per_wave: int | None = None,
    frontend_regs_override: int | None = None,
    math_regs_override: int | None = None,
):
    num_experts_per_rank = num_experts // num_ranks
    num_experts_per_wave = num_experts_per_wave or num_experts_per_rank
    assert num_experts_per_rank % num_experts_per_wave == 0
    num_scale_groups = hidden // SCALE_GRANULARITY
    num_routes = num_tokens * num_topk
    num_m_blocks = T.ceildiv(capacity, block_m)
    num_n_blocks = T.ceildiv(l1_n, block_n)
    num_k_blocks = hidden // block_k
    num_k_sub = block_k // SCALE_GRANULARITY
    assert block_k % SCALE_GRANULARITY == 0
    # WG0 is the frontend: warps 0-1 dispatch routes and warps 2-3 issue TMA.
    # WG1+ are WGMMA consumers; each owns an N fragment of the CTA tile.
    warp_size = 32
    warpgroup_size = 128
    dispatch_warps = 2
    producer_warps = 2
    frontend_warps = dispatch_warps + producer_warps
    dispatch_threads = dispatch_warps * warp_size
    producer_begin = dispatch_threads
    producer_threads = producer_warps * warp_size
    producer_end = frontend_warps * warp_size
    math_begin = producer_end
    num_math_threads = threads - math_begin
    assert producer_end == producer_begin + producer_threads == warpgroup_size
    assert num_math_threads > 0 and num_math_threads % warpgroup_size == 0
    math_warpgroups = num_math_threads // warpgroup_size
    assert math_warpgroups in (2, 4)
    num_output_scale_groups = block_n // (2 * SCALE_GRANULARITY)
    num_l1_scale_groups = l1_n // (2 * SCALE_GRANULARITY)
    tma_block_n = min(block_n, 256)
    num_tma_n_blocks = block_n // tma_block_n
    # Budgets must leave the CTA register pool some slack: 128*fe + 256*math
    # exactly at 65536 (e.g. 32/240) compiles but deadlocks at run time.
    # Spilling tracks the frontend budget, not the math one -- 40/48/56 give
    # 72/16/0 bytes of spill -- so keep the frontend at 64 for a spill-free build.
    frontend_registers = frontend_regs_override or (32 if num_math_threads == 512 else 64)
    math_registers = math_regs_override or (112 if num_math_threads == 512 else 192)
    dispatch_leader_lane = 0
    route_threads = num_math_threads
    assert route_threads % warp_size == 0

    @T.prim_func
    def main(
        x: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        x_sf: T.Tensor((num_tokens, num_scale_groups), T.float32),
        topk_idx: T.Tensor((num_tokens, num_topk), T.int32),
        topk_weights: T.Tensor((num_tokens, num_topk), T.float32),
        route_counts: T.Tensor((num_ranks, num_experts), T.int32),
        recv_counts: T.Tensor((num_experts_per_rank,), T.int32),
        route_slots: T.Tensor((num_tokens, num_topk), T.int32),
        arrivals: T.Tensor((num_experts_per_rank, T.ceildiv(capacity, block_m)), T.uint32),
        m_tasks: T.Tensor((num_experts_per_rank * num_m_blocks,), T.int32),
        num_m_tasks: T.Tensor((1,), T.int32),
        recv_x: T.Tensor((num_experts_per_rank, capacity, hidden), T.float8_e4m3fn),
        recv_x_sf: T.Tensor((num_experts_per_rank, capacity, num_scale_groups), T.float32),
        recv_weights: T.Tensor((num_experts_per_rank, capacity), T.float32),
        src_ranks: T.Tensor((num_experts_per_rank, capacity), T.int32),
        src_tokens: T.Tensor((num_experts_per_rank, capacity), T.int32),
        src_topk: T.Tensor((num_experts_per_rank, capacity), T.int32),
        l1_weight: T.Tensor((num_experts_per_rank, l1_n, hidden), T.float8_e4m3fn),
        l1_weight_sf: T.Tensor((num_experts_per_rank, l1_n // SCALE_GRANULARITY, hidden // SCALE_GRANULARITY), T.float32),
        l2_x: T.Tensor((num_experts_per_rank, capacity, l1_n // 2), T.float8_e4m3fn),
        l2_x_sf: T.Tensor((num_experts_per_rank, capacity, l1_n // (2 * SCALE_GRANULARITY)), T.float32),
        barrier: T.Tensor((num_ranks,), T.int32),
    ):
        T.annotate_pass_configs({tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
        with T.Kernel(num_sms, threads=threads) as bid:
            tx = T.get_thread_binding()
            src_rank = T.alloc_local((1,), T.int32)
            src_rank[0] = T.get_rank()

            # A TMA stage holds num_k_sub contiguous SCALE_GRANULARITY-deep
            # sub-tiles: one barrier round-trip covers all of them, while WGMMA
            # still consumes them one scale group at a time.
            a_shared = T.alloc_shared((pipeline_stages, num_k_sub, block_m, SCALE_GRANULARITY), T.float8_e4m3fn)
            b_shared = T.alloc_shared((pipeline_stages, num_k_sub, block_n, SCALE_GRANULARITY), T.float8_e4m3fn)
            out_shared = T.alloc_shared((block_m, block_n // 2), T.float8_e4m3fn)
            stage_barriers = T.alloc_barrier([producer_threads] * pipeline_stages + [num_math_threads] * pipeline_stages)

            # Routing runs on the math warpgroups, which hold the large budget.
            route_tid = tx - math_begin
            if bid == 0:
                if tx >= math_begin:
                    for reset_expert in T.serial(route_tid, num_experts, route_threads):
                        route_counts[src_rank[0], reset_expert] = 0
                    T.sync_threads(7, route_threads)

                    for assign_route in T.serial(route_tid, num_routes, route_threads):
                        assign_token = assign_route // num_topk
                        assign_topk = assign_route % num_topk
                        assign_expert = topk_idx[assign_token, assign_topk]
                        if assign_expert >= 0 and assign_expert < num_experts:
                            route_slots[assign_token, assign_topk] = T.atomic_add(route_counts[src_rank[0], assign_expert], 1, memory_order="relaxed", return_prev=True)
                        else:
                            route_slots[assign_token, assign_topk] = -1
                    T.sync_threads(7, route_threads)

                    publish_warp = route_tid // warp_size
                    for publish_rank in T.serial(publish_warp, num_ranks, route_threads // warp_size):
                        if publish_rank != src_rank[0]:
                            T.put_warp(T.address_of(route_counts[src_rank[0], 0]), T.address_of(route_counts[src_rank[0], 0]), num_experts, dst_pe=publish_rank, unroll_factor=8)

                T.barrier_blocks(barrier[0])

                if tx >= math_begin:
                    for count_local_expert in T.serial(route_tid, num_experts_per_rank, route_threads):
                        recv_count = T.alloc_var(T.int32, init=0)
                        recv_expert = src_rank[0] * num_experts_per_rank + count_local_expert
                        for count_rank in T.serial(num_ranks):
                            recv_count += route_counts[count_rank, recv_expert]
                        recv_counts[count_local_expert] = recv_count

                    for prefix_route in T.serial(route_tid, num_routes, route_threads):
                        prefix_token = prefix_route // num_topk
                        prefix_topk = prefix_route % num_topk
                        prefix_expert = topk_idx[prefix_token, prefix_topk]
                        prefix_slot = T.alloc_var(T.int32, init=route_slots[prefix_token, prefix_topk])
                        if prefix_token < num_tokens and prefix_expert >= 0 and prefix_expert < num_experts and prefix_slot >= 0:
                            for prefix_rank in T.serial(num_ranks):
                                if prefix_rank < src_rank[0]:
                                    prefix_slot += route_counts[prefix_rank, prefix_expert]
                            route_slots[prefix_token, prefix_topk] = prefix_slot

                    T.sync_threads(7, route_threads)
                    if route_tid == 0:
                        task_cursor = T.alloc_var(T.int32, init=0)
                        for task_expert in T.serial(num_experts_per_rank):
                            task_expert_m_blocks = T.ceildiv(T.min(recv_counts[task_expert], capacity), block_m)
                            for task_m in T.serial(num_m_blocks):
                                if task_m < task_expert_m_blocks:
                                    m_tasks[task_cursor] = task_expert * num_m_blocks + task_m
                                    task_cursor += 1
                        num_m_tasks[0] = task_cursor

            T.sync_grid()

            dispatch_warp = tx // warp_size
            dispatch_lane = tx % warp_size
            # setmaxnreg is only honored when the dec/inc dominates the specialized
            # code itself: with a separate if/else, ptxas cannot prove the
            # deallocating threads never reach the high-pressure path and drops both.
            if tx < math_begin:
                T.dec_max_nreg(frontend_registers)
                if tx < dispatch_threads:
                    # WG0 warps 0-1 publish route metadata and pull remote activation rows.
                    for metadata_route in T.serial(bid * dispatch_threads + tx, num_routes, num_sms * dispatch_threads):
                        metadata_token = metadata_route // num_topk
                        metadata_topk = metadata_route % num_topk
                        metadata_expert = topk_idx[metadata_token, metadata_topk]
                        metadata_slot = route_slots[metadata_token, metadata_topk]
                        if metadata_expert >= 0 and metadata_slot >= 0 and metadata_slot < capacity:
                            metadata_rank = metadata_expert // num_experts_per_rank
                            metadata_local_expert = metadata_expert % num_experts_per_rank
                            T.st(recv_weights[metadata_local_expert, metadata_slot], topk_weights[metadata_token, metadata_topk], dst_pe=metadata_rank)
                            T.st(src_tokens[metadata_local_expert, metadata_slot], metadata_token, dst_pe=metadata_rank)
                            T.st(src_topk[metadata_local_expert, metadata_slot], metadata_topk, dst_pe=metadata_rank)
                            T.st(src_ranks[metadata_local_expert, metadata_slot], src_rank[0], scope="sys", sem="release", dst_pe=metadata_rank)

                    for pull_idx in T.serial(bid * dispatch_warps + dispatch_warp, num_m_tasks[0] * block_m, num_sms * dispatch_warps):
                        pull_m_task = m_tasks[pull_idx // block_m]
                        pull_expert = pull_m_task // num_m_blocks
                        pull_slot = (pull_m_task % num_m_blocks) * block_m + pull_idx % block_m
                        if pull_slot < recv_counts[pull_expert]:
                            if dispatch_lane == dispatch_leader_lane:
                                T.wait_ge(src_ranks[pull_expert, pull_slot], 0, scope=T.WaitScope.SYS, semantics=T.WaitSemantics.ACQUIRE)
                            T.sync_warp()
                            pull_rank = src_ranks[pull_expert, pull_slot]
                            pull_token = src_tokens[pull_expert, pull_slot]
                            T.get_warp(T.address_of(x[pull_token, 0]), T.address_of(recv_x[pull_expert, pull_slot, 0]), hidden, src_pe=pull_rank, unroll_factor=8)
                            T.get_warp(T.address_of(x_sf[pull_token, 0]), T.address_of(recv_x_sf[pull_expert, pull_slot, 0]), num_scale_groups, src_pe=pull_rank, unroll_factor=8)
                            T.sync_warp()
                            if dispatch_lane == dispatch_leader_lane:
                                T.atom_add(arrivals[pull_expert, pull_slot // block_m], 1, scope="gpu", sem="release")

                if tx >= producer_begin and tx < producer_end:
                    # WG0 warps 2-3 keep the shared-memory pipeline filled with TMA.
                    producer_step = T.alloc_var(T.int32, init=0)
                    for producer_task in T.serial(bid, num_m_tasks[0] * num_n_blocks, num_sms):
                        producer_n = producer_task % num_n_blocks
                        producer_m_task = m_tasks[producer_task // num_n_blocks]
                        producer_m = producer_m_task % num_m_blocks
                        producer_expert = producer_m_task // num_m_blocks
                        if producer_n * block_n < l1_n:
                                producer_arrivals = T.min(block_m, recv_counts[producer_expert] - producer_m * block_m)
                                if tx == producer_begin:
                                    T.wait_ge(arrivals[producer_expert, producer_m], producer_arrivals, scope=T.WaitScope.GPU, semantics=T.WaitSemantics.ACQUIRE)
                                T.sync_threads(5, producer_threads)
                                for producer_k in T.serial(num_k_blocks):
                                    producer_stage = (producer_step + producer_k) % pipeline_stages
                                    producer_phase = ((producer_step + producer_k) // pipeline_stages) & 1
                                    T.mbarrier_wait_parity(stage_barriers[pipeline_stages + producer_stage], producer_phase ^ 1)
                                    for producer_ks in T.unroll(num_k_sub):
                                        producer_sf_k = producer_k * num_k_sub + producer_ks
                                        T.tma_copy(
                                            recv_x[
                                                producer_expert,
                                                producer_m * block_m : (producer_m + 1) * block_m,
                                                producer_sf_k * SCALE_GRANULARITY : (producer_sf_k + 1) * SCALE_GRANULARITY,
                                            ],
                                            a_shared[producer_stage, producer_ks, :, :],
                                            barrier=stage_barriers[producer_stage],
                                        )
                                        for producer_n_block in T.serial(num_tma_n_blocks):
                                            T.tma_copy(
                                                l1_weight[
                                                    producer_expert,
                                                    producer_n * block_n
                                                    + producer_n_block * tma_block_n : producer_n * block_n
                                                    + (producer_n_block + 1) * tma_block_n,
                                                    producer_sf_k * SCALE_GRANULARITY : (producer_sf_k + 1) * SCALE_GRANULARITY,
                                                ],
                                                b_shared[
                                                    producer_stage,
                                                    producer_ks,
                                                    producer_n_block
                                                    * tma_block_n : (producer_n_block + 1) * tma_block_n,
                                                    :,
                                                ],
                                                barrier=stage_barriers[producer_stage],
                                            )
                                    T.mbarrier_arrive(stage_barriers[producer_stage])
                                producer_step += num_k_blocks

            else:
                T.inc_max_nreg(math_registers)
                # WG1+ consume TMA stages, run WGMMA, and emit quantized L1 rows.
                partial = T.alloc_fragment((block_m, block_n), T.float32)
                # FP32 running sum: a BF16 accumulator would need a quarter-rate
                # F2FP per element pair to narrow the WGMMA output every k-step,
                # and it costs an order of magnitude of accuracy (1.7e-4 -> 1.4e-5).
                accum = T.alloc_fragment((block_m, block_n), T.float32)
                gate = T.alloc_fragment((block_m, block_n // 2), T.float32)
                up = T.alloc_fragment((block_m, block_n // 2), T.float32)
                gate_grouped = T.reshape(gate, (block_m, num_output_scale_groups, SCALE_GRANULARITY))
                amax = T.alloc_fragment((block_m, num_output_scale_groups), T.float32)
                scale = T.alloc_fragment((block_m, num_output_scale_groups), T.float32)
                act_scale = T.alloc_fragment((block_m,), T.float32)
                weight_scale = T.alloc_local((2 * num_output_scale_groups,), T.float32)
                consumer_step = T.alloc_var(T.int32, init=0)
                for consumer_task in T.serial(bid, num_m_tasks[0] * num_n_blocks, num_sms):
                    consumer_n = consumer_task % num_n_blocks
                    consumer_m_task = m_tasks[consumer_task // num_n_blocks]
                    consumer_m = consumer_m_task % num_m_blocks
                    consumer_expert = consumer_m_task // num_m_blocks
                    if consumer_n * block_n < l1_n:
                            T.clear(partial)
                            T.clear(accum)
                            for consumer_k in T.serial(num_k_blocks):
                                consumer_stage = (consumer_step + consumer_k) % pipeline_stages
                                consumer_phase = ((consumer_step + consumer_k) // pipeline_stages) & 1
                                T.mbarrier_wait_parity(stage_barriers[consumer_stage], consumer_phase)
                                # One TMA stage spans num_k_sub scale groups; WGMMA and promotion
                                # still run per SCALE_GRANULARITY so the per-128 scales stay exact.
                                for consumer_ks in T.unroll(num_k_sub):
                                    consumer_sf_k = consumer_k * num_k_sub + consumer_ks
                                    for scale_group in T.serial(num_output_scale_groups):
                                        weight_scale[2 * scale_group] = l1_weight_sf[consumer_expert, consumer_n * num_output_scale_groups + scale_group, consumer_sf_k]
                                        weight_scale[2 * scale_group + 1] = l1_weight_sf[consumer_expert, num_l1_scale_groups + consumer_n * num_output_scale_groups + scale_group, consumer_sf_k]
                                    for i in T.Parallel(block_m):
                                        act_scale[i] = recv_x_sf[consumer_expert, consumer_m * block_m + i, consumer_sf_k]
                                    T.gemm(
                                        a_shared[consumer_stage, consumer_ks, :, :],
                                        b_shared[consumer_stage, consumer_ks, :, :],
                                        partial, transpose_B=True, clear_accum=True)
                                    for i, j in T.Parallel(block_m, block_n):
                                        accum[i, j] = partial[i, j] * (
                                            act_scale[i] * weight_scale[2 * (j // (2 * SCALE_GRANULARITY)) + (j % 16) // 8]
                                        ) + accum[i, j]
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
                            T.reduce_absmax(gate_grouped, amax, dim=2)
                            for i, scale_group in T.Parallel(block_m, num_output_scale_groups):
                                scale[i, scale_group] = T.max(amax[i, scale_group], 1e-4) / FP8_MAX
                                l2_x_sf[
                                    consumer_expert,
                                    consumer_m * block_m + i,
                                    consumer_n * num_output_scale_groups + scale_group,
                                ] = scale[i, scale_group]
                            for i, j in T.Parallel(block_m, block_n // 2):
                                gate[i, j] = T.clamp(gate[i, j] / scale[i, j // SCALE_GRANULARITY], -FP8_MAX, FP8_MAX)
                            T.copy(gate, out_shared)
                            T.copy(
                                out_shared,
                                l2_x[
                                    consumer_expert,
                                    consumer_m * block_m,
                                    consumer_n * (block_n // 2),
                                ],
                            )

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
    threads: int = 384,
    pipeline_stages: int = 3,
    num_experts_per_wave: int | None = None,
    use_put_warp_scatter: bool = False,
):
    num_experts_per_wave = num_experts_per_wave or num_experts_per_rank
    assert num_experts_per_rank % num_experts_per_wave == 0
    num_m_blocks = T.ceildiv(capacity, block_m)
    num_n_blocks = T.ceildiv(hidden, block_n)
    num_k_blocks = intermediate_hidden // block_k
    reduce_block_m = 8
    reduce_block_h = 128
    num_reduce_n_blocks = T.ceildiv(hidden, reduce_block_h)
    num_reduce_m_blocks = T.ceildiv(num_tokens, reduce_block_m)
    num_reduce_tiles = num_reduce_n_blocks * num_reduce_m_blocks
    # WG0 is phase-specialized: warps 0-1 are idle during GEMM, while warps
    # 2-3 issue TMA. WG1-2 run WGMMA and direct scatter; after the grid
    # barrier, all WG0 warps reduce the top-k slots.
    warp_size = 32
    warpgroup_size = 128
    reduce_warps = 4
    producer_begin_warp = 2
    producer_warps = 2
    math_begin_warp = 4
    math_warpgroups = 2
    reduce_threads = reduce_warps * warp_size
    producer_begin = producer_begin_warp * warp_size
    producer_threads = producer_warps * warp_size
    producer_end = producer_begin + producer_threads
    math_begin = math_begin_warp * warp_size
    num_math_threads = math_warpgroups * warpgroup_size
    assert producer_end == math_begin == reduce_threads
    assert block_m == 64
    assert block_n % (math_warpgroups * 8) == 0
    assert threads == math_begin + num_math_threads

    @T.prim_func
    def main(
        a: T.Tensor((num_experts_per_rank, capacity, intermediate_hidden), T.float8_e4m3fn),
        b: T.Tensor((num_experts_per_rank, hidden, intermediate_hidden), T.float8_e4m3fn),
        a_sf: T.Tensor((num_experts_per_rank, capacity, intermediate_hidden // SCALE_GRANULARITY), T.float32),
        b_sf: T.Tensor((num_experts_per_rank, hidden // SCALE_GRANULARITY, intermediate_hidden // SCALE_GRANULARITY), T.float32),
        recv_counts: T.Tensor((num_experts_per_rank,), T.int32),
        m_tasks: T.Tensor((num_experts_per_rank * num_m_blocks,), T.int32),
        num_m_tasks: T.Tensor((1,), T.int32),
        src_ranks: T.Tensor((num_experts_per_rank, capacity), T.int32),
        src_tokens: T.Tensor((num_experts_per_rank, capacity), T.int32),
        src_topk: T.Tensor((num_experts_per_rank, capacity), T.int32),
        combine: T.Tensor((num_tokens, num_topk, hidden), T.bfloat16),
        barrier: T.Tensor((num_ranks,), T.int32),
        out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        assert capacity > 0
        with T.Kernel(num_sms, threads=threads) as bid:
            tx = T.get_thread_binding()
            a_shared = T.alloc_shared((pipeline_stages, block_m, block_k), T.float8_e4m3fn)
            b_shared = T.alloc_shared((pipeline_stages, block_n, block_k), T.float8_e4m3fn)
            a_sf_shared = T.alloc_shared((pipeline_stages, block_m), T.float32)
            scatter_shared = T.alloc_shared((block_m, block_n), T.bfloat16)
            reduce_shared = T.alloc_shared((reduce_block_m, reduce_block_h), T.bfloat16)
            stage_barriers = T.alloc_barrier([producer_threads] * pipeline_stages + [num_math_threads] * pipeline_stages)

            if tx < math_begin:
                T.dec_max_nreg(48)
            else:
                T.inc_max_nreg(208)

            if tx >= producer_begin and tx < producer_end:
                # WG0 warps 2-3 keep the L2 TMA stages filled.
                producer_step = T.alloc_var(T.int32, init=0)
                for producer_task in T.serial(bid, num_m_tasks[0] * num_n_blocks, num_sms):
                    producer_n = producer_task % num_n_blocks
                    producer_m_task = m_tasks[producer_task // num_n_blocks]
                    producer_m = producer_m_task % num_m_blocks
                    producer_expert = producer_m_task // num_m_blocks
                    if (
                        producer_expert < num_experts_per_rank
                        and producer_n * block_n < hidden
                        and num_k_blocks * block_k == intermediate_hidden
                    ):
                            for producer_k in T.serial(num_k_blocks):
                                producer_stage = (producer_step + producer_k) % pipeline_stages
                                producer_phase = ((producer_step + producer_k) // pipeline_stages) & 1
                                T.mbarrier_wait_parity(stage_barriers[pipeline_stages + producer_stage], producer_phase ^ 1)
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
                                a_sf_shared[producer_stage, tx - producer_begin] = a_sf[producer_expert, producer_m * block_m + tx - producer_begin, producer_k]
                                T.mbarrier_arrive(stage_barriers[producer_stage])
                            producer_step += num_k_blocks

            elif tx >= math_begin:
                # WG1-2 run WGMMA and scatter their BF16 column pairs remotely.
                partial = T.alloc_fragment((block_m, block_n), T.float32)
                accum = T.alloc_fragment((block_m, block_n), T.bfloat16)
                act_scale = T.alloc_fragment((block_m,), T.float32)
                weight_scale = T.alloc_local((2,), T.float32)
                consumer_step = T.alloc_var(T.int32, init=0)
                scatter_dst_rank = T.alloc_var(T.int32, init=0)
                scatter_dst_token = T.alloc_var(T.int32, init=0)
                scatter_dst_topk = T.alloc_var(T.int32, init=0)

                for consumer_task in T.serial(bid, num_m_tasks[0] * num_n_blocks, num_sms):
                    consumer_n = consumer_task % num_n_blocks
                    consumer_m_task = m_tasks[consumer_task // num_n_blocks]
                    consumer_m = consumer_m_task % num_m_blocks
                    consumer_expert = consumer_m_task // num_m_blocks
                    if (
                        consumer_expert < num_experts_per_rank
                        and consumer_n * block_n < hidden
                        and num_k_blocks * block_k == intermediate_hidden
                    ):
                            T.clear(partial)
                            T.clear(accum)
                            for consumer_k in T.serial(num_k_blocks):
                                consumer_stage = (consumer_step + consumer_k) % pipeline_stages
                                consumer_phase = ((consumer_step + consumer_k) // pipeline_stages) & 1
                                T.mbarrier_wait_parity(stage_barriers[consumer_stage], consumer_phase)
                                T.gemm(a_shared[consumer_stage, :, :], b_shared[consumer_stage, :, :], partial, transpose_B=True, clear_accum=True)
                                for i in T.Parallel(block_m):
                                    act_scale[i] = a_sf_shared[consumer_stage, i]
                                weight_scale[0] = b_sf[consumer_expert, consumer_n * 2, consumer_k]
                                weight_scale[1] = b_sf[consumer_expert, consumer_n * 2 + 1, consumer_k]
                                for i, j in T.Parallel(block_m, block_n):
                                    accum[i, j] = T.cast(partial[i, j], T.bfloat16) * T.cast(
                                        act_scale[i] * weight_scale[j // 128], T.bfloat16
                                    ) + accum[i, j]
                                T.mbarrier_arrive(stage_barriers[pipeline_stages + consumer_stage])
                            consumer_step += num_k_blocks
                            if use_put_warp_scatter:
                                # Stage the complete tile once, then let each math warp
                                # scatter eight rows with aligned 16-byte remote stores.
                                T.copy(accum, scatter_shared)
                                T.sync_threads(4, num_math_threads)
                                scatter_warp = (tx - math_begin) // warp_size
                                for row_in_warp in T.serial(block_m // (num_math_threads // warp_size)):
                                    row = scatter_warp * (block_m // (num_math_threads // warp_size)) + row_in_warp
                                    pool_row = consumer_m * block_m + row
                                    if pool_row < recv_counts[consumer_expert]:
                                        if tx % warp_size == 0:
                                            scatter_dst_rank = src_ranks[consumer_expert, pool_row]
                                            scatter_dst_token = src_tokens[consumer_expert, pool_row]
                                            scatter_dst_topk = src_topk[consumer_expert, pool_row]
                                        dst_rank = T.shfl_sync(scatter_dst_rank, 0)
                                        dst_token = T.shfl_sync(scatter_dst_token, 0)
                                        dst_topk = T.shfl_sync(scatter_dst_topk, 0)
                                        if (
                                            dst_rank >= 0
                                            and dst_rank < num_ranks
                                            and dst_token >= 0
                                            and dst_token < num_tokens
                                            and dst_topk >= 0
                                            and dst_topk < num_topk
                                        ):
                                            T.put_warp(
                                                T.address_of(scatter_shared[row, 0]),
                                                T.address_of(combine[dst_token, dst_topk, consumer_n * block_n]),
                                                block_n,
                                                dst_pe=dst_rank,
                                                unroll_factor=1,
                                            )
                                T.sync_threads(4, num_math_threads)
                            else:
                                # Direct scatter maps fragment owners to packed BF16 stores.
                                scatter_math_thread = tx - math_begin
                                scatter_wg = scatter_math_thread // warpgroup_size
                                scatter_warp_in_wg = (scatter_math_thread % warpgroup_size) // warp_size
                                scatter_lane = scatter_math_thread % warp_size
                                for scatter_row_half in T.serial(2):
                                    row = scatter_warp_in_wg * 16 + scatter_row_half * 8 + scatter_lane // 4
                                    pool_row = consumer_m * block_m + row
                                    if pool_row < recv_counts[consumer_expert]:
                                        scatter_dst_rank = src_ranks[consumer_expert, pool_row]
                                        scatter_dst_token = src_tokens[consumer_expert, pool_row]
                                        scatter_dst_topk = src_topk[consumer_expert, pool_row]
                                        if (
                                            scatter_dst_rank >= 0
                                            and scatter_dst_rank < num_ranks
                                            and scatter_dst_token >= 0
                                            and scatter_dst_token < num_tokens
                                            and scatter_dst_topk >= 0
                                            and scatter_dst_topk < num_topk
                                        ):
                                            for scatter_col_chunk in T.serial(block_n // math_warpgroups // 8):
                                                scatter_col = scatter_wg * (block_n // math_warpgroups) + scatter_col_chunk * 8 + (scatter_lane % 4) * 2
                                                scatter_value_lo = T.alloc_var(T.uint16, init=T.reinterpret(accum[row, scatter_col], T.uint16))
                                                scatter_value_hi = T.alloc_var(T.uint16, init=T.reinterpret(accum[row, scatter_col + 1], T.uint16))
                                                scatter_value = T.alloc_var(T.uint32, init=T.cast(scatter_value_lo, T.uint32) | (T.cast(scatter_value_hi, T.uint32) << 16))
                                                T.st(combine[scatter_dst_token, scatter_dst_topk, consumer_n * block_n + scatter_col], scatter_value, dst_pe=scatter_dst_rank)

            T.fence_sys()
            T.sync_grid()
            if bid == 0:
                T.barrier_blocks(barrier[0])
            T.sync_grid()

            if tx < reduce_threads:
                # After every remote scatter is visible, WG0 reduces top-k into out.
                reduce_accum = T.alloc_fragment((reduce_block_m, reduce_block_h), T.float32)
                for reduce_tile in T.serial(bid, num_reduce_tiles, num_sms):
                    reduce_n = reduce_tile % num_reduce_n_blocks
                    reduce_m = reduce_tile // num_reduce_n_blocks
                    T.clear(reduce_accum)
                    for topk_slot in T.serial(num_topk):
                        for i, j in T.Parallel(reduce_block_m, reduce_block_h):
                            if reduce_m * reduce_block_m + i < num_tokens:
                                reduce_accum[i, j] += combine[reduce_m * reduce_block_m + i, topk_slot, reduce_n * reduce_block_h + j]
                    T.copy(reduce_accum, reduce_shared)
                    T.copy(
                        reduce_shared,
                        out[
                            reduce_m * reduce_block_m,
                            reduce_n * reduce_block_h,
                        ],
                    )
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
    weight_scale_bytes = num_experts_per_rank * ((2 * intermediate_hidden // 128) * (hidden // 128) + (hidden // 128) * (intermediate_hidden // 128)) * fp32
    pool_bytes = (
        num_experts_per_rank
        * capacity
        * (
            hidden * fp8
            + (hidden // 128) * fp32
            + 4 * i32
            + intermediate_hidden * fp8
            + (intermediate_hidden // 128) * fp32
        )
    )
    input_bytes = num_tokens * (
        hidden * fp8
        + (hidden // 128) * fp32
        + num_topk * (3 * i32 + fp32)
        + (num_topk + 1) * hidden * bf16
    )
    total_bytes = weight_bytes + weight_scale_bytes + pool_bytes + input_bytes + 2**27
    return (total_bytes + 2**20 - 1) // 2**20 * 2**20


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
    model_name, model = resolve_model_config(args)
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
    average_recv = (num_tokens * num_local_ranks * num_topk + num_experts - 1) // num_experts
    capacity = args.capacity if args.capacity is not None else (max(average_recv * 2, 64) + 63) // 64 * 64
    assert capacity >= 64 and capacity % 64 == 0

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert rank == local_rank and num_ranks == num_local_ranks
    num_sms = args.num_sms or torch.cuda.get_device_properties(local_rank).multi_processor_count
    allocator = get_allocator(
        size=_allocator_size_bytes(num_tokens, hidden, intermediate_hidden, num_experts_per_rank, num_topk, capacity),
        device=f"cuda:{local_rank}",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_local_ranks,
        group=group,
        use_vmm=True,
    )

    shape_family, l1_config, l2_config = select_manual_warp_configs(hidden, intermediate_hidden, num_tokens, num_topk, num_experts_per_rank, num_sms)
    if args.l1_block_k is not None:
        l1_config["block_k"] = args.l1_block_k
    if args.l1_stages is not None:
        l1_config["pipeline_stages"] = args.l1_stages
    for phase, config in (("l1", l1_config), ("l2", l2_config)):
        requested = getattr(args, f"{phase}_experts_per_wave", None)
        if requested is not None:
            assert requested > 0 and num_experts_per_rank % requested == 0
            config["num_experts_per_wave"] = requested
    l2_scatter = getattr(args, "l2_scatter", "auto")
    use_put_warp_scatter = l2_scatter == "warp" or (l2_scatter == "auto" and num_tokens >= 256)
    kernel_specs = [
        fused_l1_swiglu_manual_warp_kernel(
            num_tokens, hidden, 2 * intermediate_hidden, num_experts, num_topk, num_ranks, capacity, num_sms,
            activation_clamp=activation_clamp,
            frontend_regs_override=args.l1_frontend_regs,
            math_regs_override=args.l1_math_regs, **l1_config,
        ),
        fused_l2_scatter_reduce_manual_warp_kernel(
            num_tokens, hidden, intermediate_hidden, num_experts_per_rank, num_topk, num_ranks, capacity, num_sms, **l2_config,
            use_put_warp_scatter=use_put_warp_scatter,
        ),
    ]
    kernels = [tilelang.compile(spec, compile_once=True, compile_group=group) for spec in kernel_specs]
    for kernel in kernels:
        kernel.initialize(allocator=allocator)
    fused_l1, fused_l2 = kernels

    if local_rank == 0 and args.print_source:
        for kernel in kernels:
            print(kernel.get_kernel_source())

    torch.manual_seed(args.seed + local_rank)
    x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    x_fp8_src, x_sf_src = per_token_cast_to_fp8(x_bf16)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda")
    topk_weights_src, topk_idx_src = torch.topk(scores, num_topk, dim=-1, sorted=False)
    topk_idx_src = topk_idx_src.to(torch.int32)

    l1_bf16 = torch.randn((num_experts_per_rank, 2 * intermediate_hidden, hidden), dtype=torch.bfloat16, device="cuda") * 0.05
    l2_bf16 = torch.randn((num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device="cuda") * 0.05
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
    l1_fp8_kernel = interleave_gate_up_weights(l1_fp8_src)
    l1_fp8 = allocator_tensor(l1_fp8_kernel.shape, l1_fp8_kernel.dtype, allocator=allocator).copy_(l1_fp8_kernel)
    del l1_fp8_kernel
    l1_sf = allocator_tensor(l1_sf_src.shape, l1_sf_src.dtype, allocator=allocator).copy_(l1_sf_src)
    l2_fp8 = allocator_tensor(l2_fp8_src.shape, l2_fp8_src.dtype, allocator=allocator).copy_(l2_fp8_src)
    l2_sf = allocator_tensor(l2_sf_src.shape, l2_sf_src.dtype, allocator=allocator).copy_(l2_sf_src)

    route_counts = allocator_tensor((num_ranks, num_experts), torch.int32, allocator=allocator)
    recv_counts = allocator_tensor((num_experts_per_rank,), torch.int32, allocator=allocator)
    recv_x = allocator_tensor((num_experts_per_rank, capacity, hidden), torch.float8_e4m3fn, allocator=allocator)
    recv_x_sf = allocator_tensor((num_experts_per_rank, capacity, hidden // SCALE_GRANULARITY), torch.float32, allocator=allocator)
    recv_weights = allocator_tensor((num_experts_per_rank, capacity), torch.float32, allocator=allocator)
    src_ranks = allocator_tensor((num_experts_per_rank, capacity), torch.int32, allocator=allocator)
    src_tokens = allocator_tensor((num_experts_per_rank, capacity), torch.int32, allocator=allocator)
    src_topk = allocator_tensor((num_experts_per_rank, capacity), torch.int32, allocator=allocator)
    route_slots = allocator_tensor((num_tokens, num_topk), torch.int32, allocator=allocator)
    arrivals = allocator_tensor((num_experts_per_rank, (capacity + 63) // 64), torch.uint32, allocator=allocator)
    m_tasks = allocator_tensor((num_experts_per_rank * ((capacity + 63) // 64),), torch.int32, allocator=allocator)
    num_m_tasks = allocator_tensor((1,), torch.int32, allocator=allocator)
    l2_x = allocator_tensor((num_experts_per_rank, capacity, intermediate_hidden), torch.float8_e4m3fn, allocator=allocator)
    l2_x_sf = allocator_tensor((num_experts_per_rank, capacity, intermediate_hidden // SCALE_GRANULARITY), torch.float32, allocator=allocator)
    combine = allocator_tensor((num_tokens, num_topk, hidden), torch.bfloat16, allocator=allocator)
    out = allocator_tensor((num_tokens, hidden), torch.bfloat16, allocator=allocator)

    def reset_state():
        route_counts.zero_()
        barrier.zero_()
        recv_counts.zero_()
        arrivals.zero_()
        recv_x.zero_()
        recv_x_sf.zero_()
        recv_weights.zero_()
        src_ranks.fill_(-1)
        combine.zero_()
        torch.cuda.synchronize()
        dist.barrier(group=group)

    def run_pipeline(check_capacity: bool = False):
        fused_l1(
            x, x_sf, topk_idx, topk_weights, route_counts, recv_counts, route_slots, arrivals,
            m_tasks, num_m_tasks, recv_x, recv_x_sf, recv_weights, src_ranks, src_tokens, src_topk,
            l1_fp8, l1_sf, l2_x, l2_x_sf, barrier,
        )
        if check_capacity:
            local_max = recv_counts.max()
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=group)
            assert local_max.item() <= capacity, f"expert capacity {capacity} is smaller than received routes {local_max.item()}"
        fused_l2(
            l2_x, l2_fp8, l2_x_sf, l2_sf, recv_counts, m_tasks, num_m_tasks, src_ranks,
            src_tokens, src_topk, combine, barrier, out,
        )
        return out

    reset_state()
    actual = run_pipeline(check_capacity=True)
    torch.cuda.synchronize()
    dist.barrier(group=group)

    if args.check:
        expected = torch_reference(
            x_fp8_src, x_sf_src, topk_idx_src, topk_weights_src, l1_fp8_src,
            l1_sf_src, l2_fp8_src, l2_sf_src, group, activation_clamp,
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
        latency = do_bench(run_pipeline, warmup=0, rep=args.rep, post_fn=reset_state, group=group)
        if local_rank == 0:
            print(
                f"tilescale sm90 fp8 mega moe: model={model_name} family={shape_family} "
                f"M={num_tokens} H={hidden} IH={intermediate_hidden} E={num_experts} "
                f"topk={num_topk} capacity={capacity} "
                f"epw={l1_config['num_experts_per_wave']}/{l2_config['num_experts_per_wave']} "
                f"l2_scatter={'warp' if use_put_warp_scatter else 'direct'} "
                f"latency={latency * 1000:.1f} us"
            )

    if args.profile_phases > 0:
        reset_state()
        for _ in range(args.warmup):
            run_pipeline()
            reset_state()

        samples = []
        for _ in range(args.profile_phases):
            dist.barrier(group=group)
            events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            events[0].record()
            fused_l1(
                x, x_sf, topk_idx, topk_weights, route_counts, recv_counts, route_slots, arrivals,
                m_tasks, num_m_tasks, recv_x, recv_x_sf, recv_weights, src_ranks, src_tokens, src_topk,
                l1_fp8, l1_sf, l2_x, l2_x_sf, barrier,
            )
            events[1].record()
            fused_l2(
                l2_x, l2_fp8, l2_x_sf, l2_sf, recv_counts, m_tasks, num_m_tasks, src_ranks,
                src_tokens, src_topk, combine, barrier, out,
            )
            events[2].record()
            events[2].synchronize()
            local = torch.tensor(
                [events[0].elapsed_time(events[1]), events[1].elapsed_time(events[2])],
                dtype=torch.float32,
                device="cuda",
            )
            gathered = [torch.empty_like(local) for _ in range(num_ranks)]
            dist.all_gather(gathered, local, group=group)
            if local_rank == 0:
                samples.append(torch.stack(gathered).cpu())
            reset_state()

        if local_rank == 0:
            stacked = torch.stack(samples)
            max_rank_median = stacked.max(dim=1).values.median(dim=0).values * 1000
            rank_medians = stacked.median(dim=0).values * 1000
            print(
                f"phase profile: samples={args.profile_phases} max-rank median "
                f"l1={max_rank_median[0]:.1f} us l2={max_rank_median[1]:.1f} us "
                f"total={max_rank_median[0] + max_rank_median[1]:.1f} us"
            )
            for phase_idx, phase_name in enumerate(("l1", "l2")):
                rank_values = ", ".join(
                    f"r{r}={rank_medians[r, phase_idx]:.1f}" for r in range(num_ranks)
                )
                print(f"phase profile {phase_name} rank medians (us): {rank_values}")

    allocator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--model-config", choices=tuple(MODEL_CONFIGS), default="smoke")
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--intermediate-hidden", type=int, default=None)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--num-topk", type=int, default=None)
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=None)
    parser.add_argument("--l1-experts-per-wave", type=int, default=None)
    parser.add_argument("--l2-experts-per-wave", type=int, default=None)
    parser.add_argument("--l2-scatter", choices=("auto", "direct", "warp"), default="auto")
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diff-tol", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--profile-phases", type=int, default=0)
    parser.add_argument("--num-sms", type=int, default=None)
    parser.add_argument("--l1-block-k", type=int, default=None)
    parser.add_argument("--l1-frontend-regs", type=int, default=None)
    parser.add_argument("--l1-math-regs", type=int, default=None)
    parser.add_argument("--l1-stages", type=int, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-source", action="store_true")
    args = parser.parse_args()
    torch.multiprocessing.spawn(main, args=(args.num_processes, args), nprocs=args.num_processes)
