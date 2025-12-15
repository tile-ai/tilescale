# For internode only
# This op is distributed

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add parent folder to path

import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
from tilelang.distributed.utils import init_dist
from utils import Config, gen_inputs  # noqa: F403
from argparse import ArgumentParser
import math

from get_dispatch_layout import get_dispatch_layout

# tilelang.disable_cache()
os.environ['NCCL_DEBUG'] = 'WARN'  # silence NCCL log


NUM_TIMEOUT_CYCLES = 200000000000
NUM_MAX_NVL_PEERS = 8

def align_up(a, b):
    return math.ceil(a / b) * b

def ceil_div(a, b):
    return math.ceil(a / b)

def translate_dst_rdma_rank(dst_rdma_rank, nvl_rank):
    return dst_rdma_rank



@tilelang.jit
def cached_notify_combine_kernel(
    kNumTMABytesPerWarp: int,
    num_channels: int,
):
    threads = max(128, 32 * num_channels)

    @T.macro
    def nvshmem_sync_with_same_gpu_idx():
        # TODO: support low latency mode
        T.nvshmem_sync_all()


    # FIXME:
    @T.prim_func
    def cached_notify_combine_main(
        rdma_clean_offset: T.int32,
        rdma_num_int_clean: T.int32,
        nvl_clean_offset: T.int32,
        nvl_num_int_clean: T.int32,
        combined_rdma_head: T.Tensor([], dtype="int32"), # FIXME
        num_combined_tokens: T.int32,
        num_channels: T.int32,
        rdma_channel_prefix_matrix: T.Tensor([], dtype="int32"), # FIXME
        rdma_rank_prefix_sum: T.Tensor([], dtype="int32"), # FIXME
        combined_nvl_head: T.Tensor([], dtype="int32"), # FIXME
        rdma_buffer_ptr: T.Tensor([], "uint8"),
        nvl_buffer_ptrs: T.Tensor([], "uint8"),
        barrier_signal_ptrs: T.Tensor([], "int32"),
        rank: T.int32,
        num_ranks: T.int32,
        is_cached_dispatch: T.bool,

    ):
        with T.Kernel(num_channels * 2, threads=threads) as bx:
            sm_id = bx
            thread_id = T.get_thread_binding()
            num_threads = threads
            num_warps = num_threads // 32
            lane_id = thread_id % 32

            nvl_rank = rank % NUM_MAX_NVL_PEERS

            num_rdma_ranks = num_ranks // NUM_MAX_NVL_PEERS

            rdma_rank = rank // NUM_MAX_NVL_PEERS

            if sm_id == 0:
                qps_per_rdma_rank = T.ibgda_get_state("num_rc_per_pe") * T.ibgda_get_state("num_devices_initialized")
                for i in T.serial(thread_id, qps_per_rdma_rank * (num_rdma_ranks - 1), num_threads):
                    dst_rdma_rank = (i // qps_per_rdma_rank + rdma_rank + 1) % num_rdma_ranks
                    qp_id = i % qps_per_rdma_rank
                    T.nvshmemi_ibgda_quiet(
                        translate_dst_rdma_rank(dst_rdma_rank, nvl_rank),
                        qp_id
                    )
                # TODO: Check whether it's -> _syncthreads()
                T.sync_threads()

                if thread_id == 32:
                    nvshmem_sync_with_same_gpu_idx()
                
                T.barrier_blocks()

                
    return cached_notify_combine_main


def cached_notify_combine(
    num_ranks,
    num_sms,
    ##### symm buffers #####
    send_head: torch.Tensor,
    channel_head_idx: torch.Tensor,
    channel_tail_idx: torch.Tensor,
    barrier_signal: torch.Tensor,    
    allocator
):
    kernel = cached_notify_combine_kernel(num_ranks, num_sms)
    kernel.initialize(allocator=allocator)

    kernel(send_head, channel_head_idx, channel_tail_idx, barrier_signal)

"""
0 -> nvlsender
1 -> nvlandrdmaforwarder
2 -> rdmareceiver
3 -> coordinator
"""
@tilelang.jit
def combine_kernel(
    rank, num_ranks,
    num_combined_tokens,
    num_max_rdma_chunked_send_tokens,
    num_max_rdma_chunked_recv_tokens,
    num_max_nvl_chunked_send_tokens,
    num_max_nvl_chunked_recv_tokens,
    hidden, 
    num_topk, 
    num_channels,
    dtype: str = 'bfloat16',
    num_nvl_bytes: int = 2e9,
    num_rdma_bytes: int = 1e9,
):
    
    num_tokens = T.dynamic("num_tokens")
    MAX_INT = 2147483647
    NUM_WAIT_NANOSECONDS = 500
    size_of_int4 = 4 * 4
    size_of_float = 4
    size_of_source_meta = 8
    size_of_uint64 = 64 // 8
    kNumCombineForwarderWarps = 24
    NUM_MAX_NVL_PEERS = 8
    kNumTMABytesPerSenderWarp = 16384
    kNumTMABytesPerForwarderWarp = 9248

    smem_size = max(kNumTMABytesPerSenderWarp * NUM_MAX_NVL_PEERS, kNumTMABytesPerForwarderWarp * kNumCombineForwarderWarps)
    num_rdma_ranks = num_ranks // NUM_MAX_NVL_PEERS
    num_warps_per_forwarder = max(kNumCombineForwarderWarps // num_rdma_ranks, 1)
    num_forwarder_warps = num_rdma_ranks * num_warps_per_forwarder
    threads = (num_forwarder_warps + 1) * 32

    kNumRDMARanks = num_rdma_ranks
    kNumTopkRDMARanks = min(num_rdma_ranks, 8)
    kNumWarpsPerForwarder = max(1, kNumCombineForwarderWarps // kNumRDMARanks)
    kNumForwarders = kNumRDMARanks * kNumWarpsPerForwarder
    kNumRDMAReceivers = kNumForwarders - NUM_MAX_NVL_PEERS

    assert num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0
    assert num_max_nvl_chunked_recv_tokens // num_rdma_ranks > max(num_max_rdma_chunked_send_tokens, num_max_nvl_chunked_send_tokens)
    assert num_max_nvl_chunked_recv_tokens // num_rdma_ranks - num_warps_per_forwarder >= num_max_nvl_chunked_send_tokens
    assert num_max_rdma_chunked_send_tokens >= num_warps_per_forwarder


    assert dtype == "bfloat16"

    num_sms = num_channels * 2

    @T.macro
    def sync_forwarder_smem():
        T.bar_sync(0, (kNumForwarders + 1) * 32,)

    @T.macro
    def sync_rdma_receiver_smem():
        T.bar_sync(1, (kNumRDMAReceivers + 1) * 32)

    @T.macro
    def sync_large_warp(dst_rdma_rank: T.int32):
        if kNumWarpsPerForwarder == 1:
            T.sync_warp()
        else:
            T.bar_sync(dst_rdma_rank + 2, kNumWarpsPerForwarder * 32)

    @T.macro
    def combine_token():
        ...

    @T.prim_func
    def combine_main(
        # outputs
        combined_x: T.ptr,
        combined_topk_weights: T.ptr,

        # inputs
        is_combined_token_in_rank: T.ptr,
        x: T.ptr,
        topk_weights: T.ptr,
        bias_0: T.ptr,
        bias_1: T.ptr,
        combined_rdma_head: T.ptr,
        combined_nvl_head: T.ptr,
        src_meta: T.ptr,
        rdma_channel_prefix_matrix: T.ptr,
        rdma_rank_prefix_sum: T.ptr,
        gbl_channel_prefix_matrix: T.ptr,

        # nvl asymmetric memory
        buffer_ptrs: T.ptr,
        # rdma symmetric memory
        rdma_buffer_ptr: T.Tensor([num_rdma_bytes], "uint8"),
    ):
        with T.Kernel(num_sms, threads=threads) as bx:
            tx = T.get_thread_binding()
            sm_id = bx
            num_threads = threads
            num_warps = num_threads // 32
            thread_id = tx
            lane_id = tx % 32
            # num_channels already defined
            channel_id = sm_id // 2

            is_forwarder_sm = sm_id % 2 == 1

            T.device_assert(num_topk <= 32)
            # sizeof(int4) = 4 * 4 = 16 sizeof(bfloat16) = 2
            T.device_assert(hidden % (8) == 0)

            hidden_int4 = hidden // 8
            hidden_bytes = hidden_int4 * 16
            T.mbarrier_init()
            num_bytes_per_token = align_up(hidden_int4 * 16 + num_topk * 4 + 8, 16)

            rdma_rank = rank // NUM_MAX_NVL_PEERS
            nvl_rank = rank % NUM_MAX_NVL_PEERS

            warp_id = T.alloc_var("int32")

            warp_role = T.alloc_var("uint8") 

            warp_id = thread_id // 32

            if not is_forwarder_sm:
                # sender
                if warp_id < NUM_MAX_NVL_PEERS:
                    warp_role = 0 
                    warp_id = (warp_id + channel_id) % NUM_MAX_NVL_PEERS
                elif warp_id < kNumForwarders:
                    warp_role = 2
                    warp_id = warp_id - NUM_MAX_NVL_PEERS
                else:
                    warp_role = 3
                    warp_id = 0
            else:
                if warp_id < kNumForwarders:
                    warp_role = 1
                    warp_id = (warp_id + channel_id) % kNumForwarders
                else:
                    warp_role = 3
                    warp_id = 0

            T.device_assert(num_warps == kNumForwarders + 1)
            num_max_nvl_chunked_recv_tokens_per_rdma = num_max_nvl_chunked_recv_tokens // kNumRDMARanks
            smem_tma_buffer = T.alloc_shared([smem_size], "uint8")
            ## FIXME: need to check
            smem_buffer = smem_tma_buffer 


            combined_nvl_head_idx = T.alloc_var("int32", init=0)
            if warp_role == 0:
                dst_nvl_rank = warp_id

                tma_buffer = smem_tma_buffer[dst_nvl_rank * kNumTMABytesPerSenderWarp:
                                            (dst_nvl_rank + 1) * kNumTMABytesPerSenderWarp]
                
                tma_mbarrier = 1

                tma_phase = 0

                if T.elect_one_sync():
                    T.init_barrier_gpu(
                        tma_mbarrier,
                        1
                    )

                    T.fence_barrier_init()

                    T.device_assert(num_bytes_per_token + 8 <= kNumTMABytesPerSenderWarp)

                T.sync_warp()

                token_start_idx = T.alloc_var('int32')
                token_end_idx = T.alloc_var('int32')

                if lane_id < kNumRDMARanks:
                    prefix_idx = (lane_id * NUM_MAX_NVL_PEERS + dst_nvl_rank) * num_channels + channel_id

                    token_start_idx = gbl_channel_prefix_matrix[prefix_idx]

                    token_end_idx = T.if_then_else(
                        prefix_idx == num_channels * num_ranks - 1,
                        num_tokens,
                        gbl_channel_prefix_matrix[prefix_idx + 1]
                    )

                T.sync_warp()


                cached_channel_head_idx = T.alloc_var('int32')
                cached_channel_tail_idx = T.alloc_var('int32')

                T.device_assert(kNumRDMARanks <= 32)

                current_rdma_idx = T.alloc_var('int32')
                current_rdma_idx = channel_id % kNumRDMARanks

                while True:
                    if T.warp_all(
                        token_start_idx >= token_end_idx,
                        0xffffffff
                    ):
                        T.loop_break()
                    
                    is_lane_ready = T.alloc_var('bool')
                    is_lane_ready = False

                    start_time = T.alloc_var("int64")

                    start_time = T.get_clock()

                    while True:
                        num_used_slots = cached_channel_tail_idx - cached_channel_head_idx
                        is_lane_ready = lane_id < kNumRDMARanks and token_start_idx < token_end_idx and num_max_nvl_chunked_recv_tokens_per_rdma - num_used_slots >= num_max_nvl_chunked_send_tokens
                        if T.warp_any(
                            is_lane_ready,
                            0xffffffff
                        ):
                            T.loop_break()

                        if lane_id < kNumRDMARanks and token_start_idx < token_end_idx:
                            T.ld(
                                nvl_channel_head[] + lane_id,
                                cached_channel_head_idx,
                                sem="volatile",
                            )

                        if T.get_clock() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks:
                            # TODO: emit error message
                            T.device_assert(False)

                        for i in range(kNumRDMARanks):
                            current_rdma_idx = (current_rdma_idx + 1) % kNumRDMARanks
                            if T.warp_shfl(0xffffffff, (token_start_idx >= token_end_idx) or (not is_lane_ready), current_rdma_idx):
                                T.loop_continue()
                            # NOTE: 1. don't inline, don't denote this is pure.
                            # 2. token_idx need to cast to int64 from int32
                            token_idx: T.int64 = T.warp_shfl(0xffffffff, token_start_idx, current_rdma_idx)
                            num_tokens_in_chunk = T.warp_shfl(0xffffffff, min(num_max_nvl_chunked_send_tokens, token_end_idx - token_start_idx), current_rdma_idx)

                            ## send by chunk

                            for chunk_idx in range(num_tokens_in_chunk):
                                dst_slot_idx = T.alloc_var("int32", init=0)
                                if lane_id == current_rdma_idx:
                                    dst_slot_idx = (cached_channel_tail_idx) % num_max_nvl_chunked_recv_tokens_per_rdma
                                    cached_channel_tail_idx += 1
                                    dst_slot_idx = current_rdma_idx * num_max_nvl_chunked_recv_tokens_per_rdma + dst_slot_idx
                                
                                dst_slot_idx = T.shfl_sync(0xffffffff, dst_slot_idx, current_rdma_idx)

                                ## TODO: load data
                                shifed_buffer = nvl_channel_x.buffer() + dst_slot_idx * num_bytes_per_token
                                # shifted_x = x + token_idx * hidden_int4
                                # shifted_x = x[token_idx: token_idx, hidden]
                                T.tma_store_wait(0)

                                if T.elect_one_sync():
                                    # TODO: bind tma load 1d function
                                    T.tma_load_1d(
                                        T.address_of(tma_buffer),
                                        T.address_of(x[token_idx, hidden]),
                                        T.address_of(tma_mbarrier),
                                        hidden_bytes,
                                    )

                                    T.mbarrier_arrive_and_expect_tx(
                                        T.address_of(tma_mbarrier),
                                        hidden_bytes
                                    )
                                
                                T.sync_warp()
                                T.mbarrier_wait(
                                    T.address_of(tma_mbarrier),
                                    tma_phase
                                )

                                if lane_id == num_topk:
                                    T.ld(
                                        T.address_of(src_meta[token_idx, 0]),
                                        T.address_of(tma_buffer[hidden_bytes]),
                                        nc=True
                                    )

                                if lane_id < num_topk:
                                    T.ld(
                                        T.address_of(topk_weights[token_idx, lane_id]),
                                        T.address_of(tma_buffer[hidden_bytes + size_of_source_meta + lane_id * size_of_float]),
                                        nc=True
                                    )
                                
                                # TODO: add fence
                                T.tma_store_fence()
                                T.sync_warp()

                                if T.elect_one_sync():
                                    T.tma_store_1d(
                                        T.address_of(tma_buffer),
                                        T.address_of(shifted_x_buffers),
                                        num_bytes_per_token,
                                        False
                                    )
                                
                                token_idx += 1
                            
                            if lane_id == current_rdma_idx:
                                token_start_idx = token_idx

                        T.tma_store_wait(0)

                        T.sync_warp()

                        if lane_id < kNumRDMARanks and is_lane_ready:
                            T.st(
                                nvl_channel_tail, ## FIXME
                                cached_channel_tail_idx,
                                scope="sys",
                                sem="release"
                            )

            else:
                # combiner and coordinator
                ##### RDMA channel data
                rdma_channel_data_num_bytes = num_max_rdma_chunked_recv_tokens * num_bytes_per_token * 1
                rdma_channel_data_total_bytes = rdma_channel_data_num_bytes * kNumRDMARanks * num_channels * 2

                rdma_channel_data_send_ptr = T.address_of(rdma_buffer_ptr[rdma_channel_data_num_bytes * kNumRDMARanks * channel_id])
                rdma_channel_data_recv_ptr = T.address_of(rdma_buffer_ptr[rdma_channel_data_num_bytes * kNumRDMARanks * (channel_id + num_channels)])

                ##### RDMA head tail
                rdma_channel_head_num_bytes = 1 * size_of_uint64
                rdma_channel_head_total_bytes = rdma_channel_head_num_bytes * kNumRDMARanks * num_channels
                rdma_channel_head_send_ptr = T.address_of(rdma_buffer_ptr[rdma_channel_data_total_bytes + rdma_channel_head_num_bytes * kNumRDMARanks * channel_id])
                # rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels)

                ##### NOTE: rdma_buffer_ptr = rdma_buffer_ptr + rdma_channel_data_total_bytes + rdma_channel_head_total_bytes
                rdma_channel_tail_num_bytes = 1 * size_of_uint64
                rdma_channel_tail_total_bytes = rdma_channel_head_num_bytes * kNumRDMARanks * num_channels
                rdma_channel_tail_send_ptr = T.address_of(rdma_buffer_ptr[rdma_channel_data_total_bytes + rdma_channel_head_total_bytes + rdma_channel_head_num_bytes * kNumRDMARanks * channel_id])

                

                local_nvl_buffer = ...

                ## copy buffer_ptrs 
                nvl_buffers = ...

            
                nvl_channel_x = ...
                
                nvl_channel_head = ... # T.alloc_local([NUM_MAX_NVL_PEERS], "handle")
                nvl_channel_tail = ... # T.alloc_local([1], "handle")

                forwarder_nvl_head = T.alloc_shared([kNumForwarders, NUM_MAX_NVL_PEERS], "int32", scope="shared")
                forwarder_retired = T.alloc_shared([kNumForwarders], "bool", scope="shared")
                rdma_receiver_rdma_head = T.alloc_shared([kNumRDMAReceivers, kNumRDMARanks], "int32", scope="shared")
                rdma_receiver_retired = T.alloc_shared([kNumRDMAReceivers], "bool", scope="shared")

                if warp_role == 1:
                    dst_rdma_rank = warp_id // kNumWarpsPerForwarder
                    sub_warp_id = warp_id % kNumWarpsPerForwarder

                    
                    # TODO: how to impl?
                    send_buffer = ...

                    T.device_assert(kNumWarpsPerForwarder == 1 or kNumRDMARanks + 2 <= 16, "Barriers are not enough")

                    kNumStages = 2
                    kNumTMALoadBytes = size_of_int4 * 32
                    kNumTMABufferBytesPerStage = kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1) + 16
                    T.device_assert(kNumTMABufferBytesPerStage * kNumStages <= kNumTMABytesPerForwarderWarp, "TMA buffer is not larger enough")

                    smem_ptr = smem_buffer[warp_id * kNumStages * kNumTMABufferBytesPerStage]

                    """
                    auto tma_mbarrier = [=](const int& i) {
                        return reinterpret_cast<uint64_t*>(smem_ptr + i * kNumTMABufferBytesPerStage + kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1));
                    };

                    smem_buffer[warp_id * kNumStages * kNumTMABufferBytesPerStage + i * kNumTMABufferBytesPerStage + kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1)]
                    
                    """

                    tma_phase = T.alloc_local([kNumStages], "uint32")

                    for i in range(kNumStages):
                        tma_phase[i] = 0

                    if lane_id < kNumStages:
                        T.mbarrier_init(T.address_of(smem_buffer[warp_id * kNumStages * kNumTMABufferBytesPerStage + lane_id * kNumTMABufferBytesPerStage + kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1)]),
                            32)
                        T.fence_barrier_init()
                    T.sync_warp()

                    # FIXME: nvl advance

                    T.device_assert(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers")

                    if lane_id < NUM_MAX_NVL_PEERS:
                        forwarder_nvl_head[warp_id][lane_id] = 0
                    
                    if lane_id == 0:
                        forwarder_retired[warp_id] = False
                    
                    sync_forwarder_smem()

                    cached_nvl_channel_tail_idx = 0
                    num_tokens_to_combine = T.alloc_var("int32", init=rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id])
                    num_tokens_prefix = T.alloc_var("int32")
                    num_tokens_prefix = T.if_then_else(channel_id == 0, 0, rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1])
                    num_tokens_to_combine -= num_tokens_prefix
                    num_tokens_prefix += T.if_then_else(dst_rdma_rank == 0, 0, rdma_rank_prefix_sum[dst_rdma_rank - 1])
                    # int32 
                    combined_nvl_head_idx += num_tokens_prefix * NUM_MAX_NVL_PEERS 

                    
                    for token_start_idx in T.serial(0, num_tokens_to_combine, num_max_rdma_chunked_send_tokens):
                        token_end_idx = T.min(token_start_idx + num_max_rdma_chunked_send_tokens, num_tokens_to_combine)
                        num_chunked_tokens = token_end_idx - token_start_idx

                        start_time = T.get_clock()
                        while sub_warp_id == 0 and lane_id == 0:
                            num_used_slots = T.alloc_var("int32")

                            T.ld(
                               rdma_buffer_ptr[rdma_channel_data_total_bytes + rdma_channel_head_num_bytes * kNumRDMARanks * channel_id + dst_rdma_rank * rdma_channel_head_num_bytes],
                               num_used_slots,
                               sem="volatile"
                            )

                            num_used_slots = token_start_idx - num_used_slots
                            if num_max_rdma_chunked_recv_tokens - num_used_slots >= num_chunked_tokens:
                                T.loop_break()
                            
                            if T.get_clock() - start_time > NUM_TIMEOUT_CYCLES:
                                # trap
                                T.device_assert(False)
                        
                        sync_large_warp()

                        for token_idx in T.serial(token_start_idx + sub_warp_id, token_end_idx, kNumWarpsPerForwarder):
                            T.device_assert(kNumRDMARanks <= 32, "Invalid number of RDMA peers")
                            expected_head = T.alloc_var("int32", init=-1)

                            if lane_id < NUM_MAX_NVL_PEERS:
                                T.ld(
                                    combined_nvl_head[token_idx * NUM_MAX_NVL_PEERS + lane_id],
                                    expected_head,
                                    nc=True
                                )
                                forwarder_nvl_head[warp_id][lane_id] = T.if_then_else(expected_head < 0,  -expected_head - 1, expected_head)
                            start_time = T.get_clock()

                            while cached_nvl_channel_tail_idx <= expected_head:
                                T.ld(
                                    nvl_buffers, # FIXME
                                    cached_nvl_channel_tail_idx,
                                    sem='acquire',
                                    scope="sys"
                                )

                                if T.get_clock() - start_time > NUM_TIMEOUT_CYCLES and lane_id < NUM_MAX_NVL_PEERS:
                                    # trap
                                    T.device_assert(False)
                            combine_token()

                            if lane_id < NUM_MAX_NVL_PEERS:
                                forwarder_nvl_head[warp_id][lane_id] = T.if_then_else(
                                    expected_head < 0,
                                    -expected_head - 1,
                                    expected_head + 1
                                )

                        sync_large_warp()

                        if (sub_warp_id == kNumWarpsPerForwarder - 1):
                            if dst_rdma_rank != rdma_rank:
                                ## FIXME
                                ...
                            else:
                                T.fence_sys()
                            
                            T.sync_warp()

                            if T.elect_one_sync():
                                T.nvshmemi_ibgda_amo_nonfetch_add(
                                    T.address_of(rdma_buffer_ptr[rdma_channel_data_total_bytes + rdma_channel_head_total_bytes + rdma_channel_head_num_bytes * kNumRDMARanks * channel_id + rdma_rank * rdma_channel_tail_num_bytes]),
                                    num_chunked_tokens,
                                    translate_dst_rdma_rank(dst_rdma_rank, nvl_rank),
                                    channel_id,
                                    dst_rdma_rank == rdma_rank
                                )
                    T.sync_warp()
                    if T.elect_one_sync():
                        forwarder_retired[warp_id] = True
                elif warp_role == 2:
                    T.device_assert(kNumRDMARanks <= 32)
                    if lane_id < kNumRDMARanks:
                        rdma_receiver_rdma_head[warp_id][lane_id] = 0
                    if lane_id == 0:
                        rdma_receiver_retired[warp_id] = False
                    
                    sync_rdma_receiver_smem()

                    token_start_idx = T.alloc_var('int32')
                    token_end_idx = T.alloc_var('int32')

                    # get_channel_task_range
                    num_tokens_per_sm = ceil_div(num_combined_tokens, num_channels)
                    token_start_idx = T.min(num_tokens_per_sm * channel_id, num_combined_tokens)
                    token_end_idx = T.min(token_start_idx + num_tokens_per_sm, num_combined_tokens)
                    cached_channel_tail_idx = T.alloc_var('int32', init=0)

                    for i in T.serial(token_start_idx + warp_id, token_end_idx, kNumRDMAReceivers):
                        T.device_assert(kNumRDMARanks <= 32, "Invalid number of RDMA peers")
                        expected_head = T.alloc_var('int32', init=-1)

                        if lane_id < kNumRDMARanks:
                            T.ld(
                                combined_rdma_head[token_idx, lane_id],
                                expected_head,
                                nc=True
                            )
                            if expected_head < 0:
                                rdma_receiver_rdma_head[warp_id][lane_id] = -expected_head - 1
                            else:
                                rdma_receiver_rdma_head[warp_id][lane_id] = expected_head
                        start_time = T.get_clock()
                        while cached_channel_tail_idx <= expected_head:
                            T.ld(rdma_buffer_ptr[rdma_channel_data_total_bytes + 
                                                rdma_channel_head_total_bytes +
                                                rdma_channel_head_num_bytes * kNumRDMARanks * channel_id + 
                                                lane_id * rdma_channel_tail_num_bytes],
                                cached_channel_tail_idx,
                                scope="sys",
                                sem="acquire"
                            )

                            if T.get_clock() - start_time > NUM_TIMEOUT_CYCLES:
                                T.device_assert(False)

                        T.sync_warp()     

                        # TODO: combine token
                        combine_token()
                    T.sync_warp()
                    if T.elect_one_sync():
                        rdma_receiver_retired[warp_id] = True
                else:
                    if is_forwarder_sm:
                        sync_forwarder_smem()
                    else:
                        sync_rdma_receiver_smem()
                    num_warps_per_rdma_rank = kNumForwarders // kNumRDMARanks

                    last_rdma_head = T.alloc_var('int32', init=0)
                    last_nvl_head = T.alloc_local([kNumRDMARanks], "int32")
                    # initialize to zero
                    for i in range(kNumRDMARanks):
                        last_nvl_head[i] = 0

                    dst_rdma_rank = T.if_then_else(
                        lane_id < kNumRDMARanks,
                        lane_id,
                        0
                    )
                    dst_nvl_rank = T.if_then_else(
                        lane_id < NUM_MAX_NVL_PEERS,
                        lane_id,
                        0
                    )
                    T.device_assert(kNumCombineForwarderWarps <= 32, "Invalid number of forwarder warps")

                    while True:
                        if not is_forwarder_sm and T.warp_all(
                            lane_id >= kNumRDMAReceivers or rdma_receiver_retired[lane_id],
                            0xffffffff, 
                        ):
                            T.loop_break()
                        if (is_forwarder_sm and T.warp_all(
                            lane_id >= kNumForwarders or forwarder_retired[lane_id]),
                            0xffffffff, 
                        ):
                            T.loop_break()
                        min_head = T.alloc_var("int", init=MAX_INT)

                        #
                        if (not is_forwarder_sm):
                            for i in T.unroll(0, kNumRDMAReceivers):
                                if (not rdma_receiver_retired[i]):
                                    min_head = T.min(min_head, rdma_receiver_rdma_head[i][dst_rdma_rank])
                            if (min_head != MAX_INT and min_head >= last_rdma_head + num_max_rdma_chunked_send_tokens and
                                lane_id < kNumRDMARanks):
                                T.nvshmemi_ibgda_amo_nonfetch_add(
                                    rdma_buffer_ptr[rdma_channel_data_total_bytes + 
                                                    rdma_channel_head_num_bytes * kNumRDMARanks * channel_id + 
                                                    rdma_channel_head_num_bytes * rdma_rank],
                                    min_head - last_rdma_head,
                                    translate_dst_rdma_rank(dst_rdma_rank, nvl_rank),
                                    channel_id + num_channels,
                                    dst_rdma_rank == rdma_rank
                                )
                                last_rdma_head = min_head
                        else:
                            for i in T.unroll(0, kNumRDMARanks):
                                min_head = MAX_INT
                                for j in T.unroll(0, num_warps_per_rdma_rank):
                                    if not forwarder_retired[i * num_warps_per_rdma_rank + j]:
                                        min_head = T.min(min_head, forwarder_nvl_head[i * num_warps_per_rdma_rank + j][dst_nvl_rank])

                                if min_head != MAX_INT and min_head > last_nvl_head[i] and lane_id < NUM_MAX_NVL_PEERS:
                                    T.st(
                                        # FIXME: how to represent nvl_channel_head
                                        T.address_of(nvl_channel_head[dst_nvl_rank]) + i,
                                        min_head,
                                        scope="sys",
                                        sem="relaxed",
                                    )
                                    last_nvl_head[i] = min_head

                        T.nanosleep(NUM_WAIT_NANOSECONDS)
                            
    return combine_main
