# B200 allgather-GEMM with tcgen05 2-CTA MMA and NVSwitch multicast.
#
# The grid is split into compute SMs (persistent 2-CTA tcgen05 clusters that
# consume the gathered A local-shard-first) and comm SMs (TMA-load the local
# A shard to SMEM, TMA-store it into the multicast VA so one store lands on
# every rank, then publish per-256-row readiness with multimem.red).
#
# Layout: A_local (M_per_rank, K) bf16, B (N_per_rank, K) bf16 row-major,
# C (M, N_per_rank) = gathered_A @ B^T. Cluster tile 256x256x64, 5 stages,
# 192 threads: warps 0-3 epilogue, warp 4 TMA producer, warp 5 MMA (CTA 0).

import argparse

import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.distributed import init_dist
from tilelang import get_allocator

COMM_CHUNKS = 2


# ThreadSync would inject __syncthreads() inside the comm warps' guarded loop
# (divergent at the task tail -> deadlock); roles synchronize via mbarriers.
@tilelang.jit(compile_once=True, pass_configs={"tl.disable_thread_storage_sync": True})
def ag_gemm_tcgen5_kernel(M, N, K, num_ranks, group_size_m, num_stages):
    dtype = T.bfloat16
    accum_dtype = T.float32
    block_M = 128            # per-CTA rows; cluster covers 256
    block_N = 256            # cluster cols
    half_N = block_N // 2    # per-CTA slice of B
    block_K = 64
    store_block_N = 64
    comm_block_M = 256
    comm_block_K = 128

    M_per_rank = M // num_ranks
    N_per_rank = N // num_ranks
    sm_num = driver.get_num_sms()

    m_clusters_total = M // (2 * block_M)
    m_clusters_local = M_per_rank // (2 * block_M)
    n_blocks = N_per_rank // block_N
    k_blocks = K // block_K
    total_tiles = m_clusters_total * n_blocks
    local_tiles = m_clusters_local * n_blocks

    comm_k_blocks = K // comm_block_K
    comm_tasks_per_rank = m_clusters_local * comm_k_blocks

    def tile_coords(tile_id, local_rank):
        """Local-first scheduler at cluster (256-row) granularity.

        Local tiles use Super-M grouping for B reuse; remote tiles walk peer
        ranks round-robin inside each shard row so traffic is consumed in
        arrival-friendly order.
        """
        is_local = tile_id < local_tiles
        local_super_rows = (m_clusters_local // group_size_m) * group_size_m
        final_rows = m_clusters_local - local_super_rows
        final_rows_safe = T.max(final_rows, 1)
        super_tiles = group_size_m * n_blocks

        is_super_tile = tile_id < local_super_rows * n_blocks
        local_remainder_id = tile_id - local_super_rows * n_blocks
        local_by = T.if_then_else(
            is_super_tile,
            group_size_m * (tile_id // super_tiles) + tile_id % group_size_m,
            local_super_rows + local_remainder_id % final_rows_safe,
        )
        local_bx = T.if_then_else(
            is_super_tile,
            (tile_id % super_tiles) // group_size_m,
            local_remainder_id // final_rows_safe,
        )

        remote_tile_id = tile_id - local_tiles
        target_shard = remote_tile_id // ((num_ranks - 1) * n_blocks)
        idx_in_shard = remote_tile_id % ((num_ranks - 1) * n_blocks)
        peer_rank_offset = idx_in_shard % (num_ranks - 1)
        peer_rank = peer_rank_offset + T.if_then_else(peer_rank_offset >= local_rank, 1, 0)
        remote_by = peer_rank * m_clusters_local + target_shard
        remote_bx = idx_in_shard // (num_ranks - 1)

        by = T.if_then_else(is_local, local_rank * m_clusters_local + local_by, remote_by)
        bx = T.if_then_else(is_local, local_bx, remote_bx)
        return is_local, by, bx, by - local_rank * m_clusters_local

    @T.prim_func
    def main(
        A_local: T.Tensor((M_per_rank, K), dtype),
        B: T.Tensor((N_per_rank, K), dtype),
        mcast_A: T.Tensor((M, K), dtype),
        gathered_A: T.Tensor((M, K), dtype),
        mcast_signal: T.Tensor((m_clusters_total,), T.uint32),
        local_signal: T.Tensor((m_clusters_total,), T.uint32),
        barriers: T.Tensor((2, num_ranks), T.int32),
        C: T.Tensor((M, N_per_rank), dtype),
        num_comm_sms: T.int32,
    ):
        with T.Kernel(sm_num, threads=192, cluster_dims=2) as bid:
            local_rank = T.get_rank()
            num_comp_clusters = (sm_num - num_comm_sms) // 2
            waves = T.ceildiv(total_tiles, num_comp_clusters)
            comm_workers_per_signal = T.min(num_comm_sms * COMM_CHUNKS, comm_k_blocks)

            tx = T.get_thread_binding()
            cta_id = T.block_rank_in_cluster()
            T.assume(cta_id < 2)

            # Publish the hoisted tmem allocation to all warps (auto sync is
            # disabled above; this one is unconditional and block-wide).
            T.sync_threads(0, 192)
            T.barrier_blocks(barriers[0, 0])

            if bid < num_comp_clusters * 2:
                A_shared = T.alloc_shared((num_stages, block_M, block_K), dtype)
                B_shared = T.alloc_shared((num_stages, half_N, block_K), dtype)
                C_tmem_0 = T.alloc_tmem([block_M, block_N], accum_dtype)
                C_tmem_1 = T.alloc_tmem([block_M, block_N], accum_dtype)
                C_local = T.alloc_fragment((block_M, store_block_N), accum_dtype)
                loaded = T.alloc_cluster_barrier([32 * 2] * num_stages)
                consumed = T.alloc_cluster_barrier([1] * num_stages)
                tmem_full = T.alloc_cluster_barrier([1] * 2)
                tmem_empty = T.alloc_cluster_barrier([128 * 2] * 2)

                cluster_id = bid // 2

                if tx < 128:  # warps 0-3: epilogue
                    for w in T.serial(waves):
                        tile_id = w * num_comp_clusters + cluster_id
                        if tile_id < total_tiles:
                            _, by_e, bx_e, _ = tile_coords(tile_id, local_rank)
                            by = T.alloc_var(T.int32)
                            bx = T.alloc_var(T.int32)
                            by = by_e
                            bx = bx_e
                            T.mbarrier_wait_parity(tmem_full[w & 1], (w // 2) & 1)
                            for i in T.unroll(block_N // store_block_N):
                                if (w & 1) == 0:
                                    T.copy(C_tmem_0[:, i * store_block_N:(i + 1) * store_block_N], C_local)
                                else:
                                    T.copy(C_tmem_1[:, i * store_block_N:(i + 1) * store_block_N], C_local)
                                T.copy(
                                    C_local,
                                    C[by * 2 * block_M + cta_id * block_M:by * 2 * block_M + (cta_id + 1) * block_M,
                                      bx * block_N + i * store_block_N:bx * block_N + (i + 1) * store_block_N])
                            T.mbarrier_arrive(tmem_empty[w & 1], 0)

                elif tx < 160:  # warp 4: TMA producer
                    for w in T.serial(waves):
                        tile_id = w * num_comp_clusters + cluster_id
                        if tile_id < total_tiles:
                            is_local, by_e, bx_e, local_by_e = tile_coords(tile_id, local_rank)
                            by = T.alloc_var(T.int32)
                            bx = T.alloc_var(T.int32)
                            local_by = T.alloc_var(T.int32)
                            by = by_e
                            bx = bx_e
                            local_by = local_by_e
                            if not is_local:
                                T.wait_ge(local_signal[by], comm_workers_per_signal)
                            for k in T.serial(k_blocks):
                                phase = w * k_blocks + k
                                T.mbarrier_wait_parity(consumed[phase % num_stages],
                                                       ((phase // num_stages) & 1) ^ 1)
                                if is_local:
                                    T.tma_copy(
                                        A_local[local_by * 2 * block_M + cta_id * block_M:
                                                local_by * 2 * block_M + (cta_id + 1) * block_M,
                                                k * block_K:(k + 1) * block_K],
                                        A_shared[phase % num_stages, :, :],
                                        barrier=loaded[phase % num_stages],
                                    )
                                else:
                                    T.tma_copy(
                                        gathered_A[by * 2 * block_M + cta_id * block_M:
                                                   by * 2 * block_M + (cta_id + 1) * block_M,
                                                   k * block_K:(k + 1) * block_K],
                                        A_shared[phase % num_stages, :, :],
                                        barrier=loaded[phase % num_stages],
                                    )
                                T.tma_copy(
                                    B[bx * block_N + cta_id * half_N:bx * block_N + (cta_id + 1) * half_N,
                                      k * block_K:(k + 1) * block_K],
                                    B_shared[phase % num_stages, :, :],
                                    barrier=loaded[phase % num_stages],
                                )
                                T.mbarrier_arrive(loaded[phase % num_stages], 0)

                elif (tx < 192) and (cta_id == 0):  # warp 5: tcgen05 MMA
                    for w in T.serial(waves):
                        tile_id = w * num_comp_clusters + cluster_id
                        if tile_id < total_tiles:
                            T.mbarrier_wait_parity(tmem_empty[w & 1], ((w // 2) & 1) ^ 1)
                            for k in T.serial(k_blocks):
                                phase = w * k_blocks + k
                                T.mbarrier_wait_parity(loaded[phase % num_stages],
                                                       (phase // num_stages) & 1)
                                if w & 1 == 0:
                                    T.tcgen05_gemm(
                                        A_shared[phase % num_stages, :, :],
                                        B_shared[phase % num_stages, :, :],
                                        C_tmem_0, transpose_B=True,
                                        mbar=consumed[phase % num_stages],
                                        clear_accum=k == 0, use_2cta=True,
                                    )
                                else:
                                    T.tcgen05_gemm(
                                        A_shared[phase % num_stages, :, :],
                                        B_shared[phase % num_stages, :, :],
                                        C_tmem_1, transpose_B=True,
                                        mbar=consumed[phase % num_stages],
                                        clear_accum=k == 0, use_2cta=True,
                                    )
                            T.tcgen05_mma_arrive(tmem_full[w & 1], arrive_2cta=True)
            else:
                A_comm_shared = T.alloc_shared((COMM_CHUNKS, comm_block_M, comm_block_K), dtype)
                A_comm_ready = T.alloc_barrier([1] * COMM_CHUNKS)
                comm_sm_id = bid - num_comp_clusters * 2
                warp_id = T.get_warp_idx_sync()
                lane_id = T.get_lane_idx()
                comm_chunk = warp_id % COMM_CHUNKS
                if warp_id < COMM_CHUNKS:
                    for it in T.serial(T.ceildiv(comm_tasks_per_rank, num_comm_sms * COMM_CHUNKS)):
                        task_id = (it * num_comm_sms + comm_sm_id) * COMM_CHUNKS + comm_chunk
                        if task_id < comm_tasks_per_rank:
                            comm_by = task_id // comm_k_blocks
                            k = task_id - comm_by * comm_k_blocks
                            global_comm_by = local_rank * m_clusters_local + comm_by
                            T.tma_copy(
                                A_local[comm_by * comm_block_M:(comm_by + 1) * comm_block_M,
                                        k * comm_block_K:(k + 1) * comm_block_K],
                                A_comm_shared[comm_chunk, :, :],
                                barrier=A_comm_ready[comm_chunk],
                                leader_thread_extent=32,
                            )
                            if lane_id == 0:
                                T.barrier_arrive(A_comm_ready[comm_chunk])
                            T.mbarrier_wait_parity(A_comm_ready[comm_chunk], it & 1)
                            T.tma_copy(
                                A_comm_shared[comm_chunk, :, :],
                                mcast_A[global_comm_by * comm_block_M:(global_comm_by + 1) * comm_block_M,
                                        k * comm_block_K:(k + 1) * comm_block_K],
                                leader_thread_extent=32,
                            )
                            T.tma_store_wait(0, False)
                            if k + num_comm_sms * COMM_CHUNKS >= comm_k_blocks and lane_id == 0:
                                T.multimem_signal_add(mcast_signal[global_comm_by], 1)

    return main


@tilelang.jit(compile_once=True)
def reset_state_kernel(signal_blocks, num_ranks, threads=256):
    @T.prim_func
    def main(
        local_signal: T.Tensor((signal_blocks,), T.uint32),
        barriers: T.Tensor((2, num_ranks), T.int32),
        do_barrier: T.bool,
    ):
        with T.Kernel(1, threads=threads):
            tid = T.get_thread_binding(0)
            if do_barrier:
                T.sync_blocks(barriers[1, 0])
            for i in T.serial(T.ceildiv(signal_blocks, threads)):
                signal_idx = i * threads + tid
                if signal_idx < signal_blocks:
                    local_signal[signal_idx] = 0
            if tid < num_ranks:
                barriers[0, tid] = 0
            if do_barrier:
                T.fence_sys()
                T.barrier_blocks(barriers[1, 0])

    return main


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    dtype = torch.bfloat16
    M, N, K = args.M, args.N, args.K

    assert M % (num_local_ranks * 256) == 0
    assert N % (num_local_ranks * 256) == 0
    assert K % 128 == 0

    M_per_rank = M // num_local_ranks
    N_per_rank = N // num_local_ranks
    signal_blocks = M // 256

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    dtype_bytes = 2
    mcast_bytes = M * K * dtype_bytes + signal_blocks * 4 + 4096
    regular_bytes = (M_per_rank * K + N_per_rank * K + M * N_per_rank) * dtype_bytes + 4096
    allocator = get_allocator(
        size=max(2**30, regular_bytes + 2**28),
        device=f"cuda:{local_rank}",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_local_ranks,
        group=group,
        use_vmm=True,
        mcast_size=mcast_bytes,
    )

    kernel = ag_gemm_tcgen5_kernel(M, N, K, num_local_ranks, args.group_size_m, args.pipeline_stages)
    reset_kernel = reset_state_kernel(signal_blocks, num_local_ranks)
    kernel.compile_group = group
    reset_kernel.compile_group = group
    kernel.initialize(allocator=allocator)
    reset_kernel.initialize(allocator=allocator)
    if local_rank == 0 and args.print_source:
        print(kernel.get_kernel_source())

    torch.manual_seed(42 + local_rank)
    A = tilelang.tensor((M_per_rank, K), dtype, allocator=allocator).normal_()
    B = tilelang.tensor((N_per_rank, K), dtype, allocator=allocator).normal_()
    C = tilelang.tensor((M, N_per_rank), dtype, allocator=allocator)
    barriers = tilelang.tensor((2, num_local_ranks), torch.int32, allocator=allocator).zero_()

    mcast_A_flat, gathered_A_flat = allocator._allocate_mcast_tensor((M * K,), dtype)
    mcast_signal, local_signal = allocator._allocate_mcast_tensor((signal_blocks,), torch.uint32)
    mcast_A = mcast_A_flat.view(M, K)
    gathered_A = gathered_A_flat.view(M, K)
    dist.barrier(group)
    reset_kernel(local_signal, barriers, False)
    torch.cuda.synchronize()
    dist.barrier(group)

    def run_kernel(comm_sms):
        kernel(A, B, mcast_A, gathered_A, mcast_signal, local_signal, barriers, C, comm_sms)

    def prepare():
        reset_kernel(local_signal, barriers, True)

    # correctness
    run_kernel(args.num_comm_sms)
    torch.cuda.synchronize()
    dist.barrier(group)

    ag_ref = torch.empty((M, K), dtype=dtype, device=f"cuda:{local_rank}")
    dist.all_gather_into_tensor(ag_ref, A, group)
    C_ref = torch.matmul(ag_ref, B.T)
    max_diff = (C_ref.float() - C.float()).abs().max().item()
    torch.testing.assert_close(C, C_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(gathered_A, ag_ref, atol=1e-2, rtol=1e-2)
    print(f"rank {local_rank} check passed. max_diff={max_diff}")
    prepare()
    torch.cuda.synchronize()
    dist.barrier(group)

    # reference-style timing: reset + barrier are outside the event window
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    sms_values = [int(s) for s in args.tune_comm_sms.split(",") if s] if args.tune_comm_sms \
        else [args.num_comm_sms]
    for sms in sms_values:
        for _ in range(args.warmup):
            prepare()
            run_kernel(sms)
        torch.cuda.synchronize()
        dist.barrier(group)
        total = 0.0
        for _ in range(args.rep):
            prepare()
            torch.cuda.synchronize()
            dist.barrier(group)
            start.record()
            run_kernel(sms)
            end.record()
            end.synchronize()
            total += start.elapsed_time(end)
        ms = torch.tensor([total / args.rep], device="cuda")
        dist.all_reduce(ms, op=dist.ReduceOp.MAX, group=group)
        if local_rank == 0:
            tf = 2 * M * N_per_rank * K / (ms.item() * 1e-3) / 1e12
            print(f"tilelang tcgen5 ag_gemm: {ms.item():.3f} ms | {tf:.1f} TFLOPS | comm_sms={sms}")

    allocator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--M", type=int, default=32768)
    parser.add_argument("--N", type=int, default=16384)
    parser.add_argument("--K", type=int, default=2048)
    parser.add_argument("--num-comm-sms", type=int, default=4)
    parser.add_argument("--group-size-m", type=int, default=12)
    parser.add_argument("--pipeline-stages", type=int, default=5)
    parser.add_argument("--tune-comm-sms", type=str, default="")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=10)
    parser.add_argument("--print-source", action="store_true")
    args = parser.parse_args()
    torch.multiprocessing.spawn(main, args=(args.num_processes, args), nprocs=args.num_processes, join=True)
