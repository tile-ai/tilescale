# B200 GEMM + reduce-scatter fused kernel with tcgen05 2-CTA MMA.
#
# Every SM computes (no comm SMs); the
# reduce-scatter IS the epilogue — each 128x64 output chunk is staged to SMEM
# and hardware-reduced straight into the owner rank's C over NVLink with
# cp.reduce.async.bulk (T.atomic_add(use_tma=True, dst_pe=...)), no signals.
# Scheduling starts every rank on the next rank's M segment so remote traffic
# is spread across the whole kernel.
#
# A (M, K_per_rank) bf16, B (N, K_per_rank) bf16, C (M_per_rank, N) bf16:
# C = reduce_scatter_m(A @ B^T). C must be a symmetric allocator tensor.

import argparse

import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.distributed import init_dist
from tilelang import get_allocator


# disable_shared_memory_reuse: the merge pass would alias C_shared with
# A_shared (liveness analysis assumes sequential execution, but the epilogue
# warps write C_shared while producer/MMA warps still stream A concurrently).
@tilelang.jit(compile_once=True, pass_configs={
    "tl.disable_thread_storage_sync": True,
    "tl.disable_shared_memory_reuse": True,
})
def gemm_rs_tcgen5_kernel(M, N, K_total, num_ranks, group_size_m, num_stages, epilogue="tma"):
    dtype = T.bfloat16
    accum_dtype = T.float32
    block_M = 128
    block_N = 256
    half_N = block_N // 2
    block_K = 64
    store_block_N = 64

    K = K_total // num_ranks
    M_per_rank = M // num_ranks
    sm_num = driver.get_num_sms()
    num_clusters = sm_num // 2

    m_clusters_total = M // (2 * block_M)
    m_clusters_local = M_per_rank // (2 * block_M)
    n_blocks = N // block_N
    k_blocks = K // block_K
    total_tiles = m_clusters_total * n_blocks
    local_tiles = m_clusters_local * n_blocks

    def tile_coords(tile_id, local_rank):
        """Rotated Super-M scheduler: rank r starts on rank (r+1)'s segment."""
        rot = (tile_id + (local_rank + 1) * local_tiles) % total_tiles
        super_rows = (m_clusters_total // group_size_m) * group_size_m
        final_rows = m_clusters_total - super_rows
        final_rows_safe = T.max(final_rows, 1)
        super_tiles = group_size_m * n_blocks

        is_super_tile = rot < super_rows * n_blocks
        remainder_id = rot - super_rows * n_blocks
        by = T.if_then_else(
            is_super_tile,
            group_size_m * (rot // super_tiles) + rot % group_size_m,
            super_rows + remainder_id % final_rows_safe,
        )
        bx = T.if_then_else(
            is_super_tile,
            (rot % super_tiles) // group_size_m,
            remainder_id // final_rows_safe,
        )
        return by, bx

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M_per_rank, N), dtype),
    ):
        with T.Kernel(sm_num, threads=192, cluster_dims=2) as bid:
            local_rank = T.get_rank()
            waves = T.ceildiv(total_tiles, num_clusters)

            tx = T.get_thread_binding()
            cta_id = T.block_rank_in_cluster()
            T.assume(cta_id < 2)

            A_shared = T.alloc_shared((num_stages, block_M, block_K), dtype)
            B_shared = T.alloc_shared((num_stages, half_N, block_K), dtype)
            C_tmem_0 = T.alloc_tmem([block_M, block_N], accum_dtype)
            C_tmem_1 = T.alloc_tmem([block_M, block_N], accum_dtype)
            C_local = T.alloc_fragment((block_M, store_block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, store_block_N), dtype)
            loaded = T.alloc_cluster_barrier([32 * 2] * num_stages)
            consumed = T.alloc_cluster_barrier([1] * num_stages)
            tmem_full = T.alloc_cluster_barrier([1] * 2)
            tmem_empty = T.alloc_cluster_barrier([128 * 2] * 2)

            cluster_id = bid // 2

            # Publish the hoisted tmem allocation (auto sync disabled above).
            T.sync_threads(0, 192)

            if tx < 128:  # warps 0-3: epilogue = the reduce-scatter
                for w in T.serial(waves):
                    tile_id = w * num_clusters + cluster_id
                    if tile_id < total_tiles:
                        by_e, bx_e = tile_coords(tile_id, local_rank)
                        by = T.alloc_var(T.int32)
                        bx = T.alloc_var(T.int32)
                        by = by_e
                        bx = bx_e
                        dst_rank = by // m_clusters_local
                        store_m = (by % m_clusters_local) * 2 * block_M + cta_id * block_M
                        T.mbarrier_wait_parity(tmem_full[w & 1], (w // 2) & 1)
                        for i in T.unroll(block_N // store_block_N):
                            if (w & 1) == 0:
                                T.copy(C_tmem_0[:, i * store_block_N:(i + 1) * store_block_N], C_local)
                            else:
                                T.copy(C_tmem_1[:, i * store_block_N:(i + 1) * store_block_N], C_local)
                            if epilogue == "store":
                                # debug arm: plain local store (single-rank only)
                                T.copy(C_local,
                                       C[store_m:store_m + block_M,
                                         bx * block_N + i * store_block_N:bx * block_N + (i + 1) * store_block_N])
                            elif epilogue == "addx2":
                                for r, jj in T.Parallel(block_M, store_block_N // 2):
                                    T.atomic_addx2(
                                        C[store_m + r, bx * block_N + i * store_block_N + jj * 2],
                                        C_local[r, jj * 2],
                                        dst_pe=dst_rank,
                                    )
                            else:
                                T.copy(C_local, C_shared)
                                T.fence_proxy_async()
                                T.sync_threads(1, 128)
                                T.atomic_add(
                                    C[store_m:store_m + block_M,
                                      bx * block_N + i * store_block_N:bx * block_N + (i + 1) * store_block_N],
                                    C_shared,
                                    use_tma=True,
                                    tma_wait_complete=True,
                                    dst_pe=dst_rank,
                                )
                                T.sync_threads(1, 128)
                        T.mbarrier_arrive(tmem_empty[w & 1], 0)

            elif tx < 160:  # warp 4: TMA producer
                for w in T.serial(waves):
                    tile_id = w * num_clusters + cluster_id
                    if tile_id < total_tiles:
                        by_e, bx_e = tile_coords(tile_id, local_rank)
                        by = T.alloc_var(T.int32)
                        bx = T.alloc_var(T.int32)
                        by = by_e
                        bx = bx_e
                        for k in T.serial(k_blocks):
                            phase = w * k_blocks + k
                            T.mbarrier_wait_parity(consumed[phase % num_stages],
                                                   ((phase // num_stages) & 1) ^ 1)
                            T.tma_copy(
                                A[by * 2 * block_M + cta_id * block_M:
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
                    tile_id = w * num_clusters + cluster_id
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

    return main


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    dtype = torch.bfloat16
    M, N, K_total = args.M, args.N, args.K

    assert M % (num_local_ranks * 256) == 0
    assert N % 256 == 0
    assert (K_total // num_local_ranks) % 64 == 0

    M_per_rank = M // num_local_ranks
    K = K_total // num_local_ranks

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    allocator = get_allocator(
        size=max(2**28, M_per_rank * N * 2 + 2**20),
        device=f"cuda:{local_rank}",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_local_ranks,
        group=group,
    )

    kernel = gemm_rs_tcgen5_kernel(M, N, K_total, num_local_ranks, args.group_size_m, args.pipeline_stages,
                                   args.epilogue)
    kernel.compile_group = group
    kernel.initialize(allocator=allocator)
    if local_rank == 0 and args.print_source:
        print(kernel.get_kernel_source())

    torch.manual_seed(42 + local_rank)
    dev = f"cuda:{local_rank}"
    A = torch.empty((M, K), dtype=dtype, device=dev).normal_()
    B = torch.empty((N, K), dtype=dtype, device=dev).normal_()
    C = tilelang.tensor((M_per_rank, N), dtype, allocator=allocator)

    def prepare():
        # All ranks must zero C before any rank starts reducing into peers.
        torch.cuda.synchronize()
        dist.barrier(group)
        C.zero_()
        torch.cuda.synchronize()
        dist.barrier(group)

    prepare()
    kernel(A, B, C)
    torch.cuda.synchronize()
    dist.barrier(group)

    partial = torch.matmul(A.float(), B.float().T)
    dist.all_reduce(partial, group=group)
    C_ref = partial[local_rank * M_per_rank:(local_rank + 1) * M_per_rank].bfloat16()
    max_diff = (C_ref.float() - C.float()).abs().max().item()
    # The hardware reduction accumulates in bf16, rounding the running sum
    # once per arriving partial (~ulp(|C|) each); the fp32 reference has none
    # of that error, so allow num_ranks rounding steps at bf16 precision.
    atol = num_ranks * C_ref.abs().max().item() * 2**-8
    torch.testing.assert_close(C.float(), C_ref.float(), atol=atol, rtol=5e-2)
    print(f"rank {local_rank} check passed. max_diff={max_diff} (atol={atol:.2f})")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(args.warmup):
        prepare()
        kernel(A, B, C)
    total = 0.0
    for _ in range(args.rep):
        prepare()
        start.record()
        kernel(A, B, C)
        end.record()
        end.synchronize()
        total += start.elapsed_time(end)
    ms = torch.tensor([total / args.rep], device="cuda")
    dist.all_reduce(ms, op=dist.ReduceOp.MAX, group=group)
    if local_rank == 0:
        tf = 2 * M * N * K / (ms.item() * 1e-3) / 1e12
        print(f"tilelang tcgen5 gemm_rs: {ms.item():.3f} ms | {tf:.1f} TFLOPS/GPU")

    allocator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--M", type=int, default=32768)
    parser.add_argument("--N", type=int, default=6144)
    parser.add_argument("--K", type=int, default=16384, help="Total K across ranks")
    parser.add_argument("--group-size-m", type=int, default=8)
    parser.add_argument("--epilogue", type=str, default="tma", choices=["tma", "addx2", "store"])
    parser.add_argument("--pipeline-stages", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=10)
    parser.add_argument("--print-source", action="store_true")
    args = parser.parse_args()
    torch.multiprocessing.spawn(main, args=(args.num_processes, args), nprocs=args.num_processes, join=True)
