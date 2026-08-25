# Persistent 2CTA grouped GEMM with tcgen05 warp-specialization on SM100.
# Two modes:
# 1. Standard: A[G*M, K], B[G*N, K], C[G*M, N] — 2D TMA
# 2. NUMA: A_packed[tiles*32, 64], B_packed[tiles*32, 64] — 1D tiled TMA with die-aware scheduling
#
# Computation: C_g = A_g @ B_g^T for each group g

import torch
import tilelang
import tilelang.language as T
from tvm import tirx
from tilelang.carver.arch import driver
from tilelang.profiler import do_bench


SM_SCHEDULE = [
    0, 1, 2, 3, 4, 5, 70, 71, 6, 7, 72, 73, 74, 75, 76, 77,
    8, 9, 10, 11, 12, 13, 78, 79, 14, 15, 80, 81, 82, 83, 84, 85,
    16, 17, 18, 19, 20, 21, 86, 87, 22, 23, 88, 89, 90, 91, 92, 93,
    24, 25, 26, 27, 28, 29, 94, 95, 30, 31, 96, 97, 98, 99, 100, 101,
    32, 33, 34, 35, 36, 37, 102, 103, 38, 39, 104, 105, 106, 107, 108, 109,
    40, 41, 42, 43, 110, 111, 44, 45, 112, 113, 114, 115, 116, 117,
    46, 47, 48, 49, 118, 119, 50, 51, 120, 121, 122, 123, 124, 125,
    52, 53, 54, 55, 126, 127, 56, 57, 128, 129, 130, 131, 132, 133,
    58, 59, 60, 61, 134, 135, 62, 63, 136, 137, 138, 139, 140, 141,
    142, 143, 144, 145, 146, 147, 64, 65, 66, 67, 68, 69,
]

NUM_SMS_PER_DIE = 74


def _exact_n_tile_coords(local_id, m_clusters, n_blocks):
    """Exact-N rasterization: full 8-wide n-stripes, then a tail stripe of
    n_blocks % 8 columns. The eager tracer re-executes the kernel body in its
    own scope, so this helper must live at module level."""
    full_stripes = n_blocks // 8
    n_rem = n_blocks % 8
    if n_rem == 0:
        bx_cluster = local_id // 8 % m_clusters
        by = local_id // (m_clusters * 8) * 8 + local_id % 8
        return bx_cluster, by
    full_tiles = m_clusters * full_stripes * 8
    rid = local_id - full_tiles
    bx_cluster = T.if_then_else(local_id < full_tiles,
                                local_id // 8 % m_clusters, rid // n_rem)
    by = T.if_then_else(local_id < full_tiles,
                        local_id // (m_clusters * 8) * 8 + local_id % 8,
                        full_stripes * 8 + rid % n_rem)
    return bx_cluster, by


@tilelang.jit
def grouped_gemm_persistent_2cta(
    A, B, G, M, N,
    block_M, block_N, store_block_N, block_K,
    in_dtype, out_dtype, accum_dtype, num_stages,
):
    """Standard 2D TMA grouped GEMM (non-NUMA)."""
    G_M, K, G_N = T.const("G_M, K, G_N")

    A: T.Tensor[[G_M, K], in_dtype]
    B: T.Tensor[[G_N, K], in_dtype]
    C = T.empty((G_M, N), out_dtype)

    sm_num = driver.get_num_sms()
    num_clusters = sm_num // 2
    half_N = block_N // 2

    m_blocks = M // block_M
    n_blocks = N // block_N
    m_clusters = m_blocks // 2
    k_blocks = K // block_K

    # Exact-N rasterization: full 8-wide n-stripes first (for B-tile L2
    # reuse), then one narrower tail stripe of n_blocks % 8 columns. No
    # padded tiles — the old pad-to-8 scheme wasted up to 20% compute on
    # shapes like N=5120 (n_blocks=20). (The CUDA reference instead keeps
    # uniform 8-wide stripes over an UNPADDED tile count, which mis-covers
    # and writes out of bounds when n_blocks % 8 != 0.)
    blocks_per_group = m_clusters * n_blocks
    total_tiles = G * blocks_per_group

    waves = T.ceildiv(total_tiles, num_clusters)

    with T.Kernel(sm_num, threads=256, cluster_dims=2) as (block_id):
        A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((num_stages, half_N, block_K), in_dtype)
        C_tmem_0 = T.alloc_tmem([block_M, block_N], accum_dtype)
        C_tmem_1 = T.alloc_tmem([block_M, block_N], accum_dtype)
        C_local = T.alloc_fragment((block_M, store_block_N), accum_dtype)
        C_local_cast = T.alloc_fragment((block_M, block_N), out_dtype)
        C_shared = T.alloc_shared((block_M, store_block_N), out_dtype)
        loaded = T.alloc_cluster_barrier([32 * 2] * num_stages)
        consumed = T.alloc_cluster_barrier([1] * num_stages)
        tmem_full = T.alloc_cluster_barrier([1] * 2)
        tmem_empty = T.alloc_cluster_barrier([128 * 2] * 2)

        tx = T.get_thread_binding()
        cta_id = T.block_rank_in_cluster()
        T.assume(cta_id < 2)
        cluster_id = block_id // 2

        if tx < 32:  # warp 0: TMA load producer
            for w in T.serial(waves):
                tile_id = num_clusters * w + cluster_id
                if tile_id < total_tiles:
                    eid = tile_id // blocks_per_group
                    local_id = tile_id % blocks_per_group
                    bx_cluster, by = _exact_n_tile_coords(local_id, m_clusters, n_blocks)
                    bx = bx_cluster * 2 + cta_id
                    off_m = eid * m_blocks + bx
                    off_n = eid * 2 * n_blocks + by * 2 + cta_id

                    for k in T.serial(k_blocks):
                        phase = w * k_blocks + k
                        T.mbarrier_wait_parity(consumed[phase % num_stages], ((phase // num_stages) & 1) ^ 1)
                        T.tma_copy(
                            A[off_m * block_M : (off_m + 1) * block_M, k * block_K : (k + 1) * block_K],
                            A_shared[phase % num_stages, :, :],
                            barrier=loaded[phase % num_stages],
                        )
                        T.tma_copy(
                            B[off_n * half_N : (off_n + 1) * half_N, k * block_K : (k + 1) * block_K],
                            B_shared[phase % num_stages, :, :],
                            barrier=loaded[phase % num_stages],
                        )
                        T.mbarrier_arrive(loaded[phase % num_stages], 0)

        elif tx < 64 and cta_id == 0:  # warp 1: tcgen05 MMA consumer
            for w in T.serial(waves):
                tile_id = num_clusters * w + cluster_id
                if tile_id < total_tiles:
                    T.mbarrier_wait_parity(tmem_empty[w & 1], ((w // 2) & 1) ^ 1)
                    for k in T.serial(k_blocks):
                        phase = w * k_blocks + k
                        T.mbarrier_wait_parity(loaded[phase % num_stages], (phase // num_stages) & 1)
                        if w & 1 == 0:
                            T.tcgen05_gemm(
                                A_shared[phase % num_stages, :, :],
                                B_shared[phase % num_stages, :, :],
                                C_tmem_0, transpose_B=True,
                                mbar=consumed[phase % num_stages], clear_accum=k == 0, use_2cta=True,
                            )
                        else:
                            T.tcgen05_gemm(
                                A_shared[phase % num_stages, :, :],
                                B_shared[phase % num_stages, :, :],
                                C_tmem_1, transpose_B=True,
                                mbar=consumed[phase % num_stages], clear_accum=k == 0, use_2cta=True,
                            )
                    T.tcgen05_mma_arrive(tmem_full[w & 1], arrive_2cta=True)

        elif 128 <= tx < 256:  # warps 4-7: epilogue
            for w in T.serial(waves):
                tile_id = num_clusters * w + cluster_id
                if tile_id < total_tiles:
                    eid = tile_id // blocks_per_group
                    local_id = tile_id % blocks_per_group
                    bx_cluster, by = _exact_n_tile_coords(local_id, m_clusters, n_blocks)
                    bx = bx_cluster * 2 + cta_id
                    off_m = eid * m_blocks + bx

                    T.mbarrier_wait_parity(tmem_full[w & 1], (w // 2) & 1)
                    if by < n_blocks:
                        for i in T.unroll(T.ceildiv(block_N, store_block_N)):
                            if (w & 1) == 0:
                                T.copy(C_tmem_0[:, i * store_block_N : (i + 1) * store_block_N], C_local)
                            else:
                                T.copy(C_tmem_1[:, i * store_block_N : (i + 1) * store_block_N], C_local)
                            T.copy(C_local,
                                   C[off_m * block_M : (off_m + 1) * block_M,
                                     by * block_N + i * store_block_N : by * block_N + (i + 1) * store_block_N])
                        T.mbarrier_arrive(tmem_empty[w & 1], 0)
                    else:
                        T.mbarrier_arrive(tmem_empty[w & 1], 0)
    return C


@tilelang.jit
def grouped_gemm_persistent_2cta_numa(
    A_packed, B_packed, sm_schedule,
    G, M, N, K, page_A, page_B,
    block_M, block_N, store_block_N, block_K,
    in_dtype, out_dtype, accum_dtype, num_stages,
):
    """NUMA-aware grouped GEMM with 1D tiled TMA + die-aware scheduling.

    A_packed/B_packed are in 1D tiled format [total_tiles*32, 64] from numa_pack.
    sm_schedule is int32[148] mapping hardware smid → logical SM ordering.
    """
    total_A_rows, total_B_rows = T.const("total_A_rows, total_B_rows")

    A_packed: T.Tensor[[total_A_rows, 64], in_dtype]
    B_packed: T.Tensor[[total_B_rows, 64], in_dtype]
    sm_schedule: T.Tensor[[148], "int32"]
    C = T.empty((G * M, N), out_dtype)

    sm_num = driver.get_num_sms()
    half_N = block_N // 2

    # Cluster-wide tile: 256×256 (2 CTAs × 128 each for M, full 256 for N)
    m_blocks_cluster = M // (block_M * 2)  # M / 256
    n_blocks_cluster = N // block_N        # N / 256
    k_iters = K // block_K

    # Die-aware scheduling
    num_clusters_per_die = NUM_SMS_PER_DIE // 2  # 37
    blocks_per_group = m_blocks_cluster * n_blocks_cluster
    tiles_per_die = (G // 2) * blocks_per_group
    waves = T.ceildiv(tiles_per_die, num_clusters_per_die)
    group_size = min(8, n_blocks_cluster)

    with T.Scale("die", 2) as die:
        with T.Scale("sm-cluster", 2, cluster_size=2) as cta_id:
            with T.Scale("sm", sm_num // 2 // 2, num_sms_per_die=NUM_SMS_PER_DIE, cluster_size=2, sm_schedule=sm_schedule) as local_cluster:
                with T.Scale("thread", 224) as tx:
                    A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
                    B_shared = T.alloc_shared((num_stages, half_N, block_K), in_dtype)
                    C_tmem_0 = T.alloc_tmem([block_M, block_N], accum_dtype)
                    C_tmem_1 = T.alloc_tmem([block_M, block_N], accum_dtype)
                    C_local = T.alloc_fragment((block_M, store_block_N), accum_dtype)
                    loaded = T.alloc_cluster_barrier([16] * num_stages)
                    consumed = T.alloc_cluster_barrier([1] * num_stages)
                    tmem_full = T.alloc_cluster_barrier([1] * 2)
                    tmem_empty = T.alloc_cluster_barrier([128 * 2] * 2)

                    T.assume(cta_id < 2)

                    if 128 <= tx < 160:  # warp 4: A TMA producer
                        for w in T.serial(waves):
                            local_tile_id = num_clusters_per_die * w + local_cluster
                            if local_tile_id < tiles_per_die:
                                eid_local = local_tile_id // blocks_per_group
                                local_id = local_tile_id % blocks_per_group
                                m_cluster_idx = (local_id // group_size) % m_blocks_cluster
                                n_idx = (local_id % group_size) + (local_id // group_size) // m_blocks_cluster * group_size

                                for k in T.serial(k_iters):
                                    phase = w * k_iters + k
                                    T.mbarrier_wait_parity_cta(consumed[phase % num_stages], ((phase // num_stages) & 1) ^ 1)

                                    if tx < 132:
                                        sub = tx - 128
                                        l_a = cta_id * 4 + sub
                                        logical_tile_a = k + l_a * k_iters + m_cluster_idx * k_iters * 8 + eid_local * k_iters * 8 * m_blocks_cluster
                                        tile_a_2 = logical_tile_a * 2
                                        hash_a = tirx.call_pure_extern("int32", "__popc", (page_A * 1024 + tile_a_2) & 0x2AD3EF) & 1
                                        phys_tile_a = tile_a_2 + (hash_a ^ die)
                                        T.tma_copy_per_thread(
                                            A_packed[phys_tile_a * 32 : (phys_tile_a + 1) * 32, 0:64],
                                            A_shared[phase % num_stages, sub * 32 : (sub + 1) * 32, :],
                                            barrier=loaded[phase % num_stages],
                                        )

                    elif 160 <= tx < 192:  # warp 5: B TMA producer
                        for w in T.serial(waves):
                            local_tile_id = num_clusters_per_die * w + local_cluster
                            if local_tile_id < tiles_per_die:
                                eid_local = local_tile_id // blocks_per_group
                                local_id = local_tile_id % blocks_per_group
                                n_idx = (local_id % group_size) + (local_id // group_size) // m_blocks_cluster * group_size

                                for k in T.serial(k_iters):
                                    phase = w * k_iters + k
                                    T.mbarrier_wait_parity_cta(consumed[phase % num_stages], ((phase // num_stages) & 1) ^ 1)

                                    if tx < 164:
                                        sub = tx - 160
                                        l_b = cta_id * 4 + sub
                                        logical_tile_b = k + l_b * k_iters + n_idx * k_iters * 8 + eid_local * k_iters * 8 * n_blocks_cluster
                                        tile_b_2 = logical_tile_b * 2
                                        hash_b = tirx.call_pure_extern("int32", "__popc", (page_B * 1024 + tile_b_2) & 0x2AD3EF) & 1
                                        phys_tile_b = tile_b_2 + (hash_b ^ die)
                                        T.tma_copy_per_thread(
                                            B_packed[phys_tile_b * 32 : (phys_tile_b + 1) * 32, 0:64],
                                            B_shared[phase % num_stages, sub * 32 : (sub + 1) * 32, :],
                                            barrier=loaded[phase % num_stages],
                                        )

                    elif 192 <= tx < 224 and cta_id == 0:  # warp 6: tcgen05 MMA consumer
                        for w in T.serial(waves):
                            local_tile_id = num_clusters_per_die * w + local_cluster
                            if local_tile_id < tiles_per_die:
                                T.mbarrier_wait_parity_cta(tmem_empty[w & 1], ((w // 2) & 1) ^ 1)
                                for k in T.serial(k_iters):
                                    phase = w * k_iters + k
                                    T.mbarrier_wait_parity_cta(loaded[phase % num_stages], (phase // num_stages) & 1)
                                    if w & 1 == 0:
                                        T.tcgen05_gemm(
                                            A_shared[phase % num_stages, :, :],
                                            B_shared[phase % num_stages, :, :],
                                            C_tmem_0, transpose_B=True,
                                            mbar=consumed[phase % num_stages], clear_accum=k == 0, use_2cta=True,
                                        )
                                    else:
                                        T.tcgen05_gemm(
                                            A_shared[phase % num_stages, :, :],
                                            B_shared[phase % num_stages, :, :],
                                            C_tmem_1, transpose_B=True,
                                            mbar=consumed[phase % num_stages], clear_accum=k == 0, use_2cta=True,
                                        )
                                T.tcgen05_mma_arrive(tmem_full[w & 1], arrive_2cta=True)

                    elif tx < 128:  # warps 0-3: epilogue (TMA store to standard C)
                        for w in T.serial(waves):
                            local_tile_id = num_clusters_per_die * w + local_cluster
                            if local_tile_id < tiles_per_die:
                                eid_local = local_tile_id // blocks_per_group
                                local_id = local_tile_id % blocks_per_group
                                m_cluster_idx = (local_id // group_size) % m_blocks_cluster
                                n_idx = (local_id % group_size) + (local_id // group_size) // m_blocks_cluster * group_size
                                eid = eid_local + die * (G // 2)
                                # bx within global M dimension
                                bx = m_cluster_idx * 2 + cta_id
                                off_m = eid * (M // block_M) + bx

                                T.mbarrier_wait_parity_cta(tmem_full[w & 1], (w // 2) & 1)

                                for i in T.unroll(T.ceildiv(block_N, store_block_N)):
                                    if (w & 1) == 0:
                                        T.copy(C_tmem_0[:, i * store_block_N : (i + 1) * store_block_N], C_local)
                                    else:
                                        T.copy(C_tmem_1[:, i * store_block_N : (i + 1) * store_block_N], C_local)
                                    T.copy(C_local, C[off_m * block_M, n_idx * block_N + i * store_block_N])

                                T.mbarrier_arrive(tmem_empty[w & 1], 0)
    return C


def main():
    G, M, N, K = 32, 1024, 4096, 6144
    block_M, block_N, block_K = 128, 256, 64
    store_block_N = 64
    in_dtype, out_dtype, accum_dtype = T.bfloat16, T.bfloat16, T.float
    num_stages = 6
    numa_num_stages = 5
    NUM_ITERATIONS = 10

    a = torch.randint(-3, 4, (G * M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randint(-3, 4, (G * N, K), dtype=torch.bfloat16, device="cuda")
    ref_c = torch.cat([a[g * M : (g + 1) * M].float() @ b[g * N : (g + 1) * N].float().T for g in range(G)], dim=0).to(torch.bfloat16)
    total_flops = 2 * G * M * N * K

    print(f"\n{'='*60}")
    print(f"Grouped GEMM: G={G}, M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    # Test TileLang NUMA first. NUMA remapping depends on allocation coloring;
    # running other benchmarks first can move these allocations to a slower base.
    A_tl_numa = None
    B_tl_numa = None
    try:
        from tilelang.distributed.numa import NUMATensor

        sm_sched_tensor = torch.tensor(SM_SCHEDULE, dtype=torch.int32, device="cuda")

        A_tl_numa = NUMATensor.from_torch(a.view(G, M, K))
        B_tl_numa = NUMATensor.from_torch(b.view(G, N, K))

        c_numa_tl = grouped_gemm_persistent_2cta_numa(
            A_tl_numa.as_1d_tiled(), B_tl_numa.as_1d_tiled(), sm_sched_tensor,
            G, M, N, K, A_tl_numa.page_id, B_tl_numa.page_id,
            block_M, block_N, store_block_N, block_K,
            in_dtype, out_dtype, accum_dtype, numa_num_stages,
        )
        torch.testing.assert_close(c_numa_tl, ref_c, rtol=1e-2, atol=1e-2)
        print("[TileLang NUMA]  correctness passed.")

        tl_numa_latency = do_bench(
            lambda: grouped_gemm_persistent_2cta_numa(
                A_tl_numa.as_1d_tiled(), B_tl_numa.as_1d_tiled(), sm_sched_tensor,
                G, M, N, K, A_tl_numa.page_id, B_tl_numa.page_id,
                block_M, block_N, store_block_N, block_K,
                in_dtype, out_dtype, accum_dtype, numa_num_stages,
            ),
            backend="cupti",
        )
        print(f"[TileLang NUMA]  latency: {tl_numa_latency:.4f} ms | {total_flops / (tl_numa_latency / 1e3) / 1e12:.2f} TFLOPS")
        del c_numa_tl
    except Exception as e:
        import traceback
        print(f"[TileLang NUMA]  skipped ({e})")
        traceback.print_exc()
    finally:
        if A_tl_numa is not None:
            A_tl_numa.free()
        if B_tl_numa is not None:
            B_tl_numa.free()

    # Test standard version
    c = grouped_gemm_persistent_2cta(
        a, b, G, M, N, block_M, block_N, store_block_N, block_K,
        in_dtype, out_dtype, accum_dtype, num_stages,
    )
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("[TileLang]       correctness passed.")

    tl_latency = do_bench(
        lambda: grouped_gemm_persistent_2cta(
            a, b, G, M, N, block_M, block_N, store_block_N, block_K,
            in_dtype, out_dtype, accum_dtype, num_stages,
        ),
        backend="cupti",
    )
    print(f"[TileLang]       latency: {tl_latency:.4f} ms | {total_flops / (tl_latency / 1e3) / 1e12:.2f} TFLOPS")

    torch_latency = do_bench(
        lambda: torch.bmm(a.view(G, M, K), b.view(G, N, K).transpose(1, 2)),
        backend="cupti",
    )
    print(f"[cuBLAS]         latency: {torch_latency:.4f} ms | {total_flops / (torch_latency / 1e3) / 1e12:.2f} TFLOPS")


if __name__ == "__main__":
    main()
