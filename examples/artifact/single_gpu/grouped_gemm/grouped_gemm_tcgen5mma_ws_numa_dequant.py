# NUMA-aware persistent 2CTA grouped GEMM with fused FP8->BF16 dequantization
# for B200's dual-die GPU.
#
# A/B inputs and the C output use NUMA-packed layouts: 4KB pages interleaved
# across the two dies by a popc hash of the physical page id, so each die's
# working set stays in its local HBM/L2 partition. Packing/unpacking is done
# with tilelang.distributed.numa.NUMATensor (A: 32x64 bf16 pages, B: 64x64
# fp8 pages, C: 8x256 bf16 pages).
#
# Role layout (352 threads, 2-CTA cluster):
#   tx   0..127 : epilogue (tmem -> smem row -> packed C, die-local)
#   tx 128..131 : producer A (per-thread 1D-tiled TMA, 4x4KB)
#   tx 160..163 : producer B fp8 (per-thread TMA, 4x2KB page halves)
#   tx 192..223 : tcgen05 MMA consumer (CTA0)
#   tx 224..351 : dequantizer (fp8 smem -> *scale -> bf16 smem)

import sys

import torch
import tilelang
import tilelang.language as T
from tvm import tirx
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


@tilelang.jit(pass_configs={"tl.disable_thread_storage_sync": True})
def grouped_gemm_numa_2cta_dequant(
    A_pk, B_pk, C_pk, scale_inv, sm_schedule,
    G, M, N, K, page_A, page_B, page_C, num_stages,
):
    A_ROWS, B_ROWS, C_ELEMS, S_ROWS, S_COLS = T.const("A_ROWS, B_ROWS, C_ELEMS, S_ROWS, S_COLS")

    A_pk: T.Tensor[[A_ROWS, 64], T.bfloat16]
    B_pk: T.Tensor[[B_ROWS, 64], T.float8_e4m3]
    C_pk: T.Tensor[[C_ELEMS], T.bfloat16]
    scale_inv: T.Tensor[[S_ROWS, S_COLS], T.float32]
    sm_schedule: T.Tensor[[148], "int32"]

    sm_num = 148
    block_M, block_N, block_K = 128, 256, 64
    store_block_N = 64
    half_N = block_N // 2

    m_blocks_cluster = M // (block_M * 2)
    n_blocks = N // block_N
    k_iters = K // block_K

    clusters_per_die = NUM_SMS_PER_DIE // 2
    n_padded = (n_blocks + 7) // 8 * 8
    blocks_mn = m_blocks_cluster * n_padded
    tiles_per_die = (G // 2) * blocks_mn
    waves = T.ceildiv(tiles_per_die, clusters_per_die)

    with T.Scale("die", 2) as die:
        with T.Scale("sm-cluster", 2, cluster_size=2) as cta_id:
            with T.Scale("sm", sm_num // 2 // 2, num_sms_per_die=NUM_SMS_PER_DIE, cluster_size=2, sm_schedule=sm_schedule) as local_cluster:
                with T.Scale("thread", 352) as tx:
                    A_shared = T.alloc_shared((num_stages, block_M, block_K), T.bfloat16)
                    B_fp8_shared = T.alloc_shared((num_stages, half_N, block_K), T.float8_e4m3)
                    B_shared = T.alloc_shared((num_stages, half_N, block_K), T.bfloat16)
                    C_tmem_0 = T.alloc_tmem([block_M, block_N], T.float)
                    C_tmem_1 = T.alloc_tmem([block_M, block_N], T.float)
                    C_local = T.alloc_fragment((block_M, store_block_N), T.float)
                    loaded = T.alloc_cluster_barrier([8] * num_stages)
                    b_fp8_full = T.alloc_barrier([4] * num_stages)
                    smem_full = T.alloc_cluster_barrier([128 * 2] * num_stages)
                    consumed = T.alloc_cluster_barrier([1] * num_stages)
                    tmem_full = T.alloc_cluster_barrier([1] * 2)
                    tmem_empty = T.alloc_cluster_barrier([128 * 2] * 2)

                    T.assume(cta_id < 2)

                    if 128 <= tx < 160:  # warp 4: producer A (4 active threads)
                        for w in T.serial(waves):
                            tile_id = clusters_per_die * w + local_cluster
                            if tile_id < tiles_per_die and tx < 132:
                                g = tile_id // blocks_mn
                                local_id = tile_id % blocks_mn
                                m_idx = local_id // 8 % m_blocks_cluster
                                for k in T.serial(k_iters):
                                    phase = w * k_iters + k
                                    l_a = cta_id * 4 + (tx - 128)
                                    tile_a = k + l_a * k_iters + m_idx * k_iters * 8 + g * k_iters * 8 * m_blocks_cluster
                                    t2 = tile_a * 2
                                    hash_a = tirx.call_pure_extern("int32", "__popc", (page_A * 1024 + t2) & 0x2AD3EF) & 1
                                    phys_a = t2 + (hash_a ^ die)
                                    T.mbarrier_wait_parity(consumed[phase % num_stages], ((phase // num_stages) & 1) ^ 1)
                                    T.tma_copy_per_thread(
                                        A_pk[phys_a * 32 : (phys_a + 1) * 32, 0:64],
                                        A_shared[phase % num_stages, (tx - 128) * 32 : (tx - 127) * 32, :],
                                        barrier=loaded[phase % num_stages],
                                    )

                    if 160 <= tx < 192:  # warp 5: producer B fp8 (4 active threads)
                        for w in T.serial(waves):
                            tile_id = clusters_per_die * w + local_cluster
                            if tile_id < tiles_per_die and tx < 164:
                                g = tile_id // blocks_mn
                                local_id = tile_id % blocks_mn
                                by = T.min(local_id // (m_blocks_cluster * 8) * 8 + local_id % 8, n_blocks - 1)
                                for k in T.serial(k_iters):
                                    phase = w * k_iters + k
                                    l_b = cta_id * 4 + (tx - 160)
                                    page_b = k + (l_b // 2) * k_iters + by * 4 * k_iters + g * 4 * k_iters * n_blocks
                                    t2 = page_b * 2
                                    hash_b = tirx.call_pure_extern("int32", "__popc", (page_B * 1024 + t2) & 0x2AD3EF) & 1
                                    phys_b = t2 + (hash_b ^ die)
                                    T.mbarrier_wait_parity(consumed[phase % num_stages], ((phase // num_stages) & 1) ^ 1)
                                    T.tma_copy_per_thread(
                                        B_pk[(phys_b * 2 + l_b % 2) * 32 : (phys_b * 2 + l_b % 2 + 1) * 32, 0:64],
                                        B_fp8_shared[phase % num_stages, (tx - 160) * 32 : (tx - 159) * 32, :],
                                        barrier=b_fp8_full[phase % num_stages],
                                    )

                    if 224 <= tx < 352:  # dequantizer
                        for w in T.serial(waves):
                            tile_id = clusters_per_die * w + local_cluster
                            if tile_id < tiles_per_die:
                                g = tile_id // blocks_mn
                                local_id = tile_id % blocks_mn
                                by = T.min(local_id // (m_blocks_cluster * 8) * 8 + local_id % 8, n_blocks - 1)
                                scale_row = (g * n_blocks + by) * 2 + cta_id
                                for k in T.serial(k_iters):
                                    phase = w * k_iters + k
                                    scale = scale_inv[scale_row, k * block_K // 128]
                                    T.mbarrier_wait_parity(b_fp8_full[phase % num_stages], (phase // num_stages) & 1)
                                    for i, j in T.Parallel(half_N, block_K):
                                        B_shared[phase % num_stages, i, j] = (
                                            B_fp8_shared[phase % num_stages, i, j].astype(T.float32) * scale
                                        ).astype(T.bfloat16)
                                    T.fence_proxy_async()
                                    T.mbarrier_arrive(smem_full[phase % num_stages], 0)

                    if (192 <= tx) and (tx < 224) and (cta_id == 0):  # warp 6: tcgen05 MMA consumer
                        for w in T.serial(waves):
                            tile_id = clusters_per_die * w + local_cluster
                            if tile_id < tiles_per_die:
                                T.mbarrier_wait_parity(tmem_empty[w & 1], ((w // 2) & 1) ^ 1)
                                for k in T.serial(k_iters):
                                    phase = w * k_iters + k
                                    T.mbarrier_wait_parity(loaded[phase % num_stages], (phase // num_stages) & 1)
                                    T.mbarrier_wait_parity(smem_full[phase % num_stages], (phase // num_stages) & 1)
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

                    if tx < 128:  # epilogue: standard-layout C
                        for w in T.serial(waves):
                            tile_id = clusters_per_die * w + local_cluster
                            if tile_id < tiles_per_die:
                                g = tile_id // blocks_mn
                                local_id = tile_id % blocks_mn
                                m_idx = local_id // 8 % m_blocks_cluster
                                by = local_id // (m_blocks_cluster * 8) * 8 + local_id % 8

                                T.mbarrier_wait_parity(tmem_full[w & 1], (w // 2) & 1)
                                if n_padded == n_blocks:  # python-level: no padding guard needed
                                    for i in T.unroll(4):
                                        if (w & 1) == 0:
                                            T.copy(C_tmem_0[:, i * store_block_N : (i + 1) * store_block_N], C_local)
                                        else:
                                            T.copy(C_tmem_1[:, i * store_block_N : (i + 1) * store_block_N], C_local)
                                        for r, jj in T.Parallel(block_M, store_block_N):
                                            tile_c = (by + ((cta_id * 128 + r) // 8) * n_blocks
                                                      + m_idx * n_blocks * 32 + g * n_blocks * 32 * m_blocks_cluster)
                                            t2c = tile_c * 2
                                            hash_c = tirx.call_pure_extern("int32", "__popc", (page_C * 1024 + t2c) & 0x2AD3EF) & 1
                                            C_pk[(t2c + (hash_c ^ die)) * 2048 + ((cta_id * 128 + r) % 8) * 256
                                                 + i * store_block_N + jj] = C_local[r, jj].astype(T.bfloat16)
                                else:
                                    if by < n_blocks:
                                        for i in T.unroll(4):
                                            if (w & 1) == 0:
                                                T.copy(C_tmem_0[:, i * store_block_N : (i + 1) * store_block_N], C_local)
                                            else:
                                                T.copy(C_tmem_1[:, i * store_block_N : (i + 1) * store_block_N], C_local)
                                            for r, jj in T.Parallel(block_M, store_block_N):
                                                tile_c = (by + ((cta_id * 128 + r) // 8) * n_blocks
                                                          + m_idx * n_blocks * 32 + g * n_blocks * 32 * m_blocks_cluster)
                                                t2c = tile_c * 2
                                                hash_c = tirx.call_pure_extern("int32", "__popc", (page_C * 1024 + t2c) & 0x2AD3EF) & 1
                                                C_pk[(t2c + (hash_c ^ die)) * 2048 + ((cta_id * 128 + r) % 8) * 256
                                                     + i * store_block_N + jj] = C_local[r, jj].astype(T.bfloat16)
                                T.mbarrier_arrive(tmem_empty[w & 1], 0)


def main():
    from tilelang.distributed.numa import NUMATensor

    G, M, N, K = 32, 1024, 4096, 6144
    for arg in sys.argv[1:]:
        if arg.startswith("--shape="):
            G, M, N, K = (int(x) for x in arg[len("--shape="):].split(","))
    num_stages = 5

    a0 = torch.randint(-3, 4, (G, M, K), dtype=torch.bfloat16, device="cuda")
    b0 = torch.randint(-3, 4, (G, N, K), dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
    scale_inv = torch.ones(G * N // 128, K // 128, dtype=torch.float32, device="cuda")

    # NUMA-packed tensors: every tile shape is one 4KB page.
    A = NUMATensor.from_torch(a0.view(G * M, K))                        # 32x64 bf16
    B = NUMATensor.from_torch(b0.view(G * N, K), tileK=64, tileMN=64)   # 64x64 fp8
    C = NUMATensor.from_torch(
        torch.zeros(G * M, N, dtype=torch.bfloat16, device="cuda"),
        tileK=256, tileMN=8)                                            # 8x256 bf16

    a_pk = A.as_1d_tiled()
    b_pk = B.as_1d_tiled()
    c_pk = C.as_1d_tiled().reshape(-1)
    sm_sched = torch.tensor(SM_SCHEDULE, dtype=torch.int32, device="cuda")

    def run():
        grouped_gemm_numa_2cta_dequant(
            a_pk, b_pk, c_pk, scale_inv, sm_sched,
            G, M, N, K, A.page_id, B.page_id, C.page_id, num_stages)

    run()
    c_out = C.to_torch().view(G, M, N)
    ref = torch.bmm(a0.float(), b0.float().transpose(1, 2)).bfloat16()
    torch.testing.assert_close(c_out, ref, rtol=1e-2, atol=1e-2)
    print("[TileLang NUMA dequant] correctness passed (packed C).")

    latency = do_bench(run, backend="cupti")
    total_flops = 2 * G * M * N * K
    print(f"[TileLang NUMA dequant] latency: {latency:.4f} ms | {total_flops / (latency / 1e3) / 1e12:.2f} TFLOPS")

    A.free()
    B.free()
    C.free()


if __name__ == "__main__":
    main()
