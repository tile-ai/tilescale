# Persistent 2CTA grouped GEMM with fused FP8->BF16 weight dequantization on
# SM100 (Blackwell).
#
# A[G*M, K] bf16, B[G*N, K] fp8e4m3, scale_inv[G*N/128, K/128] fp32.
# C_g = A_g @ dequant(B_g)^T, dequant(b[n, k]) = fp8(b) * scale_inv[n/128, k/128].
#
# Role layout (320 threads, 2-CTA cluster, mirrors the CUDA reference):
#   tx   0..127 : epilogue (tmem -> registers -> global)
#   tx 128..159 : producer warp (A cluster TMA + B fp8 CTA-local TMA)
#   tx 160..191 : tcgen05 MMA consumer (CTA0 only)
#   tx 192..319 : dequantizer (fp8 smem -> *scale -> bf16 smem)

import sys

import torch
import tilelang
import tilelang.language as T
from grouped_gemm_tcgen5mma_ws import _exact_n_tile_coords
from tilelang.carver.arch import driver
from tilelang.profiler import do_bench


@tilelang.jit
def grouped_gemm_persistent_2cta_dequant(
    A, B, scale_inv, G, M, N,
    block_M, block_N, store_block_N, block_K,
    out_dtype, accum_dtype, num_stages, skip_dequant=0,
):
    G_M, K, G_N, S_ROWS, S_COLS = T.const("G_M, K, G_N, S_ROWS, S_COLS")

    A: T.Tensor[[G_M, K], T.bfloat16]
    B: T.Tensor[[G_N, K], T.float8_e4m3]
    scale_inv: T.Tensor[[S_ROWS, S_COLS], T.float32]
    C = T.empty((G_M, N), out_dtype)

    sm_num = driver.get_num_sms()
    num_clusters = sm_num // 2
    half_N = block_N // 2

    m_blocks = M // block_M
    n_blocks = N // block_N
    m_clusters = m_blocks // 2
    k_blocks = K // block_K

    # Exact-N rasterization (see grouped_gemm_tcgen5mma_ws): no padded tiles.
    blocks_per_group = m_clusters * n_blocks
    total_tiles = G * blocks_per_group
    waves = T.ceildiv(total_tiles, num_clusters)

    with T.Kernel(sm_num, threads=320, cluster_dims=2) as (block_id):
        A_shared = T.alloc_shared((num_stages, block_M, block_K), T.bfloat16)
        B_fp8_shared = T.alloc_shared((num_stages, half_N, block_K), T.float8_e4m3)
        B_shared = T.alloc_shared((num_stages, half_N, block_K), T.bfloat16)
        C_tmem_0 = T.alloc_tmem([block_M, block_N], accum_dtype)
        C_tmem_1 = T.alloc_tmem([block_M, block_N], accum_dtype)
        C_local = T.alloc_fragment((block_M, store_block_N), accum_dtype)
        C_local_cast = T.alloc_fragment((block_M, block_N), out_dtype)
        C_shared = T.alloc_shared((block_M, store_block_N), out_dtype)
        loaded = T.alloc_cluster_barrier([32 * 2] * num_stages)      # A TMA done
        b_fp8_full = T.alloc_barrier([32] * num_stages)              # B fp8 local TMA done
        smem_full = T.alloc_cluster_barrier([128 * 2] * num_stages)  # dequant done
        consumed = T.alloc_cluster_barrier([1] * num_stages)         # MMA consumed stage
        tmem_full = T.alloc_cluster_barrier([1] * 2)
        tmem_empty = T.alloc_cluster_barrier([128 * 2] * 2)

        tx = T.get_thread_binding()
        cta_id = T.block_rank_in_cluster()
        T.assume(cta_id < 2)
        cluster_id = block_id // 2

        if 128 <= tx < 160:  # producer warp: A (cluster TMA) + B fp8 (local TMA)
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
                        T.mbarrier_arrive(loaded[phase % num_stages], 0)
                        if skip_dequant < 3:
                            T.tma_copy(
                                B[off_n * half_N : (off_n + 1) * half_N, k * block_K : (k + 1) * block_K],
                                B_fp8_shared[phase % num_stages, :, :],
                                barrier=b_fp8_full[phase % num_stages],
                            )
                            T.mbarrier_arrive(b_fp8_full[phase % num_stages])

        elif 192 <= tx < 320:  # dequantizer: fp8 -> *scale -> bf16 (128 threads)
            for w in T.serial(waves):
                tile_id = num_clusters * w + cluster_id
                if tile_id < total_tiles:
                    eid = tile_id // blocks_per_group
                    local_id = tile_id % blocks_per_group
                    _, by = _exact_n_tile_coords(local_id, m_clusters, n_blocks)
                    scale_row = (eid * n_blocks + by) * 2 + cta_id

                    for k in T.serial(k_blocks):
                        phase = w * k_blocks + k
                        scale = scale_inv[scale_row, k * block_K // 128]
                        if skip_dequant < 3:
                            T.mbarrier_wait_parity(b_fp8_full[phase % num_stages], (phase // num_stages) & 1)
                        if skip_dequant == 0:
                            for i, j in T.Parallel(half_N, block_K):
                                B_shared[phase % num_stages, i, j] = (
                                    B_fp8_shared[phase % num_stages, i, j].astype(T.float32) * scale
                                ).astype(T.bfloat16)
                            T.fence_proxy_async()
                        T.mbarrier_arrive(smem_full[phase % num_stages], 0)

        elif 160 <= tx < 192 and cta_id == 0:  # tcgen05 MMA consumer
            for w in T.serial(waves):
                tile_id = num_clusters * w + cluster_id
                if tile_id < total_tiles:
                    T.mbarrier_wait_parity(tmem_empty[w & 1], ((w // 2) & 1) ^ 1)
                    for k in T.serial(k_blocks):
                        phase = w * k_blocks + k
                        T.mbarrier_wait_parity(loaded[phase % num_stages], (phase // num_stages) & 1)
                        if skip_dequant < 2:
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

        elif tx < 128:  # epilogue
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


def main():
    G, M, N, K = 32, 1024, 4096, 6144
    for arg in sys.argv[1:]:
        if arg.startswith("--shape="):
            G, M, N, K = (int(x) for x in arg[len("--shape="):].split(","))
    block_M, block_N, block_K = 128, 256, 64
    store_block_N = 64
    num_stages = 5
    NUM_ITERATIONS = 10

    a3 = torch.randint(-3, 4, (G, M, K), dtype=torch.bfloat16, device="cuda")
    b3 = torch.randint(-3, 4, (G, N, K), dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
    scale_inv = torch.ones(G * N // 128, K // 128, dtype=torch.float32, device="cuda")

    a = a3.reshape(G * M, K)
    b = b3.reshape(G * N, K)

    def run():
        return grouped_gemm_persistent_2cta_dequant(
            a, b, scale_inv, G, M, N, block_M, block_N, store_block_N, block_K,
            T.bfloat16, T.float, num_stages)

    c = run()
    ref = torch.bmm(a3.float(), b3.float().transpose(1, 2)).bfloat16()
    torch.testing.assert_close(c.view(G, M, N), ref, rtol=1e-2, atol=1e-2)
    print("[TileLang dequant] correctness passed.")

    latency = do_bench(run, backend="cupti")
    total_flops = 2 * G * M * N * K
    print(f"[TileLang dequant] latency: {latency:.4f} ms | {total_flops / (latency / 1e3) / 1e12:.2f} TFLOPS")


if __name__ == "__main__":
    main()
