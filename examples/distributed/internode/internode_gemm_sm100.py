"""Blackwell (sm_100) warp-specialised persistent GEMM over a row range.

Ported from ``examples/gemm_sm100/gemm_tcgen5mma_ws_persistent.py``, which reaches
1650 TFLOP/s on a B200 against cuBLAS's 1419 -- 1.16x. The naive
``T.copy``/``T.gemm`` loop this replaces managed only 470 TFLOP/s, so the fused
inter-node kernels were bottlenecked on compute by 3.5x, not on the network.

What makes it fast, all of which the naive version lacks:

* **tcgen05 MMA into tensor memory.** Accumulators live in TMEM
  (``T.alloc_tmem``) rather than registers, so a 128x256 tile does not blow the
  register budget. Two of them, alternated by wave parity, let the next MMA start
  while the previous result is still being drained.
* **Warp specialisation.** Warp 0 issues TMA loads, warp 1 issues the MMAs, warps
  4-7 run the epilogue. Each stays resident on its own job instead of the whole
  CTA marching through load-compute-store in lockstep.
* **Persistent grid.** One CTA per SM, striding over output tiles, so tile setup
  is paid once per SM rather than once per tile.
* **Explicit mbarrier phases.** ``num_stages`` deep pipelining with parity
  computed by hand; this is what overlaps the TMA loads with the MMAs.

The row range is what this file adds. ``m_offset`` is a runtime argument so the
host can launch the same kernel over locally-owned rows and over each peer's rows
separately -- the split-launch overlap the AG-GEMM example relies on. Everything
else follows the upstream example.

Shape constraints inherited from it: ``K % (2 * block_K) == 0`` and
``n_blocks % (2 * group_size) == 0``.
"""

# NOTE: no `from __future__ import annotations` here. This file defines a
# T.prim_func, and PEP 563 would turn `T.Tensor((M, K), in_dtype)` into a string
# evaluated against module globals, where the closure locals M/K/in_dtype do not
# exist -- "NameError: name 'M' is not defined" at trace time.
import tilelang.language as T
from tilelang.carver.arch import driver


def tcgen05_gemm_range_kernel(
    M: int,
    N: int,
    K: int,
    m_rows: int,
    block_M: int = 128,
    block_N: int = 256,
    block_K: int = 64,
    store_block_N: int = 64,
    num_stages: int = 4,
    group_size: int = 8,
    in_dtype: str = "bfloat16",
    out_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
):
    """``C[m_offset : m_offset + m_rows] = A[same rows] @ B``.

    ``m_rows`` is compile-time (it fixes the grid and wave count); ``m_offset`` is
    a kernel argument so one compiled kernel serves every rank's row block.
    """
    sm_num = driver.get_num_sms()
    m_blocks = T.ceildiv(m_rows, block_M)
    n_blocks = T.ceildiv(N, block_N)
    k_blocks = T.ceildiv(K, block_K)
    waves = T.ceildiv(m_blocks * n_blocks, sm_num)
    if K % (2 * block_K):
        raise ValueError(f"K={K} must be a multiple of 2*block_K={2 * block_K}")
    if n_blocks % (2 * group_size):
        raise ValueError(
            f"n_blocks={n_blocks} must be a multiple of 2*group_size={2 * group_size}; "
            "adjust --gemm-block-n or group_size"
        )

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), out_dtype),
        m_offset: T.int32,
    ):
        with T.Kernel(sm_num, threads=256) as block_id:
            A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((num_stages, block_K, block_N), in_dtype)
            # Double-buffered accumulators so wave w+1's MMAs overlap wave w's drain.
            C_tmem_0 = T.alloc_tmem([block_M, block_N], accum_dtype)
            C_tmem_1 = T.alloc_tmem([block_M, block_N], accum_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, store_block_N), out_dtype)
            loaded = T.alloc_barrier([32] * num_stages)
            consumed = T.alloc_barrier([1] * num_stages)
            tmem_full = T.alloc_barrier([1] * 2)
            tmem_empty = T.alloc_barrier([128] * 2)

            tx = T.get_thread_binding()

            if tx < 32:  # warp 0: TMA loads
                for w in T.unroll(waves):
                    tile_id = sm_num * w + block_id
                    bx = (tile_id // group_size) % m_blocks
                    by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size
                    if bx * block_M < m_rows and by * block_N < N:
                        row = m_offset + bx * block_M
                        for k in T.serial(k_blocks):
                            phase = w * k_blocks + k
                            T.mbarrier_wait_parity(consumed[phase % num_stages],
                                                   ((phase // num_stages) & 1) ^ 1)
                            T.tma_copy(
                                A[row:row + block_M, k * block_K:(k + 1) * block_K],
                                A_shared[phase % num_stages, :, :],
                                barrier=loaded[phase % num_stages],
                            )
                            T.tma_copy(
                                B[k * block_K:(k + 1) * block_K,
                                  by * block_N:(by + 1) * block_N],
                                B_shared[phase % num_stages, :, :],
                                barrier=loaded[phase % num_stages],
                            )
                            T.mbarrier_arrive(loaded[phase % num_stages])

            elif tx < 64:  # warp 1: issue tcgen05 MMAs
                for w in T.unroll(waves):
                    tile_id = sm_num * w + block_id
                    bx = (tile_id // group_size) % m_blocks
                    by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size
                    if bx * block_M < m_rows and by * block_N < N:
                        T.mbarrier_wait_parity(tmem_empty[w & 1], ((w // 2) & 1) ^ 1)
                        for k in T.serial(k_blocks):
                            phase = w * k_blocks + k
                            T.mbarrier_wait_parity(loaded[phase % num_stages],
                                                   (phase // num_stages) & 1)
                            if w & 1 == 0:
                                T.tcgen05_gemm(
                                    A_shared[phase % num_stages, :, :],
                                    B_shared[phase % num_stages, :, :],
                                    C_tmem_0, False, False,
                                    mbar=consumed[phase % num_stages],
                                    clear_accum=k == 0,
                                )
                            else:
                                T.tcgen05_gemm(
                                    A_shared[phase % num_stages, :, :],
                                    B_shared[phase % num_stages, :, :],
                                    C_tmem_1, False, False,
                                    mbar=consumed[phase % num_stages],
                                    clear_accum=k == 0,
                                )
                        T.tcgen05_mma_arrive(tmem_full[w & 1])

            elif 128 <= tx < 256:  # warps 4-7: epilogue
                for w in T.unroll(waves):
                    tile_id = sm_num * w + block_id
                    bx = (tile_id // group_size) % m_blocks
                    by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size
                    if bx * block_M < m_rows and by * block_N < N:
                        row = m_offset + bx * block_M
                        T.mbarrier_wait_parity(tmem_full[w & 1], (w // 2) & 1)
                        if (w & 1) == 0:
                            T.copy(C_tmem_0, C_local)
                        else:
                            T.copy(C_tmem_1, C_local)
                        T.mbarrier_arrive(tmem_empty[w & 1])
                        for i in T.unroll(T.ceildiv(block_N, store_block_N)):
                            T.copy(C_local[:, i * store_block_N:(i + 1) * store_block_N], C_shared)
                            T.copy(C_shared, C[row, by * block_N + i * store_block_N])

    return main
