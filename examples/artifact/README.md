# TileScale High-Performance Kernels (SM100 / B200)

TileLang-DSL kernels for NVIDIA Blackwell, covering the single-GPU and
multi-GPU workloads evaluated in the TileNest paper. Every kernel is
correctness-verified against PyTorch references; performance numbers below
were measured on an 8×B200 (SM100a) node and match or exceed hand-optimized
CUDA implementations of the same algorithms.

## Single-GPU

| Kernel | File | Measured performance (representative shapes) |
|---|---|---|
| Grouped GEMM BF16 | `single_gpu/grouped_gemm/grouped_gemm_tcgen5mma_ws.py` | up to ~1630 TFLOPS across 24 MoE expert-GEMM shapes |
| Grouped GEMM W_FP8·A_BF16 (fused dequant) | `single_gpu/grouped_gemm/grouped_gemm_tcgen5mma_ws_dequant.py` | up to ~1110 TFLOPS across 24 shapes |
| Die-aware (NUMA) FP8 dequant grouped GEMM, packed C | `single_gpu/grouped_gemm/grouped_gemm_tcgen5mma_ws_numa_dequant.py` | up to ~1125 TFLOPS; die-local 4KB-page layouts for A/B/C |
| FlashAttention fwd, 2-CTA (d128 / d256, MHA+GQA) | `single_gpu/flashattention/attention_kernel_2sm.py` | 1074–1543 TFLOPS (48–69% MFU) at B=8, S=1K–16K; competitive with FlashAttention-4 |
| FlashAttention fwd, 1-CTA | `single_gpu/flashattention/attention_kernel_1sm.py` (+`_d256`) | correctness-verified 1-SM variants |

Common techniques: persistent 2-CTA tcgen05 clusters (256×256×64 tiles,
5-stage SMEM pipeline), warp-specialized roles synchronized purely by
mbarriers, tensor-memory double-buffered accumulator waves, exact-N
rasterization (full 8-wide n-stripes plus one `n_blocks % 8` tail stripe —
padding-based schemes waste up to 33% compute on non-multiple-of-8 shapes).
The attention kernels additionally use per-role register budgets
(`tl.outline_warp_spec_branches`), FA4-style 3-pass softmax with a
polynomial/SFU-mixed exp2, and TMEM alias allocation.

The NUMA kernels map the MoE group axis onto B200's two dies so each die's
working set stays in its local HBM/L2 partition; inputs and outputs use
4KB-page die-interleaved layouts handled by
`tilelang.distributed.numa.NUMATensor`.

## Multi-GPU (8×B200, NVLink; single node, no IMEX required)

| Kernel | File | Measured performance |
|---|---|---|
| AllGather-GEMM BF16 | `multi_gpu/allgather_gemm/example_allgather_gemm_tcgen5.py` | 0.255 ms @ M=32K, N=16K, K=2K (8 ranks) |
| AllGather + FP8-dequant + GEMM | `multi_gpu/allgather_gemm/example_ag_dequant_gemm_tcgen5.py` | 0.320 ms @ same shape |
| GEMM-ReduceScatter BF16 | `multi_gpu/gemm_reduce_scatter/example_gemm_rs_tcgen5.py` | 0.783 ms / 1053 TFLOPS/GPU @ M=32K, N=6K, K=16K |
| GEMM-ReduceScatter W_FP8·A_BF16 | `multi_gpu/gemm_reduce_scatter/example_dequant_gemm_rs_tcgen5.py` | 0.801 ms / 1029 TFLOPS/GPU |
| MoE token Dispatch + Combine | `multi_gpu/moe_dispatch_combine/` | dispatch 716 / combine 662 GB/s BF16 (620 / 664 FP8) at 128 SMs; faster than DeepEP through one harness |

Communication mechanisms (all expressed with TileLang intrinsics):
- **AllGather**: NVSwitch multicast — a TMA store into the multicast VA lands
  on every rank at once; per-256-row readiness is published with
  `T.multimem_signal_add` and consumed with `T.wait_ge`. Comm SMs and compute
  SMs share one persistent grid, and compute consumes its local shard first.
- **ReduceScatter**: no signals — the epilogue hardware-reduces each 128×64
  output chunk straight into the owner rank's buffer with
  `T.atomic_add(use_tma=True, dst_pe=...)` (`cp.reduce.async.bulk` over
  NVLink); a rotated Super-M schedule spreads remote traffic over the kernel.
- **MoE**: fused count → exchange → scatter dispatch (`T.match_any_sync`
  dedup, tokens land at their final compact index, no copy epilogue),
  store-back + local-reduce combine, `T.put_warp` transport.

Launch: `torch.multiprocessing.spawn`; symmetric buffers via
`tilelang.get_allocator(is_distributed=True[, mcast_size=...])` — fabric
handles when IMEX channels are configured, with an automatic POSIX-fd
(SCM_RIGHTS) fallback on single-node setups without IMEX.

## Benchmarking notes

Kernels in the 1–15 ms/iteration range are sensitive to GPU boost-clock
trajectories: short benchmark windows can show ±10–18% phantom differences
between equally fast kernels. Verify suspicious gaps with sustained,
power-limited alternating loops.
