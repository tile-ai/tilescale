"""Fused inter-node allgather + GEMM, on the hierarchical (2D) collective.

``C = allgather(A_shard) @ B``: ``A`` is sharded across ranks along ``M``, ``B`` is
replicated. The flat version of this (``example_internode_ag_gemm.py``) is built on the
flat allgather and so inherits its collapse at 16 GPUs; this one uses ``Allgather2D``
and the tcgen05 GEMM.

Two modes:

* ``serial`` -- allgather, then one GEMM over all ``M`` rows.
* ``overlap`` -- the rows this rank *owns* need no communication, so their GEMM runs on
  the main stream while the collective runs on a side stream. The remaining rows are
  then two row ranges either side of ours, so this costs three GEMM launches instead of
  one; whether that pays depends on how comm-bound the shape is.

Why overlap is worth re-testing here even though it stopped paying before
------------------------------------------------------------------------
With the flat allgather and a *slow* GEMM, split-launch overlap took the fused kernel
from 1.362 to 1.110 ms. Once the GEMM became tcgen05-fast the comm dominated so
completely (0.725 ms of allgather against 0.188 ms of GEMM) that overlap bought
nothing -- 0.915 against 0.906 serial.

The 2D collective changes that balance again: the allgather is now ~3x faster, so the
GEMM is a much larger fraction of the total and there is real work to hide behind. The
mode is a flag rather than a decision because the crossover moves with the shape.

Note ``T.Kernel`` grids and warp specialisation: the GEMM wants ``--gemm-threads 256``,
not the collective's 1024, or the launch exceeds the block limit.
"""

# NOTE: no `from __future__ import annotations` here -- see internode_2d.
import argparse
import os

import torch
import torch.distributed as dist

from internode_2d import Allgather2D, add_2d_args, pick_intra
from internode_common import (
    Context,
    TL_DTYPES,
    TORCH_DTYPES,
    add_common_args,
    bench_vs_torch,
    check,
    prepare_env,
)
from internode_gemm_sm100 import tcgen05_gemm_range_kernel


def main() -> int:
    parser = add_2d_args(add_common_args(argparse.ArgumentParser(description=__doc__)))
    parser.add_argument("--m-per-rank", type=int, default=512)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--gemm-block-n", type=int, default=256)
    parser.add_argument("--mode", choices=("serial", "overlap"), default="serial")
    args = parser.parse_args()

    prepare_env()
    world = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    M_per_rank, N, K = args.m_per_rank, args.n, args.k
    M = M_per_rank * world
    # The collective moves A, so --numel is derived rather than given.
    args.numel = M * K

    itemsize = torch.empty((), dtype=TORCH_DTYPES[args.dtype]).element_size()
    intra = pick_intra(args.intra)
    # A_full + B + C in the arena, plus a multicast copy of A_full.
    arena = int(2.5 * (M * K + K * N + M * N) * itemsize) + (1 << 26)
    ctx = Context(arena_bytes=arena,
                  mcast_bytes=M * K * itemsize if intra == "multimem" else 0)

    torch_dtype, tl_dtype = TORCH_DTYPES[args.dtype], TL_DTYPES[args.dtype]
    ctx.log(
        f"ag_gemm_2d: world={ctx.world_size} M={M} (per-rank {M_per_rank}) N={N} K={K} "
        f"mode={args.mode} intra={intra} chunks={args.chunks} dtype={args.dtype}"
    )

    ag = Allgather2D(ctx, M * K, torch_dtype, tl_dtype, args, intra=intra)

    gemm_full = ctx.compile(
        tcgen05_gemm_range_kernel(M, N, K, M, block_M=args.block_m,
                                  block_N=args.gemm_block_n, block_K=args.block_k)
    )
    gemm_block = None
    if args.mode == "overlap":
        gemm_block = ctx.compile(
            tcgen05_gemm_range_kernel(M, N, K, M_per_rank, block_M=args.block_m,
                                      block_N=args.gemm_block_n, block_K=args.block_k)
        )

    B = ctx.tensor((K, N), torch_dtype)
    C = ctx.tensor((M, N), torch_dtype)
    A_shard = ag.shard.view(M_per_rank, K)
    A_full = ag.out.view(M, K)
    A_shard.copy_((torch.randn(M_per_rank, K, device=B.device) * 0.02).to(torch_dtype))
    B.copy_((torch.randn(K, N, device=B.device) * 0.02).to(torch_dtype))
    C.zero_()

    gemm_stream = torch.cuda.Stream()

    def launch():
        if args.mode == "serial":
            ag.launch()
            gemm_full(A_full, B, C, 0)
            return
        # Our own rows are already local, so their GEMM does not wait for anything.
        main_stream = torch.cuda.current_stream()
        gemm_stream.wait_stream(main_stream)
        with torch.cuda.stream(gemm_stream):
            gemm_block(A_full, B, C, ctx.rank * M_per_rank)
        ag.launch()
        main_stream.wait_stream(gemm_stream)
        # Everything except our own block, one launch per peer row block. The
        # per-rank kernel is reused, so m_offset is all that changes.
        for peer in range(ctx.world_size):
            if peer != ctx.rank:
                gemm_block(A_full, B, C, peer * M_per_rank)

    torch.cuda.synchronize()
    dist.barrier(ctx.group)
    launch()
    torch.cuda.synchronize()

    ref_a = torch.empty(M, K, dtype=torch_dtype, device=B.device)
    dist.all_gather_into_tensor(ref_a.view(-1), ag.shard, group=ctx.group)
    ref = (ref_a.float() @ B.float()).to(torch_dtype)
    failures = check(ctx, C, ref, "ag_gemm_2d")

    if not args.no_bench and failures == 0:
        ref_buf = torch.empty(M, K, dtype=torch_dtype, device=B.device)
        out_buf = torch.empty(M, N, dtype=torch_dtype, device=B.device)

        def run_ref():
            dist.all_gather_into_tensor(ref_buf.view(-1), ag.shard, group=ctx.group)
            torch.matmul(ref_buf, B, out=out_buf)

        bench_vs_torch(ctx, args, "ag_gemm_2d", launch, run_ref, 0,
                       tflops=2 * M * N * K / 1e12)

    ctx.close()
    if ctx.is_leader:
        print("PASS" if failures == 0 else f"FAIL: {failures} rank(s) mismatched", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
