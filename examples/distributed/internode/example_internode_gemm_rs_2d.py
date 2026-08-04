"""Fused inter-node GEMM + reduce-scatter, on the hierarchical (2D) collective.

``C_shard = reduce_scatter(A @ B)``, with ``K`` sharded across ranks: every rank holds
``A (M x K/W)`` and ``B (K/W x N)``, computes a full-size *partial* product, and the
collective sums the partials and scatters the result along ``M``.

The GEMM writes straight into the collective's input buffer, which on the multimem path
is the multicast allocation -- so there is no copy between compute and communication.
That is the one integration detail worth noting: ``ReduceScatter2D.inp`` is this rank's
own view of that buffer, and it is a perfectly ordinary tensor to write into.

Serial only, and that is a correctness decision rather than an omission
----------------------------------------------------------------------
The dependency runs the wrong way for the trick that works in AG-GEMM: there the rows a
rank owns need no communication, so their GEMM can run during the collective. Here the
GEMM *produces* what is sent, so nothing can be sent before it finishes.

The flat version tried computing the peer's block first, launching the collective, then
computing our own -- and it silently mismatched 5.9M of 8.4M elements, because the
reduce-scatter kernel also folds in this rank's own contribution and so read a block
that had not been written yet. Real overlap needs the collective decomposed into a
per-peer put plus a separate accumulate, which is a kernel change rather than a
scheduling one. Left serial rather than shipping a fast wrong answer.
"""

# NOTE: no `from __future__ import annotations` here -- see internode_2d.
import argparse
import os

import torch
import torch.distributed as dist

from internode_2d import ReduceScatter2D, add_2d_args, pick_intra
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
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k-per-rank", type=int, default=512)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--gemm-block-n", type=int, default=256)
    args = parser.parse_args()

    prepare_env()
    world = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    M, N, K_per_rank = args.m, args.n, args.k_per_rank
    args.numel = M * N

    itemsize = torch.empty((), dtype=TORCH_DTYPES[args.dtype]).element_size()
    intra = pick_intra(args.intra)
    arena = int(3.5 * (M * K_per_rank + K_per_rank * N + M * N) * itemsize) + (1 << 26)
    ctx = Context(arena_bytes=arena,
                  mcast_bytes=M * N * itemsize if intra == "multimem" else 0)

    torch_dtype, tl_dtype = TORCH_DTYPES[args.dtype], TL_DTYPES[args.dtype]
    ctx.log(
        f"gemm_rs_2d: world={ctx.world_size} M={M} N={N} K={K_per_rank}/rank "
        f"intra={intra} chunks={args.chunks} dtype={args.dtype}"
    )

    rs = ReduceScatter2D(ctx, M * N, torch_dtype, tl_dtype, args, intra=intra)

    gemm = ctx.compile(
        tcgen05_gemm_range_kernel(M, N, K_per_rank, M, block_M=args.block_m,
                                  block_N=args.gemm_block_n, block_K=args.block_k)
    )

    A = ctx.tensor((M, K_per_rank), torch_dtype)
    B = ctx.tensor((K_per_rank, N), torch_dtype)
    A.copy_((torch.randn(M, K_per_rank, device=A.device) * 0.02).to(torch_dtype))
    B.copy_((torch.randn(K_per_rank, N, device=A.device) * 0.02).to(torch_dtype))
    # The GEMM writes the collective's input directly -- no staging copy.
    partial = rs.inp.view(M, N)

    def launch():
        gemm(A, B, partial, 0)
        rs.launch()

    torch.cuda.synchronize()
    dist.barrier(ctx.group)
    launch()
    torch.cuda.synchronize()

    ref_full = (A.float() @ B.float()).to(torch_dtype)
    ref = torch.empty(M // ctx.world_size * N, dtype=torch_dtype, device=A.device)
    dist.reduce_scatter_tensor(ref, ref_full.reshape(-1).contiguous(),
                              op=dist.ReduceOp.SUM, group=ctx.group)
    failures = check(ctx, rs.out, ref, "gemm_rs_2d")

    if not args.no_bench and failures == 0:
        gemm_buf = torch.empty(M, N, dtype=torch_dtype, device=A.device)
        ref_buf = torch.empty_like(ref)

        def run_ref():
            torch.matmul(A, B, out=gemm_buf)
            dist.reduce_scatter_tensor(ref_buf, gemm_buf.reshape(-1),
                                       op=dist.ReduceOp.SUM, group=ctx.group)

        bench_vs_torch(ctx, args, "gemm_rs_2d", launch, run_ref, 0,
                       tflops=2 * M * N * K_per_rank / 1e12)

    ctx.close()
    if ctx.is_leader:
        print("PASS" if failures == 0 else f"FAIL: {failures} rank(s) mismatched", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
