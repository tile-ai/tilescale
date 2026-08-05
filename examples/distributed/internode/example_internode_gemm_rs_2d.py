"""Fused inter-node GEMM + reduce-scatter, on the hierarchical (2D) collective.

``C_shard = reduce_scatter(A @ B)``, with ``K`` sharded across ranks: every rank holds
``A (M x K/W)`` and ``B (K/W x N)``, computes a full-size *partial* product, and the
collective sums the partials and scatters the result along ``M``.

The GEMM writes straight into the collective's input buffer, which on the multimem path
is the multicast allocation -- so there is no copy between compute and communication.
That is the one integration detail worth noting: ``ReduceScatter2D.inp`` is this rank's
own view of that buffer, and it is a perfectly ordinary tensor to write into.

``--mode pipeline``: overlap by node slot, adapted from triton-dist's tile swizzle
--------------------------------------------------------------------------------
The dependency runs the wrong way for the trick AG-GEMM uses -- here the GEMM *produces*
what is sent -- but it can still be split. Output row block *g* belongs to global rank *g*,
and one node's ranks occupy a contiguous block of ``M``, so compute the rows the *other*
node needs first, hand them to the fabric, and compute our own while they fly.

That remote-first ordering is Triton-distributed's ``swizzle_tiled_m_with_padding``, which
renumbers GEMM tiles so a rank computes rank *r+1*'s block first and its own last. **Their
exact swizzle does not transfer**: theirs rotates per rank, which suits per-peer pushes where
every block has one destination, whereas ``multimem.ld_reduce`` needs *every* local rank to
have written a segment, so a per-rank rotation leaves each rank ready on a segment its
siblings are not. Ours needs the same idea at coarser grain in a *common* order, one barrier
per node slot.

**And it loses, so ``serial`` stays the default.** At ``k-per-rank 2048`` serial 0.334 ms /
411 TF against pipeline 0.365 / 376; at 4096, serial 0.429 / 640 against 0.476 / 578. Two
costs swallow the gain: the two barriers are ~30-50 us each, and splitting one GEMM into two
half-height launches worsens the persistent tcgen05 grid's wave quantisation. Together they
exceed the fabric time hidden -- even though comm (~0.237 ms, near-fixed since the output is
``M*N`` whatever ``K`` is) and compute (0.031 ms at K/rank 512 rising to ~0.248 at 4096) are
comparable at the larger shapes.

The flag stays because the balance moves with barrier cost: with a device-side barrier this
should invert. Three overlap attempts have now reached that same conclusion.
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
    parser.add_argument("--mode", choices=("serial", "pipeline"), default="serial",
                        help="pipeline computes the other node's rows first and reduces them "
                             "while our own node's rows are still being computed")
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
        f"intra={intra} mode={args.mode} chunks={args.chunks} dtype={args.dtype}"
    )

    rs = ReduceScatter2D(ctx, M * N, torch_dtype, tl_dtype, args, intra=intra)

    nodes = ctx.world_size // ctx.local_world_size
    rows_per_node = M // nodes
    if args.mode == "pipeline" and M % nodes:
        raise SystemExit(f"--m {M} must divide by {nodes} nodes for --mode pipeline")
    gemm = ctx.compile(
        tcgen05_gemm_range_kernel(M, N, K_per_rank, M, block_M=args.block_m,
                                  block_N=args.gemm_block_n, block_K=args.block_k)
    )
    gemm_node = ctx.compile(
        tcgen05_gemm_range_kernel(M, N, K_per_rank, rows_per_node, block_M=args.block_m,
                                  block_N=args.gemm_block_n, block_K=args.block_k)
    ) if args.mode == "pipeline" else None

    A = ctx.tensor((M, K_per_rank), torch_dtype)
    B = ctx.tensor((K_per_rank, N), torch_dtype)
    A.copy_((torch.randn(M, K_per_rank, device=A.device) * 0.02).to(torch_dtype))
    B.copy_((torch.randn(K_per_rank, N, device=A.device) * 0.02).to(torch_dtype))
    # The GEMM writes the collective's input directly -- no staging copy.
    partial = rs.inp.view(M, N)

    my_node = ctx.rank // ctx.local_world_size

    def launch_pipeline():
        # The other node's rows are what the fabric needs, so they go first.
        for n in range(nodes):
            if n == my_node:
                continue
            gemm_node(A, B, partial, n * rows_per_node)
        dist.barrier(ctx.group)          # every rank has finished those rows
        rs.reduce_remote()
        targets = rs.issue_puts()        # fabric starts
        gemm_node(A, B, partial, my_node * rows_per_node)  # overlaps the transfer
        dist.barrier(ctx.group)
        rs.reduce_own()
        for g, target in enumerate(targets):
            rs.finish_group(g, target)

    def launch():
        if args.mode == "pipeline":
            launch_pipeline()
            return
        gemm(A, B, partial, 0)
        # The GEMM *produces* the collective's input, and the reduce reads every local
        # rank's copy of it through the multicast VA. Stream order only sequences our own
        # GEMM before our own reduce -- it says nothing about a sibling's GEMM, so
        # without this fence we reduce whatever a slower sibling has written so far.
        #
        # ReduceScatter2D advertises "no barrier needed", and that is true when the input
        # is filled once before the loop, as in the standalone example. A producer inside
        # the loop breaks that precondition. It passed on the 8-GPU single-node proxy --
        # where ranks stay tightly synchronised -- and mismatched on 11 of 16 ranks
        # across two nodes, which is exactly the shape of a skew-dependent race.
        dist.barrier(ctx.group)
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
