"""Fused inter-node allgather + GEMM, on the hierarchical (2D) collective.

``C = allgather(A_shard) @ B``: ``A`` is sharded across ranks along ``M``, ``B`` is
replicated. The flat version of this (``example_internode_ag_gemm.py``) is built on the
flat allgather and so inherits its collapse at 16 GPUs; this one uses ``Allgather2D``
and the tcgen05 GEMM.

Three modes:

* ``serial`` -- allgather, then one GEMM over all ``M`` rows. Forfeits the point of
  fusing, and is here as the reference.
* ``overlap`` -- GEMM this rank's own row block during the collective, then the other
  fifteen after. Hides only 1/16 of the compute, so it is worth ~3%.
* ``pipeline`` -- GEMM a whole *node's* row blocks as soon as that node's rows are
  complete. Since global rank is ``node * lws + local`` and the row block index is the
  global rank, a node's rows are **contiguous**, so this is one GEMM launch per node
  rather than per rank. Our own node's rows are complete after the intra-node broadcast
  alone -- they never touch the fabric -- so half the GEMM (at 2 nodes) runs while the
  fabric hop is still in flight.

What is being hidden, and why this arrangement rather than an in-kernel wait
---------------------------------------------------------------------------
Unfused torch is ``all_gather`` then ``matmul``, strictly serial: 0.204 + 0.170 ms at the
default shape. Serial fusion only inherits the collective's advantage (0.156 + 0.183 =
0.339, about 1.10x). The prize is overlap: with the fabric hop hidden under compute the
floor is ``max(comm, gemm)``, and cuBLAS being 8% faster than our GEMM in isolation stops
mattering because that GEMM is covering the network.

``pipeline`` gets at that with nothing but stream and event ordering over the collective's
existing steps, plus one extra barrier -- so it reuses only kernels that are already
verified, and its correctness rests on the barriers rather than on new signalling.
**It is nonetheless unverified on hardware**: every kernel it needs lowers, but the
cluster has had no two free nodes (and latterly no free node) since it was written. Deliberately **not** an in-kernel signal wait:
that deadlocked before, because a 2048-CTA GEMM grid fills every SM with CTAs waiting on
a signal and the comm kernel never gets scheduled. Capping the GEMM at 132 persistent CTAs
fixed the hang and was *slower than serial*. Doing it properly needs an SM partition like
Triton-distributed's ``num_sync_sms``/``num_p2p_sms``, which is not something to tune
blind.

Multimem only: on the pull path ``publish_own`` reads siblings' shards rather than writing
to them, so the ordering is different. Falls back to ``overlap`` there.

Note the GEMM wants 256 threads, not the collective's 1024, or warp specialisation
overflows the block limit.
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
    # serial is the default because it is the mode that has actually been verified on
    # hardware. pipeline is implemented and every kernel it needs lowers at both 16-GPU
    # and proxy shapes, but no node has been free long enough to run it -- flip the
    # default once run_2d_proxy.sh confirms it.
    parser.add_argument("--mode", choices=("serial", "overlap", "pipeline"),
                        default="serial")
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
    ag = Allgather2D(ctx, M * K, torch_dtype, tl_dtype, args, intra=intra)
    lws = ctx.local_world_size
    nodes = ctx.world_size // lws
    mode = args.mode
    if mode == "pipeline" and intra != "multimem":
        ctx.log("  note: --mode pipeline needs multimem; falling back to overlap")
        mode = "overlap"

    ctx.log(
        f"ag_gemm_2d: world={ctx.world_size} M={M} (per-rank {M_per_rank}) N={N} K={K} "
        f"mode={mode} intra={intra} chunks={args.chunks} dtype={args.dtype}"
    )

    gemm_full = gemm_block = gemm_node = None
    if mode == "serial":
        gemm_full = ctx.compile(
            tcgen05_gemm_range_kernel(M, N, K, M, block_M=args.block_m,
                                      block_N=args.gemm_block_n, block_K=args.block_k)
        )
    elif mode == "overlap":
        gemm_block = ctx.compile(
            tcgen05_gemm_range_kernel(M, N, K, M_per_rank, block_M=args.block_m,
                                      block_N=args.gemm_block_n, block_K=args.block_k)
        )
    else:
        # One node's worth of rows, which is contiguous -- see Allgather2D.rows_of_node.
        gemm_node = ctx.compile(
            tcgen05_gemm_range_kernel(M, N, K, lws * M_per_rank, block_M=args.block_m,
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
    comm_stream = torch.cuda.Stream()

    my_node = ctx.rank // lws

    def launch():
        main_stream = torch.cuda.current_stream()
        if mode == "serial":
            ag.launch()
            gemm_full(A_full, B, C, 0)
            return
        if mode == "overlap":
            # Our own rows are already local, so their GEMM waits on nothing.
            gemm_stream.wait_stream(main_stream)
            with torch.cuda.stream(gemm_stream):
                gemm_block(A_full, B, C, ctx.rank * M_per_rank)
            ag.launch()
            main_stream.wait_stream(gemm_stream)
            for peer in range(ctx.world_size):
                if peer != ctx.rank:
                    gemm_block(A_full, B, C, peer * M_per_rank)
            return
        # pipeline: the fabric hop runs on its own stream while we finish and then
        # consume our own node's rows, which never cross the fabric.
        ag.rail_hop(stream=comm_stream)
        ag.publish_own()
        # Our node's rows are complete once every sibling has published. This barrier is
        # ordered after publish_own on the main stream, so it does not wait for the
        # fabric hop -- that is still in flight on comm_stream.
        dist.barrier(ctx.group)
        gemm_node(A_full, B, C, ag.rows_of_node(my_node, M_per_rank))
        main_stream.wait_stream(comm_stream)
        ag.publish_remote()
        dist.barrier(ctx.group)
        for n in range(nodes):
            if n != my_node:
                gemm_node(A_full, B, C, ag.rows_of_node(n, M_per_rank))

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
