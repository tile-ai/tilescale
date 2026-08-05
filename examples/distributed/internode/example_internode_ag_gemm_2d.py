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

What is being hidden
--------------------
Unfused torch is ``all_gather`` then ``matmul``, strictly serial: 0.204 + 0.170 ms at the
default shape. Serial fusion only inherits the collective's advantage. Overlap puts the
floor at ``max(comm, gemm)``, which also stops cuBLAS being 8% faster than our GEMM from
mattering, since that GEMM is covering the network.

Measured at 16 GPUs, M=8192 N=4096 K=4096, and the ordering **inverted** once the collective
got faster: with the slower collective pipeline 0.430 ms beat serial 0.481; with rail-group
pipelining, serial 0.373 (736.8 TF) beats pipeline 0.441 (622.9) against torch's 0.500
(550.0) -- 1.34x against 1.13x. Less exposed fabric leaves less to hide, and the extra
barrier and split launches stop paying. Re-measure whenever the collective changes.

Done with stream and event ordering, never an in-kernel signal wait: see the deadlock note
in the API reference. Multimem only -- on the pull path ``publish_own`` reads siblings'
shards rather than writing to them, so the ordering differs; falls back to ``overlap``.

The GEMM wants 256 threads, not the collective's 1024, or warp specialisation overflows the
block limit.
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
    # serial is the default again, and the reversal is the point. When the collective was
    # slower, pipelining the fabric hop under our own node's GEMM won by 12% (0.430 ms
    # against 0.481). Once rail-group pipelining made the collective ~50% faster there was
    # far less fabric left to hide, and the pipeline's extra barrier and split GEMM launches
    # cost more than they save: serial 0.373 ms / 736.8 TF against pipeline 0.441 / 622.9,
    # so 1.34x torch against 1.13x. Re-measure whenever the collective changes -- the answer
    # is a function of the comm/compute balance, not a property of this kernel.
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
        ag.consume_groups()
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
