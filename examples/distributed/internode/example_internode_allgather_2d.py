"""Hierarchical (2D) inter-node allgather: rail-aligned GIN overlapped with NVLink.

The flat allgather in ``example_internode_allgather.py`` pushes every rank's shard
to all ``world_size - 1`` peers over GIN. That is optimal with one GPU per node and
catastrophic with eight: each rank puts 15 shards onto the NIC, including the 7
peers sitting on the same machine, and it measures 2.7 GB/s against torch's 309.

This version splits the collective along the topology, into three kernels:

* ``inter`` -- **rail-aligned inter-node exchange.** Rank ``(node, local)`` puts its
  shard only to the ranks holding the *same* ``local`` index on other nodes, i.e.
  ``peer = other_node * local_world_size + local``. Every rank drives its own NIC
  and only ``1/local_world_size`` of the data crosses the fabric.
* ``intra_local`` -- **our own node's slots, pulled from siblings' input buffers.**
* ``intra_remote`` -- **the other nodes' slots, pulled from siblings' output.**

Why the split of phase 2 into two kernels is the whole point
------------------------------------------------------------
The naive version runs one intra-node allgather after the inter-node one, with a
barrier between: ~0.31 ms of NIC then ~0.5 ms of NVLink, strictly serial.

But a sibling's *own* shard is already in the symmetric arena before the
collective starts. Pulling it needs no ordering with anything, so the slots of our
own node can be filled **concurrently with the fabric transfer**, on a side
stream. Only the other nodes' slots genuinely depend on every rank's ``inter``
having landed, and those are the only ones behind the barrier. That halves the
NVLink traffic on the critical path and hides the rest under the NIC.

``intra_local`` therefore reads the peer's ``shard`` while ``intra_remote`` reads
the peer's ``out`` -- the only difference between the two, which is why they are
one kernel parameterised by ``source``.

Why a host barrier
------------------
``intra_remote`` reads a *local peer's* output buffer, so it must not start until
that peer has finished ``inter``. Stream ordering only orders this rank's own work,
and the GIN signal only proves our own rail partner delivered -- neither says
anything about a sibling on the same node. A grid-wide device barrier across ranks
would need every participating CTA co-resident, which is the deadlock documented
in ``example_internode_ag_gemm.py``.

``dist.barrier`` is therefore the honest first version; it costs tens of
microseconds against a collective of ~0.6 ms. Replacing it with a device-side
arrive/wait on a symmetric flag (the ``*_barrier_gpu`` helpers in
``tilelang/language/distributed/sync.py``) is the obvious next step, and the
kernels below do not change when it lands.

Requires real intra-node peer pointers, so it does **not** work under
``TILESCALE_SKIP_PEER_IPC=1``.
"""

# NOTE: no `from __future__ import annotations` here -- see the allgather example.
import argparse

import torch
import torch.distributed as dist

import tilelang.language as T
from tilelang.distributed.bench import do_bench

from internode_common import (
    SIGNAL_DATA,
    Context,
    TL_DTYPES,
    TORCH_DTYPES,
    add_common_args,
    check,
    prepare_env,
    report,
)


def ag2d_inter_kernel(
    shard_numel: int, chunks: int, threads: int, world_size: int, local_world_size: int,
    dtype: str, signal_id: int = SIGNAL_DATA,
):
    """Rail-aligned inter-node exchange, plus the local copy of our own shard."""
    nodes = world_size // local_world_size
    rails = nodes - 1
    chunk_numel = shard_numel // chunks

    @T.prim_func
    def main(
        shard: T.Tensor((shard_numel,), dtype),
        out: T.Tensor((world_size * shard_numel,), dtype),
        rank: T.int32,
        signal_target: T.int32,
    ):
        with T.Kernel(chunks, threads=threads) as bx:
            base = bx * chunk_numel
            local_rank = rank % local_world_size
            node = rank // local_world_size
            # Same local index on every other node: one NIC per rank, no fan-out.
            for step in range(rails):
                peer = ((node + step + 1) % nodes) * local_world_size + local_rank
                T.nccl_gin.put_signal(
                    src=shard[base],
                    dst=out[rank * shard_numel + base],
                    size=chunk_numel,
                    peer=peer,
                    signal_id=signal_id,
                    scope="block",
                )
            for i in T.Parallel(chunk_numel):
                out[rank * shard_numel + base + i] = shard[base + i]
            T.nccl_gin.wait_signal(least=signal_target, signal_id=signal_id, scope="block")

    return main


def ag2d_intra_kernel(
    shard_numel: int, chunks: int, threads: int, world_size: int, local_world_size: int,
    dtype: str, source: str,
):
    """Intra-node allgather over NVLink: one CTA per (sibling, node slot, chunk).

    ``source="shard"`` pulls each sibling's *input* buffer into the slots of our own
    node. Those bytes exist before the collective starts, so this needs no barrier
    and overlaps the fabric transfer. ``source="out"`` pulls the slots belonging to
    the other nodes, which exist only once every rank's ``inter`` has landed.

    ``src_pe`` is a **global** rank. ``get_remote_base_ptr`` returns 0 for a peer it
    considers inter-node, so passing a local rank yields a null base and faults on
    every node but node 0 -- where local and global ranks happen to coincide.
    """
    nodes = world_size // local_world_size
    chunk_numel = shard_numel // chunks
    # One slot per sibling for our own node; nodes-1 slots for the rest.
    slots = 1 if source == "shard" else nodes - 1
    blocks = (local_world_size - 1) * slots * chunks

    @T.prim_func
    def main(
        shard: T.Tensor((shard_numel,), dtype),
        out: T.Tensor((world_size * shard_numel,), dtype),
        rank: T.int32,
    ):
        with T.Kernel(blocks, threads=threads) as bx:
            c = bx % chunks
            k = (bx // chunks) % slots
            step = (bx // chunks) // slots
            local_rank = rank % local_world_size
            node = rank // local_world_size
            # Rotate so concurrent CTAs do not all read the same sibling first.
            lp = (local_rank + step + 1) % local_world_size
            peer = rank - local_rank + lp  # global rank of local rank lp
            if source == "shard":
                T.get_block(
                    src=T.address_of(shard[c * chunk_numel]),
                    dst=T.address_of(out[(node * local_world_size + lp) * shard_numel +
                                         c * chunk_numel]),
                    size=chunk_numel,
                    src_pe=peer,
                )
            else:
                n = (node + k + 1) % nodes
                # The arena is symmetric, so one offset names the same bytes on both
                # sides.
                off = (n * local_world_size + lp) * shard_numel + c * chunk_numel
                T.get_block(
                    src=T.address_of(out[off]),
                    dst=T.address_of(out[off]),
                    size=chunk_numel,
                    src_pe=peer,
                )

    return main


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--no-overlap", action="store_true",
                        help="run the own-node pull after the barrier instead of "
                             "concurrently with the fabric transfer")
    args = parser.parse_args()

    prepare_env()
    ctx = Context()

    lws = ctx.local_world_size
    if ctx.world_size % lws:
        raise SystemExit(f"world_size {ctx.world_size} must be divisible by LOCAL_WORLD_SIZE {lws}")
    nodes = ctx.world_size // lws
    if nodes < 2:
        raise SystemExit("this example is for >= 2 nodes; use the flat allgather otherwise")
    if args.numel % ctx.world_size:
        raise SystemExit(f"--numel {args.numel} must be divisible by world_size {ctx.world_size}")
    shard_numel = args.numel // ctx.world_size
    if shard_numel % args.chunks:
        raise SystemExit(f"shard {shard_numel} must be a multiple of --chunks {args.chunks}")

    torch_dtype = TORCH_DTYPES[args.dtype]
    tl_dtype = TL_DTYPES[args.dtype]
    ctx.log(
        f"allgather_2d: world={ctx.world_size} nodes={nodes} local={lws} "
        f"numel={args.numel} shard={shard_numel} chunks={args.chunks} "
        f"contexts={args.gin_contexts} overlap={not args.no_overlap} dtype={args.dtype}"
    )

    inter = ctx.compile(
        ag2d_inter_kernel(shard_numel, args.chunks, args.threads, ctx.world_size, lws, tl_dtype),
        expect=("tl::gin::put_signal_addr", "tl::gin::wait_signal"),
        gin_contexts=args.gin_contexts,
    )
    intra_args = (shard_numel, args.chunks, args.threads, ctx.world_size, lws, tl_dtype)
    intra_local = ctx.compile(ag2d_intra_kernel(*intra_args, source="shard"))
    intra_remote = ctx.compile(ag2d_intra_kernel(*intra_args, source="out"))

    shard = ctx.tensor((shard_numel,), torch_dtype)
    out = ctx.tensor((args.numel,), torch_dtype)
    shard.copy_(
        torch.arange(shard_numel, device=shard.device, dtype=torch.float32).to(torch_dtype)
        + ctx.rank * 1000.0
    )
    out.zero_()

    # Only the rail peers signal us, so the per-launch total is (nodes-1)*chunks.
    per_launch = (nodes - 1) * args.chunks
    target = [0]
    side = torch.cuda.Stream()

    def launch():
        target[0] += per_launch
        main_stream = torch.cuda.current_stream()
        if args.no_overlap:
            inter(shard, out, ctx.rank, target[0])
            dist.barrier(ctx.group)
            intra_local(shard, out, ctx.rank)
        else:
            # Siblings' inputs already exist, so this races the NIC rather than
            # waiting behind it.
            side.wait_stream(main_stream)
            with torch.cuda.stream(side):
                intra_local(shard, out, ctx.rank)
            inter(shard, out, ctx.rank, target[0])
            main_stream.wait_stream(side)
            dist.barrier(ctx.group)
        intra_remote(shard, out, ctx.rank)

    torch.cuda.synchronize()
    dist.barrier(ctx.group)
    launch()
    torch.cuda.synchronize()

    ref = torch.empty_like(out)
    dist.all_gather_into_tensor(ref, shard, group=ctx.group)
    failures = check(ctx, out, ref, "allgather_2d")

    if not args.no_bench and failures == 0:
        dist.barrier(ctx.group)

        def run_ref():
            dist.all_gather_into_tensor(ref, shard, group=ctx.group)

        # Time torch first and last: on a shared cluster its NCCL number drifts by
        # more than the effect we are measuring.
        pre_ms = do_bench(run_ref, warmup=args.warmup, rep=args.rep, group=ctx.group)
        tl_ms = do_bench(launch, warmup=args.warmup, rep=args.rep, group=ctx.group)
        post_ms = do_bench(run_ref, warmup=args.warmup, rep=args.rep, group=ctx.group)
        moved = shard.numel() * shard.element_size() * (ctx.world_size - 1)
        report(ctx, "allgather_2d", tl_ms, min(pre_ms, post_ms), moved)
        ctx.log(f"  torch reps: {pre_ms:.3f} / {post_ms:.3f} ms (drift shows contention)")

    ctx.close()
    if ctx.is_leader:
        print("PASS" if failures == 0 else f"FAIL: {failures} rank(s) mismatched", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
