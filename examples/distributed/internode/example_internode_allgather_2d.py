"""Hierarchical (2D) inter-node allgather: rail-aligned GIN plus an NVSwitch broadcast.

The flat allgather in ``example_internode_allgather.py`` pushes every rank's shard
to all ``world_size - 1`` peers over GIN. That is optimal with one GPU per node and
catastrophic with eight: each rank puts 15 shards onto the NIC, including the 7
peers sitting on the same machine, and it measures 2.7 GB/s against torch's 309.

Decompose along the topology instead. Only ``1/local_world_size`` of the data has to
cross the fabric, and it crosses **rail-aligned** -- rank ``(node, local)`` exchanges
only with the same ``local`` index on other nodes, so every rank drives its own NIC
with no fan-out. The rest is finished inside the node.

Two implementations of that intra-node half, ``--intra multimem|pull``:

* ``multimem`` (default where the hardware allows it) issues ``multimem.st`` against
  the multicast VA, so **one store reaches every local rank** and the NVSwitch does
  the fan-out. A rank writes only the two shards it owns, ~31 MB, rather than reading
  14 of them.
* ``pull`` reads each sibling's buffer with ``T.get_block``. Portable, and the only
  option without multicast.

Measured, 16 GPUs on 2 nodes, 240 MB bf16:

```
flat              2.7 GB/s
2D pull         358.6 GB/s   torch 309   1.16x
2D multimem     403.6 GB/s   torch 309   1.31x
```

``--mc-tiles`` matters more than anything else here: 8 tiles per CTA gives 355 GB/s,
32 gives 404, and it is flat from there (64/128/256 all within noise). Below 32 each
thread is doing too little to cover the store latency, since the tile width is pinned.

Overlap: what the phases actually depend on
-------------------------------------------
Both paths hinge on the same observation. A rank's *own* shard already exists before
the collective starts, so publishing it is ordered against nothing and runs on a side
stream **concurrently with the fabric transfer**. Only the other nodes' shards depend
on the network. ``--no-overlap`` measures 311 GB/s against 404, and the pull path
without it was 260 -- below torch -- so this is where the speed is, not in the 2D
decomposition alone.

The two broadcast kernels therefore differ only in which buffer they read, and are one
kernel parameterised by ``source``.

Barriers
--------
* ``multimem``: a rank writes only the slots it owns, into everyone. Nothing reads a
  peer's buffer, so no barrier is needed *during* the collective -- but the output is
  complete only once every sibling has published, hence one barrier at the end.
* ``pull``: reads a sibling's output, so it must not start until that sibling's rail
  transfer has landed, which stream order cannot express -- hence one barrier in the
  middle. A grid-wide device barrier across ranks would need every participating CTA
  co-resident, the deadlock documented in ``example_internode_ag_gemm.py``.

Either way ``dist.barrier`` costs tens of microseconds against a collective of a few
hundred. Replacing it with a device-side arrive/wait on a symmetric flag (the
``*_barrier_gpu`` helpers in ``tilelang/language/distributed/sync.py``) is the obvious
next step, and none of the kernels change when it lands.

Requires real intra-node peer pointers, so it does **not** work under
``TILESCALE_SKIP_PEER_IPC=1``.
"""

# NOTE: no `from __future__ import annotations` here -- see the allgather example.
import argparse
import functools
import os

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


def ag2d_rail_kernel(
    shard_numel: int, chunks: int, threads: int, world_size: int, local_world_size: int,
    dtype: str, signal_id: int = SIGNAL_DATA, to_out: bool = False,
):
    """Rail-aligned inter-node exchange: our shard to the same local index elsewhere.

    The inbox is indexed by *sender node*, which keeps slots disjoint without any
    rotation arithmetic: only rank ``(n, l)`` ever writes our slot ``n``.

    ``to_out=True`` writes straight into the output buffer at the sender's global rank
    slot, which is what the pull path wants; the multimem path lands in a separate
    ``railbuf`` because its output lives in the multicast buffer and a GIN put has to
    target the registered arena window.
    """
    nodes = world_size // local_world_size
    chunk_numel = shard_numel // chunks
    slots = world_size if to_out else nodes

    @T.prim_func
    def main(
        shard: T.Tensor((shard_numel,), dtype),
        inbox: T.Tensor((slots * shard_numel,), dtype),
        rank: T.int32,
        signal_target: T.int32,
    ):
        with T.Kernel(chunks, threads=threads) as bx:
            base = bx * chunk_numel
            local_rank = rank % local_world_size
            node = rank // local_world_size
            slot = rank if to_out else node
            for step in range(nodes - 1):
                n = (node + step + 1) % nodes
                T.nccl_gin.put_signal(
                    src=shard[base],
                    dst=inbox[slot * shard_numel + base],
                    size=chunk_numel,
                    peer=n * local_world_size + local_rank,
                    signal_id=signal_id,
                    scope="block",
                )
            if to_out:
                # The pull path reads its own slot from `out`, so it has to be filled.
                for i in T.Parallel(chunk_numel):
                    inbox[rank * shard_numel + base + i] = shard[base + i]
            T.nccl_gin.wait_signal(least=signal_target, signal_id=signal_id, scope="block")

    return main


def ag2d_bcast_kernel(
    shard_numel: int, threads: int, world_size: int, local_world_size: int,
    dtype: str, source: str, tiles_per_cta: int = 8,
):
    """Publish a shard to every local rank's output slot with one ``multimem.st``.

    ``source="shard"`` publishes our own shard, which exists up front and so needs no
    ordering. ``source="rail"`` publishes the shards that arrived from other nodes.

    ``block_N`` is forced to ``2 * threads`` and is not tunable: bf16 multimem lowers
    to packed x2 instructions, so the staging fragment's layout must be exactly one
    contiguous pair per thread. Any other tile width lets the paired ``T.copy`` infer a
    wider vectorisation and layout inference fails outright with "requires the local
    fragment layout to preserve canonical pair ownership". That pins each thread at one
    4-byte store per tile, so ``tiles_per_cta`` is the only work-per-thread knob.
    """
    nodes = world_size // local_world_size
    slots = 1 if source == "shard" else nodes - 1
    block_N = 2 * threads
    ctas = (shard_numel // block_N) // tiles_per_cta

    @T.prim_func
    def main(
        src: T.Tensor(((1 if source == "shard" else nodes) * shard_numel,), dtype),
        out_mc: T.Tensor((world_size * shard_numel,), dtype),
        rank: T.int32,
    ):
        with T.Kernel(slots * ctas, threads=threads) as bx:
            c = bx % ctas
            k = bx // ctas
            local_rank = rank % local_world_size
            node = rank // local_world_size
            n = node if source == "shard" else (node + k + 1) % nodes
            # Our own shard is a standalone buffer; rail arrivals are indexed by the
            # node that sent them.
            src_base = 0 if source == "shard" else n * shard_numel
            dst_base = (n * local_world_size + local_rank) * shard_numel
            buf = T.alloc_fragment((block_N,), dtype)
            for j in T.serial(tiles_per_cta):
                off = (c * tiles_per_cta + j) * block_N
                T.copy(src[src_base + off:src_base + off + block_N], buf)
                T.multimem_st(buf, out_mc[dst_base + off:dst_base + off + block_N])

    return main


def ag2d_pull_kernel(
    shard_numel: int, chunks: int, threads: int, world_size: int, local_world_size: int,
    dtype: str, source: str,
):
    """Portable intra-node half: read each sibling's buffer with ``T.get_block``.

    ``source="shard"`` reads siblings' inputs to fill our own node's slots;
    ``source="out"`` reads their outputs for the other nodes' slots.

    ``src_pe`` is a **global** rank. ``get_remote_base_ptr`` returns 0 for a peer it
    considers inter-node, so passing a local rank yields a null base and faults on
    every node but node 0, where the two numberings coincide.
    """
    nodes = world_size // local_world_size
    chunk_numel = shard_numel // chunks
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
    parser.add_argument("--intra", choices=("multimem", "pull", "auto"), default="auto",
                        help="intra-node half: NVSwitch multimem.st broadcast, the "
                             "portable get_block pull, or multimem when available")
    parser.add_argument("--mc-threads", type=int, default=512,
                        help="threads per CTA on the multimem path; the tile is "
                             "2*threads, fixed by the packed-x2 fragment layout")
    parser.add_argument("--mc-tiles", type=int, default=32,
                        help="contiguous tiles each multimem CTA loops over; the tile "
                             "width is pinned, so this is the only work-per-thread knob")
    parser.add_argument("--no-overlap", action="store_true",
                        help="publish our own shard after the barrier instead of "
                             "concurrently with the fabric transfer")
    args = parser.parse_args()

    prepare_env()
    # The multicast buffer is sized before the allocator exists, so the output length
    # has to be known here rather than after Context.
    world = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    itemsize = torch.empty((), dtype=TORCH_DTYPES[args.dtype]).element_size()
    intra_mode = args.intra
    if intra_mode == "auto":
        intra_mode = "multimem" if Context.supports_multicast() else "pull"
    ctx = Context(mcast_bytes=args.numel * itemsize if intra_mode == "multimem" else 0)

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
    if intra_mode == "multimem":
        unit = 2 * args.mc_threads * args.mc_tiles
        if shard_numel % unit:
            raise SystemExit(
                f"shard {shard_numel} must be a multiple of 2*--mc-threads*--mc-tiles "
                f"= {unit}")

    torch_dtype = TORCH_DTYPES[args.dtype]
    tl_dtype = TL_DTYPES[args.dtype]
    ctx.log(
        f"allgather_2d: world={ctx.world_size} nodes={nodes} local={lws} "
        f"numel={args.numel} shard={shard_numel} chunks={args.chunks} intra={intra_mode} "
        f"mc_threads={args.mc_threads} mc_tiles={args.mc_tiles} "
        f"overlap={not args.no_overlap} contexts={args.gin_contexts} dtype={args.dtype}"
    )

    use_mc = intra_mode == "multimem"
    rail = ctx.compile(
        ag2d_rail_kernel(shard_numel, args.chunks, args.threads, ctx.world_size, lws,
                         tl_dtype, to_out=not use_mc),
        expect=("tl::gin::put_signal_addr", "tl::gin::wait_signal"),
        gin_contexts=args.gin_contexts,
    )
    if use_mc:
        build = functools.partial(
            ag2d_bcast_kernel, shard_numel, args.mc_threads, ctx.world_size, lws, tl_dtype,
            tiles_per_cta=args.mc_tiles,
        )
        pub_own = ctx.compile(build(source="shard"))
        pub_remote = ctx.compile(build(source="rail"))
    else:
        build = functools.partial(
            ag2d_pull_kernel, shard_numel, args.chunks, args.threads, ctx.world_size, lws,
            tl_dtype,
        )
        pub_own = ctx.compile(build(source="shard"))
        pub_remote = ctx.compile(build(source="out"))

    shard = ctx.tensor((shard_numel,), torch_dtype)
    if use_mc:
        out_mc, out = ctx.mcast_tensor((args.numel,), torch_dtype)
        railbuf = ctx.tensor((nodes * shard_numel,), torch_dtype)
        railbuf.zero_()
    else:
        out = ctx.tensor((args.numel,), torch_dtype)
        out_mc, railbuf = out, out
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
        if use_mc:
            if args.no_overlap:
                rail(shard, railbuf, ctx.rank, target[0])
                pub_own(shard, out_mc, ctx.rank)
            else:
                # Our own shard exists already, so publishing it races the NIC rather
                # than waiting behind it.
                side.wait_stream(main_stream)
                with torch.cuda.stream(side):
                    pub_own(shard, out_mc, ctx.rank)
                rail(shard, railbuf, ctx.rank, target[0])
                main_stream.wait_stream(side)
            pub_remote(railbuf, out_mc, ctx.rank)
            # We wrote into every sibling, and they wrote into us: our output is only
            # complete once they have all finished.
            dist.barrier(ctx.group)
        else:
            if args.no_overlap:
                rail(shard, out, ctx.rank, target[0])
                dist.barrier(ctx.group)
                pub_own(shard, out, ctx.rank)
            else:
                side.wait_stream(main_stream)
                with torch.cuda.stream(side):
                    pub_own(shard, out, ctx.rank)
                rail(shard, out, ctx.rank, target[0])
                main_stream.wait_stream(side)
                # The pull below reads a sibling's output, which only its rail transfer
                # fills; nothing in stream order says that has happened.
                dist.barrier(ctx.group)
            pub_remote(shard, out, ctx.rank)

    torch.cuda.synchronize()
    dist.barrier(ctx.group)
    launch()
    torch.cuda.synchronize()

    ref = torch.empty(args.numel, dtype=torch_dtype, device=out.device)
    dist.all_gather_into_tensor(ref, shard, group=ctx.group)
    failures = check(ctx, out, ref, "allgather_2d")

    if not args.no_bench and failures == 0:
        dist.barrier(ctx.group)

        def run_ref():
            dist.all_gather_into_tensor(ref, shard, group=ctx.group)

        # Time torch either side of ourselves: its first reading on a shared cluster is
        # cold by more than the effect being measured.
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
