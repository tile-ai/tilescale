"""Hierarchical (2D) inter-node reduce-scatter: NVLink reduce, then rail-aligned GIN.

The transpose of ``example_internode_allgather_2d.py``, and it collapses at 16 GPUs
for the same reason the flat allgather does: the flat reduce-scatter has every rank
push a slice to all ``world_size - 1`` peers over GIN, including the 7 siblings on
the same machine.

View the input as ``[nodes][local_world_size]`` segments of ``shard_numel``, indexed
by the owning rank. Rank ``(node, l)`` wants the sum over all 16 ranks of segment
``(node, l)``. Split that sum by node:

    sum over ranks  =  sum over n' of ( sum over l' of rank (n',l')'s segment )

which is exactly two phases:

* ``intra`` -- **intra-node reduce-scatter over NVLink.** Each rank pulls, from every
  local sibling, the part of that sibling's input belonging to *our* rail index ``l``,
  and sums it. We end holding a partial sum for segment ``(n, l)`` for every ``n``.
* ``rail`` -- **rail-aligned inter-node exchange.** Segment ``(n, l)``'s partial goes
  to rank ``(n, l)``, our rail partner on node ``n``; we receive ``nodes`` partials
  for our own segment and sum them.

No host barrier
---------------
Unlike the 2D allgather, this needs none. ``intra`` reads siblings' *input* buffers,
which are untouched by the collective, so nothing has to happen first. ``rail``
sends only bytes this rank produced, and its GIN signal already proves the arrivals
landed. The two kernels are ordered by the stream, and that is sufficient.

That also means the reduction direction is the cheap one to fuse with a GEMM: see
``example_internode_gemm_rs.py``.

Overlapping the two phases along the node axis
----------------------------------------------
Run back to back, the NVLink reduce (~0.5 ms) and the NIC transfer (~0.3 ms) are
strictly serial. They can overlap, but *which axis* you cut matters:

* **By chunk (does not work).** Splitting the shard into groups so group *g*'s
  transfer overlaps group *g+1*'s reduce measures 3.3 ms at 4 groups against 0.81 ms
  unsplit -- 4x *worse*. Each group's rail launch is then only ``chunks/groups`` CTAs
  moving a few MB, and RDMA is bandwidth-bound only once messages are large. This is
  the same trap as sizing the grid by payload instead of by peers and channels.
* **By node slot (what this does).** The partial for node ``n`` only needs the reduce
  for slot ``n``. So reduce the *other* nodes' slots first, start their transfer on a
  side stream, and reduce our *own* slot -- which nothing on the network waits for --
  concurrently. The transfer stays one whole message.

  This is worth only 3.6% (0.833 -> 0.803 ms), much less than the phase times suggest,
  because ``put`` returns once the RDMA is *issued*: the flight time was already being
  absorbed by ``reduce``'s ``wait_signal``. What is left on the critical path is the
  NVLink reduce itself, and the remaining 8% gap to torch is there -- staging each
  sibling's slice in ``scratch`` before summing costs ~500 MB of avoidable HBM traffic
  per rank. Fusing the remote read into the sum needs a vectorised elementwise peer
  load; ``T.ld`` is scalar and the ``fp32_sum`` accumulator restriction rules out the
  obvious shapes. That is the next thing to try.

Hence four kernels rather than two: ``intra`` parameterised by ``slots="remote"|"own"``
(the same shape as the 2D allgather's ``source`` parameter), then ``put``, then
``reduce``. ``--no-overlap`` runs them serially for comparison.

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
    fp32_sum,
    prepare_env,
    report,
)


def rs2d_intra_kernel(
    shard_numel: int, chunks: int, threads: int, world_size: int, local_world_size: int,
    dtype: str, slots: str,
):
    """Intra-node reduce-scatter: pull every sibling's slice for our rail index, sum it.

    ``slots="remote"`` handles the node slots whose partials get sent over the fabric;
    ``slots="own"`` handles the slot we keep, which no peer waits for and so can run
    concurrently with that transfer.

    One CTA per (node slot, chunk), so ``chunks`` alone sets the grid -- 16 CTAs on
    148 SMs at the rail kernel's ``--chunks 8``, which measured 0.31x of torch. This
    kernel carries no GIN signal, so its chunk count is free to differ from the rail
    kernel's and is set by ``--intra-chunks`` (1024 measured best, 0.91x; the gain is
    flat past ~512).

    The pull lands each sibling's slice in its own
    ``scratch`` slot and the sum is a separate pass, because ``T.get_block`` is a bulk
    copy with no reduce-on-arrival -- the same reason the flat kernels reduce on the
    receiving side.

    ``src_pe`` is a **global** rank: ``get_remote_base_ptr`` returns 0 for a peer it
    considers inter-node, so a local rank yields a null base and faults on every node
    but node 0, where the two numberings coincide.
    """
    nodes = world_size // local_world_size
    chunk_numel = shard_numel // chunks
    node_slots = nodes - 1 if slots == "remote" else 1

    @T.prim_func
    def main(
        inp: T.Tensor((world_size * shard_numel,), dtype),
        scratch: T.Tensor((world_size * shard_numel,), dtype),
        partial: T.Tensor((nodes * shard_numel,), dtype),
        rank: T.int32,
    ):
        with T.Kernel(node_slots * chunks, threads=threads) as bx:
            c = bx % chunks
            k = bx // chunks
            local_rank = rank % local_world_size
            node_base = rank - local_rank  # global rank of local rank 0
            node = rank // local_world_size
            n = (node + k + 1) % nodes if slots == "remote" else node
            # Every sibling holds a slice destined for our rail index at the *same*
            # symmetric offset, so only the peer changes across the loop.
            base = c * chunk_numel
            src_off = (n * local_world_size + local_rank) * shard_numel + base
            for step in range(local_world_size):
                lp = (local_rank + step) % local_world_size
                T.get_block(
                    src=T.address_of(inp[src_off]),
                    dst=T.address_of(scratch[(n * local_world_size + lp) * shard_numel + base]),
                    size=chunk_numel,
                    src_pe=node_base + lp,
                )
            # cp_block spreads the copy over the whole CTA, so the sum below must not
            # start before every thread's share of it has landed.
            T.sync_threads()
            for i in T.Parallel(chunk_numel):
                partial[n * shard_numel + base + i] = T.cast(
                    fp32_sum(
                        local_world_size,
                        lambda s: T.cast(
                            scratch[(n * local_world_size + s) * shard_numel + base + i],
                            "float32"),
                    ),
                    dtype,
                )

    return main


def rs2d_put_kernel(
    shard_numel: int, chunks: int, threads: int, world_size: int, local_world_size: int,
    dtype: str, signal_id: int = SIGNAL_DATA,
):
    """Send each other node's partial to its rail owner. One whole message per CTA."""
    nodes = world_size // local_world_size
    chunk_numel = shard_numel // chunks

    @T.prim_func
    def main(
        partial: T.Tensor((nodes * shard_numel,), dtype),
        railbuf: T.Tensor((nodes * shard_numel,), dtype),
        rank: T.int32,
    ):
        with T.Kernel(chunks, threads=threads) as bx:
            base = bx * chunk_numel
            local_rank = rank % local_world_size
            node = rank // local_world_size
            for step in range(nodes - 1):
                n = (node + step + 1) % nodes
                # Rail partner on node n. Senders to one receiver come from distinct
                # nodes, so indexing the inbox by sender node keeps slots disjoint.
                T.nccl_gin.put_signal(
                    src=partial[n * shard_numel + base],
                    dst=railbuf[node * shard_numel + base],
                    size=chunk_numel,
                    peer=n * local_world_size + local_rank,
                    signal_id=signal_id,
                    scope="block",
                )

    return main


def rs2d_reduce_kernel(
    shard_numel: int, chunks: int, threads: int, world_size: int, local_world_size: int,
    dtype: str, signal_id: int = SIGNAL_DATA,
):
    """Add our own node's partial, wait for the rail arrivals, sum the ``nodes`` slots."""
    nodes = world_size // local_world_size
    chunk_numel = shard_numel // chunks

    @T.prim_func
    def main(
        partial: T.Tensor((nodes * shard_numel,), dtype),
        railbuf: T.Tensor((nodes * shard_numel,), dtype),
        out: T.Tensor((shard_numel,), dtype),
        rank: T.int32,
        signal_target: T.int32,
    ):
        with T.Kernel(chunks, threads=threads) as bx:
            base = bx * chunk_numel
            node = rank // local_world_size
            for i in T.Parallel(chunk_numel):
                railbuf[node * shard_numel + base + i] = partial[node * shard_numel + base + i]
            T.nccl_gin.wait_signal(least=signal_target, signal_id=signal_id, scope="block")
            for i in T.Parallel(chunk_numel):
                out[base + i] = T.cast(
                    fp32_sum(
                        nodes,
                        lambda s: T.cast(railbuf[s * shard_numel + base + i], "float32"),
                    ),
                    dtype,
                )

    return main


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--intra-chunks", type=int, default=1024,
                        help="chunking of the intra-node phase; sets its grid to "
                             "nodes*intra_chunks/groups and is independent of --chunks")
    parser.add_argument("--no-overlap", action="store_true",
                        help="reduce our own node's slot before the fabric transfer "
                             "instead of concurrently with it")
    args = parser.parse_args()

    prepare_env()
    ctx = Context()

    lws = ctx.local_world_size
    if ctx.world_size % lws:
        raise SystemExit(f"world_size {ctx.world_size} must be divisible by LOCAL_WORLD_SIZE {lws}")
    nodes = ctx.world_size // lws
    if nodes < 2:
        raise SystemExit("this example is for >= 2 nodes; use the flat reduce-scatter otherwise")
    if args.numel % ctx.world_size:
        raise SystemExit(f"--numel {args.numel} must be divisible by world_size {ctx.world_size}")
    shard_numel = args.numel // ctx.world_size
    if shard_numel % args.chunks:
        raise SystemExit(f"shard {shard_numel} must be a multiple of --chunks {args.chunks}")
    if shard_numel % args.intra_chunks:
        raise SystemExit(
            f"shard {shard_numel} must be a multiple of --intra-chunks {args.intra_chunks}")
    # wait_signal divides the grid-wide target by the granted context count, so a rail
    # grid smaller than that count rounds the target down -- to 0 in the worst case,
    # which turns the wait into a no-op.
    if args.chunks % args.gin_contexts:
        raise SystemExit(
            f"--chunks {args.chunks} must be a multiple of --gin-contexts "
            f"{args.gin_contexts}, or the signal target rounds down")

    torch_dtype = TORCH_DTYPES[args.dtype]
    tl_dtype = TL_DTYPES[args.dtype]
    ctx.log(
        f"reduce_scatter_2d: world={ctx.world_size} nodes={nodes} local={lws} "
        f"numel={args.numel} shard={shard_numel} chunks={args.chunks} "
        f"intra_chunks={args.intra_chunks} overlap={not args.no_overlap} "
        f"contexts={args.gin_contexts} dtype={args.dtype}"
    )

    intra_args = (shard_numel, args.intra_chunks, args.threads, ctx.world_size, lws, tl_dtype)
    intra_remote = ctx.compile(rs2d_intra_kernel(*intra_args, slots="remote"))
    intra_own = ctx.compile(rs2d_intra_kernel(*intra_args, slots="own"))
    rail_args = (shard_numel, args.chunks, args.threads, ctx.world_size, lws, tl_dtype)
    put = ctx.compile(
        rs2d_put_kernel(*rail_args),
        expect=("tl::gin::put_signal_addr",),
        gin_contexts=args.gin_contexts,
    )
    reduce_ = ctx.compile(
        rs2d_reduce_kernel(*rail_args),
        expect=("tl::gin::wait_signal",),
        gin_contexts=args.gin_contexts,
    )

    inp = ctx.tensor((args.numel,), torch_dtype)
    scratch = ctx.tensor((args.numel,), torch_dtype)
    partial = ctx.tensor((nodes * shard_numel,), torch_dtype)
    railbuf = ctx.tensor((nodes * shard_numel,), torch_dtype)
    out = ctx.tensor((shard_numel,), torch_dtype)
    # Small magnitudes: a bf16 sum over 16 ranks of arange values would land outside
    # any sensible tolerance.
    inp.copy_(
        (torch.arange(args.numel, device=inp.device, dtype=torch.float32) % 7 + ctx.rank)
        .to(torch_dtype)
    )
    for t in (scratch, partial, railbuf, out):
        t.zero_()

    per_launch = (nodes - 1) * args.chunks
    target = [0]
    side = torch.cuda.Stream()
    ready = torch.cuda.Event()

    def launch():
        target[0] += per_launch
        main_stream = torch.cuda.current_stream()
        # The other nodes' slots are what the fabric transfer needs, so they go first.
        intra_remote(inp, scratch, partial, ctx.rank)
        if args.no_overlap:
            intra_own(inp, scratch, partial, ctx.rank)
            put(partial, railbuf, ctx.rank)
        else:
            ready.record(main_stream)
            side.wait_event(ready)
            with torch.cuda.stream(side):
                put(partial, railbuf, ctx.rank)
            # Nothing on the network waits for our own slot, so it reduces while the
            # message is in flight.
            intra_own(inp, scratch, partial, ctx.rank)
            main_stream.wait_stream(side)
        reduce_(partial, railbuf, out, ctx.rank, target[0])

    torch.cuda.synchronize()
    dist.barrier(ctx.group)
    launch()
    torch.cuda.synchronize()

    ref = torch.empty_like(out)
    dist.reduce_scatter_tensor(ref, inp, op=dist.ReduceOp.SUM, group=ctx.group)
    failures = check(ctx, out, ref, "reduce_scatter_2d")

    if not args.no_bench and failures == 0:
        ref_buf = torch.empty_like(out)

        def run_ref():
            dist.reduce_scatter_tensor(ref_buf, inp, op=dist.ReduceOp.SUM, group=ctx.group)

        # Time torch either side of ourselves: on a shared cluster its first reading
        # is cold by more than the effect being measured.
        dist.barrier(ctx.group)
        pre_ms = do_bench(run_ref, warmup=args.warmup, rep=args.rep, group=ctx.group)
        tl_ms = do_bench(launch, warmup=args.warmup, rep=args.rep, group=ctx.group)
        post_ms = do_bench(run_ref, warmup=args.warmup, rep=args.rep, group=ctx.group)
        moved = out.numel() * out.element_size() * (ctx.world_size - 1)
        report(ctx, "reduce_scatter_2d", tl_ms, min(pre_ms, post_ms), moved)
        ctx.log(f"  torch reps: {pre_ms:.3f} / {post_ms:.3f} ms (drift shows contention)")

    ctx.close()
    if ctx.is_leader:
        print("PASS" if failures == 0 else f"FAIL: {failures} rank(s) mismatched", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
