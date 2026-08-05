"""Hierarchical (2D) inter-node collectives, as reusable pieces.

Everything here exists because a *flat* collective collapses once there is more than
one GPU per node: each rank puts ``world_size - 1`` shards onto the NIC, including the
ones destined for siblings on the same machine. At 16 GPUs that measures ~2.6 GB/s
against torch's ~310.

The fix is to split every collective along the topology, and all four collectives in
this directory reduce to the same two halves:

* **rail** -- the fabric hop. Rank ``(node, local)`` exchanges only with the *same*
  ``local`` index on other nodes, ``peer = other_node * lws + local``. Every rank
  drives its own NIC with no fan-out, and only ``1/lws`` of the data crosses.
* **intra** -- finish inside the node. Either through the NVSwitch (``multimem.st`` to
  broadcast, ``multimem.ld_reduce`` to reduce, one instruction reaching every local
  rank) or, without multicast, by reading peers' buffers with ``T.get_block``.

Allgather is rail-then-broadcast, reduce-scatter is reduce-then-rail, allreduce is the
two composed, and the fused GEMM kernels wrap one of them around
``tcgen05_gemm_range_kernel``. So they share these kernels rather than restating them.

Measured at 16 GPUs on 2 nodes, 240 MB bf16 (bus bandwidth, ``nbytes*(W-1)/W/time``):

```
                   flat    2D pull   2D multimem    torch   best vs torch
allgather           2.7      358.6         403.6    309.0       1.31x
reduce_scatter      2.5      293.9         397.2    321.6       1.24x
```

The roofline is ~700 GB/s: each GPU must take one shard over its own NIC
(15.7 MB / 47.6 GB/s = 0.33 ms) and 14 over NVLink (220 MB / 670 GB/s = 0.33 ms), and
those two legs are almost perfectly balanced at 8 GPUs against 8 400-Gbps NICs. The
remaining gap is that the second intra-node phase cannot start until the fabric hop
has landed; see ``Allgather2D`` for what overlaps and what does not.

Three traps worth knowing before editing any of this
----------------------------------------------------
1. **``src_pe``/``dst_pe`` are GLOBAL ranks.** ``get_remote_base_ptr`` returns 0 for a
   peer it considers inter-node, so a *local* rank yields a null base and faults --
   but only on nodes other than node 0, where the two numberings coincide.
2. **A multimem tile is exactly ``2 * threads`` wide, and that is not tunable.** bf16
   lowers to packed x2, so the staging fragment must be one contiguous pair per
   thread; any other width lets the paired ``T.copy`` infer a wider vectorisation and
   layout inference fails with "requires the local fragment layout to preserve
   canonical pair ownership". Work per thread therefore has to come from
   ``tiles_per_cta``, and it matters a lot: 1 tile gives 317 GB/s, 32 gives ~400.
3. **``wait_signal`` divides the grid-wide target by the granted GIN context count**,
   so a rail grid smaller than that count rounds the target down -- to 0 in the worst
   case, which turns the wait into a silent no-op. Keep ``chunks % gin_contexts == 0``.
"""

# NOTE: no `from __future__ import annotations` here. T.prim_func resolves parameter
# annotations at runtime via get_type_hints, and PEP 563 would turn `T.Tensor(...)`
# into a string evaluated against module globals, where the closure locals do not
# exist.
import functools

import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T

from internode_common import SIGNAL_DATA, fp32_sum

# Smallest per-group slice worth pipelining; below this the extra launches cost more than
# the overlap saves. See _Base.__init__ for the measurement.
MIN_GROUP_BYTES = 1_500_000

# --mc-tiles has to scale with the buffer for the reduce-carrying collectives, and getting
# it wrong is expensive in both directions. Measured at 16 GPUs: allreduce wants 32 tiles
# per CTA at 240 MiB (0.944 ms against 1.086 at 4) and 4 at 48 MiB (0.282 against 0.361 at
# 32) -- a fat-CTA publish starves the GPU once the shard is small, and a thin one wastes
# scheduling once it is large. reduce_scatter behaves the same way (48 MiB: 0.138 ms at the
# scaled value against 0.194 at 32).
#
# Allgather does *not*: it wants 32 at every size measured (48 MiB 0.164 ms at 32 against
# 0.233 at 4; 240 MiB 0.501 against 0.585). Its intra-node half is two publishes and no
# reduce, so it is switch-bound rather than occupancy-bound, and fewer fatter CTAs win.
# Hence the example overrides this rather than a single global law covering both.
# One tile per 480 KB of shard. 240 KB was tried and regressed allreduce at 120 MiB
# (0.747 ms against 0.521), so the coarser divisor stands. This is a fit to three sizes,
# not a law: the true optimum also depends on the group count and on which collective, and
# --mc-tiles overrides it. See the tuning table in CLAUDE.md for what is left on the table.
MC_TILE_BYTES_PER_TILE = 480 << 10


def pick_mc_tiles(shard_numel, itemsize, requested):
    """Tiles per multicast CTA: as asked if given, else scaled to the shard. See above."""
    if requested:
        return requested
    want = max(1, (shard_numel * itemsize) // MC_TILE_BYTES_PER_TILE)
    tiles = 1
    while tiles * 2 <= min(want, 32):
        tiles *= 2
    return tiles


# --------------------------------------------------------------------- fabric hop


def rail_put_kernel(
    shard_numel: int, chunks: int, threads: int, world_size: int, local_world_size: int,
    dtype: str, signal_id: int = SIGNAL_DATA, src_per_node: bool = False,
    wait: bool = False, chunk_lo: int = 0, chunk_count: int = 0,
):
    """Rail-aligned put of one shard to the same local index on every other node.

    ``chunk_lo``/``chunk_count`` restrict the launch to a slice of the shard, for the
    pipelined path: the grid becomes ``chunk_count`` CTAs covering chunks
    ``[chunk_lo, chunk_lo + chunk_count)``. **The per-put size is unchanged** -- only the
    number of concurrent puts drops -- which is the whole point. An earlier attempt at
    pipelining split the shard into more, smaller messages and came out 4x slower,
    because RDMA is bandwidth-bound only once messages are large.

    The inbox is indexed by *sender node*, which keeps slots disjoint with no rotation
    arithmetic: only rank ``(n, l)`` ever writes our slot ``n``.

    ``src_per_node`` selects what is sent. Allgather sends the same shard to every rail
    peer (one source slot); reduce-scatter sends a different per-node partial to each
    (``nodes`` source slots, indexed by destination node).

    ``wait=True`` folds the arrival wait into this launch. Leave it False when the
    consumer is a separate kernel that can wait itself, which lets the put kernel
    retire and free its SMs while the RDMA is still in flight.
    """
    nodes = world_size // local_world_size
    chunk_numel = shard_numel // chunks
    src_slots = nodes if src_per_node else 1
    grid = chunk_count or chunks

    @T.prim_func
    def main(
        src: T.Tensor((src_slots * shard_numel,), dtype),
        inbox: T.Tensor((nodes * shard_numel,), dtype),
        rank: T.int32,
        signal_target: T.int32,
    ):
        with T.Kernel(grid, threads=threads) as bx:
            base = (chunk_lo + bx) * chunk_numel
            local_rank = rank % local_world_size
            node = rank // local_world_size
            for step in range(nodes - 1):
                n = (node + step + 1) % nodes
                src_off = (n * shard_numel if src_per_node else 0) + base
                T.nccl_gin.put_signal(
                    src=src[src_off],
                    dst=inbox[node * shard_numel + base],
                    size=chunk_numel,
                    peer=n * local_world_size + local_rank,
                    signal_id=signal_id,
                    scope="block",
                )
            if wait:
                T.nccl_gin.wait_signal(least=signal_target, signal_id=signal_id,
                                       scope="block")

    return main


def rail_wait_kernel(
    shard_numel: int, chunks: int, threads: int, world_size: int, local_world_size: int,
    dtype: str, signal_id: int = SIGNAL_DATA, chunk_count: int = 0,
):
    """Wait for one group's rail arrivals, nothing else.

    Separate from the put so a group's puts can be issued and the CTAs retire, leaving
    the RDMA in flight while the previous group's data is published over NVLink.

    The grid must match the *sender's* chunk count. ``wait_signal`` divides the target by
    the granted GIN context count, and a put issued from sender CTA ``b`` increments the
    receiver's signal through context ``b % contexts`` -- so a wait grid wider than the
    sender's would park CTAs on contexts nothing ever signals, and a narrower one would
    round the target down. Hence ``chunks % gin_contexts == 0`` per group, checked host
    side.
    """
    nodes = world_size // local_world_size
    grid = chunk_count or chunks

    @T.prim_func
    def main(
        inbox: T.Tensor((nodes * shard_numel,), dtype),
        signal_target: T.int32,
    ):
        with T.Kernel(grid, threads=threads) as bx:
            T.nccl_gin.wait_signal(least=signal_target, signal_id=signal_id, scope="block")
            # Keep `inbox` in the signature: the wait is what makes it readable, so the
            # dependency is real even though this kernel does not touch the bytes.
            if bx >= grid:
                inbox[0] = inbox[0]

    return main


def rail_sum_kernel(
    shard_numel: int, chunks: int, threads: int, world_size: int, local_world_size: int,
    dtype: str, signal_id: int = SIGNAL_DATA,
):
    """Reduce-scatter tail: add our own node's partial, wait, sum the ``nodes`` slots."""
    nodes = world_size // local_world_size
    chunk_numel = shard_numel // chunks

    @T.prim_func
    def main(
        partial: T.Tensor((nodes * shard_numel,), dtype),
        inbox: T.Tensor((nodes * shard_numel,), dtype),
        out: T.Tensor((shard_numel,), dtype),
        rank: T.int32,
        signal_target: T.int32,
    ):
        with T.Kernel(chunks, threads=threads) as bx:
            base = bx * chunk_numel
            node = rank // local_world_size
            for i in T.Parallel(chunk_numel):
                inbox[node * shard_numel + base + i] = partial[node * shard_numel + base + i]
            T.nccl_gin.wait_signal(least=signal_target, signal_id=signal_id, scope="block")
            for i in T.Parallel(chunk_numel):
                out[base + i] = T.cast(
                    fp32_sum(nodes,
                             lambda s: T.cast(inbox[s * shard_numel + base + i], "float32")),
                    dtype,
                )

    return main


def rs_sum_kernel(
    shard_numel: int, chunks: int, threads: int, world_size: int, local_world_size: int,
    dtype: str, elem_lo: int = 0, span_numel: int = 0,
):
    """Reduce-scatter tail for one group: our own partial plus this group's arrivals.

    Like ``allreduce_sum_kernel`` it reads ``partial`` as a term instead of copying it into
    the inbox first, which saves writing and re-reading a shard of HBM, and it carries no
    ``wait_signal`` so its grid is free of the sender's chunk count.
    """
    nodes = world_size // local_world_size
    span = span_numel or shard_numel
    chunk_numel = span // chunks

    @T.prim_func
    def main(
        partial: T.Tensor((nodes * shard_numel,), dtype),
        inbox: T.Tensor((nodes * shard_numel,), dtype),
        out: T.Tensor((shard_numel,), dtype),
        rank: T.int32,
    ):
        with T.Kernel(chunks, threads=threads) as bx:
            base = elem_lo + bx * chunk_numel
            node = rank // local_world_size
            for i in T.Parallel(chunk_numel):
                out[base + i] = T.cast(
                    T.cast(partial[node * shard_numel + base + i], "float32") + fp32_sum(
                        nodes - 1,
                        lambda k: T.cast(
                            inbox[((node + k + 1) % nodes) * shard_numel + base + i],
                            "float32"),
                    ),
                    dtype,
                )

    return main


def allreduce_sum_kernel(
    vec_numel: int, shard_numel: int, chunks: int, threads: int, world_size: int,
    local_world_size: int, dtype: str, slot: int,
):
    """Sum one node slot of the merged allreduce: our own partial plus the arrivals.

    Two things this deliberately does not do.

    It does **not** copy our own partial into the inbox first, as the two-hop tail did --
    it just reads ``partial`` as one more term. That saves writing and re-reading a whole
    shard of HBM per slot.

    It does **not** wait on the signal; ``rail_wait_kernel`` does that separately. Folding
    the wait in would tie this grid to the sender's chunk count, and the first version did
    exactly that: 4 CTAs for a whole shard of copy-and-sum, which made the merged
    allreduce *slower* than composing the two halves (1.559 ms against 1.086). The wait
    needs a narrow grid matched to the sender's contexts; the arithmetic wants a wide one.
    """
    nodes = world_size // local_world_size
    lo = slot * shard_numel
    chunk_numel = shard_numel // chunks

    @T.prim_func
    def main(
        partial: T.Tensor((vec_numel,), dtype),
        inbox: T.Tensor((nodes * vec_numel,), dtype),
        reduced: T.Tensor((vec_numel,), dtype),
        rank: T.int32,
    ):
        with T.Kernel(chunks, threads=threads) as bx:
            base = lo + bx * chunk_numel
            node = rank // local_world_size
            for i in T.Parallel(chunk_numel):
                reduced[base + i] = T.cast(
                    T.cast(partial[base + i], "float32") + fp32_sum(
                        nodes - 1,
                        lambda k: T.cast(
                            inbox[((node + k + 1) % nodes) * vec_numel + base + i],
                            "float32"),
                    ),
                    dtype,
                )

    return main


# ------------------------------------------------------- intra-node via NVSwitch


def mc_bcast_kernel(
    shard_numel: int, threads: int, world_size: int, local_world_size: int, dtype: str,
    slots: str, tiles_per_cta: int = 32, span_numel: int = 0, elem_lo: int = 0,
    node_slot: int = -1,
):
    """Publish the shards we own into every local rank's slot with one ``multimem.st``.

    ``span_numel``/``elem_lo`` publish only that slice of each shard, for the pipelined
    path. Both are **compile-time**: multimem lowering has to prove the multicast region
    is in bounds, and a runtime offset defeats that with "multimem packed multicast region
    must be provably in bounds or use a tile-aligned all-or-none dynamic partition". So a
    group gets its own compiled kernel rather than an offset argument.

    ``slots="own"`` publishes our own shard, which exists before the collective starts
    and so is ordered against nothing. ``slots="remote"`` publishes what arrived over
    the fabric. See trap 2 in the module docstring for why the tile width is fixed.
    """
    nodes = world_size // local_world_size
    groups = 1 if node_slot >= 0 else {"own": 1, "remote": nodes - 1}.get(slots, nodes)
    block_N = 2 * threads
    span = span_numel or shard_numel
    ctas = (span // block_N) // tiles_per_cta

    @T.prim_func
    def main(
        src: T.Tensor(((1 if slots == "own" else nodes) * shard_numel,), dtype),
        out_mc: T.Tensor((world_size * shard_numel,), dtype),
        rank: T.int32,
    ):
        with T.Kernel(groups * ctas, threads=threads) as bx:
            c = bx % ctas
            k = bx // ctas
            local_rank = rank % local_world_size
            node = rank // local_world_size
            if node_slot >= 0:
                n = node_slot  # one absolute slot: the per-slot allreduce pipeline
            elif slots == "own":
                n = node
            elif slots == "remote":
                n = (node + k + 1) % nodes
            else:
                n = k  # "all": every node slot, for the merged allreduce
            src_base = (0 if slots == "own" else n * shard_numel) + elem_lo
            dst_base = (n * local_world_size + local_rank) * shard_numel + elem_lo
            buf = T.alloc_fragment((block_N,), dtype)
            for j in T.serial(tiles_per_cta):
                off = (c * tiles_per_cta + j) * block_N
                T.copy(src[src_base + off:src_base + off + block_N], buf)
                T.multimem_st(buf, out_mc[dst_base + off:dst_base + off + block_N])

    return main


def mc_reduce_kernel(
    shard_numel: int, threads: int, world_size: int, local_world_size: int, dtype: str,
    slots: str, tiles_per_cta: int = 32,
):
    """Sum every local rank's segment for our rail index, reduced by the NVSwitch.

    One ``multimem.ld_reduce`` against the multicast VA returns the sum over every
    bound device, so the rank reads only the bytes it keeps instead of pulling all
    ``lws`` contributions into a staging buffer and reading them back. That staging
    traffic -- ~210 MB of NVLink plus ~500 MB of HBM per rank -- is exactly what kept
    the portable path below under torch.

    ``slots="remote"`` handles the node slots whose partials cross the fabric;
    ``slots="own"`` handles the one we keep, which no peer waits for. ``"all"`` does both in
    a single launch, which is what a small buffer wants: splitting them exists only to
    overlap ``own`` with the fabric, and once the transfer is short that overlap is worth
    less than the launch it costs.
    """
    nodes = world_size // local_world_size
    groups = {"remote": nodes - 1, "own": 1}.get(slots, nodes)
    block_N = 2 * threads
    ctas = (shard_numel // block_N) // tiles_per_cta

    @T.prim_func
    def main(
        inp_mc: T.Tensor((world_size * shard_numel,), dtype),
        partial: T.Tensor((nodes * shard_numel,), dtype),
        rank: T.int32,
    ):
        with T.Kernel(groups * ctas, threads=threads) as bx:
            c = bx % ctas
            k = bx // ctas
            local_rank = rank % local_world_size
            node = rank // local_world_size
            if slots == "remote":
                n = (node + k + 1) % nodes
            elif slots == "own":
                n = node
            else:
                n = k  # "all": every node slot in one launch
            src_base = (n * local_world_size + local_rank) * shard_numel
            dst_base = n * shard_numel
            acc = T.alloc_fragment((block_N,), dtype)
            for j in T.serial(tiles_per_cta):
                off = (c * tiles_per_cta + j) * block_N
                T.multimem_ld_reduce(
                    inp_mc[src_base + off:src_base + off + block_N], acc,
                    reduce_op=T.MultimemReduceOp.ADD,
                )
                T.copy(acc, partial[dst_base + off:dst_base + off + block_N])

    return main


# ------------------------------------------------------ intra-node, portable path


def pull_bcast_kernel(
    shard_numel: int, chunks: int, threads: int, world_size: int, local_world_size: int,
    dtype: str, slots: str,
):
    """Portable broadcast: read each sibling's buffer instead of publishing to it.

    ``slots="own"`` reads siblings' inputs to fill our own node's output slots;
    ``slots="remote"`` reads their outputs for the other nodes' slots. See trap 1 for
    why ``src_pe`` must be a global rank.
    """
    nodes = world_size // local_world_size
    chunk_numel = shard_numel // chunks
    groups = 1 if slots == "own" else nodes - 1
    blocks = (local_world_size - 1) * groups * chunks

    @T.prim_func
    def main(
        shard: T.Tensor((shard_numel,), dtype),
        out: T.Tensor((world_size * shard_numel,), dtype),
        rank: T.int32,
    ):
        with T.Kernel(blocks, threads=threads) as bx:
            c = bx % chunks
            k = (bx // chunks) % groups
            step = (bx // chunks) // groups
            local_rank = rank % local_world_size
            node = rank // local_world_size
            # Rotate so concurrent CTAs do not all read the same sibling first.
            lp = (local_rank + step + 1) % local_world_size
            peer = rank - local_rank + lp
            if slots == "own":
                T.get_block(
                    src=T.address_of(shard[c * chunk_numel]),
                    dst=T.address_of(out[(node * local_world_size + lp) * shard_numel +
                                         c * chunk_numel]),
                    size=chunk_numel, src_pe=peer,
                )
            else:
                n = (node + k + 1) % nodes
                off = (n * local_world_size + lp) * shard_numel + c * chunk_numel
                T.get_block(
                    src=T.address_of(out[off]), dst=T.address_of(out[off]),
                    size=chunk_numel, src_pe=peer,
                )

    return main


def pull_reduce_kernel(
    shard_numel: int, chunks: int, threads: int, world_size: int, local_world_size: int,
    dtype: str, slots: str,
):
    """Portable reduce: stage every sibling's slice, then sum the slots.

    Needs a ``scratch`` of ``world_size * shard_numel``, which is the cost that keeps
    it behind ``mc_reduce_kernel``. Its grid is one CTA per (node slot, chunk), so
    ``chunks`` here should be much larger than the rail kernel's -- 16 CTAs on 148 SMs
    measures 99.9 GB/s against 293.9 at 1024.
    """
    nodes = world_size // local_world_size
    chunk_numel = shard_numel // chunks
    groups = nodes - 1 if slots == "remote" else 1

    @T.prim_func
    def main(
        inp: T.Tensor((world_size * shard_numel,), dtype),
        scratch: T.Tensor((world_size * shard_numel,), dtype),
        partial: T.Tensor((nodes * shard_numel,), dtype),
        rank: T.int32,
    ):
        with T.Kernel(groups * chunks, threads=threads) as bx:
            c = bx % chunks
            k = bx // chunks
            local_rank = rank % local_world_size
            node = rank // local_world_size
            node_base = rank - local_rank
            n = (node + k + 1) % nodes if slots == "remote" else node
            base = c * chunk_numel
            # Every sibling holds the slice for our rail index at the same symmetric
            # offset, so only the peer changes across the loop.
            src_off = (n * local_world_size + local_rank) * shard_numel + base
            for step in range(local_world_size):
                lp = (local_rank + step) % local_world_size
                T.get_block(
                    src=T.address_of(inp[src_off]),
                    dst=T.address_of(scratch[(n * local_world_size + lp) * shard_numel + base]),
                    size=chunk_numel, src_pe=node_base + lp,
                )
            # cp_block spreads the copy over the whole CTA, so the sum must not start
            # before every thread's share has landed.
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


# ------------------------------------------------------------------ host drivers


def workable_chunks(shard_numel, threads, world_size, local_world_size, dtype,
                    chunks, gin_contexts, log=None):
    """Largest chunk count <= ``chunks`` whose put size actually lowers.

    ``put_signal`` fails with "Can't fetch the lanes of a scalable vector at a compile
    time" for a non-contiguous set of transfer sizes, characterised in
    debug_put_size_lowering.py. Which sizes those are depends on ``numel``, so perfectly
    ordinary buffer lengths are otherwise unusable: 240 MB works at ``--chunks 8`` while
    120 MB does not, because its 491520-element put lands in the bad set.

    Halving the count doubles the put size and moves off the bad value -- and larger puts
    are what RDMA wants anyway.

    Probed with a plain ``tilelang.compile`` rather than ``ctx.compile``. Not because a
    collective compile cannot report a failure -- ``_maybe_compile_once`` does ship the
    root's traceback through its ``all_gather_object``, so a root lowering failure fails
    cleanly on every rank. The reason is that each node runs a different interpreter and
    NCCL (see run_internode.sh), so "every rank rejects the same size" is an assumption
    about two toolchains rather than a guarantee; probing locally keeps the retry loop from
    depending on it.
    """
    while True:
        try:
            tilelang.compile(rail_put_kernel(shard_numel, chunks, threads, world_size,
                                             local_world_size, dtype, wait=True))
            return chunks
        except Exception as exc:  # noqa: BLE001 - only this lowering failure is retried
            if "lanes of a scalable" not in str(exc) or chunks <= gin_contexts:
                raise
            chunks //= 2
            if log:
                log(f"  put size did not lower; retrying with --chunks {chunks}")


def nodes_of(ctx) -> int:
    return ctx.world_size // ctx.local_world_size


def pick_intra(mode: str) -> str:
    """Resolve ``auto`` against the hardware. See Context.supports_multicast."""
    from internode_common import Context

    if mode != "auto":
        return mode
    return "multimem" if Context.supports_multicast() else "pull"


class _Base:
    """Shared plumbing: shapes, signal bookkeeping, side stream."""

    def __init__(self, ctx, shard_numel, torch_dtype, tl_dtype, args, intra,
                 signal_id=SIGNAL_DATA):
        self.ctx, self.args, self.intra = ctx, args, intra
        self.signal_id = signal_id
        self.lws = ctx.local_world_size
        self.nodes = ctx.world_size // self.lws
        self.shard_numel = shard_numel
        self.torch_dtype, self.tl_dtype = torch_dtype, tl_dtype
        if self.nodes < 2:
            raise SystemExit("the 2D collectives need >= 2 nodes")
        itemsize = torch.empty((), dtype=torch_dtype).element_size()
        # Resolved onto self, never back onto `args`. Several of these knobs are rewritten
        # from the requested value (pick_mc_tiles here, workable_chunks and the
        # MIN_GROUP_BYTES cap below), and mutating the shared namespace would make a caller
        # that builds more than one collective -- an autotuner sweeping candidates, or
        # allreduce building both halves -- silently inherit the previous one's rewrites and
        # report a config it did not run.
        self.mc_tiles = pick_mc_tiles(shard_numel, itemsize, args.mc_tiles)
        self.chunks = workable_chunks(shard_numel, args.threads, ctx.world_size, self.lws,
                                      tl_dtype, args.chunks, args.gin_contexts, ctx.log)
        if self.chunks % args.gin_contexts:
            raise SystemExit(
                f"--chunks {self.chunks} must be a multiple of --gin-contexts "
                f"{args.gin_contexts}: wait_signal divides the target by the granted "
                f"context count and would round it down")
        if intra == "multimem":
            # self.mc_tiles, not args.mc_tiles: the latter is 0 in auto mode.
            unit = 2 * args.mc_threads * self.mc_tiles
            if shard_numel % unit:
                raise SystemExit(
                    f"shard {shard_numel} must be a multiple of "
                    f"2*--mc-threads*--mc-tiles = {unit}")
        elif shard_numel % args.intra_chunks:
            raise SystemExit(
                f"shard {shard_numel} must be a multiple of --intra-chunks "
                f"{args.intra_chunks}")
        # Cap the pipeline depth by *slice size*, not just by the divisibility rules.
        # More groups hide more of the NVLink phases, but each group costs a put, a wait, a
        # sum and a publish launch -- roughly 6 launches -- and once a slice is small those
        # launches dominate. Measured: allreduce with 8 groups is 0.95 ms at a 15.7 MB
        # shard (1.96 MB per slice) and 0.61 ms at a 3.1 MB shard, where the fabric alone
        # needs only 0.13 -- i.e. 0.5 ms of pure overhead, and 0.50x torch.
        want = args.rail_groups
        shard_bytes = shard_numel * itemsize
        reason = ""
        while want > 1 and shard_bytes // want < MIN_GROUP_BYTES:
            want //= 2
            reason = (f"a {shard_bytes // args.rail_groups / 1e6:.2f} MB slice is too small "
                      f"to pay for its launches")
        # A group's grid must be a whole number of chunks and a multiple of the context
        # count, so a chunk count reduced by workable_chunks() caps the depth too.
        while want > 1 and (self.chunks % want
                            or (self.chunks // want) % args.gin_contexts):
            want //= 2
            reason = reason or (f"--chunks {self.chunks} cannot be split {args.rail_groups} "
                                f"ways at {args.gin_contexts} contexts")
        if want != args.rail_groups:
            ctx.log(f"  rail-groups {args.rail_groups} -> {want}: {reason}")
        self.rail_groups = want
        # Only rail peers signal us.
        self.per_launch = (self.nodes - 1) * self.chunks
        self._target = 0
        self.side = torch.cuda.Stream()

    def phases(self):
        """[(name, callable, nvlink_bytes_per_gpu)] for --phases timing.

        Timing a kernel alone is not the same as its cost inside the loop: the rail
        launch measured on its own exposes the full RDMA round trip every iteration
        (0.62 ms), where in steady state consecutive iterations overlap and its real
        contribution is ~0.39 ms. Subtracting the intra phases from the total is the
        honest way to attribute the fabric time; this is for finding which phase to
        attack, not for a bandwidth claim.
        """
        raise NotImplementedError

    @property
    def signals_used(self) -> int:
        """How many consecutive GIN signals this collective occupies, from signal_id.

        One per pipelined group. A composition must space its halves by this much:
        arrivals are cumulative and a wait does not consume them, so an overlapping range
        lets one half's wait be satisfied by the other half's bytes -- which corrupted
        exactly the second half of the allreduce output before this existed.
        """
        return getattr(self, "groups", 1)

    def _bump(self):
        self._target += self.per_launch
        return self._target

    def _rail_args(self, group=0):  # noqa: D401
        # Each pipelined group needs its own signal: arrivals are cumulative and a wait
        # does not consume them, so a shared signal would let group 0's wait be satisfied
        # by group 1's bytes. 32 signals are provisioned.
        return (self.shard_numel, self.chunks, self.args.threads,
                self.ctx.world_size, self.lws, self.tl_dtype, self.signal_id + group)

    def _mc_args(self):
        return (self.shard_numel, self.args.mc_threads, self.ctx.world_size, self.lws,
                self.tl_dtype)

    def _pull_args(self, chunks):
        return (self.shard_numel, chunks, self.args.threads, self.ctx.world_size,
                self.lws, self.tl_dtype)


class Allgather2D(_Base):
    """``shard`` on every rank -> the concatenation of all shards, on every rank.

    Buffers are owned here: write input into ``.shard`` and read ``.out``.

    What overlaps, and why that is where the speed is
    -------------------------------------------------
    A rank's *own* shard already exists, so publishing it is ordered against nothing
    and runs on a side stream **concurrently with the fabric hop**. Only the other
    nodes' shards depend on the network. Serially instead: 311 GB/s against 404.

    ``pub_remote`` cannot join that overlap -- it publishes bytes that have not arrived
    yet -- and at 0.19 ms it is the whole remaining gap to the ~700 GB/s roofline.
    Pipelining it into the rail hop is the next optimisation, but the rail chunking has
    to stay coarse: chunking it finely to overlap made reduce-scatter 4x *slower*,
    because each launch then moved a few MB from a couple of CTAs.
    """

    def __init__(self, ctx, numel, torch_dtype, tl_dtype, args, intra="auto",
                 signal_id=SIGNAL_DATA, shard=None):
        intra = pick_intra(intra)
        if numel % ctx.world_size:
            raise SystemExit(f"numel {numel} must be divisible by {ctx.world_size}")
        super().__init__(ctx, numel // ctx.world_size, torch_dtype, tl_dtype, args, intra,
                         signal_id)
        self.numel = numel
        use_mc = intra == "multimem"

        # --- fabric hop ---
        # G == 1: put and wait are one kernel; with nothing to overlap, a separate wait
        # launch is pure latency.
        # G > 1: one put kernel and one wait kernel per group, each on its own signal, so
        # group g's arrival can be published over NVLink while group g+1 is still in
        # flight. Both nodes issue groups in order on one stream, and a put from sender
        # CTA b signals through context b % contexts, so per-QP ordering makes the groups
        # arrive in order -- which is what lets an early group be published early.
        self.groups = self.rail_groups if use_mc else 1
        per_group = self.chunks // self.groups
        if self.groups > 1:
            if self.chunks % self.groups or per_group % args.gin_contexts:
                raise SystemExit(
                    f"--chunks {self.chunks} must divide by --rail-groups {self.groups} "
                    f"into a multiple of --gin-contexts {args.gin_contexts}; got "
                    f"{per_group} chunks per group")
            self.rail_puts = [
                ctx.compile(
                    rail_put_kernel(*self._rail_args(g), chunk_lo=g * per_group,
                                    chunk_count=per_group),
                    expect=("tl::gin::put_signal_addr",), gin_contexts=args.gin_contexts,
                )
                for g in range(self.groups)
            ]
            self.rail_waits = [
                ctx.compile(
                    rail_wait_kernel(*self._rail_args(g), chunk_count=per_group),
                    expect=("tl::gin::wait_signal",), gin_contexts=args.gin_contexts,
                )
                for g in range(self.groups)
            ]
        else:
            self.rail = ctx.compile(
                rail_put_kernel(*self._rail_args(), wait=True),
                expect=("tl::gin::put_signal_addr", "tl::gin::wait_signal"),
                gin_contexts=args.gin_contexts,
            )
        self.per_group_signals = (nodes_of(ctx) - 1) * per_group

        if use_mc:
            mc = functools.partial(mc_bcast_kernel, *self._mc_args(),
                                   tiles_per_cta=self.mc_tiles)
            span = self.shard_numel // self.groups
            self.group_numel = span
            # One kernel per group: the slice offset must be compile-time, see
            # mc_bcast_kernel. Our own shard is sliced too, so a fused producer can publish
            # group g while group g's fabric hop is in flight instead of publishing the
            # whole shard between the two hops.
            self.pub_own_k = ctx.compile(mc(slots="own"))
            self.pub_own_ks = [
                ctx.compile(mc(slots="own", span_numel=span, elem_lo=g * span))
                for g in range(self.groups)
            ] if self.groups > 1 else [self.pub_own_k]
            self.pub_remote_ks = [
                ctx.compile(mc(slots="remote", span_numel=span, elem_lo=g * span))
                for g in range(self.groups)
            ]
        else:
            build = functools.partial(pull_bcast_kernel, *self._pull_args(self.chunks))
            self.pub_own_k = ctx.compile(build(slots="own"))
            self.pub_remote_ks = [ctx.compile(build(slots="remote"))]

        # `shard` lets a caller feed an existing arena tensor straight in -- allreduce
        # passes the reduce-scatter's output, which avoids a full-shard copy between
        # the two halves.
        self.shard = ctx.tensor((self.shard_numel,), torch_dtype) if shard is None else shard
        if use_mc:
            # A GIN put must target the registered arena window, and the output has to
            # live in the multicast buffer -- different allocations, so the fabric hop
            # lands in `railbuf` and is published from there.
            self.out_mc, self.out = ctx.mcast_tensor((numel,), torch_dtype)
            self.inbox = ctx.tensor((self.nodes * self.shard_numel,), torch_dtype)
            self.inbox.zero_()
        else:
            self.out = ctx.tensor((numel,), torch_dtype)
            self.out_mc = self.inbox = self.out
        self.out.zero_()
        self._use_mc = use_mc

    def phases(self):
        shard_bytes = self.shard_numel * self.shard.element_size()
        if self._use_mc:
            return [
                ("rail_nic", lambda: self.rail_hop(), 0),
                ("pub_own", self.publish_own, shard_bytes * self.lws),
                ("pub_remote", lambda: [self.publish_remote(g) for g in range(self.groups)],
                 shard_bytes * self.lws * (self.nodes - 1)),
            ]
        return [
            ("rail_nic", lambda: self.rail(self.shard, self.out, self.ctx.rank, self._bump()), 0),
            ("pull_own", lambda: self.pub_own_k(self.shard, self.out, self.ctx.rank),
             shard_bytes * (self.lws - 1)),
            ("pull_remote", lambda: self.pub_remote_ks[0](self.shard, self.out, self.ctx.rank),
             shard_bytes * (self.lws - 1) * (self.nodes - 1)),
        ]

    # --- steps, exposed so a fused kernel can interleave compute between them ---
    #
    # launch() below is the plain path. A consumer that wants to compute on the rows it
    # already has, while the fabric hop is still running, needs the steps separately:
    # see example_internode_ag_gemm_2d.py --mode pipeline. Multimem only, because on the
    # pull path publish_own reads siblings' *shards* and publish_remote reads their
    # *out*, so neither is a pure local write and the staging is different.

    def rail_hop(self, stream=None):
        """Start the fabric hop.

        With one group this also waits, since there is nothing to overlap. With several,
        it only *issues* every group's puts -- in group order on one stream, so per-QP
        ordering makes them arrive in order -- and the waits are left to
        ``consume_groups``, which interleaves them with the NVLink publishes.
        """
        self._targets = [self._bump_group(g) for g in range(self.groups)]
        run = self._issue_all if self.groups > 1 else self._rail_one
        if stream is None:
            run()
            return
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()

    def _rail_one(self):
        self.rail(self.shard, self.inbox, self.ctx.rank, self._targets[0])

    def _issue_all(self):
        for g in range(self.groups):
            self.issue_group(g, self._targets[g])

    def bump_groups(self):
        """Reserve this launch's signal targets without issuing anything yet."""
        self._targets = [self._bump_group(g) for g in range(self.groups)]
        return self._targets

    def issue_group(self, group, target):
        """Start group `group`'s fabric put. Its input slice must already be final."""
        self.rail_puts[group](self.shard, self.inbox, self.ctx.rank, target)

    def wait_group(self, group, target):
        self.rail_waits[group](self.inbox, target)

    def consume_groups(self):
        """Publish each group over NVLink as soon as that group has landed.

        This is where the pipelining pays: group g's ~0.1 ms of multicast broadcast runs
        while group g+1 is still crossing the fabric, instead of the whole broadcast
        sitting after the whole hop.
        """
        if self.groups == 1:
            self.publish_remote()
            return
        for g in range(self.groups):
            self.rail_waits[g](self.inbox, self._targets[g])
            self.publish_remote(g)

    def _bump_group(self, group):
        # Each group has its own signal, so each needs its own running target.
        if not hasattr(self, "_gtargets"):
            self._gtargets = [0] * self.groups
        self._gtargets[group] += self.per_group_signals
        return self._gtargets[group]

    def publish_own(self, group=None):
        """Broadcast our own shard to every local rank. Ordered against nothing.

        ``group=None`` publishes the whole shard in one launch, which is what the plain
        path wants. A group index publishes just that slice, for the fused allreduce.
        """
        if group is None:
            self.pub_own_k(self.shard, self.out_mc, self.ctx.rank)
        else:
            self.pub_own_ks[group](self.shard, self.out_mc, self.ctx.rank)

    def publish_remote(self, group=0):
        """Broadcast what arrived over the fabric. Needs that group's wait to have run."""
        self.pub_remote_ks[group](self.inbox, self.out_mc, self.ctx.rank)

    def rows_of_node(self, node, rows_per_rank):
        """First output row belonging to `node`.

        Global rank is ``node * lws + local``, and row block index is global rank, so a
        node's row blocks are *contiguous* -- which is why the pipelined GEMM needs one
        launch per node rather than one per rank.
        """
        return node * self.lws * rows_per_rank

    def launch(self):
        ctx, args = self.ctx, self.args
        main_stream = torch.cuda.current_stream()
        if self._use_mc:
            if args.no_overlap:
                self.rail_hop()
                self.publish_own()
            else:
                # Our own shard exists already, so publishing it races the fabric rather
                # than waiting behind it.
                self.side.wait_stream(main_stream)
                with torch.cuda.stream(self.side):
                    self.publish_own()
                self.rail_hop()
                main_stream.wait_stream(self.side)
            self.consume_groups()
            # We published into every sibling and they into us, so the output is
            # complete only once they have all finished.
            dist.barrier(ctx.group)
        else:
            # The pull path needs its own slot present in `out`, and the rail kernel
            # writes the sender's global-rank slot, so route it straight there.
            target = self._bump()
            if args.no_overlap:
                self.rail(self.shard, self.out, ctx.rank, target)
                dist.barrier(ctx.group)
                self.pub_own_k(self.shard, self.out, ctx.rank)
            else:
                self.side.wait_stream(main_stream)
                with torch.cuda.stream(self.side):
                    self.pub_own_k(self.shard, self.out, ctx.rank)
                self.rail(self.shard, self.out, ctx.rank, target)
                main_stream.wait_stream(self.side)
                dist.barrier(ctx.group)
            self.pub_remote_ks[0](self.shard, self.out, ctx.rank)


class ReduceScatter2D(_Base):
    """Sum of every rank's ``.inp``, restricted to this rank's shard, into ``.out``.

    Needs **no barrier** *provided the input is not written in the same iteration*: the
    intra phase reads siblings' input buffers, which the collective itself never writes,
    and the rail phase sends only bytes this rank produced, with the GIN signal proving
    arrival. Stream order is then sufficient.

    A caller that **produces** ``inp`` each iteration -- a fused GEMM, say -- breaks that
    precondition and must fence between producing and launching, because the reduce reads
    every local rank's copy and stream order says nothing about a sibling's producer. See
    example_internode_gemm_rs_2d.py, which mismatched 11 of 16 ranks without it.

    Overlap runs along the node axis -- reduce the other nodes' slots, start their
    transfer, then reduce our own slot, which nothing on the network waits for. That is
    worth only ~4%, because the put kernel returns once the RDMA is *issued* and the
    flight time was already absorbed by the wait in the tail kernel.
    """

    def __init__(self, ctx, numel, torch_dtype, tl_dtype, args, intra="auto",
                 signal_id=SIGNAL_DATA):
        intra = pick_intra(intra)
        if numel % ctx.world_size:
            raise SystemExit(f"numel {numel} must be divisible by {ctx.world_size}")
        super().__init__(ctx, numel // ctx.world_size, torch_dtype, tl_dtype, args, intra,
                         signal_id)
        self.numel = numel
        use_mc = intra == "multimem"

        if use_mc:
            build = functools.partial(mc_reduce_kernel, *self._mc_args(),
                                      tiles_per_cta=self.mc_tiles)
        else:
            build = functools.partial(pull_reduce_kernel, *self._pull_args(args.intra_chunks))
        # With one group the fabric hop is a single transfer, so there is nothing for the
        # own-slot reduce to hide behind and splitting it just costs a launch.
        # self.rail_groups is the depth after the size cap, not what was asked for.
        self.fused_reduce = self.rail_groups == 1 and use_mc
        if self.fused_reduce:
            self.red_all = ctx.compile(build(slots="all"))
        else:
            self.red_remote = ctx.compile(build(slots="remote"))
            self.red_own = ctx.compile(build(slots="own"))
        # Grouped fabric hop, same idea as Allgather2D: group g's arithmetic runs while
        # group g+1 is still crossing the fabric. Per-put size is unchanged.
        self.groups = self.rail_groups
        per_group = self.chunks // self.groups
        if self.chunks % self.groups or per_group % args.gin_contexts:
            raise SystemExit(
                f"--chunks {self.chunks} must divide by --rail-groups {self.groups} into a "
                f"multiple of --gin-contexts {args.gin_contexts}; got {per_group}")
        span = self.shard_numel // self.groups
        self.put_k, self.wait_k, self.sum_k = [], [], []
        for g in range(self.groups):
            self.put_k.append(ctx.compile(
                rail_put_kernel(*self._rail_args(g), src_per_node=True,
                                chunk_lo=g * per_group, chunk_count=per_group),
                expect=("tl::gin::put_signal_addr",), gin_contexts=args.gin_contexts))
            self.wait_k.append(ctx.compile(
                rail_wait_kernel(*self._rail_args(g), chunk_count=per_group),
                expect=("tl::gin::wait_signal",), gin_contexts=args.gin_contexts))
            self.sum_k.append(ctx.compile(
                rs_sum_kernel(self.shard_numel, args.intra_chunks // self.groups,
                              args.threads, ctx.world_size, self.lws, tl_dtype,
                              elem_lo=g * span, span_numel=span)))
        self.per_group_signals = (self.nodes - 1) * per_group
        self._gtargets = [0] * self.groups

        if use_mc:
            self.inp_mc, self.inp = ctx.mcast_tensor((numel,), torch_dtype)
            self.scratch = None
        else:
            self.inp = ctx.tensor((numel,), torch_dtype)
            self.inp_mc = self.inp
            self.scratch = ctx.tensor((numel,), torch_dtype)
        self.partial = ctx.tensor((self.nodes * self.shard_numel,), torch_dtype)
        self.inbox = ctx.tensor((self.nodes * self.shard_numel,), torch_dtype)
        self.out = ctx.tensor((self.shard_numel,), torch_dtype)
        for t in (self.scratch, self.partial, self.inbox, self.out):
            if t is not None:
                t.zero_()
        self._use_mc = use_mc

    def _reduce(self, kernel):
        if self._use_mc:
            kernel(self.inp_mc, self.partial, self.ctx.rank)
        else:
            kernel(self.inp, self.scratch, self.partial, self.ctx.rank)

    def phases(self):
        shard_bytes = self.shard_numel * self.out.element_size()
        return [
            ("rail_put", lambda: self.put_k[0](self.partial, self.inbox, self.ctx.rank,
                                               self._bump()), 0),
            ("red_remote", lambda: self._reduce(self.red_remote),
             shard_bytes * self.lws * (self.nodes - 1)),
            ("red_own", lambda: self._reduce(self.red_own), shard_bytes * self.lws),
        ]

    def start(self):
        """Reduce over NVLink and get the fabric moving; return the per-group targets.

        Split out of ``launch`` so a consumer can interleave work with the per-group
        finishes -- the fused allreduce starts the allgather's hop for group g the moment
        this rank's group g is summed, instead of after every group.
        """
        ctx, args = self.ctx, self.args
        main_stream = torch.cuda.current_stream()
        targets = []
        for g in range(self.groups):
            self._gtargets[g] += self.per_group_signals
            targets.append(self._gtargets[g])
        issue = lambda: [self.put_k[g](self.partial, self.inbox, ctx.rank, targets[g])
                         for g in range(self.groups)]
        if self.fused_reduce:
            self._reduce(self.red_all)
            issue()
            return targets
        self._reduce(self.red_remote)
        if args.no_overlap:
            self._reduce(self.red_own)
            issue()
        else:
            ready = torch.cuda.Event()
            ready.record(main_stream)
            self.side.wait_event(ready)
            with torch.cuda.stream(self.side):
                issue()
            # Nothing on the network waits for our own slot, so it reduces in flight.
            self._reduce(self.red_own)
            main_stream.wait_stream(self.side)
        return targets

    def finish_group(self, group, target):
        """Wait for group `group`'s arrivals and sum it into ``out``."""
        self.wait_k[group](self.inbox, target)
        self.sum_k[group](self.partial, self.inbox, self.out, self.ctx.rank)

    def launch(self):
        for g, target in enumerate(self.start()):
            self.finish_group(g, target)


class Allreduce2D(_Base):
    """Sum of every rank's ``.inp``, on every rank, into ``.out`` -- in one fabric hop.

    Composing ReduceScatter2D with Allgather2D is correct and simple, and it measured
    1.088 ms against torch's 0.949 (0.87x). The two halves do not overlap at all -- the
    total is exactly their sum -- because the allgather cannot start until the
    reduce-scatter has produced the shard it broadcasts.

    So merge them. The insight is that the rail peer can send its partials for **every**
    node slot rather than only the one this rank owns, which lets this rank finish all
    slots locally and removes the second hop entirely:

    1. ``mc_reduce`` -- NVSwitch-reduce over local ranks, giving this rank a partial for
       each of the ``nodes`` slots at its own rail index.
    2. ``rail`` -- one hop, carrying that whole ``nodes * shard`` vector.
    3. ``tail`` -- add our own vector, wait, sum the ``nodes`` arrivals: now every slot at
       our rail index holds the global sum.
    4. ``mc_bcast(slots="all")`` -- publish all of them to every local rank.

    Fabric bytes are identical to the two-hop version, and the floor is the same: a 2-node
    allreduce must move the node-sum each way, ``N/lws`` per rank, so 31.4 MB at 47.6 GB/s
    = 0.66 ms at this size. What the merge buys is one serialisation instead of two, one
    barrier instead of three, and far fewer launches.
    """

    def __init__(self, ctx, numel, torch_dtype, tl_dtype, args, intra="auto",
                 signal_id=SIGNAL_DATA):
        intra = pick_intra(intra)
        if intra != "multimem":
            raise SystemExit("the merged allreduce needs multimem; use --algo composed")
        if numel % ctx.world_size:
            raise SystemExit(f"numel {numel} must be divisible by {ctx.world_size}")
        super().__init__(ctx, numel // ctx.world_size, torch_dtype, tl_dtype, args, intra,
                         signal_id)
        self.numel = numel
        vec = self.nodes * self.shard_numel
        shard = self.shard_numel
        cps = self.chunks // self.nodes  # chunks per node slot
        if self.chunks % self.nodes or cps % args.gin_contexts:
            raise SystemExit(
                f"--chunks {self.chunks} must split into {self.nodes} slots of a multiple "
                f"of --gin-contexts {args.gin_contexts}; got {cps} per slot")

        mc = functools.partial(mc_reduce_kernel, *self._mc_args(),
                               tiles_per_cta=self.mc_tiles)
        self.red_remote = ctx.compile(mc(slots="remote"))
        self.red_own = ctx.compile(mc(slots="own"))

        # One put / tail / publish per node slot, each on its own signal. Slot m occupies
        # a contiguous chunk range of the vector, so the existing chunk_lo/chunk_count
        # slicing covers it. The offsets have to be compile-time (multimem needs a
        # provably in-bounds region), so this is one kernel per absolute slot and the host
        # picks which to call -- our own node index is known there.
        rail = (vec, self.chunks, args.threads, ctx.world_size, self.lws, tl_dtype)
        self.put_k, self.wait_k, self.sum_k, self.pub_k = [], [], [], []
        for m in range(self.nodes):
            self.put_k.append(ctx.compile(
                rail_put_kernel(*rail, signal_id + m, chunk_lo=m * cps, chunk_count=cps),
                expect=("tl::gin::put_signal_addr",), gin_contexts=args.gin_contexts))
            self.wait_k.append(ctx.compile(
                rail_wait_kernel(*rail, signal_id + m, chunk_count=cps),
                expect=("tl::gin::wait_signal",), gin_contexts=args.gin_contexts))
            self.sum_k.append(ctx.compile(
                allreduce_sum_kernel(vec, shard, args.intra_chunks, args.threads,
                                     ctx.world_size, self.lws, tl_dtype, m)))
            self.pub_k.append(ctx.compile(
                mc_bcast_kernel(*self._mc_args(), slots="all",
                                tiles_per_cta=self.mc_tiles, node_slot=m)))

        self.inp_mc, self.inp = ctx.mcast_tensor((numel,), torch_dtype)
        self.out_mc, self.out = ctx.mcast_tensor((numel,), torch_dtype)
        self.partial = ctx.tensor((vec,), torch_dtype)
        self.inbox = ctx.tensor((self.nodes * vec,), torch_dtype)
        self.reduced = ctx.tensor((vec,), torch_dtype)
        for t in (self.partial, self.inbox, self.reduced, self.out):
            t.zero_()
        self.per_slot_signals = (self.nodes - 1) * cps
        self._stargets = [0] * self.nodes
        self.my_node = ctx.rank // self.lws
        # Send the slots we can send first, then ours: red_own is still writing our own
        # slot while the first put is in flight, and sending it early was a real race --
        # it corrupted exactly the second half of the output.
        self.slot_order = [m for m in range(self.nodes) if m != self.my_node] + [self.my_node]

    def launch(self):
        ctx = self.ctx
        main_stream = torch.cuda.current_stream()
        targets = []
        for m in range(self.nodes):
            self._stargets[m] += self.per_slot_signals
            targets.append(self._stargets[m])

        # Reduce the slots the fabric needs first, so the hop starts as early as possible.
        self.red_remote(self.inp_mc, self.partial, ctx.rank)
        for m in self.slot_order[:-1]:
            self.put_k[m](self.partial, self.inbox, ctx.rank, targets[m])
        # Our own slot: only now may it be sent.
        self.red_own(self.inp_mc, self.partial, ctx.rank)
        self.put_k[self.my_node](self.partial, self.inbox, ctx.rank, targets[self.my_node])

        # Per-slot pipeline: sum and publish each slot as its arrival lands, in the order
        # they were sent, so slot A's NVLink publish overlaps slot B's transfer.
        for m in self.slot_order:
            self.wait_k[m](self.inbox, targets[m])
            self.sum_k[m](self.partial, self.inbox, self.reduced, ctx.rank)
            self.pub_k[m](self.reduced, self.out_mc, ctx.rank)
        # We published into every sibling and they into us.
        dist.barrier(ctx.group)


def report_phases(ctx, coll, args):
    """Time each kernel of a 2D collective on its own. See ``_Base.phases``."""
    from tilelang.distributed.bench import do_bench

    for name, fn, nvlink_bytes in coll.phases():
        dist.barrier(ctx.group)
        ms = do_bench(fn, warmup=args.warmup, rep=args.rep, group=ctx.group)
        if nvlink_bytes:
            ctx.log(f"  {name:12s} {ms:7.3f} ms   {nvlink_bytes / 1e6 / ms:6.1f} GB/s "
                    f"NVLink/GPU")
        else:
            # One shard per rank crosses the fabric, on that rank's own NIC.
            mb = coll.shard_numel * coll.out.element_size() / 1e6
            ctx.log(f"  {name:12s} {ms:7.3f} ms   {mb / ms:6.1f} GB/s per NIC "
                    f"(isolated; overstates -- see phases())")


def fused_allreduce_launch(rs, ag, ctx):
    """Reduce-scatter and allgather with their fabric hops overlapped.

    Composed serially the two halves add up exactly -- 0.550 + 0.500 ms -- because the
    allgather cannot start until the reduce-scatter has produced the shard it broadcasts.
    But that is only true *per group*: the allgather's hop for group g needs nothing but
    the reduce-scatter's sum for group g. So push each group across the fabric as soon as
    it is summed, and the second hop overlaps the first's remaining groups instead of
    following all of them.

    The two halves must already hold disjoint signal ranges; see the example.
    """
    if ag.groups == 1:
        # Nothing to interleave, so take the fewest launches: the allgather's put and wait
        # are one kernel here, and the publishes are whole-shard. At small sizes this path
        # is what wins -- the pipeline's extra launches cost more than its overlap saves.
        for g, target in enumerate(rs.start()):
            rs.finish_group(g, target)
        ag.rail_hop()
        ag.publish_own()
        ag.publish_remote()
        dist.barrier(ctx.group)
        return
    rs_targets = rs.start()
    ag_targets = ag.bump_groups()
    for g, target in enumerate(rs_targets):
        rs.finish_group(g, target)
        # out slice g is final, and ag.shard *is* rs.out, so this group can fly now --
        # and publishing our own copy of that slice needs no network at all, so it runs
        # while the slice is crossing the fabric rather than between the two hops.
        ag.issue_group(g, ag_targets[g])
        ag.publish_own(g)
    for g, target in enumerate(ag_targets):
        ag.wait_group(g, target)
        ag.publish_remote(g)
    dist.barrier(ctx.group)


def add_2d_args(parser):
    """Knobs shared by every 2D example. Defaults are the measured optima at 16 GPUs."""
    parser.add_argument("--intra", choices=("multimem", "pull", "auto"), default="auto",
                        help="intra-node half: NVSwitch multimem, the portable "
                             "get_block pull, or multimem when the hardware allows it")
    parser.add_argument("--mc-threads", type=int, default=512,
                        help="threads per CTA on the multimem path; the tile is "
                             "2*threads, fixed by the packed-x2 fragment layout")
    parser.add_argument("--mc-tiles", type=int, default=0,
                        help="contiguous tiles each multimem CTA loops over; the tile width "
                             "is pinned, so this is the only work-per-thread knob. 0 scales "
                             "it to the shard, which matters: the best value is 32 at "
                             "240 MiB and 4 at 48 MiB")
    parser.add_argument("--intra-chunks", type=int, default=1024,
                        help="chunking of the pull path's intra phase; sets its grid, "
                             "and is independent of --chunks because it carries no signal")
    # 2 measured best on allgather: 472.8 GB/s against 395.2 unsplit (1.53x torch vs
    # 1.28x). 4 needs --gin-contexts 2 to keep a group's grid a multiple of the context
    # count, and losing contexts costs more than the extra group gains (434.0). Raising
    # --chunks to 16 to keep 4 groups at 4 contexts hits the put-size lowering bug.
    parser.add_argument("--rail-groups", type=int, default=2,
                        help="split the fabric hop into this many groups so each group's "
                             "NVLink publish overlaps the next group's transfer; the "
                             "per-put size is unchanged, only per-group parallelism drops")
    parser.add_argument("--no-overlap", action="store_true",
                        help="run the phases serially, to show what the overlap buys")
    # add_common_args defaults --chunks to 64, which suits the flat collectives. The
    # rail kernel here wants far fewer, larger messages: at 64 it fails to lower with
    # "Can't fetch the lanes of a scalable vector", and in the single-node proxy
    # gemm_rs_2d returned a *wrong* answer on one rank rather than erroring. Root cause
    # not yet found, so default to the tuned value and treat large chunk counts as
    # unsupported here.
    parser.set_defaults(chunks=8)
    return parser
