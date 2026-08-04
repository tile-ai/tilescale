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

import tilelang.language as T

from internode_common import SIGNAL_DATA, fp32_sum


# --------------------------------------------------------------------- fabric hop


def rail_put_kernel(
    shard_numel: int, chunks: int, threads: int, world_size: int, local_world_size: int,
    dtype: str, signal_id: int = SIGNAL_DATA, src_per_node: bool = False,
    wait: bool = False,
):
    """Rail-aligned put of one shard to the same local index on every other node.

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

    @T.prim_func
    def main(
        src: T.Tensor((src_slots * shard_numel,), dtype),
        inbox: T.Tensor((nodes * shard_numel,), dtype),
        rank: T.int32,
        signal_target: T.int32,
    ):
        with T.Kernel(chunks, threads=threads) as bx:
            base = bx * chunk_numel
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


# ------------------------------------------------------- intra-node via NVSwitch


def mc_bcast_kernel(
    shard_numel: int, threads: int, world_size: int, local_world_size: int, dtype: str,
    slots: str, tiles_per_cta: int = 32,
):
    """Publish the shards we own into every local rank's slot with one ``multimem.st``.

    ``slots="own"`` publishes our own shard, which exists before the collective starts
    and so is ordered against nothing. ``slots="remote"`` publishes what arrived over
    the fabric. See trap 2 in the module docstring for why the tile width is fixed.
    """
    nodes = world_size // local_world_size
    groups = 1 if slots == "own" else nodes - 1
    block_N = 2 * threads
    ctas = (shard_numel // block_N) // tiles_per_cta

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
            n = node if slots == "own" else (node + k + 1) % nodes
            src_base = 0 if slots == "own" else n * shard_numel
            dst_base = (n * local_world_size + local_rank) * shard_numel
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
    ``slots="own"`` handles the one we keep, which no peer waits for.
    """
    nodes = world_size // local_world_size
    groups = nodes - 1 if slots == "remote" else 1
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
            n = (node + k + 1) % nodes if slots == "remote" else node
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
        if args.chunks % args.gin_contexts:
            raise SystemExit(
                f"--chunks {args.chunks} must be a multiple of --gin-contexts "
                f"{args.gin_contexts}: wait_signal divides the target by the granted "
                f"context count and would round it down")
        if intra == "multimem":
            unit = 2 * args.mc_threads * args.mc_tiles
            if shard_numel % unit:
                raise SystemExit(
                    f"shard {shard_numel} must be a multiple of "
                    f"2*--mc-threads*--mc-tiles = {unit}")
        elif shard_numel % args.intra_chunks:
            raise SystemExit(
                f"shard {shard_numel} must be a multiple of --intra-chunks "
                f"{args.intra_chunks}")
        # Only rail peers signal us.
        self.per_launch = (self.nodes - 1) * args.chunks
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

    def _bump(self):
        self._target += self.per_launch
        return self._target

    def _rail_args(self):
        return (self.shard_numel, self.args.chunks, self.args.threads,
                self.ctx.world_size, self.lws, self.tl_dtype, self.signal_id)

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

        # The put and the wait are one kernel here: with nothing else to overlap, a
        # separate wait launch is pure latency.
        self.rail = ctx.compile(
            rail_put_kernel(*self._rail_args(), wait=True),
            expect=("tl::gin::put_signal_addr", "tl::gin::wait_signal"),
            gin_contexts=args.gin_contexts,
        )
        if use_mc:
            build = functools.partial(mc_bcast_kernel, *self._mc_args(),
                                      tiles_per_cta=args.mc_tiles)
        else:
            build = functools.partial(pull_bcast_kernel, *self._pull_args(args.chunks))
        self.pub_own = ctx.compile(build(slots="own"))
        self.pub_remote = ctx.compile(build(slots="remote"))

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
        rail = lambda: self.rail(self.shard, self.inbox, self.ctx.rank, self._bump())
        if self._use_mc:
            return [
                ("rail_nic", rail, 0),
                ("pub_own", lambda: self.pub_own(self.shard, self.out_mc, self.ctx.rank),
                 shard_bytes * self.lws),
                ("pub_remote",
                 lambda: self.pub_remote(self.inbox, self.out_mc, self.ctx.rank),
                 shard_bytes * self.lws * (self.nodes - 1)),
            ]
        return [
            ("rail_nic", lambda: self.rail(self.shard, self.out, self.ctx.rank, self._bump()), 0),
            ("pull_own", lambda: self.pub_own(self.shard, self.out, self.ctx.rank),
             shard_bytes * (self.lws - 1)),
            ("pull_remote", lambda: self.pub_remote(self.shard, self.out, self.ctx.rank),
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
        """Issue the fabric put and wait for the arrivals. Blocks `stream`, not the CPU."""
        target = self._bump()
        if stream is None:
            self.rail(self.shard, self.inbox, self.ctx.rank, target)
            return
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.rail(self.shard, self.inbox, self.ctx.rank, target)

    def publish_own(self):
        """Broadcast our own shard to every local rank. Ordered against nothing."""
        self.pub_own(self.shard, self.out_mc, self.ctx.rank)

    def publish_remote(self):
        """Broadcast what arrived over the fabric. Needs rail_hop to have completed."""
        self.pub_remote(self.inbox, self.out_mc, self.ctx.rank)

    def rows_of_node(self, node, rows_per_rank):
        """First output row belonging to `node`.

        Global rank is ``node * lws + local``, and row block index is global rank, so a
        node's row blocks are *contiguous* -- which is why the pipelined GEMM needs one
        launch per node rather than one per rank.
        """
        return node * self.lws * rows_per_rank

    def launch(self):
        ctx, args = self.ctx, self.args
        target = self._bump()
        main_stream = torch.cuda.current_stream()
        if self._use_mc:
            if args.no_overlap:
                self.rail(self.shard, self.inbox, ctx.rank, target)
                self.pub_own(self.shard, self.out_mc, ctx.rank)
            else:
                self.side.wait_stream(main_stream)
                with torch.cuda.stream(self.side):
                    self.pub_own(self.shard, self.out_mc, ctx.rank)
                self.rail(self.shard, self.inbox, ctx.rank, target)
                main_stream.wait_stream(self.side)
            self.pub_remote(self.inbox, self.out_mc, ctx.rank)
            # We published into every sibling and they into us, so the output is
            # complete only once they have all finished.
            dist.barrier(ctx.group)
        else:
            # The pull path needs its own slot present in `out`, and the rail kernel
            # writes the sender's global-rank slot, so route it straight there.
            if args.no_overlap:
                self.rail(self.shard, self.out, ctx.rank, target)
                dist.barrier(ctx.group)
                self.pub_own(self.shard, self.out, ctx.rank)
            else:
                self.side.wait_stream(main_stream)
                with torch.cuda.stream(self.side):
                    self.pub_own(self.shard, self.out, ctx.rank)
                self.rail(self.shard, self.out, ctx.rank, target)
                main_stream.wait_stream(self.side)
                dist.barrier(ctx.group)
            self.pub_remote(self.shard, self.out, ctx.rank)


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
                                      tiles_per_cta=args.mc_tiles)
        else:
            build = functools.partial(pull_reduce_kernel, *self._pull_args(args.intra_chunks))
        self.red_remote = ctx.compile(build(slots="remote"))
        self.red_own = ctx.compile(build(slots="own"))
        self.put = ctx.compile(
            rail_put_kernel(*self._rail_args(), src_per_node=True),
            expect=("tl::gin::put_signal_addr",),
            gin_contexts=args.gin_contexts,
        )
        self.tail = ctx.compile(
            rail_sum_kernel(*self._rail_args()),
            expect=("tl::gin::wait_signal",),
            gin_contexts=args.gin_contexts,
        )

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
            ("rail_put", lambda: self.put(self.partial, self.inbox, self.ctx.rank,
                                          self._bump()), 0),
            ("rail_sum", lambda: self.tail(self.partial, self.inbox, self.out,
                                           self.ctx.rank, self._target), 0),
            ("red_remote", lambda: self._reduce(self.red_remote),
             shard_bytes * self.lws * (self.nodes - 1)),
            ("red_own", lambda: self._reduce(self.red_own), shard_bytes * self.lws),
        ]

    def launch(self):
        ctx, args = self.ctx, self.args
        target = self._bump()
        main_stream = torch.cuda.current_stream()
        # The other nodes' slots are what the fabric hop needs, so they go first.
        self._reduce(self.red_remote)
        if args.no_overlap:
            self._reduce(self.red_own)
            self.put(self.partial, self.inbox, ctx.rank, target)
        else:
            ready = torch.cuda.Event()
            ready.record(main_stream)
            self.side.wait_event(ready)
            with torch.cuda.stream(self.side):
                self.put(self.partial, self.inbox, ctx.rank, target)
            self._reduce(self.red_own)
            main_stream.wait_stream(self.side)
        self.tail(self.partial, self.inbox, self.out, ctx.rank, target)


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


def add_2d_args(parser):
    """Knobs shared by every 2D example. Defaults are the measured optima at 16 GPUs."""
    parser.add_argument("--intra", choices=("multimem", "pull", "auto"), default="auto",
                        help="intra-node half: NVSwitch multimem, the portable "
                             "get_block pull, or multimem when the hardware allows it")
    parser.add_argument("--mc-threads", type=int, default=512,
                        help="threads per CTA on the multimem path; the tile is "
                             "2*threads, fixed by the packed-x2 fragment layout")
    parser.add_argument("--mc-tiles", type=int, default=32,
                        help="contiguous tiles each multimem CTA loops over; the tile "
                             "width is pinned, so this is the only work-per-thread knob")
    parser.add_argument("--intra-chunks", type=int, default=1024,
                        help="chunking of the pull path's intra phase; sets its grid, "
                             "and is independent of --chunks because it carries no signal")
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
