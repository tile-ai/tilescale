"""Hierarchical (2D) inter-node allreduce: 2D reduce-scatter then 2D allgather.

The flat allreduce in ``example_internode_allreduce.py`` has every rank push to all
``world_size - 1`` peers over GIN, which is fine with one GPU per node and collapses
with eight -- see ``internode_2d`` for the measurement and the fix.

There is nothing new here: allreduce *is* reduce-scatter followed by allgather, and
both halves are already 2D and already beat torch on their own. So this example is
composition, and the interesting parts are the two seams:

* **The shard buffer is shared, not copied.** ``Allgather2D`` takes the reduce-scatter's
  output as its input, so the two halves hand over in place. Both live in the arena, so
  this is just a pointer.
* **The halves need disjoint GIN signal *ranges*.** Signal state is cumulative and a wait
  does not consume it, so an overlap lets one half's wait be satisfied by the other's
  bytes -- silently, and only under repetition. Each half occupies ``signals_used`` of
  them (one per pipelined group), so the allgather starts at
  ``SIGNAL_DATA + rs.signals_used`` rather than at a hardcoded second signal. Getting this
  wrong by one corrupted exactly the second half of the output. 32 signals are
  provisioned.

Volume-wise this is the right decomposition for large buffers: two-shot moves
``2(W-1)N/W`` bytes against one-shot's ``(W-1)N``. One-shot only wins at small ``W``
and even then not here, because it reduces over the full buffer rather than a shard --
see the flat example's ``--algo oneshot`` for that measurement.
"""

# NOTE: no `from __future__ import annotations` here -- see internode_2d.
import argparse
import os

import torch
import torch.distributed as dist

from internode_2d import (
    Allgather2D,
    Allreduce2D,
    ReduceScatter2D,
    add_2d_args,
    fused_allreduce_launch,
    pick_intra,
)
from internode_common import (
    SIGNAL_DATA,
    Context,
    TL_DTYPES,
    TORCH_DTYPES,
    add_common_args,
    bench_vs_torch,
    check,
    prepare_env,
)


def main() -> int:
    parser = add_2d_args(add_common_args(argparse.ArgumentParser(description=__doc__)))
    # composed wins on measurement: 1.086 ms against merged's 1.296. The single hop saves
    # a serialisation but the NVLink publish cost is identical, and with only `nodes` slots
    # the merged pipeline is too coarse-grained to make up the difference.
    parser.add_argument("--algo", choices=("fused", "merged", "composed"), default="fused",
                        help="merged: one fabric hop carrying every node partial. "
                             "composed: reduce-scatter then allgather, which is simpler "
                             "and works without multicast, but the halves cannot overlap")
    # Allreduce wants a finer pipeline than the one-hop collectives: it has two fabric
    # hops to overlap, so 8 groups beats 2 (0.95 ms vs 1.02). Each group's grid must be a
    # multiple of the GIN context count, and 8 groups of chunks=8 leaves one chunk per
    # group, hence one context. Allgather goes the other way -- it prefers 2 groups on 4
    # contexts (0.500 ms vs 0.561 at 8/1) -- so these defaults are per example, not global.
    parser.set_defaults(rail_groups=8, gin_contexts=1)
    args = parser.parse_args()

    prepare_env()
    # The multicast buffer is sized before the allocator exists. Both halves want one
    # -- the reduce-scatter reduces its input through the switch, the allgather
    # broadcasts into its output -- hence two buffers of `numel`.
    world = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    itemsize = torch.empty((), dtype=TORCH_DTYPES[args.dtype]).element_size()
    intra = pick_intra(args.intra)
    if args.algo == "merged" and intra != "multimem":
        args.algo = "composed"
    if args.algo == "fused" and intra != "multimem":
        args.algo = "composed"
    # Both algorithms want two multicast buffers of `numel`: one reduced through the
    # switch, one broadcast out of it.
    ctx = Context(mcast_bytes=2 * args.numel * itemsize if intra == "multimem" else 0)

    torch_dtype, tl_dtype = TORCH_DTYPES[args.dtype], TL_DTYPES[args.dtype]
    nodes = ctx.world_size // ctx.local_world_size
    ctx.log(
        f"allreduce_2d: world={ctx.world_size} nodes={nodes} local={ctx.local_world_size} "
        f"numel={args.numel} shard={args.numel // ctx.world_size} chunks={args.chunks} "
        f"algo={args.algo} intra={intra} mc_threads={args.mc_threads} mc_tiles={args.mc_tiles} "
        f"overlap={not args.no_overlap} contexts={args.gin_contexts} dtype={args.dtype}"
    )

    if args.algo == "merged":
        ar = Allreduce2D(ctx, args.numel, torch_dtype, tl_dtype, args, intra=intra)
        inp, out, launch = ar.inp, ar.out, ar.launch
    else:
        rs = ReduceScatter2D(ctx, args.numel, torch_dtype, tl_dtype, args, intra=intra,
                             signal_id=SIGNAL_DATA)
        # Hand over in place, and on a *disjoint signal range* -- not merely a different
        # signal. Each half occupies one signal per pipelined group, so with
        # --rail-groups 2 the reduce-scatter holds {0,1}; starting the allgather at
        # SIGNAL_PHASE2 == 1 overlapped it and its group-0 wait was satisfied by the
        # reduce-scatter's group-1 arrivals, corrupting the second half of the output.
        ag = Allgather2D(ctx, args.numel, torch_dtype, tl_dtype, args, intra=intra,
                         signal_id=SIGNAL_DATA + rs.signals_used, shard=rs.out)
        inp, out = rs.inp, ag.out

        if args.algo == "fused":
            def launch():
                fused_allreduce_launch(rs, ag, ctx)
        else:
            def launch():
                rs.launch()
                ag.launch()
    # Small magnitudes: a bf16 sum over 16 ranks of arange values would land outside any
    # sensible tolerance.
    inp.copy_(
        (torch.arange(args.numel, device=inp.device, dtype=torch.float32) % 7 + ctx.rank)
        .to(torch_dtype)
    )
    out.zero_()

    torch.cuda.synchronize()
    dist.barrier(ctx.group)
    launch()
    torch.cuda.synchronize()

    ref = inp.clone()
    dist.all_reduce(ref, op=dist.ReduceOp.SUM, group=ctx.group)
    failures = check(ctx, out, ref, "allreduce_2d")

    if not args.no_bench and failures == 0:
        ref_buf = torch.empty_like(inp)

        def run_ref():
            ref_buf.copy_(inp)
            dist.all_reduce(ref_buf, op=dist.ReduceOp.SUM, group=ctx.group)

        # Allreduce convention: 2(W-1)/W, both directions of the two-shot exchange.
        moved = 2 * inp.numel() * inp.element_size() * (ctx.world_size - 1) // ctx.world_size
        bench_vs_torch(ctx, args, "allreduce_2d", launch, run_ref, moved)

    ctx.close()
    if ctx.is_leader:
        print("PASS" if failures == 0 else f"FAIL: {failures} rank(s) mismatched", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
