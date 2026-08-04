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
* **The halves need different GIN signals.** Signal state is cumulative and a wait does
  not consume it, so if both used ``SIGNAL_DATA`` the allgather's wait would already be
  satisfied by the reduce-scatter's arrivals and would return without waiting for
  anything -- silently, and only under repetition. 32 signals are provisioned.

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

from internode_2d import Allgather2D, ReduceScatter2D, add_2d_args, pick_intra
from internode_common import (
    SIGNAL_DATA,
    SIGNAL_PHASE2,
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
    args = parser.parse_args()

    prepare_env()
    # The multicast buffer is sized before the allocator exists. Both halves want one
    # -- the reduce-scatter reduces its input through the switch, the allgather
    # broadcasts into its output -- hence two buffers of `numel`.
    world = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    itemsize = torch.empty((), dtype=TORCH_DTYPES[args.dtype]).element_size()
    intra = pick_intra(args.intra)
    ctx = Context(mcast_bytes=2 * args.numel * itemsize if intra == "multimem" else 0)

    torch_dtype, tl_dtype = TORCH_DTYPES[args.dtype], TL_DTYPES[args.dtype]
    nodes = ctx.world_size // ctx.local_world_size
    ctx.log(
        f"allreduce_2d: world={ctx.world_size} nodes={nodes} local={ctx.local_world_size} "
        f"numel={args.numel} shard={args.numel // ctx.world_size} chunks={args.chunks} "
        f"intra={intra} mc_threads={args.mc_threads} mc_tiles={args.mc_tiles} "
        f"overlap={not args.no_overlap} contexts={args.gin_contexts} dtype={args.dtype}"
    )

    rs = ReduceScatter2D(ctx, args.numel, torch_dtype, tl_dtype, args, intra=intra,
                         signal_id=SIGNAL_DATA)
    # Hand over in place, and on a different signal: see the module docstring.
    ag = Allgather2D(ctx, args.numel, torch_dtype, tl_dtype, args, intra=intra,
                     signal_id=SIGNAL_PHASE2, shard=rs.out)

    inp, out = rs.inp, ag.out
    # Small magnitudes: a bf16 sum over 16 ranks of arange values would land outside any
    # sensible tolerance.
    inp.copy_(
        (torch.arange(args.numel, device=inp.device, dtype=torch.float32) % 7 + ctx.rank)
        .to(torch_dtype)
    )
    out.zero_()

    def launch():
        rs.launch()
        ag.launch()

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
