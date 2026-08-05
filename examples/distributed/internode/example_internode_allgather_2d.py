"""Hierarchical (2D) inter-node allgather: rail-aligned GIN plus an NVSwitch broadcast.

The flat allgather in ``example_internode_allgather.py`` pushes every rank's shard to
all ``world_size - 1`` peers over GIN. That is optimal with one GPU per node and
catastrophic with eight: each rank puts 15 shards onto the NIC, including the 7 for
siblings on the same machine, and it measures 2.7 GB/s against torch's 309.

The kernels live in ``internode_2d`` -- shared with the other three collectives, which
are the same two halves recombined -- so this file is just the shape, the reference and
the measurement. Read ``internode_2d.Allgather2D`` for what overlaps and why, and the
module docstring there for the three traps worth knowing before editing any of it.

Measured, 16 GPUs on 2 nodes, 240 MB bf16:

```
flat              2.7 GB/s
2D pull         358.6 GB/s   torch 309   1.16x
2D multimem     403.6 GB/s   torch 309   1.31x   (triton-dist push_2d: 290)
```

against a roofline of ~700 GB/s -- one shard over this rank's own NIC
(15.7 MB / 47.6 GB/s = 0.33 ms) and fourteen over NVLink (220 MB / 670 GB/s = 0.33 ms),
two legs that happen to balance at 8 GPUs against 8 400-Gbps NICs. ``--phases`` prints
where the time actually goes.
"""

# NOTE: no `from __future__ import annotations` here -- see internode_2d.
import argparse

import torch
import torch.distributed as dist

from internode_2d import Allgather2D, add_2d_args, pick_intra, report_phases
from internode_common import (
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
    parser.add_argument("--phases", action="store_true",
                        help="time each kernel on its own, to locate headroom")
    # Allgather's intra-node half is two multicast publishes and no reduce, so it is
    # switch-bound rather than occupancy-bound and wants fatter CTAs than the
    # reduce-carrying collectives: 32 tiles gives 0.164 ms against 0.233 at 4 for 48 MB,
    # and 0.501 against 0.585 for 240 MB.
    #
    # But not at the smallest sizes. At 32 MB (a 2 MB shard) the curve peaks at 16, and 32 is
    # well past it: tiles 2/8/16/32 measure 183.7 / 210.4 / 216.9 / 194.5 GB/s. So this is a
    # threshold fitted to three shard sizes -- 2 MB wants 16, 3 MB and 15.7 MB want 32 -- and
    # nothing finer. --mc-tiles overrides it.
    parser.set_defaults(mc_tiles=0)
    args = parser.parse_args()

    prepare_env()
    # The multicast buffer is sized before the allocator exists, so the output length
    # has to be known here rather than after Context.
    itemsize = torch.empty((), dtype=TORCH_DTYPES[args.dtype]).element_size()
    intra = pick_intra(args.intra)
    ctx = Context(mcast_bytes=args.numel * itemsize if intra == "multimem" else 0)

    torch_dtype, tl_dtype = TORCH_DTYPES[args.dtype], TL_DTYPES[args.dtype]
    nodes = ctx.world_size // ctx.local_world_size
    ctx.log(
        f"allgather_2d: world={ctx.world_size} nodes={nodes} local={ctx.local_world_size} "
        f"numel={args.numel} shard={args.numel // ctx.world_size} chunks={args.chunks} "
        f"intra={intra} mc_threads={args.mc_threads} mc_tiles={args.mc_tiles} "
        f"overlap={not args.no_overlap} contexts={args.gin_contexts} dtype={args.dtype}"
    )

    if not args.mc_tiles:
        args.mc_tiles = 16 if (args.numel // ctx.world_size) * itemsize < 3_000_000 else 32
    ag = Allgather2D(ctx, args.numel, torch_dtype, tl_dtype, args, intra=intra)
    ag.shard.copy_(
        torch.arange(ag.shard_numel, device=ag.shard.device, dtype=torch.float32)
        .to(torch_dtype) + ctx.rank * 1000.0
    )

    torch.cuda.synchronize()
    dist.barrier(ctx.group)
    ag.launch()
    torch.cuda.synchronize()

    ref = torch.empty_like(ag.out)
    dist.all_gather_into_tensor(ref, ag.shard, group=ctx.group)
    failures = check(ctx, ag.out, ref, "allgather_2d")

    if failures == 0:
        if args.phases:
            report_phases(ctx, ag, args)
        if not args.no_bench:
            moved = ag.shard.numel() * ag.shard.element_size() * (ctx.world_size - 1)
            bench_vs_torch(
                ctx, args, "allgather_2d", ag.launch,
                lambda: dist.all_gather_into_tensor(ref, ag.shard, group=ctx.group), moved,
            )

    ctx.close()
    if ctx.is_leader:
        print("PASS" if failures == 0 else f"FAIL: {failures} rank(s) mismatched", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
