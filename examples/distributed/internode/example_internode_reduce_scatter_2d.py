"""Hierarchical (2D) inter-node reduce-scatter: NVSwitch reduce, then rail-aligned GIN.

The transpose of the 2D allgather, and the flat version collapses at 16 GPUs for the
same reason: every rank pushes a slice to all 15 peers over GIN, including the 7 for
siblings on the same machine.

The kernels live in ``internode_2d``. Read ``internode_2d.ReduceScatter2D`` for why this
direction needs no barrier at all, and the module docstring there for the traps.

Measured, 16 GPUs on 2 nodes, 240 MB bf16:

```
flat              2.5 GB/s
2D pull         293.9 GB/s   torch 319   0.92x
2D multimem     397.2 GB/s   torch 322   1.24x
```

The pull path structurally cannot reach torch: reducing on the consumer means moving
``(lws-1)/lws`` of the input over NVLink into a staging buffer and reading it back,
~210 MB of NVLink plus ~500 MB of avoidable HBM per rank. That was exactly the 8% it
lost, and no tuning removes it. ``multimem.ld_reduce`` reduces in the switch instead --
the same NVLS mechanism torch is using here, which is what makes the comparison fair
rather than us fighting a switch-reduce with a copy loop.
"""

# NOTE: no `from __future__ import annotations` here -- see internode_2d.
import argparse

import torch
import torch.distributed as dist

from internode_2d import ReduceScatter2D, add_2d_args, pick_intra, report_phases
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
    args = parser.parse_args()

    prepare_env()
    itemsize = torch.empty((), dtype=TORCH_DTYPES[args.dtype]).element_size()
    intra = pick_intra(args.intra)
    ctx = Context(mcast_bytes=args.numel * itemsize if intra == "multimem" else 0)

    torch_dtype, tl_dtype = TORCH_DTYPES[args.dtype], TL_DTYPES[args.dtype]
    nodes = ctx.world_size // ctx.local_world_size
    ctx.log(
        f"reduce_scatter_2d: world={ctx.world_size} nodes={nodes} "
        f"local={ctx.local_world_size} numel={args.numel} "
        f"shard={args.numel // ctx.world_size} chunks={args.chunks} intra={intra} "
        f"mc_threads={args.mc_threads} mc_tiles={args.mc_tiles} "
        f"intra_chunks={args.intra_chunks} overlap={not args.no_overlap} "
        f"contexts={args.gin_contexts} dtype={args.dtype}"
    )

    rs = ReduceScatter2D(ctx, args.numel, torch_dtype, tl_dtype, args, intra=intra)
    # Small magnitudes: a bf16 sum over 16 ranks of arange values would land outside any
    # sensible tolerance.
    rs.inp.copy_(
        (torch.arange(args.numel, device=rs.inp.device, dtype=torch.float32) % 7 + ctx.rank)
        .to(torch_dtype)
    )

    torch.cuda.synchronize()
    dist.barrier(ctx.group)
    rs.launch()
    torch.cuda.synchronize()

    ref = torch.empty_like(rs.out)
    dist.reduce_scatter_tensor(ref, rs.inp, op=dist.ReduceOp.SUM, group=ctx.group)
    failures = check(ctx, rs.out, ref, "reduce_scatter_2d")

    if failures == 0:
        if args.phases:
            report_phases(ctx, rs, args)
        if not args.no_bench:
            ref_buf = torch.empty_like(rs.out)
            moved = rs.out.numel() * rs.out.element_size() * (ctx.world_size - 1)
            bench_vs_torch(
                ctx, args, "reduce_scatter_2d", rs.launch,
                lambda: dist.reduce_scatter_tensor(ref_buf, rs.inp, op=dist.ReduceOp.SUM,
                                                   group=ctx.group),
                moved,
            )

    ctx.close()
    if ctx.is_leader:
        print("PASS" if failures == 0 else f"FAIL: {failures} rank(s) mismatched", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
