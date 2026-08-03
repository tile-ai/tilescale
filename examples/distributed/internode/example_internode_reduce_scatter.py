"""Inter-node reduce-scatter over NCCL GIN.

Every rank starts with a full-length input and ends with the elementwise sum of
all ranks' inputs, restricted to its own shard. The algorithm is the transpose of
the one-shot allgather: rank ``r`` pushes the slice destined for rank ``p`` into
slot ``r`` of ``p``'s scratch buffer, then each rank reduces the ``world_size``
slices it has collected.

The reduction happens on the receiving rank rather than in flight because GIN
2.28.9 has no remote atomic or reduce-on-put; the only device op that moves
payload is a plain put. Landing the contributions in separate slots and summing
locally costs one extra read of the scratch buffer, which is HBM-local and cheap
next to a network hop.

Shape of the work
-----------------
``grid = chunks`` -- one CTA per chunk, each owning that chunk end to end: it
sends the chunk to every peer, waits, then reduces it. Sizing the grid by chunks
rather than by payload keeps messages large, which is what RDMA needs (see the
allgather docstring for the measurement that motivated this).

The grid is *not* ``peers * chunks`` here, unlike the allgather. A CTA has to
reduce the whole chunk it owns, so splitting a chunk across CTAs by peer would
make the reduction depend on slices other CTAs received -- a grid-wide barrier
inside one kernel, which needs a cooperative launch. Looping over peers inside
the CTA keeps every dependency CTA-local. Puts to different peers land on
different QPs regardless, since a context has one QP per peer.

Scratch is allocated from the arena like everything else -- a GIN destination has
to live in the registered window.
"""

# NOTE: no `from __future__ import annotations` here. T.prim_func resolves the
# parameter annotations at runtime via get_type_hints, and PEP 563 would turn
# `T.Tensor((shard_numel,), dtype)` into a string evaluated against module
# globals -- where the closure locals shard_numel/dtype do not exist.
import argparse

import torch
import torch.distributed as dist

import tilelang.language as T
from tilelang.distributed.bench import do_bench

from internode_common import (
    SIGNAL_DATA,
    per_launch_signals,
    report_tuning,
    tune_grid,
    fp32_sum,
    Context,
    TL_DTYPES,
    TORCH_DTYPES,
    add_common_args,
    check,
    prepare_env,
    report,
)


def reduce_scatter_kernel(shard_numel: int, chunks: int, threads: int, world_size: int, dtype: str,
                          signal_id: int = SIGNAL_DATA):
    """Scatter each peer's slice, then sum the slices that arrive for this rank.

    One launch does both phases. The wait sits between them, so the reduction
    only reads slots whose payload has been signalled as landed.
    """
    peers = world_size - 1
    chunk_numel = shard_numel // chunks

    @T.prim_func
    def main(
        inp: T.Tensor((world_size * shard_numel,), dtype),
        scratch: T.Tensor((world_size * shard_numel,), dtype),
        out: T.Tensor((shard_numel,), dtype),
        rank: T.int32,
        signal_target: T.int32,
    ):
        with T.Kernel(chunks, threads=threads) as bx:
            base = bx * chunk_numel
            # `peers` is a Python int, so this unrolls into independent puts.
            for step in range(peers):
                peer = (rank + step + 1) % world_size
                # Send the part of our input that belongs to `peer`, into the
                # slot `peer` reserves for us. Symmetric arena, so the index we
                # write is the index the peer reads.
                T.nccl_gin.put_signal(
                    src=inp[peer * shard_numel + base],
                    dst=scratch[rank * shard_numel + base],
                    size=chunk_numel,
                    peer=peer,
                    signal_id=signal_id,
                    scope="block",
                )
            # Our own contribution needs no network hop.
            for i in T.Parallel(chunk_numel):
                scratch[rank * shard_numel + base + i] = inp[rank * shard_numel + base + i]
            T.nccl_gin.wait_signal(least=signal_target, signal_id=signal_id, scope="block")
            # world_size is a Python int, so this unrolls into one fp32
            # expression. See fp32_sum for why it must not use an accumulator.
            for i in T.Parallel(chunk_numel):
                out[base + i] = T.cast(
                    fp32_sum(
                        world_size,
                        lambda s: T.cast(scratch[s * shard_numel + base + i], "float32"),
                    ),
                    dtype,
                )

    return main


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()

    prepare_env()
    ctx = Context()

    if args.numel % ctx.world_size:
        raise SystemExit(f"--numel {args.numel} must be divisible by world_size {ctx.world_size}")
    shard_numel = args.numel // ctx.world_size
    peers = ctx.world_size - 1
    if not args.tune and shard_numel % args.chunks:
        raise SystemExit(f"shard {shard_numel} must be a multiple of --chunks {args.chunks}")

    torch_dtype = TORCH_DTYPES[args.dtype]
    itemsize = torch.empty((), dtype=torch_dtype).element_size()
    ctx.log(
        f"reduce_scatter: world_size={ctx.world_size} nodes={ctx.num_nodes} "
        f"numel={args.numel} shard={shard_numel} chunks={args.chunks} "
        f"put={shard_numel // args.chunks * itemsize / 1024:.0f}KiB threads={args.threads} "
        f"gin_contexts={args.gin_contexts} dtype={args.dtype}"
    )

    inp = ctx.tensor((args.numel,), torch_dtype)
    scratch = ctx.tensor((args.numel,), torch_dtype)
    out = ctx.tensor((shard_numel,), torch_dtype)
    # Small magnitudes: a bf16 sum of 16 ranks' worth of arange values would
    # otherwise land outside any sensible tolerance.
    inp.copy_(
        (torch.arange(args.numel, device=inp.device, dtype=torch.float32) % 7 + ctx.rank).to(
            torch_dtype
        )
    )
    scratch.zero_()
    out.zero_()

    ref = torch.empty_like(out)
    dist.reduce_scatter_tensor(ref, inp, op=dist.ReduceOp.SUM, group=ctx.group)
    moved = out.numel() * out.element_size() * peers
    ref_buf = torch.empty_like(out)

    def time_torch() -> float:
        return do_bench(
            lambda: dist.reduce_scatter_tensor(ref_buf, inp, op=dist.ReduceOp.SUM, group=ctx.group),
            warmup=args.warmup, rep=args.rep, group=ctx.group,
        )

    configs = tune_grid(args, signals_per_config=1)
    torch_before = time_torch() if not args.no_bench else float("nan")

    rows = []
    failures = 0
    for cfg in configs:
        if shard_numel % cfg["chunks"]:
            continue
        kernel = ctx.compile(
            reduce_scatter_kernel(
                shard_numel, cfg["chunks"], cfg["threads"], ctx.world_size,
                TL_DTYPES[args.dtype], signal_id=cfg["signals"][0],
            ),
            expect=("tl::gin::put_signal_addr", "tl::gin::wait_signal"),
            gin_contexts=cfg["gin_contexts"],
        )
        per_launch = per_launch_signals(peers, cfg["chunks"], args.signal_div)
        target = [0]

        def launch(kernel=kernel, target=target, per_launch=per_launch):
            target[0] += per_launch
            kernel(inp, scratch, out, ctx.rank, target[0])

        out.zero_()
        scratch.zero_()
        torch.cuda.synchronize()
        dist.barrier(ctx.group)
        launch()
        torch.cuda.synchronize()
        bad = check(ctx, out, ref, f"reduce_scatter[c{cfg['chunks']}/x{cfg['gin_contexts']}]")
        failures += bad

        ms = float("inf")
        if bad == 0 and not args.no_bench:
            dist.barrier(ctx.group)
            ms = do_bench(launch, warmup=args.warmup, rep=args.rep, group=ctx.group)
        rows.append({
            **cfg, "ok": bad == 0, "ms": ms,
            "gbps": (moved / (ms * 1e-3) / 1e9) if ms not in (float("inf"), 0) else 0.0,
        })
        dist.barrier(ctx.group)

    if not args.no_bench:
        report_tuning(ctx, "reduce_scatter", rows, torch_before, time_torch(), moved)

    ctx.close()
    if ctx.is_leader:
        print("PASS" if failures == 0 else f"FAIL: {failures} rank(s) mismatched", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
