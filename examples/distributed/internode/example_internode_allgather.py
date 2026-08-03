"""Inter-node allgather over NCCL GIN.

Each rank holds one shard and ends up with every rank's shard concatenated in
rank order. A rank pushes its own shard directly into slot ``rank`` of every
other rank's output, then waits for the slots it does not own.

Shape of the work, and why
--------------------------
One CTA issues one large put. The grid is ``(world_size - 1) * chunks``, so it is
sized by *peers and channels*, not by payload -- 8 CTAs for a 2-node run, not
512. This mirrors Triton-distributed's inter-node sender, which launches
``grid = (n_nodes - 1,)`` with ``num_warps=32`` and hands each block a single
``putmem_signal_block`` covering a whole shard.

The earlier version here inverted that: it launched one CTA per 8192-element
block, so an 8 MB shard became 512 separate 16 KB puts with 512 signal
increments. RDMA at 400 Gbps is bandwidth-bound only once messages are large;
512 small messages on one queue pair is latency-bound, which is what held this
kernel to ~2 GB/s against torch NCCL's ~30 GB/s.

``chunks`` is not a tiling parameter. One GIN context is one QP per peer, hence
one NIC; splitting a peer's transfer into ``chunks`` pieces lets ``make_gin()``
place each piece on a different context and use several NICs at once. Splitting
beyond the granted context count only shrinks messages for no extra
parallelism, so ``--chunks`` and ``--gin-contexts`` want to match.

Signals are cumulative
----------------------
GIN signals are running totals that a wait does not consume, so ``least`` must be
a per-launch target rather than a constant. With a constant, the second launch
finds the counter already at the target and returns *without waiting for any
data* -- correct on the first call only, and it silently turns a benchmark into a
measurement of nothing. ``signal_target`` is therefore passed in and advanced by
the host, the same way Triton-distributed threads its ``signal_target`` through
``NVSHMEM_SIGNAL_SET``.

Launch: see run_internode.sh.
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
    Context,
    TL_DTYPES,
    TORCH_DTYPES,
    add_common_args,
    check,
    prepare_env,
    report,
)


def allgather_kernel(shard_numel: int, chunks: int, threads: int, world_size: int, dtype: str,
                     signal_id: int = SIGNAL_DATA):
    """One CTA per (peer, chunk); each issues a single large put.

    ``peer`` is selected by block index rather than looped inside the CTA so
    every put is independent and can sit on its own context.
    """
    peers = world_size - 1
    chunk_numel = shard_numel // chunks
    blocks = peers * chunks
    # Each block also copies one slice of the local shard into place, so the
    # slices tile the shard exactly once across the grid.
    copy_numel = shard_numel // blocks

    @T.prim_func
    def main(
        shard: T.Tensor((shard_numel,), dtype),
        out: T.Tensor((world_size * shard_numel,), dtype),
        rank: T.int32,
        signal_target: T.int32,
    ):
        with T.Kernel(blocks, threads=threads) as bx:
            peer_idx = bx // chunks
            chunk_idx = bx % chunks
            # Rotating the peer by rank keeps concurrent senders from all
            # hammering the same destination first.
            peer = (rank + peer_idx + 1) % world_size
            # Destination is slot `rank` of the peer's output. The arena is
            # symmetric, so the index written is the index the peer reads.
            T.nccl_gin.put_signal(
                src=shard[chunk_idx * chunk_numel],
                dst=out[rank * shard_numel + chunk_idx * chunk_numel],
                size=chunk_numel,
                peer=peer,
                signal_id=signal_id,
                scope="block",
            )
            # The local shard never crosses the network; copy it while the puts
            # are in flight. T.Parallel spreads this across the CTA's threads --
            # a T.serial loop here would have every thread redundantly walk the
            # whole slice.
            for i in T.Parallel(copy_numel):
                out[rank * shard_numel + bx * copy_numel + i] = shard[bx * copy_numel + i]
            T.nccl_gin.wait_signal(least=signal_target, signal_id=signal_id, scope="block")

    return main


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()

    prepare_env()
    ctx = Context()

    # --numel is the gathered size, so it must split evenly into shards.
    if args.numel % ctx.world_size:
        raise SystemExit(f"--numel {args.numel} must be divisible by world_size {ctx.world_size}")
    shard_numel = args.numel // ctx.world_size
    peers = ctx.world_size - 1
    # Only validate the single requested config here. Under --tune each config is
    # checked in the loop and unusable ones are skipped rather than fatal.
    if not args.tune:
        blocks = peers * args.chunks
        if shard_numel % args.chunks:
            raise SystemExit(f"shard {shard_numel} must be a multiple of --chunks {args.chunks}")
        if shard_numel % blocks:
            raise SystemExit(
                f"shard {shard_numel} must be a multiple of (world_size-1)*chunks = {blocks} "
                "so the local copy tiles evenly"
            )

    torch_dtype = TORCH_DTYPES[args.dtype]
    ctx.log(
        f"allgather: world_size={ctx.world_size} nodes={ctx.num_nodes} "
        f"numel={args.numel} shard={shard_numel} dtype={args.dtype} "
        + (
            f"tune chunks={args.tune_chunks} contexts={args.tune_contexts} "
            f"threads={args.tune_threads}"
            if args.tune
            else f"chunks={args.chunks} contexts={args.gin_contexts} threads={args.threads}"
        )
    )

    shard = ctx.tensor((shard_numel,), torch_dtype)
    out = ctx.tensor((args.numel,), torch_dtype)
    # Rank-dependent values so a slot filled by the wrong peer, or not at all,
    # cannot compare equal by accident.
    shard.copy_(
        torch.arange(shard_numel, device=shard.device, dtype=torch.float32).to(torch_dtype)
        + ctx.rank * 1000.0
    )
    out.zero_()

    # Golden result, once. Each rank sends its shard to world_size-1 peers; that
    # egress is what the link has to carry.
    ref = torch.empty_like(out)
    dist.all_gather_into_tensor(ref, shard, group=ctx.group)
    moved = shard.numel() * shard.element_size() * peers
    ref_buf = torch.empty_like(out)

    def time_torch() -> float:
        return do_bench(
            lambda: dist.all_gather_into_tensor(ref_buf, shard, group=ctx.group),
            warmup=args.warmup, rep=args.rep, group=ctx.group,
        )

    configs = tune_grid(args, signals_per_config=1)
    torch_before = time_torch() if not args.no_bench else float("nan")

    rows = []
    failures = 0
    for cfg in configs:
        blocks = peers * cfg["chunks"]
        if shard_numel % cfg["chunks"] or shard_numel % blocks:
            continue
        kernel = ctx.compile(
            allgather_kernel(
                shard_numel, cfg["chunks"], cfg["threads"], ctx.world_size,
                TL_DTYPES[args.dtype], signal_id=cfg["signals"][0],
            ),
            expect=("tl::gin::put_signal_addr", "tl::gin::wait_signal"),
            gin_contexts=cfg["gin_contexts"],
            wait_ctx0=args.wait_ctx0,
        )
        # This config's slot starts at zero, so its own running total is all that
        # matters -- see tune_grid for why the slots are not shared.
        per_launch = per_launch_signals(peers, cfg["chunks"], args.signal_div)
        target = [0]

        def launch(kernel=kernel, target=target, per_launch=per_launch):
            target[0] += per_launch
            kernel(shard, out, ctx.rank, target[0])

        out.zero_()
        torch.cuda.synchronize()
        dist.barrier(ctx.group)
        launch()
        torch.cuda.synchronize()
        bad = check(ctx, out, ref, f"allgather[c{cfg['chunks']}/x{cfg['gin_contexts']}]")
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
        report_tuning(ctx, "allgather", rows, torch_before, time_torch(), moved)

    ctx.close()
    if ctx.is_leader:
        print("PASS" if failures == 0 else f"FAIL: {failures} rank(s) mismatched", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
