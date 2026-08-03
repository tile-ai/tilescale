"""Inter-node allreduce over NCCL GIN, one-shot or two-shot.

Every rank ends with the elementwise sum of all ranks' inputs, full length.
``--algo`` picks the algorithm, and which one wins is decided by rank count, not
by size:

* ``twoshot`` (default) -- reduce-scatter then allgather, so each rank reduces one
  shard and broadcasts it. Two dependent phases. Sends ``2 * (W-1) * N / W``.
* ``oneshot`` -- push the whole input to every peer, sum locally. One network
  phase. Sends ``(W-1) * N`` per rank.

Network volume alone says they tie at W=2 (the ratio is ``W / 2``) and that
one-shot should then win on having half the phases. **Measured, it loses**: 30.3
GB/s against two-shot's 43.4 at chunks=8, 64 MB shard, two nodes. Volume parity is
not parity, because one-shot reduces over the *full* buffer rather than a shard --
roughly twice the HBM traffic, plus a full-length local copy into scratch. Both
readings of this file's history were wrong in turn, first that one-shot only wins
when latency dominates, then that it must win at W=2; the profile decided it.

``oneshot`` is kept because the balance shifts with rank count and dtype, and it is
the cheaper shape when the reduction is trivial relative to the transfer.

Shape of the work
-----------------
``grid = chunks``, one CTA per chunk, owning that chunk end to end. Every
dependency stays inside one CTA, so no grid-wide barrier is needed and no
cooperative launch. See the allgather docstring for why the grid is sized by
chunks and channels rather than by payload.

Two-shot's phases use different signal slots. Signals are cumulative running
totals, so if both phases counted into one slot the phase-2 wait could be
satisfied by phase-1 arrivals and read shards that had not been reduced yet. Both
slots advance by the same amount per launch, so one ``signal_target`` serves both.

One-shot needs ``world_size * N`` of scratch against two-shot's ``N``, so it wants
a larger arena.
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
    SIGNAL_PHASE2,
    Context,
    TL_DTYPES,
    TORCH_DTYPES,
    add_common_args,
    check,
    prepare_env,
    report,
)


def allreduce_oneshot_kernel(numel: int, chunks: int, threads: int, world_size: int, dtype: str,
                             signal_id: int = SIGNAL_DATA):
    """Push the whole input to every peer, then reduce locally. One network phase.

    Volume, for ``W`` ranks and ``N`` elements: one-shot sends ``(W-1) * N`` per
    rank, two-shot ``2 * (W-1) * N / W`` -- ratio ``W / 2``, so equal at W=2.
    Equal bytes in one phase instead of two looks like a clear win, and it is not:
    measured 30.3 GB/s against two-shot's 43.4 (chunks=8, 64 MB shard, two nodes).
    The reduction here spans the whole buffer rather than a shard, so it costs
    about twice the HBM traffic and adds a full-length local copy into scratch.
    Not the default; see the module docstring.

    Each rank lands its contribution in slot ``rank`` of every peer's scratch, so
    the reduction is a straight sum over the ``world_size`` slots. This rank's own
    slot is filled by a local copy rather than a branch, since ``rank`` is a
    runtime value and a trace-time branch on it is not available.

    Scratch is ``world_size * numel``: bigger than the two-shot's, and the reason
    this needs a larger arena.
    """
    peers = world_size - 1
    chunk_numel = numel // chunks

    @T.prim_func
    def main(
        inp: T.Tensor((numel,), dtype),
        scratch: T.Tensor((world_size * numel,), dtype),
        out: T.Tensor((numel,), dtype),
        rank: T.int32,
        signal_target: T.int32,
    ):
        with T.Kernel(chunks, threads=threads) as bx:
            base = bx * chunk_numel
            for step in range(peers):
                peer = (rank + step + 1) % world_size
                # Slot by *sender* rank, so on the peer this lands in the slot it
                # reads for us. Symmetric arena, so the index matches.
                T.nccl_gin.put_signal(
                    src=inp[base],
                    dst=scratch[rank * numel + base],
                    size=chunk_numel,
                    peer=peer,
                    signal_id=signal_id,
                    scope="block",
                )
            # Our own contribution, no network hop. Reads inp, so it does not race
            # the puts above.
            for i in T.Parallel(chunk_numel):
                scratch[rank * numel + base + i] = inp[base + i]
            T.nccl_gin.wait_signal(least=signal_target, signal_id=signal_id, scope="block")
            for i in T.Parallel(chunk_numel):
                out[base + i] = T.cast(
                    fp32_sum(
                        world_size,
                        lambda s: T.cast(scratch[s * numel + base + i], "float32"),
                    ),
                    dtype,
                )

    return main


def allreduce_kernel(shard_numel: int, chunks: int, threads: int, world_size: int, dtype: str,
                     signal_id: int = SIGNAL_DATA, signal_id2: int = SIGNAL_PHASE2):
    """Scatter-reduce into an owned chunk, then broadcast every owned chunk.

    ``out`` doubles as the phase-2 destination and as the phase-1 reduction
    target: the reduced chunk is written to ``out[rank * shard + base]``, which is
    exactly where the allgather wants it, so no copy sits between the phases.
    """
    peers = world_size - 1
    chunk_numel = shard_numel // chunks

    @T.prim_func
    def main(
        inp: T.Tensor((world_size * shard_numel,), dtype),
        scratch: T.Tensor((world_size * shard_numel,), dtype),
        out: T.Tensor((world_size * shard_numel,), dtype),
        rank: T.int32,
        signal_target: T.int32,
    ):
        with T.Kernel(chunks, threads=threads) as bx:
            base = bx * chunk_numel
            # ---- phase 1: scatter-reduce ----
            for step in range(peers):
                peer = (rank + step + 1) % world_size
                T.nccl_gin.put_signal(
                    src=inp[peer * shard_numel + base],
                    dst=scratch[rank * shard_numel + base],
                    size=chunk_numel,
                    peer=peer,
                    signal_id=signal_id,
                    scope="block",
                )
            for i in T.Parallel(chunk_numel):
                scratch[rank * shard_numel + base + i] = inp[rank * shard_numel + base + i]
            T.nccl_gin.wait_signal(least=signal_target, signal_id=signal_id, scope="block")

            # world_size is a Python int, so this unrolls into one fp32
            # expression. See fp32_sum for why it must not use an accumulator.
            for i in T.Parallel(chunk_numel):
                out[rank * shard_numel + base + i] = T.cast(
                    fp32_sum(
                        world_size,
                        lambda s: T.cast(scratch[s * shard_numel + base + i], "float32"),
                    ),
                    dtype,
                )

            # ---- phase 2: allgather the reduced chunks ----
            # Reading out[rank*shard + base] as the put source is safe: this CTA
            # wrote exactly those bytes above, and a put by the same coop is
            # ordered after the writes it depends on.
            for step in range(peers):
                peer = (rank + step + 1) % world_size
                T.nccl_gin.put_signal(
                    src=out[rank * shard_numel + base],
                    dst=out[rank * shard_numel + base],
                    size=chunk_numel,
                    peer=peer,
                    signal_id=signal_id2,
                    scope="block",
                )
            T.nccl_gin.wait_signal(least=signal_target, signal_id=signal_id2, scope="block")

    return main


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    # twoshot by measurement, not by theory. One-shot moves the same network bytes
    # at W=2, but reduces over the full buffer instead of a shard -- ~2x the HBM
    # traffic plus a full-length local copy -- and lost: 30.3 GB/s against
    # two-shot's 43.4 at chunks=8. See allreduce_oneshot_kernel.
    parser.add_argument("--algo", choices=("twoshot", "oneshot"), default="twoshot")
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
        f"allreduce[{args.algo}]: world_size={ctx.world_size} nodes={ctx.num_nodes} "
        f"numel={args.numel} shard={shard_numel} chunks={args.chunks} "
        f"put={shard_numel // args.chunks * itemsize / 1024:.0f}KiB threads={args.threads} "
        f"gin_contexts={args.gin_contexts} dtype={args.dtype}"
    )

    inp = ctx.tensor((args.numel,), torch_dtype)
    # one-shot needs a slot per rank; two-shot only reduces its own shard.
    scratch_numel = ctx.world_size * args.numel if args.algo == "oneshot" else args.numel
    scratch = ctx.tensor((scratch_numel,), torch_dtype)
    out = ctx.tensor((args.numel,), torch_dtype)
    inp.copy_(
        (torch.arange(args.numel, device=inp.device, dtype=torch.float32) % 7 + ctx.rank).to(
            torch_dtype
        )
    )
    scratch.zero_()
    out.zero_()

    ref = inp.clone()
    dist.all_reduce(ref, op=dist.ReduceOp.SUM, group=ctx.group)
    # Both phases send one shard to each of world_size-1 peers.
    moved = 2 * shard_numel * inp.element_size() * peers
    ref_buf = inp.clone()

    def time_torch() -> float:
        return do_bench(
            lambda: dist.all_reduce(ref_buf, op=dist.ReduceOp.SUM, group=ctx.group),
            warmup=args.warmup, rep=args.rep, group=ctx.group,
        )

    # two-shot burns two signal slots per config, one per phase.
    configs = tune_grid(args, signals_per_config=1 if args.algo == "oneshot" else 2)
    torch_before = time_torch() if not args.no_bench else float("nan")

    rows = []
    failures = 0
    for cfg in configs:
        span = args.numel if args.algo == "oneshot" else shard_numel
        if span % cfg["chunks"]:
            continue
        if args.algo == "oneshot":
            func = allreduce_oneshot_kernel(
                args.numel, cfg["chunks"], cfg["threads"], ctx.world_size,
                TL_DTYPES[args.dtype], signal_id=cfg["signals"][0],
            )
        else:
            func = allreduce_kernel(
                shard_numel, cfg["chunks"], cfg["threads"], ctx.world_size,
                TL_DTYPES[args.dtype], signal_id=cfg["signals"][0], signal_id2=cfg["signals"][1],
            )
        kernel = ctx.compile(
            func,
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
        bad = check(ctx, out, ref, f"allreduce[{args.algo}/c{cfg['chunks']}/x{cfg['gin_contexts']}]")
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
        report_tuning(ctx, f"allreduce[{args.algo}]", rows, torch_before, time_torch(), moved)

    ctx.close()
    if ctx.is_leader:
        print("PASS" if failures == 0 else f"FAIL: {failures} rank(s) mismatched", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
