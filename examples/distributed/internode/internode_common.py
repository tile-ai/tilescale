"""Shared setup for the inter-node GIN collectives.

Every example here follows the same shape: bring up a process group with the
network enabled, allocate symmetric buffers from the TileScale allocator (GIN can
only address the registered arena), run a kernel that moves bytes with
``T.nccl_gin``, check the result against torch, and time both.

The pieces live here rather than in each example because the environment setup is
easy to get subtly wrong -- ``init_dist`` disables InfiniBand by default, and a
cudaMalloc arena cannot be registered as an NCCL window -- and a silent mistake
in either shows up as a passing test that moved no data over the fabric.
"""

from __future__ import annotations

import argparse
import functools
import operator
import os

import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T

# Signal slots. The reduce-scatter half of allreduce and its allgather half must
# not share a slot: signals are cumulative totals, so two phases counting into
# one slot cannot be told apart.
SIGNAL_DATA = 0
SIGNAL_PHASE2 = 1


def fp32_sum(count: int, term):
    """Fold ``term(0) + ... + term(count-1)`` into one add-expression.

    ``term`` maps a source index to a PrimExpr; callers cast to float32 inside it,
    because a bf16 running sum over 16 ranks loses enough low bits to fail a
    tolerance check.

    Both the loop and the fold live here, outside any traced function, and that
    placement is the point. Two shapes that look more natural both fail inside a
    ``T.prim_func``:

    * ``acc = T.cast(...)`` then ``acc = acc + ...`` -- in a kernel body the eager
      builder treats assignment of a PrimExpr as a TIR variable *declaration*, so
      the reassignment emits a second variable (``acc_1``) and the enclosing
      ``T.Parallel`` frame rejects it.
    * ``fp32_sum([... for src in range(n)])`` -- the builder rewrites ``for``
      statements *and comprehension for-clauses* into TIR loops, so the
      comprehension raises ``'ForFrame' object is not iterable``.

    Calling a plain Python helper sidesteps both: the fold runs at trace time and
    only its result, a single expression, is emitted. That keeps the copy
    vectorisable.
    """
    return functools.reduce(operator.add, (term(k) for k in range(count)))


def prepare_env() -> None:
    """Set the environment GIN needs, before ``init_dist`` is imported or run.

    ``init_dist`` sets ``NCCL_IB_DISABLE=1`` unless it is already set, which would
    keep every transfer inside shared memory and make an "inter-node" benchmark
    measure nothing. The VMM and GIN flags are hard requirements of window
    registration; asserting them here turns a missing Device API into an error at
    startup instead of a null devcomm read inside a kernel.
    """
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ.setdefault("TILESCALE_USE_VMM", "1")
    os.environ.setdefault("TILESCALE_USE_GIN", "1")
    os.environ.setdefault("NCCL_DEBUG", "ERROR")


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--numel",
        type=int,
        default=1 << 22,
        help="elements in the collective's logical input, summed over ranks",
    )
    parser.add_argument("--block", type=int, default=8192, help="elements per CTA")
    # Chunks exist to spread a peer's transfer across GIN contexts (one context
    # is one QP per peer, so one NIC), NOT to parallelise within a channel.
    # Splitting further than the context count only shrinks messages.
    parser.add_argument(
        "--chunks",
        type=int,
        default=64,
        # Tuned on two idle nodes, 64 MB shard: the reduce variants climb with
        # chunk count and peak at 64 (allreduce 41.5 -> 47.2 GB/s from 4 to 64,
        # reduce_scatter 44.0 -> 46.2 from 16 to 64; 128 is worse), because their
        # CTAs also carry the reduction and want more of them. Allgather is flat
        # from 2 to 64, so 64 is a safe shared default.
        help="puts per peer; each becomes one CTA issuing one large put",
    )
    parser.add_argument(
        "--gin-contexts",
        type=int,
        default=4,
        # Contexts are insurance against a busy NIC, not a win on an idle one.
        # On two *idle* nodes one context already reaches line rate and extra
        # contexts cost ~1% (allgather 47.6 GB/s at 1 vs 47.0 at 4). On a NIC
        # shared with another tenant's job, one context collapsed to 23 GB/s while
        # 2-4 held ~44. Defaulting to 4 trades that 1% for the contended case.
        # The device clamps to what the devcomm granted (4 here, though the
        # allocator asks for 8) and scales the wait target to match.
        help="-DTL_GIN_CONTEXTS: spread CTAs over n GIN contexts (QPs); 1 pins to context 0",
    )
    parser.add_argument(
        "--signal-div",
        type=int,
        default=0,
        help="DEBUG ONLY: divide the wait target; under-waits, so any result is invalid",
    )
    parser.add_argument(
        "--wait-ctx0",
        action="store_true",
        help="-DTL_GIN_WAIT_CTX0: puts spread over contexts, every wait on context 0",
    )
    # 1024 threads matches triton-dist's num_warps=32 for its inter-node send
    # blocks: one CTA cooperatively driving one large put.
    parser.add_argument("--threads", type=int, default=1024)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--no-bench", action="store_true", help="check correctness only")
    parser.add_argument("--print-source", action="store_true")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="sweep the grid below in one process, verify each, report the best vs torch",
    )
    parser.add_argument("--tune-chunks", default="2,4,8,16", help="--tune: chunk counts to try")
    parser.add_argument("--tune-contexts", default="1,2,4", help="--tune: GIN context counts")
    parser.add_argument("--tune-threads", default="1024", help="--tune: threads per CTA")
    return parser


# GIN_SIGNAL_COUNT in nccl_window.py. Signals are "guaranteed to start at id=0",
# so ids [0, 32) are usable.
MAX_SIGNALS = 32


def tune_grid(args, signals_per_config: int = 1):
    """Configurations to try, each with its own signal slots.

    Why distinct slots per config rather than one shared slot: signals are
    cumulative and nothing resets them, and ``wait_signal`` divides the target by
    the device's ``context_span()``. A config that changes the context count would
    therefore divide a total accumulated under a *different* span, so the target
    would be wrong from that point on. Giving each config fresh slots means every
    counter starts at zero under exactly one span, which makes ``--tune-contexts``
    safe to sweep in a single process.
    """
    if not args.tune:
        return [{
            "chunks": args.chunks,
            "gin_contexts": args.gin_contexts,
            "threads": args.threads,
            "signals": tuple(range(signals_per_config)),
        }]

    ints = lambda s: [int(x) for x in str(s).split(",") if x != ""]
    grid = []
    for contexts in ints(args.tune_contexts):
        for chunks in ints(args.tune_chunks):
            for threads in ints(args.tune_threads):
                # A context must back the same number of CTAs for one wait target
                # to be right, so the span has to divide the chunk count.
                if chunks % max(1, contexts):
                    continue
                base = len(grid) * signals_per_config
                if base + signals_per_config > MAX_SIGNALS:
                    break
                grid.append({
                    "chunks": chunks,
                    "gin_contexts": contexts,
                    "threads": threads,
                    "signals": tuple(range(base, base + signals_per_config)),
                })
    if not grid:
        raise SystemExit("--tune grid is empty; check --tune-chunks / --tune-contexts")
    return grid


def report_tuning(ctx, name: str, rows, torch_ms_before: float, torch_ms_after: float, moved: int):
    """Print every configuration, then the winner against torch.

    torch is timed twice, before and after the sweep. If the two disagree the
    fabric moved under us and the ratios are not trustworthy -- worth knowing,
    since another tenant's job on the same NICs shifts these by up to 2x.
    """
    if not ctx.is_leader:
        return
    gbps = lambda ms: moved / (ms * 1e-3) / 1e9
    print(f"\n===== {name}: tuning results ({len(rows)} configs) =====", flush=True)
    print(f"{'chunks':>7} {'ctx':>4} {'thr':>5} {'ms':>9} {'GB/s':>8}  status", flush=True)
    for r in sorted(rows, key=lambda r: -(r["gbps"] if r["ok"] else -1)):
        print(
            f"{r['chunks']:>7} {r['gin_contexts']:>4} {r['threads']:>5} "
            f"{r['ms']:>9.3f} {r['gbps']:>8.1f}  {'PASS' if r['ok'] else 'FAIL'}",
            flush=True,
        )
    good = [r for r in rows if r["ok"]]
    if not good:
        print("no configuration passed", flush=True)
        return
    best = max(good, key=lambda r: r["gbps"])
    t_before, t_after = gbps(torch_ms_before), gbps(torch_ms_after)
    t_ms = min(torch_ms_before, torch_ms_after)          # torch at its best
    t_best = gbps(t_ms)
    drift = abs(t_before - t_after) / max(t_before, t_after)
    print(
        f"\n{name}: BEST chunks={best['chunks']} contexts={best['gin_contexts']} "
        f"threads={best['threads']}\n"
        f"  tilescale {best['ms']:.3f} ms  {best['gbps']:.1f} GB/s\n"
        f"  torch     {t_ms:.3f} ms  {t_best:.1f} GB/s   "
        f"(measured {t_before:.1f} then {t_after:.1f} GB/s, drift {drift * 100:.0f}%)\n"
        f"  speedup   {best['gbps'] / t_best:.2f}x vs torch's best of the two",
        flush=True,
    )
    if drift > 0.15:
        print("  WARNING: torch drifted >15% across the sweep; fabric was not stable", flush=True)


def per_launch_signals(peers: int, chunks: int, signal_div: int = 0) -> int:
    """Signal increments this rank actually receives per launch.

    Each of ``peers`` senders signals once per chunk, so the honest target is
    ``peers * chunks``. This is the only value that makes ``wait_signal`` mean
    "all my data has landed".

    ``signal_div`` divides it, and exists solely to investigate why multiple GIN
    contexts hang (see nccl_gin.h). It makes the wait weaker than the data, so the
    kernel can return before the payload arrives -- which shows up as bandwidth
    above what the hardware can carry. Any number produced with signal_div set is
    not a measurement of the collective.
    """
    total = peers * chunks
    if not signal_div:
        return total
    if total % signal_div:
        raise SystemExit(f"--signal-div {signal_div} must divide peers*chunks = {total}")
    return total // signal_div


TORCH_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
TL_DTYPES = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}


class Context:
    """Process group, allocator and topology for one rank."""

    def __init__(self, arena_bytes: int = 1 << 30):
        from tilelang.distributed.host import init_dist

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.local_world_size = int(
            os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count())
        )
        # Staged so a hang can be attributed. init_dist, allocator construction
        # (which creates the GIN devcomm and registers the arena window) and
        # compile are all collective; without markers they are one opaque block.
        if os.environ.get("TL_STAGE_TRACE"):
            print(f"[rank?] init_dist: enter local_rank={self.local_rank}", flush=True)
        self.rank, self.world_size, self.group, self.node_info = init_dist(
            self.local_rank, self.local_world_size, return_node_info=True
        )
        self.trace("init_dist: done")
        self.num_nodes = self.node_info.num_nodes if self.node_info is not None else 1
        self.trace(
            f"allocator: enter bytes={arena_bytes} nodes={self.num_nodes} "
            f"world={self.world_size} (devcomm + window register)"
        )
        self.allocator = tilelang.get_allocator(
            size=arena_bytes,
            device="cuda",
            is_distributed=True,
            local_rank=self.local_rank,
            num_local_ranks=self.local_world_size,
            group=self.group,
            node_info=self.node_info,
        )
        self.trace("allocator: done (arena window live)")

    @property
    def is_leader(self) -> bool:
        return self.rank == 0

    def tensor(self, shape, dtype: torch.dtype):
        """Allocate from the arena. Required: only the arena is a GIN window."""
        return tilelang.tensor(shape, dtype, allocator=self.allocator)

    def log(self, msg: str) -> None:
        if self.is_leader:
            print(msg, flush=True)

    def trace(self, msg: str) -> None:
        """Print from every rank, tagged.

        ``log`` is leader-only, so a non-leader rank cannot report progress at
        all: it looks identical whether it hung in setup, hung in compile, or
        ran fine. That ambiguity is what made the first two-node hang
        undiagnosable. Enabled by TL_STAGE_TRACE=1 to keep normal runs quiet.
        """
        if os.environ.get("TL_STAGE_TRACE"):
            print(f"[rank{self.rank}] {msg}", flush=True)

    def compile(self, func, *, expect: tuple[str, ...] = (), gin_contexts: int | None = None,
                wait_ctx0: bool = False):
        # Every rank checks the tokens, not just the leader. compile_once makes
        # this a collective, so a leader-only assertion aborts rank 0 while the
        # others march on into close()'s barrier and hang until the outer
        # timeout -- turning a clear assertion failure into a mystery stall.
        #
        # gin_contexts becomes -DTL_GIN_CONTEXTS=n. compile_flags is part of the
        # cache key, so each setting gets its own cache entry -- which is also
        # the only thing that keeps a sweep honest, since the key does not cover
        # the device headers this define lives in.
        flags = None if gin_contexts is None else [f"-DTL_GIN_CONTEXTS={int(gin_contexts)}"]
        if wait_ctx0:
            flags = (flags or []) + ["-DTL_GIN_WAIT_CTX0=1"]
        if os.environ.get("TL_GIN_DEBUG"):
            flags = (flags or []) + ["-DTL_GIN_DEBUG=1"]
        self.trace(f"compile: enter (collective in compile_once) flags={flags}")
        kernel = tilelang.compile(
            func, compile_once=True, compile_group=self.group, compile_flags=flags
        )
        self.trace("compile: lowered")
        if expect:
            source = kernel.get_kernel_source()
            # Without this the kernel still compiles and silently moves nothing,
            # which would read as a fast and correct-looking result.
            for token in expect:
                assert token in source, f"lowering did not emit {token!r}"
            assert "nccl_gin.h" in source, "generated code is missing the GIN header"
        kernel.initialize(allocator=self.allocator)
        self.trace("compile: initialized")
        return kernel

    def close(self) -> None:
        # allocator.close() is collective, so every rank must reach it even if
        # this rank's check failed. Guard the barrier: if a peer already died,
        # blocking here forever converts its error into a timeout on this rank
        # and buries the real message.
        self.trace("close: barrier")
        try:
            dist.barrier(self.group)
        except Exception as exc:  # noqa: BLE001 - report and keep tearing down
            print(f"[rank{self.rank}] close: barrier failed: {exc}", flush=True)
        self.trace("close: allocator")
        self.allocator.close()
        dist.destroy_process_group()
        self.trace("close: done")


def check(ctx: Context, got: torch.Tensor, want: torch.Tensor, name: str) -> int:
    """Compare on every rank and aggregate, so one bad rank fails the run."""
    # bf16 accumulation order differs between a tree reduction and our linear
    # one, so compare with a tolerance rather than exactly.
    if got.dtype in (torch.bfloat16, torch.float16):
        ok = torch.allclose(got.float(), want.float(), rtol=6e-2, atol=6e-2)
    else:
        ok = torch.allclose(got, want, rtol=1e-5, atol=1e-5)
    if not ok:
        diff = (got.float() - want.float()).abs()
        bad = (diff > 6e-2).nonzero().flatten()
        print(
            f"[rank {ctx.rank}] {name} MISMATCH: {bad.numel()}/{got.numel()} differ, "
            f"max |diff| {diff.max().item():.4g}, first at {bad[0].item() if bad.numel() else -1}",
            flush=True,
        )
    status = torch.tensor([0 if ok else 1], device=got.device, dtype=torch.int32)
    dist.all_reduce(status, group=ctx.group)
    failures = int(status.item())
    if failures == 0:
        ctx.log(f"{name}: correct on all {ctx.world_size} ranks")
    return failures


def report(ctx: Context, name: str, tl_ms: float, ref_ms: float, moved_bytes: int) -> None:
    """Print both timings plus the bus bandwidth each implies.

    ``moved_bytes`` is the payload that has to cross a rank's network link, not
    the buffer size, so the number is comparable between collectives with
    different algorithmic volumes.
    """
    if not ctx.is_leader:
        return
    tl_gbps = moved_bytes / (tl_ms * 1e-3) / 1e9
    ref_gbps = moved_bytes / (ref_ms * 1e-3) / 1e9
    speedup = ref_ms / tl_ms if tl_ms > 0 else float("nan")
    print(
        f"{name:<16} tilescale {tl_ms:8.3f} ms {tl_gbps:7.1f} GB/s | "
        f"torch {ref_ms:8.3f} ms {ref_gbps:7.1f} GB/s | speedup {speedup:5.2f}x",
        flush=True,
    )
