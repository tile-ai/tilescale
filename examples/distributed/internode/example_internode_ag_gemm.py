"""Inter-node AllGather-GEMM over NCCL GIN.

``C = allgather(A_shard) @ B``. Activations are sharded by row across ranks and
gathered before the GEMM; ``B`` is replicated, so every rank computes the full
output. This is the shape TP inference uses in front of a column-parallel matmul.

Two kernels on one stream, not one fused kernel
-----------------------------------------------
The allgather runs as its own launch, then the GEMM. Stream ordering is what makes
it correct: the allgather kernel ends with ``wait_signal``, so every remote row has
landed before the GEMM's first load.

The tempting fused version -- have some CTAs of the GEMM grid issue the puts while
the rest wait on the signal -- deadlocks as soon as the grid stops being
co-resident. A CTA waiting on a signal occupies an SM, and if the CTAs that would
have issued the puts are still queued behind it, nothing progresses. The GEMM grid
here is ``ceildiv(M,128) * ceildiv(N,128)`` blocks, far more than fit at once, so
that failure is the default rather than the exception. Overlapping safely needs the
comm kernel on a *separate stream* (which is how Triton-distributed does it, with
its own reduction/comm streams) so both are resident independently.

Cost of not overlapping: the comm and the GEMM serialize, so the total is their
sum rather than the max. That is the honest baseline to beat, and the reason the
report below prints the two phases separately.
"""

# NOTE: no `from __future__ import annotations` here -- see the allgather example.
import argparse

import torch
import torch.distributed as dist

import tilelang.language as T
from tilelang.distributed.bench import do_bench

from example_internode_allgather import allgather_kernel
from internode_gemm_sm100 import tcgen05_gemm_range_kernel
from internode_common import (
    SIGNAL_DATA,
    Context,
    TL_DTYPES,
    TORCH_DTYPES,
    add_common_args,
    check,
    per_launch_signals,
    prepare_env,
)


def gemm_kernel(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    threads: int,
    dtype: str,
    accum_dtype: str = "float32",
):
    """Plain tiled GEMM over the gathered activations.

    Accumulate in fp32 and store in ``dtype``: a bf16 accumulator over a
    K-length dot product drifts far outside any tolerance worth setting.
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def gemm_overlap_kernel(
    M: int, N: int, K: int, M_per_rank: int,
    block_M: int, block_N: int, block_K: int, threads: int, dtype: str,
    signal_id: int = SIGNAL_DATA, accum_dtype: str = "float32",
):
    """GEMM that waits per output tile, so it overlaps with the allgather.

    Output rows partition by rank: a tile whose rows come from this rank's own
    shard needs no remote data and starts immediately, while a tile owned by a
    peer waits on the GIN signal first. With two ranks that means half the grid
    runs while the network is still busy.

    This only works with the allgather on a *separate stream*. Both kernels must
    be resident at once: the waiting CTAs hold SMs, so if the comm kernel were
    queued behind them on the same stream nothing would ever signal. The comm grid
    is small (chunks CTAs) so it schedules alongside the GEMM.

    ``owner`` is block-uniform, so every thread in a CTA takes the same branch and
    the whole-CTA wait is well formed.
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
        rank: T.int32,
        signal_target: T.int32,
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            owner = (by * block_M) // M_per_rank
            if owner != rank:
                T.nccl_gin.wait_signal(least=signal_target, signal_id=signal_id, scope="block")
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def gemm_overlap_persistent_kernel(
    M: int, N: int, K: int, M_per_rank: int,
    block_M: int, block_N: int, block_K: int, threads: int, dtype: str, sm_num: int,
    signal_id: int = SIGNAL_DATA, accum_dtype: str = "float32",
):
    """Persistent overlapped GEMM: a fixed grid that leaves SMs for the comm kernel.

    The non-persistent version deadlocks. Its grid is ceildiv(M,block_M) *
    ceildiv(N,block_N) CTAs -- 2048 for an 8192x4096 output -- and the ones waiting
    on the GIN signal occupy every SM, so the comm kernel on the side stream never
    gets scheduled to send the signal they are waiting for. Correctness passed on a
    single launch and the benchmark then hung, which is exactly that race.

    Fixing it needs SM partitioning, not just separate streams: cap the grid at
    ``sm_num`` CTAs so the comm kernel's much smaller grid always has room to run
    alongside. This is what Triton-distributed's num_sync_sms / num_p2p_sms split
    is for.

    Each CTA strides over the tile space. Re-waiting per tile is free after the
    first: the signal is cumulative, so a satisfied wait returns immediately.
    """
    n_blocks = T.ceildiv(N, block_N)
    total_tiles = T.ceildiv(M, block_M) * n_blocks

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
        rank: T.int32,
        signal_target: T.int32,
    ):
        with T.Kernel(sm_num, threads=threads) as bid:
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            for t in T.serial(T.ceildiv(total_tiles, sm_num)):
                tile = bid + t * sm_num
                if tile < total_tiles:
                    pid_m = tile // n_blocks
                    pid_n = tile % n_blocks
                    owner = (pid_m * block_M) // M_per_rank
                    if owner != rank:
                        T.nccl_gin.wait_signal(least=signal_target, signal_id=signal_id,
                                               scope="block")
                    T.clear(C_local)
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                        T.copy(A[pid_m * block_M, k * block_K], A_shared)
                        T.copy(B[k * block_K, pid_n * block_N], B_shared)
                        T.gemm(A_shared, B_shared, C_local)
                    T.copy(C_local, C[pid_m * block_M, pid_n * block_N])

    return main


def gemm_range_kernel(
    M: int, N: int, K: int, m_rows: int,
    block_M: int, block_N: int, block_K: int, threads: int, dtype: str,
    accum_dtype: str = "float32",
):
    """GEMM over a contiguous row range, chosen at launch by ``m_offset``.

    This is what makes overlap work without any in-kernel waiting, and therefore
    without the co-residency deadlock. The host issues three launches: the comm on
    a side stream, this kernel over the rows this rank already owns (no dependency,
    so it runs while the network is busy), then after a stream wait, this kernel
    again over each peer's rows.

    Preferred over the persistent + in-kernel-wait version, which was both
    deadlock-prone and slower: capping the grid at the SM count costs more than
    the overlap gains (1.429 ms against 1.362 for the plain serial path).
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
        m_offset: T.int32,
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(m_rows, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            row = m_offset + by * block_M
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[row, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[row, bx * block_N])

    return main


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--m-per-rank", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=64)
    # Separate from --threads: the collectives want 1024 threads to drive one big
    # put, but TileLang adds warp-specialisation threads on top of the GEMM's
    # request, so 1024 there overflows the 1024-thread block limit (it launched
    # 1152 and CUDA rejected it). 256 is the usual choice for 128x128x64 tiles.
    parser.add_argument("--gemm-threads", type=int, default=256)
    # tcgen05 is the Blackwell path (warp-specialised persistent, TMEM
    # accumulators): 1334 TFLOP/s against the naive loop's 470 on this shape.
    parser.add_argument("--gemm-impl", choices=("tcgen05", "naive"), default="tcgen05")
    parser.add_argument("--gemm-block-n", type=int, default=256)
    parser.add_argument("--mode", choices=("serial", "split", "persistent"), default="split",
                        help="split = comm on a side stream overlapped with the local-row GEMM")
    parser.add_argument("--overlap", action="store_true", help=argparse.SUPPRESS)
    # Must leave room for the comm kernel's CTAs or the waiting GEMM CTAs deadlock.
    parser.add_argument("--gemm-sms", type=int, default=132,
                        help="--overlap: persistent GEMM CTAs, below the SM count")
    args = parser.parse_args()

    prepare_env()
    ctx = Context(arena_bytes=1 << 31)

    peers = ctx.world_size - 1
    M_per_rank, N, K = args.m_per_rank, args.n, args.k
    M = M_per_rank * ctx.world_size
    shard_numel = M_per_rank * K
    if shard_numel % args.chunks or shard_numel % (peers * args.chunks):
        raise SystemExit(f"M_per_rank*K = {shard_numel} must divide by chunks and peers*chunks")

    torch_dtype = TORCH_DTYPES[args.dtype]
    tl_dtype = TL_DTYPES[args.dtype]
    ctx.log(
        f"ag_gemm: world={ctx.world_size} nodes={ctx.num_nodes} M={M} (per rank {M_per_rank}) "
        f"N={N} K={K} chunks={args.chunks} contexts={args.gin_contexts} dtype={args.dtype}"
    )

    ag = ctx.compile(
        allgather_kernel(shard_numel, args.chunks, args.threads, ctx.world_size, tl_dtype,
                         signal_id=SIGNAL_DATA),
        expect=("tl::gin::put_signal_addr", "tl::gin::wait_signal"),
        gin_contexts=args.gin_contexts,
    )
    if args.mode == "persistent":
        gemm = ctx.compile(
            gemm_overlap_persistent_kernel(M, N, K, M_per_rank, args.block_m, args.block_n,
                                           args.block_k, args.gemm_threads, tl_dtype,
                                           sm_num=args.gemm_sms, signal_id=SIGNAL_DATA),
            expect=("tl::gin::wait_signal",),
            gin_contexts=args.gin_contexts,
        )
    elif args.gemm_impl == "tcgen05":
        # One compiled kernel per row-range size: the full M for serial, one
        # rank's block for split.
        rows = M if args.mode == "serial" else M_per_rank
        gemm = ctx.compile(
            tcgen05_gemm_range_kernel(M, N, K, rows, block_M=args.block_m,
                                      block_N=args.gemm_block_n, block_K=args.block_k)
        )
    elif args.mode == "split":
        gemm = ctx.compile(
            gemm_range_kernel(M, N, K, M_per_rank, args.block_m, args.block_n, args.block_k,
                              args.gemm_threads, tl_dtype)
        )
    else:
        gemm = ctx.compile(
            gemm_kernel(M, N, K, args.block_m, args.block_n, args.block_k, args.gemm_threads,
                        tl_dtype)
        )
    # priority=-1: nudge the scheduler to start the comm kernel promptly.
    comm_stream = torch.cuda.Stream(priority=-1) if args.mode != "serial" else None

    # A_shard and A_full must both live in the arena: A_full is a GIN destination,
    # and A_shard is a GIN source.
    A_shard = ctx.tensor((shard_numel,), torch_dtype)
    A_full = ctx.tensor((ctx.world_size * shard_numel,), torch_dtype)
    B = ctx.tensor((K * N,), torch_dtype)
    C = ctx.tensor((M * N,), torch_dtype)

    gen = torch.Generator(device="cuda").manual_seed(1234 + ctx.rank)
    A_shard.copy_(torch.randn(shard_numel, generator=gen, device="cuda", dtype=torch.float32) * 0.05)
    # B is replicated, so every rank must generate the same values.
    genb = torch.Generator(device="cuda").manual_seed(999)
    B.copy_(torch.randn(K * N, generator=genb, device="cuda", dtype=torch.float32) * 0.05)
    A_full.zero_()
    C.zero_()

    A2, B2, C2 = A_full.view(M, K), B.view(K, N), C.view(M, N)
    per_launch = per_launch_signals(peers, args.chunks, args.signal_div)
    target = [0]

    def launch():
        target[0] += per_launch
        if args.mode == "split":
            main_s = torch.cuda.current_stream()
            comm_stream.wait_stream(main_s)
            with torch.cuda.stream(comm_stream):
                ag(A_shard, A_full, ctx.rank, target[0])
            # Local rows need nothing from the network, so this GEMM overlaps the
            # whole allgather.
            gemm(A2, B2, C2, ctx.rank * M_per_rank)
            main_s.wait_stream(comm_stream)
            for step in range(peers):
                peer = (ctx.rank + step + 1) % ctx.world_size
                gemm(A2, B2, C2, peer * M_per_rank)
        elif args.mode == "persistent":
            # Comm on its own stream so it stays resident next to the GEMM; the
            # GEMM's per-tile wait is what orders the data, not the stream.
            comm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(comm_stream):
                ag(A_shard, A_full, ctx.rank, target[0])
            gemm(A2, B2, C2, ctx.rank, target[0])
            torch.cuda.current_stream().wait_stream(comm_stream)
        else:
            ag(A_shard, A_full, ctx.rank, target[0])
            if args.gemm_impl == "tcgen05":
                gemm(A2, B2, C2, 0)          # whole M in one launch
            else:
                gemm(A2, B2, C2)

    torch.cuda.synchronize()
    dist.barrier(ctx.group)
    launch()
    torch.cuda.synchronize()

    # Reference: gather with torch, then matmul in fp32 to keep the tolerance
    # about our accumulation rather than the reference's.
    A_ref = torch.empty_like(A_full)
    dist.all_gather_into_tensor(A_ref, A_shard, group=ctx.group)
    C_ref = (A_ref.view(M, K).float() @ B2.float()).to(torch_dtype)
    failures = check(ctx, C2, C_ref, "ag_gemm")

    if not args.no_bench and failures == 0:
        dist.barrier(ctx.group)
        ref_ag = torch.empty_like(A_full)

        def run_torch():
            dist.all_gather_into_tensor(ref_ag, A_shard, group=ctx.group)
            torch.matmul(ref_ag.view(M, K), B2)

        tl_ms = do_bench(launch, warmup=args.warmup, rep=args.rep, group=ctx.group)
        ref_ms = do_bench(run_torch, warmup=args.warmup, rep=args.rep, group=ctx.group)
        # Split the phases so the serialization cost is visible.
        def ag_only():
            target[0] += per_launch
            ag(A_shard, A_full, ctx.rank, target[0])

        ag_ms = do_bench(ag_only, warmup=args.warmup, rep=args.rep, group=ctx.group)
        gemm_ms = float("nan")
        if args.mode == "serial":
            g = ((lambda: gemm(A2, B2, C2, 0)) if args.gemm_impl == "tcgen05"
                 else (lambda: gemm(A2, B2, C2)))
            gemm_ms = do_bench(g, warmup=args.warmup, rep=args.rep, group=ctx.group)
        flops = 2 * M * N * K
        if ctx.is_leader:
            print(
                f"ag_gemm[{args.mode}/{args.gemm_impl}]  tilescale {tl_ms:.3f} ms  {flops / (tl_ms * 1e-3) / 1e12:.1f} TFLOP/s"
                f"  (allgather alone {ag_ms:.3f}, gemm alone {gemm_ms:.3f})\n"
                f"         torch     {ref_ms:.3f} ms  {flops / (ref_ms * 1e-3) / 1e12:.1f} TFLOP/s"
                f"  | speedup {ref_ms / tl_ms:.2f}x",
                flush=True,
            )

    ctx.close()
    if ctx.is_leader:
        print("PASS" if failures == 0 else f"FAIL: {failures} rank(s) mismatched", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
