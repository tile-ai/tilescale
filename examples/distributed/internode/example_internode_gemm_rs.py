"""Inter-node GEMM-ReduceScatter over NCCL GIN.

``C_local = reduce_scatter(A @ B)`` with ``K`` sharded across ranks: every rank
multiplies its own K-slice to get a full-size partial product, then the partials
are summed and scattered so each rank keeps one row-block. This is the shape that
follows a row-parallel matmul in TP inference -- the transpose of the AG-GEMM
case.

Why there is no overlapped mode yet
----------------------------------
The dependency runs the opposite way from AG-GEMM: the GEMM *produces* what the
network sends, so nothing can go out until some output exists. The obvious trick --
compute the peer's row block, start the reduce-scatter, then compute our own block
-- is **wrong**, and measurably so: it mismatched on 5.9M of 8.4M elements.

The reason is that reduce_scatter_kernel is monolithic. It does not only send the
peer's block; it also copies *this* rank's block into scratch as its own
contribution to the sum. Launching it before that block is computed makes it
reduce stale data. Reordering the launches cannot fix this, because the kernel
needs every block ready before it starts.

Real overlap needs the reduce-scatter split into two pieces: a per-peer "put my
block for peer p" that can fire as soon as that block is computed, and a separate
"accumulate what arrived" that runs after the local block is done. That is a
kernel-level decomposition, not a scheduling change, and is the next step here.

The reduce-scatter kernel is the verified one from
``example_internode_reduce_scatter.py``, reused unchanged: the partial product is
laid out exactly as its input expects (rank-major row blocks).
"""

# NOTE: no `from __future__ import annotations` here -- see the allgather example.
import argparse

import torch
import torch.distributed as dist

from tilelang.distributed.bench import do_bench

from example_internode_ag_gemm import gemm_range_kernel
from internode_gemm_sm100 import tcgen05_gemm_range_kernel
from example_internode_reduce_scatter import reduce_scatter_kernel
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


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--m-per-rank", type=int, default=2048)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k-per-rank", type=int, default=4096)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--gemm-threads", type=int, default=256)
    parser.add_argument("--gemm-impl", choices=("tcgen05", "naive"), default="tcgen05")
    parser.add_argument("--gemm-block-n", type=int, default=256)
    # serial only. See the module docstring: the naive split is incorrect.
    parser.add_argument("--mode", choices=("serial",), default="serial")
    args = parser.parse_args()

    prepare_env()
    ctx = Context(arena_bytes=1 << 31)

    peers = ctx.world_size - 1
    M_per_rank, N, K_per_rank = args.m_per_rank, args.n, args.k_per_rank
    M = M_per_rank * ctx.world_size
    shard_numel = M_per_rank * N          # what each rank keeps after the scatter
    if shard_numel % args.chunks:
        raise SystemExit(f"M_per_rank*N = {shard_numel} must be a multiple of --chunks")

    torch_dtype = TORCH_DTYPES[args.dtype]
    tl_dtype = TL_DTYPES[args.dtype]
    ctx.log(
        f"gemm_rs[{args.mode}]: world={ctx.world_size} nodes={ctx.num_nodes} M={M} "
        f"(per rank {M_per_rank}) N={N} K_per_rank={K_per_rank} chunks={args.chunks} "
        f"contexts={args.gin_contexts} dtype={args.dtype}"
    )

    if args.gemm_impl == "tcgen05":
        gemm = ctx.compile(
            tcgen05_gemm_range_kernel(M, N, K_per_rank, M_per_rank, block_M=args.block_m,
                                      block_N=args.gemm_block_n, block_K=args.block_k)
        )
    else:
        gemm = ctx.compile(
            gemm_range_kernel(M, N, K_per_rank, M_per_rank, args.block_m, args.block_n,
                              args.block_k, args.gemm_threads, tl_dtype)
        )
    rs = ctx.compile(
        reduce_scatter_kernel(shard_numel, args.chunks, args.threads, ctx.world_size, tl_dtype,
                              signal_id=SIGNAL_DATA),
        expect=("tl::gin::put_signal_addr", "tl::gin::wait_signal"),
        gin_contexts=args.gin_contexts,
    )

    A = ctx.tensor((M * K_per_rank,), torch_dtype)
    B = ctx.tensor((K_per_rank * N,), torch_dtype)
    partial = ctx.tensor((M * N,), torch_dtype)       # GIN source
    scratch = ctx.tensor((M * N,), torch_dtype)       # GIN destination
    out = ctx.tensor((shard_numel,), torch_dtype)

    gen = torch.Generator(device="cuda").manual_seed(4321 + ctx.rank)
    A.copy_(torch.randn(M * K_per_rank, generator=gen, device="cuda", dtype=torch.float32) * 0.05)
    B.copy_(torch.randn(K_per_rank * N, generator=gen, device="cuda", dtype=torch.float32) * 0.05)
    partial.zero_()
    scratch.zero_()
    out.zero_()

    A2, B2, P2 = A.view(M, K_per_rank), B.view(K_per_rank, N), partial.view(M, N)
    per_launch = per_launch_signals(peers, args.chunks, args.signal_div)
    target = [0]

    def launch():
        target[0] += per_launch
        # Every row block must exist before the reduce-scatter starts; see the
        # module docstring for why this cannot simply be reordered for overlap.
        for r in range(ctx.world_size):
            gemm(A2, B2, P2, r * M_per_rank)
        rs(partial, scratch, out, ctx.rank, target[0])

    torch.cuda.synchronize()
    dist.barrier(ctx.group)
    launch()
    torch.cuda.synchronize()

    # Reference in fp32 so the tolerance reflects our accumulation, not torch's.
    P_ref = (A2.float() @ B2.float())
    C_ref = torch.empty(shard_numel, device=out.device, dtype=torch.float32)
    dist.reduce_scatter_tensor(C_ref, P_ref.reshape(-1).contiguous(), op=dist.ReduceOp.SUM,
                               group=ctx.group)
    failures = check(ctx, out, C_ref.to(torch_dtype), "gemm_rs")

    if not args.no_bench and failures == 0:
        dist.barrier(ctx.group)
        ref_p = torch.empty(M * N, device=out.device, dtype=torch_dtype)
        ref_o = torch.empty(shard_numel, device=out.device, dtype=torch_dtype)

        def run_torch():
            torch.matmul(A2, B2, out=ref_p.view(M, N))
            dist.reduce_scatter_tensor(ref_o, ref_p, op=dist.ReduceOp.SUM, group=ctx.group)

        tl_ms = do_bench(launch, warmup=args.warmup, rep=args.rep, group=ctx.group)
        ref_ms = do_bench(run_torch, warmup=args.warmup, rep=args.rep, group=ctx.group)
        flops = 2 * M * N * K_per_rank
        if ctx.is_leader:
            print(
                f"gemm_rs[{args.gemm_impl}]  tilescale {tl_ms:.3f} ms  "
                f"{flops / (tl_ms * 1e-3) / 1e12:.1f} TFLOP/s\n"
                f"          torch     {ref_ms:.3f} ms  {flops / (ref_ms * 1e-3) / 1e12:.1f} "
                f"TFLOP/s  | speedup {ref_ms / tl_ms:.2f}x",
                flush=True,
            )

    ctx.close()
    if ctx.is_leader:
        print("PASS" if failures == 0 else f"FAIL: {failures} rank(s) mismatched", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
