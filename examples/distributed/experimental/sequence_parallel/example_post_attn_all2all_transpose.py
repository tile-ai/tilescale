"""Intranode head-to-sequence all-to-all with a fused transpose.

Input:  [B, H_PE, S, D]   - partial heads, full sequence per rank
Output: [B, S_PE, NH, D]  - partial sequence, full heads per rank
"""

import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing

import tilelang
import tilelang.language as T
from tilelang.distributed.allocator import get_allocator
from tilelang.distributed.bench import do_bench
from tilelang.distributed.host import init_dist


_TORCH_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
_TL_DTYPES = {
    "bf16": T.bfloat16,
    "fp16": T.float16,
    "fp32": T.float32,
}


def torch_reference(src: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    num_ranks = dist.get_world_size(group)
    batch, heads_per_rank, seq_len, head_dim = src.shape
    seq_per_rank = seq_len // num_ranks

    send = [src[:, :, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous() for rank in range(num_ranks)]
    recv = [torch.empty(batch, heads_per_rank, seq_per_rank, head_dim, dtype=src.dtype, device=src.device) for _ in range(num_ranks)]
    dist.all_to_all(recv, send, group=group)
    return torch.cat([part.transpose(1, 2) for part in recv], dim=2)


@tilelang.jit(compile_once=True)
def post_attn_all2all_transpose_kernel(num_ranks, batch, num_heads, seq_per_rank, head_dim, dtype=T.float16):
    heads_per_rank = num_heads // num_ranks
    seq_len = seq_per_rank * num_ranks

    @T.prim_func
    def main(
        src: T.Tensor((batch, heads_per_rank, seq_len, head_dim), dtype),
        dst: T.Tensor((batch, seq_per_rank, num_heads, head_dim), dtype),
    ):
        with T.Kernel(batch * seq_per_rank, num_ranks, threads=128) as (bx, dst_rank):
            rank = T.get_rank()
            batch_idx = bx // seq_per_rank
            seq_idx = bx % seq_per_rank
            src_seq_idx = dst_rank * seq_per_rank + seq_idx

            for head_idx in T.serial(heads_per_rank):
                T.put_block(
                    src=T.address_of(src[batch_idx, head_idx, src_seq_idx, 0]),
                    dst=T.address_of(dst[batch_idx, seq_idx, rank * heads_per_rank + head_idx, 0]),
                    size=head_dim,
                    dst_pe=dst_rank,
                )

    return main


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    assert args.seq_len % num_local_ranks == 0, "seq-len must be divisible by num-processes"
    assert args.num_heads % num_local_ranks == 0, "num-heads must be divisible by num-processes"

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert rank == local_rank and num_ranks == num_local_ranks, "only support single-node launch for now"

    torch_dtype = _TORCH_DTYPES[args.dtype]
    tl_dtype = _TL_DTYPES[args.dtype]
    seq_per_rank = args.seq_len // num_ranks
    heads_per_rank = args.num_heads // num_ranks
    numel = args.batch_size * (heads_per_rank * args.seq_len + seq_per_rank * args.num_heads) * args.head_dim
    allocator = get_allocator(
        size=max(numel * torch.empty((), dtype=torch_dtype).element_size() + 4096, 2**22),
        device=f"cuda:{local_rank}",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group,
    )

    kernel = post_attn_all2all_transpose_kernel(
        num_ranks,
        args.batch_size,
        args.num_heads,
        seq_per_rank,
        args.head_dim,
        tl_dtype,
    )
    kernel.compile_group = group
    kernel.initialize(allocator=allocator)
    if rank == 0 and args.print_source:
        print(kernel.get_kernel_source())

    src_peers = tilelang.tensor(
        (args.batch_size, heads_per_rank, args.seq_len, args.head_dim),
        torch_dtype,
        allocator=allocator,
        return_peers=True,
    )
    dst_peers = tilelang.tensor(
        (args.batch_size, seq_per_rank, args.num_heads, args.head_dim),
        torch_dtype,
        allocator=allocator,
        return_peers=True,
    )
    src = src_peers[rank]
    dst = dst_peers[rank]
    src.normal_(mean=0.0, std=0.5)
    dst.zero_()
    dist.barrier(group)

    expected = torch_reference(src, group)
    dist.barrier(group)
    kernel(src, dst)
    torch.cuda.synchronize()
    dist.barrier(group)
    torch.testing.assert_close(dst, expected, atol=args.atol, rtol=args.rtol)
    print(f"rank {rank} check passed")

    latency_ms = do_bench(
        lambda: kernel(src, dst),
        warmup=args.warmup,
        rep=args.rep,
        group=group,
    )
    if rank == 0:
        print(f"post-attention transpose all-to-all time: {latency_ms * 1000:.2f} us")

    allocator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=tuple(_TORCH_DTYPES), default="fp16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--print-source", action="store_true")
    args = parser.parse_args()

    torch.multiprocessing.spawn(main, args=(args.num_processes, args), nprocs=args.num_processes, join=True)
