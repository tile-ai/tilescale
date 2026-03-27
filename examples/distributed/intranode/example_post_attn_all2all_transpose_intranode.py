"""
Intranode post-attention all-to-all (transpose) using tilescale IPC API.

Input:  [B, H_PE, S,   D] — partial heads, full sequence per rank
Output: [B, S_PE, NH,  D] — partial sequence, full heads per rank

Rank r sends src[:, :, p*S_PE:(p+1)*S_PE, :] (shape [B, H_PE, S_PE, D])
to rank p's dst[:, :, r*H_PE:(r+1)*H_PE, :] (shape [B, S_PE, H_PE, D])
after transposing dims 1 and 2.
"""
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing
import tilelang
import tilelang.language as T
from tilelang.distributed import init_dist, perf_fn

dtype_map = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def torch_reference(src, group, H_PE, S_PE):
    """dist.all_to_all reference implementation."""
    PE_num = dist.get_world_size(group)
    B, _, S, D = src.shape
    NH = H_PE * PE_num

    # send [B, H_PE, S_PE, D] to each rank
    input_list = [src[:, :, p * S_PE : (p + 1) * S_PE, :].contiguous() for p in range(PE_num)]
    output_list = [torch.empty(B, H_PE, S_PE, D, dtype=src.dtype, device=src.device) for _ in range(PE_num)]
    dist.all_to_all(output_list, input_list, group=group)

    result = torch.empty(B, S_PE, NH, D, dtype=src.dtype, device=src.device)
    for r in range(PE_num):
        # output_list[r] is [B, H_PE, S_PE, D] from rank r; transpose to [B, S_PE, H_PE, D]
        result[:, :, r * H_PE : (r + 1) * H_PE, :] = output_list[r].transpose(1, 2)
    return result


def kernel_post_attn_all2all_transpose(PE_num, B, NH, S_PE, D, dtype="float16"):
    H_PE = NH // PE_num
    S = S_PE * PE_num
    NUM_BLOCKS_X = B * S_PE

    @T.prim_func
    def main(
        data_src: T.Tensor((B, H_PE, S, D), dtype),
        data_dst: T.Tensor((B, S_PE, NH, D), dtype),
    ):
        with T.Kernel(NUM_BLOCKS_X, PE_num, threads=128) as (bx, target_pe):
            rank = T.alloc_local([1], "uint64")
            rank[0] = T.get_rank()

            batch_idx = bx // S_PE
            seq_idx = bx % S_PE
            src_seq_idx = target_pe * S_PE + seq_idx

            for head_idx in T.serial(H_PE):
                T.put_block(
                    src=T.address_of(data_src[batch_idx, head_idx, src_seq_idx, 0]),
                    dst=T.address_of(data_dst[batch_idx, seq_idx, rank[0] * H_PE + head_idx, 0]),
                    size=D,
                    dst_pe=target_pe,
                )

    return main


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    dtype = dtype_map[args.dtype]
    device = "cuda"
    B, NH, S, D = args.batch_size, args.num_heads, args.seq_len, args.head_dim
    PE_num = num_local_ranks
    assert S % PE_num == 0 and NH % PE_num == 0
    S_PE = S // PE_num
    H_PE = NH // PE_num

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    allocator = tilelang.get_allocator(
        size=2**30, device=device, is_distributed=True,
        local_rank=local_rank, num_local_ranks=num_local_ranks, group=group,
    )

    func = kernel_post_attn_all2all_transpose(PE_num, B, NH, S_PE, D, args.dtype)
    kernel = tilelang.compile(func)
    kernel.initialize(allocator=allocator)

    if local_rank == 0 and args.print_source:
        print(kernel.get_kernel_source())

    src_bufs = tilelang.tensor((B, H_PE, S, D), dtype, allocator=allocator, return_peers=True)
    dst_bufs = tilelang.tensor((B, S_PE, NH, D), dtype, allocator=allocator, return_peers=True)

    src_bufs[local_rank].normal_(mean=0.0, std=0.5)
    dst_bufs[local_rank].zero_()
    dist.barrier(group)

    torch_out = torch_reference(src_bufs[local_rank], group, H_PE, S_PE)
    dist.barrier(group)

    def ipc_all2all():
        kernel(src_bufs[local_rank], dst_bufs[local_rank])
        torch.cuda.synchronize()
        dist.barrier(group)

    ipc_all2all()

    result = dst_bufs[local_rank].clone()
    if torch.allclose(result, torch_out, atol=1e-3, rtol=1e-3):
        print(f"rank {local_rank} check passed. \u2705")
    else:
        diff = (result - torch_out).abs()
        print(f"rank {local_rank} check FAILED. max_diff={diff.max():.5f}")

    t = perf_fn(ipc_all2all, warmup=args.warmup, rep=args.repeat)
    print(f"rank {local_rank} avg time: {t:.3f} ms")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="float16", choices=list(dtype_map))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--print_source", action="store_true")
    args = parser.parse_args()

    torch.multiprocessing.spawn(main, args=(args.num_processes, args), nprocs=args.num_processes)
