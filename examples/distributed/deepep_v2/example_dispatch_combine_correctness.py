"""Standalone runnable correctness check for the DeepEP-EPv2-aligned dispatch/combine port.

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MASTER_PORT=30071 \\
    python examples/distributed/deepep_v2/example_dispatch_combine_correctness.py

For DeepEP's own headline shape:

    python examples/distributed/deepep_v2/example_dispatch_combine_correctness.py \\
        --tokens 8192 --hidden 7168 --topk 8 --experts 256 --num-processes 8 --num-sms 64
"""

import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing

from tilelang.distributed.host import init_dist

from buffer import Buffer
import reference


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    torch.manual_seed(1234 + rank)
    device = f"cuda:{local_rank}"
    x = torch.randn(args.tokens, args.hidden, dtype=torch.bfloat16, device=device)
    topk_idx, topk_weights = reference.make_topk(args.tokens, args.topk, args.experts, device, args.masked_ratio)

    buf = Buffer(
        group=group,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        num_max_tokens_per_rank=args.tokens,
        hidden=args.hidden,
        num_topk=args.topk,
        num_experts=args.experts,
        dtype=torch.bfloat16,
        num_sms=args.num_sms,
        dispatch_threads=args.dispatch_threads,
        combine_threads=args.combine_threads,
    )

    recv_x, recv_topk_idx, recv_topk_weights, handle, _ = buf.dispatch(x, topk_idx, topk_weights)
    # `dispatch` returns the full receive capacity; the reference only wants
    # the rows that were actually written. Reading the count synchronises.
    n = handle.num_recv_tokens
    recv_x, recv_topk_idx, recv_topk_weights = recv_x[:n], recv_topk_idx[:n], recv_topk_weights[:n]
    if rank == 0:
        print(f"num_recv_tokens={handle.num_recv_tokens} total_capacity={buf.total_capacity}")

    expert_out = reference.simulate_expert_compute(recv_x, recv_topk_idx, recv_topk_weights)
    combined, _ = buf.combine(expert_out, handle)
    expected = reference.reference_combined(x, topk_weights, topk_idx)
    err = (combined.float() - expected.float()).norm().item()
    denom = expected.float().norm().item()
    # `denom` is zero only when every selection was masked off.
    rel_l2 = err / denom if denom > 0 else err
    passed = rel_l2 < 0.05
    print(f"rank {rank}: rel_l2_error={rel_l2:.6f} passed={passed}")
    assert passed, f"rank {rank}: mismatch, rel_l2_error={rel_l2}"

    buf.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    # Fraction of top-k selections marked unselected (-1), DeepEP's marker.
    parser.add_argument("--masked-ratio", type=float, default=0.0)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--num-sms", type=int, default=64)
    parser.add_argument("--dispatch-threads", type=int, default=512)
    parser.add_argument("--combine-threads", type=int, default=256)
    args = parser.parse_args()
    torch.multiprocessing.spawn(main, args=(args.num_processes, args), nprocs=args.num_processes, join=True)
