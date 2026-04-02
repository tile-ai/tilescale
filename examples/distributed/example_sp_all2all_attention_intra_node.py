from __future__ import annotations

import argparse
from itertools import accumulate

import tilelang
import torch
import torch.distributed as dist
import torch.multiprocessing

from tilelang.distributed import init_dist, perf_fn
from sp_all2all_attention_intra_node import (
    create_sp_all2all_attention_context_intra_node,
    fused_sp_all2all_attn_intra_node,
)


def torch_pre_attn_qkv_a2a_reference(group, q_input, k_input, v_input, skip_q_a2a=False):
    world_size = dist.get_world_size(group)

    def _a2a(data_src):
        a2a_input = data_src.permute(2, 1, 0, 3).contiguous()
        a2a_heads, a2a_seq_per_pe, a2a_batch, a2a_head_dim = a2a_input.shape
        assert a2a_heads % world_size == 0

        a2a_output = torch.empty(
            (world_size, a2a_heads // world_size, a2a_seq_per_pe, a2a_batch, a2a_head_dim),
            dtype=a2a_input.dtype,
            device=a2a_input.device,
            requires_grad=False,
        )
        dist.all_to_all_single(a2a_output, a2a_input, group=group)
        return (
            a2a_output.permute(3, 0, 2, 1, 4)
            .reshape(a2a_batch, a2a_seq_per_pe * world_size, a2a_heads // world_size, a2a_head_dim)
            .contiguous()
        )

    q_output = None if skip_q_a2a else _a2a(q_input)
    k_output = _a2a(k_input)
    v_output = _a2a(v_input)
    return q_output, k_output, v_output


def torch_attention_reference(q_out, k_out, v_out, is_causal, q_start_offsets=None):
    # q_out: [B, S, Hq_local, D], k/v_out: [B, S, Hkv_local, D]
    batch_size, seq_len, q_heads, head_dim = q_out.shape
    kv_heads = k_out.shape[2]
    assert q_heads % kv_heads == 0
    groups = q_heads // kv_heads
    out_list = []
    for b in range(batch_size):
        q_b = q_out[b].permute(1, 0, 2).unsqueeze(0).contiguous()  # [1, Hq, S, D]
        k_b = k_out[b].permute(1, 0, 2).unsqueeze(0).contiguous()  # [1, Hkv, S, D]
        v_b = v_out[b].permute(1, 0, 2).unsqueeze(0).contiguous()  # [1, Hkv, S, D]

        k_b = k_b.repeat_interleave(groups, dim=1)
        v_b = v_b.repeat_interleave(groups, dim=1)

        attn_mask = None
        if is_causal:
            q_positions = torch.arange(seq_len, device=q_out.device)[:, None]
            k_positions = torch.arange(k_out.shape[1], device=q_out.device)[None, :]
            attn_mask = k_positions <= q_positions

        out_b = torch.nn.functional.scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=attn_mask)
        out_b = out_b.squeeze(0).permute(1, 0, 2).contiguous()  # [S, Hq, D]
        out_list.append(out_b)

    return torch.cat(out_list, dim=0)  # [B*S, Hq, D]


def pack_local_qkv_for_all2all(
    q_input,
    k_input,
    v_input,
    packed_local,
    local_world_size,
    q_heads_per_rank,
    kv_heads_per_rank,
):
    # q_input: [B, S_local, Hq_global, D], k/v_input: [B, S_local, Hkv_global, D]
    batch_size, seq_per_pe, _, head_dim = q_input.shape
    packed_heads_per_rank = q_heads_per_rank + 2 * kv_heads_per_rank
    packed_total_heads = packed_heads_per_rank * local_world_size

    packed_view = packed_local.view(batch_size, seq_per_pe, packed_total_heads, head_dim)
    packed_view.fill_(0)

    for dst_rank in range(local_world_size):
        base = dst_rank * packed_heads_per_rank
        q_slice = q_input[:, :, dst_rank * q_heads_per_rank : (dst_rank + 1) * q_heads_per_rank, :]
        k_slice = k_input[:, :, dst_rank * kv_heads_per_rank : (dst_rank + 1) * kv_heads_per_rank, :]
        v_slice = v_input[:, :, dst_rank * kv_heads_per_rank : (dst_rank + 1) * kv_heads_per_rank, :]

        packed_view[:, :, base : base + q_heads_per_rank, :].copy_(q_slice)
        packed_view[:, :, base + q_heads_per_rank : base + q_heads_per_rank + kv_heads_per_rank, :].copy_(k_slice)
        packed_view[:, :, base + q_heads_per_rank + kv_heads_per_rank : base + q_heads_per_rank + 2 * kv_heads_per_rank, :].copy_(v_slice)


class FusedSequenceParallelAll2AllAttn(torch.nn.Module):
    def __init__(
        self,
        pg: torch.distributed.ProcessGroup,
        batch_size: int,
        q_head: int,
        kv_head: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        head_dim: int,
        input_dtype=torch.float16,
        output_dtype=torch.float16,
        device="cuda",
        is_causal=True,
        enable_zig_zag=True,
        allocator=None,
    ):
        super(FusedSequenceParallelAll2AllAttn, self).__init__()
        self.pg = pg
        self.rank = pg.rank()
        self.world_size = pg.size()

        self.batch_size = batch_size
        self.q_head = q_head
        self.kv_head = kv_head
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k
        self.head_dim = head_dim
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.device = device
        self.is_causal = is_causal
        self.enable_zig_zag = enable_zig_zag
        self.allocator = allocator

        assert self.q_head % self.world_size == 0, "q_head should be divisible by world_size"
        assert self.kv_head % self.world_size == 0, "kv_head should be divisible by world_size"
        self.q_head_per_rank = self.q_head // self.world_size
        self.kv_head_per_rank = self.kv_head // self.world_size
        self.max_q_shard_len = self.max_seqlen_q // self.world_size

        self.ctx = create_sp_all2all_attention_context_intra_node(
            self.batch_size,
            self.q_head_per_rank,
            self.kv_head_per_rank,
            self.max_seqlen_k,
            self.max_q_shard_len,
            self.head_dim,
            self.input_dtype,
            self.output_dtype,
            self.rank,
            self.world_size,
            self.device,
            self.allocator,
        )

    def forward(self, packed_qkv_shards, cu_seqlens_q, cu_seqlens_k, print_source=False):
        output_buffer = self.ctx.attn_output_buffer

        fused_sp_all2all_attn_intra_node(
            self.ctx,
            packed_qkv_shards,
            output_buffer,
            cu_seqlens_q,
            cu_seqlens_k,
            self.max_q_shard_len,
            self.rank,
            self.world_size,
            self.q_head_per_rank,
            self.kv_head_per_rank,
            self.is_causal,
            self.enable_zig_zag,
            print_source,
        )

        return output_buffer


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    dtype = torch.float16
    device = "cuda"

    batch_size = args.batch_size
    q_head = args.q_head
    kv_head = args.kv_head
    max_seqlen_q = args.max_seqlen_q
    max_seqlen_k = args.max_seqlen_k
    head_dim = args.head_dim
    is_causal = args.is_causal
    enable_zig_zag = args.zig_zag

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    try:
        assert rank == local_rank and num_ranks == num_local_ranks, "only support single node for now"

        seqlens_q = args.seqlens_q
        seqlens_k = args.seqlens_k
        assert len(seqlens_q) == batch_size and len(seqlens_k) == batch_size
        assert q_head % num_ranks == 0, "q_head should be divisible by world size"
        assert kv_head % num_ranks == 0, "kv_head should be divisible by world size"
        for s in seqlens_q + seqlens_k:
            assert s % num_ranks == 0, "all2all requires per-batch sequence length divisible by world size"

        cu_seqlens_q_list = [0] + list(accumulate(seqlens_q))
        cu_seqlens_k_list = [0] + list(accumulate(seqlens_k))
        cu_seqlens_q = torch.tensor(cu_seqlens_q_list, dtype=torch.int32, device=device) // num_ranks
        cu_seqlens_k = torch.tensor(cu_seqlens_k_list, dtype=torch.int32, device=device)

        allocator = tilelang.get_allocator(
            size=2**30,
            device=device,
            is_distributed=True,
            local_rank=local_rank,
            num_local_ranks=num_local_ranks,
            group=group,
        )

        seq_per_rank_max = max_seqlen_k // num_ranks
        q_head_per_rank = q_head // num_ranks
        kv_head_per_rank = kv_head // num_ranks
        packed_heads_per_rank = q_head_per_rank + 2 * kv_head_per_rank
        packed_total_heads = packed_heads_per_rank * num_ranks

        packed_qkv_shards = tilelang.tensor(
            (batch_size * seq_per_rank_max, packed_total_heads, head_dim),
            dtype=dtype,
            allocator=allocator,
            return_peers=True,
        )

        q_input = torch.randn((batch_size, seq_per_rank_max, q_head, head_dim), dtype=dtype, device=device)
        k_input = torch.randn((batch_size, seq_per_rank_max, kv_head, head_dim), dtype=dtype, device=device)
        v_input = torch.randn((batch_size, seq_per_rank_max, kv_head, head_dim), dtype=dtype, device=device)
        local_q_batch_lens = [s // num_ranks for s in seqlens_q]
        local_q_start_offsets = [rank * local_len for local_len in local_q_batch_lens]

        pack_local_qkv_for_all2all(
            q_input,
            k_input,
            v_input,
            packed_qkv_shards[local_rank],
            num_ranks,
            q_head_per_rank,
            kv_head_per_rank,
        )

        dist.barrier(group)

        tilescale_module = FusedSequenceParallelAll2AllAttn(
            group,
            batch_size,
            q_head,
            kv_head,
            max_seqlen_q,
            max_seqlen_k,
            head_dim,
            dtype,
            dtype,
            device,
            is_causal,
            enable_zig_zag,
            allocator=allocator,
        )

        tilescale_out = tilescale_module(packed_qkv_shards, cu_seqlens_q, cu_seqlens_k, print_source=args.print_source)
        # valid_q_tokens = int(cu_seqlens_q[-1].item())

        torch_q_out, torch_k_out, torch_v_out = torch_pre_attn_qkv_a2a_reference(group, q_input, k_input, v_input)
        torch_out = torch_attention_reference(torch_q_out, torch_k_out, torch_v_out, is_causal, local_q_start_offsets)

        torch_out_local = []
        tilescale_out_local = []
        torch_cursor = 0
        tilescale_cursor = 0
        for local_len in local_q_batch_lens:
            torch_start = torch_cursor + rank * local_len
            torch_end = torch_start + local_len
            torch_out_local.append(torch_out[torch_start:torch_end])

            tilescale_start = tilescale_cursor
            tilescale_end = tilescale_start + local_len
            tilescale_out_local.append(tilescale_out[tilescale_start:tilescale_end])

            torch_cursor += local_len * num_ranks
            tilescale_cursor += local_len

        torch_out = torch.cat(torch_out_local, dim=0)
        tilescale_out = torch.cat(tilescale_out_local, dim=0)

        atol = 1e-2
        rtol = 1e-2
        if torch.allclose(torch_out, tilescale_out, atol=atol, rtol=rtol):
            print(f"rank {local_rank} check passed.✅")
        else:
            diff = torch.abs(torch_out - tilescale_out)
            print(f"rank {local_rank} check failed.❌ max_diff={torch.max(diff).item():.6f}")
            print(f"torch_out: {torch_out.shape}, tilelang_out: {tilescale_out.shape}")

        tl_t = perf_fn(lambda: tilescale_module(packed_qkv_shards, cu_seqlens_q, cu_seqlens_k), warmup=5, rep=5)
        # if isinstance(tl_t, (tuple, list)):
        #     tl_t = tl_t[0]
        # if isinstance(tl_t, torch.Tensor):
        #     tl_t = tl_t.detach().float()
        #     tl_t = tl_t.mean().item() if tl_t.numel() > 1 else tl_t.item()
        # else:
        #     tl_t = float(tl_t)
        print(f"rank {local_rank} tilescale time: {tl_t:.2f} ms")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=2, help="Number of processes to spawn (default: 2)")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--q_head", type=int, default=32, help="local num q heads per rank")
    parser.add_argument("--kv_head", type=int, default=8, help="local num kv heads per rank")
    parser.add_argument("--max_seqlen_q", type=int, default=4096, help="max sequence length of q")
    parser.add_argument("--max_seqlen_k", type=int, default=4096, help="max sequence length of k/v")
    parser.add_argument("--head_dim", type=int, default=128, help="head dim")
    parser.add_argument("--seqlens_q", type=int, nargs="+", default=[4096, 4096], help="sequence lengths of q")
    parser.add_argument("--seqlens_k", type=int, nargs="+", default=[4096, 4096], help="sequence lengths of k/v")
    parser.add_argument("--is_causal", action="store_true", help="causal")
    parser.add_argument(
        "--zig-zag",
        "--no-zig-zag",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable zig zag opt",
    )
    parser.add_argument("--print_source", action="store_true", help="print kernel source")

    args = parser.parse_args()
    num_processes = args.num_processes

    torch.multiprocessing.spawn(main, args=(num_processes, args), nprocs=num_processes)
