import torch
import torch.distributed as dist
import torch.multiprocessing
import tilelang
from tilelang.distributed import init_dist, dtype_map
import argparse
import os

tilelang.disable_cache()


def torch_pre_attn_qkv_a2a_reference(group, q_input, k_input, v_input, skip_q_a2a=False, debug=False, tensor_name="", rank=0):
    """
    PyTorch golden reference matching pre_attn_qkv_pack_a2a semantics.

    q_input shape: [BATCH_SIZE, SEQ_PER_PE, Q_NUM_HEADS, HEAD_DIM]
    k_input shape: [BATCH_SIZE, SEQ_PER_PE, KV_NUM_HEADS, HEAD_DIM]
    v_input shape: [BATCH_SIZE, SEQ_PER_PE, KV_NUM_HEADS, HEAD_DIM]

    q_output shape: [BATCH_SIZE, SEQ_LEN, Q_NUM_HEADS_PER_PE, HEAD_DIM] (or None if skip)
    k_output shape: [BATCH_SIZE, SEQ_LEN, KV_NUM_HEADS_PER_PE, HEAD_DIM]
    v_output shape: [BATCH_SIZE, SEQ_LEN, KV_NUM_HEADS_PER_PE, HEAD_DIM]

    Args:
        group: Distributed process group
        q_input: Q input on current PE
        k_input: K input on current PE
        v_input: V input on current PE
        skip_q_a2a: Whether to skip q all-to-all
        debug: Whether to print tensor shapes and values
        tensor_name: Name of tensor for debug output
        rank: Current rank for debug output

    Returns:
        [q_output, k_output, v_output]
    """
    world_size = dist.get_world_size(group)

    def _a2a(data_src, name=""):
        # Match target path: [B, S_local, H, D] -> [H, S_local, B, D] -> all_to_all_single -> [B, S_global, H_local, D]
        if debug and rank == 0:
            print(f"\n=== {tensor_name} {name} ===")
            print(f"Input shape: {data_src.shape}")
            print(f"Input tensor:\n{data_src}")

        a2a_input = data_src.permute(2, 1, 0, 3).contiguous()
        a2a_heads, a2a_seq_per_pe, a2a_batch, a2a_head_dim = a2a_input.shape

        if debug and rank == 0:
            print(f"After permute [2,1,0,3]: {a2a_input.shape}")

        if a2a_heads < world_size:
            assert world_size % a2a_heads == 0
            repeats = world_size // a2a_heads
            a2a_input = torch.repeat_interleave(a2a_input, repeats=repeats, dim=0).contiguous()
            a2a_heads, a2a_seq_per_pe, a2a_batch, a2a_head_dim = a2a_input.shape
            if debug and rank == 0:
                print(f"After repeat_interleave (repeats={repeats}): {a2a_input.shape}")

        assert a2a_heads % world_size == 0
        a2a_output = torch.empty(
            (world_size, a2a_heads // world_size, a2a_seq_per_pe, a2a_batch, a2a_head_dim),
            dtype=a2a_input.dtype,
            device=a2a_input.device,
            requires_grad=False,
        )

        if debug and rank == 0:
            print(f"Before all_to_all_single - send shape: {a2a_input.shape}, recv shape: {a2a_output.shape}")

        dist.all_to_all_single(a2a_output, a2a_input, group=group)

        if debug and rank == 0:
            print(f"After all_to_all_single: {a2a_output.shape}")

        result = (
            a2a_output.permute(3, 0, 2, 1, 4)
            .reshape(a2a_batch, a2a_seq_per_pe * world_size, a2a_heads // world_size, a2a_head_dim)
            .contiguous()
        )

        if debug and rank == 0:
            print(f"After final reshape: {result.shape}")
            print(f"Output tensor:\n{result}")

        return result

    q_output = None if skip_q_a2a else _a2a(q_input, "(Q)")
    k_output = _a2a(k_input, "(K)")
    v_output = _a2a(v_input, "(V)")
    return [q_output, k_output, v_output]


def custom_ipc_all_to_all(
    data_src_peers,
    data_dst_peers,
    local_rank,
    local_world_size,
    HEADS_PER_PE,
    SEQ_PER_PE,
    stream
):
    """
    P2P IPC-based all-to-all dimension swap matching the PyTorch golden reference.
    Executes asynchronous memory copies pulling from remote buffers to local buffer.
    """
    rank_orders = [(local_rank + i) % local_world_size for i in range(local_world_size)]
    
    with torch.cuda.stream(stream):
        for src_rank in rank_orders:
            # PULL mechanism:
            # We want the heads belonging to our local_rank from the src_rank's data_src.
            # data_src shape on src_rank: [BATCH_SIZE, SEQ_PER_PE, NUM_HEADS, HEAD_DIM]
            src = data_src_peers[src_rank][
                :,
                :,
                local_rank * HEADS_PER_PE : (local_rank + 1) * HEADS_PER_PE,
                :
            ]
            
            # We place this data into our local data_dst at the sequence position corresponding to src_rank.
            # data_dst shape on local_rank: [BATCH_SIZE, SEQ_LEN, HEADS_PER_PE, HEAD_DIM]
            dst = data_dst_peers[local_rank][
                :,
                src_rank * SEQ_PER_PE : (src_rank + 1) * SEQ_PER_PE,
                :,
                :
            ]
            
            # Execute P2P copy
            dst.copy_(src)


def custom_ipc_pre_attn_qkv_a2a(
    q_src_peers,
    k_src_peers,
    v_src_peers,
    q_dst_peers,
    k_dst_peers,
    v_dst_peers,
    local_rank,
    local_world_size,
    q_heads_per_pe,
    kv_heads_per_pe,
    seq_per_pe,
    stream,
    skip_q_a2a=False,
):
    if not skip_q_a2a:
        custom_ipc_all_to_all(
            q_src_peers,
            q_dst_peers,
            local_rank,
            local_world_size,
            q_heads_per_pe,
            seq_per_pe,
            stream,
        )

    custom_ipc_all_to_all(
        k_src_peers,
        k_dst_peers,
        local_rank,
        local_world_size,
        kv_heads_per_pe,
        seq_per_pe,
        stream,
    )
    custom_ipc_all_to_all(
        v_src_peers,
        v_dst_peers,
        local_rank,
        local_world_size,
        kv_heads_per_pe,
        seq_per_pe,
        stream,
    )


def verify_results(custom_output, torch_output, rank, tensor_name="", tolerance=1e-3):
    """Verify output against PyTorch golden reference. Returns True if passed, False if failed."""
    if not torch.allclose(custom_output, torch_output, atol=tolerance, rtol=tolerance):
        print(f"❌ PE {rank} {tensor_name} Verification FAILED!")

        diff = torch.abs(custom_output - torch_output)
        max_diff = torch.max(diff)
        mean_diff = torch.mean(diff)

        print(f"   Max difference: {max_diff:.6f}")
        print(f"   Mean difference: {mean_diff:.6f}")
        print(f"   TileLang shape: {custom_output.shape}")
        print(f"   PyTorch shape:  {torch_output.shape}")

        # Find position with maximum difference
        max_pos = torch.unravel_index(torch.argmax(diff), diff.shape)
        print(f"   Max diff position: {max_pos}")
        print(f"   TileLang value: {custom_output[max_pos]:.6f}")
        print(f"   PyTorch value:  {torch_output[max_pos]:.6f}")

        return False
    else: 
        return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=None, help="Number of GPUs to spawn")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--gqa", type=int, default=1, help="group size of group query attn")
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--skip_q_a2a", default=False, action="store_true", help="skip q all-to-all")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--debug", default=False, action="store_true", help="print debug info with sequential integer inputs")
    return parser.parse_args()


def run_all_to_all_with_golden_reference(args, WORLD_SIZE, RANK, LOCAL_RANK, TP_GROUP, LC_GROUP):
    try:
        PE_num = WORLD_SIZE
        assert args.seq_len % PE_num == 0
        assert args.num_heads % PE_num == 0
        assert args.gqa > 0
        assert args.num_heads % args.gqa == 0

        SEQ_PER_PE = args.seq_len // PE_num
        Q_HEADS_PER_PE = args.num_heads // PE_num
        kv_num_heads = args.num_heads // args.gqa
        assert kv_num_heads % PE_num == 0
        KV_HEADS_PER_PE = kv_num_heads // PE_num
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", WORLD_SIZE))
        assert PE_num == local_world_size, "IPC mode in this example only supports single-node runs"

        allocator = tilelang.get_allocator(
            size=2**30,
            device="cuda",
            is_distributed=True,
            local_rank=LOCAL_RANK,
            num_local_ranks=local_world_size,
            group=LC_GROUP,
        )

        if RANK == 0:
            print("=== All-to-All with IPC and PyTorch Golden Reference ===")
            print(f"Batch size: {args.batch_size}")
            print(f"Sequence length: {args.seq_len}")
            print(f"Number of heads: {args.num_heads}")
            print(f"KV heads: {kv_num_heads}")
            print(f"GQA group size: {args.gqa}")
            print(f"Head dimension: {args.head_dim}")
            print(f"PE count: {PE_num}")
            print(f"Sequence per PE: {SEQ_PER_PE}")
            print(f"Q heads per PE: {Q_HEADS_PER_PE}")
            print(f"KV heads per PE: {KV_HEADS_PER_PE}")
            print(f"Skip q all-to-all: {args.skip_q_a2a}")

        dtype_torch = dtype_map[args.dtype]
        
        # PyTorch reference inputs
        if args.debug:
            # Use sequential integers for easy visualization
            q_size = args.batch_size * SEQ_PER_PE * args.num_heads * args.head_dim
            q_input = torch.arange(q_size, dtype=dtype_torch, device="cuda").reshape([args.batch_size, SEQ_PER_PE, args.num_heads, args.head_dim])
            
            k_size = args.batch_size * SEQ_PER_PE * kv_num_heads * args.head_dim
            k_input = torch.arange(k_size, dtype=dtype_torch, device="cuda").reshape([args.batch_size, SEQ_PER_PE, kv_num_heads, args.head_dim])
            v_input = torch.arange(k_size, dtype=dtype_torch, device="cuda").reshape([args.batch_size, SEQ_PER_PE, kv_num_heads, args.head_dim])
        else:
            q_input = torch.rand([args.batch_size, SEQ_PER_PE, args.num_heads, args.head_dim], dtype=dtype_torch, device="cuda")
            k_input = torch.rand([args.batch_size, SEQ_PER_PE, kv_num_heads, args.head_dim], dtype=dtype_torch, device="cuda")
            v_input = torch.rand([args.batch_size, SEQ_PER_PE, kv_num_heads, args.head_dim], dtype=dtype_torch, device="cuda")

        # PyTorch Reference run
        dist.barrier(TP_GROUP)
        torch_q_out, torch_k_out, torch_v_out = torch_pre_attn_qkv_a2a_reference(
            TP_GROUP,
            q_input,
            k_input,
            v_input,
            skip_q_a2a=args.skip_q_a2a,
            debug=args.debug,
            tensor_name=f"Rank {RANK}",
            rank=RANK,
        )

        # Custom IPC memory allocation
        q_src_peers = tilelang.tensor(
            (args.batch_size, SEQ_PER_PE, args.num_heads, args.head_dim),
            dtype_torch,
            allocator=allocator,
            return_peers=True
        )
        k_src_peers = tilelang.tensor(
            (args.batch_size, SEQ_PER_PE, kv_num_heads, args.head_dim),
            dtype_torch,
            allocator=allocator,
            return_peers=True
        )
        v_src_peers = tilelang.tensor(
            (args.batch_size, SEQ_PER_PE, kv_num_heads, args.head_dim),
            dtype_torch,
            allocator=allocator,
            return_peers=True,
        )

        q_dst_peers = tilelang.tensor(
            (args.batch_size, args.seq_len, Q_HEADS_PER_PE, args.head_dim),
            dtype_torch,
            allocator=allocator,
            return_peers=True,
        )
        k_dst_peers = tilelang.tensor(
            (args.batch_size, args.seq_len, KV_HEADS_PER_PE, args.head_dim),
            dtype_torch,
            allocator=allocator,
            return_peers=True,
        )
        v_dst_peers = tilelang.tensor(
            (args.batch_size, args.seq_len, KV_HEADS_PER_PE, args.head_dim),
            dtype_torch,
            allocator=allocator,
            return_peers=True,
        )

        # Local initialization
        q_src_peers[LOCAL_RANK].copy_(q_input)
        k_src_peers[LOCAL_RANK].copy_(k_input)
        v_src_peers[LOCAL_RANK].copy_(v_input)

        q_dst_peers[LOCAL_RANK].fill_(0.0)
        k_dst_peers[LOCAL_RANK].fill_(0.0)
        v_dst_peers[LOCAL_RANK].fill_(0.0)

        torch.cuda.synchronize()
        dist.barrier(LC_GROUP)

        # Run IPC data transfer
        stream = torch.cuda.Stream()
        custom_ipc_pre_attn_qkv_a2a(
            q_src_peers,
            k_src_peers,
            v_src_peers,
            q_dst_peers,
            k_dst_peers,
            v_dst_peers,
            LOCAL_RANK,
            local_world_size,
            Q_HEADS_PER_PE,
            KV_HEADS_PER_PE,
            SEQ_PER_PE,
            stream,
            skip_q_a2a=args.skip_q_a2a,
        )
        stream.synchronize()
        dist.barrier(LC_GROUP)

        custom_q_out = None if args.skip_q_a2a else q_dst_peers[LOCAL_RANK]
        custom_k_out = k_dst_peers[LOCAL_RANK]
        custom_v_out = v_dst_peers[LOCAL_RANK]

        if RANK == 0:
            print("Finished IPC Q/K/V all-to-all.")

        # Collect verification results
        results = []
        if not args.skip_q_a2a:
            results.append(verify_results(custom_q_out, torch_q_out, RANK, "(Q)"))
        results.append(verify_results(custom_k_out, torch_k_out, RANK, "(K)"))
        results.append(verify_results(custom_v_out, torch_v_out, RANK, "(V)"))

        # Output unified result
        if all(results):
            print(f"✅ PE {RANK} All Verification PASSED!")
        else:
            print(f"❌ PE {RANK} Verification FAILED!")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_all_to_all_with_golden_reference_spawn(local_rank: int, num_local_ranks: int, args):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    run_all_to_all_with_golden_reference(args, num_ranks, rank, local_rank, group, group)


if __name__ == "__main__":
    args = parse_args()
    if args.num_processes is None:
        args.num_processes = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    torch.multiprocessing.spawn(
        test_all_to_all_with_golden_reference_spawn,
        args=(args.num_processes, args),
        nprocs=args.num_processes,
    )
