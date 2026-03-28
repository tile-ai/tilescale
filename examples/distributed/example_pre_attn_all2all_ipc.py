import torch
import torch.distributed as dist
import torch.multiprocessing
import tilelang
from tilelang.distributed import init_dist, dtype_map
import argparse
import os

tilelang.disable_cache()


def torch_sequence_all_to_all_reference(data_src, group):
    """
    PyTorch Distributed All-to-All Golden Reference

    Input:  [BATCH_SIZE, NUM_HEADS, SEQ_PER_PE, HEAD_DIM] - full heads, partial sequence per PE
    Output: [BATCH_SIZE, HEADS_PER_PE, SEQ_LEN, HEAD_DIM] - partial heads, full sequence per PE

    Args:
        data_src: Input tensor on each PE
        group: Distributed process group

    Returns:
        Output tensor after all-to-all communication
    """
    world_size = dist.get_world_size(group)
    batch_size, num_heads, seq_per_pe, head_dim = data_src.shape
    seq_len = seq_per_pe * world_size
    heads_per_pe = num_heads // world_size

    # Step 1: Prepare input list for all_to_all
    # Split data_src by heads and create send list
    input_list = []
    for pe_idx in range(world_size):
        # For each target PE, extract the heads that should go to that PE
        start_head = pe_idx * heads_per_pe
        end_head = (pe_idx + 1) * heads_per_pe

        # Extract [BATCH_SIZE, HEADS_PER_PE, SEQ_PER_PE, HEAD_DIM] for target PE
        send_data = data_src[:, start_head:end_head, :, :].contiguous()
        input_list.append(send_data)

    # Step 2: Prepare output list for all_to_all
    output_list = []
    for _ in range(world_size):
        # Receive [BATCH_SIZE, HEADS_PER_PE, SEQ_PER_PE, HEAD_DIM] from each PE
        recv_data = torch.empty(batch_size, heads_per_pe, seq_per_pe, head_dim, dtype=data_src.dtype, device=data_src.device)
        output_list.append(recv_data)

    # Step 3: Execute all_to_all
    dist.all_to_all(output_list, input_list, group=group)

    # Step 4: Reorganize received data
    # output_list[pe_idx] contains data from PE pe_idx
    # Need to arrange by sequence dimension
    result = torch.empty(batch_size, heads_per_pe, seq_len, head_dim, dtype=data_src.dtype, device=data_src.device)

    for pe_idx in range(world_size):
        seq_start = pe_idx * seq_per_pe
        seq_end = (pe_idx + 1) * seq_per_pe
        # Place data from PE pe_idx into the correct sequence positions
        result[:, :, seq_start:seq_end, :] = output_list[pe_idx]

    return result


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
            # data_src shape on src_rank: [BATCH_SIZE, NUM_HEADS, SEQ_PER_PE, HEAD_DIM]
            src = data_src_peers[src_rank][
                :, 
                local_rank * HEADS_PER_PE : (local_rank + 1) * HEADS_PER_PE, 
                :, 
                :
            ]
            
            # We place this data into our local data_dst at the sequence position corresponding to src_rank.
            # data_dst shape on local_rank: [BATCH_SIZE, HEADS_PER_PE, SEQ_LEN, HEAD_DIM]
            dst = data_dst_peers[local_rank][
                :, 
                :, 
                src_rank * SEQ_PER_PE : (src_rank + 1) * SEQ_PER_PE, 
                :
            ]
            
            # Execute P2P copy
            dst.copy_(src)


def verify_results(custom_output, torch_output, rank, tolerance=1e-3):
    """Verify TileLang output against PyTorch golden reference"""
    if not torch.allclose(custom_output, torch_output, atol=tolerance, rtol=tolerance):
        print(f"❌ PE {rank} Verification FAILED!")

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
        print(f"✅ PE {rank} Verification PASSED!")
        return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=None, help="Number of GPUs to spawn")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--dtype", default="float16")
    return parser.parse_args()


def run_all_to_all_with_golden_reference(args, WORLD_SIZE, RANK, LOCAL_RANK, TP_GROUP, LC_GROUP):
    try:
        PE_num = WORLD_SIZE
        assert args.seq_len % PE_num == 0
        assert args.num_heads % PE_num == 0

        SEQ_PER_PE = args.seq_len // PE_num
        HEADS_PER_PE = args.num_heads // PE_num
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
            print(f"Head dimension: {args.head_dim}")
            print(f"PE count: {PE_num}")
            print(f"Sequence per PE: {SEQ_PER_PE}")
            print(f"Heads per PE: {HEADS_PER_PE}")

        dtype_torch = dtype_map[args.dtype]
        
        # PyTorch reference input
        input_data = torch.rand([args.batch_size, args.num_heads, SEQ_PER_PE, args.head_dim], dtype=dtype_torch, device="cuda")

        # PyTorch Reference run
        dist.barrier(TP_GROUP)
        torch_output = torch_sequence_all_to_all_reference(input_data, TP_GROUP)

        # Custom IPC memory allocation
        data_src_peers = tilelang.tensor(
            (args.batch_size, args.num_heads, SEQ_PER_PE, args.head_dim),
            dtype_torch,
            allocator=allocator,
            return_peers=True
        )
        data_dst_peers = tilelang.tensor(
            (args.batch_size, HEADS_PER_PE, args.seq_len, args.head_dim),
            dtype_torch,
            allocator=allocator,
            return_peers=True
        )

        # Local initialization
        data_src_peers[LOCAL_RANK].copy_(input_data)
        data_dst_peers[LOCAL_RANK].fill_(0.0)

        torch.cuda.synchronize()
        dist.barrier(LC_GROUP)

        # Run IPC data transfer
        stream = torch.cuda.Stream()
        custom_ipc_all_to_all(
            data_src_peers,
            data_dst_peers,
            LOCAL_RANK,
            local_world_size,
            HEADS_PER_PE,
            SEQ_PER_PE,
            stream
        )
        stream.synchronize()
        dist.barrier(LC_GROUP)

        custom_output = data_dst_peers[LOCAL_RANK]

        if RANK == 0:
            print("Finished IPC all-to-all.")

        verify_results(custom_output, torch_output, RANK)
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
