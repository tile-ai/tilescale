import tilelang
import tilelang.language as T
from tilelang.distributed import init_dist
import torch
import torch.distributed as dist
import argparse
from enum import IntEnum

tilelang.disable_cache()


class Direction(IntEnum):
    NORTH = 0
    SOUTH = 1
    WEST = 2
    EAST = 3
    SELF = 4


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
    debug_root_path="/home/zhengju.tang/tilescale/examples/distributed/debug/"
)
def torus_alltoall_xy(PE_num, X, Y, M, N, num_blocks, threads):
    block_M = M // num_blocks
    # Number of slots
    num_slots = PE_num
    
    @T.prim_func
    def main_route_opt(
            # For each (src, dst) pair, the real transfer size is M * N
            src: T.Tensor((PE_num * M, N), "float16"),
            dst: T.Tensor((PE_num * M, N), "float16"),
            # buffer[dst_rank, slots, *, *]: This PE save slots for transferring data chunks for the real destination rank
            buffer_transfer: T.Tensor((PE_num, num_slots, M, N), "float16"),
            # slot_counter[dst_rank, num_blocks]: Counter for each slot in each block
            slot_counter: T.Tensor((PE_num, num_blocks), "uint32"),
            # Signal for each dst buffer
            signal_transfer: T.Tensor((PE_num, num_blocks), "int32"),
            # Src idx during transfer
            src_transfer: T.Tensor((PE_num, num_slots, num_blocks), "uint32"),
            # Signal for finish
            local_finish: T.Tensor((1), "uint32"),
            global_finish: T.Tensor((1), "uint32"),
            # Barrier for all blocks
            barrier: T.Tensor((PE_num), "int32"),
    ):
        with T.Kernel(PE_num, num_blocks, threads=threads) as (bx, bz):
            tx = T.get_thread_binding()

            rank = T.alloc_local([1], "uint32")
            rank_x = T.alloc_local([1], "uint32")
            rank_y = T.alloc_local([1], "uint32")
            next_rank = T.alloc_local([1], "uint32")
            to_dir = T.alloc_local([1], "uint32")
            diff = T.alloc_local([1], "int32")
            dst_rank = T.alloc_local([1], "uint32")
            old_counter = T.alloc_local([1], "uint32")
            old_counter_shared = T.alloc_shared([PE_num, num_blocks], "uint32")
            old_signal = T.alloc_local([1], "int32")
            new_signal = T.alloc_local([1], "int32")
            cur_counter = T.alloc_local([1], "uint32")
            cur_counter_shared = T.alloc_shared([PE_num, num_blocks], "uint32")
            old_local = T.alloc_local([1], "uint32")
            old_global = T.alloc_local([1], "uint32")

            rank[0] = T.get_rank()
            rank_x[0] = T.floordiv(rank[0], Y)
            rank_y[0] = T.floormod(rank[0], Y)
            next_rank[0] = rank[0]
            
            dst_rank[0] = bx

            # Prepare for routing
            dst_rank_x = T.floordiv(dst_rank[0], Y)
            dst_rank_y = T.floormod(dst_rank[0], Y)
            to_dir[0] = Direction.SELF
            # XY-routing: first route along X-axis, then Y-axis
            if dst_rank_x != rank_x[0]:
                # Calculate shortest path in Torus X-dimension
                diff[0] = dst_rank_x - rank_x[0]
                if diff[0] > T.floordiv(X, 2):
                    diff[0] -= X
                elif diff[0] <= -T.ceildiv(X, 2):
                    diff[0] += X

                if diff[0] < 0:
                    # Send North (up): neighbor receives in its north buffer
                    to_dir[0] = Direction.NORTH
                    next_rank[0] = T.floormod(rank_x[0] + X - 1, X) * Y + rank_y[0]
                else:
                    # Send South (down): neighbor receives in its south buffer
                    to_dir[0] = Direction.SOUTH
                    next_rank[0] = T.floormod(rank_x[0] + 1, X) * Y + rank_y[0]
            elif dst_rank_y != rank_y[0]:
                # Calculate shortest path in Torus Y-dimension
                diff[0] = dst_rank_y - rank_y[0]
                if diff[0] > T.floordiv(Y, 2):
                    diff[0] -= Y
                elif diff[0] <= -T.ceildiv(Y, 2):
                    diff[0] += Y

                if diff[0] < 0:
                    # Send West (left): neighbor receives in its west buffer
                    to_dir[0] = Direction.WEST
                    next_rank[0] = rank_x[0] * Y + T.floormod(rank_y[0] + Y - 1, Y)
                else:
                    # Send East (right): neighbor receives in its east buffer
                    to_dir[0] = Direction.EAST
                    next_rank[0] = rank_x[0] * Y + T.floormod(rank_y[0] + 1, Y)

            # Phase 1: Fully use all blocks to initially send from src to the target neighbor
            # Split the tile_M to each block

            if dst_rank[0] != rank[0]:
                if tx == 0:
                    old_counter[0] = T.atom_add_remote(
                        slot_counter[dst_rank[0], bz],
                        1,
                        scope=T.MemoryScope.SYSTEM,
                        sem=T.MemorySemantic.RELEASE,
                        dst_pe=next_rank[0],
                    )
                    # Write src idx to src_transfer
                    T.st(
                        src_transfer[dst_rank[0], old_counter[0], bz],
                        rank[0],
                        scope=T.MemoryScope.SYSTEM,
                        sem=T.MemorySemantic.RELEASE,
                        dst_pe=next_rank[0],
                    )
                    old_counter_shared[dst_rank[0], bz] = old_counter[0]
                T.sync_threads()
                T.put_block(
                    T.address_of(src[dst_rank[0] * M + bz * block_M, 0]),
                    T.address_of(buffer_transfer[dst_rank[0], old_counter_shared[dst_rank[0], bz], bz * block_M, 0]),
                    block_M * N,
                    next_rank[0],
                )
                T.fence_sys(sem=T.MemorySemantic.RELEASE)
                if tx == 0:
                    # Write signal always after remote data is ready
                    T.atom_add_remote(
                        signal_transfer[dst_rank[0], bz],
                        1,
                        scope=T.MemoryScope.SYSTEM,
                        sem=T.MemorySemantic.RELEASE,
                        dst_pe=next_rank[0],
                    )
                T.sync_threads()
                T.fence_sys(sem=T.MemorySemantic.RELEASE)
                # if tx == 0:
                #     T.print(rank[0], "rank")
                #     T.print(next_rank[0], "next rank")
                #     T.print(dst_rank[0], "dst rank")
                #     T.print(old_signal[0], "old signal")
                #     T.print(old_signal_shared[dst_rank[0], bz], "old signal shared")
                #     T.print(signal_transfer[dst_rank[0], bz], "signal transfer")
            else:
                T.put_block(
                    T.address_of(src[dst_rank[0] * M + bz * block_M, 0]),
                    T.address_of(dst[rank[0] * M + bz * block_M, 0]),
                    block_M * N,
                    -1,
                )
                if tx == 0:
                    old_local[0] = T.atom_add(
                        local_finish[0],
                        1,
                        scope=T.MemoryScope.GPU,
                        sem=T.MemorySemantic.RELEASE,
                    )
                T.sync_threads()
                T.fence_cta(sem=T.MemorySemantic.RELEASE)

            T.fence_sys(sem=T.MemorySemantic.RELEASE)

            # Phase 2: Each block handles one final dst data in each slot of buffer of current rank and check whether to transfer
            old_signal[0] = 0
            with T.While(signal_transfer[dst_rank[0], bz] >= old_signal[0]):
                new_signal[0] = T.wait_ne(signal_transfer[dst_rank[0], bz], old_signal[0], scope=T.MemoryScope.GPU)
                # if tx == 0:
                #     T.print(rank[0], "rank")
                #     T.print(new_signal[0], "new signal")

                # Termination signal is -1
                if new_signal[0] < old_signal[0]:
                    T.loop_break()
                # We send all intermediate buffer according to the signal
                for slot_idx in T.serial(old_signal[0], new_signal[0]):
                    src_idx = src_transfer[dst_rank[0], slot_idx, bz]
                    # Handle the transfer signal
                    if dst_rank[0] != rank[0]:
                        if tx == 0:
                            cur_counter[0] = T.atom_add_remote(
                                slot_counter[dst_rank[0], bz],
                                1,
                                scope=T.MemoryScope.SYSTEM,
                                sem=T.MemorySemantic.RELEASE,
                                dst_pe=next_rank[0],
                            )
                            # Write src idx to src_transfer
                            T.st(
                                src_transfer[dst_rank[0], cur_counter[0], bz],
                                src_idx,
                                scope=T.MemoryScope.SYSTEM,
                                sem=T.MemorySemantic.RELEASE,
                                dst_pe=next_rank[0],
                            )
                            cur_counter_shared[dst_rank[0], bz] = cur_counter[0]
                        T.sync_threads()
                        T.put_block(
                            T.address_of(buffer_transfer[dst_rank[0], slot_idx, bz * block_M, 0]),
                            T.address_of(buffer_transfer[dst_rank[0], cur_counter_shared[dst_rank[0], bz], bz * block_M, 0]),
                            block_M * N,
                            dst_pe=next_rank[0],
                        )
                        T.fence_sys(sem=T.MemorySemantic.RELEASE)
                        if tx == 0:
                            # Write signal always after remote data is ready
                            T.atom_add_remote(
                                signal_transfer[dst_rank[0], bz],
                                1,
                                scope=T.MemoryScope.SYSTEM,
                                sem=T.MemorySemantic.RELEASE,
                                dst_pe=next_rank[0],
                            )
                        T.sync_threads()
                        T.fence_sys(sem=T.MemorySemantic.RELEASE)
                        # if tx == 0:
                        #     T.print(rank[0], "rank")
                        #     T.print(slot_idx, "slot idx")
                        #     T.print(next_rank[0], "transfer to rank")
                    else:
                        # Current rank is the real destination of this chunk of data, the real source rank is the buffer index
                        # if tx == 0:
                        #     T.print(rank[0], "rank")
                        #     T.print(src_transfer[dst_rank[0], slot_idx, bz], "src transfer")
                        #     T.print(slot_idx, "slot idx")
                        T.put_block(
                            T.address_of(buffer_transfer[dst_rank[0], slot_idx, bz * block_M, 0]),
                            T.address_of(dst[src_transfer[dst_rank[0], slot_idx, bz] * M + bz * block_M, 0]),
                            block_M * N,
                            -1,
                        )
                        T.fence_cta(sem=T.MemorySemantic.RELEASE)
                        T.sync_threads()
                        if tx == 0:
                            old_local[0] = T.atom_add(
                                local_finish[0],
                                1,
                                scope=T.MemoryScope.GPU,
                                sem=T.MemorySemantic.RELEASE,
                            )
                            # if bz == 0 and tx == 0:
                            #     T.print(rank[0], "dst rank")
                            if old_local[0] + 1 == PE_num * num_blocks:
                                for i in T.serial(PE_num):
                                    old_global[0] = T.atom_add_remote(
                                        global_finish[0],
                                        1,
                                        scope=T.MemoryScope.SYSTEM,
                                        sem=T.MemorySemantic.RELEASE,
                                        dst_pe=i,
                                    )
                                if old_global[0] + 1 == PE_num:
                                    # Send termination signals to wake up all waiting blocks on all PEs
                                    for remote_pe in T.serial(PE_num):
                                        for dst_rank_idx in T.serial(PE_num):
                                            for bz_idx in T.serial(num_blocks):
                                                T.st(
                                                    signal_transfer[dst_rank_idx, bz_idx],
                                                    -1,
                                                    scope=T.MemoryScope.SYSTEM,
                                                    sem=T.MemorySemantic.RELEASE,
                                                    dst_pe=remote_pe,
                                                )
                        T.sync_threads()
                
                # Update old signal for next iteration
                old_signal[0] = new_signal[0]
                T.sync_threads()

            T.fence_sys(sem=T.MemorySemantic.RELEASE)

    return main_route_opt


def run_torus_alltoall(local_rank, num_ranks, args):
    NUM_SM = 148
    PE_num = args.PE_num
    X, Y = args.X, args.Y
    M, N = args.M, args.N
    blocks = args.blocks
    block_M, block_N = M // blocks, N
    threads = 512

    num_blocks = blocks
    num_blocks = min(num_blocks, NUM_SM // PE_num)

    local_rank, num_ranks, group_size = init_dist(local_rank, num_ranks)
    allocator = tilelang.get_allocator(
        size=2**34,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group_size,
    )

    kernel = torus_alltoall_xy(PE_num, X, Y, M, N, num_blocks, threads)
    kernel.initialize(allocator=allocator)

    src = tilelang.tensor((PE_num * M, N), torch.float16, allocator=allocator).random_()
    dst = tilelang.tensor((PE_num * M, N), torch.float16, allocator=allocator).zero_()

    dst_ref = torch.zeros((PE_num * M, N), dtype=torch.float16, device="cuda")
    dist.all_to_all_single(dst_ref, src, group=group_size)
    torch.cuda.synchronize()

    buffer_transfer = tilelang.tensor((PE_num, PE_num, M, N), torch.float16,
                                      allocator=allocator).fill_(-1)

    # Signals for each buffer slot in each direction
    slot_counter = tilelang.tensor((PE_num, num_blocks), torch.uint32, allocator=allocator).fill_(0)
    signal_transfer = tilelang.tensor((PE_num, num_blocks),
                                      torch.int32,
                                      allocator=allocator).fill_(0)

    src_transfer = tilelang.tensor((PE_num, PE_num, num_blocks),
                                    torch.uint32,
                                    allocator=allocator).fill_(PE_num)

    local_finish = tilelang.tensor((1), torch.uint32, allocator=allocator).fill_(0)
    global_finish = tilelang.tensor((1), torch.uint32, allocator=allocator).fill_(0)
    barrier = tilelang.tensor((PE_num), torch.int32, allocator=allocator).zero_()

    torch.cuda.synchronize()
    dist.barrier(group_size)

    kernel(src, dst, buffer_transfer, slot_counter, signal_transfer, src_transfer, local_finish, global_finish, barrier)

    torch.cuda.synchronize()
    dist.barrier(group_size)

    print(f"Rank {local_rank} TileLang AllToAll XY Routing Finished.")

    if torch.allclose(dst, dst_ref, atol=1e-2, rtol=1e-2):
        print(f"Rank {local_rank} Verification Passed! ✅")
    else:
        max_diff = (dst - dst_ref).abs().max()
        print(f"Rank {local_rank} Verification Failed! ❌ Max diff: {max_diff}")
        # Find differences
        diff_mask = (dst != dst_ref)
        diff_count = diff_mask.sum().item()

        if diff_count > 0:
            # Check each rank's symmetric buffer at every M boundary
            print(f"Rank {local_rank} found {diff_count} differences")
            print(f"Rank {local_rank} checking symmetric buffer positions at M={M} boundaries:")
            for rank_idx in range(PE_num):
                start_idx = rank_idx * M
                # Check first element of each rank's buffer
                print(
                    f"Rank {local_rank} Buffer[{rank_idx}][0,0]: dst={dst[start_idx, 0].item():.6f}, dst_ref={dst_ref[start_idx, 0].item():.6f}"
                )
            for slot in range(PE_num):
                print(
                    f"Rank {local_rank} slot={slot}: buffer={buffer_transfer[local_rank, slot, 0, 0].item():.6f}, src_transfer={src_transfer[local_rank, slot, 0].item():.6f}"
                )
            # Print first 10 differences with their coordinates
            print(f"Rank {local_rank} first 10 differences:")
            diff_indices = torch.nonzero(diff_mask, as_tuple=False)
            for i in range(min(10, diff_indices.shape[0])):
                idx = diff_indices[i]
                row, col = idx[0].item(), idx[1].item()
                print(
                    f"Rank {local_rank} Diff[{i}] at ({row}, {col}): dst={dst[row, col].item():.6f}, dst_ref={dst_ref[row, col].item():.6f}"
                )
    # print(f"Rank {local_rank} slot_counter: {slot_counter}, signal_transfer: {signal_transfer}, src_transfer: {src_transfer}")

    if args.benchmark:
        # Warmup
        for _ in range(args.warmup):
            # Reinitialize
            buffer_transfer.fill_(-1)
            src_transfer.fill_(PE_num)
            slot_counter.zero_()
            signal_transfer.zero_()
            local_finish.zero_()
            global_finish.zero_()
            kernel(src, dst, buffer_transfer, slot_counter, signal_transfer, src_transfer, local_finish, global_finish, barrier)
            torch.cuda.synchronize()
        dist.barrier(group_size)

        # Reinitialize
        buffer_transfer.fill_(-1)
        src_transfer.fill_(PE_num)
        slot_counter.zero_()
        signal_transfer.zero_()
        local_finish.zero_()
        global_finish.zero_()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        num_iters = args.iter
        start_event.record()
        for _ in range(num_iters):
            # torch.cuda.profiler.start()
            # with torch.cuda.nvtx.range("alltoall_xy_routing_benchmark"):
            kernel(src, dst, buffer_transfer, slot_counter, signal_transfer, src_transfer, local_finish, global_finish, barrier)
            # torch.cuda.profiler.stop()
            torch.cuda.synchronize()
            dist.barrier(group_size)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event) / num_iters
        # All-to-all total data moved: each rank sends (PE_num - 1) * M * N elements
        # and receives (PE_num - 1) * M * N elements.
        # For bandwidth calculation, we usually use the amount of data sent per rank.
        total_data_bytes = PE_num * M * N * 2  # float16 = 2 bytes
        bandwidth_gbps = (total_data_bytes / 1e9) / (elapsed_time_ms / 1e3)
        print(f"Rank {local_rank} Average Latency: {elapsed_time_ms:.4f} ms, Effective Bandwidth: {bandwidth_gbps:.4f} GB/s")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=7168)
    parser.add_argument("--PE_num", type=int, default=8)
    parser.add_argument("--X", type=int, default=2)
    parser.add_argument("--Y", type=int, default=4)
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--iter", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--blocks", type=int, default=1, help="Number of blocks")
    args = parser.parse_args()

    torch.multiprocessing.spawn(run_torus_alltoall, args=(args.PE_num, args), nprocs=args.PE_num)
