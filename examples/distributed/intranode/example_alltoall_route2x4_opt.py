import os
import tilelang
import tilelang.language as T
from tilelang.distributed import init_dist
import torch
import torch.distributed as dist
import argparse
from enum import IntEnum

# tilelang.disable_cache()

os.environ["TILELANG_USE_DISTRIBUTED"] = "1"

def _dbg(rank: int, msg: str, debug: bool) -> None:
    if debug:
        print(f"[Rank {rank}] {msg}", flush=True)


class Direction(IntEnum):
    NORTH = 0
    SOUTH = 1
    WEST = 2
    EAST = 3
    SELF = 4


def compute_next_hop(src_rank, dst_rank, X, Y):
    """Compute the next hop from src_rank towards dst_rank using XY-routing on a 2D torus."""
    if src_rank == dst_rank:
        return src_rank  # self, no hop needed

    src_x, src_y = src_rank // Y, src_rank % Y
    dst_x, dst_y = dst_rank // Y, dst_rank % Y

    if dst_x != src_x:
        # Route along X-axis first
        diff = dst_x - src_x
        if diff > X // 2:
            diff -= X
        elif diff <= -X // 2:
            diff += X

        if diff < 0:
            # North
            next_x = (src_x + X - 1) % X
            return next_x * Y + src_y
        else:
            # South
            next_x = (src_x + 1) % X
            return next_x * Y + src_y
    else:
        # Route along Y-axis
        diff = dst_y - src_y
        if diff > Y // 2:
            diff -= Y
        elif diff <= -Y // 2:
            diff += Y

        if diff < 0:
            # West
            next_y = (src_y + Y - 1) % Y
            return src_x * Y + next_y
        else:
            # East
            next_y = (src_y + 1) % Y
            return src_x * Y + next_y


def compute_expected_slots(PE_num, X, Y):
    """
    Pre-compute the number of incoming slots each PE will receive for each dst_rank,
    tracing the full XY-routing path through the torus.

    Returns a list of lists of shape (PE_num, PE_num) where expected_slots[me][dst_rank] is
    the number of data chunks that PE 'me' will receive in its buffer for destination 'dst_rank'.

    This includes both:
    - Final reception: dst_rank == me, each other PE sends one chunk → PE_num - 1 slots
    - Forwarding: dst_rank != me, data passes through me on the way to dst_rank

    For each (src, final_dst) pair where src != final_dst, we trace the full multi-hop
    XY-routing path. Every intermediate PE and the final destination PE each receive one slot.
    """
    # expected_slots[receiver_pe][dst_rank] = count of incoming slots
    expected = [[0] * PE_num for _ in range(PE_num)]

    for src in range(PE_num):
        for final_dst in range(PE_num):
            if src == final_dst:
                continue
            # Trace the full path from src to final_dst
            current = src
            while current != final_dst:
                next_hop = compute_next_hop(current, final_dst, X, Y)
                # next_hop receives one slot for dst_rank=final_dst
                expected[next_hop][final_dst] += 1
                current = next_hop

    return expected


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
    # debug_root_path="/home/zhengju.tang/tilescale/examples/distributed/debug/"
)
def torus_alltoall_xy(PE_num, X, Y, M, N, num_blocks, num_warps, threads, enable_profiling=False):
    block_M = M // num_blocks
    tile_M = block_M // num_warps
    # Number of slots
    num_slots = PE_num
    
    @T.prim_func
    def main_route_opt(
            # For each (src, dst) pair, the real transfer size is M * N
            src: T.Tensor((PE_num * M, N), "float16"),
            dst: T.Tensor((PE_num * M, N), "float16"),
            # buffer[dst_rank, slots, *, *]: This PE save slots for transferring data chunks for the real destination rank
            buffer_transfer: T.Tensor((PE_num, num_slots, M, N), "float16"),
            # slot_counter[dst_rank, num_blocks, num_warps]: Counter for allocating slot indices
            slot_counter: T.Tensor((PE_num, num_blocks, num_warps), "uint32"),
            # Per-slot ready flag: writer sets signal=run_id (RELEASE); reader wait_eq(signal, run_id) (ACQUIRE)
            # signal_transfer[dst_rank, slot, num_blocks, num_warps]
            signal_transfer: T.Tensor((PE_num, num_slots, num_blocks, num_warps), "int32"),
            # Src idx during transfer
            src_transfer: T.Tensor((PE_num, num_slots, num_blocks, num_warps), "uint32"),
            # Pre-computed expected incoming slot counts per (PE, dst_rank)
            # expected_slots[pe, dst_rank]: how many slots PE 'pe' will receive for dst_rank
            expected_slots: T.Tensor((PE_num, PE_num), "int32"),
            # Run generation: writers set signal = run_id, readers wait_eq(signal, run_id). Must be != 0.
            run_id: T.Tensor((1,), "int32"),
            # Profiling timestamps: [dst_rank, slot_idx, phase, block, warp]
            # phase 0 = phase1 start
            # phase 1 = phase1 end
            # phase 2 = phase2 iteration start (before wait_eq)
            # phase 3 = phase2 after wait_eq
            # phase 4 = phase2 iteration end (after put+signal)
            timestamps: T.Tensor((PE_num, num_slots, 5, num_blocks, num_warps), "int64"),
    ):
        with T.Kernel(PE_num, num_blocks, threads=threads) as (bx, bz):
            tx = T.get_thread_binding()
            warp_id = tx // 32

            rank = T.alloc_local([1], "uint32")
            rank_x = T.alloc_local([1], "uint32")
            rank_y = T.alloc_local([1], "uint32")
            next_rank = T.alloc_local([1], "uint32")
            to_dir = T.alloc_local([1], "uint32")
            diff = T.alloc_local([1], "int32")
            dst_rank = T.alloc_local([1], "uint32")
            old_counter = T.alloc_local([1], "uint32")
            slot_physical = T.alloc_local([1], "uint32")
            old_counter_shared = T.alloc_shared([PE_num, num_blocks, num_warps], "uint32")
            cur_counter = T.alloc_local([1], "uint32")
            cur_slot_physical = T.alloc_local([1], "uint32")
            cur_counter_shared = T.alloc_shared([PE_num, num_blocks, num_warps], "uint32")
            slot_flag = T.alloc_local([1], "int32")
            slot_flag_shared = T.alloc_shared([num_warps], "int32")
            num_expected = T.alloc_local([1], "int32")
            ts_val = T.alloc_local([1], "int64")

            rank[0] = T.get_rank()
            rank_x[0] = T.floordiv(rank[0], Y)
            rank_y[0] = T.floormod(rank[0], Y)
            next_rank[0] = rank[0]
            
            dst_rank[0] = bx

            # Read pre-computed expected slot count for this PE and dst_rank
            num_expected[0] = expected_slots[rank[0], dst_rank[0]]

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

            if enable_profiling:
                if tx % 32 == 0:
                    ts_val[0] = T.get_clock()
                    timestamps[dst_rank[0], 0, 0, bz, warp_id] = ts_val[0]
                T.sync_warp()

            chunk_M = T.min(block_M - warp_id * tile_M, tile_M)
            if dst_rank[0] != rank[0]:
                if tx % 32 == 0:
                    old_counter[0] = T.atom_add_remote(
                        slot_counter[dst_rank[0], bz, warp_id],
                        1,
                        scope=T.MemoryScope.SYSTEM,
                        sem=T.MemorySemantic.RELEASE,
                        dst_pe=next_rank[0],
                    )
                    slot_physical[0] = T.floormod(old_counter[0], num_slots)
                    old_counter_shared[dst_rank[0], bz, warp_id] = slot_physical[0]
                    T.st(
                        src_transfer[dst_rank[0], slot_physical[0], bz, warp_id],
                        rank[0],
                        scope=T.MemoryScope.SYSTEM,
                        sem=T.MemorySemantic.RELEASE,
                        dst_pe=next_rank[0],
                    )
                T.sync_warp()
                if next_rank[0] == dst_rank[0]:
                    # Last hop: write directly to dst on the destination PE
                    T.put_warp(
                        T.address_of(src[dst_rank[0] * M + bz * block_M + warp_id * tile_M, 0]),
                        T.address_of(dst[rank[0] * M + bz * block_M + warp_id * tile_M, 0]),
                        chunk_M * N,
                        next_rank[0],
                    )
                else:
                    T.put_warp(
                        T.address_of(src[dst_rank[0] * M + bz * block_M + warp_id * tile_M, 0]),
                        T.address_of(buffer_transfer[dst_rank[0], old_counter_shared[dst_rank[0], bz, warp_id], bz * block_M + warp_id * tile_M, 0]),
                        chunk_M * N,
                        next_rank[0],
                    )
                T.fence_sys(sem=T.MemorySemantic.RELEASE)
                if tx % 32 == 0:
                    T.st(
                        signal_transfer[dst_rank[0], old_counter_shared[dst_rank[0], bz, warp_id], bz, warp_id],
                        run_id[0],
                        scope=T.MemoryScope.SYSTEM,
                        sem=T.MemorySemantic.RELEASE,
                        dst_pe=next_rank[0],
                    )
                T.sync_warp()
                T.fence_sys(sem=T.MemorySemantic.RELEASE)
            else:
                T.put_warp(
                    T.address_of(src[dst_rank[0] * M + bz * block_M + warp_id * tile_M, 0]),
                    T.address_of(dst[rank[0] * M + bz * block_M + warp_id * tile_M, 0]),
                    chunk_M * N,
                    -1,
                )
                T.sync_warp()
                T.fence_cta(sem=T.MemorySemantic.RELEASE)

            if enable_profiling:
                if tx % 32 == 0:
                    ts_val[0] = T.get_clock()
                    timestamps[dst_rank[0], 0, 1, bz, warp_id] = ts_val[0]
                T.sync_warp()
            
            # Phase 2: Poll per-slot ready flags sequentially and process each slot.
            # Use pre-computed expected_slots count to know exactly how many slots to process,
            # eliminating the need for termination signals.
            for slot_idx in T.serial(num_slots):
                # Skip if no more expected slots or this block doesn't receive for this dst_rank
                if slot_idx >= num_expected[0]:
                    T.loop_break()

                if enable_profiling:
                    if tx % 32 == 0:
                        ts_val[0] = T.get_clock()
                        timestamps[dst_rank[0], slot_idx, 2, bz, warp_id] = ts_val[0]

                if tx % 32 == 0:
                    slot_flag[0] = T.wait_eq(
                        signal_transfer[dst_rank[0], slot_idx, bz, warp_id],
                        run_id[0],
                        scope=T.MemoryScope.SYSTEM,
                        semantic=T.MemorySemantic.ACQUIRE,
                    )
                    slot_flag_shared[warp_id] = slot_flag[0]
                T.sync_warp()
                # Ensure consumer sees producer's RELEASE writes: no reorder of subsequent reads above wait
                T.fence_sys(sem=T.MemorySemantic.ACQUIRE)

                if enable_profiling:
                    if tx % 32 == 0:
                        ts_val[0] = T.get_clock()
                        timestamps[dst_rank[0], slot_idx, 3, bz, warp_id] = ts_val[0]
                    T.sync_warp()

                src_idx = src_transfer[dst_rank[0], slot_idx, bz, warp_id]
                # Handle the transfer
                if dst_rank[0] != rank[0]:
                    # Forward to next hop
                    if tx % 32 == 0:
                        cur_counter[0] = T.atom_add_remote(
                            slot_counter[dst_rank[0], bz, warp_id],
                            1,
                            scope=T.MemoryScope.SYSTEM,
                            sem=T.MemorySemantic.RELEASE,
                            dst_pe=next_rank[0],
                        )
                        cur_slot_physical[0] = T.floormod(cur_counter[0], num_slots)
                        cur_counter_shared[dst_rank[0], bz, warp_id] = cur_slot_physical[0]
                        T.st(
                            src_transfer[dst_rank[0], cur_slot_physical[0], bz, warp_id],
                            src_idx,
                            scope=T.MemoryScope.SYSTEM,
                            sem=T.MemorySemantic.RELEASE,
                            dst_pe=next_rank[0],
                        )
                    T.sync_warp()
                    if next_rank[0] == dst_rank[0]:
                        # Last hop: write directly to dst on the destination PE
                        T.put_warp(
                            T.address_of(buffer_transfer[dst_rank[0], slot_idx, bz * block_M + warp_id * tile_M, 0]),
                            T.address_of(dst[src_idx * M + bz * block_M + warp_id * tile_M, 0]),
                            chunk_M * N,
                            dst_pe=next_rank[0],
                        )
                    else:
                        T.put_warp(
                            T.address_of(buffer_transfer[dst_rank[0], slot_idx, bz * block_M + warp_id * tile_M, 0]),
                            T.address_of(buffer_transfer[dst_rank[0], cur_counter_shared[dst_rank[0], bz, warp_id], bz * block_M + warp_id * tile_M, 0]),
                            chunk_M * N,
                            dst_pe=next_rank[0],
                        )
                    T.fence_sys(sem=T.MemorySemantic.RELEASE)
                    T.sync_warp()
                    if tx % 32 == 0:
                        T.st(
                            signal_transfer[dst_rank[0], cur_counter_shared[dst_rank[0], bz, warp_id], bz, warp_id],
                            run_id[0],
                            scope=T.MemoryScope.SYSTEM,
                            sem=T.MemorySemantic.RELEASE,
                            dst_pe=next_rank[0],
                        )
                    T.sync_warp()
                    T.fence_sys(sem=T.MemorySemantic.RELEASE)
                else:
                    # Final destination: data already written to dst by the last-hop sender.
                    # Just synchronize, no local copy needed.
                    T.sync_warp()

                if enable_profiling:
                    if tx % 32 == 0:
                        ts_val[0] = T.get_clock()
                        timestamps[dst_rank[0], slot_idx, 4, bz, warp_id] = ts_val[0]
                    T.sync_warp()

            T.sync_warp()

    return main_route_opt


def run_torus_alltoall(local_rank, num_ranks, args):
    debug = getattr(args, "debug", False)
    _dbg(local_rank, "run_torus_alltoall started", debug)

    NUM_SM = 148
    PE_num = args.PE_num
    X, Y = args.X, args.Y
    M, N = args.M, args.N
    blocks = args.blocks
    threads = 512
    assert threads % 32 == 0, "threads must be divisible by 32"
    num_warps = threads // 32

    num_blocks = blocks
    num_blocks = min(num_blocks, NUM_SM // PE_num)

    local_rank, num_ranks, group_size = init_dist(local_rank, num_ranks)
    _dbg(local_rank, "init_dist done", debug)

    allocator = tilelang.get_allocator(
        size=2**34,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group_size,
    )

    # Pre-compute expected slot counts based on XY-routing topology
    expected_slots_host = compute_expected_slots(PE_num, X, Y)
    if local_rank == 0:
        print(f"Expected slots per PE (rows=receiver, cols=dst_rank):")
        for pe in range(PE_num):
            print(f"  PE {pe}: {expected_slots_host[pe]}")

    enable_profiling = args.profile
    kernel = torus_alltoall_xy(PE_num, X, Y, M, N, num_blocks, num_warps, threads, enable_profiling=enable_profiling)
    kernel.initialize(allocator=allocator)

    src = tilelang.tensor((PE_num * M, N), torch.float16, allocator=allocator).random_()
    dst = tilelang.tensor((PE_num * M, N), torch.float16, allocator=allocator).zero_()

    dst_ref = torch.zeros((PE_num * M, N), dtype=torch.float16, device="cuda")
    dist.all_to_all_single(dst_ref, src, group=group_size)
    torch.cuda.synchronize()

    buffer_transfer = tilelang.tensor((PE_num, PE_num, M, N), torch.float16,
                                      allocator=allocator).fill_(-1)

    # Signals for each buffer slot in each direction
    slot_counter = tilelang.tensor((PE_num, num_blocks, num_warps), torch.uint32, allocator=allocator).fill_(0)
    # Per-slot ready flags: 0=not ready, 1=ready
    signal_transfer = tilelang.tensor((PE_num, PE_num, num_blocks, num_warps),
                                      torch.int32,
                                      allocator=allocator).fill_(0)

    src_transfer = tilelang.tensor((PE_num, PE_num, num_blocks, num_warps),
                                    torch.uint32,
                                    allocator=allocator).fill_(PE_num)

    # Pre-computed expected slots tensor (same for all PEs)
    expected_slots_tensor = tilelang.tensor((PE_num, PE_num), torch.int32, allocator=allocator)
    expected_slots_tensor.copy_(torch.tensor(expected_slots_host, dtype=torch.int32, device="cuda"))

    # Profiling timestamps: [dst_rank, num_slots, 5 phases, num_blocks, num_warps]
    num_slots = PE_num
    timestamps_tensor = tilelang.tensor(
        (PE_num, num_slots, 5, num_blocks, num_warps), torch.int64, allocator=allocator).zero_()

    # Run generation: must be != 0 so wait_eq(signal, run_id) does not pass on unwritten memory
    run_id_tensor = tilelang.tensor((1,), torch.int32, allocator=allocator)

    torch.cuda.synchronize()
    _dbg(local_rank, "pre-verify: before barrier", debug)
    dist.barrier(group_size)
    _dbg(local_rank, "pre-verify: after barrier, launching kernel run_id=1", debug)

    run_id_tensor.copy_(torch.tensor([1], dtype=torch.int32, device="cuda"))
    if enable_profiling:
        timestamps_tensor.zero_()
    kernel(src, dst, buffer_transfer, slot_counter, signal_transfer, src_transfer, expected_slots_tensor, run_id_tensor, timestamps_tensor)

    _dbg(local_rank, "verify kernel returned", debug)
    torch.cuda.synchronize()
    _dbg(local_rank, "pre-verify: before post barrier", debug)
    dist.barrier(group_size)
    _dbg(local_rank, "verify done", debug)

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
                    f"Rank {local_rank} slot={slot}: buffer={buffer_transfer[local_rank, slot, 0, 0].item():.6f}, src_transfer={src_transfer[local_rank, slot, 0, 0].item():.6f}"
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

    if args.benchmark:
        run_id = 2  # 1 used for verification
        for w in range(args.warmup):
            slot_counter.zero_()
            torch.cuda.synchronize()
            _dbg(local_rank, f"warmup {w}: before barrier", debug)
            dist.barrier(group_size)
            _dbg(local_rank, f"warmup {w}: launching kernel run_id={run_id}", debug)
            run_id_tensor.copy_(torch.tensor([run_id], dtype=torch.int32, device="cuda"))
            run_id += 1
            kernel(src, dst, buffer_transfer, slot_counter, signal_transfer, src_transfer, expected_slots_tensor, run_id_tensor, timestamps_tensor)
            _dbg(local_rank, f"warmup {w}: kernel returned", debug)
            torch.cuda.synchronize()
            dist.barrier(group_size)

        num_iters = args.iter
        elapsed_time_ms = []
        for i in range(num_iters):
            slot_counter.zero_()
            torch.cuda.synchronize()
            _dbg(local_rank, f"iter {i}: before barrier", debug)
            dist.barrier(group_size)
            _dbg(local_rank, f"iter {i}: launching kernel run_id={run_id + i}", debug)
            run_id_tensor.copy_(torch.tensor([run_id + i], dtype=torch.int32, device="cuda"))
            if enable_profiling:
                timestamps_tensor.zero_()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            # torch.cuda.profiler.start()
            # with torch.cuda.nvtx.range("alltoall_xy_routing_benchmark"):
            kernel(src, dst, buffer_transfer, slot_counter, signal_transfer, src_transfer, expected_slots_tensor, run_id_tensor, timestamps_tensor)
            # torch.cuda.synchronize()
            # dist.barrier(group_size)
            # torch.cuda.profiler.stop()
            end_event.record()
            torch.cuda.synchronize()
            dist.barrier(group_size)
            _dbg(local_rank, f"iter {i}: kernel returned", debug)
            elapsed_time_ms.append(start_event.elapsed_time(end_event))
        torch.cuda.synchronize()

        elapsed_time_ms = sum(elapsed_time_ms) / num_iters
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
    parser.add_argument("--profile", action="store_true", help="Print phase2 per-iteration timing from T.get_clock()")
    parser.add_argument("--debug", action="store_true", help="Print debug messages at barriers/kernel launch")
    args = parser.parse_args()

    torch.multiprocessing.spawn(run_torus_alltoall, args=(args.PE_num, args), nprocs=args.PE_num)
