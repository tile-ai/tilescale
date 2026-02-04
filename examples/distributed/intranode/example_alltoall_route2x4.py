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


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
},debug_root_path="/home/zhengju.tang/tilescale/examples/distributed/debug/")
def torus_alltoall_xy(PE_num, X, Y, M, N, block_M, block_N, threads):

    @T.prim_func
    def main(
        # For each (src, dst) pair, the real transfer size is M * N
        src: T.Tensor((PE_num * M, N), "float16"),
        dst: T.Tensor((PE_num * M, N), "float16"),
        # buffer[src_rank, dst_rank, *, *]: This PE save a slot for transferring data chunks for the real destination rank
        buffer_transfer: T.Tensor((PE_num, PE_num, M, N), "float16"),
        # Signal for each buffer
        signal_transfer: T.Tensor((PE_num, PE_num), "uint32"),
        # Signal for finish
        local_finish: T.Tensor((1), "uint32"),
        global_finish: T.Tensor((1), "uint32"),
        # Barrier for all blocks
        barrier: T.Tensor((PE_num), "int32"),
    ):
        with T.Kernel(PE_num, PE_num, T.ceildiv(M, block_M), threads=threads) as (bx, by, bz):
            tx = T.get_thread_binding()
            
            rank = T.alloc_local([1], "uint32")
            rank_x = T.alloc_local([1], "uint32")
            rank_y = T.alloc_local([1], "uint32")
            next_rank = T.alloc_local([1], "uint32")
            to_dir = T.alloc_local([1], "uint32")
            diff = T.alloc_local([1], "int32")
            src_rank = T.alloc_local([1], "uint32")
            dst_rank = T.alloc_local([1], "uint32")
            old_local = T.alloc_local([1], "uint32")
            old_global = T.alloc_local([1], "uint32")
            num_block_M = T.alloc_local([1], "uint32")

            rank[0] = T.get_rank()
            rank_x[0] = T.floordiv(rank[0], Y)
            rank_y[0] = T.floormod(rank[0], Y)
            next_rank[0] = rank[0]

            num_block_M[0] = T.ceildiv(M, block_M)
            src_rank[0] = bx
            dst_rank[0] = by
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
                elif diff[0] < -T.floordiv(X, 2):
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
                elif diff[0] < -T.floordiv(Y, 2):
                    diff[0] += Y
                
                if diff[0] < 0:
                    # Send West (left): neighbor receives in its west buffer
                    to_dir[0] = Direction.WEST
                    next_rank[0] = rank_x[0] * Y + T.floormod(rank_y[0] + Y - 1, Y)
                else:
                    # Send East (right): neighbor receives in its east buffer
                    to_dir[0] = Direction.EAST
                    next_rank[0] = rank_x[0] * Y + T.floormod(rank_y[0] + 1, Y)

            # Phase 1: Initial send from src to the target neighbor
            if src_rank[0] == rank[0]:
                if dst_rank[0] != rank[0]:
                    T.put_block(
                        T.address_of(src[dst_rank[0] * M + bz * block_M, 0]),
                        T.address_of(buffer_transfer[rank[0], dst_rank[0], bz * block_M, 0]),
                        # T.address_of(dst[rank[0] * M + by * block_M, 0]),
                        block_M * N,
                        next_rank[0],
                    )
                    if tx == 0:
                        T.st(
                            signal_transfer[rank[0], dst_rank[0]],
                            rank[0],
                            scope=T.MemoryScope.SYSTEM,
                            sem=T.MemorySemantic.RELEASE,
                            dst_pe=next_rank[0]
                        )
                    T.sync_threads()
                else:
                    T.put_block(
                        T.address_of(src[dst_rank[0] * M + bz * block_M, 0]),
                        T.address_of(dst[rank[0] * M + bz * block_M, 0]),
                        block_M * N,
                        -1,
                    )

            T.barrier_blocks(barrier)
            
            # Phase 2: Each block handles one final dst data in one direction buffer of current rank and check whether to transfer
            # Signal values: represent the src_rank
            with T.While(global_finish[0] < PE_num):
                if tx == 0:
                    T.wait_le(signal_transfer[bx, dst_rank[0]], PE_num, scope=T.MemoryScope.SYSTEM)
                T.sync_threads()

                if signal_transfer[bx, dst_rank[0]] < PE_num:
                    # Only handle the transfer signal
                    if to_dir[0] != Direction.SELF:
                        T.put_block(
                            T.address_of(buffer_transfer[bx, dst_rank[0], bz * block_M, 0]),
                            T.address_of(buffer_transfer[bx, dst_rank[0], bz * block_M, 0]),
                            block_M * N,
                            next_rank[0],
                        )
                        if tx == 0:
                            T.st(
                                signal_transfer[bx, dst_rank[0]],
                                bx,
                                scope=T.MemoryScope.SYSTEM,
                                sem=T.MemorySemantic.RELEASE,
                                dst_pe=next_rank[0],
                            )
                        T.sync_threads()
                    else:
                        # Current rank is the real destination of this chunk of data, the real source rank is the buffer index
                        T.put_block(
                            T.address_of(buffer_transfer[bx, dst_rank[0], bz * block_M, 0]),
                            T.address_of(dst[bx * M + bz * block_M, 0]),
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
                            if old_local[0] + 2 == PE_num * num_block_M[0]:
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
                                        for src_rank in T.serial(PE_num):
                                            for dst_rank in T.serial(PE_num):
                                                T.st(
                                                    signal_transfer[src_rank, dst_rank],
                                                    PE_num,
                                                    scope=T.MemoryScope.SYSTEM,
                                                    sem=T.MemorySemantic.RELEASE,
                                                    dst_pe=remote_pe,
                                                )
                        T.sync_threads()

            T.barrier_blocks(barrier)

    return main

def run_torus_alltoall(local_rank, num_ranks, args):
    PE_num = args.PE_num
    X, Y = args.X, args.Y
    M, N = args.M, args.N
    block_M, block_N = M, N
    threads = 128

    local_rank, num_ranks, group_size = init_dist(local_rank, num_ranks)
    allocator = tilelang.get_allocator(
        size=2**32,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group_size,
    )
    
    kernel = torus_alltoall_xy(PE_num, X, Y, M, N, block_M, block_N, threads)
    kernel.initialize(allocator=allocator)
    
    src = tilelang.tensor((PE_num * M, N), torch.float16, allocator=allocator).random_()
    dst = tilelang.tensor((PE_num * M, N), torch.float16, allocator=allocator).zero_()
    
    buffer_transfer = tilelang.tensor((PE_num, PE_num, M, N), torch.float16, allocator=allocator).zero_()
    
    # Signals for each buffer slot in each direction
    signal_transfer = tilelang.tensor((PE_num, PE_num), torch.uint32, allocator=allocator).fill_(PE_num + 1)
    local_finish = tilelang.tensor((1), torch.uint32, allocator=allocator).fill_(0)
    global_finish = tilelang.tensor((1), torch.uint32, allocator=allocator).fill_(0)
    barrier = tilelang.tensor((PE_num), torch.int32, allocator=allocator).zero_()
    
    torch.cuda.synchronize()
    dist.barrier(group_size)
    
    kernel(src, dst, buffer_transfer, signal_transfer, local_finish, global_finish, barrier)

    torch.cuda.synchronize()
    dist.barrier(group_size)
        
    print(f"Rank {local_rank} TileLang AllToAll XY Routing Finished.")
    
    dst_ref = torch.zeros((PE_num * M, N), dtype=torch.float16, device="cuda")
    dist.all_to_all_single(dst_ref, src, group=group_size)
    torch.cuda.synchronize()

    if torch.allclose(dst, dst_ref, atol=1e-2, rtol=1e-2):
        print(f"Rank {local_rank} Verification Passed! ✅")
    else:
        max_diff = (dst - dst_ref).abs().max()
        print(f"Rank {local_rank} Verification Failed! ❌ Max diff: {max_diff}")

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--PE_num", type=int, default=8)
    parser.add_argument("--X", type=int, default=2)
    parser.add_argument("--Y", type=int, default=4)
    args = parser.parse_args()
    
    torch.multiprocessing.spawn(run_torus_alltoall, args=(args.PE_num, args), nprocs=args.PE_num)
