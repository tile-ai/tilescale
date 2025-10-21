import os
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
import torch
import torch.distributed as dist
import torch.multiprocessing
import argparse
from tilelang.distributed import init_dist

tilelang.disable_cache()

@tilelang.jit(
    pass_configs={
        "tl.disable_warp_specialized": True,
        "tl.disable_tma_lower": True
    })
def copy_and_barrier_all_intra_node_kernel(local_rank,
                                            rank,
                                            num_ranks,
                                            M,
                                            K,
                                            block_M,
                                            block_K,
                                            threads,
                                            dtype="float16"):

    M_per_rank = T.ceildiv(M, num_ranks)
    sm_num = driver.get_num_sms()
    m_blocks = T.ceildiv(M_per_rank, block_M)
    k_blocks = T.ceildiv(K, block_K)
    waves = T.ceildiv(m_blocks * k_blocks, sm_num)
    
    @T.macro
    def copy_kernel(
            src: T.Tensor((M_per_rank, K), dtype),
            dst: T.Tensor((M, K), dtype),
            data_shared: T.Tensor((block_M, block_K), dtype),
            block_id
    ):
        for w in T.serial(waves):
            tile_id = sm_num * w + block_id
            bx = tile_id % m_blocks
            by = tile_id // m_blocks
            
            if by < k_blocks:
                T.copy(src[bx * block_M, by * block_K], data_shared)
                T.copy(data_shared, dst[rank * M_per_rank + bx * block_M, by * block_K])
    
    
    @T.macro
    def barrier_all_intra_node_non_atomic(
        sync_buffer: T.Tensor((3 * num_ranks), "uint32"),
        block_id
    ):
        if block_id == 0:
            # _barrier_all_intra_node_non_atomic_once_block(local_rank, rank, num_ranks, sync_buffer, target_value)
            T.barrier_all_blocks_sys(sync_buffer)
            
        # barrier all CTAs
        T.sync_grids(sync_buffer[2 * num_ranks])

        # if block_id == 0:
        #     # _barrier_all_intra_node_non_atomic_once_block(local_rank, rank, num_ranks, sync_buffer, target_value)
        #     T.barrier_all_blocks_sys(sync_buffer[num_ranks])
            
        # # barrier all CTAs
        # T.sync_grids(sync_buffer[2 * num_ranks])
        
    
    @T.prim_func
    def local_copy(
            A: T.Tensor((M_per_rank, K), dtype),
            ag_buffer: T.Tensor((M, K), dtype),
            signal_buffer: T.Tensor((num_ranks), "uint32"),
            sync_buffer: T.Tensor((3 * num_ranks), "uint32"),
            # clk_beg: T.Tensor((sm_num), "uint64"),
            # clk_end: T.Tensor((sm_num), "uint64"),
  
    ):
        with T.Kernel(sm_num, threads=threads) as (block_id):
            data_shared = T.alloc_shared((block_M, block_K), dtype)
            clk_beg_local = T.alloc_local((1), "uint64")
            clk_end_local = T.alloc_local((1), "uint64")
            
            tid = T.get_thread_binding(0)
            
            T.annotate_layout({data_shared: tilelang.layout.make_swizzled_layout(data_shared)})
            
            # if tid == 0:
            #     clk_beg_local[0] = T.get_clock()
            # This synchonization cost approximately 150000 cycles (0.1 ms)
            barrier_all_intra_node_non_atomic(sync_buffer, block_id)
            # if tid == 0:
            #     clk_end_local[0] = T.get_clock()
            copy_kernel(A, ag_buffer, data_shared, block_id)
            tx = T.get_thread_binding(0)
            if block_id == 0 and tx < num_ranks:  # set symm barrier
                if tx == rank:
                    signal_buffer[tx] = 1
                else:
                    signal_buffer[tx] = 0
            if tid == 0:
                clk_beg_local[0] = T.get_clock()
            barrier_all_intra_node_non_atomic(sync_buffer, block_id)
            if tid == 0:
                clk_end_local[0] = T.get_clock()
            # if tid == 0:
            #     clk_beg[block_id] = clk_beg_local[0]
            #     clk_end[block_id] = clk_end_local[0]
        

    return local_copy


def main(local_rank: int, num_ranks: int, args: argparse.Namespace):
    dtype = torch.float16
    M = args.M if args else 8192
    K = args.K if args else 8192
    M_per_rank = M // num_ranks
    
    BLOCK_M = 64
    BLOCK_K = 64
    threads = 128

    rank, num_ranks, group = init_dist(local_rank, num_ranks)
    allocator = tilelang.get_allocator(
        size=2**30,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group)
    kernel = copy_and_barrier_all_intra_node_kernel(
        local_rank=local_rank,
        rank=local_rank,
        num_ranks=num_ranks,
        M=M,
        K=K,
        block_M=BLOCK_M,
        block_K=BLOCK_K,
        threads=threads,
    )
    kernel.initialize(allocator=allocator)
    if local_rank == 0:
        print(kernel.get_kernel_source())
    
    M_per_rank = M // num_ranks

    A = tilelang.tensor((M_per_rank, K), dtype, allocator=allocator).normal_()
    ag_buffer = tilelang.tensor((M, K), dtype, allocator=allocator)
    signal_buffer = tilelang.tensor((num_ranks,),
                                    torch.uint32,
                                    allocator=allocator)
    sync_buffer = tilelang.tensor((3 * num_ranks,),
                                    torch.uint32,
                                    allocator=allocator)

    kernel(A, ag_buffer, signal_buffer, sync_buffer)
    
    ref_ag = torch.zeros((M, K), dtype=dtype).cuda()
    ref_ag[rank * M_per_rank:(rank + 1) * M_per_rank, :].copy_(A)
    
    if torch.allclose(ref_ag, ag_buffer, atol=1e-6, rtol=1e-6):
        print(f"rank {local_rank} check passed.✅")
    else:
        print(f"rank {local_rank} check failed.❌")
        print(f"ref_ag: {ref_ag}, ag_buffer: {ag_buffer}")
        # raise ValueError("Test failed")

    dist.destroy_process_group()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-processes', type=int, default=2, help='Number of processes to spawn (default: 2)')
    parser.add_argument('--M', type=int, default=4096, help='M dimension')
    parser.add_argument('--K', type=int, default=4096, help='K dimension')
    args = parser.parse_args()
    num_processes = args.num_processes

    torch.multiprocessing.spawn(main, args=(num_processes, args), nprocs=num_processes)
