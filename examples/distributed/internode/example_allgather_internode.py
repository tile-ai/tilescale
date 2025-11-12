# NODES=2 NODE_RANK=0 MASTER_IP=ip0 bash tilelang/distributed/launch.sh ./examples/distributed/internode/example_allgather_internode.py
# NODES=2 NODE_RANK=1 MASTER_IP=ip0 bash tilelang/distributed/launch.sh ./examples/distributed/internode/example_allgather_internode.py

# todo: add benchmark and impl for wait_eq u64, also stricter test

import os
import tilelang
import tilelang.language as T
import argparse
import torch
import torch.distributed as dist
from tilelang.distributed import init_distributed, dtype_map, perf_fn, wait_eq
import pynvshmem
from dataclasses import dataclass, field

from cuda import cudart, cuda


os.environ['NCCL_DEBUG'] = 'WARN'  # silence NCCL log

@dataclass
class AllGatherInternodeContext:
    # tensor info
    M_per_rank: int
    M: int = field(init=False)
    N: int
    dtype: str
    torch_dtype: torch.dtype = field(init=False)

    # rank info
    rank: int
    num_local_ranks: int
    num_ranks: int
    local_rank: int = field(init=False)
    num_nodes: int = field(init=False)
    node_rank: int = field(init=False)

    # workspace
    barriers: list[torch.Tensor] = field(init=False)
    barrier: torch.Tensor = field(init=False)
    internode_comm_bufs: list[torch.Tensor] = field(init=False)
    # internode_comm_buf: torch.Tensor = field(init=False)

    # streams
    internode_stream: torch.cuda.Stream = field(init=False)
    intranode_stream: torch.cuda.Stream = field(init=False)

    def __post_init__(self):
        self.M = self.M_per_rank * self.num_ranks
        self.local_rank = self.rank % self.num_local_ranks
        self.num_nodes = self.num_ranks // self.num_local_ranks
        self.node_rank = self.rank // self.num_local_ranks
        self.torch_dtype = dtype_map[self.dtype]

        self.create_workspace()

        self.internode_stream = torch.cuda.Stream()
        self.intranode_stream = torch.cuda.Stream()

        pynvshmem.nvshmem_barrier_all()
        torch.cuda.synchronize()

    def create_workspace(self):
        self.barriers = pynvshmem.nvshmem_create_tensor_list_intra_node([self.num_nodes,], torch.uint64)
        self.barrier = self.barriers[self.local_rank]
        self.barrier.fill_(0)


@tilelang.jit
def put_internode_kernel(
    num_nodes: int,
    num_local_ranks: int,
    M_per_rank: int,
    M: int,
    N: int,
    dtype: str,
    threads: int = 256
):

    @T.prim_func
    def main(
        dst: T.Tensor([M, N], "int32"),  # type: ignore
        barrier: T.Tensor([num_nodes], "uint64"),  # type: ignore
    ):
        with T.Kernel(num_nodes-1, threads=threads) as (bx):
            rank = T.get_pe()
            node_rank = rank // num_local_ranks
            peer = (rank+ (bx+1) * num_local_ranks) % (num_nodes * num_local_ranks)
            T.putmem_signal_nbi_block(
                T.address_of(dst[rank * M_per_rank, 0]),
                T.address_of(dst[rank * M_per_rank, 0]), 
                M_per_rank * N * dtype_map[dtype].itemsize,
                T.address_of(barrier[node_rank]),
                1,
                T.Amo.SIGNAL_SET,
                peer
            )
    return main


def tl_allgather_internode(
    src: torch.Tensor,
    dst: list[torch.Tensor],
    ctx: AllGatherInternodeContext,
    debug: bool = False,
):
    # 0. local copy and barrier
    cudart.cudaMemcpy(
        dst[ctx.local_rank][ctx.rank * ctx.M_per_rank, 0].data_ptr(),
        src.data_ptr(),
        ctx.M_per_rank * ctx.N * ctx.torch_dtype.itemsize,
        cudart.cudaMemcpyKind.cudaMemcpyDefault
    )
    pynvshmem.nvshmem_barrier_all()
    dist.barrier()
    torch.cuda.synchronize()
    
    # 1. perform inter-node comm
    # push to all peers with same local rank and signal on barrier
    with torch.cuda.stream(ctx.internode_stream):
        kernel = put_internode_kernel(ctx.num_nodes, ctx.num_local_ranks, ctx.M_per_rank, ctx.M, ctx.N, ctx.dtype)
        if debug and ctx.rank == 0:
            print(kernel.get_kernel_source())
        kernel(dst[ctx.local_rank], ctx.barrier)

    with torch.cuda.stream(ctx.intranode_stream):
        # 2. perform intra-node cp-engine based gather to overlap with inter-node comm
        for i in range(ctx.num_local_ranks-1):
            tgt_local_rank = (ctx.local_rank + i + 1) % ctx.num_local_ranks
            tgt_rank = tgt_local_rank + ctx.node_rank * ctx.num_local_ranks
            cudart.cudaMemcpyAsync(
                dst[ctx.local_rank][tgt_rank * ctx.M_per_rank, 0].data_ptr(),
                dst[tgt_local_rank][tgt_rank * ctx.M_per_rank, 0].data_ptr(),
                ctx.M_per_rank * ctx.N * ctx.torch_dtype.itemsize,
                cudart.cudaMemcpyKind.cudaMemcpyDefault,
                ctx.intranode_stream.cuda_stream
            )

        # 3. wait for data from other nodes sent to intra-node peers and gather
        for i in range(ctx.num_nodes-1):
            tgt_node_rank = (ctx.node_rank + i + 1) % ctx.num_nodes
            for tgt_local_rank in range (ctx.num_local_ranks):
                tgt_rank = tgt_local_rank + tgt_node_rank * ctx.num_local_ranks
                cuda.cuStreamWaitValue64(
                    ctx.intranode_stream.cuda_stream,
                    ctx.barriers[tgt_local_rank][tgt_node_rank].data_ptr(),
                    1,
                    cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
                )
                cudart.cudaMemcpyAsync(
                    dst[ctx.local_rank][tgt_rank * ctx.M_per_rank, 0].data_ptr(),
                    dst[tgt_local_rank][tgt_rank * ctx.M_per_rank, 0].data_ptr(),
                    ctx.M_per_rank * ctx.N * ctx.torch_dtype.itemsize,
                    cudart.cudaMemcpyKind.cudaMemcpyDefault,
                    ctx.intranode_stream.cuda_stream
                )
                
    ctx.intranode_stream.wait_stream(ctx.internode_stream)


def main(M_per_rank: int, N: int, dtype: str, debug: bool = False):
    WORLD_SIZE, RANK, LOCAL_RANK, TP_GROUP, LC_GROUP = init_distributed(
        return_tp_group=True, return_lc_group=True)
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE'))
    assert WORLD_SIZE % local_world_size == 0
    nodes: int = WORLD_SIZE // local_world_size
    assert nodes >= 2, "This example is for inter-node allgather"
    node_rank = RANK // local_world_size
    
    # gather WORLD_SIZE*[M_per_rank, N]->[M, N]
    if debug:
        dtype = 'int32'
        torch_dtype = torch.int32
        src = torch.full([M_per_rank, N], RANK, dtype=torch.int32, device='cuda')
        dst = pynvshmem.nvshmem_create_tensor_list_intra_node([M_per_rank * WORLD_SIZE, N], torch.int32)
        dst[LOCAL_RANK].fill_(-1)
    else:
        torch_dtype = dtype_map[dtype]
        src = torch.randn([M_per_rank, N], dtype=torch_dtype, device='cuda')
        dst = pynvshmem.nvshmem_create_tensor_node([M_per_rank * WORLD_SIZE, N], torch_dtype)
    ctx = AllGatherInternodeContext(M_per_rank, N, "int32", RANK, local_world_size, WORLD_SIZE)

    pynvshmem.nvshmem_barrier_all()
    dist.barrier(TP_GROUP)
    tl_allgather_internode(src, dst, ctx, debug)
    pynvshmem.nvshmem_barrier_all()
    dist.barrier(TP_GROUP)
    
    if debug:
        print(dst[LOCAL_RANK])

    # torch ref
    ref_dst = torch.empty_like(dst[LOCAL_RANK])
    dist.barrier(TP_GROUP)
    dist.all_gather_into_tensor(ref_dst, src, TP_GROUP)
    dist.barrier(TP_GROUP)
    assert torch.allclose(dst[LOCAL_RANK], ref_dst)
    print(f'[node={node_rank}, local_rank={LOCAL_RANK}] All check passed.✅')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--M_per_rank', type=int, default=1024, help='Number of rows of the local tensor')
    parser.add_argument('--N', type=int, default=1024, help='Number of columns of the local tensor')
    parser.add_argument('--dtype', type=str, default='float32', help='Data type')
    parser.add_argument('-debug', action='store_true', default=False, help='Enable debug mode')
    args = parser.parse_args()

    main(args.M_per_rank, args.N, args.dtype, args.debug)
    