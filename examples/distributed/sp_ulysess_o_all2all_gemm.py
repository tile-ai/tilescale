import torch
from typing import Optional
import itertools
import tilelang
import tilelang.language as T
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing
from tilelang.distributed import init_dist, supports_p2p_native_atomic, cuda_stream_max_priority

class SpUlysessOAll2AllGemmKernel:

    def __init__(
        self,
        world_group: torch.distributed.ProcessGroup,
        nnodes: int,
        sp_size: int,
        max_batch: int,
        num_head: int,
        max_seqlen: int,
        head_dim: int,
        max_num_comm_buf: int,
        input_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        a2a_only: bool = True,
        fuse_sync: bool = True,
    ):
        self.world_group = world_group
        self.world_size = world_group.size()
        self.rank = world_group.rank()
        self.nnodes = nnodes
        assert self.world_size % nnodes == 0, f"world_size {self.world_size} must be divisible by nnodes {nnodes}"
        self.local_world_size = self.world_size // nnodes
        self.local_rank = self.rank % self.local_world_size
        self.sp_size = sp_size
        assert self.local_world_size % self.sp_size == 0, f"local_world_size {self.local_world_size} must be divisible by sp_size {sp_size}"
        self.sp_rank = self.local_rank % self.sp_size
        self.max_batch = max_batch
        self.num_head = num_head
        self.max_seqlen = max_seqlen
        self.head_dim = head_dim
        self.max_num_comm_buf = max_num_comm_buf
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.a2a_only = a2a_only
        assert self.a2a_only, "Only support a2a_only mode"
        self.fuse_sync = fuse_sync

        self.compute_stream = torch.cuda.Stream(priority=cuda_stream_max_priority())
        self.cp_event = torch.cuda.Event(enable_timing=False)
        self.ready_event = torch.cuda.Event(enable_timing=False)
        self.compute_event = torch.cuda.Event(enable_timing=False)

        self.p2p_atomic_supported = supports_p2p_native_atomic()
        self.max_sms = torch.cuda.get_device_properties("cuda").multi_processor_count

        # GEMM config
        self.BLOCK_SIZE_M = 128
        self.BLOCK_SIZE_N = 256
        self.BLOCK_SIZE_K = 64
        self.GROUP_SIZE_M = 4
        self.A2A_TILE_M = 128
        self.A2A_TILE_N = 256
        self.max_gemm_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        self.num_warps = 8
        self.num_stages = 3
        self.warp_specialize = False

        self.init_symm_buffer()
        self.init_local_buffer()

    def __del__(self):
        self.finalize()

    def finalize(self):
        self.deinit_symm_buffer()

    def init_symm_buffer(self):
        max_local_seq = self.max_seqlen // self.sp_size
        self._comm_output_buffer = nvshmem_create_tensor(
            [self.max_num_comm_buf, self.max_batch, max_local_seq, self.num_head * self.head_dim], self.input_dtype)
        self._barrier_buffer = nvshmem_create_tensor(
            [triton.cdiv(self.max_batch * self.max_seqlen, self.BLOCK_SIZE_M) * self.num_head], torch.int32)
        self._barrier_buffer.zero_()
        self._intra_node_sync_buffer = nvshmem_create_tensor([self.sp_size * self.max_sms], torch.int32)
        self._intra_node_sync_buffer.zero_()
        self._sp_group_sync_buffer = nvshmem_create_tensor([self.world_size], torch.int32)
        self._sp_group_sync_buffer.zero_()

    def deinit_symm_buffer(self):
        if hasattr(self, "_comm_output_buffer"):
            nvshmem_free_tensor_sync(self._comm_output_buffer)
            del self._comm_output_buffer
        if hasattr(self, "_barrier_buffer"):
            nvshmem_free_tensor_sync(self._barrier_buffer)
            del self._barrier_buffer
        if hasattr(self, "_intra_node_sync_buffer"):
            nvshmem_free_tensor_sync(self._intra_node_sync_buffer)
            del self._intra_node_sync_buffer
        if hasattr(self, "_sp_group_sync_buffer"):
            nvshmem_free_tensor_sync(self._sp_group_sync_buffer)
            del self._sp_group_sync_buffer

    def init_local_buffer(self):
        self._cum_seqlen_gpu = torch.empty([self.sp_size + 1], dtype=torch.int32, device="cuda")

    def sp_group_barrier_all_intra_node(self, stream=None):
        stream = torch.cuda.current_stream() if stream is None else stream
        sp_local_rank = self.local_rank % self.sp_size
        with torch.cuda.stream(stream):
            barrier_all_intra_node_atomic_cas_block[(1, )](sp_local_rank, self.rank, self.sp_size,
                                                           self._sp_group_sync_buffer)

    def reset_cusum_seq_lens(self, local_seqlen, seq_lens_cpu=None):
        if seq_lens_cpu is None:
            seq_lens_cpu = [local_seqlen] * self.sp_size
        else:
            seq_lens_cpu = seq_lens_cpu.tolist()
        assert local_seqlen == seq_lens_cpu[
            self.local_rank % self.
            sp_size], f"local_seqlen {local_seqlen} != seq_lens_cpu[{self.local_rank % self.sp_size}]={seq_lens_cpu[self.local_rank % self.sp_size]}"
        cum_seqlen_cpu = [0] + list(itertools.accumulate(seq_lens_cpu))
        self._cum_seq_len_cpu_tuple = tuple(cum_seqlen_cpu)

    def forward(self, inputs: torch.Tensor, weight: torch.Tensor, seq_lens_cpu: Optional[torch.Tensor] = None,
                bias: Optional[torch.Tensor] = None, output: Optional[torch.Tensor] = None,
                a2a_output: Optional[torch.Tensor] = None, transpose_weight: bool = False, num_comm_sms: int = -1,
                sm_margin: int = 0):
        if num_comm_sms == -1:
            num_comm_sms = self.world_size
        assert num_comm_sms >= 0, "num_comm_sms must be non-negative"
        assert len(weight.shape) == 2, f"weight must be 2D tensor, got {len(weight)}D"
        assert len(inputs.shape) == 4, f"inputs must be 4D tensor, got {len(inputs)}D"
        bs, total_seq_len, local_head, head_dim = inputs.shape
        assert head_dim == self.head_dim, f"head_dim {head_dim} must be equal to self.head_dim {self.head_dim}"
        assert weight.is_contiguous(), f"weight must be contiguous, got {weight.shape}"
        assert inputs.is_contiguous(), f"inputs must be contiguous, got {inputs.shape}"
        assert not transpose_weight, "transpose_weight is not supported in this kernel"

        if not transpose_weight:
            N = weight.shape[0]
            K = weight.shape[1]
        else:
            N = weight.shape[1]
            K = weight.shape[0]

        if seq_lens_cpu is not None:
            assert seq_lens_cpu.is_cpu, "seq_lens_cpu must be a CPU tensor"
            assert seq_lens_cpu.dtype == torch.int32, "seq_lens_cpu must be int32"
            assert seq_lens_cpu.is_contiguous(), "seq_lens_cpu must be contiguous"

            seq_lens_cpu_tuple = tuple(seq_lens_cpu.tolist())
            local_seq_len = seq_lens_cpu_tuple[self.sp_rank]
            M = local_seq_len * bs
        else:
            assert total_seq_len % self.sp_size == 0, f"total_seq_len {total_seq_len} must be divisible by sp_size {self.sp_size}"
            local_seq_len = total_seq_len // self.sp_size
            M = local_seq_len * bs

        self.reset_cusum_seq_lens(local_seqlen=local_seq_len, seq_lens_cpu=seq_lens_cpu)

        gemm_input_a = self._comm_output_buffer.view(-1)[:M * K].view([M, K])

        cur_stream = torch.cuda.current_stream()

        # self._barrier_buffer.zero_()
        # if not self.fuse_sync:
        #     self.sp_group_barrier_all_intra_node(cur_stream)

        # self.ready_event.record(cur_stream)
        # self.compute_stream.wait_event(self.ready_event)

        # grid = (num_comm_sms, )
        # kernel_all2all_push_intra_node_nvl[grid](
        #     inputs,
        #     gemm_input_a,
        #     self._cum_seq_len_cpu_tuple,
        #     self._cum_seqlen_gpu,
        #     self._barrier_buffer,
        #     self._intra_node_sync_buffer,  # no need to initialize
        #     local_head,
        #     local_head * self.sp_size,
        #     self.head_dim,
        #     self.sp_size,
        #     self.rank,
        #     self.sp_rank,
        #     self.A2A_TILE_M,
        #     self.A2A_TILE_N,
        #     self.GROUP_SIZE_M,
        #     num_comm_sms,
        #     self.fuse_sync,
        #     self.p2p_atomic_supported,
        #     VEC=(16 // inputs.dtype.itemsize),
        #     num_warps=32,
        # )

        # if output is None:
        #     output = torch.empty([bs, local_seq_len, N], device=inputs.device, dtype=self.output_dtype)

        # assert len(output.shape) == 3, f"output must be 4D tensor, got {len(output)}D"
        # assert output.shape[0] == bs, f"output batch size {output.shape[0]} must be equal to input batch size {bs}"
        # assert output.shape[
        #     1] == local_seq_len, f"output seq_len {output.shape[1]} must be equal to local_seq_len {local_seq_len}"
        # assert output.shape[2] == N, f"output head {output.shape[2]} must be equal to output size {N}"
        # assert output.is_contiguous(), f"output must be contiguous, got {output.shape}"

        # assert self.max_gemm_sms - num_comm_sms - sm_margin > 0, f"max_gemm_sms {self.max_gemm_sms} - num_comm_sms {num_comm_sms} - sm_margin {sm_margin} must be greater than 0"
        # gemm_config = triton.Config(
        #     {
        #         'BLOCK_SIZE_M': self.BLOCK_SIZE_M, 'BLOCK_SIZE_N': self.BLOCK_SIZE_N, 'BLOCK_SIZE_K': self.BLOCK_SIZE_K,
        #         'GROUP_SIZE_M': self.GROUP_SIZE_M, 'A2A_TILE_M': self.A2A_TILE_M, 'A2A_TILE_N': self.A2A_TILE_N,
        #         'NUM_GEMM_SMS': self.max_gemm_sms - num_comm_sms - sm_margin
        #     }, num_stages=self.num_stages, num_warps=self.num_warps)

        # with torch.cuda.stream(self.compute_stream):
        #     matmul_descriptor_persistent(self.sp_rank, self.sp_size, gemm_input_a, weight, bias, output,
        #                                  self._barrier_buffer, gemm_config, self.warp_specialize)

        # if a2a_output is not None:
        #     assert a2a_output.shape == (
        #         bs, local_seq_len, local_head * self.sp_size, head_dim
        #     ), f"a2a_output shape {a2a_output.shape} must be equal to (bs, local_seq_len, local_head * self.sp_size, head_dim) ({bs}, {local_seq_len}, {local_head * self.sp_size}, {head_dim})"
        #     assert a2a_output.is_contiguous(), f"a2a_output must be contiguous, got {a2a_output.shape}"
        #     a2a_output.copy_(gemm_input_a.view(bs, local_seq_len, local_head * self.sp_size * head_dim))
        #     ret = (output, a2a_output)
        # else:
        #     ret = (output, )

        # self.compute_event.record(self.compute_stream)
        # cur_stream.wait_event(self.compute_event)

        # return ret

    def post_attn_a2a(
        self,
        inputs: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor] = None,
        return_comm_buf: bool = False,
        comm_buf_idx: int = 0,
        num_comm_sms: int = -1,
    ):
        if num_comm_sms == -1:
            num_comm_sms = self.world_size
        assert num_comm_sms >= 0, "num_comm_sms must be non-negative"
        assert len(inputs.shape) == 4, f"inputs must be 4D tensor, got {len(inputs)}D"
        bs, total_seq_len, local_head, head_dim = inputs.shape
        assert head_dim == self.head_dim, f"head_dim {head_dim} must be equal to self.head_dim {self.head_dim}"
        assert inputs.is_contiguous(), f"inputs must be contiguous, got {inputs.shape}"

        if seq_lens_cpu is not None:
            assert seq_lens_cpu.is_cpu, "seq_lens_cpu must be a CPU tensor"
            assert seq_lens_cpu.dtype == torch.int32, "seq_lens_cpu must be int32"
            assert seq_lens_cpu.is_contiguous(), "seq_lens_cpu must be contiguous"

            seq_lens_cpu_tuple = tuple(seq_lens_cpu.tolist())
            local_seq_len = seq_lens_cpu_tuple[self.sp_rank]
            M = local_seq_len * bs
        else:
            assert total_seq_len % self.sp_size == 0, f"total_seq_len {total_seq_len} must be divisible by sp_size {self.sp_size}"
            local_seq_len = total_seq_len // self.sp_size
            M = local_seq_len * bs

        K = local_head * self.sp_size * head_dim

        self.reset_cusum_seq_lens(local_seqlen=local_seq_len, seq_lens_cpu=seq_lens_cpu)

        assert comm_buf_idx < self.max_num_comm_buf, f"comm_buf_idx {comm_buf_idx} must be less than num_comm_buf {self.max_num_comm_buf}"
        gemm_input_a = self._comm_output_buffer[comm_buf_idx].view(-1)[:M * K].view([M, K])

        cur_stream = torch.cuda.current_stream()

        # if not self.fuse_sync:
        #     self.sp_group_barrier_all_intra_node(cur_stream)

        # grid = (self.max_gemm_sms, )
        # kernel_all2all_push_intra_node_nvl[grid](
        #     inputs,
        #     gemm_input_a,
        #     self._cum_seq_len_cpu_tuple,
        #     self._cum_seqlen_gpu,
        #     self._barrier_buffer,
        #     self._intra_node_sync_buffer,  # no need to initialize
        #     local_head,
        #     local_head * self.sp_size,
        #     self.head_dim,
        #     self.sp_size,
        #     self.rank,
        #     self.sp_rank,
        #     256,
        #     256,
        #     16,
        #     num_comm_sms,
        #     self.fuse_sync,
        #     self.p2p_atomic_supported,
        #     VEC=(16 // inputs.dtype.itemsize),
        #     SKIP_BARRIER=True,
        #     num_warps=32,
        # )

        # if return_comm_buf:
        #     return gemm_input_a
        # else:
        #     self.sp_group_barrier_all_intra_node(cur_stream)
        #     return gemm_input_a.clone()

    def post_attn_a2a_no_cpy(
        self,
        inputs: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor] = None,
        comm_buf_idx: int = 0,
        num_comm_sms: int = -1,
    ):
        return self.post_attn_a2a(
            inputs,
            seq_lens_cpu,
            return_comm_buf=True,
            comm_buf_idx=comm_buf_idx,
            num_comm_sms=num_comm_sms,
        )

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "fp8e4m3": torch.float8_e4m3fn,
    "fp8e5m2": torch.float8_e5m2,
    "s8": torch.int8,
    "s32": torch.int32,
}

def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    dtype = DTYPE_MAP[args.dtype]
    
    gemm_a2a_op = SpUlysessOAll2AllGemmKernel(
            world_group=group,
            nnodes=1, # only support intra node for now
            sp_size=args.sp_size,
            max_batch=args.bs,
            num_head=args.nh,
            max_seqlen=args.seq_len,
            head_dim=args.hd,
            max_num_comm_buf=3,
            input_dtype=dtype,
            output_dtype=dtype,
            a2a_only=True,
            fuse_sync=args.fuse_sync,
        )
    
    # dtype = torch.float16
    # M = args.M if args else 8192
    # N = args.N if args else 8192
    # K = args.K if args else 8192
    # M_per_rank = M // num_local_ranks
    # N_per_rank = N // num_local_ranks

    # BLOCK_M = 128
    # BLOCK_N = 128
    # BLOCK_K = 64
    # threads = 256

    # allocator = tilelang.get_allocator(
    #     size=2**30,
    #     device="cuda",
    #     is_distributed=True,
    #     local_rank=local_rank,
    #     num_local_ranks=num_local_ranks,
    #     group=group)
    # kernel = tilelang.compile(gemm_kernel(M, N, K, num_ranks, BLOCK_M, BLOCK_N, BLOCK_K, threads))
    # kernel.initialize(allocator=allocator)
    # if local_rank == 0:
    #     print(kernel.get_kernel_source())

    # A = tilelang.tensor((M_per_rank, K), dtype, allocator=allocator).normal_()
    # B = tilelang.tensor((K, N_per_rank), dtype, allocator=allocator).normal_()
    # C = tilelang.tensor((M, N_per_rank), dtype, allocator=allocator)
    # ag_buffer = tilelang.tensor((M, K), dtype, allocator=allocator, return_peers=True)
    # signal_buffer = tilelang.tensor((num_local_ranks,),
    #                                 torch.int32,
    #                                 allocator=allocator,
    #                                 return_peers=True)
    # signal_buffer[rank].fill_(0)
    # ag_buffer[rank][rank * M_per_rank:(rank + 1) * M_per_rank, :].copy_(A)

    # dist.barrier(group)

    # ag_stream = torch.cuda.Stream()
    # signal_target = 1

    # tilelang_C = ag_gemm_op(A, B, C, ag_buffer, signal_buffer, M_per_rank, K, signal_target, rank,
    #                         group, num_local_ranks, num_local_ranks, kernel, ag_stream)

    # torch_ag_buffer = torch.empty([M, K], dtype=dtype, device="cuda")
    # torch_C = torch_ag_gemm(group, A, B, torch_ag_buffer)

    # if torch.allclose(torch_C, tilelang_C, atol=1e-6, rtol=1e-6):
    #     print(f"rank {local_rank} check passed.✅")
    # else:
    #     print(f"rank {local_rank} check failed.❌")
    #     print(f"torch_C: {torch_C}, tilelang_C: {tilelang_C}")
    #     raise ValueError("Test failed")

    # dist.destroy_process_group()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-processes', type=int, default=2, help='Number of processes to spawn (default: 2)')
    parser.add_argument("bs", type=int)
    parser.add_argument("nh", type=int)
    parser.add_argument("seq_len", type=int)
    parser.add_argument("hd", type=int)
    parser.add_argument("out_features", type=int)
    parser.add_argument("--num_comm_sm", type=int, required=True, help="num sm for a2a")
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--sm_margin", default=0, type=int, help="sm margin")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--has_bias", default=False, action="store_true", help="whether have bias")
    parser.add_argument("--copy_a2a_output", default=False, action="store_true", help="whether to copy a2a output")
    parser.add_argument(
        "--fastacc",
        default=False,
        action="store_true",
        help="whether to use fast accumulation (FP8 Gemm only)",
    )
    parser.add_argument(
        "--transpose_weight",
        dest="transpose_weight",
        action=argparse.BooleanOptionalAction,
        help="transpose weight",
        default=False,
    )
    parser.add_argument("--fuse_sync", default=False, action="store_true", help="fuse sync into all2all kernel")
    parser.add_argument(
        "--verify",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="run once to verify correctness",
    )
    parser.add_argument(
        "--save_gemm_input",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="save gemm input",
    )
    parser.add_argument("--dp", default=False, action="store_true", help="dp per rank")
    parser.add_argument("--sp_size", default=0, type=int, help="sp size")
    parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
    args = parser.parse_args()
    num_processes = args.num_processes

    torch.multiprocessing.spawn(main, args=(num_processes, args), nprocs=num_processes)
