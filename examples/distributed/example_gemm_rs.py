import torch
import argparse
import os
import tilelang
from tilelang.distributed.utils import init_distributed, generate_data
from gemm_rs_kernel import gemm_rs
from gemm_rs_utils import create_gemm_rs_context
from typing import Optional

tilelang.disable_cache()


def torch_gemm_rs(
    input: torch.Tensor,  # [M, local_k]
    weight: torch.Tensor,  # [N, local_K]
    bias: Optional[torch.Tensor],
    TP_GROUP,
):
    M, local_K = input.shape
    N = weight.shape[0]
    output = torch.matmul(input, weight.T)
    if bias:
        output = output + bias
    rs_output = torch.empty((M // WORLD_SIZE, N), dtype=output.dtype, device=input.device)
    torch.distributed.reduce_scatter_tensor(rs_output, output, group=TP_GROUP)
    return rs_output


class GemmRS(torch.nn.Module):

    def __init__(
        self,
        tp_group: torch.distributed.ProcessGroup,
        max_M: int,
        N: int,
        K: int,
        input_dtype: torch.dtype,
        output_dtype: torch.dtype,
        local_world_size: int = -1,
    ):
        super().__init__()
        self.tp_group = tp_group
        self.rank: int = tp_group.rank()
        self.world_size = tp_group.size()
        self.local_world_size = local_world_size if local_world_size != -1 else self.world_size
        self.local_rank = self.rank % self.local_world_size

        self.max_M: int = max_M
        self.N = N
        self.K = K
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype

        self.rs_stream: torch.cuda.Stream = torch.cuda.Stream(priority=-1)

        self.ctx = create_gemm_rs_context(max_M, N, self.rank, self.world_size,
                                          self.local_world_size, output_dtype, self.rs_stream)

    def forward(
        self,
        input: torch.Tensor,  # [M, local_K]
        weight: torch.Tensor,  # [N, local_K]
        bias: Optional[torch.Tensor] = None,
        persistent: bool = True,
    ):
        assert input.shape[0] <= self.max_M and weight.shape[0] == self.N

        return gemm_rs(input, weight, self.ctx, persistent)


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "s8": torch.int8,
    "s32": torch.int32,
}

THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 6e-2,
    torch.float8_e4m3fn: 1e-2,
    torch.float8_e5m2: 1e-2,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--warmup", default=20, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="float16", type=str, help="data type")

    parser.add_argument(
        "--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--check", default=False, action="store_true", help="correctness check")
    parser.add_argument("--verify-iters", default=1, type=int)
    parser.add_argument(
        "--persistent",
        action=argparse.BooleanOptionalAction,
        default=torch.cuda.get_device_capability() >= (9, 0))

    parser.add_argument(
        "--transpose_weight",
        dest="transpose_weight",
        action=argparse.BooleanOptionalAction,
        help="transpose weight",
        default=True,
    )
    parser.add_argument("--has_bias", default=False, action="store_true", help="whether have bias")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    # init
    args = parse_args()

    # TODO: remove this
    args.persistent = False

    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    WORLD_SIZE, RANK, LOCAL_RANK, TP_GROUP = init_distributed(return_tp_group=True)

    input_dtype = DTYPE_MAP[args.dtype]
    output_dtype = input_dtype
    atol = THRESHOLD_MAP[output_dtype]
    rtol = THRESHOLD_MAP[output_dtype]

    assert args.M % TP_GROUP.size() == 0
    assert args.K % TP_GROUP.size() == 0
    local_K = args.K // TP_GROUP.size()

    scale = TP_GROUP.rank() + 1

    def _make_data(M):
        data_config = [
            ((M, local_K), input_dtype, (0.01 * scale, 0)),  # A
            ((args.N, local_K), input_dtype, (0.01 * scale, 0)),  # B
            (  # bias
                None if not args.has_bias else ((M, args.N), input_dtype, (1, 0))),
        ]
        generator = generate_data(data_config)
        input, weight, bias = next(generator)
        return input, weight, bias

    gemm_rs_op = GemmRS(TP_GROUP, args.M, args.N, args.K, input_dtype, output_dtype,
                        LOCAL_WORLD_SIZE)

    torch.cuda.empty_cache()
    input_list = [
        # _make_data(random.randint(1, args.M // WORLD_SIZE) * WORLD_SIZE) for _ in range(args.verify_iters)
        _make_data(args.M) for _ in range(args.verify_iters)
    ]
    dist_out_list, torch_out_list = [], []

    # torch impl
    for input, weight, bias in input_list:
        torch_out = torch_gemm_rs(
            input,
            weight,
            bias,
            TP_GROUP,
        )
        torch_out_list.append(torch_out)

    # dist triton impl
    for input, weight, bias in input_list:
        dist_out = gemm_rs_op.forward(input, weight, bias, args.persistent)
        dist_out_list.append(dist_out)
    # verify
    # for idx, (torch_out, dist_out) in enumerate(zip(torch_out_list, dist_out_list)):
    #     assert_allclose(torch_out, dist_out, atol=atol, rtol=rtol, verbose=False)

    # print(f"RANK[{RANK}]: pass.")
    # exit(0)
