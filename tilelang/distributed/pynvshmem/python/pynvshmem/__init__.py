import sys
import torch
import torch.distributed
from cuda import cuda, cudart
from typing import Optional

try:
    from _pynvshmem import *  # noqa: F403
except Exception as e:
    print(
        "please add NVSHMEM library path to LD_LIBRARY_PATH and try again",
        flush=True,
        file=sys.stderr,
    )
    raise e


def _CUDA_CHECK(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {err}: {cuda.cuGetErrorName(err)}")
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cuda Error: {err}: {cudart.cudaGetErrorString(err)}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


def broadcast_cpu(tensor: torch.Tensor, src: int, group: torch.distributed.ProcessGroup):
    if not tensor.is_cuda:
        tensor_gpu = tensor.cuda()
        torch.distributed.broadcast(tensor_gpu, src=src, group=group)
        tensor.copy_(tensor_gpu)
    else:
        torch.distributed.broadcast(tensor, src=src, group=group)
    torch.cuda.synchronize()


def init_nvshmem_by_uniqueid(group: torch.distributed.ProcessGroup):
    rank, nranks = group.rank(), group.size()
    if rank == 0:
        unique_id: bytes = nvshmemx_get_uniqueid()  # noqa: F405
        unique_id = torch.frombuffer(unique_id, dtype=torch.uint8).clone()
    else:
        unique_id = torch.empty(128, dtype=torch.uint8)

    broadcast_cpu(tensor=unique_id, group=group, src=0)

    unique_id = unique_id.numpy().tobytes()
    nvshmemx_init_attr_with_uniqueid(rank, nranks, unique_id)  # noqa: F405
    nvshmem_barrier_all()
    torch.cuda.synchronize()


"""Host-side signaling functions."""


def write_i32(tensor: torch.Tensor, value: int, stream: Optional[torch.cuda.Stream] = None):
    """Atomic write an int32 value to a tensor.
    Args:
        tensor (torch.Tensor): The tensor to write to, must be of dtype torch.int32.
        value (int): The value to write.
        stream (Optional[torch.cuda.Stream]): The CUDA stream to use for the operation.
            If None, the current stream will be used.
    """
    assert isinstance(tensor, torch.Tensor) and tensor.dtype == torch.int32, \
        f"tensor must be a torch.Tensor with dtype torch.int32, but got {tensor.dtype}"
    assert tensor.numel() == 1, "tensor must have exactly one element"
    if stream is None:
        stream = torch.cuda.current_stream()
    (err,) = cuda.cuStreamWriteValue32(
        stream.cuda_stream,
        tensor.data_ptr(),
        value,
        cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
    )
    _CUDA_CHECK(err)


def write_u64(tensor: torch.Tensor, value: int, stream: Optional[torch.cuda.Stream] = None):
    """Atomic write an uint64 value to a tensor.
    Args:
        tensor (torch.Tensor): The tensor to write to, must be of dtype torch.uint64.
        value (int): The value to write.
        stream (Optional[torch.cuda.Stream]): The CUDA stream to use for the operation.
            If None, the current stream will be used.
    """
    assert isinstance(tensor, torch.Tensor) and tensor.dtype == torch.uint64, \
        f"tensor must be a torch.Tensor with dtype torch.uint64, but got {tensor.dtype}"
    assert tensor.numel() == 1, "tensor must have exactly one element"
    if stream is None:
        stream = torch.cuda.current_stream()
    (err,) = cuda.cuStreamWriteValue64(
        stream.cuda_stream,
        tensor.data_ptr(),
        value,
        cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
    )
    _CUDA_CHECK(err)


# team node
NVSHMEM_TEAM_INVALID = -1
NVSHMEM_TEAM_WORLD = 0
NVSHMEM_TEAM_WORLD_INDEX = 0
NVSHMEM_TEAM_SHARED = 1
NVSHMEM_TEAM_SHARED_INDEX = 1
NVSHMEMX_TEAM_NODE = 2
NVSHMEM_TEAM_NODE_INDEX = 2
NVSHMEMX_TEAM_SAME_MYPE_NODE = 3
NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX = 3
NVSHMEMI_TEAM_SAME_GPU = 4
NVSHMEM_TEAM_SAME_GPU_INDEX = 4
NVSHMEMI_TEAM_GPU_LEADERS = 5
NVSHMEM_TEAM_GPU_LEADERS_INDEX = 5
NVSHMEM_TEAMS_MIN = 6
NVSHMEM_TEAM_INDEX_MAX = sys.maxsize

# class nvshmemi_cmp_type(Enum):
NVSHMEM_CMP_EQ = 0
NVSHMEM_CMP_NE = 1
NVSHMEM_CMP_GT = 2
NVSHMEM_CMP_LE = 3
NVSHMEM_CMP_LT = 4
NVSHMEM_CMP_GE = 5
NVSHMEM_CMP_SENTINEL = sys.maxsize

# class nvshmemi_amo_t(Enum):
NVSHMEMI_AMO_ACK = 1
NVSHMEMI_AMO_INC = 2
NVSHMEMI_AMO_SET = 3
NVSHMEMI_AMO_ADD = 4
NVSHMEMI_AMO_AND = 5
NVSHMEMI_AMO_OR = 6
NVSHMEMI_AMO_XOR = 7
NVSHMEMI_AMO_SIGNAL = 8
NVSHMEM_SIGNAL_SET = 9
NVSHMEM_SIGNAL_ADD = 10
NVSHMEMI_AMO_SIGNAL_SET = NVSHMEM_SIGNAL_SET  # Note - NVSHMEM_SIGNAL_SET == 9
NVSHMEMI_AMO_SIGNAL_ADD = NVSHMEM_SIGNAL_ADD  # Note - NVSHMEM_SIGNAL_ADD == 10
NVSHMEMI_AMO_END_OF_NONFETCH = 11  # end of nonfetch atomics
NVSHMEMI_AMO_FETCH = 12
NVSHMEMI_AMO_FETCH_INC = 13
NVSHMEMI_AMO_FETCH_ADD = 14
NVSHMEMI_AMO_FETCH_AND = 15
NVSHMEMI_AMO_FETCH_OR = 16
NVSHMEMI_AMO_FETCH_XOR = 17
NVSHMEMI_AMO_SWAP = 18
NVSHMEMI_AMO_COMPARE_SWAP = 19
NVSHMEMI_AMO_OP_SENTINEL = sys.maxsize
