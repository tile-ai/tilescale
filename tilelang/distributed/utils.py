import torch
import torch.distributed as dist
import datetime
import os
import inspect
from typing import List, Union, Tuple, Callable, Sequence, Optional
from contextlib import contextmanager

import importlib.metadata

cuda_python_version = importlib.metadata.version("cuda-python")
from packaging import version
if version.parse(cuda_python_version) >= version.parse("12.8.0"):
    from cuda.bindings import driver as cuda
    from cuda.bindings import runtime as cudart
else:
    from cuda import cuda, cudart

import ctypes
from tilelang.distributed.common.ipc_ext import _create_ipc_handle, _sync_ipc_handles, _create_tensor
from functools import lru_cache

dtype_map = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "s8": torch.int8,
    "s32": torch.int32,
    "float32": torch.float32,
}


def init_dist(local_rank: int, num_local_ranks: int):
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))

    sig = inspect.signature(dist.init_process_group)
    params = {
        'backend': 'nccl',
        'init_method': f'tcp://{ip}:{port}',
        'world_size': num_nodes * num_local_ranks,
        'rank': node_rank * num_local_ranks + local_rank,
    }
    if 'device_id' in sig.parameters:
        # noinspection PyTypeChecker
        params['device_id'] = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    torch.cuda.set_device(local_rank)

    return dist.get_rank(), dist.get_world_size(), dist.new_group(
        list(range(num_local_ranks * num_nodes)))


def init_distributed(return_tp_group=False, init_nvshmem=True):
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    torch.cuda.set_device(LOCAL_RANK)
    assert torch.distributed.is_initialized()
    TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")

    torch.cuda.synchronize()
    if init_nvshmem:
        import pynvshmem
        pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)

    if return_tp_group:
        return WORLD_SIZE, RANK, LOCAL_RANK, TP_GROUP
    else:
        return WORLD_SIZE, RANK, LOCAL_RANK


def create_tensor(shape: List[int], dtype: torch.dtype) -> torch.Tensor:
    # NOTE(wt): We discovered that IPC only works with tensors explicitly allocated by `cudaMalloc` somehow.
    return _create_tensor(shape, dtype)


# IPC related functions
def get_local_ipc_handle(data: torch.Tensor):
    p = ctypes.c_void_p(data.data_ptr())
    handle = _create_ipc_handle(p.value)
    return handle


def create_dist_tensor(local_rank: int, num_local_ranks: int, data: torch.Tensor, rank: int,
                       group: dist.ProcessGroup):
    assert num_local_ranks == group.size()
    # Synchronize device IDs
    device_ids = [
        None,
    ] * group.size()
    local_device_id = local_rank
    dist.all_gather_object(device_ids, local_device_id, group)

    # Synchronize IPC handles
    ipc_handles = [
        None,
    ] * group.size()
    local_ipc_handle = get_local_ipc_handle(data)
    dist.all_gather_object(ipc_handles, local_ipc_handle, group)
    buffer_ptrs_gpu = torch.empty(group.size(), dtype=torch.uint64, device="cuda")
    _sync_ipc_handles(rank, device_ids,
                      ctypes.c_void_p(buffer_ptrs_gpu.data_ptr()).value, ipc_handles, None)
    return buffer_ptrs_gpu


@contextmanager
def with_torch_deterministic(mode: bool, warn_only: bool = True):
    old_mode = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(mode, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(old_mode, warn_only=warn_only)


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype.itemsize == 1 and dtype.is_floating_point


def _make_tensor(
    shape: List[Union[int, Callable[[], int]]],
    dtype: torch.dtype,
    init_args: Union[Tuple[float, float], Tuple[int, int]],
    device: str = "cuda",
):
    """
    rand() * scale + bias
    randint(-scale, scale) + bias
    """
    if isinstance(shape, Sequence):
        shape = tuple([x() if isinstance(x, Callable) else x for x in shape])
    elif isinstance(shape, int):
        shape = (shape,)
    elif isinstance(shape, Callable):
        shape = shape()
    else:
        raise ValueError(f"unsupported shape {shape}")

    scale, bias = init_args
    if dtype in [torch.float16, torch.bfloat16, torch.float32]:
        out = (torch.rand(shape, dtype=dtype, device=device) * 2 - 1) * scale + bias
    elif dtype == torch.int8:
        out = torch.randint(-scale, scale, shape, dtype=torch.int8, device=device)
        out = out + bias
    elif is_fp8_dtype(dtype):
        out = (torch.rand(shape, dtype=torch.float16, device=device) * 2 - 1) * scale + bias
        with with_torch_deterministic(False):
            out = out.to(dtype)
    else:
        raise ValueError(f"unsupported dtype {dtype}")

    return out


def generate_data(configs):
    while True:
        yield (_make_tensor(*args) if args else None for args in configs)


def dist_print(*args, **kwargs):
    """A wrapped distributed version of the built-in `print` function.
    Args:
        allowed_ranks (List[int] or "all"): The ranks that are allowed to print. Default: [0].
        prefix (bool): Whether to add a prefix indicating the rank number. Default: False.
        need_sync (bool): Whether to synchronize all ranks before printing. Default: False.
    Note:
        This function requires the environment variables "RANK" and "WORLD_SIZE" to be set.
    Example:
    ```
    dist_print("Hello, world!", allowed_ranks=[0, 1], prefix=True, need_sync=True)
    ```
    """
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    prefix = False
    if "allowed_ranks" in kwargs:
        allowed_ranks = kwargs["allowed_ranks"]
        if isinstance(allowed_ranks, str) and allowed_ranks == "all":
            allowed_ranks = list(range(world_size))

        del kwargs["allowed_ranks"]
    else:
        allowed_ranks = [0]
    if "prefix" in kwargs:
        prefix = kwargs["prefix"]

        del kwargs["prefix"]

    need_sync = False
    if "need_sync" in kwargs:
        need_sync = kwargs["need_sync"]

        del kwargs["need_sync"]

    for allowed in allowed_ranks:
        if need_sync:
            torch.distributed.barrier()
        if rank == allowed:
            if prefix:
                print(f"[rank:{rank}]", end="")
            print(*args, **kwargs)


def perf_fn(fn: Callable, warmup: int, rep: int):
    """Benchmark a function `fn` by running it `warmup` times for warm-up and then `rep` times for measurement.
    Returns the output of the last run and the average time per run in milliseconds.
    Args:
        fn (Callable): The function to benchmark.
        warmup (int): The number of warm-up runs.
        rep (int): The number of measurement runs.
    Returns:
        output: The output of the last run of `fn`.
        float: The average time per run in milliseconds.
    """
    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)
    for i in range(warmup + rep):
        if i == warmup:
            st.record()
        output = fn()
    ed.record()
    st.wait()
    ed.wait()
    torch.cuda.current_stream().synchronize()
    return output, st.elapsed_time(ed) / rep


def CUDA_CHECK(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {err}: {cuda.cuGetErrorName(err)}")
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cuda Error: {err}: {cudart.cudaGetErrorString(err)}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


@lru_cache
def supports_p2p_native_atomic():
    """Check if native atomic operations are supported for peer-to-peer (P2P) access between CUDA devices 0 and 1."""

    assert torch.cuda.is_available() and torch.cuda.device_count() > 1

    # force create CUDA context
    (err,) = cudart.cudaFree(0)
    CUDA_CHECK(err)

    (err, support) = cudart.cudaDeviceGetP2PAttribute(
        cudart.cudaDeviceP2PAttr.cudaDevP2PAttrNativeAtomicSupported, 0, 1)
    CUDA_CHECK(err)
    return support == 1


def set_signal(signal_tensor: torch.Tensor,
               signal: int,
               stream: Optional[torch.cuda.Stream] = None):
    stream = stream or torch.cuda.current_stream()
    if signal_tensor.dtype == torch.int32:
        (err,) = cuda.cuStreamWriteValue32(
            stream.cuda_stream,
            signal_tensor.data_ptr(),
            signal,
            cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        )
        CUDA_CHECK(err)
    else:
        raise Exception(f"Unsupported signal dtype {signal_tensor.dtype}")


def wait_eq(signal_tensor: torch.Tensor,
            signal: int,
            stream: Optional[torch.cuda.Stream] = None,
            require_i64=False):
    stream = stream or torch.cuda.current_stream()
    if signal_tensor.dtype == torch.int32:
        (err,) = cuda.cuStreamWaitValue32(
            stream.cuda_stream,
            signal_tensor.data_ptr(),
            signal,
            cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        )
        CUDA_CHECK(err)
    else:
        raise Exception(f"Unsupported signal dtype {signal_tensor.dtype}")
