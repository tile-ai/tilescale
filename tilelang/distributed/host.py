from __future__ import annotations

import inspect
import os
import operator
from functools import lru_cache

import torch
import torch.distributed as dist

try:
    from cuda.bindings import driver as cuda
    from cuda.bindings import runtime as cudart
except ImportError:
    try:
        from cuda import cuda, cudart
    except ImportError as exc:
        raise ImportError(
            "TileScale distributed host helpers require cuda-python; "
            "install the 'distributed' extra with `pip install 'tilescale[distributed]'`."
        ) from exc


def CUDA_CHECK(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {err}: {cuda.cuGetErrorName(err)}")
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cuda Error: {err}: {cudart.cudaGetErrorString(err)}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


def init_dist(local_rank: int, num_local_ranks: int, master_port: int | None = None):
    """Initialize the currently supported single-node NCCL process group.

    Single-node ``torchrun`` variables are accepted. Multi-node groups are
    rejected until TileScale creates a node-local allocator group.
    """
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_DEBUG", "ERROR")

    if not 0 <= local_rank < num_local_ranks:
        raise ValueError(f"local_rank must be in [0, {num_local_ranks}), got {local_rank}")

    if "LOCAL_WORLD_SIZE" in os.environ:
        launcher_local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        launcher_world_size = int(os.environ.get("WORLD_SIZE", launcher_local_world_size))
        launcher_rank = int(os.environ.get("RANK", local_rank))
        if launcher_local_world_size != num_local_ranks:
            raise ValueError(f"num_local_ranks must match torchrun LOCAL_WORLD_SIZE: {num_local_ranks} != {launcher_local_world_size}")
        if launcher_world_size % launcher_local_world_size != 0:
            raise ValueError("torchrun WORLD_SIZE must be divisible by LOCAL_WORLD_SIZE")
        num_nodes = launcher_world_size // launcher_local_world_size
        node_rank = int(os.environ.get("GROUP_RANK", launcher_rank // launcher_local_world_size))
        launcher_local_rank = int(os.environ.get("LOCAL_RANK", launcher_rank))
        if num_nodes == 1 and (launcher_rank != local_rank or launcher_local_rank != local_rank):
            raise ValueError(
                "local_rank must match torchrun RANK and LOCAL_RANK for a single-node launch: "
                f"local_rank={local_rank}, RANK={launcher_rank}, LOCAL_RANK={launcher_local_rank}"
            )
    else:
        num_nodes = int(os.environ.get("NNODES", "1"))
        node_rank = int(os.environ.get("NODE_RANK", "0"))
        legacy_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        legacy_rank = int(os.environ.get("RANK", "0"))
        if legacy_world_size != 1 or legacy_rank != 0:
            raise NotImplementedError(
                "Ambiguous WORLD_SIZE/RANK launcher environment without LOCAL_WORLD_SIZE; "
                "TileScale v0 supports local spawn or single-node torchrun only."
            )

    if num_nodes != 1 or node_rank != 0:
        raise NotImplementedError(
            "TileScale v0 currently supports only single-node process groups; multi-node launch requires a node-local allocator group."
        )

    torch.cuda.set_device(local_rank)

    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = master_port if master_port is not None else int(os.getenv("TILESCALE_MASTER_PORT", os.getenv("MASTER_PORT", "8361")))

    sig = inspect.signature(dist.init_process_group)
    params = {
        "backend": "nccl",
        "init_method": f"tcp://{ip}:{port}",
        "world_size": num_local_ranks,
        "rank": local_rank,
    }
    if "device_id" in sig.parameters:
        params["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**params)

    return dist.get_rank(), dist.get_world_size(), dist.group.WORLD


@lru_cache
def supports_p2p_native_atomic():
    """Check native atomic support for peer-to-peer access between CUDA devices 0 and 1."""

    assert torch.cuda.is_available() and torch.cuda.device_count() > 1

    (err,) = cudart.cudaFree(0)
    CUDA_CHECK(err)

    (err, support) = cudart.cudaDeviceGetP2PAttribute(cudart.cudaDeviceP2PAttr.cudaDevP2PAttrNativeAtomicSupported, 0, 1)
    CUDA_CHECK(err)
    return support == 1


def _validate_signal_value(signal: int, bits: int) -> int:
    try:
        value = operator.index(signal)
    except TypeError as exc:
        raise TypeError("signal must be an integer") from exc
    if not 0 <= value < 1 << bits:
        raise ValueError(f"signal must fit in an unsigned {bits}-bit value")
    return value


def _validate_signal_tensor(
    signal_tensor: torch.Tensor,
    allowed_dtypes: tuple[torch.dtype, ...],
    stream: torch.cuda.Stream | None,
) -> torch.cuda.Stream:
    if not isinstance(signal_tensor, torch.Tensor):
        raise TypeError("signal_tensor must be a torch.Tensor")
    if signal_tensor.dtype not in allowed_dtypes:
        names = ", ".join(str(dtype) for dtype in allowed_dtypes)
        raise TypeError(f"signal_tensor dtype must be one of {names}, got {signal_tensor.dtype}")
    if not signal_tensor.is_cuda:
        raise ValueError("signal_tensor must be a CUDA tensor")
    if signal_tensor.numel() == 0:
        raise ValueError("signal_tensor must contain at least one element")
    if not signal_tensor.is_contiguous():
        raise ValueError("signal_tensor must be contiguous")

    stream = stream or torch.cuda.current_stream(signal_tensor.device)
    stream_device = stream.device
    if isinstance(stream_device, int):
        stream_device = torch.device("cuda", stream_device)
    else:
        stream_device = torch.device(stream_device)
    if stream_device != signal_tensor.device:
        raise ValueError(f"stream device {stream_device} does not match signal tensor device {signal_tensor.device}")
    return stream


def set_signal(signal_tensor: torch.Tensor, signal: int, stream: torch.cuda.Stream | None = None):
    signal = _validate_signal_value(signal, 32)
    stream = _validate_signal_tensor(signal_tensor, (torch.int32, torch.uint32), stream)
    (err,) = cuda.cuStreamWriteValue32(
        stream.cuda_stream,
        signal_tensor.data_ptr(),
        signal,
        cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
    )
    CUDA_CHECK(err)


def wait_eq(signal_tensor: torch.Tensor, signal: int, stream: torch.cuda.Stream | None = None, require_i64=False):
    bits = 64 if require_i64 else 32
    dtypes = (torch.int64, torch.uint64) if require_i64 else (torch.int32, torch.uint32)
    signal = _validate_signal_value(signal, bits)
    stream = _validate_signal_tensor(signal_tensor, dtypes, stream)
    wait = cuda.cuStreamWaitValue64 if require_i64 else cuda.cuStreamWaitValue32
    (err,) = wait(
        stream.cuda_stream,
        signal_tensor.data_ptr(),
        signal,
        cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
    )
    CUDA_CHECK(err)


def cuda_stream_max_priority():
    ret = cudart.cudaDeviceGetStreamPriorityRange()
    CUDA_CHECK(ret[0])
    return ret[2]
