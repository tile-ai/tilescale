from __future__ import annotations

import inspect
import os
import operator
from dataclasses import dataclass
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


@dataclass
class NodeTopology:
    """Multi-node topology information for distributed execution.

    Attributes:
        node_rank: Rank of this node (0 to num_nodes-1)
        num_nodes: Total number of nodes in the distributed job
        local_world_size: Number of GPUs per node
        node_local_group: NCCL process group containing only ranks on this node
    """
    node_rank: int
    num_nodes: int
    local_world_size: int
    node_local_group: dist.ProcessGroup


def CUDA_CHECK(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {err}: {cuda.cuGetErrorName(err)}")
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cuda Error: {err}: {cudart.cudaGetErrorString(err)}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


def init_dist(
    local_rank: int,
    num_local_ranks: int,
    master_port: int | None = None,
    return_node_info: bool = False,
):
    """Initialize an NCCL process group with single-node or multi-node support.

    Args:
        local_rank: Local rank on this node (0 to num_local_ranks-1)
        num_local_ranks: Number of ranks per node
        master_port: Optional master port (defaults to TILESCALE_MASTER_PORT or MASTER_PORT)
        return_node_info: When True, also return the :class:`NodeTopology` describing
            the multi-node layout. Defaults to False so that existing single-node
            callers keep the historical three-value return.

    Returns:
        ``(rank, world_size, global_group)`` by default, or
        ``(rank, world_size, global_group, node_info)`` when ``return_node_info``
        is True. ``node_info`` is None for a single-node launch.

    Note:
        A multi-node launch requires ``return_node_info=True``; the resulting
        ``node_info`` must be forwarded to :func:`tilelang.get_allocator` so the
        allocator restricts IPC/VMM handle exchange to node-local peers.
    """
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_DEBUG", "ERROR")

    if not 0 <= local_rank < num_local_ranks:
        raise ValueError(f"local_rank must be in [0, {num_local_ranks}), got {local_rank}")

    # Detect topology from environment
    if "LOCAL_WORLD_SIZE" in os.environ:
        # torchrun style variables
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
        global_rank = launcher_rank
        global_world_size = launcher_world_size
    else:
        # Manual environment variables (NNODES, NODE_RANK)
        num_nodes = int(os.environ.get("NNODES", "1"))
        node_rank = int(os.environ.get("NODE_RANK", "0"))
        global_rank = int(os.environ.get("RANK", local_rank))
        global_world_size = int(os.environ.get("WORLD_SIZE", num_local_ranks))

        # Validate consistency
        if global_world_size != num_nodes * num_local_ranks:
            raise ValueError(
                f"Inconsistent configuration: WORLD_SIZE ({global_world_size}) != "
                f"NNODES ({num_nodes}) * num_local_ranks ({num_local_ranks})"
            )

    # Set device
    torch.cuda.set_device(local_rank)

    # Initialize global process group
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = master_port if master_port is not None else int(os.getenv("TILESCALE_MASTER_PORT", os.getenv("MASTER_PORT", "8361")))

    sig = inspect.signature(dist.init_process_group)
    params = {
        "backend": "nccl",
        "init_method": f"tcp://{ip}:{port}",
        "world_size": global_world_size,
        "rank": global_rank,
    }
    if "device_id" in sig.parameters:
        params["device_id"] = torch.device(f"cuda:{local_rank}")
    # Opt-in shorter group timeout. NCCL's 10-minute default means a mismatched
    # or stuck setup collective busy-waits at 100% GPU utilisation and gets
    # killed by the launcher's outer timeout before the watchdog ever reports,
    # which leaves no diagnostic at all. Unset -> unchanged default behaviour.
    _pg_timeout = os.getenv("TL_PG_TIMEOUT_SEC")
    if _pg_timeout and "timeout" in sig.parameters:
        import datetime

        params["timeout"] = datetime.timedelta(seconds=int(_pg_timeout))
    dist.init_process_group(**params)

    if num_nodes == 1:
        if return_node_info:
            return dist.get_rank(), dist.get_world_size(), dist.group.WORLD, None
        return dist.get_rank(), dist.get_world_size(), dist.group.WORLD

    if not return_node_info:
        raise ValueError(
            f"a multi-node launch was detected (num_nodes={num_nodes}), which requires "
            "init_dist(..., return_node_info=True) so the node-local topology can be "
            "forwarded to the allocator via get_allocator(node_info=...)"
        )

    # Every rank must build the same subgroups in the same order: dist.new_group
    # is collective over the default group, so ranks cannot create only their own.
    node_local_group = None
    for node in range(num_nodes):
        group = dist.new_group(
            ranks=[node * num_local_ranks + i for i in range(num_local_ranks)])
        if node == node_rank:
            node_local_group = group

    node_info = NodeTopology(
        node_rank=node_rank,
        num_nodes=num_nodes,
        local_world_size=num_local_ranks,
        node_local_group=node_local_group,
    )

    return global_rank, global_world_size, dist.group.WORLD, node_info


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
