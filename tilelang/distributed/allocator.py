from __future__ import annotations

import ctypes
import ctypes.util
import contextlib
import os
import operator
import threading
import warnings
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from tilelang.distributed.shared_memory import (
    tensor_from_ptr,
    _create_ipc_handle,
    _open_ipc_handle,
    _close_ipc_handle,
    _vmm_malloc,
    _vmm_free,
    _create_vmm_handle,
    _open_vmm_handle,
    _close_vmm_handle,
    _supports_vmm,
    _supports_vmm_fabric,
    _supports_multicast,
    _mc_create,
    _mc_export_handle,
    _mc_import_handle,
    _mc_add_device,
    _mc_bind_mem,
    _mc_map,
    _mc_release_handle,
    _mc_unmap,
    _mc_get_aligned_size,
)
from tilelang.utils.target import parse_device

if TYPE_CHECKING:
    from tilelang.distributed.host import NodeTopology

__all__ = ["BaseAllocator", "get_allocator"]

# Distributed metadata table layout. Keep in sync with
# src/tl_templates/cuda/distributed/meta_layout.h, which documents the fields
# and is shared by the device helpers and the host remote-TMA remapping.
_META_GLOBAL_RANK = 0
_META_GLOBAL_WORLD_SIZE = 1
_META_NODE_RANK = 2
_META_NUM_NODES = 3
_META_LOCAL_RANK = 4
_META_LOCAL_WORLD_SIZE = 5
_META_GLOBAL_DEV_COMM = 6
_META_INTERNODE_DEV_COMM = 7
_META_ARENA_WINDOW = 8
_META_ARENA_BASE = 9
_META_PEER_BASE = 10

_dtype_to_str = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float64: "float64",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.uint16: "uint16",
    torch.uint32: "uint32",
    torch.uint64: "uint64",
    torch.int8: "int8",
    torch.bool: "bool",
}


def _element_size_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _prod_shape(shape: tuple[int, ...] | int) -> int:
    if not isinstance(shape, (list, tuple)):
        shape = (shape,)
    p = 1
    for d in shape:
        try:
            d = operator.index(d)
        except TypeError as exc:
            raise TypeError("shape dimensions must be integers") from exc
        if d < 0:
            raise ValueError("negative dimension in shape")
        p *= d
    return p


def _align_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


# helper: load CUDA runtime library
def _load_cudart():
    name = ctypes.util.find_library("cudart") or "libcudart.so"
    try:
        lib = ctypes.CDLL(name)
    except OSError as e:
        raise RuntimeError(f"cannot load the CUDA runtime library ({name}): {e}. Install a CUDA runtime that provides libcudart.") from e
    return lib


_libcudart = _load_cudart()
# setup signatures
_libcudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_libcudart.cudaMalloc.restype = ctypes.c_int
_libcudart.cudaFree.argtypes = [ctypes.c_void_p]
_libcudart.cudaFree.restype = ctypes.c_int
_libcudart.cudaGetErrorString.argtypes = [ctypes.c_int]
_libcudart.cudaGetErrorString.restype = ctypes.c_char_p
# optional set device
if hasattr(_libcudart, "cudaSetDevice"):
    _libcudart.cudaSetDevice.argtypes = [ctypes.c_int]
    _libcudart.cudaSetDevice.restype = ctypes.c_int


_TRUE_ENV_VALUES = frozenset({"1", "true", "on", "yes", "y"})
_FALSE_ENV_VALUES = frozenset({"0", "false", "off", "no", "n", ""})


def _parse_bool_env(name: str, value: str) -> bool:
    """Parse a boolean environment variable, refusing values we cannot read.

    An unrecognised value used to fall through to False, so setting
    TILESCALE_USE_VMM=true to *enable* VMM silently disabled it instead.
    """
    normalized = value.strip().lower()
    if normalized in _TRUE_ENV_VALUES:
        return True
    if normalized in _FALSE_ENV_VALUES:
        return False
    raise ValueError(
        f"{name} must be a boolean value "
        f"({'/'.join(sorted(_TRUE_ENV_VALUES))} or {'/'.join(sorted(v for v in _FALSE_ENV_VALUES if v))}), got {value!r}"
    )


def _resolve_use_vmm(
    use_vmm: bool | None,
    is_distributed: bool = False,
    needs_window: bool = False,
) -> bool:
    """Resolve whether to use VMM based on env var and hardware support.

    Two different capabilities are involved. Mapping a peer's arena into this
    process needs an *exportable fabric* handle, which requires an IMEX channel.
    Registering the arena as an NCCL window for GIN needs only a driver-level
    (VMM) allocation, which a POSIX-FD handle type also provides. So on a node
    with no IMEX channel, fabric is unavailable but GIN is still reachable --
    hence the weaker ``_supports_vmm`` check when a window is wanted.
    """
    env_val = os.environ.get("TILESCALE_USE_VMM", None)
    if env_val is not None:
        return _parse_bool_env("TILESCALE_USE_VMM", env_val)
    if use_vmm is not None:
        return use_vmm
    if is_distributed and _supports_vmm_fabric():
        return True
    # A cudaMalloc arena cannot be registered as an NCCL window, so GIN would be
    # dead on arrival; prefer VMM whenever it can be allocated at all.
    return bool(needs_window) and _supports_vmm()


def _resolve_register_window(is_multi_node: bool) -> tuple[bool, bool]:
    """Resolve whether to register the arena as an NCCL window for GIN.

    Returns ``(requested, required)``. ``required`` marks an explicit opt-in via
    ``TILESCALE_USE_GIN``, where a failure to register must raise rather than
    silently fall back -- inter-node traffic would otherwise be quietly dead.
    Auto mode only attempts registration for a genuine multi-node job, so
    single-node runs never take on an NCCL Device API dependency.
    """
    env_val = os.environ.get("TILESCALE_USE_GIN", None)
    if env_val is not None:
        requested = _parse_bool_env("TILESCALE_USE_GIN", env_val)
        return requested, requested
    return is_multi_node, False


class BaseAllocator:
    func: callable | None = None

    def __init__(
        self,
        size: int,
        device: str | torch.device | int | None = None,
        is_distributed: bool = False,
        local_rank: int | None = None,
        num_local_ranks: int | None = None,
        group: dist.ProcessGroup | None = None,
        align: int = 256,
        use_vmm: bool | None = None,
        mcast_size: int | None = None,
        node_info: NodeTopology | None = None,
    ) -> None:
        # Keep potentially failing local parsing inside the first collective
        # stage below. Otherwise one rank can exit while its peers wait forever
        # in the first configuration collective.
        self.size = size
        self._use_vmm_requested = use_vmm
        self._use_vmm = False
        self._base_ptr = ctypes.c_void_p(0)
        self._ptr = ctypes.c_void_p(0)
        self._device_request = device
        self._device = None
        self._is_distributed = is_distributed
        self._local_rank = local_rank
        self._num_local_ranks = num_local_ranks
        self._group = group
        self._node_info = node_info
        self._is_multi_node = (node_info is not None and node_info.num_nodes > 1)
        self._align = align
        self._lock = threading.RLock()
        self._mcast_size_requested = mcast_size
        # table items:
        # 1. local_rank, size: 8 bytes
        # 2. num_local_ranks, size: 8 bytes
        # 3. buffer_ptrs, size: 8 bytes * num_local_ranks
        # total size: 16 + 8 * num_local_ranks
        # Only _init_table fills these in, so keep them defined for the
        # non-distributed path where `table` is None and there is no table.
        self._table = None
        self._table_size = 0
        self._buffer_ptrs = None
        self._peer_ptr_values: list[int] = []
        self._device_ids = None
        self._initialized = False
        self._closed = False
        self._construction_collectives_started = False
        # Multicast state
        self._mcast_base_ptr = 0
        self._mcast_ptr = 0
        self._mcast_phys_ptr = 0
        self._mcast_aligned_size = 0
        self._mcast_handle = 0
        self._use_multicast = False
        self._group_size = 1
        self._group_root_global_rank = 0
        # NCCL/GIN arena window state. Zero means "no window registered", which
        # is the normal single-node case and keeps the intra-node peer-pointer
        # path free of any NCCL Device API dependency.
        self._arena_window = 0
        self._arena_window_comm = 0
        self._arena_window_size = 0
        # The GIN devcomm, created only once the window registers. Holding the
        # object alive is what keeps its device pointer valid.
        self._dev_comm = None
        # GIN gets its own process group. ncclDevCommCreate needs a communicator
        # that still supports symmetric memory, and a torch communicator loses
        # that after its first collective -- the call then *segfaults* rather
        # than failing, so it cannot be attempted and recovered from. Allocator
        # construction runs several collectives on the caller's group before GIN
        # setup, so a private group is the only safe option.
        self._gin_group = None
        self._register_window, self._require_window = _resolve_register_window(self._is_multi_node)

        if self._is_distributed:
            if self._group is None:
                raise ValueError("group must be provided when is_distributed is True")
            if not dist.is_initialized():
                raise RuntimeError("torch.distributed must be initialized before creating a distributed allocator")

            # For multi-node, use node-local group for IPC/VMM operations
            # For single-node, use the provided group as-is
            if self._is_multi_node:
                self._allocator_group = self._node_info.node_local_group
                self._global_group = self._group
            else:
                self._allocator_group = self._group
                self._global_group = self._group

            self._group_size = dist.get_world_size(self._allocator_group)
            group_rank = dist.get_rank(self._allocator_group)

        try:
            if self._is_distributed:
                self._construction_collectives_started = True
                # init_dist selects the current CUDA device before allocator
                # construction, which makes this first NCCL object collective
                # safe even when parsing the requested device fails locally.
                self._collective_stage("validate local configuration", self._prepare_local_configuration)
                self._validate_distributed_configuration(group_rank)
                self._collective_stage(
                    "select allocator device",
                    lambda: self._set_device("allocator initialization"),
                )
                self._collective_stage("resolve process-group root", self._resolve_group_root)
                if self._mcast_size_requested is not None:
                    self._collective_stage("validate multicast configuration", self._prepare_multicast)
                self._collective_stage("allocate base storage", self._alloc_base)
                if self._mcast_size_requested is not None:
                    self._init_multicast_buffer()
                self._init_arena_window()
                self._init_table()
            else:
                self._prepare_local_configuration()
                if self._mcast_size_requested is not None:
                    raise ValueError("mcast_size requires is_distributed=True")
                self._alloc_base()
            self._initialized = True
        except Exception as init_error:
            self._closed = True
            rollback_error = self._rollback_failed_initialization()
            if rollback_error is not None:
                raise rollback_error from init_error
            raise

    def _prepare_local_configuration(self) -> None:
        """Validate and normalize configuration before allocating resources."""

        def positive_integer(value, name: str) -> int:
            try:
                value = operator.index(value)
            except TypeError as exc:
                raise TypeError(f"{name} must be an integer") from exc
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
            return value

        self.size = positive_integer(self.size, "size")
        self._align = positive_integer(self._align, "align")
        if self._mcast_size_requested is not None:
            self._mcast_size_requested = positive_integer(self._mcast_size_requested, "mcast_size")
        self._device = parse_device(self._device_request)
        self._use_vmm = _resolve_use_vmm(
            self._use_vmm_requested,
            self._is_distributed,
            needs_window=self._register_window,
        )

    def _validate_distributed_configuration(self, group_rank: int) -> None:
        """Collectively validate invariants before any allocation is created."""
        local_config = {
            "size": self.size,
            "align": self._align,
            "use_vmm": self._use_vmm,
            "mcast_size": self._mcast_size_requested,
            "local_rank": self._local_rank,
            "num_local_ranks": self._num_local_ranks,
            "device": self._device,
        }
        configurations = [None] * self._group_size
        dist.all_gather_object(configurations, local_config, group=self._allocator_group)

        failures = []
        reference = configurations[0]
        invariant_keys = ("size", "align", "use_vmm", "mcast_size", "num_local_ranks")
        for rank, config in enumerate(configurations):
            # The allocator group is node-local, so a rank's index within it is
            # its local rank in both the single-node and multi-node cases.
            if config["local_rank"] != rank:
                failures.append(f"rank {rank} reports local_rank={config['local_rank']!r}")
            if config["device"] != config["local_rank"]:
                failures.append(f"rank {rank} reports device={config['device']!r}, expected local_rank={config['local_rank']!r}")

            for key in invariant_keys:
                if config[key] != reference[key]:
                    failures.append(f"rank {rank} reports {key}={config[key]!r}, expected {reference[key]!r}")

        if group_rank != self._local_rank:
            failures.append(f"this process has group rank {group_rank}, but local_rank={self._local_rank!r}")
        if self._num_local_ranks != self._group_size:
            failures.append(f"num_local_ranks={self._num_local_ranks!r}, but process-group size is {self._group_size}")
        if self._mcast_size_requested is not None and not self._use_vmm:
            failures.append("mcast_size requires use_vmm=True")

        if failures:
            raise ValueError("invalid distributed allocator configuration (" + "; ".join(failures) + ")")
        self._device_ids = [config["device"] for config in configurations]

    def _resolve_group_root(self) -> None:
        # For multi-node, resolve global rank of node-local group root
        # For single-node, resolve as before
        group_to_resolve = self._allocator_group if self._is_multi_node else self._group
        if hasattr(dist, "get_global_rank"):
            self._group_root_global_rank = dist.get_global_rank(group_to_resolve, 0)
        elif group_to_resolve is not dist.group.WORLD:
            raise RuntimeError("this PyTorch version cannot resolve the global rank of a subgroup")

    def _prepare_multicast(self) -> None:
        if not _supports_multicast():
            raise RuntimeError("Multicast unavailable; check GPU, driver, fabric, and IMEX configuration")
        self._mcast_aligned_size = _mc_get_aligned_size(self._mcast_size_requested, self._num_local_ranks)

    def _rollback_failed_initialization(self) -> RuntimeError | None:
        """Rollback construction without exposing freed owners to peer mappings."""
        if self._is_distributed and self._construction_collectives_started:
            try:
                self._collective_stage("rollback imported mappings", self._free_remote_mappings)
            except Exception as rollback_error:  # noqa: BLE001
                return RuntimeError(
                    "distributed allocator initialization failed and rollback of imported "
                    "mappings also failed; owned allocations were intentionally retained "
                    f"to avoid dangling peer mappings ({rollback_error})"
                )

            # Any one of the three is worth a rollback, and they are created in
            # order (group, devcomm, window), so a failure at any point leaves a
            # different subset behind. _free_arena_window releases whichever exist.
            if self._arena_window or self._dev_comm is not None or self._gin_group is not None:
                try:
                    self._collective_stage(
                        "rollback arena NCCL window",
                        self._free_arena_window,
                        group=self._global_group,
                    )
                except Exception as rollback_error:  # noqa: BLE001
                    return RuntimeError(
                        "distributed allocator initialization failed and deregistration of the "
                        "arena NCCL window also failed; owned allocations were intentionally "
                        f"retained to avoid freeing memory behind a live window ({rollback_error})"
                    )

            try:
                self._collective_stage("rollback owned allocations", self._free_local_allocations)
            except Exception as rollback_error:  # noqa: BLE001
                return RuntimeError(
                    "distributed allocator initialization failed and rollback of owned "
                    f"allocations also failed; remaining resources were retained ({rollback_error})"
                )
            return None

        try:
            self._free()
        except Exception as rollback_error:  # noqa: BLE001
            return RuntimeError(
                f"allocator initialization failed and local rollback also failed; remaining resources were retained ({rollback_error})"
            )
        return None

    def _collective_stage(self, stage: str, operation, group: dist.ProcessGroup | None = None):
        """Run a local operation and make every rank observe any exception.

        ``group`` defaults to the node-local allocator group. Stages whose
        collective spans nodes (NCCL window registration) must pass the global
        group, or a failure on one node leaves the other node waiting forever.
        """
        if group is None:
            group = self._allocator_group
        group_size = dist.get_world_size(group)
        local_exception = None
        result = None
        try:
            result = operation()
        except Exception as exc:  # noqa: BLE001 - propagated with rank context
            local_exception = exc

        local_status = None
        if local_exception is not None:
            local_status = f"{type(local_exception).__name__}: {local_exception}"
        statuses = [None] * group_size
        dist.all_gather_object(statuses, local_status, group=group)
        failures = [f"rank {rank}: {status}" for rank, status in enumerate(statuses) if status is not None]
        if failures:
            error = RuntimeError(f"distributed allocator stage '{stage}' failed ({'; '.join(failures)})")
            if local_exception is not None:
                raise error from local_exception
            raise error
        return result

    @property
    def device(self) -> int:
        return self._device

    def _alloc_base(self):
        self._set_device("allocator initialization")

        if self._use_vmm:
            ptr_val = _vmm_malloc(self.size)
            self._base_ptr.value = ptr_val
        else:
            rc = _libcudart.cudaMalloc(ctypes.byref(self._base_ptr), ctypes.c_size_t(self.size))
            if rc != 0:
                msg = _libcudart.cudaGetErrorString(rc)
                raise RuntimeError(f"cudaMalloc failed: {rc} {msg.decode() if msg else ''}")
        self._ptr.value = self._base_ptr.value

    def _init_arena_window(self):
        """Register the whole arena as one NCCL window for GIN inter-node access.

        Registration is collective over the *global* group and 4096-byte
        aligned, so it happens once here rather than per tensor. One handle plus
        a peer index names any rank's bytes, and the arena is symmetric, so
        ``local_ptr - arena_base`` is the offset valid on every rank.
        """
        if not self._register_window:
            return

        from tilelang.distributed import nccl_window as _win

        def check_support():
            if not _win.supports_device_api():
                raise RuntimeError(_win.unavailable_reason())
            # Measured on <node> with NCCL 2.28.9: ncclCommWindowRegister
            # rejects a plain cudaMalloc pointer with "invalid argument" under
            # both NCCL_WIN_COLL_SYMMETRIC and flags=0, while a VMM-mapped arena
            # registers cleanly. The NIC needs the driver-level allocation that
            # only the VMM path (or ncclMemAlloc) produces.
            if not self._use_vmm:
                raise RuntimeError(
                    "NCCL window registration requires a VMM-backed arena; the cudaMalloc "
                    "backend cannot be registered. Enable VMM (unset TILESCALE_USE_VMM=0) "
                    "for inter-node GIN support")

        try:
            self._collective_stage(
                "check NCCL Device API support", check_support, group=self._global_group)
        except Exception as exc:  # noqa: BLE001 - optional unless explicitly requested
            if self._require_window:
                raise RuntimeError(
                    "TILESCALE_USE_GIN requested NCCL window registration, but the NCCL "
                    f"Device API is unavailable: {exc}"
                ) from exc
            warnings.warn(
                "inter-node run without GIN: the arena could not be registered as an NCCL "
                f"window ({exc}); inter-node primitives will be unavailable",
                RuntimeWarning,
                stacklevel=2,
            )
            self._register_window = False
            return

        def make_gin_group():
            # device_id makes the new communicator eager, so _comm_ptr() is
            # non-null right away; without it the comm is created lazily on
            # first use, and the first use would be the collective that
            # invalidates it for ncclDevCommCreate.
            self._gin_group = dist.new_group(
                ranks=dist.get_process_group_ranks(self._global_group),
                backend="nccl",
                device_id=torch.device("cuda", self._device),
            )

        # new_group is collective over WORLD and every rank must reach it, so it
        # runs as its own stage rather than inside the registration closure.
        self._collective_stage("create GIN process group", make_gin_group, group=self._global_group)

        def create_dev_comm():
            # Before any collective touches this group -- see make_gin_group.
            comm_ptr = _win.get_comm_ptr(self._gin_group)
            if not comm_ptr:
                raise RuntimeError(
                    "could not obtain the raw ncclComm_t for the GIN process group; "
                    "torch.distributed must expose ProcessGroupNCCL._comm_ptr()"
                )
            self._arena_window_comm = comm_ptr
            self._dev_comm = _win.create_dev_comm(comm_ptr)

        self._collective_stage("create GIN devcomm", create_dev_comm, group=self._global_group)

        def register():
            base = self._base_ptr.value
            if base is None or base == 0:
                raise RuntimeError("arena base pointer is null; cannot register an NCCL window")
            if base % _win.NCCL_WIN_REQUIRED_ALIGNMENT:
                raise RuntimeError(
                    f"arena base {base:#x} is not {_win.NCCL_WIN_REQUIRED_ALIGNMENT}-byte aligned; "
                    "NCCL window registration requires NCCL_WIN_REQUIRED_ALIGNMENT"
                )
            size = _align_up(self.size, _win.NCCL_WIN_REQUIRED_ALIGNMENT)
            if size > self.size:
                raise RuntimeError(
                    f"arena size {self.size} is not a multiple of "
                    f"{_win.NCCL_WIN_REQUIRED_ALIGNMENT}; registering {size} bytes would run "
                    "past the allocation"
                )
            # The same communicator the devcomm was created from: a window
            # handle is only meaningful to the devcomm that shares its comm.
            self._arena_window_size = size
            self._arena_window = _win.register_window(
                self._arena_window_comm, base, size, _win.NCCL_WIN_COLL_SYMMETRIC)

        # Spans nodes, so it must be gated on the global group.
        self._collective_stage("register arena NCCL window", register, group=self._global_group)

    def _init_multicast_buffer(self):
        """Create multicast object and map, following multi-process fabric pattern."""
        num_devices = self._num_local_ranks
        aligned = self._mcast_aligned_size

        # Allocate physical memory (reuses vmm_malloc, same fabric handle type)
        def allocate_physical_storage():
            self._mcast_phys_ptr = _vmm_malloc(aligned)

        self._collective_stage("allocate multicast physical storage", allocate_physical_storage)

        # Rank 0 creates MC object, exports fabric handle; broadcast to all
        def create_and_export():
            if self._local_rank != 0:
                return None
            self._mcast_handle = _mc_create(aligned, num_devices)
            return bytes(_mc_export_handle(self._mcast_handle))

        mcast_fabric_bytes = self._collective_stage("create multicast object", create_and_export)

        def broadcast_handle():
            # The multicast object spans one node's devices, and each node's
            # local rank 0 creates its own. Broadcasting over the global group
            # would have ranks on different nodes pass different `src` values to
            # the same collective, so this must stay node-local.
            obj_list = [mcast_fabric_bytes]
            dist.broadcast_object_list(obj_list, src=self._group_root_global_rank, group=self._allocator_group)
            return obj_list[0]

        mcast_fabric_bytes = self._collective_stage("broadcast multicast object", broadcast_handle)

        # Non-rank-0 import the MC handle
        def import_handle():
            if self._local_rank != 0:
                self._mcast_handle = _mc_import_handle(mcast_fabric_bytes)

        self._collective_stage("import multicast object", import_handle)

        # Each rank adds its own device
        self._collective_stage(
            "add devices to multicast object",
            lambda: _mc_add_device(self._mcast_handle, self._device),
        )

        # Each rank binds its own physical memory
        self._collective_stage(
            "bind multicast physical storage",
            lambda: _mc_bind_mem(self._mcast_handle, self._mcast_phys_ptr, aligned),
        )

        # Each rank maps the MC object to a local VA
        def map_multicast_object():
            self._mcast_base_ptr = _mc_map(self._mcast_handle, aligned, num_devices)
            self._mcast_ptr = self._mcast_base_ptr

        self._collective_stage("map multicast object", map_multicast_object)

        # Release handle (backing persists due to mapping)
        def release_handle():
            handle = self._mcast_handle
            self._mcast_handle = 0
            _mc_release_handle(handle)

        self._collective_stage("release multicast handles", release_handle)
        self._use_multicast = True

    def _allocate_mcast_tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate from multicast buffer (bump-pointer).

        Returns:
            (mcast_tensor, local_tensor):
                mcast_tensor: backed by MC VA, for multimem read instructions
                local_tensor: backed by physical VA, for writing data
        """
        with self._lock:
            return self._allocate_mcast_tensor_locked(shape, dtype)

    def _allocate_mcast_tensor_locked(self, shape: tuple[int, ...], dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if self._closed or not self._initialized:
            raise RuntimeError("cannot allocate from a closed or uninitialized allocator")
        if not self._use_multicast:
            raise RuntimeError("Multicast buffer not initialized")

        numel = _prod_shape(shape)
        itemsize = _element_size_bytes(dtype)
        bytes_needed = numel * itemsize
        bytes_alloc = _align_up(bytes_needed, self._align)

        current_offset = self._mcast_ptr - self._mcast_base_ptr
        if current_offset + bytes_alloc > self._mcast_aligned_size:
            raise MemoryError(
                f"Mcast allocation failed: Requesting {bytes_alloc} bytes, but only "
                f"{self._mcast_aligned_size - current_offset} bytes available "
                f"(total mcast size: {self._mcast_aligned_size} bytes)."
            )

        dtype_str = _dtype_to_str.get(dtype)
        if dtype_str is None:
            dtype_str = str(dtype).split(".")[-1]
        if isinstance(shape, tuple):
            shape = list(shape)
        elif not isinstance(shape, list):
            shape = [shape]

        mcast_t = tensor_from_ptr(self._mcast_ptr, shape, dtype_str, self._device, False)
        local_t = tensor_from_ptr(self._mcast_phys_ptr + current_offset, shape, dtype_str, self._device, False)
        self._mcast_ptr += bytes_alloc
        return mcast_t, local_t

    def close(self):
        """Explicitly free resources with proper distributed coordination.

        Must be called collectively by all ranks before process group destruction.
        Safe to call multiple times.
        """
        with self._lock:
            if self._closed:
                return

            # A partially completed close may have already removed mappings or
            # owner storage. Disable allocations immediately, while leaving
            # _closed false so all ranks can retry the collective teardown.
            self._initialized = False
            if self._is_distributed and self._group is not None:
                self._collective_stage(
                    "select allocator device for close",
                    lambda: self._set_device("allocator close"),
                )
                # Imported mappings must be gone on every rank before any owner
                # allocation is released. Each stage propagates local failures
                # so all ranks either advance together or return the same error.
                self._collective_stage(
                    "synchronize device work",
                    lambda: torch.cuda.synchronize(self._device),
                )
                self._collective_stage("release imported mappings", self._free_remote_mappings)
                # Deregistration is collective over the global group and must
                # precede freeing the arena the window points at. Destroying the
                # devcomm and the GIN group are collective too, so this stage runs
                # when any of the three exists -- gating on the window alone would
                # skip resources left behind by a partially failed init.
                if self._arena_window or self._dev_comm is not None or self._gin_group is not None:
                    self._collective_stage(
                        "deregister arena NCCL window",
                        self._free_arena_window,
                        group=self._global_group,
                    )
                self._collective_stage("release owned allocations", self._free_local_allocations)
            else:
                self._set_device("allocator close")
                self._free_remote_mappings()
                self._free_arena_window()
                self._free_local_allocations()

            self._closed = True

    def _set_device(self, operation: str):
        if self._device is None:
            return
        rc = _libcudart.cudaSetDevice(int(self._device))
        if rc != 0:
            msg = _libcudart.cudaGetErrorString(rc)
            detail = msg.decode() if msg else ""
            raise RuntimeError(f"cudaSetDevice failed during {operation}: {rc} {detail}")

    def _free(self):
        """Best-effort non-collective teardown used by failed construction/destruction."""
        self._set_device("allocator cleanup")
        self._free_remote_mappings()
        self._free_arena_window()
        self._free_local_allocations()

    def _free_remote_mappings(self):
        if getattr(self, "_mcast_base_ptr", 0) and self._mcast_base_ptr:
            mcast_base_ptr = self._mcast_base_ptr
            self._mcast_base_ptr = 0
            self._mcast_ptr = 0
            _mc_unmap(mcast_base_ptr, self._mcast_aligned_size, self._num_local_ranks)

        if getattr(self, "_mcast_handle", 0) and self._mcast_handle:
            mcast_handle = self._mcast_handle
            self._mcast_handle = 0
            _mc_release_handle(mcast_handle)

        peer_ptr_values = getattr(self, "_peer_ptr_values", [])
        for peer_rank, peer_ptr in enumerate(peer_ptr_values):
            if peer_rank == self._local_rank or peer_ptr == 0:
                continue
            if self._use_vmm:
                # The C++ VMM closer performs best-effort RAII cleanup after an
                # error, so its pointer is consumed even when it raises.
                peer_ptr_values[peer_rank] = 0
                _close_vmm_handle(peer_ptr)
            else:
                _close_ipc_handle(peer_ptr)
                peer_ptr_values[peer_rank] = 0
        self._peer_ptr_values = []
        self._buffer_ptrs = None

    def _free_dev_comm(self):
        """Destroy the GIN devcomm. Collective, and must precede window teardown."""
        dev_comm = getattr(self, "_dev_comm", None)
        if dev_comm is None:
            return
        # Clear first so a failed destroy is not retried against a handle NCCL may
        # already have released, and so kernels cannot read a stale pointer.
        self._dev_comm = None
        if self._table is not None and len(self._table) > _META_INTERNODE_DEV_COMM:
            self._table[_META_GLOBAL_DEV_COMM] = 0
            self._table[_META_INTERNODE_DEV_COMM] = 0

        from tilelang.distributed import nccl_window as _win

        _win.destroy_dev_comm(dev_comm)

    def _free_arena_window(self):
        """Tear down the whole GIN stack: devcomm, then window, then its group.

        Ordered innermost-first, since each resource references the one after it.
        No step short-circuits the rest: a partially failed init can leave any
        subset of the three behind, and each one is skipped individually rather
        than by returning early.
        """
        # The devcomm references the communicator the window belongs to, so it has
        # to go first regardless of whether a window was ever registered.
        self._free_dev_comm()

        window = getattr(self, "_arena_window", 0)
        if window:
            comm_ptr = getattr(self, "_arena_window_comm", 0)
            # Clear first: a failed deregistration must not be retried against a
            # handle NCCL may already have destroyed.
            self._arena_window = 0
            self._arena_window_comm = 0
            self._arena_window_size = 0
            if self._table is not None and len(self._table) > _META_ARENA_WINDOW:
                self._table[_META_ARENA_WINDOW] = 0

            from tilelang.distributed import nccl_window as _win

            _win.deregister_window(comm_ptr, window)

        # Last: the communicator both of the above were created from.
        self._free_gin_group()

    def _free_gin_group(self):
        """Destroy the private GIN process group, after its devcomm and window."""
        gin_group = getattr(self, "_gin_group", None)
        if gin_group is None:
            return
        self._gin_group = None
        # Collective, and only safe once nothing references the communicator.
        dist.destroy_process_group(gin_group)

    def _free_local_allocations(self):
        if getattr(self, "_mcast_phys_ptr", 0) and self._mcast_phys_ptr:
            mcast_phys_ptr = self._mcast_phys_ptr
            self._mcast_phys_ptr = 0
            _vmm_free(mcast_phys_ptr)
        self._use_multicast = False

        if getattr(self, "_base_ptr", None) and self._base_ptr.value:
            if getattr(self, "_use_vmm", False):
                base_ptr = self._base_ptr.value
                self._base_ptr = ctypes.c_void_p(0)
                self._ptr = ctypes.c_void_p(0)
                _vmm_free(base_ptr)
            else:
                rc = _libcudart.cudaFree(self._base_ptr)
                if rc != 0:
                    msg = _libcudart.cudaGetErrorString(rc)
                    raise RuntimeError(f"cudaFree failed: {rc} {msg.decode() if msg else ''}")
                self._base_ptr = ctypes.c_void_p(0)
                self._ptr = ctypes.c_void_p(0)

        self._table = None

    def _init_table(self):
        # Synchronize handles (VMM or IPC)
        handles = [None] * self._group_size

        # With a single rank on the node there is no peer to map, so the export is
        # pure overhead -- and it is not always possible: a VMM arena allocated
        # with a POSIX-FD handle type (the fallback when no IMEX channel exists)
        # cannot be exported as a fabric handle. Skipping keeps a GIN-only run
        # working on nodes where intra-node peer mapping is unavailable.
        skip_peer_handles = self._group_size == 1

        def create_handle():
            if self._use_vmm:
                return _create_vmm_handle(self._base_ptr.value)
            return _create_ipc_handle(self._base_ptr.value)

        if skip_peer_handles:
            handles = [b""]
        else:
            local_handle = self._collective_stage("export allocation handles", create_handle)
            local_handle = self._collective_stage(
                "serialize allocation handle",
                lambda: bytes(local_handle),
            )
            dist.all_gather_object(handles, local_handle, group=self._allocator_group)

        def allocate_peer_pointer_table():
            self._buffer_ptrs = torch.empty(self._group_size, dtype=torch.uint64, device=f"cuda:{self._device}")

        self._collective_stage("allocate peer pointer table", allocate_peer_pointer_table)

        def import_handles():
            # Record every mapping immediately after it is opened. This state is
            # required if another rank reports a failure at the end of the stage.
            self._peer_ptr_values = [0] * self._group_size
            for peer_rank, handle in enumerate(handles):
                if peer_rank == self._local_rank:
                    self._peer_ptr_values[peer_rank] = self._base_ptr.value
                elif skip_peer_handles:
                    continue
                elif self._use_vmm:
                    self._peer_ptr_values[peer_rank] = _open_vmm_handle(handle)
                else:
                    self._peer_ptr_values[peer_rank] = _open_ipc_handle(handle)

            host_ptrs = torch.tensor(self._peer_ptr_values, dtype=torch.uint64)
            self._buffer_ptrs.copy_(host_ptrs)

        self._collective_stage("import allocation handles", import_handles)

        def finalize_pointer_table():
            # Layout is defined once in
            # src/tl_templates/cuda/distributed/meta_layout.h and must match the
            # device helpers in distributed.h and the host-side remote TMA
            # remapping in src/cuda/runtime.cc.
            if self._is_multi_node:
                global_rank = dist.get_rank(self._global_group)
                global_world_size = dist.get_world_size(self._global_group)
                node_rank = self._node_info.node_rank
                num_nodes = self._node_info.num_nodes
            else:
                global_rank = self._local_rank
                global_world_size = self._num_local_ranks
                node_rank = 0
                num_nodes = 1
            local_rank = self._local_rank
            local_world_size = self._num_local_ranks

            self._table_size = _META_PEER_BASE + local_world_size
            self._table = torch.empty(self._table_size, dtype=torch.uint64, device="cpu")
            self._table[_META_GLOBAL_RANK] = global_rank
            self._table[_META_GLOBAL_WORLD_SIZE] = global_world_size
            self._table[_META_NODE_RANK] = node_rank
            self._table[_META_NUM_NODES] = num_nodes
            self._table[_META_LOCAL_RANK] = local_rank
            self._table[_META_LOCAL_WORLD_SIZE] = local_world_size
            # Device pointer to the GIN devcomm, or zero when GIN is unavailable;
            # tl::gin::available() tests exactly this slot. The devcomm covers the
            # whole global communicator, so the same handle serves both slots --
            # the inter-node entry is kept distinct for a future rail-local comm.
            dev_comm_ptr = self._dev_comm.device_ptr if self._dev_comm is not None else 0
            self._table[_META_GLOBAL_DEV_COMM] = dev_comm_ptr
            self._table[_META_INTERNODE_DEV_COMM] = dev_comm_ptr
            # Zero window means no GIN; kernels must check before using it.
            self._table[_META_ARENA_WINDOW] = self._arena_window
            self._table[_META_ARENA_BASE] = self._base_ptr.value or 0
            self._table[_META_PEER_BASE:] = self._buffer_ptrs

        self._collective_stage("finalize peer pointer table", finalize_pointer_table)

    def initialized(self) -> bool:
        return self._initialized

    def _allocate_tensor(
        self, shape: tuple[int, ...], dtype: torch.dtype, return_peers=False, take_ownership: bool = False
    ) -> torch.Tensor:
        with self._lock:
            return self._allocate_tensor_locked(shape, dtype, return_peers, take_ownership)

    def _allocate_tensor_locked(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        return_peers: bool,
        take_ownership: bool,
    ) -> torch.Tensor:
        if self._closed or not self._initialized:
            raise RuntimeError("cannot allocate from a closed or uninitialized allocator")
        if take_ownership:
            raise NotImplementedError("BaseAllocator does not yet support transferring allocation ownership to a tensor")
        numel = _prod_shape(shape)
        itemsize = _element_size_bytes(dtype)
        bytes_needed = numel * itemsize

        bytes_alloc = _align_up(bytes_needed, self._align)

        current_offset = int(self._ptr.value) - int(self._base_ptr.value)
        if current_offset + bytes_alloc > self.size:
            bytes_available = self.size - current_offset
            raise MemoryError(
                f"Allocation failed: Requesting {bytes_alloc} bytes, but only "
                f"{bytes_available} bytes are available in the pre-allocated buffer "
                f"(total size: {self.size} bytes)."
            )

        if not isinstance(self._ptr, ctypes.c_void_p):
            raise TypeError("self._ptr must be ctypes.c_void_p")
        cur_ptr_val = int(self._ptr.value)
        if cur_ptr_val == 0:
            raise RuntimeError("null device pointer")

        dtype_str = _dtype_to_str.get(dtype)
        if dtype_str is None:
            dtype_str = str(dtype).split(".")[-1]

        if isinstance(shape, tuple):
            shape = list(shape)
        elif not isinstance(shape, list):
            shape = [shape]

        t = tensor_from_ptr(cur_ptr_val, shape, dtype_str, self._device, take_ownership)

        if return_peers:
            peer_ts = []
            for i in range(self._group_size):
                if i == self._local_rank:
                    peer_ts.append(t)
                else:
                    peer_ptr_val = int(self._buffer_ptrs[i]) + current_offset
                    # Peer pointers are mapped into the current rank's CUDA VA
                    # space. Keep the torch tensor on the current device so
                    # kernel adapters validate all arguments against this rank.
                    peer_t = tensor_from_ptr(peer_ptr_val, shape, dtype_str, self._device, False)
                    peer_ts.append(peer_t)

        new_ptr_val = cur_ptr_val + bytes_alloc
        self._ptr.value = new_ptr_val

        return peer_ts if return_peers else t

    @property
    def ptr(self) -> int:
        return int(self._ptr.value) if self._ptr and self._ptr.value else 0

    @property
    def table(self) -> torch.Tensor:
        return self._table

    @property
    def table_size(self) -> int:
        return self._table_size

    @property
    def arena_window(self) -> int:
        """``ncclWindow_t`` for the whole arena, or 0 when GIN is unavailable."""
        return self._arena_window

    @property
    def arena_base(self) -> int:
        """Arena base address; subtract from a local pointer to get a window offset."""
        return int(self._base_ptr.value) if self._base_ptr and self._base_ptr.value else 0

    def window_offset(self, ptr: int) -> int:
        """Convert a local arena pointer to the offset valid on every rank.

        The arena is symmetric, so the same offset names the corresponding bytes
        on any peer -- the identical subtraction the intra-node peer-pointer path
        performs, just paired with a window handle instead of a peer base.
        """
        base = self.arena_base
        if not base:
            raise RuntimeError("allocator has no arena base; cannot compute a window offset")
        ptr = operator.index(ptr)
        if not base <= ptr < base + self.size:
            raise ValueError(
                f"pointer {ptr:#x} is outside the arena [{base:#x}, {base + self.size:#x})"
            )
        return ptr - base

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        if getattr(self, "_is_distributed", False):
            try:
                warnings.warn(
                    "distributed BaseAllocator was not closed collectively; imported mappings "
                    "will be released best-effort, while owned allocations are retained until "
                    "CUDA context teardown to avoid dangling mappings on peer ranks",
                    ResourceWarning,
                    stacklevel=2,
                )
            finally:
                with contextlib.suppress(Exception):
                    self._set_device("allocator destruction")
                    self._free_remote_mappings()
            return
        with contextlib.suppress(Exception):
            self._free()


def get_allocator(
    size: int = 2**30,
    device: str = "cuda",
    is_distributed: bool = True,
    local_rank: int = 0,
    num_local_ranks: int = 1,
    group: dist.ProcessGroup | None = None,
    use_vmm: bool | None = None,
    mcast_size: int | None = None,
    node_info: NodeTopology | None = None,
) -> BaseAllocator:
    return BaseAllocator(
        size,
        device=device,
        is_distributed=is_distributed,
        local_rank=local_rank,
        num_local_ranks=num_local_ranks,
        group=group,
        use_vmm=use_vmm,
        mcast_size=mcast_size,
        node_info=node_info,
    )
