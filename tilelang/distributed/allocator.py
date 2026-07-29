from __future__ import annotations

import ctypes
import ctypes.util
import contextlib
import os
import operator
import threading
import warnings

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

__all__ = ["BaseAllocator", "get_allocator"]

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


def _resolve_use_vmm(use_vmm: bool | None, is_distributed: bool = False) -> bool:
    """Resolve whether to use VMM based on env var and hardware support."""
    env_val = os.environ.get("TILESCALE_USE_VMM", None)
    if env_val is not None:
        return _parse_bool_env("TILESCALE_USE_VMM", env_val)
    if use_vmm is not None:
        return use_vmm
    return is_distributed and _supports_vmm_fabric()


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

        if self._is_distributed:
            if self._group is None:
                raise ValueError("group must be provided when is_distributed is True")
            if not dist.is_initialized():
                raise RuntimeError("torch.distributed must be initialized before creating a distributed allocator")

            self._group_size = dist.get_world_size(self._group)
            group_rank = dist.get_rank(self._group)

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
        self._use_vmm = _resolve_use_vmm(self._use_vmm_requested, self._is_distributed)

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
        dist.all_gather_object(configurations, local_config, group=self._group)

        failures = []
        reference = configurations[0]
        invariant_keys = ("size", "align", "use_vmm", "mcast_size", "num_local_ranks")
        for rank, config in enumerate(configurations):
            if config["local_rank"] != rank:
                failures.append(f"rank {rank} reports local_rank={config['local_rank']!r}")
            if config["device"] != rank:
                failures.append(f"rank {rank} reports device={config['device']!r}")
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
        if hasattr(dist, "get_global_rank"):
            self._group_root_global_rank = dist.get_global_rank(self._group, 0)
        elif self._group is not dist.group.WORLD:
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

    def _collective_stage(self, stage: str, operation):
        """Run a local operation and make every rank observe any exception."""
        local_exception = None
        result = None
        try:
            result = operation()
        except Exception as exc:  # noqa: BLE001 - propagated with rank context
            local_exception = exc

        local_status = None
        if local_exception is not None:
            local_status = f"{type(local_exception).__name__}: {local_exception}"
        statuses = [None] * self._group_size
        dist.all_gather_object(statuses, local_status, group=self._group)
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
            obj_list = [mcast_fabric_bytes]
            dist.broadcast_object_list(obj_list, src=self._group_root_global_rank, group=self._group)
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
                self._collective_stage("release owned allocations", self._free_local_allocations)
            else:
                self._set_device("allocator close")
                self._free_remote_mappings()
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

        def create_handle():
            if self._use_vmm:
                return _create_vmm_handle(self._base_ptr.value)
            return _create_ipc_handle(self._base_ptr.value)

        local_handle = self._collective_stage("export allocation handles", create_handle)
        local_handle = self._collective_stage(
            "serialize allocation handle",
            lambda: bytes(local_handle),
        )
        dist.all_gather_object(handles, local_handle, group=self._group)

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
                elif self._use_vmm:
                    self._peer_ptr_values[peer_rank] = _open_vmm_handle(handle)
                else:
                    self._peer_ptr_values[peer_rank] = _open_ipc_handle(handle)

            host_ptrs = torch.tensor(self._peer_ptr_values, dtype=torch.uint64)
            self._buffer_ptrs.copy_(host_ptrs)

        self._collective_stage("import allocation handles", import_handles)

        def finalize_pointer_table():
            self._table_size = 2 + self._group_size
            self._table = torch.empty(self._table_size, dtype=torch.uint64, device="cpu")
            self._table[0] = self._local_rank
            self._table[1] = self._group_size
            self._table[2:] = self._buffer_ptrs

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
    )
