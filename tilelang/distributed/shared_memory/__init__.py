"""Shared memory allocator for distributed communication (IPC + VMM/fabric).

All ops registered via TVM FFI under tl.shared_memory.* namespace.
"""

import ctypes
import ctypes.util
import operator
import warnings

import torch
import tvm_ffi

# ---------- TVM FFI function handles ----------


def _missing_shared_memory_func(name):
    def _missing(*args, **kwargs):
        raise RuntimeError(
            f"TileScale shared-memory FFI function '{name}' is unavailable. "
            "This usually means TileLang was built without CUDA shared-memory support. "
            "Rebuild with CUDA enabled to use distributed shared-memory allocations."
        )

    return _missing


def _get_required_global_func(name):
    func = tvm_ffi.get_global_func(name, allow_missing=True)
    if func is None:
        return _missing_shared_memory_func(name)
    return func


def _get_capability_global_func(name):
    func = tvm_ffi.get_global_func(name, allow_missing=True)
    if func is None:
        return lambda *args, **kwargs: False
    return func


_vmm_malloc = _get_required_global_func("tl.shared_memory.vmm_malloc")
_vmm_free = _get_required_global_func("tl.shared_memory.vmm_free")
_create_vmm_handle = _get_required_global_func("tl.shared_memory.create_vmm_handle")
_open_vmm_handle = _get_required_global_func("tl.shared_memory.open_vmm_handle")
_close_vmm_handle = _get_required_global_func("tl.shared_memory.close_vmm_handle")
_sync_vmm_handles_raw = _get_required_global_func("tl.shared_memory.sync_vmm_handles")
# POSIX-FD export/import, for nodes with no IMEX channel where a fabric handle
# cannot be created. Capability-style lookup so an older native library still
# imports.
_create_vmm_fd_handle = _get_required_global_func("tl.shared_memory.create_vmm_fd_handle")
_open_vmm_fd_handle = _get_required_global_func("tl.shared_memory.open_vmm_fd_handle")

_create_ipc_handle = _get_required_global_func("tl.shared_memory.create_ipc_handle")
_open_ipc_handle = _get_required_global_func("tl.shared_memory.open_ipc_handle")
_close_ipc_handle = _get_required_global_func("tl.shared_memory.close_ipc_handle")
_sync_ipc_handles_raw = _get_required_global_func("tl.shared_memory.sync_ipc_handles")

_supports_vmm = _get_capability_global_func("tl.shared_memory.supports_vmm")
_supports_vmm_fabric = _get_capability_global_func("tl.shared_memory.supports_vmm_fabric")
_supports_multicast = _get_capability_global_func("tl.shared_memory.supports_multicast")

# Multicast (NVSwitch) ops
_mc_create = _get_required_global_func("tl.shared_memory.mc_create")
_mc_export_handle = _get_required_global_func("tl.shared_memory.mc_export_handle")
_mc_import_handle = _get_required_global_func("tl.shared_memory.mc_import_handle")
# POSIX-fd route, for a cluster with no IMEX channel (see mc_export_fd_handle).
_mc_export_fd_handle = _get_required_global_func("tl.shared_memory.mc_export_fd_handle")
_mc_open_fd_handle = _get_required_global_func("tl.shared_memory.mc_open_fd_handle")
_multicast_uses_fd = _get_capability_global_func("tl.shared_memory.multicast_uses_fd")
_mc_add_device = _get_required_global_func("tl.shared_memory.mc_add_device")
_mc_bind_mem = _get_required_global_func("tl.shared_memory.mc_bind_mem")
_mc_map = _get_required_global_func("tl.shared_memory.mc_map")
_mc_release_handle = _get_required_global_func("tl.shared_memory.mc_release_handle")
_mc_unmap = _get_required_global_func("tl.shared_memory.mc_unmap")
_mc_get_aligned_size = _get_required_global_func("tl.shared_memory.mc_get_aligned_size")


# ---------- tensor_from_ptr (pure Python, no C++ torch dependency) ----------

_dtype_str_to_torch = {
    "float32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "double": torch.float64,
    "int32": torch.int32,
    "int": torch.int32,
    "int64": torch.int64,
    "long": torch.int64,
    "uint8": torch.uint8,
    "byte": torch.uint8,
    "uint16": torch.uint16,
    "int8": torch.int8,
    "bool": torch.bool,
    "uint32": torch.uint32,
    "uint64": torch.uint64,
}

# __cuda_array_interface__ typestr mapping
_torch_dtype_to_typestr = {
    torch.float32: "<f4",
    torch.float64: "<f8",
    torch.float16: "<f2",
    torch.int8: "<i1",
    torch.int16: "<i2",
    torch.int32: "<i4",
    torch.int64: "<i8",
    torch.uint8: "<u1",
    torch.bool: "|b1",
    # bfloat16/uint32/uint64 handled specially
}


class _ExternalCUDAArray:
    """Minimal __cuda_array_interface__ provider for zero-copy tensor creation."""

    def __init__(self, ptr: int, shape: tuple, typestr: str, owner=None):
        # PyTorch retains the producer for the lifetime of the resulting
        # storage. Keeping the allocation here therefore also protects aliases.
        self._owner = owner
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": shape,
            "typestr": typestr,
            "version": 3,
            "strides": None,
        }


def tensor_from_ptr(
    ptr_val: int,
    shape: list,
    dtype_str: str = "float32",
    device: int = 0,
    take_ownership: bool = False,
    _owner=None,
) -> torch.Tensor:
    """Create a CUDA tensor viewing external device memory (zero-copy)."""
    if take_ownership:
        raise NotImplementedError("tensor_from_ptr does not yet support ownership transfer")
    if ptr_val == 0:
        raise RuntimeError("Received null pointer (0).")

    dtype = _dtype_str_to_torch.get(dtype_str)
    if dtype is None:
        raise ValueError(f"Unsupported dtype string: '{dtype_str}'")

    if not isinstance(shape, (list, tuple)):
        shape = (shape,)
    try:
        shape = tuple(operator.index(dim) for dim in shape)
    except TypeError as exc:
        raise TypeError("shape dimensions must be integers") from exc
    if any(dim < 0 for dim in shape):
        raise ValueError("shape dimensions must be non-negative")

    numel = 1
    for s in shape:
        numel *= s
    if numel == 0:
        return torch.empty(shape, dtype=dtype, device=f"cuda:{device}")

    typestr = _torch_dtype_to_typestr.get(dtype)
    if typestr is not None:
        # Standard path via __cuda_array_interface__
        arr = _ExternalCUDAArray(ptr_val, shape, typestr, _owner)
        tensor = torch.as_tensor(arr, device=f"cuda:{device}")
    else:
        # bfloat16 / uint32 / uint64: create as matching-size int type, then view
        element_size = torch.empty((), dtype=dtype).element_size()
        if element_size == 2:
            # proxy_dtype = torch.int16
            proxy_typestr = "<i2"
        elif element_size == 4:
            # proxy_dtype = torch.int32
            proxy_typestr = "<i4"
        elif element_size == 8:
            # proxy_dtype = torch.int64
            proxy_typestr = "<i8"
        else:
            raise ValueError(f"Cannot handle dtype {dtype} with element_size={element_size}")

        arr = _ExternalCUDAArray(ptr_val, shape, proxy_typestr, _owner)
        t = torch.as_tensor(arr, device=f"cuda:{device}")
        tensor = t.view(dtype)

    if _owner is not None:
        tensor.untyped_storage()._tilelang_managed_allocation = _owner
    return tensor


# ---------- Higher-level Python wrappers ----------


def _sync_vmm_handles(rank, device_ids, buffer_ptrs_gpu_addr, all_gathered_handles):
    """Compatibility wrapper: packs handles into a single bytes blob and calls FFI."""
    num = len(device_ids)
    # all_gathered_handles is a list of bytearrays (or bytes)
    # Pack into single contiguous bytes blob
    # handle_size = len(all_gathered_handles[0]) if all_gathered_handles[0] is not None else 0
    packed = b""
    for h in all_gathered_handles:
        packed += bytes(h)
    _sync_vmm_handles_raw(rank, num, buffer_ptrs_gpu_addr, packed)


def _sync_ipc_handles(rank, device_ids, buffer_ptrs_gpu_addr, all_gathered_handles, root_unique_id_opt=None):
    """Compatibility wrapper for IPC handle sync."""
    num = len(device_ids)
    packed = b""
    for h in all_gathered_handles:
        packed += bytes(h)
    _sync_ipc_handles_raw(rank, num, buffer_ptrs_gpu_addr, packed)


def _create_tensor(shape, dtype):
    """Create a CUDA tensor (simple cudaMalloc-backed)."""
    return torch.empty(shape, dtype=dtype, device="cuda")


class _ManagedAllocation:
    """Own one cudaMallocManaged allocation until every tensor alias is gone."""

    def __init__(self, ptr: int, cudart):
        self.ptr = ptr
        self.cudart = cudart

    def __del__(self):
        ptr = getattr(self, "ptr", 0)
        if not ptr:
            return
        # Consume the pointer before cudaFree so finalization can never retry a
        # pointer whose state is uncertain after a runtime error.
        self.ptr = 0
        rc = self.cudart.cudaFree(ctypes.c_void_p(ptr))
        if rc != 0:
            warnings.warn(
                f"cudaFree failed while releasing managed allocation: error code {rc}",
                ResourceWarning,
                stacklevel=2,
            )


def create_host_device_tensor(shape, dtype):
    """Create host/device tensor views backed by one CUDA managed allocation."""
    if not isinstance(shape, (list, tuple)):
        shape = (shape,)
    try:
        shape = tuple(operator.index(dim) for dim in shape)
    except TypeError as exc:
        raise TypeError("shape dimensions must be integers") from exc
    if any(dim <= 0 for dim in shape):
        raise ValueError("managed tensor shape dimensions must be positive")

    numel = 1
    for s in shape:
        numel *= s

    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    cudart = ctypes.CDLL(ctypes.util.find_library("cudart") or "libcudart.so")
    cudart.cudaMallocManaged.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    cudart.cudaMallocManaged.restype = ctypes.c_int
    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    cudart.cudaFree.restype = ctypes.c_int

    ptr = ctypes.c_void_p()
    rc = cudart.cudaMallocManaged(ctypes.byref(ptr), ctypes.c_size_t(nbytes), 1)
    if rc != 0:
        raise RuntimeError(f"cudaMallocManaged failed with error code {rc}")

    allocation = _ManagedAllocation(ptr.value, cudart)
    buffer = (ctypes.c_byte * nbytes).from_address(ptr.value)
    buffer._tilelang_managed_allocation = allocation
    host = torch.frombuffer(buffer, dtype=dtype, count=numel).reshape(shape)
    host.untyped_storage()._tilelang_managed_allocation = allocation
    device = tensor_from_ptr(
        ptr.value,
        list(shape),
        str(dtype).split(".")[-1],
        torch.cuda.current_device(),
        False,
        allocation,
    )
    return host, device


__all__ = [
    "tensor_from_ptr",
    "_create_tensor",
    "_create_ipc_handle",
    "_open_ipc_handle",
    "_close_ipc_handle",
    "_sync_ipc_handles",
    "create_host_device_tensor",
    "_supports_vmm",
    "_supports_vmm_fabric",
    "_vmm_malloc",
    "_vmm_free",
    "_create_vmm_handle",
    "_create_vmm_fd_handle",
    "_open_vmm_fd_handle",
    "_open_vmm_handle",
    "_close_vmm_handle",
    "_sync_vmm_handles",
    "_supports_multicast",
    "_mc_export_fd_handle",
    "_mc_open_fd_handle",
    "_multicast_uses_fd",
]
