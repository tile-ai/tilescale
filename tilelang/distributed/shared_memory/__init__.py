"""Shared memory allocator for distributed communication (IPC + VMM/fabric)."""

from tilelang.distributed.shared_memory._C import (  # type: ignore[import]
    tensor_from_ptr,
    _create_tensor,
    _create_ipc_handle,
    _sync_ipc_handles,
    create_host_device_tensor,
    _supports_vmm_fabric,
    _vmm_malloc,
    _vmm_free,
    _create_vmm_handle,
    _open_vmm_handle,
    _close_vmm_handle,
    _sync_vmm_handles,
)

__all__ = [
    "tensor_from_ptr",
    "_create_tensor",
    "_create_ipc_handle",
    "_sync_ipc_handles",
    "create_host_device_tensor",
    "_supports_vmm_fabric",
    "_vmm_malloc",
    "_vmm_free",
    "_create_vmm_handle",
    "_open_vmm_handle",
    "_close_vmm_handle",
    "_sync_vmm_handles",
]
