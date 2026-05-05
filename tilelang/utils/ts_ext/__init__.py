from importlib import import_module as _imp

_C = _imp("tilescale_ext._C")

tensor_from_ptr = _C.tensor_from_ptr
_create_tensor = _C._create_tensor
_create_ipc_handle = _C._create_ipc_handle
_sync_ipc_handles = _C._sync_ipc_handles
create_host_device_tensor = _C.create_host_device_tensor

# VMM operations
_supports_vmm_fabric = _C._supports_vmm_fabric
_vmm_malloc = _C._vmm_malloc
_vmm_free = _C._vmm_free
_create_vmm_handle = _C._create_vmm_handle
_open_vmm_handle = _C._open_vmm_handle
_close_vmm_handle = _C._close_vmm_handle
_sync_vmm_handles = _C._sync_vmm_handles

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
    "_C",
]
