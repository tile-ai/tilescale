"""Distributed runtime helpers, loaded only when their APIs are requested."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "do_bench": (".bench", "do_bench"),
    "perf_fn": (".bench", "perf_fn"),
    "CUDA_CHECK": (".host", "CUDA_CHECK"),
    "cuda_stream_max_priority": (".host", "cuda_stream_max_priority"),
    "init_dist": (".host", "init_dist"),
    "set_signal": (".host", "set_signal"),
    "supports_p2p_native_atomic": (".host", "supports_p2p_native_atomic"),
    "wait_eq": (".host", "wait_eq"),
    "NvidiaSmiUtil": (".topology", "NvidiaSmiUtil"),
    "ensure_nvml_initialized": (".topology", "ensure_nvml_initialized"),
    "has_fullmesh_nvlink": (".topology", "has_fullmesh_nvlink"),
    "has_fullmesh_nvlink_pynvml": (".topology", "has_fullmesh_nvlink_pynvml"),
    "tensor_from_ptr": (".shared_memory", "tensor_from_ptr"),
    "_create_tensor": (".shared_memory", "_create_tensor"),
    "_create_ipc_handle": (".shared_memory", "_create_ipc_handle"),
    "_sync_ipc_handles": (".shared_memory", "_sync_ipc_handles"),
    "_supports_vmm": (".shared_memory", "_supports_vmm"),
    "_supports_vmm_fabric": (".shared_memory", "_supports_vmm_fabric"),
    "_create_vmm_fd_handle": (".shared_memory", "_create_vmm_fd_handle"),
    "_open_vmm_fd_handle": (".shared_memory", "_open_vmm_fd_handle"),
    "_vmm_malloc": (".shared_memory", "_vmm_malloc"),
    "_vmm_free": (".shared_memory", "_vmm_free"),
    "_create_vmm_handle": (".shared_memory", "_create_vmm_handle"),
    "_open_vmm_handle": (".shared_memory", "_open_vmm_handle"),
    "_close_vmm_handle": (".shared_memory", "_close_vmm_handle"),
    "_sync_vmm_handles": (".shared_memory", "_sync_vmm_handles"),
    "_supports_multicast": (".shared_memory", "_supports_multicast"),
    "nccl_supports_device_api": (".nccl_window", "supports_device_api"),
    "nccl_version": (".nccl_window", "nccl_version"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
