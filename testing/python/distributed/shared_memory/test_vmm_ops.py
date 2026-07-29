"""
VMM (Virtual Memory Management) and IPC shared-memory operations.

Single-GPU tests run directly under pytest.
Multi-GPU tests use torch.multiprocessing.spawn and are configured for 4 GPUs.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import threading

import pytest
import torch
import torch.distributed as dist

import tilelang.distributed.allocator as allocator_mod
import tilelang.distributed.shared_memory as shared_memory_mod
import tilelang.testing
from tilelang import tvm
from testing.python.distributed._utils import distributed_test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _libcudart():
    return ctypes.CDLL(ctypes.util.find_library("cudart") or "libcudart.so")


def _skip_if_no_fabric():
    from tilelang.distributed.shared_memory import _supports_vmm_fabric

    if not _supports_vmm_fabric():
        pytest.skip("VMM fabric unavailable; check GPU, driver, and IMEX configuration")


# ---------------------------------------------------------------------------
# Single-GPU tests
# ---------------------------------------------------------------------------


@tilelang.testing.requires_cuda
def test_supports_fabric():
    from tilelang.distributed.shared_memory import _supports_vmm_fabric

    result = _supports_vmm_fabric()
    assert isinstance(result, bool)


def test_resolve_use_vmm_defaults_to_supported_fabric(monkeypatch):
    monkeypatch.delenv("TILESCALE_USE_VMM", raising=False)
    monkeypatch.setattr(allocator_mod, "_supports_vmm_fabric", lambda: True)

    assert allocator_mod._resolve_use_vmm(None, is_distributed=True)
    assert not allocator_mod._resolve_use_vmm(None, is_distributed=False)
    assert not allocator_mod._resolve_use_vmm(False, is_distributed=True)


def test_resolve_use_vmm_falls_back_without_fabric(monkeypatch):
    monkeypatch.delenv("TILESCALE_USE_VMM", raising=False)
    monkeypatch.setattr(allocator_mod, "_supports_vmm_fabric", lambda: False)

    assert not allocator_mod._resolve_use_vmm(None, is_distributed=True)
    assert allocator_mod._resolve_use_vmm(True, is_distributed=True)


def test_resolve_use_vmm_env_override(monkeypatch):
    monkeypatch.setenv("TILESCALE_USE_VMM", "0")
    assert not allocator_mod._resolve_use_vmm(None, is_distributed=True)

    monkeypatch.setenv("TILESCALE_USE_VMM", "1")
    assert allocator_mod._resolve_use_vmm(None, is_distributed=False)


def test_load_cudart_never_falls_back_to_cuda_driver(monkeypatch):
    lookups = []
    loads = []

    def find_library(name):
        lookups.append(name)
        return None

    def load_library(name):
        loads.append(name)
        raise OSError("not found")

    monkeypatch.setattr(allocator_mod.ctypes.util, "find_library", find_library)
    monkeypatch.setattr(allocator_mod.ctypes, "CDLL", load_library)

    with pytest.raises(RuntimeError, match="libcudart"):
        allocator_mod._load_cudart()

    assert lookups == ["cudart"]
    assert loads == ["libcudart.so"]


def test_tensor_from_ptr_rejects_unimplemented_ownership_transfer():
    from tilelang.distributed.shared_memory import tensor_from_ptr

    with pytest.raises(NotImplementedError, match="ownership transfer"):
        tensor_from_ptr(1, [1], take_ownership=True)


@pytest.mark.parametrize(
    ("function_name", "args", "message"),
    [
        ("_vmm_malloc", (-1,), "size must be > 0"),
        ("_vmm_free", (0,), "ptr must be a non-zero address"),
        ("_create_vmm_handle", (-1,), "ptr must be a non-zero address"),
        ("_close_vmm_handle", (1,), "ptr must be pointer-aligned"),
        ("_create_ipc_handle", (0,), "ptr must be a non-zero address"),
        ("_close_ipc_handle", (-1,), "ptr must be a non-zero address"),
        ("_mc_create", (1, -1), "num_devices must be > 0"),
        ("_mc_export_handle", (0,), "mc_handle must be a non-zero handle"),
        ("_mc_add_device", (1, -1), "device_id must be >= 0"),
        ("_mc_bind_mem", (1, 0, 1), "ptr must be a non-zero address"),
        ("_mc_bind_mem", (1, 8, -1), "size must be > 0"),
        ("_mc_map", (1, -1, 1), "size must be > 0"),
        ("_mc_map", (1, 1, -1), "num_devices must be > 0"),
        ("_mc_release_handle", (0,), "mc_handle must be a non-zero handle"),
        ("_mc_unmap", (0, 1, 1), "mc_ptr must be a non-zero address"),
        ("_mc_unmap", (8, -1, 1), "size must be > 0"),
        ("_mc_get_aligned_size", (-1, 1), "size must be > 0"),
        ("_mc_get_aligned_size", (1, -1), "num_devices must be > 0"),
    ],
)
def test_shared_memory_ffi_rejects_invalid_integer_boundaries(function_name, args, message):
    operation = getattr(shared_memory_mod, function_name)

    with pytest.raises(tvm.error.InternalError, match=message):
        operation(*args)


@pytest.mark.parametrize(
    ("function_name", "handle_bytes"),
    [
        ("_open_vmm_handle", b""),
        ("_open_ipc_handle", b""),
        ("_mc_import_handle", b""),
    ],
)
def test_shared_memory_ffi_rejects_malformed_handles(function_name, handle_bytes):
    operation = getattr(shared_memory_mod, function_name)

    with pytest.raises(tvm.error.InternalError, match="handle_bytes must contain exactly"):
        operation(handle_bytes)


def test_open_vmm_handle_rejects_zero_serialized_size():
    serialized_size_bytes = ctypes.sizeof(ctypes.c_size_t)
    serialized_handle = bytes(serialized_size_bytes + 64)

    with pytest.raises(tvm.error.InternalError, match="serialized allocation size must be > 0"):
        shared_memory_mod._open_vmm_handle(serialized_handle)


@pytest.mark.parametrize(
    ("function_name", "rank", "num_ranks", "table_address", "packed", "message"),
    [
        ("_sync_vmm_handles_raw", 0, 0, 8, b"", "num_ranks must be > 0"),
        ("_sync_ipc_handles_raw", -1, 1, 8, bytes(64), "rank must be >= 0"),
        (
            "_sync_vmm_handles_raw",
            1,
            1,
            8,
            bytes(ctypes.sizeof(ctypes.c_size_t) + 64),
            "rank must be smaller than num_ranks",
        ),
        (
            "_sync_ipc_handles_raw",
            0,
            1,
            0,
            bytes(64),
            "buffer_ptrs_gpu_addr must be a non-zero address",
        ),
        ("_sync_vmm_handles_raw", 0, 1, 8, b"", "packed_handles must contain exactly"),
        ("_sync_ipc_handles_raw", 0, 1, 8, b"", "packed_handles must contain exactly"),
    ],
)
def test_sync_ffi_rejects_invalid_rank_address_and_packed_handles(function_name, rank, num_ranks, table_address, packed, message):
    operation = getattr(shared_memory_mod, function_name)

    with pytest.raises(tvm.error.InternalError, match=message):
        operation(rank, num_ranks, table_address, packed)


@pytest.mark.parametrize(
    ("use_vmm", "close_name"),
    [(False, "_close_ipc_handle"), (True, "_close_vmm_handle")],
)
def test_allocator_free_closes_nonlocal_peer_mappings(monkeypatch, use_vmm, close_name):
    closed = []
    monkeypatch.setattr(allocator_mod, close_name, closed.append)

    allocator = object.__new__(allocator_mod.BaseAllocator)
    allocator._device = None
    allocator._local_rank = 1
    allocator._num_local_ranks = 3
    allocator._use_vmm = use_vmm
    allocator._use_multicast = False
    allocator._mcast_base_ptr = 0
    allocator._mcast_phys_ptr = 0
    allocator._peer_ptr_values = [101, 202, 303]
    allocator._buffer_ptrs = object()
    allocator._base_ptr = ctypes.c_void_p(0)

    allocator._free()

    assert closed == [101, 303]
    assert allocator._peer_ptr_values == []
    assert allocator._buffer_ptrs is None


def test_allocator_close_uses_two_phase_collective_teardown(monkeypatch):
    events = []
    group = object()
    allocator = object.__new__(allocator_mod.BaseAllocator)
    allocator._closed = False
    allocator._initialized = True
    allocator._is_distributed = True
    allocator._lock = threading.RLock()
    allocator._group = group
    allocator._group_size = 2
    allocator._device = 3
    allocator._set_device = lambda operation: events.append(("set_device", operation))
    allocator._free_remote_mappings = lambda: events.append(("free_remote", None))
    allocator._free_local_allocations = lambda: events.append(("free_local", None))

    def collective_stage(stage, operation):
        events.append(("collective_stage", stage))
        return operation()

    allocator._collective_stage = collective_stage

    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append(("synchronize", device)))

    allocator.close()
    allocator.close()

    assert events == [
        ("collective_stage", "select allocator device for close"),
        ("set_device", "allocator close"),
        ("collective_stage", "synchronize device work"),
        ("synchronize", 3),
        ("collective_stage", "release imported mappings"),
        ("free_remote", None),
        ("collective_stage", "release owned allocations"),
        ("free_local", None),
    ]
    assert allocator._closed
    assert not allocator._initialized


def test_failed_close_disables_allocation_and_can_be_retried(monkeypatch):
    events = []
    fail_remote_stage = True
    allocator = object.__new__(allocator_mod.BaseAllocator)
    allocator._closed = False
    allocator._initialized = True
    allocator._is_distributed = True
    allocator._lock = threading.RLock()
    allocator._group = object()
    allocator._group_size = 2
    allocator._device = 0
    allocator._set_device = lambda operation: events.append(("set_device", operation))
    allocator._free_remote_mappings = lambda: events.append(("free_remote", None))
    allocator._free_local_allocations = lambda: events.append(("free_local", None))

    def collective_stage(stage, operation):
        nonlocal fail_remote_stage
        events.append(("collective_stage", stage))
        result = operation()
        if stage == "release imported mappings" and fail_remote_stage:
            fail_remote_stage = False
            raise RuntimeError("injected peer close failure")
        return result

    allocator._collective_stage = collective_stage
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)

    with pytest.raises(RuntimeError, match="injected peer close failure"):
        allocator.close()

    assert not allocator._initialized
    assert not allocator._closed
    with pytest.raises(RuntimeError, match="closed or uninitialized"):
        allocator._allocate_tensor((1,), torch.float32)
    allocator._use_multicast = True
    with pytest.raises(RuntimeError, match="closed or uninitialized"):
        allocator._allocate_mcast_tensor((1,), torch.float32)

    allocator.close()
    assert allocator._closed
    assert not allocator._initialized
    assert events.count(("free_remote", None)) == 2
    assert events.count(("free_local", None)) == 1


def test_peer_mapping_cleanup_can_resume_without_double_close(monkeypatch):
    calls = []
    fail_once = True

    def close_handle(ptr):
        nonlocal fail_once
        calls.append(ptr)
        if ptr == 303 and fail_once:
            fail_once = False
            raise RuntimeError("close failed")

    monkeypatch.setattr(allocator_mod, "_close_ipc_handle", close_handle)

    allocator = object.__new__(allocator_mod.BaseAllocator)
    allocator._closed = True
    allocator._local_rank = 1
    allocator._use_vmm = False
    allocator._mcast_base_ptr = 0
    allocator._mcast_handle = 0
    allocator._peer_ptr_values = [101, 202, 303]
    allocator._buffer_ptrs = object()

    with pytest.raises(RuntimeError, match="close failed"):
        allocator._free_remote_mappings()

    assert allocator._peer_ptr_values == [0, 202, 303]
    allocator._free_remote_mappings()
    assert calls == [101, 303, 303]
    assert allocator._peer_ptr_values == []


def test_vmm_mapping_cleanup_consumes_pointer_before_ffi_call(monkeypatch):
    calls = []

    def close_handle(ptr):
        calls.append(ptr)
        raise RuntimeError("close failed after RAII cleanup")

    monkeypatch.setattr(allocator_mod, "_close_vmm_handle", close_handle)

    allocator = object.__new__(allocator_mod.BaseAllocator)
    allocator._local_rank = 0
    allocator._use_vmm = True
    allocator._mcast_base_ptr = 0
    allocator._mcast_handle = 0
    allocator._peer_ptr_values = [101, 202]
    allocator._buffer_ptrs = object()

    with pytest.raises(RuntimeError, match="RAII cleanup"):
        allocator._free_remote_mappings()

    assert allocator._peer_ptr_values == [101, 0]
    allocator._free_remote_mappings()
    assert calls == [202]


def test_vmm_local_cleanup_consumes_pointer_before_ffi_call(monkeypatch):
    def free(_ptr):
        raise RuntimeError("free failed after RAII cleanup")

    monkeypatch.setattr(allocator_mod, "_vmm_free", free)

    allocator = object.__new__(allocator_mod.BaseAllocator)
    allocator._use_vmm = True
    allocator._use_multicast = False
    allocator._mcast_phys_ptr = 0
    allocator._base_ptr = ctypes.c_void_p(123)
    allocator._ptr = ctypes.c_void_p(123)
    allocator._table = object()

    with pytest.raises(RuntimeError, match="RAII cleanup"):
        allocator._free_local_allocations()

    assert not allocator._base_ptr.value
    assert not allocator._ptr.value


def test_cuda_free_failure_preserves_pointer_for_retry(monkeypatch):
    class FailingCUDART:
        @staticmethod
        def cudaFree(_ptr):
            return 7

        @staticmethod
        def cudaGetErrorString(_rc):
            return b"failure"

    monkeypatch.setattr(allocator_mod, "_libcudart", FailingCUDART())

    allocator = object.__new__(allocator_mod.BaseAllocator)
    allocator._closed = True
    allocator._use_vmm = False
    allocator._use_multicast = False
    allocator._mcast_phys_ptr = 0
    allocator._base_ptr = ctypes.c_void_p(123)
    allocator._ptr = ctypes.c_void_p(123)
    allocator._table = object()

    with pytest.raises(RuntimeError, match="cudaFree failed"):
        allocator._free_local_allocations()

    assert allocator._base_ptr.value == 123
    assert allocator._ptr.value == 123


def test_multicast_adds_allocator_device_not_local_rank(monkeypatch):
    added = []
    allocator = object.__new__(allocator_mod.BaseAllocator)
    allocator._closed = True
    allocator.size = 1024
    allocator._mcast_size_requested = 1024
    allocator._num_local_ranks = 2
    allocator._local_rank = 0
    allocator._device = 5
    allocator._group = object()
    allocator._group_size = 2
    allocator._group_root_global_rank = 0
    allocator._mcast_handle = 0
    allocator._mcast_aligned_size = 2048
    allocator._collective_stage = lambda _stage, operation: operation()

    monkeypatch.setattr(allocator_mod, "_vmm_malloc", lambda size: 111)
    monkeypatch.setattr(allocator_mod, "_mc_create", lambda size, count: 222)
    monkeypatch.setattr(allocator_mod, "_mc_export_handle", lambda handle: b"handle")
    monkeypatch.setattr(allocator_mod, "_mc_add_device", lambda handle, device: added.append((handle, device)))
    monkeypatch.setattr(allocator_mod, "_mc_bind_mem", lambda *args: None)
    monkeypatch.setattr(allocator_mod, "_mc_map", lambda *args: 333)
    monkeypatch.setattr(allocator_mod, "_mc_release_handle", lambda handle: None)
    monkeypatch.setattr(dist, "broadcast_object_list", lambda *args, **kwargs: None)

    allocator._init_multicast_buffer()

    assert added == [(222, 5)]
    assert allocator._mcast_handle == 0


def test_allocator_rejects_allocation_after_close():
    allocator = object.__new__(allocator_mod.BaseAllocator)
    allocator._lock = threading.RLock()
    allocator._closed = True
    allocator._initialized = False

    with pytest.raises(RuntimeError, match="closed or uninitialized"):
        allocator._allocate_tensor((1,), torch.float32)


def test_collective_stage_propagates_peer_failure(monkeypatch):
    allocator = object.__new__(allocator_mod.BaseAllocator)
    allocator._group = object()
    allocator._group_size = 2

    def all_gather_object(statuses, local_status, group):
        assert local_status is None
        assert group is allocator._group
        statuses[:] = [None, "RuntimeError: injected peer failure"]

    monkeypatch.setattr(dist, "all_gather_object", all_gather_object)

    with pytest.raises(RuntimeError, match="rank 1: RuntimeError: injected peer failure"):
        allocator._collective_stage("fault injection", lambda: 42)


def test_constructor_rolls_back_local_resource_after_peer_stage_failure(monkeypatch):
    events = []

    monkeypatch.setattr(allocator_mod, "parse_device", lambda _device: 0)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(dist, "get_global_rank", lambda _group, _rank: 0)
    monkeypatch.setattr(
        allocator_mod.BaseAllocator,
        "_validate_distributed_configuration",
        lambda self, _rank: setattr(self, "_device_ids", [0, 1]),
    )
    monkeypatch.setattr(allocator_mod.BaseAllocator, "_set_device", lambda *_args: None)

    def allocate_base(self):
        events.append("allocate base")
        self._base_ptr.value = 123
        self._ptr.value = 123

    def free_remote(self):
        events.append("free remote")

    def free_local(self):
        events.append("free local")
        assert self._base_ptr.value == 123
        self._base_ptr.value = 0
        self._ptr.value = 0

    def collective_stage(self, stage, operation):
        events.append(stage)
        result = operation()
        if stage == "allocate base storage":
            raise RuntimeError("rank 1: injected allocation failure")
        return result

    monkeypatch.setattr(allocator_mod.BaseAllocator, "_alloc_base", allocate_base)
    monkeypatch.setattr(allocator_mod.BaseAllocator, "_free_remote_mappings", free_remote)
    monkeypatch.setattr(allocator_mod.BaseAllocator, "_free_local_allocations", free_local)
    monkeypatch.setattr(allocator_mod.BaseAllocator, "_collective_stage", collective_stage)

    with pytest.raises(RuntimeError, match="rank 1: injected allocation failure"):
        allocator_mod.BaseAllocator(
            1024,
            device=0,
            is_distributed=True,
            local_rank=0,
            num_local_ranks=2,
            group=object(),
            use_vmm=False,
        )

    assert events == [
        "validate local configuration",
        "select allocator device",
        "resolve process-group root",
        "allocate base storage",
        "allocate base",
        "rollback imported mappings",
        "free remote",
        "rollback owned allocations",
        "free local",
    ]


def test_resolve_group_root_uses_nonzero_global_rank(monkeypatch):
    group = object()
    allocator = object.__new__(allocator_mod.BaseAllocator)
    allocator._group = group
    allocator._group_root_global_rank = 0

    def get_global_rank(actual_group, group_rank):
        assert actual_group is group
        assert group_rank == 0
        return 7

    monkeypatch.setattr(dist, "get_global_rank", get_global_rank)

    allocator._resolve_group_root()

    assert allocator._group_root_global_rank == 7


def test_failed_remote_rollback_retains_owned_allocations():
    events = []
    allocator = object.__new__(allocator_mod.BaseAllocator)
    allocator._is_distributed = True
    allocator._construction_collectives_started = True
    allocator._free_remote_mappings = lambda: events.append("free remote")
    allocator._free_local_allocations = lambda: events.append("free local")

    def collective_stage(stage, operation):
        operation()
        if stage == "rollback imported mappings":
            raise RuntimeError("rank 1 still has an imported mapping")

    allocator._collective_stage = collective_stage

    rollback_error = allocator._rollback_failed_initialization()

    assert "intentionally retained" in str(rollback_error)
    assert events == ["free remote"]


def test_peer_import_state_survives_peer_stage_failure(monkeypatch):
    class FakePointerTable:
        def __init__(self):
            self.values = None

        def copy_(self, values):
            self.values = values.tolist()

    pointer_table = FakePointerTable()
    closed = []
    allocator = object.__new__(allocator_mod.BaseAllocator)
    allocator._group = object()
    allocator._group_size = 2
    allocator._local_rank = 0
    allocator._device = 0
    allocator._use_vmm = False
    allocator._base_ptr = ctypes.c_void_p(111)
    allocator._mcast_base_ptr = 0
    allocator._mcast_handle = 0
    allocator._peer_ptr_values = []
    allocator._buffer_ptrs = None

    def all_gather_object(values, local_value, group):
        assert group is allocator._group
        values[:] = [local_value, b"peer handle"]

    def collective_stage(stage, operation):
        result = operation()
        if stage == "import allocation handles":
            raise RuntimeError("rank 1: injected post-import failure")
        return result

    monkeypatch.setattr(dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(
        allocator_mod.torch,
        "empty",
        lambda *_args, **_kwargs: pointer_table,
    )
    monkeypatch.setattr(allocator_mod, "_create_ipc_handle", lambda _ptr: b"local handle")
    monkeypatch.setattr(allocator_mod, "_open_ipc_handle", lambda _handle: 222)
    monkeypatch.setattr(allocator_mod, "_close_ipc_handle", closed.append)
    allocator._collective_stage = collective_stage

    with pytest.raises(RuntimeError, match="post-import failure"):
        allocator._init_table()

    assert allocator._peer_ptr_values == [111, 222]
    assert pointer_table.values == [111, 222]
    allocator._free_remote_mappings()
    assert closed == [222]
    assert allocator._peer_ptr_values == []


def test_allocator_lock_serializes_allocation(monkeypatch):
    allocator = object.__new__(allocator_mod.BaseAllocator)
    allocator._lock = threading.RLock()
    allocator._closed = False
    allocator._initialized = True
    allocator._align = 256
    allocator._base_ptr = ctypes.c_void_p(1000)
    allocator._ptr = ctypes.c_void_p(1000)
    allocator._device = 0
    allocator._group_size = 1
    allocator.size = 1024

    monkeypatch.setattr(allocator_mod, "tensor_from_ptr", lambda ptr, *_args: ptr)

    started = threading.Event()
    finished = threading.Event()
    result = []

    def allocate():
        started.set()
        result.append(allocator._allocate_tensor((1,), torch.float32))
        finished.set()

    with allocator._lock:
        thread = threading.Thread(target=allocate)
        thread.start()
        assert started.wait(timeout=1)
        assert not finished.is_set()

    thread.join(timeout=1)
    assert not thread.is_alive()
    assert result == [1000]
    assert allocator.ptr == 1256


@tilelang.testing.requires_cuda
def test_vmm_malloc_free():
    torch.cuda.set_device(0)
    _skip_if_no_fabric()
    from tilelang.distributed.shared_memory import _vmm_free, _vmm_malloc

    size = 1024 * 1024  # 1 MB
    ptr = _vmm_malloc(size)
    assert ptr != 0, "vmm_malloc returned null"

    lib = _libcudart()
    rc = lib.cudaMemset(ctypes.c_void_p(ptr), 0, ctypes.c_size_t(size))
    assert rc == 0, f"cudaMemset on VMM pointer failed: {rc}"

    _vmm_free(ptr)


@tilelang.testing.requires_cuda
def test_vmm_handle_export_import():
    torch.cuda.set_device(0)
    _skip_if_no_fabric()
    from tilelang.distributed.shared_memory import (
        _close_vmm_handle,
        _create_vmm_handle,
        _open_vmm_handle,
        _vmm_free,
        _vmm_malloc,
    )

    size = 4096
    ptr = _vmm_malloc(size)
    assert ptr != 0

    lib = _libcudart()
    lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.cudaMemcpy.restype = ctypes.c_int

    pattern = (ctypes.c_uint8 * size)(*([0xAB] * size))
    rc = lib.cudaMemcpy(ctypes.c_void_p(ptr), ctypes.byref(pattern), size, 1)
    assert rc == 0, f"cudaMemcpy H2D failed: {rc}"

    handle = _create_vmm_handle(ptr)
    assert len(handle) > 0, "handle is empty"

    ptr2 = _open_vmm_handle(handle)
    assert ptr2 != 0, "open_vmm_handle returned null"

    readback = (ctypes.c_uint8 * size)()
    rc = lib.cudaMemcpy(ctypes.byref(readback), ctypes.c_void_p(ptr2), size, 2)
    assert rc == 0, f"cudaMemcpy D2H failed: {rc}"
    assert all(b == 0xAB for b in readback), "Data mismatch after handle export/import"

    _close_vmm_handle(ptr2)
    _vmm_free(ptr)


# ---------------------------------------------------------------------------
# Multi-GPU worker functions (called by spawn)
# ---------------------------------------------------------------------------


@distributed_test(nprocs=4, require_fabric=True)
def test_distributed_vmm(local_rank: int, num_ranks: int):
    from tilelang.distributed.host import init_dist
    from tilelang.distributed.allocator import BaseAllocator

    _, _, group = init_dist(local_rank, num_ranks)

    allocator = BaseAllocator(
        size=1024 * 1024,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group,
    )

    assert allocator.initialized()
    assert allocator._use_vmm
    assert allocator._buffer_ptrs is not None
    assert allocator._buffer_ptrs.shape[0] == num_ranks
    assert allocator._buffer_ptrs[local_rank].item() != 0

    t = allocator._allocate_tensor((256,), torch.float32)
    t.fill_(float(local_rank + 1))
    torch.cuda.synchronize()
    dist.barrier()
    allocator.close()
    dist.destroy_process_group()


@distributed_test(nprocs=4)
def test_distributed_ipc_fallback(local_rank: int, num_ranks: int):
    from tilelang.distributed.allocator import BaseAllocator
    from tilelang.distributed.host import init_dist

    _, _, group = init_dist(local_rank, num_ranks)

    allocator = BaseAllocator(
        size=1024 * 1024,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group,
        use_vmm=False,
    )

    assert allocator.initialized()
    assert not allocator._use_vmm
    assert allocator._buffer_ptrs is not None
    assert allocator._buffer_ptrs.shape[0] == num_ranks

    remote_rank = (local_rank + 1) % num_ranks
    assert allocator._buffer_ptrs[remote_rank].item() != 0, f"Remote rank {remote_rank} pointer is zero"

    t = allocator._allocate_tensor((256,), torch.float32)
    t.fill_(float(local_rank + 1))
    torch.cuda.synchronize()
    dist.barrier(group)
    allocator.close()
    dist.destroy_process_group()


@distributed_test(nprocs=4)
def test_distributed_allocator_collective_fault_recovery(local_rank: int, num_ranks: int):
    from tilelang.distributed.allocator import BaseAllocator
    from tilelang.distributed.host import init_dist

    _, _, group = init_dist(local_rank, num_ranks)

    # A local validation failure must reach every rank before any rank enters
    # the next collective or allocates device storage.
    bad_size = 0 if local_rank == 0 else 1024 * 1024
    with pytest.raises(RuntimeError, match="validate local configuration"):
        BaseAllocator(
            size=bad_size,
            device="cuda",
            is_distributed=True,
            local_rank=local_rank,
            num_local_ranks=num_ranks,
            group=group,
            use_vmm=False,
        )

    allocator = BaseAllocator(
        size=1024 * 1024,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group,
        use_vmm=False,
    )

    original_free_remote_mappings = allocator._free_remote_mappings
    if local_rank == 0:

        def fail_after_releasing_remote_mappings():
            original_free_remote_mappings()
            raise RuntimeError("injected close failure")

        allocator._free_remote_mappings = fail_after_releasing_remote_mappings

    with pytest.raises(RuntimeError, match="release imported mappings"):
        allocator.close()

    assert not allocator.initialized()
    assert not allocator._closed
    with pytest.raises(RuntimeError, match="closed or uninitialized"):
        allocator._allocate_tensor((1,), torch.float32)

    allocator._free_remote_mappings = original_free_remote_mappings
    allocator.close()
    assert allocator._closed
    dist.destroy_process_group()


if __name__ == "__main__":
    tilelang.testing.main()
