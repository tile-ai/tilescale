"""Standalone probe: can a TileScale arena be registered as an NCCL GIN window?

This is deliberately not a pytest test. It needs two nodes, RoCE, and a
GIN-capable NCCL (>= 2.28.7), so it is run by hand before trusting the
allocator's window registration path.

It answers three questions, in order of risk:

1. Does ``ncclCommWindowRegister`` accept a **VMM-mapped** arena? This is the
   allocator's default backend when fabric is supported.
2. Does it accept a plain ``cudaMalloc`` arena? The fallback backend.
3. Can a GIN devcomm be created on a communicator that a window is then
   registered on, in the order the allocator uses?

Measured on <node> with NCCL 2.28.9 (single node, 2 GPUs) on 2026-07-31:

    ncclMemAlloc    COLL_SYMMETRIC  rc=0   registers
    ncclMemAlloc    flags=0         rc=0   registers
    cudaMalloc      COLL_SYMMETRIC  rc=1   invalid argument
    cudaMalloc      flags=0         rc=1   invalid argument
    vmm POSIX_FD    COLL_SYMMETRIC  rc=0   registers

So registration needs the driver-level allocation that the VMM path (or
``ncclMemAlloc``) produces; a ``cudaMalloc`` arena cannot be registered at all,
under either flag value. The allocator therefore requires VMM for GIN. Note that
requesting ``CU_MEM_HANDLE_TYPE_FABRIC`` returned ``CUDA_ERROR_NOT_PERMITTED``
without an IMEX channel configured (see ``scripts/conf_vmm.sh``), so this probe
falls back to a POSIX-FD handle type when fabric is unavailable.

Also measured there, and the reason the devcomm stage uses a private process
group: ``ncclDevCommCreate`` **segfaults** on a torch communicator that has
already carried a collective, rather than returning an error. Orderings, one
rank::

    devcomm                      -> OK
    window, devcomm              -> OK
    devcomm, window              -> OK
    devcomm, collective, window  -> OK   (devcomm survives later collectives)
    collective, devcomm          -> SEGFAULT
    collective, window, devcomm  -> SEGFAULT

``NCCL_WIN_ENABLE=0`` or ``NCCL_CUMEM_ENABLE=0`` turns the crash into the real
diagnosis: ``Communicator does not support symmetric memory!``

Launch (per node, NNODES=2, 1 rank per node is enough to cross the network):

    NCCL_IB_DISABLE=0 MASTER_ADDR=<node-ip> MASTER_PORT=29511 \
    WORLD_SIZE=2 LOCAL_WORLD_SIZE=1 NNODES=2 NODE_RANK=<0|1> RANK=<0|1> \
    python testing/python/distributed/probe_gin_window.py

``init_dist`` defaults ``NCCL_IB_DISABLE=1``, so it must be set to 0 explicitly
for any inter-node run.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import importlib.util
import os
import traceback

import torch
import torch.distributed as dist

# testing/python/distributed/<this file> -> repo root is three levels up.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
for _ in range(3):
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)

# Load nccl_window.py directly rather than via `import tilelang`: the probe must
# run without the native tilelang build, and this module depends only on ctypes
# and torch. It is still the exact code the allocator uses.
_spec = importlib.util.spec_from_file_location(
    "tilescale_probe_nccl_window",
    os.path.join(_REPO_ROOT, "tilelang", "distributed", "nccl_window.py"),
)
win = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(win)

from cuda.bindings import driver as cuda  # noqa: E402

ARENA_SIZE = 64 << 20  # 64 MiB, a multiple of NCCL_WIN_REQUIRED_ALIGNMENT


def _log(rank: int, message: str) -> None:
    print(f"[rank {rank}] {message}", flush=True)


def _cudart():
    lib = ctypes.CDLL(ctypes.util.find_library("cudart") or "libcudart.so")
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaFree.argtypes = [ctypes.c_void_p]
    return lib


def _cu(result):
    """Unwrap a cuda-python return tuple, raising on a driver error."""
    err, *rest = result if isinstance(result, tuple) else (result,)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error: {err}")
    if not rest:
        return None
    return rest[0] if len(rest) == 1 else tuple(rest)


# Mirrors vmm_malloc_impl in src/shared_memory/shared_memory.cc: pinned device
# memory with fabric handle export, mapped into a reserved VA range. Reproduced
# here (rather than imported) so the probe needs no native build.
_vmm_maps: dict[int, int] = {}


def vmm_alloc(size: int, handle_type=None) -> int:
    device = _cu(cuda.cuCtxGetDevice())
    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device
    prop.requestedHandleTypes = (
        handle_type
        if handle_type is not None else cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC)

    granularity = _cu(cuda.cuMemGetAllocationGranularity(
        prop, cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM))
    aligned = ((size + granularity - 1) // granularity) * granularity

    handle = _cu(cuda.cuMemCreate(aligned, prop, 0))
    ptr = _cu(cuda.cuMemAddressReserve(aligned, granularity, 0, 0))
    _cu(cuda.cuMemMap(ptr, aligned, 0, handle, 0))

    access = cuda.CUmemAccessDesc()
    access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = device
    access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _cu(cuda.cuMemSetAccess(ptr, aligned, [access], 1))
    _cu(cuda.cuMemRelease(handle))

    value = int(ptr)
    _vmm_maps[value] = aligned
    return value


def vmm_free(ptr: int) -> None:
    size = _vmm_maps.pop(ptr, 0)
    if not size:
        return
    cuda.cuMemUnmap(cuda.CUdeviceptr(ptr), size)
    cuda.cuMemAddressFree(cuda.CUdeviceptr(ptr), size)


def supports_vmm_fabric() -> bool:
    """Probe whether fabric handles can be exported and reimported here."""
    try:
        device = _cu(cuda.cuCtxGetDevice())
        prop = cuda.CUmemAllocationProp()
        prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device
        prop.requestedHandleTypes = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        granularity = _cu(cuda.cuMemGetAllocationGranularity(
            prop, cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM))
        handle = _cu(cuda.cuMemCreate(granularity, prop, 0))
        try:
            _cu(cuda.cuMemExportToShareableHandle(
                handle, cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC, 0))
        finally:
            cuda.cuMemRelease(handle)
        return True
    except Exception:  # noqa: BLE001 - probe result is the return value
        return False


def _vmm_handle_type(rank: int):
    """Handle type for VMM arenas: None means fabric, the allocator's default.

    Fabric handles need an IMEX channel, so fall back to a POSIX FD where it is
    unavailable -- that still exercises the VMM mapping, which is the property
    window registration actually requires.
    """
    if supports_vmm_fabric():
        return None
    _log(rank, "vmm-fabric: unavailable (no IMEX channel); using a POSIX-FD handle type")
    return cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR


def probe_backend(name: str, alloc, free, comm_ptr: int, rank: int) -> bool:
    """Allocate an arena with one backend and try to register it as a window."""
    base = 0
    window = 0
    try:
        base = alloc()
        aligned = base % win.NCCL_WIN_REQUIRED_ALIGNMENT == 0
        _log(rank, f"{name}: base={base:#x} aligned={aligned}")
        if not aligned:
            _log(rank, f"{name}: FAIL base is not {win.NCCL_WIN_REQUIRED_ALIGNMENT}-byte aligned")
            return False

        # Every rank must reach the collective registration together.
        dist.barrier()
        window = win.register_window(comm_ptr, base, ARENA_SIZE, win.NCCL_WIN_COLL_SYMMETRIC)
        _log(rank, f"{name}: PASS registered window={window:#x}")
        return True
    except Exception:  # noqa: BLE001 - the probe reports rather than raises
        _log(rank, f"{name}: FAIL\n{traceback.format_exc()}")
        return False
    finally:
        if window:
            try:
                dist.barrier()
                win.deregister_window(comm_ptr, window)
                _log(rank, f"{name}: deregistered")
            except Exception:  # noqa: BLE001
                _log(rank, f"{name}: deregister failed\n{traceback.format_exc()}")
        if base:
            free(base)


def probe_devcomm(rank: int, local_rank: int) -> bool:
    """Create a GIN devcomm the way the allocator does, then register a window on it.

    This is the ordering constraint the allocator exists to satisfy, so the probe
    reproduces it rather than testing ``ncclDevCommCreate`` in isolation.
    ``ncclDevCommCreate`` requires a communicator that still supports symmetric
    memory, and a torch communicator loses that after its **first collective** --
    the call then *segfaults* instead of returning an error, so it cannot be
    attempted and recovered from. WORLD has already run a collective by the time
    this is reached, which is exactly the case that crashes.

    The fix, and what is checked here: a private process group whose devcomm is
    created before any collective touches it. ``device_id`` makes that group's
    communicator eager, so ``_comm_ptr()`` is valid immediately -- without it the
    comm is created lazily on first use, and that first use would be the very
    collective that invalidates it.
    """
    gin_group = None
    dev_comm = None
    window = 0
    comm_ptr = 0
    base = 0
    try:
        gin_group = dist.new_group(backend="nccl", device_id=torch.device("cuda", local_rank))
        comm_ptr = win.get_comm_ptr(gin_group)
        _log(rank, f"devcomm: private group comm_ptr={comm_ptr:#x} (eager)")
        if not comm_ptr:
            _log(rank, "devcomm: FAIL private group exposed no ncclComm_t")
            return False

        # Before anything else uses this group -- see the docstring.
        dev_comm = win.create_dev_comm(comm_ptr)
        _log(rank, f"devcomm: PASS created device_ptr={dev_comm.device_ptr:#x}")

        base = vmm_alloc(ARENA_SIZE, _vmm_handle_type(rank))
        dist.barrier()
        # The same comm the devcomm came from: a window handle is only meaningful
        # to the devcomm sharing its communicator.
        window = win.register_window(comm_ptr, base, ARENA_SIZE, win.NCCL_WIN_COLL_SYMMETRIC)
        _log(rank, f"devcomm: PASS window={window:#x} on the devcomm's comm")

        # The allocator keeps running collectives on WORLD afterwards; confirm
        # that does not disturb the devcomm it created on the private group.
        probe = torch.ones(1, device=f"cuda:{local_rank}")
        dist.all_reduce(probe)
        _log(rank, f"devcomm: PASS WORLD collectives still work after setup (sum={probe.item()})")
        return True
    except Exception:  # noqa: BLE001 - the probe reports rather than raises
        _log(rank, f"devcomm: FAIL\n{traceback.format_exc()}")
        return False
    finally:
        # Innermost first: the devcomm references the comm the window belongs to,
        # and both reference the group.
        if dev_comm is not None:
            with contextlib.suppress(Exception):
                win.destroy_dev_comm(dev_comm)
        if window:
            with contextlib.suppress(Exception):
                dist.barrier()
                win.deregister_window(comm_ptr, window)
        if base:
            with contextlib.suppress(Exception):
                vmm_free(base)
        if gin_group is not None:
            with contextlib.suppress(Exception):
                dist.destroy_process_group(gin_group)


def main() -> int:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{os.environ.get('MASTER_ADDR', '127.0.0.1')}:{os.environ.get('MASTER_PORT', '29511')}",
        world_size=world_size,
        rank=rank,
        device_id=torch.device("cuda", local_rank),
    )

    version = win.nccl_version()
    _log(rank, f"nccl_version={version} device_api={win.supports_device_api()}")
    _log(rank, f"support: {win.unavailable_reason()}")
    if not win.supports_device_api():
        _log(rank, "ABORT: no NCCL Device API; point TILESCALE_NCCL_LIB at a >= 2.28.7 libnccl")
        dist.destroy_process_group()
        return 2

    # The communicator is created lazily; force it before asking for the pointer.
    dist.all_reduce(torch.ones(1, device=f"cuda:{local_rank}"))
    comm_ptr = win.get_comm_ptr(dist.group.WORLD)
    _log(rank, f"comm_ptr={comm_ptr:#x}")
    if not comm_ptr:
        _log(rank, "ABORT: could not obtain ncclComm_t via _comm_ptr()")
        dist.destroy_process_group()
        return 2

    results = {}
    cudart = _cudart()

    def cuda_alloc() -> int:
        ptr = ctypes.c_void_p(0)
        rc = cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(ARENA_SIZE))
        if rc != 0:
            raise RuntimeError(f"cudaMalloc failed: {rc}")
        return int(ptr.value)

    # Expected to fail: recorded to catch the day NCCL starts accepting these,
    # which would let the allocator drop its VMM requirement for GIN.
    cuda_ok = probe_backend("cudaMalloc", cuda_alloc,
                            lambda p: cudart.cudaFree(ctypes.c_void_p(p)), comm_ptr, rank)
    _log(rank, f"cudaMalloc: {'registers (NCCL behaviour changed)' if cuda_ok else 'rejected, as expected'}")

    vmm_handle = _vmm_handle_type(rank)
    vmm_label = "vmm-fabric" if vmm_handle is None else "vmm-posix-fd"
    results[vmm_label] = probe_backend(
        vmm_label, lambda: vmm_alloc(ARENA_SIZE, vmm_handle), vmm_free, comm_ptr, rank)

    # Last, because it is the stage that used to segfault: a crash here takes the
    # process down, so the cheaper registration answers are already recorded.
    dist.barrier()
    results["devcomm"] = probe_devcomm(rank, local_rank)

    dist.barrier()
    _log(rank, f"summary: {results}")
    dist.destroy_process_group()
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
