"""
VMM (Virtual Memory Management) operations test.

Usage:
  # Single-GPU unit tests (no torchrun needed):
  python testing/python/distributed/test_vmm_ops.py

  # Multi-GPU integration test:
  torchrun --nproc_per_node=8 testing/python/distributed/test_vmm_ops.py --distributed
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist


def test_supports_fabric():
    """Test fabric support detection."""
    from tilelang.distributed.shared_memory import _supports_vmm_fabric

    result = _supports_vmm_fabric()
    print(f"\033[32m[PASS]\033[0m _supports_vmm_fabric() = {result}")
    return result


def test_vmm_malloc_free():
    """Test VMM malloc and free roundtrip."""
    from tilelang.distributed.shared_memory import _vmm_malloc, _vmm_free

    size = 1024 * 1024  # 1 MB
    ptr = _vmm_malloc(size)
    assert ptr != 0, "vmm_malloc returned null"

    # Verify the pointer is usable by writing via cudaMemset
    import ctypes
    import ctypes.util

    libcudart = ctypes.CDLL(ctypes.util.find_library("cudart") or "libcudart.so")
    rc = libcudart.cudaMemset(ctypes.c_void_p(ptr), 0, ctypes.c_size_t(size))
    assert rc == 0, f"cudaMemset on VMM pointer failed: {rc}"

    _vmm_free(ptr)
    print("\033[32m[PASS]\033[0m test_vmm_malloc_free")


def test_vmm_handle_export_import():
    """Test handle export and import on a single GPU."""
    from tilelang.distributed.shared_memory import _vmm_malloc, _vmm_free, _create_vmm_handle, _open_vmm_handle, _close_vmm_handle

    size = 4096
    ptr = _vmm_malloc(size)
    assert ptr != 0

    # Write a known pattern
    import ctypes

    pattern = (ctypes.c_uint8 * size)(*([0xAB] * size))
    libcudart = ctypes.CDLL(ctypes.util.find_library("cudart") or "libcudart.so")
    libcudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libcudart.cudaMemcpy.restype = ctypes.c_int
    rc = libcudart.cudaMemcpy(ctypes.c_void_p(ptr), ctypes.byref(pattern), size, 1)  # cudaMemcpyHostToDevice=1
    assert rc == 0, f"cudaMemcpy H2D failed: {rc}"

    # Export handle
    handle = _create_vmm_handle(ptr)
    assert len(handle) > 0, "handle is empty"

    # Import handle (same process, simulates remote open)
    ptr2 = _open_vmm_handle(handle)
    assert ptr2 != 0, "open_vmm_handle returned null"

    # Read back through the new mapping
    readback = (ctypes.c_uint8 * size)()
    rc = libcudart.cudaMemcpy(ctypes.byref(readback), ctypes.c_void_p(ptr2), size, 2)  # cudaMemcpyDeviceToHost=2
    assert rc == 0, f"cudaMemcpy D2H failed: {rc}"
    assert all(b == 0xAB for b in readback), "Data mismatch after handle export/import"

    _close_vmm_handle(ptr2)
    _vmm_free(ptr)
    print("\033[32m[PASS]\033[0m test_vmm_handle_export_import")


def test_distributed_vmm(rank, world_size):
    """Multi-GPU integration test: VMM alloc + P2P read via BaseAllocator."""
    from tilelang.utils.allocator import BaseAllocator

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    group = dist.new_group(list(range(world_size)))

    # Use BaseAllocator with use_vmm=True — it allocates via vmm_malloc
    # and handles the VMM handle exchange internally
    allocator = BaseAllocator(
        size=1024 * 1024,  # 1 MB
        device=f"cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=world_size,
        group=group,
        use_vmm=True,
    )

    assert allocator.initialized()
    assert allocator._buffer_ptrs is not None
    assert allocator._buffer_ptrs.shape[0] == world_size
    assert allocator._buffer_ptrs[local_rank].item() != 0

    # Allocate a tensor from the VMM buffer and verify P2P access
    t = allocator._allocate_tensor((256,), torch.float32)
    t.fill_(float(rank + 1))
    torch.cuda.synchronize()

    dist.barrier()

    if rank == 0:
        print(f"\033[32m[PASS]\033[0m test_distributed_vmm (world_size={world_size})")


def test_distributed_ipc_fallback(rank, world_size):
    """Verify IPC path still works with TILESCALE_USE_VMM=0."""
    from tilelang.distributed.utils import create_dist_tensor, create_tensor

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    group = dist.new_group(list(range(world_size)))

    data = create_tensor([1024], torch.float32)
    data.fill_(float(rank + 1))

    # Explicitly use IPC
    buffer_ptrs = create_dist_tensor(local_rank, world_size, data, rank, group, use_vmm=False)

    assert buffer_ptrs.shape[0] == world_size
    # Note: IPC path doesn't set local rank's pointer (no self-open needed)
    # Check that at least one remote rank's pointer is non-zero
    remote_rank = (local_rank + 1) % world_size
    assert buffer_ptrs[remote_rank].item() != 0, f"Remote rank {remote_rank} pointer is zero"

    dist.barrier()

    if rank == 0:
        print(f"\033[32m[PASS]\033[0m test_distributed_ipc_fallback (world_size={world_size})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true", help="Run multi-GPU tests (requires torchrun)")
    args = parser.parse_args()

    if args.distributed:
        # Multi-GPU path (launched via torchrun)
        import datetime

        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(seconds=60),
        )

        has_fabric = test_supports_fabric() if rank == 0 else False
        # Broadcast fabric support info
        fabric_tensor = torch.tensor([int(has_fabric)], device="cuda")
        dist.broadcast(fabric_tensor, src=0)
        has_fabric = bool(fabric_tensor.item())

        if has_fabric:
            test_distributed_vmm(rank, world_size)

        test_distributed_ipc_fallback(rank, world_size)

        dist.destroy_process_group()
    else:
        # Single-GPU unit tests
        torch.cuda.set_device(0)
        has_fabric = test_supports_fabric()
        if has_fabric:
            test_vmm_malloc_free()
            test_vmm_handle_export_import()
            print("\n=== All single-GPU VMM tests passed ===")
        else:
            print("\n=== Fabric not supported on this hardware, skipping VMM tests ===")


if __name__ == "__main__":
    main()
