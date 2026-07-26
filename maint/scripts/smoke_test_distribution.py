#!/usr/bin/env python3
"""Smoke-test the distributed API exported by an installed TileScale package."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version


CUDA_SHARED_MEMORY_FUNCS = (
    "tl.shared_memory.vmm_malloc",
    "tl.shared_memory.vmm_free",
    "tl.shared_memory.create_vmm_handle",
    "tl.shared_memory.open_vmm_handle",
    "tl.shared_memory.close_vmm_handle",
    "tl.shared_memory.sync_vmm_handles",
    "tl.shared_memory.create_ipc_handle",
    "tl.shared_memory.open_ipc_handle",
    "tl.shared_memory.close_ipc_handle",
    "tl.shared_memory.sync_ipc_handles",
    "tl.shared_memory.supports_vmm_fabric",
    "tl.shared_memory.supports_multicast",
    "tl.shared_memory.mc_create",
    "tl.shared_memory.mc_export_handle",
    "tl.shared_memory.mc_import_handle",
    "tl.shared_memory.mc_add_device",
    "tl.shared_memory.mc_bind_mem",
    "tl.shared_memory.mc_map",
    "tl.shared_memory.mc_release_handle",
    "tl.shared_memory.mc_unmap",
    "tl.shared_memory.mc_get_aligned_size",
)
TVM_FFI_VERSION = "0.1.11"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda-shared-memory",
        action="store_true",
        help="also require CUDA host bindings, allocator import, and shared-memory FFI symbols",
    )
    args = parser.parse_args()

    try:
        distribution_version = version("tilescale")
    except PackageNotFoundError as error:
        raise RuntimeError("the tilescale distribution is not installed") from error

    try:
        tvm_ffi_version = version("apache-tvm-ffi")
    except PackageNotFoundError as error:
        raise RuntimeError("the apache-tvm-ffi runtime dependency is not installed") from error
    if tvm_ffi_version != TVM_FFI_VERSION:
        raise RuntimeError(f"apache-tvm-ffi=={TVM_FFI_VERSION} is required by this build; found {tvm_ffi_version}")

    import tilelang
    import tilelang.distributed as distributed

    if not callable(tilelang.get_allocator):
        raise RuntimeError("tilelang.get_allocator is not callable")
    if not callable(distributed.__getattr__):
        raise RuntimeError("tilelang.distributed lazy API loader is unavailable")

    if args.require_cuda_shared_memory:
        import tvm_ffi

        from tilelang.distributed import init_dist
        from tilelang.distributed.allocator import get_allocator
        from tilelang.distributed import shared_memory  # noqa: F401

        if not callable(init_dist) or not callable(get_allocator):
            raise RuntimeError("distributed host or allocator API is unavailable")

        missing = [name for name in CUDA_SHARED_MEMORY_FUNCS if tvm_ffi.get_global_func(name, allow_missing=True) is None]
        if missing:
            raise RuntimeError(f"CUDA shared-memory FFI symbols missing from wheel: {missing}")

    print(
        f"PASS tilescale={distribution_version} tilelang={tilelang.__version__} "
        f"tvm_ffi={tvm_ffi_version} cuda_shared_memory={args.require_cuda_shared_memory}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
