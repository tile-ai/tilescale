"""NUMA-aware memory allocation for B200 dual-die GPUs.

Provides NUMA-local memory placement via latency-based die detection
and tile-interleaved data packing for optimal HBM bandwidth.

Usage:
    from tilelang.distributed.numa import NUMATensor

    # Allocate on detected NUMA die
    a_numa = NUMATensor.from_torch(a_tensor, tileK=64, tileMN=32)
    page_id = a_numa.page_id

    # Get 1D tiled view for TMA kernel
    a_packed = a_numa.as_1d_tiled()

    # Cleanup
    a_numa.free()
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from functools import lru_cache

import torch
import tvm_ffi

from tilelang.distributed.shared_memory import tensor_from_ptr

_CSRC_DIR = Path(__file__).parent / "csrc"
_BUILD_DIR = Path(__file__).parent / "build"
_REMAP_TILE_BYTES = 4 * 1024

__all__ = ["NUMATensor", "ensure_built"]


def _find_nvcc():
    for path in ["/usr/local/cuda-13.2/bin/nvcc",
                 "/usr/local/cuda-13.1/bin/nvcc",
                 "/usr/local/cuda/bin/nvcc"]:
        if os.path.isfile(path):
            return path
    import shutil
    return shutil.which("nvcc")


@lru_cache(maxsize=1)
def ensure_built() -> str:
    """Build the NUMA .so if needed. Returns path to the .so."""
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    so_path = _BUILD_DIR / "libtilescale_numa.so"
    src = _CSRC_DIR / "numa.cu"
    hdr = _CSRC_DIR / "numa.h"

    if (so_path.exists() and so_path.stat().st_mtime >= src.stat().st_mtime
            and so_path.stat().st_mtime >= hdr.stat().st_mtime):
        return str(so_path)

    nvcc = _find_nvcc()
    if nvcc is None:
        raise RuntimeError("nvcc not found. Install CUDA >= 13.1 for B200 support.")

    tvm_ffi_include = os.path.join(os.path.dirname(tvm_ffi.__file__), "include")

    cmd = [
        nvcc, str(src),
        "-shared", "-o", str(so_path),
        f"-I{tvm_ffi_include}",
        "-Xcompiler", "-fPIC",
        "-O2",
        "--std=c++17",
        "-arch=sm_100a",
    ]
    print(f"[tilescale numa] Building {so_path.name}...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"NUMA module build failed:\n{result.stderr}")
    print("[tilescale numa] Build complete.", flush=True)
    return str(so_path)


@lru_cache(maxsize=1)
def _get_mod():
    so_path = ensure_built()
    return tvm_ffi.load_module(so_path)


def _dtype_bytes(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _remap_padded_elems(numel: int, elem_bytes: int) -> int:
    elems_per_tile = _REMAP_TILE_BYTES // elem_bytes
    return ((numel + elems_per_tile - 1) // elems_per_tile) * elems_per_tile


def _check_tile_shape(tileK: int, tileMN: int, elem_bytes: int):
    tile_bytes = tileK * tileMN * elem_bytes
    if tile_bytes != _REMAP_TILE_BYTES:
        raise ValueError(
            "NUMATensor uses a 4KB NUMA remap tile; "
            f"got tileK={tileK}, tileMN={tileMN}, dtype_bytes={elem_bytes} "
            f"({tile_bytes} bytes)"
        )


class NUMATensor:
    """A NUMA-aware tensor on B200 dual-die GPU.

    Data is stored in a 1D tiled layout with NUMA-interleaved pages
    for optimal die-local access.
    """

    def __init__(self, handle: int, page_id: int, shape: tuple, dtype: torch.dtype,
                 tileK: int, tileMN: int, K: int, total_MN: int):
        self.handle = handle
        self.page_id = page_id
        self.shape = shape
        self.dtype = dtype
        self.tileK = tileK
        self.tileMN = tileMN
        self.K = K
        self.total_MN = total_MN
        self._freed = False

    @classmethod
    def from_torch(cls, src: torch.Tensor, tileK: int = 64, tileMN: int = 32) -> "NUMATensor":
        """Allocate NUMA memory and pack a torch tensor into it.

        Args:
            src: Source tensor on CUDA, shape [G, M, K] or [G*M, K] for A,
                 [G, N, K] or [G*N, K] for B.
            tileK: Tile size along K dimension (default 64).
            tileMN: Tile size along M/N dimension (default 32).
        """
        mod = _get_mod()
        src = src.contiguous().to(device="cuda")

        elem_bytes = _dtype_bytes(src.dtype)
        _check_tile_shape(tileK, tileMN, elem_bytes)
        total_elems = src.numel()
        size_bytes = _remap_padded_elems(total_elems, elem_bytes) * elem_bytes

        handle = mod["tilescale_numa_alloc"](size_bytes, elem_bytes)
        if handle == 0:
            raise RuntimeError("NUMA allocation failed (cudaMallocAndGetPageId returned -1)")

        page_id = mod["tilescale_numa_get_page_id"](handle)

        # Determine K and total_MN from shape
        if src.dim() == 3:
            G, MN, K_dim = src.shape
            total_MN_dim = G * MN
        elif src.dim() == 2:
            total_MN_dim, K_dim = src.shape
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {src.dim()}D")

        src_2d = src.reshape(total_MN_dim, K_dim)
        mod["tilescale_numa_pack"](src_2d, handle, tileK, tileMN, K_dim, total_MN_dim)

        return cls(handle, page_id, tuple(src.shape), src.dtype,
                   tileK, tileMN, K_dim, total_MN_dim)

    def to_torch(self) -> torch.Tensor:
        """Unpack NUMA tensor back to standard torch layout."""
        mod = _get_mod()
        elem_bytes = _dtype_bytes(self.dtype)
        out = torch.empty(self.total_MN, self.K, dtype=self.dtype, device="cuda")
        mod["tilescale_numa_unpack"](self.handle, out, self.tileK, self.tileMN, self.K, self.total_MN)
        return out.reshape(self.shape)

    def as_1d_tiled(self, device: int = 0) -> torch.Tensor:
        """Get a torch tensor view in 1D tiled format [tiles*32, 64] for TMA."""
        total_elems = self.total_MN * self.K
        tile_elems = self.tileMN * self.tileK  # 32 * 64 = 2048
        num_tiles = (total_elems + tile_elems - 1) // tile_elems
        dtype_str = {
            torch.bfloat16: "bfloat16",
            torch.float16: "float16",
            torch.float32: "float32",
            torch.float8_e4m3fn: "float8_e4m3fn",
        }.get(self.dtype, str(self.dtype).split(".")[-1])
        return tensor_from_ptr(self.handle, [num_tiles * self.tileMN, self.tileK],
                               dtype_str, device, False)

    def free(self):
        """Free NUMA memory."""
        if not self._freed and self.handle:
            _get_mod()["tilescale_numa_free"](self.handle)
            self._freed = True
            self.handle = 0

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    def __repr__(self):
        return (f"NUMATensor(shape={self.shape}, dtype={self.dtype}, "
                f"page_id={self.page_id}, tileK={self.tileK}, tileMN={self.tileMN})")
