from pathlib import Path
import shutil
import subprocess

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[5]
_CUTLASS_INCLUDE = _REPO_ROOT / "3rdparty/cutlass/include"


def _nvcc_compile(tmp_path, name: str, source: str):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is required to compile distributed load helpers")

    source_path = tmp_path / f"{name}.cu"
    source_path.write_text(source)
    return subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-arch=sm_90",
            "-c",
            f"-I{_REPO_ROOT / 'src'}",
            f"-I{_CUTLASS_INCLUDE}",
            str(source_path),
            "-o",
            str(tmp_path / f"{name}.o"),
        ],
        capture_output=True,
        text=True,
    )


def test_weak_load_specializations_compile_with_ptxas(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is required to compile distributed load helpers")

    source_path = tmp_path / "weak_loads.cu"
    object_path = tmp_path / "weak_loads.o"
    source_path.write_text(
        """
#include <cstdint>
#include <tl_templates/cuda/distributed/ldst.h>

__global__ void weak_loads(const uint32_t *src, uint32_t *dst) {
  uint32_t cta, gpu, sys, gpu_na, sys_na;
  tl::ld<Semantic::WEAK, Scope::CTA, false, false>(src, cta);
  tl::ld<Semantic::WEAK, Scope::GPU, false, false>(src, gpu);
  tl::ld<Semantic::WEAK, Scope::SYS, false, false>(src, sys);
  tl::ld<Semantic::WEAK, Scope::GPU, false, true>(src, gpu_na);
  tl::ld<Semantic::WEAK, Scope::SYS, false, true>(src, sys_na);
  dst[0] = cta + gpu + sys + gpu_na + sys_na;
}
"""
    )
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-arch=sm_90",
            "-c",
            f"-I{_REPO_ROOT / 'src'}",
            f"-I{_CUTLASS_INCLUDE}",
            str(source_path),
            "-o",
            str(object_path),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_ldst_covers_every_supported_width(tmp_path):
    """tl::ld/tl::st must accept all 2/4/8-byte scalar types.

    The 16-bit path once bit-cast only bfloat16 and passed other 16-bit types
    straight into the "h"/"=h" asm constraint, so float16 failed to compile
    while bfloat16 worked.
    """
    result = _nvcc_compile(
        tmp_path,
        "ldst_widths",
        """
#include <cstdint>
#include <tl_templates/cuda/distributed/ldst.h>

template <typename T> __device__ void roundtrip(const T *src, T *dst) {
  T value;
  tl::ld<Semantic::WEAK, Scope::SYS, false, false>(src, value);
  tl::st<Semantic::WEAK, Scope::SYS, false>(dst, value);
  tl::ld<Semantic::ACQUIRE, Scope::SYS, false, false>(src, value);
  tl::st<Semantic::RELEASE, Scope::SYS, false>(dst, value);
}

__global__ void ldst_widths(void *p) {
  roundtrip(static_cast<const half_t *>(p), static_cast<half_t *>(p));
  roundtrip(static_cast<const bfloat16_t *>(p), static_cast<bfloat16_t *>(p));
  roundtrip(static_cast<const uint16_t *>(p), static_cast<uint16_t *>(p));
  roundtrip(static_cast<const float *>(p), static_cast<float *>(p));
  roundtrip(static_cast<const uint32_t *>(p), static_cast<uint32_t *>(p));
  roundtrip(static_cast<const double *>(p), static_cast<double *>(p));
  roundtrip(static_cast<const uint64_t *>(p), static_cast<uint64_t *>(p));
}
""",
    )
    assert result.returncode == 0, result.stderr


def test_nc_load_is_registered_for_a_single_scope(tmp_path):
    """`ld.global.nc` is unscoped, so only one scope may be instantiable.

    Registering more than one produced byte-identical unscoped loads that
    silently discarded the requested scope.
    """
    ok = _nvcc_compile(
        tmp_path,
        "nc_ok",
        """
#include <cstdint>
#include <tl_templates/cuda/distributed/ldst.h>

__global__ void nc_ok(const uint32_t *src, uint32_t *dst) {
  uint32_t plain, no_alloc;
  tl::ld<Semantic::WEAK, Scope::GPU, true, false>(src, plain);
  tl::ld<Semantic::WEAK, Scope::GPU, true, true>(src, no_alloc);
  dst[0] = plain + no_alloc;
}
""",
    )
    assert ok.returncode == 0, ok.stderr

    rejected = _nvcc_compile(
        tmp_path,
        "nc_scoped",
        """
#include <cstdint>
#include <tl_templates/cuda/distributed/ldst.h>

__global__ void nc_scoped(const uint32_t *src, uint32_t *dst) {
  uint32_t sys;
  tl::ld<Semantic::WEAK, Scope::SYS, true, false>(src, sys);
  dst[0] = sys;
}
""",
    )
    assert rejected.returncode != 0, "a scoped non-coherent load must not compile"
    assert "unsupported configuration" in rejected.stderr


def _ld_st_kernel(dtype: str, **ld_kwargs):
    import tilelang.language as T

    @T.prim_func
    def main(A: T.Tensor((8,), dtype), B: T.Tensor((8,), dtype)):
        with T.Kernel(1, threads=1):
            value = T.alloc_local((1,), dtype)
            T.ld(A[0], value[0], **ld_kwargs)
            T.st(B[0], value[0], scope="sys", sem="weak")

    return main


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64", "int32", "uint64"])
def test_ld_st_compiles_for_every_dtype(dtype):
    """T.ld/T.st must work for every scalar dtype, float16 included."""
    import tilelang

    tilelang.compile(_ld_st_kernel(dtype, scope="sys", sem="weak"))


@pytest.mark.parametrize(
    "ld_kwargs,message",
    [
        (dict(nc=True, sem="acquire"), "no memory ordering"),
        (dict(nc=True, sem="relaxed"), "no memory ordering"),
        (dict(nc=True, scope="sys"), "unscoped"),
        (dict(nc=True, scope="cta"), "unscoped"),
    ],
)
def test_ld_rejects_nc_with_ordering_or_scope(ld_kwargs, message):
    """`ld.global.nc` has no semantic or scope slot, so neither may be requested."""
    with pytest.raises(AssertionError, match=message):
        _ld_st_kernel("float32", **ld_kwargs)


def test_ld_accepts_plain_nc():
    import tilelang

    tilelang.compile(_ld_st_kernel("float32", nc=True))
    tilelang.compile(_ld_st_kernel("float32", nc=True, na=True))
