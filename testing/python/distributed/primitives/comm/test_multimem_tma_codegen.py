"""Compile-time contracts for multimem bulk TMA and signal helpers."""

from __future__ import annotations

import re

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.contrib import nvcc
from tilelang.layout import make_swizzled_layout


def _has_cuda_toolkit_13_1() -> bool:
    try:
        return nvcc.get_cuda_version() >= (13, 1)
    except (OSError, RuntimeError, ValueError):
        return False


def _requires_multimem_tma_codegen(func):
    func = tilelang.testing.requires_cuda_compute_version_ge(9, 0)(func)
    return pytest.mark.skipif(not _has_cuda_toolkit_13_1(), reason="Requires CUDA Toolkit 13.1+")(func)


def _lower_with_nvcc(kernel, *, arch: str | None = None) -> str:
    target = "cuda" if arch is None else {"kind": "cuda", "arch": arch}
    artifact = tilelang.lower(kernel, target=target, enable_device_compile=True)
    return artifact.kernel_source


def _lower_without_nvcc(kernel) -> str:
    artifact = tilelang.lower(
        kernel,
        target={"kind": "cuda", "arch": "sm_90a"},
        enable_device_compile=False,
    )
    return artifact.kernel_source


def _legal_helpers_kernel():
    @T.prim_func
    def main(
        mcast_f32: T.Tensor((4,), T.float32),
        mcast_f16: T.Tensor((8,), T.float16),
        mcast_bf16: T.Tensor((8,), T.bfloat16),
        mcast_u8: T.Tensor((16,), T.uint8),
        mcast_2d: T.Tensor((3, 8), T.float32),
        mcast_f32x2: T.Tensor((2,), T.float32x2),
        signal64: T.Tensor((1,), T.uint64),
    ):
        with T.Kernel(1, threads=1):
            shared_f32 = T.alloc_shared((4,), T.float32)
            shared_f16 = T.alloc_shared((8,), T.float16)
            shared_bf16 = T.alloc_shared((8,), T.bfloat16)
            shared_u8 = T.alloc_shared((16,), T.uint8)
            shared_2d = T.alloc_shared((3, 8), T.float32)
            shared_f32x2 = T.alloc_shared((2,), T.float32x2)

            T.multimem_tma_store(shared_f32, mcast_f32, reduce_op=T.MultimemReduceOp.ADD)
            T.multimem_tma_store(shared_f16, mcast_f16, reduce_op=T.MultimemReduceOp.ADD)
            T.multimem_tma_store(shared_f16, mcast_f16, reduce_op=T.MultimemReduceOp.MIN)
            T.multimem_tma_store(shared_f16, mcast_f16, reduce_op=T.MultimemReduceOp.MAX)
            T.multimem_tma_store(shared_bf16, mcast_bf16, reduce_op=T.MultimemReduceOp.ADD)
            T.multimem_tma_store(shared_bf16, mcast_bf16, reduce_op=T.MultimemReduceOp.MIN)
            T.multimem_tma_store(shared_bf16, mcast_bf16, reduce_op=T.MultimemReduceOp.MAX)
            T.multimem_tma_store(shared_u8, mcast_u8)
            T.multimem_tma_store(shared_2d[1, 0:4], mcast_2d[1, 0:4])
            T.multimem_tma_store(shared_f32x2, mcast_f32x2)
            T.multimem_signal_add(signal64[0], 1)

    return main


def _dynamic_rank_kernel():
    @T.prim_func
    def main(mcast: T.Tensor((8,), T.float32)):
        with T.Kernel(1, threads=1):
            rank = T.get_rank()
            shared = T.alloc_shared((4,), T.float32)
            T.multimem_tma_store(shared, mcast[rank * 4 : (rank + 1) * 4])

    return main


def _plain_none_kernel():
    @T.prim_func
    def main(mcast: T.Tensor((4,), T.float32)):
        with T.Kernel(1, threads=1):
            shared = T.alloc_shared((4,), T.float32)
            T.multimem_tma_store(shared, mcast, reduce_op=T.MultimemReduceOp.NONE)

    return main


def _invalid_size_kernel():
    @T.prim_func
    def main(mcast: T.Tensor((3,), T.float32)):
        with T.Kernel(1, threads=1):
            shared = T.alloc_shared((3,), T.float32)
            T.multimem_tma_store(shared, mcast)

    return main


def _misaligned_kernel(side: str):
    if side == "shared":

        @T.prim_func
        def main(mcast: T.Tensor((4,), T.float32)):
            with T.Kernel(1, threads=1):
                shared = T.alloc_shared((5,), T.float32)
                T.multimem_tma_store(shared[1:5], mcast)

    else:

        @T.prim_func
        def main(mcast: T.Tensor((5,), T.float32)):
            with T.Kernel(1, threads=1):
                shared = T.alloc_shared((4,), T.float32)
                T.multimem_tma_store(shared, mcast[1:5])

    return main


def _out_of_bounds_kernel():
    @T.prim_func
    def main(mcast: T.Tensor((16,), T.float32)):
        with T.Kernel(1, threads=1):
            shared = T.alloc_shared((16,), T.float32)
            T.multimem_tma_store(shared, mcast[4:20])

    return main


def _aligned_out_of_bounds_kernel():
    @T.prim_func
    def main(mcast: T.Tensor((8,), T.float32)):
        with T.Kernel(1, threads=1):
            shared = T.alloc_shared((4,), T.float32)
            T.multimem_tma_store(shared, mcast[8:12])

    return main


def _noncontiguous_2d_kernel():
    @T.prim_func
    def main(mcast: T.Tensor((2, 4), T.float32)):
        with T.Kernel(1, threads=1):
            shared = T.alloc_shared((2, 8), T.float32)
            T.multimem_tma_store(shared[:, 0:4], mcast)

    return main


def _mismatched_extent_kernel():
    @T.prim_func
    def main(mcast: T.Tensor((8,), T.float32)):
        with T.Kernel(1, threads=1):
            shared = T.alloc_shared((4,), T.float32)
            T.multimem_tma_store(shared, mcast)

    return main


def _mismatched_rank_kernel():
    @T.prim_func
    def main(mcast: T.Tensor((4,), T.float32)):
        with T.Kernel(1, threads=1):
            shared = T.alloc_shared((1, 4), T.float32)
            T.multimem_tma_store(shared, mcast)

    return main


def _mismatched_dtype_kernel():
    @T.prim_func
    def main(mcast: T.Tensor((8,), T.float16)):
        with T.Kernel(1, threads=1):
            shared = T.alloc_shared((8,), T.bfloat16)
            T.multimem_tma_store(shared, mcast)

    return main


def _unsupported_reduce_dtype_kernel():
    @T.prim_func
    def main(mcast: T.Tensor((4,), T.int32)):
        with T.Kernel(1, threads=1):
            shared = T.alloc_shared((4,), T.int32)
            T.multimem_tma_store(shared, mcast, reduce_op=T.MultimemReduceOp.ADD)

    return main


def _unsupported_subbyte_dtype_kernel():
    @T.prim_func
    def main(mcast: T.Tensor((32,), T.int4)):
        with T.Kernel(1, threads=1):
            shared = T.alloc_shared((32,), T.int4)
            T.multimem_tma_store(shared, mcast)

    return main


def _remapped_shared_layout_kernel():
    @T.prim_func
    def main(mcast: T.Tensor((16, 16), T.float16)):
        with T.Kernel(1, threads=1):
            shared = T.alloc_shared((16, 16), T.float16)
            T.annotate_layout({shared: make_swizzled_layout(shared)})
            T.multimem_tma_store(shared, mcast)

    return main


def _unsupported_f32_reduce_kernel(reduce_op: T.MultimemReduceOp):
    @T.prim_func
    def main(mcast: T.Tensor((4,), T.float32)):
        with T.Kernel(1, threads=1):
            shared = T.alloc_shared((4,), T.float32)
            T.multimem_tma_store(shared, mcast, reduce_op=reduce_op)

    return main


@_requires_multimem_tma_codegen
def test_multimem_tma_legal_helpers_compile_with_nvcc():
    source = _lower_with_nvcc(_legal_helpers_kernel(), arch="sm_90a")
    helpers = (
        "tl::multimem::cp_async_bulk",
        "tl::multimem::cp_reduce_async_bulk_add_f32",
        "tl::multimem::cp_reduce_async_bulk_add_f16",
        "tl::multimem::cp_reduce_async_bulk_min_f16",
        "tl::multimem::cp_reduce_async_bulk_max_f16",
        "tl::multimem::cp_reduce_async_bulk_add_bf16",
        "tl::multimem::cp_reduce_async_bulk_min_bf16",
        "tl::multimem::cp_reduce_async_bulk_max_bf16",
        "tl::multimem::SignalAdd<uint64_t>::run",
    )
    for helper in helpers:
        assert helper in source


def test_multimem_tma_dynamic_rank_is_runtime_guarded():
    source = _lower_without_nvcc(_dynamic_rank_kernel())
    assert "tl::multimem::cp_async_bulk" in source
    assert "tl::get_rank()" in source
    assert re.search(r"if \(\(0 <= rank\) && \(rank <= 1\)\)", source)


def test_multimem_tma_none_uses_plain_store():
    source = _lower_without_nvcc(_plain_none_kernel())
    assert "tl::multimem::cp_async_bulk" in source
    assert "tl::multimem::cp_reduce_async_bulk" not in source


_INVALID_TMA_CASES = (
    pytest.param(
        _invalid_size_kernel,
        r"divisible by 16 bytes",
        id="transaction-size",
    ),
    pytest.param(
        lambda: _misaligned_kernel("shared"),
        r"TMA shared.*16-byte aligned",
        id="shared-alignment",
    ),
    pytest.param(
        lambda: _misaligned_kernel("multicast"),
        r"TMA multicast.*16-byte aligned",
        id="multicast-alignment",
    ),
    pytest.param(
        _out_of_bounds_kernel,
        r"statically out of bounds",
        id="static-out-of-bounds",
    ),
    pytest.param(
        _aligned_out_of_bounds_kernel,
        r"statically out of bounds",
        id="aligned-static-out-of-bounds",
    ),
    pytest.param(
        _noncontiguous_2d_kernel,
        r"physically contiguous",
        id="physical-contiguity",
    ),
    pytest.param(
        _mismatched_extent_kernel,
        r"matching source and destination extents",
        id="extent",
    ),
    pytest.param(
        _mismatched_rank_kernel,
        r"regions with matching rank",
        id="rank",
    ),
    pytest.param(
        _mismatched_dtype_kernel,
        r"matching source and destination dtypes",
        id="dtype",
    ),
    pytest.param(
        _unsupported_reduce_dtype_kernel,
        r"supports float32, float16, and bfloat16",
        id="reduce-dtype",
    ),
    pytest.param(
        _unsupported_subbyte_dtype_kernel,
        r"byte-addressable element dtypes",
        id="subbyte-dtype",
    ),
    pytest.param(
        _remapped_shared_layout_kernel,
        r"layout-remapped buffers",
        id="layout-remap",
    ),
    pytest.param(
        lambda: _unsupported_f32_reduce_kernel(T.MultimemReduceOp.MIN),
        r"does not support MIN with float32",
        id="f32-min",
    ),
    pytest.param(
        lambda: _unsupported_f32_reduce_kernel(T.MultimemReduceOp.MAX),
        r"does not support MAX with float32",
        id="f32-max",
    ),
)


@pytest.mark.parametrize("kernel_factory,error", _INVALID_TMA_CASES)
def test_multimem_tma_rejects_invalid_contract(kernel_factory, error: str):
    with pytest.raises(RuntimeError, match=error):
        _lower_without_nvcc(kernel_factory())


if __name__ == "__main__":
    tilelang.testing.main()
