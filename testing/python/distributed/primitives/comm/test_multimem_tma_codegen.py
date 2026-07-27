"""Compile-time contracts for multimem bulk TMA and signal helpers."""

from __future__ import annotations

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


@_requires_multimem_tma_codegen
def test_multimem_tma_dynamic_rank_is_runtime_guarded():
    source = _lower_with_nvcc(_dynamic_rank_kernel())
    assert "tl::multimem::cp_async_bulk" in source
    assert "int rank = tl::get_rank();" in source
    assert "if ((0 <= rank) && (rank <= 1)) {" in source
    assert "(uint)16" in source


@_requires_multimem_tma_codegen
def test_multimem_tma_rejects_non_16_byte_size():
    with pytest.raises(RuntimeError, match="provably divisible by 16 bytes"):
        _lower_with_nvcc(_invalid_size_kernel())


@_requires_multimem_tma_codegen
@pytest.mark.parametrize("side", ["shared", "multicast"])
def test_multimem_tma_rejects_misaligned_start(side: str):
    with pytest.raises(RuntimeError, match=f"TMA {side}.*16-byte aligned"):
        _lower_with_nvcc(_misaligned_kernel(side))


@_requires_multimem_tma_codegen
def test_multimem_tma_rejects_static_out_of_bounds_region():
    with pytest.raises(RuntimeError, match="region is statically out of bounds"):
        _lower_with_nvcc(_out_of_bounds_kernel())


@_requires_multimem_tma_codegen
def test_multimem_tma_rejects_aligned_static_out_of_bounds_region():
    with pytest.raises(RuntimeError, match="region is statically out of bounds"):
        _lower_with_nvcc(_aligned_out_of_bounds_kernel())


@_requires_multimem_tma_codegen
def test_multimem_tma_rejects_noncontiguous_2d_region():
    with pytest.raises(RuntimeError, match="provably physically contiguous"):
        _lower_with_nvcc(_noncontiguous_2d_kernel())


@_requires_multimem_tma_codegen
def test_multimem_tma_rejects_mismatched_extent():
    with pytest.raises(RuntimeError, match="matching source and destination extents"):
        _lower_with_nvcc(_mismatched_extent_kernel())


@_requires_multimem_tma_codegen
def test_multimem_tma_rejects_mismatched_rank():
    with pytest.raises(RuntimeError, match="regions with matching rank"):
        _lower_with_nvcc(_mismatched_rank_kernel())


@_requires_multimem_tma_codegen
def test_multimem_tma_rejects_mismatched_dtype():
    with pytest.raises(RuntimeError, match="matching source and destination dtypes"):
        _lower_with_nvcc(_mismatched_dtype_kernel())


@_requires_multimem_tma_codegen
def test_multimem_tma_rejects_unsupported_reduce_dtype():
    with pytest.raises(RuntimeError, match="supports float32, float16, and bfloat16"):
        _lower_with_nvcc(_unsupported_reduce_dtype_kernel())


@_requires_multimem_tma_codegen
def test_multimem_tma_rejects_subbyte_dtype():
    with pytest.raises(RuntimeError, match="requires byte-addressable element dtypes"):
        _lower_with_nvcc(_unsupported_subbyte_dtype_kernel())


@_requires_multimem_tma_codegen
def test_multimem_tma_rejects_layout_remap():
    with pytest.raises(RuntimeError, match="does not support layout-remapped buffers"):
        _lower_with_nvcc(_remapped_shared_layout_kernel())


@_requires_multimem_tma_codegen
@pytest.mark.parametrize("reduce_op", [T.MultimemReduceOp.MIN, T.MultimemReduceOp.MAX])
def test_multimem_tma_rejects_f32_min_max(reduce_op: T.MultimemReduceOp):
    with pytest.raises(RuntimeError, match="does not support (MIN|MAX) with float32"):
        _lower_with_nvcc(_unsupported_f32_reduce_kernel(reduce_op))


if __name__ == "__main__":
    tilelang.testing.main()
