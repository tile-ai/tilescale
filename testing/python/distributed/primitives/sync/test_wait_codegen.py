from __future__ import annotations

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
)
def _wait_scope_semantics_kernel():
    @T.prim_func
    def main(
        signal_u32: T.Tensor((1,), T.uint32),
        signal_i32: T.Tensor((1,), T.int32),
        signal_u64: T.Tensor((1,), T.uint64),
        signal_i64: T.Tensor((1,), T.int64),
    ):
        with T.Kernel(1, threads=1):
            T.wait_eq(signal_u32[0], 1)
            T.wait_eq(signal_i32[0], 1)
            T.wait_eq(signal_u64[0], 1)
            T.wait_eq(signal_i64[0], 1)
            T.wait_eq(signal_i64[0], 1, peer=0)
            T.wait_ne(signal_u32[0], 1)
            T.wait_ge(signal_u32[0], 1)
            T.wait_le(signal_u32[0], 1)
            T.wait_gt(signal_u32[0], 1)
            T.wait_lt(signal_u32[0], 1)
            T.wait_eq(signal_u32[0], 1, scope="gpu")
            T.wait_eq(signal_i32[0], 1, scope="gpu")
            T.wait_eq(signal_u64[0], 1, scope="gpu")
            T.wait_eq(signal_i64[0], 1, scope="gpu")
            T.wait_eq(signal_u32[0], 1, scope="gpu", semantics="volatile")
            T.wait_eq(signal_i32[0], 1, scope="gpu", semantics="volatile")
            T.wait_eq(signal_u64[0], 1, scope="gpu", semantics="volatile")
            T.wait_eq(signal_i64[0], 1, scope="gpu", semantics="volatile")

    return main


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
)
def _multimem_signal_dtype_kernel():
    @T.prim_func
    def main(signal32: T.Tensor((1,), T.uint32), signal64: T.Tensor((1,), T.uint64)):
        with T.Kernel(1, threads=1):
            T.multimem_signal(signal32[0], 1)
            T.multimem_signal(signal64[0], 1)

    return main


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_wait_scope_semantics_codegen():
    source = _wait_scope_semantics_kernel().get_kernel_source()
    for dtype in ("uint32_t", "int32_t", "uint64_t", "int64_t"):
        assert f"tl::wait_eq<tl::WaitScope::kSys, tl::WaitSemantics::kAcquire, {dtype}>" in source
        assert f"tl::wait_eq<tl::WaitScope::kGpu, tl::WaitSemantics::kAcquire, {dtype}>" in source
        assert f"tl::wait_eq<tl::WaitScope::kGpu, tl::WaitSemantics::kVolatile, {dtype}>" in source
    for relation in ("eq", "ne", "ge", "le", "gt", "lt"):
        assert f"tl::wait_{relation}<tl::WaitScope::kSys, tl::WaitSemantics::kAcquire, uint32_t>" in source
    remote_i64_wait = "tl::wait_eq<tl::WaitScope::kSys, tl::WaitSemantics::kAcquire, int64_t>((tl::get_remote_base_ptr(0)"
    assert remote_i64_wait in source
    invalid_atomic_load = ".".join(("atom", "load"))
    assert invalid_atomic_load not in source


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_multimem_signal_infers_dtype_codegen():
    source = _multimem_signal_dtype_kernel().get_kernel_source()
    assert "tl::multimem::Signal<uint32_t>::run" in source
    assert "tl::multimem::Signal<uint64_t>::run" in source
