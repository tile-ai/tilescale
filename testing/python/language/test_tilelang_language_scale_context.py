"""Phase-2 scale context metadata plumbing tests.

These cover the frontend ``current_scale()`` context stack and the
``tl.scale_ctx.*`` annotations that ``T.copy`` / ``T.gemm`` stamp onto their
emitted tile ops. This is metadata only -- no lowering behavior changes.
"""
import tilelang
import tilelang.language as T
import tilelang.testing
import pytest
import tilelang.transform

_HAS_EXPANSION = hasattr(tilelang.transform, "PrepareScaleTileOps")
requires_expansion = pytest.mark.skipif(
    not _HAS_EXPANSION, reason="scale tile-op templates (M2) not ported yet")
from tilelang.language.scale import (
    current_scale,
    SCALE_CTX_NAME_KEY,
    SCALE_CTX_WORKGROUP_KEY,
    SCALE_CTX_PARENT_KEY,
    SCALE_CTX_PATH_KEY,
)


def _path_list(v):
    """Extract a Python list of str from a TVM Array / list of StringImm / str."""
    return [x.value if hasattr(x, "value") else str(x) for x in v]


def test_current_scale_outside_is_none():
    assert current_scale() is None


@tilelang.testing.requires_cuda
def test_current_scale_inside_and_nested():
    seen = {}

    @T.prim_func
    def kernel(out: T.Tensor((16,), T.int32)):
        with T.scale("block", workgroup=(4, 2)) as (bx, by):
            with T.scale("thread", workgroup=(32,)) as tid:
                # capture innermost context during construction
                ctx = current_scale()
                seen["name"] = ctx.name
                seen["workgroup"] = ctx.workgroup
                seen["parent"] = ctx.parent_name
                if tid == 0:
                    out[bx] = by

    # Force construction of the prim_func.
    tilelang.lower(kernel, target="cuda")
    assert seen["name"] == "thread"
    assert tuple(seen["workgroup"]) == (32,)
    assert seen["parent"] == "block"
    # After the with-block exits, no active scale.
    assert current_scale() is None


def _str_val(v):
    """Extract the Python string from a StringImm (or pass through a str)."""
    return v.value if hasattr(v, "value") else str(v)


def _int_list(v):
    """Extract a Python list of ints from an Array of IntImm (or pass through)."""
    out = []
    for x in v:
        out.append(int(x.value) if hasattr(x, "value") else int(x))
    return out


def _assert_thread_scale_ctx(a):
    """Assert annotations ``a`` carry the expected innermost (thread) scale ctx."""
    assert _str_val(a[SCALE_CTX_NAME_KEY]) == "thread"
    assert _int_list(a[SCALE_CTX_WORKGROUP_KEY]) == [128]
    assert _str_val(a[SCALE_CTX_PARENT_KEY]) == "block"


def _find_tileop_annotations(func, op_suffix):
    """Collect annotations dicts of every tl.tileop.<suffix> call in func."""
    import tvm
    from tvm import tirx as tir
    found = []

    def visit(node):
        if isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op):
            if node.op.name == f"tl.tileop.{op_suffix}":
                # annotations live on the Call node's `annotations` map
                ann = getattr(node, "annotations", None)
                if ann is None:
                    found.append({})
                else:
                    found.append({str(k): ann[k] for k in ann.keys()})
    tir.stmt_functor.post_order_visit(func.body, visit)
    return found


@tilelang.testing.requires_cuda
def test_copy_gets_scale_ctx_annotations():
    import tvm

    @T.prim_func
    def kernel(A: T.Tensor((128, 128), T.float16), B: T.Tensor((128, 128), T.float16)):
        with T.scale("block", workgroup=(2, 2)) as (bx, by):
            with T.scale("thread", workgroup=(128,)) as tid:
                smem = T.alloc_shared((64, 64), T.float16)
                T.copy(A[bx * 64, by * 64], smem)
                T.copy(smem, B[bx * 64, by * 64])

    mod = tvm.IRModule.from_expr(kernel)
    func = next(iter(mod.functions.values()))
    anns = _find_tileop_annotations(func, "copy")
    assert anns, "expected at least one tl.tileop.copy"
    for a in anns:
        _assert_thread_scale_ctx(a)


@tilelang.testing.requires_cuda
def test_async_copy_gets_scale_ctx_annotations():
    import tvm

    @T.prim_func
    def kernel(A: T.Tensor((128, 128), T.float16), B: T.Tensor((128, 128), T.float16)):
        with T.scale("block", workgroup=(2, 2)) as (bx, by):
            with T.scale("thread", workgroup=(128,)) as tid:
                smem = T.alloc_shared((64, 64), T.float16)
                T.async_copy(A[bx * 64, by * 64], smem)

    mod = tvm.IRModule.from_expr(kernel)
    func = next(iter(mod.functions.values()))
    anns = _find_tileop_annotations(func, "async_copy")
    assert anns, "expected at least one tl.tileop.async_copy"
    for a in anns:
        _assert_thread_scale_ctx(a)


@tilelang.testing.requires_cuda
def test_tma_copy_gets_scale_ctx_annotations():
    import tvm

    @T.prim_func
    def kernel(A: T.Tensor((128, 128), T.float16), B: T.Tensor((128, 128), T.float16)):
        with T.scale("block", workgroup=(2, 2)) as (bx, by):
            with T.scale("thread", workgroup=(128,)) as tid:
                smem = T.alloc_shared((64, 64), T.float16)
                bar = T.alloc_barrier([1])
                T.tma_copy(A[bx * 64, by * 64], smem, barrier=bar)

    mod = tvm.IRModule.from_expr(kernel)
    func = next(iter(mod.functions.values()))
    anns = _find_tileop_annotations(func, "tma_copy")
    assert anns, "expected at least one tl.tileop.tma_copy"
    for a in anns:
        _assert_thread_scale_ctx(a)


@tilelang.testing.requires_cuda
def test_gemm_gets_scale_ctx_annotations():
    import tvm

    @T.prim_func
    def kernel(A: T.Tensor((64, 64), T.float16), B: T.Tensor((64, 64), T.float16),
               C: T.Tensor((64, 64), T.float16)):
        with T.scale("block", workgroup=(1, 1)) as (bx, by):
            with T.scale("thread", workgroup=(128,)) as tid:
                A_s = T.alloc_shared((64, 64), T.float16)
                B_s = T.alloc_shared((64, 64), T.float16)
                C_l = T.alloc_fragment((64, 64), T.float32)
                T.copy(A, A_s)
                T.copy(B, B_s)
                T.gemm(A_s, B_s, C_l)

    mod = tvm.IRModule.from_expr(kernel)
    func = next(iter(mod.functions.values()))
    anns = _find_tileop_annotations(func, "gemm")
    assert anns, "expected at least one tl.tileop.gemm"
    for a in anns:
        _assert_thread_scale_ctx(a)


@tilelang.testing.requires_cuda
def test_tcgen05_gemm_blockscaled_gets_scale_ctx_annotations():
    # tcgen05_gemm_blockscaled emits tl.tileop.gemm directly (not via _gemm_impl),
    # and must still pick up scale ctx; its own use_2cta annotation must survive.
    import tvm

    @T.prim_func
    def kernel(A: T.Tensor((128, 64), T.float8_e4m3),
               B: T.Tensor((64, 128), T.float8_e4m3),
               C: T.Tensor((128, 128), T.float32)):
        with T.scale("block", workgroup=(1, 1)) as (bx, by):
            with T.scale("thread", workgroup=(128,)) as tid:
                A_s = T.alloc_shared((128, 64), T.float8_e4m3)
                B_s = T.alloc_shared((64, 128), T.float8_e4m3)
                C_t = T.alloc_tmem([128, 128], T.float32)
                SFA = T.alloc_tmem([128, 4], T.float32)
                SFB = T.alloc_tmem([128, 4], T.float32)
                bar = T.alloc_barrier([1])
                T.copy(A, A_s)
                T.copy(B, B_s)
                T.tcgen05_gemm_blockscaled(A_s, B_s, C_t, SFA, SFB, mbar=bar, k_start=0, sf_a_granularity_k=32, sf_b_granularity_k=32)

    mod = tvm.IRModule.from_expr(kernel)
    func = next(iter(mod.functions.values()))
    anns = _find_tileop_annotations(func, "gemm")
    assert anns, "expected at least one tl.tileop.gemm"
    for a in anns:
        _assert_thread_scale_ctx(a)


@tilelang.testing.requires_cuda
def test_user_annotations_not_overridden():
    import tvm

    @T.prim_func
    def kernel(A: T.Tensor((128, 128), T.float16), B: T.Tensor((128, 128), T.float16)):
        with T.scale("block", workgroup=(2, 2)) as (bx, by):
            with T.scale("thread", workgroup=(128,)) as tid:
                smem = T.alloc_shared((64, 64), T.float16)
                # user explicitly sets the scale_ctx.name key -> must win
                T.copy(A[bx * 64, by * 64], smem,
                       annotations={SCALE_CTX_NAME_KEY: "user_override"})

    mod = tvm.IRModule.from_expr(kernel)
    func = next(iter(mod.functions.values()))
    anns = _find_tileop_annotations(func, "copy")
    assert anns
    assert _str_val(anns[0][SCALE_CTX_NAME_KEY]) == "user_override"


@tilelang.testing.requires_cuda
def test_no_scale_ctx_without_scale():
    import tvm

    @T.prim_func
    def kernel(A: T.Tensor((128, 128), T.float16), B: T.Tensor((128, 128), T.float16)):
        with T.Kernel(2, 2, threads=128) as (bx, by):
            smem = T.alloc_shared((64, 64), T.float16)
            T.copy(A[bx * 64, by * 64], smem)
            T.copy(smem, B[bx * 64, by * 64])

    mod = tvm.IRModule.from_expr(kernel)
    func = next(iter(mod.functions.values()))
    anns = _find_tileop_annotations(func, "copy")
    assert anns
    for a in anns:
        assert SCALE_CTX_NAME_KEY not in a
        assert SCALE_CTX_WORKGROUP_KEY not in a


@tilelang.testing.requires_cuda
def test_scale_ctx_lowers_without_crash():
    # The metadata must not break lowering / source generation.
    import tvm

    @T.prim_func
    def kernel(A: T.Tensor((128, 128), T.float16), B: T.Tensor((128, 128), T.float16)):
        with T.scale("block", workgroup=(2, 2)) as (bx, by):
            with T.scale("thread", workgroup=(128,)) as tid:
                smem = T.alloc_shared((64, 64), T.float16)
                T.copy(A[bx * 64, by * 64], smem)
                T.copy(smem, B[bx * 64, by * 64])

    target = tvm.target.Target("cuda")
    with target:
        artifact = tilelang.lower(kernel, target=target)
    assert artifact.kernel_source is not None
    assert "blockIdx.x" in artifact.kernel_source
    assert "threadIdx.x" in artifact.kernel_source
    # scale_ctx keys are frontend metadata and must not leak into CUDA source.
    assert "scale_ctx" not in artifact.kernel_source


def test_scale_ctx_stack_clean_on_body_exception():
    # An exception raised inside the with-body must unwind the context managers
    # and leave the scale-context stack empty for subsequent kernels.
    import importlib
    scale_mod = importlib.import_module("tilelang.language.scale")

    assert scale_mod.current_scale() is None

    class _Boom(Exception):
        pass

    @T.prim_func
    def kernel(out: T.Tensor((16,), T.int32)):
        with T.scale("block", workgroup=(4,)) as bx:
            with T.scale("thread", workgroup=(32,)) as tid:
                if tid == 0:
                    out[bx] = bx

    # Normal construction must leave the stack empty.
    import tvm
    tvm.IRModule.from_expr(kernel)
    assert scale_mod.current_scale() is None

    # Body exception must also leave the stack empty.
    def build_failing():
        @T.prim_func
        def bad(out: T.Tensor((16,), T.int32)):
            with T.scale("block", workgroup=(4,)) as bx:
                with T.scale("thread", workgroup=(32,)) as tid:
                    raise _Boom()

        return bad

    try:
        build_failing()
    except Exception:
        pass
    assert scale_mod.current_scale() is None


def test_scale_ctx_rolls_back_on_enter_failure():
    # A failure *inside* a frame's __enter__ (after the scale ctx is pushed and
    # the loop frame entered) must roll back the scale ctx, exit the loop frame,
    # and leave the context clean so the next kernel is unaffected.
    import importlib
    import tvm
    scale_mod = importlib.import_module("tilelang.language.scale")

    assert scale_mod.current_scale() is None

    # Patch the root-block opener used inside ScaleFrame.__enter__ to raise,
    # which models an __enter__ failure that happens after the loop frame has
    # been entered and the scale ctx pushed.
    import tvm.tirx.script.builder as tb_tir

    class _EnterBoom(Exception):
        pass

    orig_block = tb_tir.sblock

    def boom_block(*a, **k):
        raise _EnterBoom()

    def build_failing():
        @T.prim_func
        def bad(out: T.Tensor((16,), T.int32)):
            # block scale does not open a root block; thread scale does, so the
            # patched opener fires inside the thread frame's __enter__.
            with T.scale("block", workgroup=(4,)) as bx:
                with T.scale("thread", workgroup=(32,)) as tid:
                    if tid == 0:
                        out[bx] = bx

        return bad

    tb_tir.sblock = boom_block
    try:
        try:
            build_failing()
        except Exception:
            pass
    finally:
        tb_tir.sblock = orig_block

    # The failed __enter__ must not have polluted the stack.
    assert scale_mod.current_scale() is None

    # A subsequent normal construction + lower must work cleanly.
    @T.prim_func
    def good(out: T.Tensor((4,), T.int32)):
        with T.scale("block", workgroup=(4,)) as bx:
            with T.scale("thread", workgroup=(32,)) as tid:
                if tid == 0:
                    out[bx] = bx

    target = tvm.target.Target("cuda")
    with target:
        artifact = tilelang.lower(good, target=target)
    assert "blockIdx.x" in artifact.kernel_source
    assert "threadIdx.x" in artifact.kernel_source
    assert scale_mod.current_scale() is None


# ---------------------------------------------------------------------------
# Phase 4: GEMM scale_ctx consumption / scope-aware dispatch.
#
# T.gemm reads the innermost `tl.scale_ctx.name` (and `tl.scale_ctx.path`) and
# routes through resolve_gemm_impl(scale_scope=..., scale_path=...). Allowed
# scopes (None / block / thread / warp / device) keep the existing
# implementation; unsupported scopes (cluster / sm / die / unknown) loud-error
# instead of silently lowering as a plain block GEMM.
#
# A direct `with T.scale("device"): T.gemm(...)` has no block/thread launch
# binding, so generative device-scale tiling would be required to make it
# runnable -- that is not implemented. It now builds fine but loud-errors in the
# pre-LowerScaleLaunch pass PrepareDeviceScaleGemm (which `tilelang.lower` runs in
# LowerAndLegalize before LowerScaleLaunch), instead of crashing in lowering.
# `device` is still in the dispatch allowed set as a reservation /
# ancestor-metadata path: a device -> block -> thread GEMM (innermost "thread")
# lowers normally and carries scale_ctx.path == ["device", "block", "thread"].
# ---------------------------------------------------------------------------


@requires_expansion
def test_gemm_device_only_builds_with_device_metadata():
    # A T.gemm directly inside a `device` scale scope (global operands, the
    # generative-target form) builds fine and stamps name="device" /
    # path=["device"] on the GEMM. (Whether it then lowers depends on the MVP
    # rewrite -- see the rewrite/reject tests below.) This test only checks the
    # build-side metadata and a clean scale-context stack.
    import importlib
    import tvm
    scale_mod = importlib.import_module("tilelang.language.scale")

    assert scale_mod.current_scale() is None

    @T.prim_func
    def kernel(A: T.Tensor((256, 256), T.float16), B: T.Tensor((256, 256), T.float16),
               C: T.Tensor((256, 256), T.float16)):
        with T.scale("device") as d:
            T.gemm(A, B, C)

    assert scale_mod.current_scale() is None
    mod = tvm.IRModule.from_expr(kernel)
    func = next(iter(mod.functions.values()))
    anns = _find_tileop_annotations(func, "gemm")
    assert anns, "expected at least one tl.tileop.gemm"
    for a in anns:
        assert _str_val(a[SCALE_CTX_NAME_KEY]) == "device"
        assert _path_list(a[SCALE_CTX_PATH_KEY]) == ["device"]


@tilelang.testing.requires_cuda
@requires_expansion
def test_gemm_device_mvp_shape_rewritten_and_lowers():
    # Step 3a-2c: an MVP-shaped direct device GEMM (global operands, 256x256x256
    # fp16) is now rewritten by PrepareDeviceScaleGemm into the device -> block ->
    # thread template and lowers to a runnable kernel (blockIdx/threadIdx/mma, no
    # scale_ctx leak).
    import tvm

    @T.prim_func
    def kernel(A: T.Tensor((256, 256), T.float16), B: T.Tensor((256, 256), T.float16),
               C: T.Tensor((256, 256), T.float16)):
        with T.scale("device") as d:
            T.gemm(A, B, C)

    target = tvm.target.Target("cuda")
    with target:
        artifact = tilelang.lower(kernel, target=target)
    src = artifact.kernel_source
    assert "blockIdx.x" in src
    assert "threadIdx.x" in src
    assert "mma" in src
    assert "scale_ctx" not in src


@tilelang.testing.requires_cuda
@requires_expansion
def test_gemm_device_mvp_correctness():
    # Step 3a-2c GPU smoke: a direct device GEMM rewritten to the template is
    # numerically correct against torch.
    import torch

    @T.prim_func
    def kernel(A: T.Tensor((256, 256), T.float16), B: T.Tensor((256, 256), T.float16),
               C: T.Tensor((256, 256), T.float16)):
        with T.scale("device") as d:
            T.gemm(A, B, C)

    compiled = tilelang.compile(kernel, target="cuda")
    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    c = torch.empty(256, 256, device="cuda", dtype=torch.float16)
    compiled(a, b, c)
    ref = (a.float() @ b.float()).half()
    assert torch.allclose(c, ref, rtol=1e-2, atol=1e-2), \
        f"maxdiff={(c.float() - ref.float()).abs().max().item()}"


@requires_expansion
def test_gemm_device_non_divisible_shape_rejected():
    # A device GEMM whose shape is not divisible by the MVP tiles must reject with
    # a divisibility / tile-size message.
    import pytest
    import tvm
    from tilelang.transform import PrepareDeviceScaleGemm

    @T.prim_func
    def kernel(A: T.Tensor((130, 256), T.float16), B: T.Tensor((256, 256), T.float16),
               C: T.Tensor((130, 256), T.float16)):
        with T.scale("device") as d:
            T.gemm(A, B, C)

    mod = tvm.IRModule.from_expr(kernel)
    with pytest.raises(Exception) as ei:
        PrepareDeviceScaleGemm()(mod)
    msg = str(ei.value)
    assert "PrepareDeviceScaleGemm" in msg
    assert "not supported yet" in msg
    assert ("divisible" in msg or "tile" in msg)


@requires_expansion
def test_gemm_device_transpose_rejected():
    # A transposed device GEMM must reject with a transpose message.
    import pytest
    import tvm
    from tilelang.transform import PrepareDeviceScaleGemm

    @T.prim_func
    def kernel(A: T.Tensor((256, 256), T.float16), B: T.Tensor((256, 256), T.float16),
               C: T.Tensor((256, 256), T.float16)):
        with T.scale("device") as d:
            T.gemm(A, B, C, transpose_A=True)

    mod = tvm.IRModule.from_expr(kernel)
    with pytest.raises(Exception) as ei:
        PrepareDeviceScaleGemm()(mod)
    msg = str(ei.value)
    assert "PrepareDeviceScaleGemm" in msg
    assert "transpose" in msg
    assert "not supported yet" in msg


@tilelang.testing.requires_cuda
@requires_expansion
def test_gemm_device_param_order_rejected():
    # MVP requires the function signature order to be (A, B, C), matching the
    # GEMM operands. A permuted signature must reject with a parameter-order
    # message (the rewrite does not reorder params).
    import pytest
    import tvm
    from tilelang.transform import PrepareDeviceScaleGemm

    @T.prim_func
    def kernel(C: T.Tensor((256, 256), T.float16), A: T.Tensor((256, 256), T.float16),
               B: T.Tensor((256, 256), T.float16)):
        with T.scale("device") as d:
            T.gemm(A, B, C)

    mod = tvm.IRModule.from_expr(kernel)
    with pytest.raises(Exception) as ei:
        PrepareDeviceScaleGemm()(mod)
    msg = str(ei.value)
    assert "PrepareDeviceScaleGemm" in msg
    assert "A, B, C parameter order" in msg
    assert "not supported yet" in msg


@tilelang.testing.requires_cuda
@requires_expansion
def test_gemm_device_multiple_gemms_rejected():
    # Two direct device-scale GEMMs in one function must reject.
    import pytest
    import tvm
    from tilelang.transform import PrepareDeviceScaleGemm

    @T.prim_func
    def kernel(A: T.Tensor((256, 256), T.float16), B: T.Tensor((256, 256), T.float16),
               C: T.Tensor((256, 256), T.float16)):
        with T.scale("device") as d:
            T.gemm(A, B, C)
            T.gemm(A, B, C)

    mod = tvm.IRModule.from_expr(kernel)
    with pytest.raises(Exception) as ei:
        PrepareDeviceScaleGemm()(mod)
    msg = str(ei.value)
    assert "PrepareScaleTileOps" in msg
    assert "multiple scale-scoped tile-op template rewrites" in msg


@tilelang.testing.requires_cuda
@requires_expansion
def test_gemm_device_sibling_side_effect_rejected():
    # A side-effect statement (a buffer store) alongside the device GEMM must
    # reject: the MVP rewrites a device scope whose body is a single T.gemm.
    import pytest
    import tvm
    from tilelang.transform import PrepareDeviceScaleGemm

    @T.prim_func
    def kernel(A: T.Tensor((256, 256), T.float16), B: T.Tensor((256, 256), T.float16),
               C: T.Tensor((256, 256), T.float16)):
        with T.scale("device") as d:
            C[0, 0] = T.float16(0)
            T.gemm(A, B, C)

    mod = tvm.IRModule.from_expr(kernel)
    with pytest.raises(Exception) as ei:
        PrepareDeviceScaleGemm()(mod)
    msg = str(ei.value)
    assert "PrepareDeviceScaleGemm" in msg
    assert "sibling statements" in msg
    assert "not supported yet" in msg


@requires_expansion
def test_wgmma_gemm_device_only_rejected_non_mvp_op():
    # Op-coverage: a direct device T.wgmma_gemm (tl.tileop.wgmma_gemm) is a
    # non-MVP op and must reject with a message naming the op. (tcgen05_gemm is in
    # the same scan set; not exercised e2e here because it needs an mbar.)
    import pytest
    import tvm
    from tilelang.transform import PrepareDeviceScaleGemm

    @T.prim_func
    def kernel(A: T.Tensor((256, 256), T.float16), B: T.Tensor((256, 256), T.float16),
               C: T.Tensor((256, 256), T.float16)):
        with T.scale("device") as d:
            T.wgmma_gemm(A, B, C)

    mod = tvm.IRModule.from_expr(kernel)
    with pytest.raises(Exception) as ei:
        PrepareDeviceScaleGemm()(mod)
    msg = str(ei.value)
    assert "PrepareDeviceScaleGemm" in msg
    assert "device-scale GEMM" in msg
    assert "wgmma_gemm" in msg
    assert "not supported yet" in msg


@tilelang.testing.requires_cuda
@requires_expansion
def test_blockscaled_gemm_device_only_rejected_extended_layout():
    # A blockscaled GEMM emits tl.tileop.gemm with an extended (>19) arg layout
    # (SFA/SFB regions + ids appended). PrepareDeviceScaleGemm must reject the
    # extended layout up front, before positional decode. Built synthetically:
    # a real device-scope tcgen05_gemm_blockscaled cannot be constructed (its
    # tmem/block_attr needs a block scope, absent under device-only), so we forge
    # a tl.tileop.gemm Call with 23 args and a device scale_ctx. The len(args)
    # check fires before any region decode, so dummy args are fine.
    import pytest
    import tvm
    from tvm import tirx as tir
    from tilelang.transform import PrepareDeviceScaleGemm

    op = tir.op.Op.get("tl.tileop.gemm")
    # 23 dummy int args (> 19 -> extended/blockscaled layout).
    dummy_args = [tir.const(0, "int32") for _ in range(23)]
    annotations = {
        SCALE_CTX_NAME_KEY: tir.StringImm("device"),
        SCALE_CTX_PATH_KEY: tvm.runtime.convert([tir.StringImm("device")]),
    }
    # The len(args) check in the pass fires before any region decode, so dummy
    # int args (which would not decode as regions) are sufficient.
    gemm_call = tir.Call("handle", op, dummy_args, annotations)

    @T.prim_func
    def kernel(out: T.Tensor((4,), T.int32)):
        out[0] = 0

    func = next(iter(tvm.IRModule.from_expr(kernel).functions.values()))
    new_body = tir.SeqStmt([tir.Evaluate(gemm_call), func.body])
    func = func.with_body(new_body)
    mod = tvm.IRModule({"main": func})

    with pytest.raises(Exception) as ei:
        PrepareDeviceScaleGemm()(mod)
    msg = str(ei.value)
    assert "PrepareDeviceScaleGemm" in msg
    assert "device-scale GEMM" in msg
    assert ("extended" in msg or "blockscaled" in msg)
    assert "not supported yet" in msg


@requires_expansion
def test_device_scale_tileop_without_template_rejected():
    # Fail-closed: a device-scope tl.tileop.* with no registered device template
    # must loud-error, not silently slip past into LowerScaleLaunch. Built
    # synthetically with tl.tileop.copy (a registered tile op that has no device
    # template) so we don't pull in real T.copy region/lowering complexity.
    import pytest
    import tvm
    from tvm import tirx as tir
    from tilelang.transform import PrepareDeviceScaleGemm

    op = tir.op.Op.get("tl.tileop.copy")
    annotations = {
        SCALE_CTX_NAME_KEY: tir.StringImm("device"),
        SCALE_CTX_PATH_KEY: tvm.runtime.convert([tir.StringImm("device")]),
    }
    copy_call = tir.Call("handle", op, [tir.const(0, "int32")], annotations)

    @T.prim_func
    def kernel(out: T.Tensor((4,), T.int32)):
        out[0] = 0

    func = next(iter(tvm.IRModule.from_expr(kernel).functions.values()))
    new_body = tir.SeqStmt([tir.Evaluate(copy_call), func.body])
    func = func.with_body(new_body)
    mod = tvm.IRModule({"main": func})

    with pytest.raises(Exception) as ei:
        PrepareDeviceScaleGemm()(mod)
    msg = str(ei.value)
    assert "PrepareScaleTileOps" in msg
    assert "scale-scoped tile op" in msg
    assert "no registered template" in msg
    assert "not supported yet" in msg


@requires_expansion
def test_scale_template_registry_resolves_device_gemm():
    # The scale-aware registry resolves ("device", "tl.tileop.gemm") to the GEMM
    # template, and returns None for an unregistered scale ("node"). The legacy
    # device_* alias and the renamed/aliased passes must also keep working.
    from tilelang.tileop.scale_template import (
        resolve_scale_template,
        ensure_default_scale_templates_registered,
    )
    ensure_default_scale_templates_registered()
    assert resolve_scale_template("device", "tl.tileop.gemm") is not None
    assert resolve_scale_template("node", "tl.tileop.gemm") is None

    # Legacy device_* registry alias still works.
    from tilelang.tileop.device_template import resolve_device_template
    assert resolve_device_template("tl.tileop.gemm") is not None

    # Pass names: new main + both compatibility aliases resolve to one object.
    from tilelang.transform import (
        PrepareScaleTileOps,
        PrepareDeviceScaleTileOps,
        PrepareDeviceScaleGemm,
    )
    assert PrepareDeviceScaleTileOps is PrepareScaleTileOps
    assert PrepareDeviceScaleGemm is PrepareScaleTileOps


@requires_expansion
def test_scale_template_registry_node_not_managed_until_registered():
    # No template registers scale_names=("node",), so the registry does not claim
    # the node scale: resolve returns None and has_scale_templates is False. (We
    # only test the registry here -- T.scale("node") has no lowering yet, so no
    # e2e lower.)
    from tilelang.tileop.scale_template import (
        resolve_scale_template,
        has_scale_templates,
        ensure_default_scale_templates_registered,
    )
    ensure_default_scale_templates_registered()
    assert has_scale_templates("device") is True
    assert has_scale_templates("node") is False
    assert resolve_scale_template("node", "tl.tileop.gemm") is None
    # thread / block are not template-managed either (hand-written kernels keep
    # flowing through the existing pipeline, not the registry).
    assert has_scale_templates("thread") is False
    assert has_scale_templates("block") is False


@requires_expansion
def test_device_scale_multi_dim_workgroup_rejected():
    # T.scale("device") supports a single SPMD workgroup dimension (the rank
    # axis). A multi-dim workgroup has no defined rank mapping and must
    # loud-error at construction instead of being silently dropped.
    import pytest

    def build():
        @T.prim_func
        def kernel(out: T.Tensor((4,), T.int32)):
            with T.scale("device", workgroup=(2, 2)) as d:
                out[0] = 0

        return kernel

    with pytest.raises(Exception) as ei:
        build()
    msg = str(ei.value)
    assert "device" in msg
    assert "workgroup" in msg
    assert "single workgroup dimension" in msg


@tilelang.testing.requires_cuda
@requires_expansion
def test_gemm_device_non_mvp_lower_rejects():
    # End-to-end: `tilelang.lower` on a non-MVP direct device GEMM (non-divisible
    # shape) loud-errors via PrepareDeviceScaleGemm (before LowerScaleLaunch),
    # not a segfault. (An MVP-shaped device GEMM now lowers; see
    # test_gemm_device_mvp_shape_rewritten_and_lowers.)
    import pytest
    import tvm

    @T.prim_func
    def kernel(A: T.Tensor((130, 256), T.float16), B: T.Tensor((256, 256), T.float16),
               C: T.Tensor((130, 256), T.float16)):
        with T.scale("device") as d:
            T.gemm(A, B, C)

    target = tvm.target.Target("cuda")
    with pytest.raises(Exception) as ei:
        with target:
            tilelang.lower(kernel, target=target)
    msg = str(ei.value)
    assert "PrepareDeviceScaleGemm" in msg
    assert "device-scale GEMM" in msg
    assert "not supported yet" in msg


@tilelang.testing.requires_cuda
@requires_expansion
def test_gemm_device_sibling_store_lower_rejects():
    # End-to-end regression guard: a device GEMM with a sibling store must
    # loud-error through the real pipeline (NormalizeScaleExpansion, before
    # LowerScaleLaunch) -- NOT segfault. The generic planner recognizes the
    # ordered device program (GEMM then store) as needing a launch-boundary stage
    # split and fail-closes (staging skeleton, milestone 11).
    import pytest
    import tvm

    @T.prim_func
    def kernel(A: T.Tensor((256, 256), T.float16), B: T.Tensor((256, 256), T.float16),
               C: T.Tensor((256, 256), T.float16)):
        with T.scale("device") as d:
            C[0, 0] = T.float16(0)
            T.gemm(A, B, C)

    target = tvm.target.Target("cuda")
    # TVM's FFI re-wraps a pass-callback NotImplementedError as RuntimeError, so
    # catch broadly and assert the message (same pattern as the non-MVP test).
    with pytest.raises(Exception) as ei:
        with target:
            tilelang.lower(kernel, target=target)
    msg = str(ei.value)
    assert "NormalizeScaleExpansion" in msg
    assert "ordered" in msg and "device" in msg
    assert "launch_boundary" in msg or "stage boundary" in msg
    assert "not implemented yet" in msg


@tilelang.testing.requires_cuda
@requires_expansion
def test_gemm_device_multiple_gemms_lower_rejects():
    # End-to-end regression guard: two device GEMMs in one scope must loud-error
    # through the real pipeline (NormalizeScaleExpansion) -- NOT segfault. The
    # generic planner recognizes an ordered device program (two ordered stages)
    # needing a launch-boundary stage split and fail-closes.
    import pytest
    import tvm

    @T.prim_func
    def kernel(A: T.Tensor((256, 256), T.float16), B: T.Tensor((256, 256), T.float16),
               C: T.Tensor((256, 256), T.float16)):
        with T.scale("device") as d:
            T.gemm(A, B, C)
            T.gemm(A, B, C)

    target = tvm.target.Target("cuda")
    with pytest.raises(Exception) as ei:
        with target:
            tilelang.lower(kernel, target=target)
    msg = str(ei.value)
    assert "NormalizeScaleExpansion" in msg
    assert "ordered" in msg and "device" in msg
    assert "launch_boundary" in msg or "stage boundary" in msg
    assert "not implemented yet" in msg


@tilelang.testing.requires_cuda
@requires_expansion
def test_gemm_device_static_loop_lower_rejects():
    # End-to-end regression guard: a static T.serial loop of device GEMMs is an
    # ordered (repeated) device stage program. It must loud-error through the real
    # pipeline (NormalizeScaleExpansion) -- NOT segfault / leak to LowerScaleLaunch.
    import pytest
    import tvm

    @T.prim_func
    def kernel(A: T.Tensor((256, 256), T.float16), B: T.Tensor((256, 256), T.float16),
               C: T.Tensor((256, 256), T.float16)):
        with T.scale("device") as d:
            for _i in T.serial(10):
                T.gemm(A, B, C)

    target = tvm.target.Target("cuda")
    with pytest.raises(Exception) as ei:
        with target:
            tilelang.lower(kernel, target=target)
    msg = str(ei.value)
    assert "NormalizeScaleExpansion" in msg
    assert "loop" in msg or "ordered" in msg
    assert "not implemented yet" in msg


@tilelang.testing.requires_cuda
@requires_expansion
def test_gemm_device_block_thread_not_rejected_by_prepare_pass():
    # A device -> block -> thread GEMM (innermost scope "thread") must pass
    # cleanly through PrepareDeviceScaleGemm (no rewrite, no reject).
    import tvm

    @T.prim_func
    def kernel(A: T.Tensor((64, 64), T.float16), B: T.Tensor((64, 64), T.float16),
               C: T.Tensor((64, 64), T.float16)):
        with T.scale("device") as d:
            with T.scale("block", workgroup=(1, 1)) as (bx, by):
                with T.scale("thread", workgroup=(128,)) as tid:
                    A_s = T.alloc_shared((64, 64), T.float16)
                    B_s = T.alloc_shared((64, 64), T.float16)
                    C_l = T.alloc_fragment((64, 64), T.float32)
                    T.copy(A, A_s)
                    T.copy(B, B_s)
                    T.gemm(A_s, B_s, C_l)

    from tilelang.transform import PrepareDeviceScaleGemm
    mod = tvm.IRModule.from_expr(kernel)
    # Must not raise.
    PrepareDeviceScaleGemm()(mod)


@tilelang.testing.requires_cuda
@requires_expansion
def test_gemm_device_block_thread_innermost_scope_is_thread():
    # Pin the current metadata semantics: scale_ctx.name is the INNERMOST scale,
    # not an ancestor. A device -> block -> thread GEMM stamps "thread" (and so
    # lowers via the allowed thread path), NOT "device". The full ancestor chain
    # is preserved in scale_ctx.path == ["device", "block", "thread"].
    import tvm

    @T.prim_func
    def kernel(A: T.Tensor((64, 64), T.float16), B: T.Tensor((64, 64), T.float16),
               C: T.Tensor((64, 64), T.float16)):
        with T.scale("device") as d:
            with T.scale("block", workgroup=(1, 1)) as (bx, by):
                with T.scale("thread", workgroup=(128,)) as tid:
                    A_s = T.alloc_shared((64, 64), T.float16)
                    B_s = T.alloc_shared((64, 64), T.float16)
                    C_l = T.alloc_fragment((64, 64), T.float32)
                    T.copy(A, A_s)
                    T.copy(B, B_s)
                    T.gemm(A_s, B_s, C_l)

    mod = tvm.IRModule.from_expr(kernel)
    func = next(iter(mod.functions.values()))
    anns = _find_tileop_annotations(func, "gemm")
    assert anns, "expected at least one tl.tileop.gemm"
    for a in anns:
        assert _str_val(a[SCALE_CTX_NAME_KEY]) == "thread"
        assert _path_list(a[SCALE_CTX_PATH_KEY]) == ["device", "block", "thread"]

    # And it lowers (through the allowed "thread" scope), with no ctx leak.
    target = tvm.target.Target("cuda")
    with target:
        artifact = tilelang.lower(kernel, target=target)
    assert "threadIdx.x" in artifact.kernel_source
    assert "scale_ctx" not in artifact.kernel_source


@tilelang.testing.requires_cuda
@requires_expansion
def test_gemm_cluster_block_thread_ancestor_cluster_not_rejected():
    # A cluster -> block -> thread GEMM has scale_ctx.path containing an ancestor
    # `cluster`, but its innermost scope is `thread` (an allowed scope). The
    # ancestor cluster must NOT cause a dispatch reject: the gate keys off the
    # innermost scope, not the path. The kernel lowers with no ctx leak.
    import tvm

    @T.prim_func
    def kernel(A: T.Tensor((64, 64), T.float16), B: T.Tensor((64, 64), T.float16),
               C: T.Tensor((64, 64), T.float16)):
        with T.scale("cluster", workgroup=(1, 1)) as (cm, cn):
            with T.scale("block", workgroup=(2, 1, 1)) as cta_id:
                with T.scale("thread", workgroup=(128,)) as tid:
                    A_s = T.alloc_shared((64, 64), T.float16)
                    B_s = T.alloc_shared((64, 64), T.float16)
                    C_l = T.alloc_fragment((64, 64), T.float32)
                    T.copy(A, A_s)
                    T.copy(B, B_s)
                    T.gemm(A_s, B_s, C_l)

    mod = tvm.IRModule.from_expr(kernel)
    func = next(iter(mod.functions.values()))
    anns = _find_tileop_annotations(func, "gemm")
    assert anns, "expected at least one tl.tileop.gemm"
    for a in anns:
        assert _str_val(a[SCALE_CTX_NAME_KEY]) == "thread"
        assert _path_list(a[SCALE_CTX_PATH_KEY]) == ["cluster", "block", "thread"]

    target = tvm.target.Target("cuda")
    with target:
        artifact = tilelang.lower(kernel, target=target)
    assert "threadIdx.x" in artifact.kernel_source
    assert "scale_ctx" not in artifact.kernel_source


@tilelang.testing.requires_cuda
def test_gemm_block_scope_lowers_and_no_ctx_leak():
    # A normal block->thread GEMM (innermost scope "thread", an allowed scope)
    # lowers through the existing GEMM path, and the scale_ctx metadata must not
    # leak into the generated CUDA source.
    import tvm

    @T.prim_func
    def kernel(A: T.Tensor((64, 64), T.float16), B: T.Tensor((64, 64), T.float16),
               C: T.Tensor((64, 64), T.float16)):
        with T.scale("block", workgroup=(1, 1)) as (bx, by):
            with T.scale("thread", workgroup=(128,)) as tid:
                A_s = T.alloc_shared((64, 64), T.float16)
                B_s = T.alloc_shared((64, 64), T.float16)
                C_l = T.alloc_fragment((64, 64), T.float32)
                T.copy(A, A_s)
                T.copy(B, B_s)
                T.gemm(A_s, B_s, C_l)

    target = tvm.target.Target("cuda")
    with target:
        artifact = tilelang.lower(kernel, target=target)
    assert artifact.kernel_source is not None
    # A 1x1 grid optimizes blockIdx.x out of the device source; threadIdx.x is
    # always present and proves the GEMM lowered through the thread-level path.
    assert "threadIdx.x" in artifact.kernel_source
    assert "scale_ctx" not in artifact.kernel_source


@tilelang.testing.requires_cuda
def test_gemm_no_scale_lowers_smoke():
    # Existing no-scale GEMM path must be unaffected: a plain T.Kernel GEMM lowers
    # and emits no scale_ctx metadata.
    import tvm

    @T.prim_func
    def kernel(A: T.Tensor((64, 64), T.float16), B: T.Tensor((64, 64), T.float16),
               C: T.Tensor((64, 64), T.float16)):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_s = T.alloc_shared((64, 64), T.float16)
            B_s = T.alloc_shared((64, 64), T.float16)
            C_l = T.alloc_fragment((64, 64), T.float32)
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.gemm(A_s, B_s, C_l)

    target = tvm.target.Target("cuda")
    with target:
        artifact = tilelang.lower(kernel, target=target)
    # A 1x1 grid optimizes blockIdx.x out; threadIdx.x proves the kernel lowered.
    assert "threadIdx.x" in artifact.kernel_source
    assert "scale_ctx" not in artifact.kernel_source


@requires_expansion
def test_gemm_resolve_impl_scope_gate():
    # Dispatch-layer contract: resolve_gemm_impl accepts the allowed scopes and
    # loud-errors on unsupported ones. Tested directly because a naturally-built
    # cluster/sm/die-scope GEMM has no thread binding and cannot reach GEMM
    # lowering (it would crash earlier on a missing thread scope), so the reject
    # is exercised at the dispatch boundary rather than via fragile IR injection.
    import tvm
    import tilelang.tileop.gemm  # noqa: F401  (registers GEMM impls)
    from tilelang.tileop.gemm.registry import resolve_gemm_impl

    target = tvm.target.Target("cuda")

    # Unsupported scopes must raise with a clear, specific message.
    for scope in ("cluster", "sm", "die", "warpgroup_bogus"):
        try:
            resolve_gemm_impl("tl.tileop.any", target, scale_scope=scope)
        except NotImplementedError as e:
            msg = str(e)
            assert "scale_ctx" in msg and scope in msg and "not supported yet" in msg, \
                f"unexpected message for scope {scope}: {msg[:200]}"
        else:
            raise AssertionError(f"scope {scope} should have raised NotImplementedError")

    # Allowed scopes must pass the gate (reaching the instruction-registry lookup,
    # which raises a *different* ValueError for the dummy instruction key, not the
    # scope NotImplementedError). NOTE: `device` passing here is a dispatch-level
    # allow / reservation, not proof that a device-scale GEMM is runnable (a direct
    # device-scope GEMM is not runnable in Batch 1; see module comment).
    for scope in (None, "block", "thread", "warp", "device"):
        try:
            resolve_gemm_impl("tl.tileop.definitely_not_registered", target, scale_scope=scope)
        except NotImplementedError as e:
            raise AssertionError(
                f"allowed scope {scope} wrongly rejected by scope gate: {str(e)[:200]}")
        except ValueError:
            pass  # expected: instruction key not registered, scope gate passed


@requires_expansion
def test_gemm_resolve_impl_path_consistency():
    # scale_path is ancestor metadata: an ancestor cluster/sm/die in the path must
    # NOT trigger a reject as long as the innermost scope is allowed. But a path
    # whose innermost element disagrees with scale_scope is contradictory metadata
    # and must loud-error.
    import tvm
    import tilelang.tileop.gemm  # noqa: F401
    from tilelang.tileop.gemm.registry import resolve_gemm_impl

    target = tvm.target.Target("cuda")

    # Ancestor cluster, innermost thread -> path does NOT reject (passes the scope
    # gate, reaching the registry lookup that raises ValueError for the dummy key).
    try:
        resolve_gemm_impl("tl.tileop.definitely_not_registered", target,
                          scale_scope="thread", scale_path=("cluster", "block", "thread"))
    except NotImplementedError as e:
        raise AssertionError(f"ancestor cluster wrongly rejected: {str(e)[:200]}")
    except ValueError:
        pass  # expected: dummy instruction not registered, gate + path check passed

    # A single-element path matching the scope passes the consistency gate.
    try:
        resolve_gemm_impl("tl.tileop.definitely_not_registered", target,
                          scale_scope="thread", scale_path=("thread",))
    except NotImplementedError as e:
        raise AssertionError(f"single-element matching path wrongly rejected: {str(e)[:200]}")
    except ValueError:
        pass  # expected: dummy instruction not registered, consistency gate passed

    # Mismatched path/scope is contradictory metadata -> loud error mentioning path.
    try:
        resolve_gemm_impl("tl.tileop.any", target,
                          scale_scope="thread", scale_path=("device", "block"))
    except ValueError as e:
        assert "scale_ctx.path" in str(e), f"unexpected message: {str(e)[:200]}"
    else:
        raise AssertionError("mismatched scale_scope/scale_path should have raised ValueError")


def _mvp_device_gemm_info(M=256, N=256, K=256):
    # Build a DeviceGemmInfo for the builder. The builder only uses M/N/K and the
    # dtypes (not the buffer objects), so placeholder buffers are fine.
    import tvm
    from tvm import tirx as tir
    from tilelang.transform.prepare_device_scale_gemm import DeviceGemmInfo
    A = tir.decl_buffer((M, K), "float16", name="A")
    B = tir.decl_buffer((K, N), "float16", name="B")
    C = tir.decl_buffer((M, N), "float16", name="C")
    return DeviceGemmInfo(A=A, B=B, C=C, M=M, N=N, K=K,
                          in_dtype="float16", out_dtype="float16", acc_dtype="float32")


@tilelang.testing.requires_cuda
@requires_expansion
def test_build_device_gemm_template_metadata():
    # Step 3a-2b: build_device_gemm_template generates the device -> block ->
    # thread tiled kernel. Its inner T.gemm must carry the thread-scope metadata
    # (name "thread", path ["device","block","thread"]).
    import tvm
    from tilelang.transform import build_device_gemm_template

    kernel = build_device_gemm_template(_mvp_device_gemm_info())
    mod = tvm.IRModule.from_expr(kernel)
    func = next(iter(mod.functions.values()))
    anns = _find_tileop_annotations(func, "gemm")
    assert anns, "expected at least one tl.tileop.gemm"
    for a in anns:
        assert _str_val(a[SCALE_CTX_NAME_KEY]) == "thread"
        assert _path_list(a[SCALE_CTX_PATH_KEY]) == ["device", "block", "thread"]


@tilelang.testing.requires_cuda
@requires_expansion
def test_build_device_gemm_template_lowers():
    # The generated template must lower: source has blockIdx/threadIdx/mma and no
    # scale_ctx leak.
    import tvm
    from tilelang.transform import build_device_gemm_template

    kernel = build_device_gemm_template(_mvp_device_gemm_info())
    target = tvm.target.Target("cuda")
    with target:
        artifact = tilelang.lower(kernel, target=target)
    src = artifact.kernel_source
    assert "blockIdx.x" in src
    assert "threadIdx.x" in src
    assert "mma" in src
    assert "scale_ctx" not in src


@tilelang.testing.requires_cuda
@requires_expansion
def test_build_device_gemm_template_correctness():
    # GPU smoke: the generated 256x256x256 fp16 template matches torch.
    import torch
    from tilelang.transform import build_device_gemm_template

    kernel = build_device_gemm_template(_mvp_device_gemm_info(256, 256, 256))
    compiled = tilelang.compile(kernel, target="cuda")
    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    c = torch.empty(256, 256, device="cuda", dtype=torch.float16)
    compiled(a, b, c)
    ref = (a.float() @ b.float()).half()
    assert torch.allclose(c, ref, rtol=1e-2, atol=1e-2), \
        f"maxdiff={(c.float() - ref.float()).abs().max().item()}"


if __name__ == "__main__":
    tilelang.testing.main()
