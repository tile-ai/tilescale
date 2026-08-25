"""Device-scale GEMM template (Phase 4 Batch 3).

GEMM-specific half of the device-scale lowering, owned by the GEMM tile op. It
decodes a direct ``with T.scale("device"): T.gemm(A, B, C)`` (a 19-arg
``tl.tileop.gemm``) into ``DeviceGemmInfo``, validating it against the MVP
template constraints (raising a precise ``NotImplementedError`` for any non-MVP
shape/dtype/op), and builds the generative ``device -> block -> thread`` tiled
template (``build_device_gemm_template``).

The generic pass ``tilelang.transform.prepare_device_scale_gemm`` drives this:
it finds device-scale GEMMs, calls ``_decode_device_gemm`` / the template
builder, and grafts the result onto the original PrimFunc. This split keeps the
GEMM-specific decode/template here (GEMM-owned) while the pass stays generic.
"""

from dataclasses import dataclass

from tvm import tirx as tir
from tvm.tirx.stmt_functor import post_order_visit

from tilelang.tileop.scale_template import (
    ScaleTileOpTemplate,
    register_scale_template,
)

_GEMM_OP_NAME = "tl.tileop.gemm"
_NON_MVP_GEMM_OP_NAMES = (
    "tl.tileop.wgmma_gemm",
    "tl.tileop.tcgen05_gemm",
)
_GEMM_OP_NAMES = (_GEMM_OP_NAME,) + _NON_MVP_GEMM_OP_NAMES

# MVP template constants (validated against the Step 3a probe: fp16 in / fp32
# accumulate, 128x128 block tile, 32 K tile, 128 threads). build_device_gemm_template
# emits a device -> block -> thread kernel with these tiles.
MVP_BM = 128
MVP_BN = 128
MVP_BK = 32
MVP_THREADS = 128
_MVP_IN_DTYPE = "float16"
_MVP_OUT_DTYPE = "float16"
_MVP_ACC_DTYPE = "float32"

# tl.tileop.gemm positional argument layout (see language/gemm_op.py _gemm_impl):
#  0 A region, 1 B region, 2 C region, 3 transpose_A, 4 transpose_B,
#  5 M, 6 N, 7 K, 8 policy, 9 clear_accum, 10 stride_a, 11 stride_b,
#  12 offset_a, 13 offset_b, 14 k_pack, 15 wg_wait, 16 mbar, 17 C_coord0, 18 C_coord1
_A_REGION, _B_REGION, _C_REGION = 0, 1, 2
_TRANSPOSE_A, _TRANSPOSE_B = 3, 4
_M, _N, _K = 5, 6, 7
_POLICY, _CLEAR_ACCUM = 8, 9
_STRIDE_A, _STRIDE_B = 10, 11
_OFFSET_A, _OFFSET_B = 12, 13
_K_PACK = 14
_WG_WAIT = 15
_MBAR = 16
_C_COORD0, _C_COORD1 = 17, 18

# A plain (non-blockscaled) tl.tileop.gemm has exactly 19 positional args. The
# block-scaled path (tcgen05_gemm_blockscaled) appends SFA/SFB regions and ids
# (args 19..22), so a longer arg list means an extended/blockscaled layout.
_PLAIN_GEMM_NUM_ARGS = 19

_GEMM_WARP_POLICY_SQUARE = 0  # GemmWarpPolicy.Square


def _int_or_none(expr):
    """Return the python int of an IntImm-like expr, or None if not a constant int."""
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    return None


def _region_buffer(region_call):
    """Recover the tir.Buffer from a tl.tileop.region call (arg0 is a BufferLoad)."""
    if not (isinstance(region_call, tir.Call) and isinstance(region_call.op, tir.op.Op)
            and region_call.op.name == "tl.tileop.region"):
        return None
    inner = region_call.args[0]
    if isinstance(inner, tir.BufferLoad):
        return inner.buffer
    return None


@dataclass
class DeviceGemmInfo:
    """Decoded MVP-relevant fields of a direct device-scale ``tl.tileop.gemm``."""
    A: tir.Buffer
    B: tir.Buffer
    C: tir.Buffer
    M: int
    N: int
    K: int
    in_dtype: str
    out_dtype: str
    acc_dtype: str


def _reject(reason: str):
    raise NotImplementedError(
        f"PrepareDeviceScaleGemm: device-scale GEMM is {reason}. This is not "
        f"supported yet. Wrap the GEMM in explicit inner block/thread scales "
        f"(device -> block -> thread -> T.gemm) for now."
    )


def _decode_device_gemm(call: tir.Call) -> DeviceGemmInfo:
    """Decode + MVP-validate a direct device-scale GEMM call.

    Always raises NotImplementedError for any non-MVP shape/dtype/op (no silent
    return -- a silently-skipped device GEMM would crash in LowerScaleLaunch).
    On success returns a DeviceGemmInfo for build_device_gemm_template.
    """
    op_name = call.op.name

    # Only the synchronous tl.tileop.gemm (covers T.gemm + blockscaled, which
    # emit tl.tileop.gemm) is an MVP candidate. wgmma / tcgen05 are not.
    if op_name in _NON_MVP_GEMM_OP_NAMES:
        _reject(f"emitted via a non-MVP op `{op_name}` (only the synchronous "
                f"T.gemm path is a generative device-scale candidate)")

    args = call.args

    # A blockscaled GEMM (tcgen05_gemm_blockscaled) emits tl.tileop.gemm with an
    # extended arg layout (SFA/SFB regions + ids appended after arg 18). Reject
    # any non-plain layout up front so the positional decode below is sound.
    if len(args) != _PLAIN_GEMM_NUM_ARGS:
        _reject(f"using an extended/blockscaled GEMM argument layout "
                f"({len(args)} args, expected {_PLAIN_GEMM_NUM_ARGS}); blockscaled "
                f"device-scale GEMM")

    # Operand buffers must be global tensors.
    A = _region_buffer(args[_A_REGION])
    B = _region_buffer(args[_B_REGION])
    C = _region_buffer(args[_C_REGION])
    if A is None or B is None or C is None:
        _reject("operands could not be decoded to global buffers")
    for buf, label in ((A, "A"), (B, "B"), (C, "C")):
        if buf.scope() != "global":
            _reject(f"operand {label} is in scope `{buf.scope()}`, not `global` "
                    f"(generative device-scale GEMM tiles global tensors)")

    # Static shapes, divisible by the MVP tiles.
    M = _int_or_none(args[_M])
    N = _int_or_none(args[_N])
    K = _int_or_none(args[_K])
    if M is None or N is None or K is None:
        _reject("a dynamic shape (M/N/K must be static IntImm)")
    if M % MVP_BM != 0 or N % MVP_BN != 0 or K % MVP_BK != 0:
        _reject(f"a shape (M={M}, N={N}, K={K}) not divisible by the MVP tile "
                f"sizes (BM={MVP_BM}, BN={MVP_BN}, BK={MVP_BK})")

    # dtypes: fp16 in, fp16 out, fp32 accumulate.
    in_dtype = A.dtype
    out_dtype = C.dtype
    if in_dtype != _MVP_IN_DTYPE or B.dtype != _MVP_IN_DTYPE:
        _reject(f"using input dtype `{in_dtype}`/`{B.dtype}` (MVP supports only "
                f"`{_MVP_IN_DTYPE}` inputs)")
    if out_dtype != _MVP_OUT_DTYPE:
        _reject(f"using output dtype `{out_dtype}` (MVP supports only "
                f"`{_MVP_OUT_DTYPE}` output)")

    # No transpose.
    if _int_or_none(args[_TRANSPOSE_A]) or _int_or_none(args[_TRANSPOSE_B]):
        _reject("using transpose_A/transpose_B (MVP supports only non-transposed "
                "operands)")

    # Square warp policy only.
    if _int_or_none(args[_POLICY]) != _GEMM_WARP_POLICY_SQUARE:
        _reject("using a non-default warp policy (MVP supports only the Square "
                "policy)")

    # No clear_accum (the template clears its own fragment).
    if _int_or_none(args[_CLEAR_ACCUM]):
        _reject("using clear_accum (MVP generates its own accumulator clear)")

    # k_pack == 1.
    if _int_or_none(args[_K_PACK]) != 1:
        _reject("using k_pack != 1 (MVP supports only k_pack == 1)")

    # wg_wait == 0 (MVP is synchronous; no explicit warpgroup wait id).
    if _int_or_none(args[_WG_WAIT]) != 0:
        _reject("using wg_wait != 0 (MVP is synchronous)")

    # No mbar: MVP requires arg16 to be the IntImm(0) placeholder. Any other
    # value (a BufferLoad mbarrier, or any non-zero/non-IntImm expr) is non-MVP.
    if _int_or_none(args[_MBAR]) != 0:
        _reject("using an explicit mbarrier (MVP is synchronous; arg16 must be "
                "the IntImm(0) placeholder)")

    # C coords must be (0, 0): the whole C tensor is the output.
    if _int_or_none(args[_C_COORD0]) != 0 or _int_or_none(args[_C_COORD1]) != 0:
        _reject("a non-zero C coordinate (MVP writes the whole C tensor)")

    # Offsets must be 0 (whole-tensor operands).
    if _int_or_none(args[_OFFSET_A]) != 0 or _int_or_none(args[_OFFSET_B]) != 0:
        _reject("a non-zero operand offset (MVP tiles whole global tensors)")

    # Row-major stride: stride_a == K, stride_b == N.
    if _int_or_none(args[_STRIDE_A]) != K or _int_or_none(args[_STRIDE_B]) != N:
        _reject("a non-default (non-row-major) operand stride")

    return DeviceGemmInfo(
        A=A, B=B, C=C, M=M, N=N, K=K,
        in_dtype=in_dtype, out_dtype=out_dtype, acc_dtype=_MVP_ACC_DTYPE,
    )


def build_device_gemm_template(info: DeviceGemmInfo,
                               BM: int = MVP_BM, BN: int = MVP_BN,
                               BK: int = MVP_BK, threads: int = MVP_THREADS):
    """Build the generative device-scale GEMM template as a PrimFunc.

    Produces the device -> block -> thread tiled kernel validated by the Step 3a
    probe (shared A/B tiles, fragment accumulator, k-loop copy + tile gemm,
    fragment->global output copy). Returns a ``tir.PrimFunc`` with the same
    ``(A, B, C)`` signature as the original device-scale GEMM.

    ``PrepareDeviceScaleGemm`` calls this to rewrite an MVP-shaped direct
    device-scale GEMM into a runnable kernel.
    """
    # Imported lazily: tilelang.language pulls in a broad surface, and this
    # module is imported during tilelang.transform package init.
    import tilelang.language as T

    M, N, K = info.M, info.N, info.K
    in_dtype = info.in_dtype
    out_dtype = info.out_dtype
    acc_dtype = info.acc_dtype

    # Explicit MVP guard: the helper is exported and may be called directly, so it
    # enforces the same contract as _decode_device_gemm rather than relying on
    # ceildiv to mask an illegal shape.
    if M % BM != 0 or N % BN != 0 or K % BK != 0:
        raise ValueError(
            f"build_device_gemm_template: shape (M={M}, N={N}, K={K}) is not "
            f"divisible by the tile sizes (BM={BM}, BN={BN}, BK={BK}).")
    if (in_dtype != _MVP_IN_DTYPE or out_dtype != _MVP_OUT_DTYPE
            or acc_dtype != _MVP_ACC_DTYPE):
        raise ValueError(
            f"build_device_gemm_template: dtype combination (in={in_dtype}, "
            f"out={out_dtype}, acc={acc_dtype}) is outside the MVP "
            f"(in/out={_MVP_IN_DTYPE}, acc={_MVP_ACC_DTYPE}).")

    @T.prim_func
    def kernel(A: T.Tensor((M, K), in_dtype),
               B: T.Tensor((K, N), in_dtype),
               C: T.Tensor((M, N), out_dtype)):
        with T.scale("device") as _d:
            with T.scale("block", workgroup=(M // BM, N // BN)) as (bx, by):
                with T.scale("thread", workgroup=(threads,)) as _tx:
                    A_s = T.alloc_shared((BM, BK), in_dtype)
                    B_s = T.alloc_shared((BK, BN), in_dtype)
                    C_f = T.alloc_fragment((BM, BN), acc_dtype)
                    T.clear(C_f)
                    for ko in T.serial(K // BK):
                        T.copy(A[bx * BM, ko * BK], A_s)
                        T.copy(B[ko * BK, by * BN], B_s)
                        T.gemm(A_s, B_s, C_f)
                    T.copy(C_f, C[bx * BM, by * BN])

    return kernel


def _is_scale_for(node):
    """True if `node` is a scale launch For loop (tl.scale annotation)."""
    if not isinstance(node, tir.For):
        return False
    ann = node.annotations
    return bool(ann) and ann.get("tl.scale") is not None


def _count_gemm_effect_stmts(stmt):
    """Count execution-affecting statements in `stmt` (for the single-gemm check).

    Returns (num_gemm_evaluates, num_other_effect_stmts). The device-scope body
    of an MVP kernel is just ``For(device scale) { Evaluate(gemm) }``: the scale
    launch For is scaffolding, not a user side-effect, so it is not counted. The
    gemm Evaluate, no-op Evaluate (assume / const), and structural wrappers
    (SeqStmt / BlockRealize / Block) are tolerated; any BufferStore / non-scale
    For / IfThenElse / Allocate / etc. counts as an "other effect" so a
    side-effect sibling can be rejected.
    """
    n_gemm = [0]
    n_other = [0]

    def visit(node):
        if isinstance(node, tir.Evaluate):
            val = node.value
            if isinstance(val, tir.Call) and isinstance(val.op, tir.op.Op):
                name = val.op.name
                if name in _GEMM_OP_NAMES:
                    n_gemm[0] += 1
                    return
                if name == "tir.assume" or name.endswith(".assume"):
                    return  # no-op
            if isinstance(val, tir.IntImm):
                return  # T.evaluate(0) no-op
            n_other[0] += 1
        elif isinstance(node, tir.For):
            if not _is_scale_for(node):
                n_other[0] += 1  # a real (non-scale) loop is a side effect
        elif isinstance(node, (tir.BufferStore, tir.IfThenElse,
                               tir.AllocBuffer, tir.DeclBuffer, tir.Bind,
                               tir.AssertStmt, tir.AttrStmt)):
            n_other[0] += 1

    post_order_visit(stmt, visit)
    return n_gemm[0], n_other[0]


def validate_device_gemm_func(info, func) -> None:
    """Function-level MVP checks for a direct device-scale GEMM (shared helper).

    Reused by both the legacy whole-function ``GemmDeviceTemplate`` (registry
    path) and the generic ``GemmDeviceExpansionTemplate`` (top-down expansion
    path), so the MVP boundary + reject messages are defined once. Raises
    ``NotImplementedError`` for: a device scope with sibling statements, extra
    params, or a non-(A, B, C) parameter order.
    """
    # The device-scope body must contain only the single GEMM (no side-effect
    # sibling statements). One gemm Evaluate, no other effects.
    n_gemm, n_other = _count_gemm_effect_stmts(func.body)
    if n_gemm != 1 or n_other != 0:
        raise NotImplementedError(
            "PrepareDeviceScaleGemm: a device-scale GEMM with sibling "
            "statements (stores / loops / control flow / allocations) is not "
            "supported yet. The MVP rewrites a device scope whose body is a "
            "single T.gemm.")

    # Exactly the (A, B, C) params.
    if len(func.params) != 3:
        raise NotImplementedError(
            "PrepareDeviceScaleGemm: a device-scale GEMM function with extra "
            "parameters (beyond A, B, C) is not supported yet.")

    # The function signature order must be exactly (A, B, C) -- the same order
    # as the GEMM operands -- so the rewritten template (built with an
    # (A, B, C) signature) is a drop-in. Verify each param's buffer is the
    # corresponding decoded GEMM operand (by buffer data var). Do not reorder.
    param_bufs = [func.buffer_map.get(p, None) for p in func.params]
    if (param_bufs[0] is None or param_bufs[1] is None or param_bufs[2] is None
            or not param_bufs[0].data.same_as(info.A.data)
            or not param_bufs[1].data.same_as(info.B.data)
            or not param_bufs[2].data.same_as(info.C.data)):
        raise NotImplementedError(
            "PrepareDeviceScaleGemm: the function signature must be in "
            "A, B, C parameter order (matching the GEMM operands). A "
            "different parameter order is not supported yet.")


def device_gemm_template_with_attrs(info, func):
    """Build the device GEMM template and copy ``func``'s attrs onto it.

    Shared by both the legacy ``rewrite`` and the expansion-template ``plan`` so
    the generated kernel preserves global_symbol / target / other attrs.
    """
    template = build_device_gemm_template(info)
    if func.attrs:
        for key, value in dict(func.attrs).items():
            template = template.with_attr(key, value)
    return template


class GemmDeviceTemplate(ScaleTileOpTemplate):
    """Device-scale template for ``T.gemm`` (and the non-MVP gemm ops it rejects).

    Claims the ``device`` scale and every GEMM tile-op name so a device-scope
    wgmma / tcgen05 / blockscaled GEMM is decoded (and loud-errored) here rather
    than slipping past the pass. ``validate`` owns the GEMM function-level MVP
    checks (single GEMM, no side-effect siblings, exactly the (A, B, C) params in
    order); ``decode`` owns the per-call MVP checks; ``rewrite`` builds the
    generative device -> block -> thread kernel.
    """

    @property
    def scale_names(self) -> tuple[str, ...]:
        return ("device",)

    @property
    def op_names(self) -> tuple[str, ...]:
        return _GEMM_OP_NAMES

    def decode(self, call, func):
        return _decode_device_gemm(call)

    def validate(self, info, func) -> None:
        validate_device_gemm_func(info, func)

    def rewrite(self, info, func):
        return device_gemm_template_with_attrs(info, func)


register_scale_template(GemmDeviceTemplate())
