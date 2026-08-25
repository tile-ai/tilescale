"""Logical launch scale helpers."""

from __future__ import annotations

import sys
import threading
from typing import Any

from tvm import tirx as tir

from tilelang.jit.exceptions import JITNoBuilderError

from .loop import serial

_local = threading.local()

WARP_SIZE = 32


class ScaleContext:
    """Immutable snapshot of a scale scope, returned by :func:`current_scale`.

    It records *where* a primitive is being emitted so the primitive can stamp
    ``tl.scale_ctx.*`` annotations onto its own IR. GEMM dispatch reads those
    annotations (``name`` / ``path``) for its scope gate and consistency check;
    the metadata does not leak into the generated CUDA source.
    """

    __slots__ = ("name", "workgroup", "parent")

    def __init__(self, name, workgroup, parent=None):
        self.name = name
        self.workgroup = tuple(workgroup)
        self.parent = parent  # ScaleContext | None

    @property
    def parent_name(self):
        return self.parent.name if self.parent is not None else None

    def __repr__(self):
        return (f"ScaleContext(name={self.name!r}, workgroup={self.workgroup}, "
                f"parent={self.parent_name!r})")


def _scale_ctx_stack() -> list:
    stack = getattr(_local, "scale_ctx_stack", None)
    if stack is None:
        stack = []
        _local.scale_ctx_stack = stack
    return stack


def _push_scale_ctx(name, workgroup):
    stack = _scale_ctx_stack()
    parent = stack[-1] if stack else None
    ctx = ScaleContext(name, workgroup, parent)
    stack.append(ctx)
    return ctx


def _pop_scale_ctx(expected) -> None:
    """Pop ``expected`` off the scale-context stack, enforcing LIFO order.

    ``expected`` is the ScaleContext returned by the matching ``_push_scale_ctx``.
    Raises if the stack top is not ``expected`` (mismatched push/pop nesting).
    """
    stack = _scale_ctx_stack()
    assert stack and stack[-1] is expected, (
        "scale context stack corrupted: push/pop not balanced (LIFO violated)")
    stack.pop()


def current_scale():
    """Return the innermost active :class:`ScaleContext`, or ``None``.

    Usable inside a ``with T.scale(...)`` (or ``T.Scale(...)``) region to query
    the current scale name / workgroup shape / parent scale.
    """
    stack = _scale_ctx_stack()
    return stack[-1] if stack else None


# Annotation keys stamped onto primitive ops to record where they were emitted.
# scale_ctx is primitive metadata: GEMM dispatch currently reads `name`/`path`
# for the scope gate and a path/name consistency check. The metadata never leaks
# into the generated CUDA source.
SCALE_CTX_NAME_KEY = "tl.scale_ctx.name"
SCALE_CTX_WORKGROUP_KEY = "tl.scale_ctx.workgroup"
SCALE_CTX_PARENT_KEY = "tl.scale_ctx.parent"
SCALE_CTX_PATH_KEY = "tl.scale_ctx.path"


def merge_scale_context_annotations(annotations):
    """Return ``annotations`` augmented with ``tl.scale_ctx.*`` keys for the
    current scale, if any.

    - Returns the input unchanged when not inside a scale scope.
    - Never overrides keys the user already set (user annotations win).
    - Does not mutate the caller's dict.

    Stamps ``tl.scale_ctx.name`` (innermost scope), ``tl.scale_ctx.workgroup``
    (innermost workgroup shape), ``tl.scale_ctx.parent`` (one level up, when
    present), and ``tl.scale_ctx.path`` (the full ancestor chain outer -> inner,
    including the current scope). Every primitive that calls this helper
    (T.copy / T.gemm / ...) gets these keys.
    """
    ctx = current_scale()
    if ctx is None:
        return annotations
    from tvm import tirx as tir
    from tvm.runtime import convert
    merged = dict(annotations) if annotations else {}
    merged.setdefault(SCALE_CTX_NAME_KEY, tir.StringImm(ctx.name))
    merged.setdefault(SCALE_CTX_WORKGROUP_KEY,
                      convert([int(x) if isinstance(x, int) else x
                               for x in ctx.workgroup]))
    if ctx.parent_name is not None:
        merged.setdefault(SCALE_CTX_PARENT_KEY, tir.StringImm(ctx.parent_name))
    # Full scope path, outer -> inner, including the current (innermost) scope.
    path = []
    node = ctx
    while node is not None:
        path.append(node.name)
        node = node.parent
    path.reverse()
    merged.setdefault(SCALE_CTX_PATH_KEY,
                      convert([tir.StringImm(n) for n in path]))
    return merged


def _cluster_active() -> bool:
    """True when the current scope is nested inside a ``cluster`` scale.

    A cluster scale's ``__enter__`` runs before the inner ``block`` scale is
    constructed, so this lets ``Scale('block', ...)`` decide whether it denotes
    the cluster-internal CTA rank (cluster present) or plain grid coords.
    """
    return getattr(_local, "cluster_depth", 0) > 0


def _push_cluster() -> None:
    _local.cluster_depth = getattr(_local, "cluster_depth", 0) + 1


def _pop_cluster() -> None:
    _local.cluster_depth = getattr(_local, "cluster_depth", 0) - 1


class ScaleFrame:
    """Context manager for one logical launch-scale axis."""

    def __init__(self, loop_frame: Any, *, opens_root_block: bool,
                 is_cluster: bool = False, scale_name: str | None = None,
                 workgroup=()):
        self.loop_frame = loop_frame
        self.opens_root_block = opens_root_block
        self.is_cluster = is_cluster
        self.scale_name = scale_name
        self.workgroup = workgroup
        self.block_frame = None
        self._ctx_token = None

    def __enter__(self) -> Any:
        loop_var = self.loop_frame.__enter__()
        if self.is_cluster:
            _push_cluster()
        if self.scale_name is not None:
            self._ctx_token = _push_scale_ctx(self.scale_name, self.workgroup)
        try:
            if self.opens_root_block and getattr(_local, "scale_root_depth", 0) == 0:
                import tvm.tirx.script.builder as tb_tir

                self.block_frame = tb_tir.sblock("tilelang_root")
                self.block_frame.__enter__()
                _local.scale_root_depth = 1
        except BaseException:
            # Roll back everything pushed before the failure (scale ctx token,
            # cluster depth) and exit the loop frame so we don't pollute the
            # context for subsequent kernels.
            if self._ctx_token is not None:
                _pop_scale_ctx(self._ctx_token)
                self._ctx_token = None
            if self.is_cluster:
                _pop_cluster()
            self.loop_frame.__exit__(*sys.exc_info())
            raise
        return loop_var

    def __exit__(self, ptype, value, trace) -> None:
        try:
            if self.block_frame is not None:
                self.block_frame.__exit__(ptype, value, trace)
                _local.scale_root_depth = 0
        finally:
            if self._ctx_token is not None:
                _pop_scale_ctx(self._ctx_token)
                self._ctx_token = None
            if self.is_cluster:
                _pop_cluster()
            self.loop_frame.__exit__(ptype, value, trace)


class MultiScaleFrame:
    """Context manager for multiple logical launch-scale axes (e.g. 2D block grid)."""

    def __init__(self, loop_frames: list, *, opens_root_block: bool,
                 is_cluster: bool = False, scale_name: str | None = None,
                 workgroup=()):
        self.loop_frames = loop_frames
        self.opens_root_block = opens_root_block
        self.is_cluster = is_cluster
        self.scale_name = scale_name
        self.workgroup = workgroup
        self.block_frame = None
        self._ctx_token = None

    def __enter__(self) -> Any:
        loop_vars = []
        entered = []  # loop_frames successfully entered, for reverse unwind
        pushed_cluster = False
        try:
            for frame in self.loop_frames:
                v = frame.__enter__()
                entered.append(frame)
                loop_vars.append(v)
            if self.is_cluster:
                _push_cluster()
                pushed_cluster = True
            if self.scale_name is not None:
                self._ctx_token = _push_scale_ctx(self.scale_name, self.workgroup)
            if self.opens_root_block and getattr(_local, "scale_root_depth", 0) == 0:
                import tvm.tirx.script.builder as tb_tir

                self.block_frame = tb_tir.sblock("tilelang_root")
                self.block_frame.__enter__()
                _local.scale_root_depth = 1
        except BaseException:
            # Reverse-unwind everything already established before re-raising so
            # the context is not left polluted for subsequent kernels.
            if self._ctx_token is not None:
                _pop_scale_ctx(self._ctx_token)
                self._ctx_token = None
            if pushed_cluster:
                _pop_cluster()
            for frame in reversed(entered):
                frame.__exit__(*sys.exc_info())
            raise
        if len(loop_vars) == 1:
            return loop_vars[0]
        return loop_vars

    def __exit__(self, ptype, value, trace) -> None:
        try:
            if self.block_frame is not None:
                self.block_frame.__exit__(ptype, value, trace)
                _local.scale_root_depth = 0
        finally:
            if self._ctx_token is not None:
                _pop_scale_ctx(self._ctx_token)
                self._ctx_token = None
            if self.is_cluster:
                _pop_cluster()
            for frame in reversed(self.loop_frames):
                frame.__exit__(ptype, value, trace)


class WarpScaleFrame:
    """Context manager for warp-level Scale axis.

    Creates a placeholder thread Var at construction time. This Var is used
    by the MMA emitter for lane_id computation and is later bound to
    threadIdx.x by the C++ lowering pass.
    """

    _stack = threading.local()

    def __init__(self, loop_frame: Any, tx_var: tir.Var, workgroup=()):
        self.loop_frame = loop_frame
        self.tx_var = tx_var
        self.workgroup = workgroup
        self.block_frame = None
        self._ctx_token = None

    @classmethod
    def Current(cls) -> WarpScaleFrame | None:
        stack = getattr(cls._stack, "frames", None)
        if stack:
            return stack[-1]
        return None

    def __enter__(self) -> Any:
        warp_var = self.loop_frame.__enter__()
        self._ctx_token = _push_scale_ctx("warp", self.workgroup)
        try:
            if getattr(_local, "scale_root_depth", 0) == 0:
                import tvm.tirx.script.builder as tb_tir

                self.block_frame = tb_tir.sblock("tilelang_root")
                self.block_frame.__enter__()
                _local.scale_root_depth = 1
            if not hasattr(self._stack, "frames"):
                self._stack.frames = []
            self._stack.frames.append(self)
        except BaseException:
            _pop_scale_ctx(self._ctx_token)
            self._ctx_token = None
            self.loop_frame.__exit__(*sys.exc_info())
            raise
        return warp_var

    def __exit__(self, ptype, value, trace) -> None:
        stack = getattr(self._stack, "frames", [])
        if stack and stack[-1] is self:
            stack.pop()
        try:
            if self.block_frame is not None:
                self.block_frame.__exit__(ptype, value, trace)
                _local.scale_root_depth = 0
        finally:
            if self._ctx_token is not None:
                _pop_scale_ctx(self._ctx_token)
                self._ctx_token = None
            self.loop_frame.__exit__(ptype, value, trace)


def _normalize_sm_schedule(schedule: Any) -> Any:
    """Keep scale annotations printer/mutator friendly.

    TIR annotations are traversed by generic TVM visitors.  Storing a full
    Buffer object there can crash older TVM printer/mutator paths, while the
    buffer data var is enough for the lowering pass to recover the Buffer from
    PrimFunc.buffer_map.
    """

    if isinstance(schedule, tir.Buffer):
        return schedule.data
    if isinstance(schedule, tir.BufferLoad):
        return schedule.buffer.data
    return schedule


def _has_tvm_ir_builder() -> bool:
    from tvm.script.ir_builder import IRBuilder

    return IRBuilder.is_in_scope()


def _is_thread_axis(name: str, bind: str | None) -> bool:
    axis = bind or name
    return axis in ("thread", "threadIdx.x")


def Scale(
    name: str,
    *extents: int | tir.PrimExpr,
    workgroup: int | tir.PrimExpr | tuple | list | None = None,
    bind: str | None = None,
    num_sms_per_die: int | tir.PrimExpr | None = None,
    cluster_size: int | tir.PrimExpr | None = None,
    sm_schedule: Any | None = None,
    swizzle: int | None = None,
    swizzle_order: str = "row",
) -> Any:
    """Create a logical launch scale axis.

    ``T.Scale`` is a frontend-only launch abstraction.  It is represented as an
    annotated serial loop and lowered by ``tl.transform.LowerScaleLaunch`` before
    normal TileLang lowering.

    The shape of a scale is given by ``workgroup`` -- the number and shape of
    workgroups at this scale under the current parent scale::

        with T.scale("block", workgroup=(m_tiles, n_tiles)) as (bm, bn):
            with T.scale("thread", workgroup=(128,)) as tx:
                ...

    ``workgroup`` is the only shape argument; there is no separate ``grid=``.
    Hierarchy is expressed by nesting scales.  The legacy positional form
    ``T.Scale("block", m, n)`` is still accepted and normalized to
    ``workgroup=(m, n)``; passing both positional extents and ``workgroup`` is
    an error.

    Supported axes:

    - ``"device"`` -> no-op scope (optional top-level wrapper)
    - ``"cluster"`` -> cluster grid coords (supports multi-dim ``workgroup=(cx, cy)``);
      sets ``cluster_dims`` from the companion inner ``block`` axis extent.
    - ``"block"`` -> if a ``cluster`` axis is present in the same nest, the
      cluster-internal rank (``block_rank_in_cluster()``); otherwise the grid
      coords (``blockIdx.x``, supports multi-dim ``workgroup=(gx, gy)``).
    - ``"warp"`` -> warp-level decomposition (thread_count = num_warps * 32)
    - ``"thread"`` -> ``threadIdx.x``
    - ``"sm-cluster"``/``"cta"`` -> ``block_rank_in_cluster()``
    - ``"die"`` -> derived from ``get_smid()``
    - ``"sm"`` -> derived from ``get_smid()`` and ``sm_schedule``

    ``swizzle`` (with optional ``swizzle_order`` "row"/"col") attaches a
    threadblock swizzle pattern to a ``block``/``cluster`` launch axis.  Unlike
    ``T.use_swizzle`` it is emitted by the lowering pass, so it composes with the
    Scale launch loops.
    """

    if not isinstance(name, str) or not name:
        raise ValueError("T.Scale name must be a non-empty string")

    # ``workgroup`` is the uniform shape argument; normalize it into the same
    # positional ``extents`` the rest of the body consumes.
    if workgroup is not None:
        if extents:
            raise ValueError(
                "T.scale(): pass the shape via workgroup= OR positional extents, not both")
        if isinstance(workgroup, (tuple, list)):
            if len(workgroup) == 0:
                raise ValueError("T.scale(): workgroup= must have at least one dimension")
            extents = tuple(workgroup)
        else:
            extents = (workgroup,)

    if swizzle is not None and name not in ("block", "cluster"):
        raise ValueError("swizzle= is only valid on 'block' or 'cluster' scale axes")

    from tilelang.language.eager.builder import Builder

    if Builder.current() is None and not _has_tvm_ir_builder():
        raise JITNoBuilderError(
            "T.Scale() can only be used inside @tilelang.jit or @T.prim_func context. No Builder is available."
        )

    if name == "device":
        # The device scale never contributes a grid dimension. With the
        # default workgroup=(1,) it is a no-op wrapper whose rank var lowers
        # to 0. With workgroup=(n,) the program is SPMD over an n-rank
        # process group: every rank executes the body once and the rank var
        # lowers to tl.get_rank(), so the kernel must be launched by all n
        # ranks with a distributed allocator installed via
        # kernel.initialize(allocator).
        wg = tuple(extents) if extents else (1,)
        if len(wg) != 1:
            raise NotImplementedError(
                "T.scale('device') supports a single workgroup dimension "
                f"(got workgroup={list(wg)})")
        annotations: dict[str, Any] = {
            "tl.scale": True,
            "tl.scale.name": "device",
            "tl.scale.workgroup": list(wg),
        }
        return ScaleFrame(
            serial(wg[0], annotations=annotations),
            opens_root_block=False,
            scale_name="device",
            workgroup=wg,
        )

    if name == "cluster":
        if len(extents) < 1:
            raise ValueError("T.Scale('cluster', ...) requires at least one extent")
        base = {
            "tl.scale": True,
            "tl.scale.name": "cluster",
            # Central workgroup metadata: the full shape at this scale.
            "tl.scale.workgroup": list(extents),
        }
        if swizzle is not None:
            base["tl.scale.swizzle"] = swizzle
            base["tl.scale.swizzle_order"] = swizzle_order
        if len(extents) == 1:
            return ScaleFrame(
                serial(extents[0], annotations=dict(base)),
                opens_root_block=False,
                is_cluster=True,
                scale_name="cluster",
                workgroup=tuple(extents),
            )
        # First axis carries the swizzle/workgroup annotation; the rest just name="cluster".
        frames = []
        for i, ext in enumerate(extents):
            anno = dict(base)
            if i > 0:
                anno.pop("tl.scale.swizzle", None)
                anno.pop("tl.scale.swizzle_order", None)
                anno.pop("tl.scale.workgroup", None)
            frames.append(serial(ext, annotations=anno))
        return MultiScaleFrame(frames, opens_root_block=False, is_cluster=True,
                               scale_name="cluster", workgroup=tuple(extents))

    if name == "warp":
        if len(extents) != 1:
            raise ValueError("T.Scale('warp', num_warps) requires exactly one extent")
        num_warps = extents[0]
        tx_var = tir.Var("_warp_tx", "int32")
        annotations = {
            "tl.scale": True,
            "tl.scale.name": "warp",
            "tl.scale.thread_var": tx_var,
        }
        return WarpScaleFrame(
            serial(num_warps, annotations=annotations),
            tx_var=tx_var,
            workgroup=(num_warps,),
        )

    if name == "block":
        if len(extents) < 1:
            raise ValueError("T.Scale('block', ...) requires at least one extent")
        base = {
            "tl.scale": True,
            "tl.scale.name": "block",
            # Central workgroup metadata: the full shape at this scale.
            "tl.scale.workgroup": list(extents),
        }
        if swizzle is not None:
            base["tl.scale.swizzle"] = swizzle
            base["tl.scale.swizzle_order"] = swizzle_order

        # Inside a cluster, `block` is the cluster-internal CTA rank. A
        # multi-dim workgroup like (2,1,1) describes the per-dim CTA split but is
        # a single scale: emit one serial loop over the flattened rank and return
        # a single rank var. The workgroup shape is preserved in metadata so the
        # lowerer can set cluster_dims = workgroup.
        #
        # Tag the role explicitly here (the frontend knows the cluster nesting)
        # so the lowerer does not have to infer it from scale-loop adjacency --
        # intervening non-scale statements (T.assume/T.alloc/...) must not change
        # the classification.
        if _cluster_active():
            base["tl.scale.is_cluster_rank"] = True
            flat = 1
            for e in extents:
                flat = flat * e
            return ScaleFrame(
                serial(flat, annotations=dict(base)),
                opens_root_block=False,
                scale_name="block",
                workgroup=tuple(extents),
            )

        # Outside a cluster, `block` denotes plain grid coords (blockIdx.x/y/z),
        # one rank var per dim.
        if len(extents) == 1:
            return ScaleFrame(
                serial(extents[0], annotations=dict(base)),
                opens_root_block=False,
                scale_name="block",
                workgroup=tuple(extents),
            )
        frames = []
        for i, ext in enumerate(extents):
            anno = dict(base)
            if i > 0:
                anno.pop("tl.scale.swizzle", None)
                anno.pop("tl.scale.swizzle_order", None)
                anno.pop("tl.scale.workgroup", None)
            frames.append(serial(ext, annotations=anno))
        return MultiScaleFrame(frames, opens_root_block=False,
                               scale_name="block", workgroup=tuple(extents))

    if len(extents) != 1:
        raise ValueError(f"T.Scale('{name}', ...) requires exactly one extent")
    extent = extents[0]

    annotations = {
        "tl.scale": True,
        "tl.scale.name": name,
    }
    if bind is not None:
        annotations["tl.scale.bind"] = bind
    if num_sms_per_die is not None:
        annotations["tl.scale.num_sms_per_die"] = num_sms_per_die
    if cluster_size is not None:
        annotations["tl.scale.cluster_size"] = cluster_size
    if sm_schedule is not None:
        annotations["tl.scale.sm_schedule"] = _normalize_sm_schedule(sm_schedule)

    return ScaleFrame(
        serial(extent, annotations=annotations),
        opens_root_block=_is_thread_axis(name, bind),
        scale_name=name,
        workgroup=(extent,),
    )


# Lowercase alias -- the preferred public spelling.  Shares Scale's full
# signature (incl. workgroup=, and notably no grid=).  Recommended form::
#
#     with T.scale("block", workgroup=(m_tiles, n_tiles)) as (bm, bn):
#         with T.scale("thread", workgroup=(128,)) as tx:
#             ...
scale = Scale
