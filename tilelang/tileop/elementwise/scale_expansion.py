"""Block-scope elementwise expansion templates (milestone 13).

The non-GEMM, non-copy compute family on the generic top-down expansion path. Two
templates live here, both claiming a ``block -> thread`` edge in the
:mod:`tilelang.tileop.scale_expansion` registry and driven by the generic
normalizer (:mod:`tilelang.transform.normalize_scale_expansion`):

- :class:`BlockFillExpansionTemplate` -- ``("block", "tl.tileop.fill")`` (the
  single-value broadcast ``T.fill`` / ``T.clear``);
- :class:`BlockElementwiseExpansionTemplate` -- ``("block",
  "tl.tileop.elementwise")`` (the binary ``T.elementwise(in0, in1, out,
  op="add")`` tile op).

The normalizer relocates the block-scope op into the following thread child,
retags its ``tl.scale_ctx.*`` to the thread path, and (via the storage/barrier
planner) inserts a CTA sync only when the op writes shared and a thread consumer
reads it. All op-specific knowledge lives here; the generic normalizer never
learns the op kind.

MVP boundary (intentionally narrow, mirrors block-copy/fill):

- the segment is exactly one ``Evaluate(Call(<op>, ...))``;
- the enclosing region is an *outermost* ``block`` scale (no cluster/device
  ancestor);
- there is exactly one immediately-following ``thread`` child region to merge into;
- elementwise: ``op="add"``, exactly two inputs + one output, **static** same
  shape, exact + one_to_one (verified via :class:`ElementwiseLayoutContract`).

Sync is not over-introduced: effects come from each operand's storage scope, so a
shared output gets a CTA sync before a thread shared read while a local/global
output gets none. The templates hard-code no barrier.
"""

from __future__ import annotations

from dataclasses import dataclass

from tvm import tirx as tir

from tilelang.tileop.scale_expansion import (
    ExpansionContext,
    ExpansionPlan,
    MemoryEffects,
    ScaleExpansionTemplate,
    ScaleSegment,
    memory_effect_read_for_scope,
    memory_effect_write_for_scope,
    register_scale_expansion_template,
)
from tilelang.tileop.scale_layout import (
    ScaleEdgeLayout,
    ScaleLayout,
    PhysicalStorageLayout,
    ElementwiseLayoutContract,
    compose_scale_edges,
    verify_elementwise_layout_contract,
    scale_edge_layout_from_fragment,
    verify_scale_edge_layout,
)

_FILL_OP_NAME = "tl.tileop.fill"
_ELEMENTWISE_OP_NAME = "tl.tileop.elementwise"
_REGION_OP_NAME = "tl.tileop.region"
_FROM_SCALE = "block"
_TO_SCALE = "thread"

# Must match ElementwiseKind in src/op/elementwise.h.
_ELEMENTWISE_ADD = 0


def _fill_call(stmt):
    """Return the ``tl.tileop.fill`` Call inside an Evaluate, or None."""
    if isinstance(stmt, tir.Evaluate):
        val = stmt.value
        if (isinstance(val, tir.Call) and isinstance(val.op, tir.op.Op)
                and val.op.name == _FILL_OP_NAME):
            return val
    return None


def _region_buffer(region_call):
    """Recover the destination ``tir.Buffer`` from a tl.tileop.region call."""
    if not (isinstance(region_call, tir.Call) and isinstance(region_call.op, tir.op.Op)
            and region_call.op.name == _REGION_OP_NAME):
        return None
    inner = region_call.args[0]
    if isinstance(inner, tir.BufferLoad):
        return inner.buffer
    return None


@dataclass
class BlockFillInfo:
    """Decoded block-scope fill segment."""

    fill_evaluate: tir.Evaluate
    effects: MemoryEffects


class BlockFillExpansionTemplate(ScaleExpansionTemplate):
    """Expand a block-scope elementwise ``T.fill`` into the thread child scale.

    The fill stays a tile op (``tl.tileop.fill`` lowers cooperatively in
    ``LowerTileOp``); "expansion" relocates it into the thread region so it runs
    inside the thread launch, with a CTA sync inserted by the generic barrier
    planner only when it writes shared memory consumed downstream.
    """

    @property
    def from_scale(self) -> str:
        return _FROM_SCALE

    @property
    def to_scale(self) -> str:
        return _TO_SCALE

    @property
    def op_names(self) -> tuple[str, ...]:
        return (_FILL_OP_NAME,)

    def match(self, segment: ScaleSegment, context: ExpansionContext) -> bool:
        return (segment.scale_name == _FROM_SCALE
                and segment.op_name == _FILL_OP_NAME
                and len(segment.stmts) == 1
                and _fill_call(segment.stmts[0]) is not None)

    def decode(self, segment: ScaleSegment, context: ExpansionContext) -> BlockFillInfo:
        if len(segment.stmts) != 1 or _fill_call(segment.stmts[0]) is None:
            raise NotImplementedError(
                "BlockFillExpansion: only a single block-scope T.fill statement "
                "may be expanded into the thread scale (a multi-statement "
                "block-scope segment is not supported yet).")
        call = _fill_call(segment.stmts[0])
        buf = _region_buffer(call.args[0]) if call.args else None
        if buf is None:
            raise NotImplementedError(
                "BlockFillExpansion: could not decode the fill destination buffer "
                "region; this form is not supported yet.")
        scope = buf.scope()
        return BlockFillInfo(fill_evaluate=segment.stmts[0],
                             effects=memory_effect_write_for_scope(scope))

    def validate(self, info: BlockFillInfo, context: ExpansionContext) -> None:
        region = context.region
        # MVP: outermost block scale only (no cluster/device ancestor).
        path = tuple(region.path)
        if path != (_FROM_SCALE,):
            raise NotImplementedError(
                f"BlockFillExpansion: a block-scope fill is only supported under "
                f"an outermost `block` scale (path == ('block',)); got path "
                f"{path}. Nested-scale fill is not supported yet.")
        thread_children = [c for c in region.child_regions()
                           if c.scale_name == _TO_SCALE]
        if len(thread_children) != 1:
            raise NotImplementedError(
                f"BlockFillExpansion: a block-scope fill requires exactly one "
                f"inner `thread` scale to expand into; found "
                f"{len(thread_children)}.")

    def plan(self, info: BlockFillInfo, context: ExpansionContext) -> ExpansionPlan:
        # Report the write effect; the generic BarrierPlanner decides whether a
        # CTA sync is needed (shared -> yes if a thread consumer reads it;
        # local/fragment -> no). The template does NOT hard-code a barrier.
        return ExpansionPlan(
            from_scale=_FROM_SCALE,
            to_scale=_TO_SCALE,
            required_child_workgroup=None,
            lowered_stmts=[info.fill_evaluate],
            barriers_before=[],
            barriers_after=[],
            effects=info.effects,
        )


register_scale_expansion_template(BlockFillExpansionTemplate())


# ---------------------------------------------------------------------------
# Block-scope binary elementwise (add).
# ---------------------------------------------------------------------------


def _elementwise_call(stmt):
    """Return the ``tl.tileop.elementwise`` Call inside an Evaluate, or None."""
    if isinstance(stmt, tir.Evaluate):
        val = stmt.value
        if (isinstance(val, tir.Call) and isinstance(val.op, tir.op.Op)
                and val.op.name == _ELEMENTWISE_OP_NAME):
            return val
    return None


def _region_buffer_and_extents(region_call):
    """Recover (buffer, [static extents]) from a tl.tileop.region call.

    The region call is ``region(buffer_load, access_type, *extents)``. Returns
    ``(buffer, extents_or_None)`` where extents is a list of python ints, or None
    if any extent is non-static (used to hard-reject dynamic shapes in E2).
    """
    if not (isinstance(region_call, tir.Call) and isinstance(region_call.op, tir.op.Op)
            and region_call.op.name == _REGION_OP_NAME):
        return None, None
    inner = region_call.args[0]
    if not isinstance(inner, tir.BufferLoad):
        return None, None
    extents = []
    for e in region_call.args[2:]:
        if isinstance(e, tir.IntImm):
            extents.append(int(e.value))
        else:
            extents = None
            break
    return inner.buffer, extents


@dataclass
class BlockElementwiseInfo:
    """Decoded block-scope elementwise segment."""

    elementwise_evaluate: tir.Evaluate
    out_buffer: object
    in0_buffer: object
    in1_buffer: object
    shape: tuple        # static output extents
    effects: MemoryEffects


def _static_int(expr):
    """Return ``expr`` as a Python int if it is a static integer, else ``None``.

    Accepts plain Python ints and constant ``tir.IntImm`` expressions; any
    symbolic / non-constant extent yields ``None`` (caller fail-closes).
    """
    if isinstance(expr, int):
        return int(expr)
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    return None


def _block_thread_scale_layout(shape, num_threads) -> ScaleLayout:
    """Build the pre-lowering CLAIMED ``block -> thread`` ownership ScaleLayout.

    This is edge (1) in the end-to-end binding model (see the "End-to-end edge
    binding" block below): the per-element round-robin ownership the template
    asserts *before* lowering, when the vectorize width is still unknown. The
    elementwise tile of ``total = product(shape)`` logical elements is owned by
    ``num_threads`` threads, each owning ``total // num_threads`` local elements,
    with the scalar (V=1) SIMT interleaving

        i -> (thread = i % num_threads, local = i // num_threads)

    so element ``i`` is claimed by thread ``i % num_threads`` at local index
    ``i // num_threads``. This is a genuine ``ScaleEdgeLayout(block -> thread)``
    (``child_shape == (num_threads,)``, ``local_shape == (total // num_threads,)``),
    NOT the old one-element-per-"thread" placeholder; it passes
    ``verify_scale_edge_layout`` as exact + one_to_one only when the partition is
    actually a clean cover (``total % num_threads == 0``), which the caller checks.

    IMPORTANT: this claimed edge equals the actual SIMT partition only when the
    real vectorize width is 1. When lowering vectorizes (V > 1) the actual
    partition is ``thread = (i // V) % num_threads`` and this edge will NOT match
    it -- use :func:`expected_actual_partition_edge` at the real V for that. This
    is the *ownership* edge (which thread owns which element); the physical
    fragment/register-slot layout is separate (see
    :func:`_placeholder_storage_layout_for`).
    """
    total = 1
    for d in shape:
        total *= d
    edge = ScaleEdgeLayout(
        from_scale=_FROM_SCALE,
        to_scale=_TO_SCALE,
        input_shape=(total,),
        child_shape=(num_threads,),
        local_shape=(total // num_threads,),
        index_map=tir.IndexMap.from_func(
            lambda i: (i % num_threads, i // num_threads), ndim=1,
            index_dtype="int32"),
        coverage="exact",
        policy="one_to_one",
    )
    return compose_scale_edges([edge])


def _parallel_op_partition_fragment(total, num_threads, vectorize_size):
    """Reconstruct the EXACT thread partition ``ParallelOp`` lowering produces.

    This independently mirrors the partition formula in
    ``src/transform/loop_partition.cc`` (``LoopPartitioner::Partition``), which is
    what ``backend::Elementwise::Lower`` calls (via ``PartitionLoop`` after
    ``ParallelOp::InferLayout(kFree)``) to turn the flat SIMT loop into a
    per-thread loop. For a flattened logical index ``flat`` over ``num_threads``
    threads with vectorize width ``V == vectorize_size`` the production formula is

        thread = (flat // V) % num_threads
        local  = (flat // V // num_threads) * V + (flat % V)

    (see ``loop_partition.cc`` lines ~201-210). We build the same mapping as a
    Python :class:`Fragment` so it can be diffed against the layout we *claim*
    (:func:`_block_thread_scale_layout`).

    IMPORTANT: this is a *reconstruction*, not the live Fragment. The real
    Fragment is transient C++ state inside ``LowerTileOp`` and is discarded -- it
    is not recoverable from post-lowering TIR (LayoutInference runs before the
    elementwise SIMT loop even exists; the op's block ``layout_map`` stays empty).
    So the shadow comparison validates that *our reconstruction of the documented
    formula* matches our hand-built edge; it does not read back the actual object.
    ``vectorize_size`` is the one input we cannot observe pre-lowering, hence the
    comparison is only made for the ``V == 1`` regime (see
    :func:`_shadow_compare_block_thread_partition`).
    """
    from tilelang.layout import Fragment

    def _fwd_thread(i):
        return tir.floormod(tir.floordiv(i, vectorize_size), num_threads)

    def _fwd_index(i):
        return (tir.floordiv(tir.floordiv(i, vectorize_size), num_threads)
                * vectorize_size + tir.floormod(i, vectorize_size))

    return Fragment([total], forward_thread_fn=_fwd_thread,
                    forward_index_fn=_fwd_index)


def _edges_equal_over_domain(edge_a, edge_b) -> bool:
    """Structural + full-domain equality of two 1D-input ScaleEdgeLayouts.

    Both edges are expected to have a single logical input axis (the flattened
    elementwise tile). Returns True iff the from/to scales, ``child_shape``,
    ``local_shape`` match and ``index_map`` agrees at every logical index in the
    input domain (a complete enumeration, not a sample -- the domains here are the
    small elementwise tiles the MVP admits).
    """
    if (edge_a.from_scale, edge_a.to_scale) != (edge_b.from_scale, edge_b.to_scale):
        return False
    if tuple(int(x) for x in edge_a.child_shape) != tuple(
            int(x) for x in edge_b.child_shape):
        return False
    if tuple(int(x) for x in edge_a.local_shape) != tuple(
            int(x) for x in edge_b.local_shape):
        return False
    in_a = tuple(int(x) for x in edge_a.input_shape)
    in_b = tuple(int(x) for x in edge_b.input_shape)
    if in_a != in_b or len(in_a) != 1:
        return False
    ma = edge_a.index_map.map_indices
    mb = edge_b.index_map.map_indices
    for i in range(in_a[0]):
        oa = tuple(int(x) for x in ma([i]))
        ob = tuple(int(x) for x in mb([i]))
        if oa != ob:
            return False
    return True


def _shadow_compare_block_thread_partition(shape, num_threads) -> None:
    """Shadow-validate our ownership edge against the ParallelOp partition.

    Reconstructs the actual ``ParallelOp`` / ``Fragment`` thread partition (via
    :func:`_parallel_op_partition_fragment` + :func:`scale_edge_layout_from_fragment`)
    and checks it equals the edge we hand-build in
    :func:`_block_thread_scale_layout`. On disagreement raises
    ``NotImplementedError`` (fail-closed): a claimed ownership layout that does not
    match what lowering will actually do must not silently pass.

    Scope/limit -- only the ``vectorize_size == 1`` regime is compared. The real
    vectorize width is chosen inside C++ (`GetVectorizeSize`, shrunk so
    ``total % (num_threads * V) == 0``) and is NOT observable from the Python
    template before lowering. When V > 1 the production partition is
    ``thread = (i // V) % num_threads`` (a vectorized round-robin), which differs
    from our V=1 edge ``i -> (i % num_threads, i // num_threads)``. Rather than
    overclaim a match we cannot prove, this comparison is performed only for V=1,
    where the production formula reduces exactly to our edge. This is shadow
    validation only; it does not drive codegen or alter lowering.
    """
    total = 1
    for d in shape:
        total *= d
    # Our claimed ownership edge.
    claimed = _block_thread_scale_layout(shape, num_threads).edges[0]
    # The reconstructed production partition, in the V=1 regime.
    frag = _parallel_op_partition_fragment(total, num_threads, vectorize_size=1)
    actual = scale_edge_layout_from_fragment(frag)
    # The reconstructed edge must itself be a legal exact/one_to_one cover.
    res = verify_scale_edge_layout(actual)
    if not res.ok:
        raise NotImplementedError(
            "layout-driven block->thread elementwise validation: the "
            f"reconstructed ParallelOp partition edge is not a legal cover: "
            f"{res.reason}")
    if not _edges_equal_over_domain(claimed, actual):
        raise NotImplementedError(
            "layout-driven block->thread elementwise validation: the claimed "
            "block->thread ownership edge does not match the reconstructed "
            "ParallelOp/Fragment thread partition (vectorize_size=1 regime). "
            "Refusing to proceed with a layout that disagrees with the actual "
            "SIMT lowering.")


# Annotation key written by the C++ elementwise lowering
# (``src/backend/common/op/elementwise.h``, ``attr::kScaleLayoutPartition``). It
# carries the ACTUAL ParallelOp loop-partition ``Fragment`` -- the real
# (thread, local) mapping built with the real vectorize width -- attached to the
# lowered statement as shadow-only metadata. It drives no codegen and is consumed
# by nothing in C++; it exists so this Python layer can recover and diff the real
# partition. See :func:`collect_elementwise_partition_fragments`.
ELEMENTWISE_PARTITION_ANNOTATION = "tl.scale_layout.partition"


def collect_elementwise_partition_fragments(stmt_or_func):
    """Collect the partition ``Fragment``s exposed by elementwise lowering.

    Walks a lowered TIR statement / PrimFunc and returns the list of ``Fragment``
    objects attached under :data:`ELEMENTWISE_PARTITION_ANNOTATION` by the C++
    ``Elementwise::Lower``. Read-only: this only reads metadata the lowering put
    there for shadow validation; it never alters lowering. Each Fragment is the
    real ParallelOp loop-partition layout (logical index -> (thread, local)) for
    one lowered ``tl.tileop.elementwise``.
    """
    from tvm import tir as _tir
    body = stmt_or_func.body if isinstance(stmt_or_func, _tir.PrimFunc) else stmt_or_func
    out = []

    def _visit(node):
        if (isinstance(node, _tir.AttrStmt)
                and node.attr_key == ELEMENTWISE_PARTITION_ANNOTATION):
            out.append(node.node)

    _tir.stmt_functor.post_order_visit(body, _visit)
    return out


def actual_partition_edge_from_fragment(fragment) -> ScaleEdgeLayout:
    """Derive the post-lowering ACTUAL ``block -> thread`` partition edge.

    Edge (3) in the binding model: a thin wrapper over
    :func:`scale_edge_layout_from_fragment` for the Fragments returned by
    :func:`collect_elementwise_partition_fragments`. The Fragment's input shape is
    the (possibly multi-dim) elementwise loop shape; the resulting edge's
    ``index_map`` maps a logical index to ``(thread, local...)`` exactly as the
    SIMT lowering does, at the real vectorize width. This is the partition lowering
    actually produced -- not the pre-lowering claimed ownership edge.
    """
    return scale_edge_layout_from_fragment(fragment)


# ---------------------------------------------------------------------------
# End-to-end edge binding (shadow-only). Three DISTINCT edges participate:
#
#   1. pre-lowering CLAIMED ownership edge -- `_block_thread_scale_layout`:
#      i -> (i % T, i // T). This is the per-element round-robin ownership the
#      template asserts before lowering, when the vectorize width V is unknown.
#   2. reconstructed EXPECTED actual partition edge --
#      `expected_actual_partition_edge`: the documented ParallelOp partition
#      formula at a *given* V (thread=(i//V)%T, local=(i//V//T)*V + i%V). This is
#      NOT pre-lowering ownership; it is what we expect the SIMT lowering to
#      actually produce once V is known.
#   3. post-lowering ACTUAL annotated edge -- the Fragment recovered from the
#      `tl.scale_layout.partition` annotation via
#      `actual_partition_edge_from_fragment`.
#
# For V == 1 all three coincide. For V > 1 edge (1) differs from (3) (the
# lowering vectorizes), while edge (2) built at the real V equals (3). These
# helpers let a test bind (1)/(2) against (3) for one concrete kernel. All
# read-only: nothing here drives codegen.
# ---------------------------------------------------------------------------


def _edge_total(edge) -> int:
    """Product of an edge's logical input shape (its flattened domain size)."""
    total = 1
    for d in edge.input_shape:
        total *= int(d)
    return total


def _edge_point(edge, flat):
    """Evaluate an edge's index_map at row-major-flattened logical index ``flat``.

    Returns ``(thread, (local...))``. Handles multi-dim input shapes (the actual
    annotated Fragment keeps the loop's per-axis shape, e.g. (64, 64)) by
    unflattening ``flat`` into per-axis indices before mapping.
    """
    in_shape = tuple(int(x) for x in edge.input_shape)
    idx = []
    f = flat
    for d in reversed(in_shape):
        idx.append(f % d)
        f //= d
    idx = list(reversed(idx))
    out = tuple(int(x) for x in edge.index_map.map_indices(idx))
    return out[0], out[1:]


def edges_agree_over_flat_domain(edge_a, edge_b) -> bool:
    """Full-domain equality of two edges over their flattened logical domain.

    Unlike :func:`_edges_equal_over_domain` (which requires identical 1D input
    shapes), this compares edges whose input shapes may differ in rank but cover
    the same number of logical elements -- e.g. a 1D ``(4096,)`` reconstruction
    against the multi-dim ``(64, 64)`` annotated Fragment. Requires equal total
    domain size, equal ``child_shape`` / ``local_shape``, and the same
    ``(thread, local...)`` image at every flattened index (complete enumeration).
    """
    if _edge_total(edge_a) != _edge_total(edge_b):
        return False
    if tuple(int(x) for x in edge_a.child_shape) != tuple(
            int(x) for x in edge_b.child_shape):
        return False
    if tuple(int(x) for x in edge_a.local_shape) != tuple(
            int(x) for x in edge_b.local_shape):
        return False
    return all(_edge_point(edge_a, f) == _edge_point(edge_b, f)
               for f in range(_edge_total(edge_a)))


def recover_vectorize_size(edge) -> int:
    """Recover the vectorize width V from an actual partition edge.

    The partition owns flattened indices ``0..V-1`` on thread 0 (consecutive
    vector lanes stay on one thread), so V is the first flattened index whose
    owning thread is not 0. A purely scalar (single-thread, or every element on
    thread 0) edge yields the full domain size; callers asserting V>1 should guard
    accordingly.
    """
    total = _edge_total(edge)
    for f in range(1, total):
        if _edge_point(edge, f)[0] != 0:
            return f
    return total


def expected_actual_partition_edge(shape, num_threads, vectorize_size):
    """Build the EXPECTED actual block->thread partition edge at a known ``V``.

    This is edge (2) in the binding model above: the documented ParallelOp
    partition (``thread=(i//V)%T``, ``local=(i//V//T)*V + i%V``) reconstructed for
    a *given* vectorize width. It is the prediction of what lowering produces once
    V is known -- explicitly NOT the pre-lowering claimed ownership edge
    (:func:`_block_thread_scale_layout`, which assumes V=1). Reuses
    :func:`_parallel_op_partition_fragment` so the formula stays in one place.
    """
    total = 1
    for d in shape:
        total *= d
    frag = _parallel_op_partition_fragment(total, num_threads,
                                           vectorize_size=vectorize_size)
    return scale_edge_layout_from_fragment(frag)


def bind_partition_edge_or_raise(predicted_edge, actual_edge,
                                 *, what="partition edge") -> None:
    """Bind a predicted edge against the actual annotated edge, or fail-closed.

    Compares ``predicted_edge`` against ``actual_edge`` (the post-lowering edge
    from :func:`actual_partition_edge_from_fragment`) over the full flattened
    domain via :func:`edges_agree_over_flat_domain`. Raises ``NotImplementedError``
    if they disagree -- the fail-closed contract: a prediction that does not match
    what lowering actually produced must not silently pass. ``what`` names the
    predicted edge in the error message (e.g. "claimed ownership edge",
    "expected actual partition edge at V=8").

    Shadow-only: this only *verifies* a prediction against recovered metadata; it
    does not drive codegen.
    """
    if not edges_agree_over_flat_domain(predicted_edge, actual_edge):
        raise NotImplementedError(
            f"layout-driven block->thread elementwise validation: the "
            f"{what} does not match the actual ParallelOp/Fragment partition "
            f"recovered from the `{ELEMENTWISE_PARTITION_ANNOTATION}` annotation. "
            f"Refusing to bind a layout that disagrees with the real SIMT "
            f"lowering.")


def _placeholder_storage_layout_for(buffer, shape) -> PhysicalStorageLayout:
    """A PLACEHOLDER identity PhysicalStorageLayout for `buffer` over `shape`.

    Unlike the ownership layout (now a real block->thread edge, see
    :func:`_block_thread_scale_layout`), this physical-storage layout is still
    only a structural storage-class check: it carries the buffer's storage class
    (memory_layer / instance_scope from its scope) and a flattened identity index
    map so the contract verifier can run. It is NOT a real physical
    (swizzle / fragment-slot) layout proof.
    """
    from tilelang.tileop.scale_expansion import classify_storage_scope
    cls = classify_storage_scope(buffer.scope())
    instance = {"shared": "block", "local": "thread", "global": "device"}[cls]
    total = 1
    for d in shape:
        total *= d
    return PhysicalStorageLayout(
        buffer=buffer,
        logical_shape=(total,),
        memory_layer=cls,
        instance_scope=instance,
        index_map=tir.IndexMap.from_func(lambda i: (i,), ndim=1,
                                         index_dtype="int32"),
    )


class BlockElementwiseExpansionTemplate(ScaleExpansionTemplate):
    """Expand a block-scope ``T.elementwise(add)`` into the thread child scale.

    The elementwise stays a tile op (``tl.tileop.elementwise`` lowers through the
    ParallelOp SIMT path); "expansion" relocates it into the thread region so it
    runs inside the thread launch, with a CTA sync inserted by the generic barrier
    planner only when the output is shared and a thread consumer reads it.
    """

    @property
    def from_scale(self) -> str:
        return _FROM_SCALE

    @property
    def to_scale(self) -> str:
        return _TO_SCALE

    @property
    def op_names(self) -> tuple[str, ...]:
        return (_ELEMENTWISE_OP_NAME,)

    def match(self, segment: ScaleSegment, context: ExpansionContext) -> bool:
        return (segment.scale_name == _FROM_SCALE
                and segment.op_name == _ELEMENTWISE_OP_NAME
                and len(segment.stmts) == 1
                and _elementwise_call(segment.stmts[0]) is not None)

    def decode(self, segment: ScaleSegment,
               context: ExpansionContext) -> BlockElementwiseInfo:
        if len(segment.stmts) != 1 or _elementwise_call(segment.stmts[0]) is None:
            raise NotImplementedError(
                "BlockElementwiseExpansion: only a single block-scope "
                "T.elementwise statement may be expanded into the thread scale.")
        call = _elementwise_call(segment.stmts[0])
        # Call args: [out_region, in0_region, in1_region, op_kind].
        if len(call.args) != 4:
            raise NotImplementedError(
                "BlockElementwiseExpansion: unexpected elementwise arg count "
                f"{len(call.args)} (expected 4).")
        op_kind = call.args[3]
        if not (isinstance(op_kind, tir.IntImm)
                and int(op_kind.value) == _ELEMENTWISE_ADD):
            raise NotImplementedError(
                "BlockElementwiseExpansion: only op='add' is supported yet.")

        out_buf, out_ext = _region_buffer_and_extents(call.args[0])
        in0_buf, in0_ext = _region_buffer_and_extents(call.args[1])
        in1_buf, in1_ext = _region_buffer_and_extents(call.args[2])
        if out_buf is None or in0_buf is None or in1_buf is None:
            raise NotImplementedError(
                "BlockElementwiseExpansion: could not decode operand buffer "
                "regions; this form is not supported yet.")
        # E2 hard-rejects symbolic/dynamic shapes even though the E1 tile op is
        # lenient: a layout contract can only be proven for static shapes.
        if out_ext is None or in0_ext is None or in1_ext is None:
            raise NotImplementedError(
                "BlockElementwiseExpansion: only static operand shapes are "
                "supported yet (dynamic extents are not).")
        if not (tuple(out_ext) == tuple(in0_ext) == tuple(in1_ext)):
            raise NotImplementedError(
                f"BlockElementwiseExpansion: operand shapes must match "
                f"(out={out_ext}, in0={in0_ext}, in1={in1_ext}); broadcast is "
                f"not supported yet.")

        eff = (memory_effect_read_for_scope(in0_buf.scope())
               .merged_with(memory_effect_read_for_scope(in1_buf.scope()))
               .merged_with(memory_effect_write_for_scope(out_buf.scope())))
        return BlockElementwiseInfo(
            elementwise_evaluate=segment.stmts[0],
            out_buffer=out_buf, in0_buffer=in0_buf, in1_buffer=in1_buf,
            shape=tuple(out_ext), effects=eff)

    def validate(self, info: BlockElementwiseInfo,
                 context: ExpansionContext) -> None:
        region = context.region
        # MVP: outermost block scale only (no cluster/device ancestor).
        path = tuple(region.path)
        if path != (_FROM_SCALE,):
            raise NotImplementedError(
                f"BlockElementwiseExpansion: a block-scope elementwise is only "
                f"supported under an outermost `block` scale (path == "
                f"('block',)); got path {path}. Nested-scale elementwise is not "
                f"supported yet.")
        thread_children = [c for c in region.child_regions()
                           if c.scale_name == _TO_SCALE]
        if len(thread_children) != 1:
            raise NotImplementedError(
                f"BlockElementwiseExpansion: a block-scope elementwise requires "
                f"exactly one inner `thread` scale to expand into; found "
                f"{len(thread_children)}.")
        # dtype: same input/output dtype (the E1 tile op also enforces this).
        if not (info.in0_buffer.dtype == info.out_buffer.dtype
                and info.in1_buffer.dtype == info.out_buffer.dtype):
            raise NotImplementedError(
                "BlockElementwiseExpansion: operand dtypes must match "
                "(no implicit cast yet).")

        # Read the inner thread workgroup (the real block->thread partition
        # size). The generic `thread` axis is a single serial launch loop whose
        # extent IS the thread count; it carries no `tl.scale.workgroup`
        # annotation (only `block`/`device`/`warp` do), so prefer the loop extent
        # and fall back to an explicit workgroup if one was attached. MVP: only a
        # 1D static thread workgroup.
        thread_child = thread_children[0]
        thread_wg = tuple(thread_child.workgroup)
        if thread_wg:
            if len(thread_wg) != 1:
                raise NotImplementedError(
                    "layout-driven block->thread elementwise validation: only a "
                    f"1D thread workgroup is supported yet; got workgroup "
                    f"{thread_wg}.")
            thread_extent = thread_wg[0]
        elif thread_child.loop is not None:
            thread_extent = thread_child.loop.extent
        else:
            raise NotImplementedError(
                "layout-driven block->thread elementwise validation: could not "
                "determine the inner thread workgroup extent (no workgroup and "
                "no scale loop).")
        num_threads = _static_int(thread_extent)
        if num_threads is None or num_threads <= 0:
            raise NotImplementedError(
                "layout-driven block->thread elementwise validation: the thread "
                f"workgroup extent must be a positive static int; got "
                f"{thread_extent!r}.")

        total = 1
        for d in info.shape:
            total *= d
        if total % num_threads != 0:
            raise NotImplementedError(
                f"layout-driven block->thread elementwise validation: the "
                f"elementwise tile of {total} elements (shape {info.shape}) is "
                f"not evenly partitioned by {num_threads} threads "
                f"({total} % {num_threads} != 0); a clean exact one_to_one "
                f"block->thread partition is required.")

        # Build and verify the REAL block->thread ownership layout (replaces the
        # old one-element-per-"thread" placeholder): num_threads threads, each
        # owning total // num_threads local elements, SIMT interleaving
        # i -> (i % num_threads, i // num_threads). The physical storage layouts
        # remain structural storage-class checks (not a swizzle/fragment-slot
        # proof). This is a pre-lowering ownership validation: the actual physical
        # thread partition is still produced by the ParallelOp/Fragment SIMT
        # lowering of tl.tileop.elementwise at lower time -- this check proves the
        # ownership edge is a legal exact/one_to_one cover before that runs.
        out_sl = _block_thread_scale_layout(info.shape, num_threads)
        contract = ElementwiseLayoutContract(
            input_scale_layouts=(out_sl, out_sl),
            output_scale_layout=out_sl,
            input_storage_layouts=(
                _placeholder_storage_layout_for(info.in0_buffer, info.shape),
                _placeholder_storage_layout_for(info.in1_buffer, info.shape)),
            output_storage_layout=_placeholder_storage_layout_for(
                info.out_buffer, info.shape),
            policy="one_to_one")
        res = verify_elementwise_layout_contract(contract)
        if not res.ok:
            raise NotImplementedError(
                f"layout-driven block->thread elementwise validation failed: "
                f"{res.reason}")

        # Shadow-compare the claimed ownership edge against a reconstruction of
        # the actual ParallelOp/Fragment thread partition (V=1 regime). This
        # proves our layout agrees with what the existing SIMT lowering does, or
        # fail-closes. It is validation only -- lowering is still owned by
        # ParallelOp/Fragment and this comparison drives no codegen.
        _shadow_compare_block_thread_partition(info.shape, num_threads)

    def plan(self, info: BlockElementwiseInfo,
             context: ExpansionContext) -> ExpansionPlan:
        # Report decoded read/write effects; the generic BarrierPlanner derives a
        # CTA sync from (writes_shared producer / thread reads_shared consumer).
        # The template hard-codes no barrier.
        return ExpansionPlan(
            from_scale=_FROM_SCALE,
            to_scale=_TO_SCALE,
            required_child_workgroup=None,
            lowered_stmts=[info.elementwise_evaluate],
            barriers_before=[],
            barriers_after=[],
            effects=info.effects,
        )


register_scale_expansion_template(BlockElementwiseExpansionTemplate())
