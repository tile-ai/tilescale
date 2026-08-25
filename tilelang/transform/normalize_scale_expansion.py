"""Generic top-down scale-expansion normalizer (Long-Term Plan skeleton).

This is the *generic*, op-agnostic driver for the top-down scale expansion
pipeline (see ``docs/compiler_internals/scale_programming_plan.md``, "Long-Term
Plan: Top-Down Scale Expansion"). It provides two things at this milestone:

1. :func:`build_region_tree` -- a **read-only** parser that turns a PrimFunc (or
   a TIR statement) into a :class:`~tilelang.tileop.scale_expansion.ScaleRegion`
   tree of nested scales and ordered :class:`ScaleSegment`\\ s.
2. :func:`plan_region_expansions` / :func:`NormalizeScaleExpansion` -- the
   planner + generic rewrite: it finds parent segments that need expansion,
   resolves a :class:`ScaleExpansionTemplate` by ``(from_scale, op_name)``, asks
   it for a generic :class:`ExpansionPlan`, then -- op-agnostically -- merges the
   plan's lowered statements and barriers into the unique compatible child region
   and deletes the parent segment. It loud-errors (``NotImplementedError``) for
   any segment that has no registered template or any unsupported merge.

The driver is strictly generic: it knows about scale order, lexical regions,
statement ordering, and template registration -- never that ``tl.tileop.copy``
moves into a ``thread`` body or that ``tl.tileop.gemm`` uses ``BM=128``. All such
knowledge belongs in :class:`ScaleExpansionTemplate` implementations. The
``("block", "tl.tileop.copy")`` template
(:mod:`tilelang.tileop.copy.scale_expansion`) is the first registered template;
``LowerScaleLaunch`` still strictly rejects any interleaving the normalizer did
not expand (raw stores, control flow, cluster-path copies, ...).
"""

from __future__ import annotations

from typing import Optional

from tvm import tirx as tir
from tvm.tirx import PrimFunc
from tvm.tirx.stmt_functor import post_order_visit
from tvm.tirx.transform import prim_func_pass

from tilelang.tileop.scale_expansion import (
    BarrierSpec,  # noqa: F401  (re-exported for template authors / tests)
    ExpansionContext,
    ExpansionPlan,  # noqa: F401
    MemoryEffects,
    ScaleRegion,
    ScaleSegment,
    resolve_scale_expansion_template,
    has_scale_expansion_templates,
    ensure_default_scale_expansion_templates_registered,
)
from tilelang.tileop.scale_barrier_planner import (
    ScaleDependencyAnalysis,
    BarrierPlanner,
)
from tilelang.tileop.scale_semantics import resolve_scale_semantics
from tilelang.tileop.scale_stage import (
    StageBoundary,
    StagePlan,
    StageProgram,
)

_ROOT_SCALE = "<root>"
_SCALE_ANN = "tl.scale"
_SCALE_NAME_ANN = "tl.scale.name"
_SCALE_WORKGROUP_ANN = "tl.scale.workgroup"

# tl.scale_ctx.* metadata keys stamped on tile-op calls by the frontend. When a
# tile op is relocated into a smaller scale by an expansion plan, these must be
# retagged to the destination scale so downstream consumers (e.g. the GEMM
# dispatch gate) see a consistent name/path -- the moved op now belongs to the
# child scale, not its original parent scale.
_CTX_NAME_KEY = "tl.scale_ctx.name"
_CTX_PATH_KEY = "tl.scale_ctx.path"


# ---------------------------------------------------------------------------
# Annotation helpers.
# ---------------------------------------------------------------------------


def _str_val(v) -> str:
    return v.value if hasattr(v, "value") else str(v)


def _scale_for_name(node) -> Optional[str]:
    """Return the scale name of a ``tl.scale`` For loop, else ``None``."""
    if not isinstance(node, tir.For):
        return None
    ann = node.annotations
    if not ann or ann.get(_SCALE_ANN) is None:
        return None
    name = ann.get(_SCALE_NAME_ANN)
    return _str_val(name) if name is not None else None


def _scale_for_workgroup(node) -> tuple:
    ann = node.annotations or {}
    wg = ann.get(_SCALE_WORKGROUP_ANN)
    if wg is None:
        return ()
    return tuple(wg)


def _tileop_name(stmt) -> Optional[str]:
    """If ``stmt`` is ``Evaluate(tl.tileop.*)``, return the op name, else None."""
    if isinstance(stmt, tir.Evaluate):
        val = stmt.value
        if (isinstance(val, tir.Call) and isinstance(val.op, tir.op.Op)
                and val.op.name.startswith("tl.tileop.")):
            return val.op.name
    return None


def _retag_tileop_to_scale(stmt, scale_name: str, path: tuple):
    """Retag a relocated tile-op statement's ``tl.scale_ctx.*`` to a new scale.

    Generic and op-agnostic: any ``Evaluate(Call(tl.tileop.*))`` has its
    ``tl.scale_ctx.name`` set to ``scale_name`` and ``tl.scale_ctx.path`` set to
    ``path`` (other annotations -- workgroup, parent, user keys -- are preserved).
    When an expansion plan moves a tile op into a smaller (child) scale, the op
    now executes in that child scale, so its context metadata must match. A
    non-tile-op statement is returned unchanged.
    """
    if not isinstance(stmt, tir.Evaluate):
        return stmt
    call = stmt.value
    if not (isinstance(call, tir.Call) and isinstance(call.op, tir.op.Op)
            and call.op.name.startswith("tl.tileop.")):
        return stmt
    ann = dict(call.annotations) if call.annotations else {}
    ann[_CTX_NAME_KEY] = tir.StringImm(scale_name)
    ann[_CTX_PATH_KEY] = [tir.StringImm(p) for p in path]
    new_call = tir.Call(call.dtype, call.op, call.args, ann)
    return tir.Evaluate(new_call)


# ---------------------------------------------------------------------------
# Statement flattening.
# ---------------------------------------------------------------------------


def _flatten(stmt) -> list:
    """Flatten one level of statements, unwrapping Seq / Block / BlockRealize.

    ``SeqStmt`` is spliced; a ``BlockRealize`` / ``Block`` is unwrapped to its
    body (its ``alloc_buffers`` are structural scaffolding, not segments). The
    result is a flat list of "logical statements" at this nesting level, in
    lexical order. Scale ``For`` loops are returned as-is (the caller turns them
    into child regions); everything else is a candidate segment statement.
    """
    if isinstance(stmt, tir.SeqStmt):
        out = []
        for s in stmt.seq:
            out.extend(_flatten(s))
        return out
    if isinstance(stmt, tir.SBlockRealize):
        return _flatten(stmt.block)
    if isinstance(stmt, tir.SBlock):
        return _flatten(stmt.body)
    return [stmt]


# ---------------------------------------------------------------------------
# Effect inference (coarse, op-agnostic).
# ---------------------------------------------------------------------------


def _buffer_scope(buffer) -> str:
    try:
        return buffer.scope()
    except Exception:  # pragma: no cover - defensive
        return "global"


def _classify_write(eff: MemoryEffects, scope: str) -> None:
    if "shared" in scope:
        eff.writes_shared = True
    elif "local" in scope or scope in ("warp", "fragment"):
        eff.writes_local = True
    else:
        eff.writes_global = True


def _classify_read(eff: MemoryEffects, scope: str) -> None:
    if "shared" in scope:
        eff.reads_shared = True
    elif "local" in scope or scope in ("warp", "fragment"):
        eff.reads_local = True
    else:
        eff.reads_global = True


def _segment_effects(stmts) -> MemoryEffects:
    """Coarse memory effects of a run of statements (which spaces read/written)."""
    eff = MemoryEffects()

    def visit(node):
        if isinstance(node, tir.BufferStore):
            _classify_write(eff, _buffer_scope(node.buffer))
        elif isinstance(node, tir.BufferLoad):
            _classify_read(eff, _buffer_scope(node.buffer))

    for st in stmts:
        post_order_visit(st, visit)
    return eff


# ---------------------------------------------------------------------------
# Region parser (read-only).
# ---------------------------------------------------------------------------


def _collect_tile_op_names(stmts) -> tuple:
    """Return every statement-level ``tl.tileop.*`` op name in ``stmts`` (in order).

    Op-agnostic: collects the op of each ``Evaluate(Call(tl.tileop.*))`` statement
    (handles a tile op grouped with siblings, e.g. a store + gemm, or two gemms).
    Only *statement-level* tile ops count -- operand wrappers like ``T.region``
    (``tl.tileop.region``) appear as call *arguments*, not as Evaluate statements,
    so they are correctly excluded. Used by the planner to fail-close on mixed /
    multi tile-op segments that cannot be cleanly expanded.
    """
    names: list = []
    for s in stmts:
        name = _tileop_name(s)
        if name is not None:
            names.append(name)
    return tuple(names)


def _make_segment(scale_name: str, path: tuple, stmts: list) -> ScaleSegment:
    """Classify a run of non-region statements into a :class:`ScaleSegment`."""
    op_name = None
    if len(stmts) == 1:
        op_name = _tileop_name(stmts[0])
    tile_op_names = _collect_tile_op_names(stmts)
    is_side_effect = op_name is None and any(
        _stmt_has_effect(s) for s in stmts)
    return ScaleSegment(
        scale_name=scale_name,
        path=path,
        stmts=list(stmts),
        op_name=op_name,
        tile_op_names=tile_op_names,
        is_side_effect=is_side_effect,
        effects=_segment_effects(stmts),
    )


def _stmt_has_effect(stmt) -> bool:
    """True if ``stmt`` carries an execution side effect (store / non-noop)."""
    if isinstance(stmt, tir.BufferStore):
        return True
    if _tileop_name(stmt) is not None:
        return True
    if isinstance(stmt, tir.Evaluate):
        val = stmt.value
        # T.assume / tir.assume and Evaluate(const) are no-ops.
        if isinstance(val, tir.IntImm):
            return False
        if isinstance(val, tir.Call) and isinstance(val.op, tir.op.Op):
            name = val.op.name
            if name == "tir.assume" or name.endswith(".assume"):
                return False
        return True
    if isinstance(stmt, (tir.IfThenElse, tir.For, tir.While)):
        return True
    return False


def _build_region(scale_name: str, path: tuple, loop, workgroup,
                  body) -> ScaleRegion:
    """Recursively build a :class:`ScaleRegion` from a scale loop body."""
    region = ScaleRegion(scale_name=scale_name, path=path, loop=loop,
                         workgroup=workgroup, items=[])
    pending: list = []

    def flush():
        if pending:
            region.items.append(_make_segment(scale_name, path, pending))
            pending.clear()

    for stmt in _flatten(body):
        child_name = _scale_for_name(stmt)
        if child_name is not None:
            flush()
            child_path = path + (child_name,)
            child = _build_region(child_name, child_path, stmt,
                                  _scale_for_workgroup(stmt), stmt.body)
            region.items.append(child)
        else:
            pending.append(stmt)
    flush()
    return region


def build_region_tree(func_or_stmt) -> ScaleRegion:
    """Parse a PrimFunc (or statement) into a read-only :class:`ScaleRegion` tree.

    The returned root region has ``scale_name == "<root>"`` and ``loop is None``;
    its ``items`` are the top-level segments and scale regions in lexical order.
    Purely structural -- it performs no rewrite and resolves no templates.
    """
    body = func_or_stmt.body if isinstance(func_or_stmt, PrimFunc) else func_or_stmt
    return _build_region(_ROOT_SCALE, (), None, (), body)


# ---------------------------------------------------------------------------
# Expandable-segment discovery + fail-closed planner skeleton.
# ---------------------------------------------------------------------------


def _stmt_contains_tile_op(stmt) -> bool:
    """True if ``stmt`` contains any ``tl.tileop.*`` call anywhere within it.

    Used to recognize a tile op nested inside control flow (e.g. a static
    ``T.serial`` loop of GEMMs), which the flat segment scan does not surface as a
    statement-level tile op.
    """
    found = [False]

    def visit(node):
        if isinstance(node, tir.Call) and isinstance(node.op, tir.op.Op) \
                and node.op.name.startswith("tl.tileop.") \
                and node.op.name != "tl.tileop.region":
            found[0] = True

    post_order_visit(stmt, visit)
    return found[0]


def _segment_loop_with_tile_op(segment: ScaleSegment):
    """Return a loop/control-flow stmt in the segment that wraps a tile op, else None.

    Detects the "ordered stage loop" shape: a (non-scale) ``For`` / ``While`` /
    ``IfThenElse`` at this scale whose body contains a tile op. Such a construct
    is an ordered scale program (repeated / conditional stages), not a single
    expandable op.
    """
    for s in segment.stmts:
        if isinstance(s, (tir.For, tir.While, tir.IfThenElse)) \
                and _stmt_contains_tile_op(s):
            return s
    return None


def _plan_stage_program(region: ScaleRegion, segment: ScaleSegment):
    """Build a :class:`StageProgram` skeleton if ``segment`` is an ordered program.

    Generic + scale-parametric: only fires for a scale whose semantics support a
    *stage boundary* (``supports_stage_boundary``, e.g. ``device``). Recognizes
    two ordered-program shapes at that scale:

    - **multiple ordered units**: more than one statement-level tile op, or a tile
      op grouped with one or more sibling side-effect statements (the device
      ``GEMM; store`` and ``GEMM; GEMM`` cases). Each unit becomes a stage,
      separated by the scale's default (stage) boundary;
    - **stage loop**: a tile op nested in a (non-scale) loop / control flow (the
      static ``T.serial`` device loop) -- modeled as a repeated stage.

    Returns the :class:`StageProgram` (whose ``requires_stage_boundary`` the
    caller checks to fail-close), or ``None`` if this is not an ordered program
    needing staging (so the normal single-op expansion path runs).
    """
    sem = resolve_scale_semantics(region.scale_name)
    if sem is None or not sem.supports_stage_boundary:
        return None
    boundary_barrier = sem.default_barrier
    if boundary_barrier is None:
        return None
    # in_kernel mirrors the StagePlanner contract: a scale that supports a stage
    # boundary but not an in-kernel barrier (device) needs a real kernel split.
    in_kernel = sem.supports_in_kernel_barrier
    boundary = StageBoundary(barrier=boundary_barrier, in_kernel=in_kernel)

    # Shape 1: a tile op wrapped in a loop / control flow -> stage loop.
    loop = _segment_loop_with_tile_op(segment)
    if loop is not None:
        return StageProgram(
            scale_name=region.scale_name,
            stages=[
                StagePlan(scale_name=region.scale_name, index=0, stmts=[loop],
                          effects=segment.effects, boundary_after=boundary),
                # The repeated-stage tail is left abstract in the skeleton.
                StagePlan(scale_name=region.scale_name, index=1, stmts=[],
                          effects=MemoryEffects.empty()),
            ],
            reason=f"static {region.scale_name} loop containing a tile op",
        )

    # Shape 2: multiple ordered effect units in one segment.
    n_tile_ops = len(segment.tile_op_names)
    has_other_effect = segment.op_name is None and any(
        _stmt_has_effect(s) and _tileop_name(s) is None for s in segment.stmts)
    n_units = n_tile_ops + (1 if has_other_effect else 0)
    if n_units > 1:
        # One stage per statement, each followed by the stage boundary except the
        # last. The skeleton keeps statements opaque; it only needs the ordering.
        units = [s for s in segment.stmts if _stmt_has_effect(s)]
        stages = []
        for i, s in enumerate(units):
            stages.append(StagePlan(
                scale_name=region.scale_name, index=i, stmts=[s],
                boundary_after=boundary if i + 1 < len(units) else None))
        return StageProgram(
            scale_name=region.scale_name,
            stages=stages,
            reason=(f"{region.scale_name} scope with {len(units)} ordered "
                    f"statements ({', '.join(segment.tile_op_names) or 'mixed'})"),
        )
    return None


def _dispatch_op_or_fail(region: ScaleRegion, segment: ScaleSegment) -> str:
    """Return the single managed tile-op name to dispatch, or fail-closed raise.

    Generic, op-agnostic gate over a surfaced expandable segment:

    - 0 tile ops (a raw side-effect segment interleaved with a child scale) ->
      raise (only template-managed tile ops may be expanded between scales);
    - >1 tile ops in one segment -> raise (multiple tile ops cannot be expanded);
    - exactly 1 tile op but the segment also has sibling statements
      (``op_name is None``) -> raise (a tile op with sibling side effects);
    - exactly 1 tile op as the whole segment -> return its op name for dispatch.

    The "multiple" / "sibling" rejects are what restore clean errors for a
    device GEMM grouped with a store or a second GEMM (instead of leaking to
    LowerScaleLaunch). Messages keep the ``PrepareDeviceScaleGemm``-compatible
    wording where the GEMM template owns it, but the generic layer is what
    decides a mixed segment is invalid -- no op-specific branch here.

    Ordered programs at a *stage-boundary* scale (``device``: GEMM+store,
    GEMM+GEMM, static loop) are recognized first and fail-closed with a
    staging-specific message (a StageProgram skeleton is built; multi-kernel
    lowering is future work).
    """
    # Ordered program at a stage-boundary scale -> recognize + fail-close with a
    # staging message (before the generic mixed/multi rejects), so the user sees
    # "ordered device program requiring a launch boundary" rather than a generic
    # "multiple tile ops".
    program = _plan_stage_program(region, segment)
    if program is not None and program.requires_stage_boundary:
        raise NotImplementedError(
            f"NormalizeScaleExpansion: recognized an ordered `{region.scale_name}` "
            f"scale program ({program.reason}) that requires a "
            f"`{program.stages[0].boundary_after.barrier.kind}` stage boundary "
            f"({program.num_stages} stages). Device stage boundary / multi-kernel "
            f"launch_boundary lowering is not implemented yet.")

    n = len(segment.tile_op_names)
    if n == 0:
        raise NotImplementedError(
            f"NormalizeScaleExpansion: a raw side-effect statement between a "
            f"`{region.scale_name}` scale and its child scale is not supported "
            f"yet. Only tile ops with a registered expansion template may be "
            f"expanded between scale levels.")
    if n > 1:
        raise NotImplementedError(
            f"NormalizeScaleExpansion: a `{region.scale_name}` scale with "
            f"multiple tile ops ({', '.join(segment.tile_op_names)}) in one "
            f"segment cannot be expanded; multiple scale-scoped tile ops in one "
            f"scope are not supported yet.")
    if segment.op_name is None:
        # Exactly one tile op, but grouped with sibling statements.
        raise NotImplementedError(
            f"NormalizeScaleExpansion: a `{region.scale_name}` scale tile op "
            f"`{segment.tile_op_names[0]}` with sibling statements (stores / "
            f"control flow / other ops) is not supported yet; place such "
            f"statements inside the innermost scale.")
    return segment.op_name


def find_expandable_segments(region: ScaleRegion) -> list:
    """Return ``(region, segment)`` pairs that require expansion, top-down.

    A segment "requires expansion" when either:

    - it contains at least one ``tl.tileop.*`` call whose scale is managed by an
      expansion template (``has_scale_expansion_templates``). This covers a clean
      single tile op *and* a mixed segment (tile op grouped with siblings, or
      more than one tile op) -- the latter is surfaced precisely so the planner
      can fail-close on it rather than silently leaking to ``LowerScaleLaunch``;
    - or it is a raw side-effect segment interleaved with a child scale region
      (planner rejects it).

    Note this no longer requires a child region or a single-op segment for a
    managed tile op: a device-scope GEMM (no child region) and a device GEMM with
    a sibling store (mixed segment) are both surfaced. A tile op nested in
    control flow (a static device loop) at a stage-boundary scale is also surfaced
    so the planner can recognize it as an ordered program and fail-close instead
    of leaking to ``LowerScaleLaunch``. Leaf-scale plain segments with no managed
    tile op are not expandable. Outermost-first ordering.
    """
    found: list = []

    def walk(reg: ScaleRegion):
        has_child_region = len(reg.child_regions()) > 0
        managed_scale = (not reg.is_root
                         and has_scale_expansion_templates(reg.scale_name))
        sem = resolve_scale_semantics(reg.scale_name) if not reg.is_root else None
        stage_scale = sem is not None and sem.supports_stage_boundary
        for item in reg.items:
            if isinstance(item, ScaleRegion):
                continue
            seg = item
            needs = False
            # Any managed tile op in the segment (clean or mixed) -> surface it;
            # the planner dispatches a clean single op and fail-closes the rest.
            if managed_scale and seg.tile_op_names:
                needs = True
            # A tile op nested in control flow at a stage-boundary scale (e.g. a
            # static device loop) -> surface it so the planner recognizes the
            # ordered program and fail-closes (else it leaks to LowerScaleLaunch).
            elif stage_scale and _segment_loop_with_tile_op(seg) is not None:
                needs = True
            # A raw side effect interleaved with a child scale -> planner rejects.
            elif seg.is_side_effect and has_child_region:
                needs = True
            if needs:
                found.append((reg, seg))
        for child in reg.child_regions():
            walk(child)

    walk(region)
    return found


def plan_region_expansions(func: PrimFunc, target=None) -> list:
    """Resolve an :class:`ExpansionPlan` for each expandable segment (fail-closed).

    Generic only: builds the region tree, finds expandable segments, resolves a
    template by ``(from_scale, op_name)``, and calls ``decode`` / ``validate`` /
    ``plan``. Raises ``NotImplementedError`` for any expandable segment with no
    registered template, or for a raw side-effect segment between scales. Returns
    the list of plans (empty when there is nothing to expand). Performs no rewrite.
    """
    ensure_default_scale_expansion_templates_registered()
    root = build_region_tree(func)
    expandable = find_expandable_segments(root)
    plans: list = []
    for region, segment in expandable:
        op_name = _dispatch_op_or_fail(region, segment)
        template = resolve_scale_expansion_template(region.scale_name, op_name)
        if template is None:
            raise NotImplementedError(
                f"NormalizeScaleExpansion: no expansion template registered for "
                f"the scale/op edge `({region.scale_name}, {op_name})`. "
                f"Top-down expansion of this scale edge is not supported yet.")
        context = ExpansionContext(region=region, func=func, target=target)
        info = template.decode(segment, context)
        template.validate(info, context)
        plans.append(template.plan(info, context))
    return plans


# ---------------------------------------------------------------------------
# Generic rewrite: splice plans into compatible child regions.
# ---------------------------------------------------------------------------


def _current_target():
    """Return the active TVM target if one is set, else None (generic)."""
    try:
        from tvm.target import Target
        return Target.current(allow_none=True)
    except Exception:  # pragma: no cover - defensive
        return None


def _barrier_to_stmt(spec: BarrierSpec) -> tir.Stmt:
    """Lower a scale-parametric :class:`BarrierSpec` to a concrete sync statement.

    Routes through :class:`StagePlanner` (which consults
    :class:`ScaleSemantics`) to decide whether the barrier is an in-kernel sync
    or a stage boundary:

    - in-kernel block sync (``scope="block"``, ``kind="sync_threads"``) lowers to
      ``tir.tvm_storage_sync(<memory_scope>)`` -- the same intrinsic
      ``T.sync_threads`` emits (``__syncthreads`` on CUDA);
    - a stage boundary (``scope="device"``, ``kind="launch_boundary"``) is
      *designed but not executable yet*: it loud-errors here. Multi-kernel
      lowering is a later milestone.

    Any other kind also loud-errors rather than silently dropping a barrier.
    """
    from tilelang.tileop.scale_barrier_planner import StagePlanner

    decision = StagePlanner().plan_stage(spec)
    if decision.stage_boundary:
        raise NotImplementedError(
            f"NormalizeScaleExpansion: a `{spec.scope}` stage boundary "
            f"(`{spec.kind}`) is not implemented yet -- device stage boundary / "
            f"multi-kernel lowering is designed but not executable. Ordered "
            f"device-scope segments are not supported yet.")
    if decision.in_kernel and spec.kind == "sync_threads":
        mem = spec.memory_scope or "shared"
        return tir.Evaluate(tir.call_intrin("int32", "tirx.tvm_storage_sync", mem))
    raise NotImplementedError(
        f"NormalizeScaleExpansion: barrier kind `{spec.kind}` at scope "
        f"`{spec.scope}` is not supported by the generic barrier emitter yet.")


def _workgroups_compatible(plan: ExpansionPlan, child: ScaleRegion) -> bool:
    """Conservative compatibility: identical static workgroup, or plan defers.

    The first implementation only reuses a child whose workgroup the plan does
    not constrain (``required_child_workgroup is None``) or matches exactly
    (structural equality of the extent list).
    """
    if plan.required_child_workgroup is None:
        return True
    req = tuple(plan.required_child_workgroup)
    have = tuple(child.workgroup)
    if len(req) != len(have):
        return False
    return all(tir.analysis.expr_deep_equal(a, b) for a, b in zip(req, have))


def _find_compatible_child(region: ScaleRegion, segment: ScaleSegment,
                           plan: ExpansionPlan) -> ScaleRegion:
    """Return the unique compatible child region for ``plan``, else loud-error.

    Conservative merge rule (milestone 6): the single immediately-following child
    region of scale ``plan.to_scale`` with a compatible workgroup. Multiple
    sibling children of that scale, or no following child, are rejected.
    """
    # Children of scale to_scale that appear after this segment in lexical order.
    seen_segment = False
    following: list = []
    for item in region.items:
        if item is segment:
            seen_segment = True
            continue
        if seen_segment and isinstance(item, ScaleRegion) \
                and item.scale_name == plan.to_scale:
            following.append(item)

    to_scale_children = [c for c in region.child_regions()
                         if c.scale_name == plan.to_scale]
    if len(to_scale_children) != 1:
        raise NotImplementedError(
            f"NormalizeScaleExpansion: expanding `{segment.op_name}` from "
            f"`{region.scale_name}` requires exactly one `{plan.to_scale}` child "
            f"to merge into; found {len(to_scale_children)} (multiple sibling "
            f"child scales are not supported yet).")
    child = to_scale_children[0]
    if all(child is not f for f in following):
        raise NotImplementedError(
            f"NormalizeScaleExpansion: the `{plan.to_scale}` child must follow "
            f"the expanded `{segment.op_name}` segment in lexical order.")
    if not _workgroups_compatible(plan, child):
        raise NotImplementedError(
            f"NormalizeScaleExpansion: the `{plan.to_scale}` child workgroup is "
            f"not compatible with the expansion plan's required workgroup.")
    return child


def _aggregate_read_effects(region: ScaleRegion) -> MemoryEffects:
    """Union the read effects of every segment in ``region`` (recursively).

    The consumer side of a producer/consumer dependency: which memory spaces the
    child region (and its descendants) read. Coarse and op-agnostic -- it relies
    on the per-segment :class:`MemoryEffects` the parser already computed.
    """
    acc = MemoryEffects()
    for item in region.items:
        if isinstance(item, ScaleRegion):
            acc = acc.merged_with(_aggregate_read_effects(item))
        else:
            acc = acc.merged_with(item.effects)
    return acc


def _derive_after_barriers(plan: ExpansionPlan, parent: ScaleRegion,
                           child: ScaleRegion) -> list:
    """Derive the barriers to insert after the relocated producer (option B).

    Generic, scale-parametric: run :class:`ScaleDependencyAnalysis` over the
    plan's producer effects and the child region's read effects at the *parent*
    scale, then ask the :class:`BarrierPlanner` for each dependency's barrier.
    This is the path that turns "block copy writes shared; thread reads shared"
    into a block ``sync_threads`` without the template hard-coding it.

    The template may also declare explicit ``plan.barriers_after`` (e.g. a
    barrier with no detectable consumer dependency); those are unioned in. Result
    is de-duplicated on (scope, kind, memory_scope).
    """
    consumer = _aggregate_read_effects(child)
    deps = ScaleDependencyAnalysis().required_dependencies(
        parent.scale_name, plan.effects, consumer)
    planner = BarrierPlanner()
    specs = [planner.plan_barrier(d) for d in deps]
    specs.extend(plan.barriers_after)
    # De-duplicate preserving order.
    seen = set()
    out = []
    for s in specs:
        key = (s.scope, s.kind, s.memory_scope)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _resolve_rewrites(func: PrimFunc, expandable: list, target=None):
    """Resolve and validate the expansion(s) for the expandable segments.

    Returns one of:

    - ``("replace_func", generated_func)`` -- a whole-function (replace_func)
      plan; the caller swaps the entire PrimFunc. Such a plan must be the only
      expansion in the function.
    - a list of ``(parent_loop, child_loop, lowered, prefix, suffix, removed)``
      merge tuples, keyed by the original scale ``For`` node identities so the
      splice pass can locate them.

    Fail-closed: any unsupported edge raises here, before any IR is rebuilt.
    """
    rewrites: list = []
    for region, segment in expandable:
        op_name = _dispatch_op_or_fail(region, segment)
        template = resolve_scale_expansion_template(region.scale_name, op_name)
        if template is None:
            raise NotImplementedError(
                f"NormalizeScaleExpansion: no expansion template registered for "
                f"the scale/op edge `({region.scale_name}, {op_name})`. "
                f"Top-down expansion of this scale edge is not supported yet.")
        context = ExpansionContext(region=region, func=func, target=target)
        # The template claims (from_scale, op_name); match() is its finer-grained
        # predicate for whether it actually handles *this* segment. A registered
        # template that declines the segment is a fail-closed case, not a silent
        # skip.
        if not template.match(segment, context):
            raise NotImplementedError(
                f"NormalizeScaleExpansion: the registered expansion template for "
                f"`({region.scale_name}, {segment.op_name})` did not match this "
                f"segment; this form is not supported yet.")
        info = template.decode(segment, context)
        template.validate(info, context)
        plan = template.plan(info, context)

        # replace_func plan: the whole enclosing PrimFunc is replaced by a
        # generated lower-scale kernel (e.g. a device-scope GEMM -> generated
        # device/block/thread func). This is whole-function, so it cannot coexist
        # with other expansions in the same function.
        if plan.kind == "replace_func":
            if len(expandable) != 1:
                raise NotImplementedError(
                    "NormalizeScaleExpansion: a whole-function (replace_func) "
                    "expansion cannot coexist with other expandable segments in "
                    "the same function; this is not supported yet.")
            return ("replace_func", plan.replacement_func)

        if plan.to_scale == region.scale_name:
            raise NotImplementedError(
                "NormalizeScaleExpansion: an expansion plan must target a "
                "different (smaller) scale than its parent.")
        child = _find_compatible_child(region, segment, plan)
        prefix = [_barrier_to_stmt(b) for b in plan.barriers_before]
        # Derive after-barriers from producer/consumer effects via the generic
        # BarrierPlanner (scale-parametric), unioned with any the plan declared.
        after_specs = _derive_after_barriers(plan, region, child)
        suffix = [_barrier_to_stmt(b) for b in after_specs]
        # Retag every relocated tile op to the destination child scale: the op
        # now executes in that scale, so its tl.scale_ctx.name / .path must match.
        # Generic -- applies to any tl.tileop.*, not just copy.
        lowered = [_retag_tileop_to_scale(s, child.scale_name, tuple(child.path))
                   for s in plan.lowered_stmts]
        rewrites.append((
            region.loop, child.loop,
            lowered, prefix, suffix,
            list(segment.stmts),
        ))
    return rewrites


def _prepend_into_child_body(child_loop: tir.For, new_prefix: list) -> tir.For:
    """Return ``child_loop`` with ``new_prefix`` prepended inside its root block.

    The child scale For body is the root ``BlockRealize`` of the child region;
    splice the prefix statements inside that block's body so they execute within
    the child scale's launch. Falls back to wrapping the body in a SeqStmt if it
    is not a BlockRealize. Scale-agnostic -- works for any parent -> child edge,
    not just block -> thread.
    """
    body = child_loop.body
    if isinstance(body, tir.SBlockRealize):
        block = body.block
        new_inner = tir.SeqStmt(list(new_prefix) + [block.body])
        new_block = tir.SBlock(
            block.iter_vars, block.reads, block.writes, block.name_hint,
            new_inner, block.init, block.alloc_buffers, block.match_buffers,
            block.annotations)
        new_body = tir.SBlockRealize(body.iter_values, body.predicate, new_block)
    else:
        new_body = tir.SeqStmt(list(new_prefix) + [body])
    return tir.For(child_loop.loop_var, child_loop.min, child_loop.extent,
                   child_loop.kind, new_body,
                   thread_binding=child_loop.thread_binding,
                   annotations=child_loop.annotations)


def _apply_rewrites(func: PrimFunc, rewrites: list) -> PrimFunc:
    """Splice every resolved rewrite into the function body, op-agnostically.

    For each ``(parent_loop, child_loop, lowered, prefix, suffix, removed)``:
    rebuild the parent scale loop so the ``removed`` parent-segment statements are
    dropped and the child scale loop is replaced by one whose root block body is
    prefixed with ``prefix + lowered + suffix`` (the plan's lowered statements and
    barriers).

    Matching uses ``ObjectRef.same_as`` (reference equality on the underlying TIR
    node) against the nodes captured during region parsing. We do our own
    recursive rebuild rather than ``ir_transform`` because ``ir_transform`` hands
    the callback freshly-rebuilt nodes, so neither python ``id`` nor ``same_as``
    would match the originally-parsed loop handles.
    """
    by_parent = list(rewrites)  # (parent_loop, child_loop, lowered, pre, suf, removed)

    def find_rewrite(node):
        for rw in by_parent:
            if node.same_as(rw[0]):
                return rw
        return None

    def rewrite_parent(parent_node, rw):
        _parent_loop, child_loop, lowered, prefix, suffix, removed = rw
        new_child = _prepend_into_child_body(
            child_loop, list(prefix) + list(lowered) + list(suffix))

        def rebuild(stmt):
            if isinstance(stmt, tir.SeqStmt):
                out = []
                for s in stmt.seq:
                    r = rebuild(s)
                    if r is not None:
                        out.append(r)
                if not out:
                    return tir.Evaluate(tir.const(0))
                return out[0] if len(out) == 1 else tir.SeqStmt(out)
            if isinstance(stmt, tir.SBlockRealize):
                blk = stmt.block
                new_block = tir.SBlock(
                    blk.iter_vars, blk.reads, blk.writes, blk.name_hint,
                    rebuild(blk.body), blk.init, blk.alloc_buffers,
                    blk.match_buffers, blk.annotations)
                return tir.SBlockRealize(stmt.iter_values, stmt.predicate, new_block)
            if stmt.same_as(child_loop):
                return new_child
            if any(stmt.same_as(s) for s in removed):
                return None  # drop the expanded parent-segment statement
            return stmt

        new_body = rebuild(parent_node.body)
        return tir.For(parent_node.loop_var, parent_node.min, parent_node.extent,
                       parent_node.kind, new_body,
                       thread_binding=parent_node.thread_binding,
                       annotations=parent_node.annotations)

    def walk(stmt):
        if isinstance(stmt, tir.For):
            rw = find_rewrite(stmt)
            if rw is not None:
                return rewrite_parent(stmt, rw)
            return tir.For(stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
                           walk(stmt.body), thread_binding=stmt.thread_binding,
                           annotations=stmt.annotations)
        if isinstance(stmt, tir.SeqStmt):
            return tir.SeqStmt([walk(s) for s in stmt.seq])
        if isinstance(stmt, tir.SBlockRealize):
            blk = stmt.block
            new_block = tir.SBlock(
                blk.iter_vars, blk.reads, blk.writes, blk.name_hint,
                walk(blk.body), blk.init, blk.alloc_buffers, blk.match_buffers,
                blk.annotations)
            return tir.SBlockRealize(stmt.iter_values, stmt.predicate, new_block)
        if isinstance(stmt, tir.IfThenElse):
            return tir.IfThenElse(
                stmt.condition, walk(stmt.then_case),
                walk(stmt.else_case) if stmt.else_case is not None else None)
        if isinstance(stmt, tir.AttrStmt):
            return tir.AttrStmt(stmt.node, stmt.attr_key, stmt.value,
                                walk(stmt.body))
        # tirx Bind / AllocBuffer / DeclBuffer are body-less scope statements
        # (visible to subsequent statements in the enclosing SeqStmt), so they
        # are leaves for this walk and fall through unchanged.
        return stmt

    return func.with_body(walk(func.body))


# ---------------------------------------------------------------------------
# Pass.
# ---------------------------------------------------------------------------


def NormalizeScaleExpansion():
    """Top-down scale-expansion normalizer pass.

    Generic, op-agnostic driver. For each expandable parent segment it resolves a
    :class:`ScaleExpansionTemplate` by ``(from_scale, op_name)``, asks it for an
    :class:`ExpansionPlan`, and then -- knowing nothing about the op -- merges
    ``plan.lowered_stmts`` (plus any required barriers) into the unique compatible
    child region of scale ``plan.to_scale`` and deletes the parent segment. It
    loud-errors (``NotImplementedError``) for any segment with no registered
    template, any raw side-effect segment between scales, or any merge that
    violates the conservative compatibility rule. A clean direct scale tree (e.g.
    the device GEMM template output, or a hand-written block/thread kernel) has no
    expandable segments and is returned unchanged.

    Runs after ``PrepareScaleTileOps`` and before ``LowerScaleLaunch``.
    """

    def pass_fn(func: PrimFunc, mod, ctx):
        ensure_default_scale_expansion_templates_registered()
        root = build_region_tree(func)
        if not root.child_regions():
            return func  # no scale regions -> nothing to do

        expandable = find_expandable_segments(root)
        if not expandable:
            return func  # clean direct tree (device GEMM output, etc.)

        # Resolve the expansion(s) (fail-closed). This returns either a
        # whole-function replacement or a list of merge rewrites.
        resolved = _resolve_rewrites(func, expandable, target=_current_target())

        # replace_func: the whole PrimFunc is replaced by a generated lower-scale
        # kernel (the generated func already carries the original attrs /
        # global_symbol via the template builder).
        if isinstance(resolved, tuple) and resolved[0] == "replace_func":
            return resolved[1]

        # Otherwise, apply the merge rewrites generically: splice each plan into
        # its compatible child region and drop the parent segment.
        return _apply_rewrites(func, resolved)

    return prim_func_pass(pass_fn, opt_level=0)


__all__ = [
    "build_region_tree",
    "find_expandable_segments",
    "plan_region_expansions",
    "NormalizeScaleExpansion",
    "has_scale_expansion_templates",
]
