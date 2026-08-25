"""Block-scope copy expansion template (Top-Down Scale Expansion, milestone 6).

The op-specific half of the ``block -> thread`` cooperative-copy expansion. It
claims the ``("block", "tl.tileop.copy")`` edge in the
:mod:`tilelang.tileop.scale_expansion` registry and is driven by the generic
normalizer (:mod:`tilelang.transform.normalize_scale_expansion`): the normalizer
parses the region tree, finds the block-scope copy segment, resolves this
template, and asks it to ``decode`` / ``validate`` / ``plan``. The template
returns a generic :class:`ExpansionPlan`; the normalizer performs the
op-agnostic merge into the thread child and inserts the barrier.

All block/copy/thread-specific knowledge lives here. The generic normalizer
never learns that the op is a copy -- it only consumes the plan.

MVP boundary (validated in :meth:`validate`):

- the segment is exactly one ``Evaluate(Call(tl.tileop.copy, ...))``;
- the enclosing region is an *outermost* ``block`` scale (no cluster/device
  ancestor) -- enforced by the normalizer's region path, re-checked here;
- there is exactly one immediately-following ``thread`` child region to merge
  into (the normalizer owns the compatibility / uniqueness check).

The copy's src/dst storage scopes are decoded to report accurate memory effects
(milestone 14): any direction is accepted (global->shared, shared->shared,
shared->global, global->global), and the destination scope decides barriers --
a copy into shared yields a CTA sync before a thread shared read; a copy into
global yields none (program-order at block scale). The template hard-codes no
barrier; the generic BarrierPlanner derives it from the reported effects.
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

_COPY_OP_NAME = "tl.tileop.copy"
_REGION_OP_NAME = "tl.tileop.region"
_FROM_SCALE = "block"
_TO_SCALE = "thread"


@dataclass
class BlockCopyInfo:
    """Decoded block-scope copy segment (the single copy Evaluate + effects)."""

    copy_evaluate: tir.Evaluate
    effects: MemoryEffects


def _copy_call(stmt) -> tir.Call:
    """Return the ``tl.tileop.copy`` Call inside an Evaluate, or None."""
    if isinstance(stmt, tir.Evaluate):
        val = stmt.value
        if (isinstance(val, tir.Call) and isinstance(val.op, tir.op.Op)
                and val.op.name == _COPY_OP_NAME):
            return val
    return None


def _region_buffer(region_call):
    """Recover the ``tir.Buffer`` from a tl.tileop.region call (arg0 BufferLoad)."""
    if not (isinstance(region_call, tir.Call) and isinstance(region_call.op, tir.op.Op)
            and region_call.op.name == _REGION_OP_NAME):
        return None
    inner = region_call.args[0]
    if isinstance(inner, tir.BufferLoad):
        return inner.buffer
    return None


def _copy_effects(call: tir.Call) -> MemoryEffects:
    """Report accurate read/write effects from the copy's src/dst storage scopes.

    ``tl.tileop.copy(src_region, dst_region, ...)``: arg0 is read, arg1 is
    written. The destination scope is what decides barrier needs -- a copy whose
    destination is shared (global->shared, shared->shared) reports ``writes_shared``
    so the generic BarrierPlanner derives a CTA sync when a thread consumer reads
    shared; a copy into global (shared->global, global->global) reports
    ``writes_global`` (program-order-only at block scale -> no spurious block sync).
    This replaces the previous hard-coded global->shared assumption.
    """
    eff = MemoryEffects()
    src = _region_buffer(call.args[0]) if len(call.args) >= 1 else None
    dst = _region_buffer(call.args[1]) if len(call.args) >= 2 else None
    if src is not None:
        eff = eff.merged_with(memory_effect_read_for_scope(src.scope()))
    if dst is not None:
        eff = eff.merged_with(memory_effect_write_for_scope(dst.scope()))
    return eff


class BlockCopyExpansionTemplate(ScaleExpansionTemplate):
    """Expand a block-scope cooperative ``T.copy`` into the thread child scale.

    The copy is left intact as a tile op (``tl.tileop.copy`` lowers
    CTA-cooperatively in ``LowerTileOp``); "expansion" here means relocating it
    into the thread region so it executes inside the thread launch and a CTA
    barrier separates it from later thread reads of shared memory. The actual
    relocation + barrier insertion is performed generically by the normalizer
    from the returned :class:`ExpansionPlan`.
    """

    @property
    def from_scale(self) -> str:
        return _FROM_SCALE

    @property
    def to_scale(self) -> str:
        return _TO_SCALE

    @property
    def op_names(self) -> tuple[str, ...]:
        return (_COPY_OP_NAME,)

    def match(self, segment: ScaleSegment, context: ExpansionContext) -> bool:
        return (segment.scale_name == _FROM_SCALE
                and segment.op_name == _COPY_OP_NAME
                and len(segment.stmts) == 1
                and _copy_call(segment.stmts[0]) is not None)

    def decode(self, segment: ScaleSegment, context: ExpansionContext) -> BlockCopyInfo:
        if len(segment.stmts) != 1:
            raise NotImplementedError(
                "BlockCopyExpansion: only a single block-scope T.copy statement "
                "may be expanded into the thread scale (a multi-statement "
                "block-scope segment is not supported yet).")
        stmt = segment.stmts[0]
        call = _copy_call(stmt)
        if call is None:
            raise NotImplementedError(
                "BlockCopyExpansion: the block-scope segment is not a single "
                "tl.tileop.copy and cannot be expanded.")
        return BlockCopyInfo(copy_evaluate=stmt, effects=_copy_effects(call))

    def validate(self, info: BlockCopyInfo, context: ExpansionContext) -> None:
        region = context.region
        # MVP: the copy must live in an *outermost* block scale -- no cluster /
        # device ancestor. The region path is (..., "block"); anything before
        # "block" means a nesting we do not support yet (cluster -> block -> copy).
        path = tuple(region.path)
        if path != (_FROM_SCALE,):
            raise NotImplementedError(
                f"BlockCopyExpansion: a block-scope copy is only supported under "
                f"an outermost `block` scale (path == ('block',)); got path "
                f"{path}. cluster -> block -> T.copy -> thread is not supported yet.")
        # There must be exactly one thread child region to merge into. The
        # generic normalizer owns the strict compatibility/uniqueness check; this
        # is an early, op-meaningful error.
        thread_children = [c for c in region.child_regions()
                           if c.scale_name == _TO_SCALE]
        if len(thread_children) != 1:
            raise NotImplementedError(
                f"BlockCopyExpansion: a block-scope copy requires exactly one "
                f"inner `thread` scale to expand into; found "
                f"{len(thread_children)}.")

    def plan(self, info: BlockCopyInfo, context: ExpansionContext) -> ExpansionPlan:
        # Report the decoded src/dst effects; the generic BarrierPlanner derives
        # the CTA barrier from the producer (e.g. writes_shared) / consumer (e.g.
        # thread reads_shared) pair. The template does NOT hard-code sync_threads
        # (option B): a copy into shared gets a sync before a thread shared read; a
        # copy into global gets none. A template could still declare an explicit
        # barrier for a dependency the generic analysis cannot see.
        return ExpansionPlan(
            from_scale=_FROM_SCALE,
            to_scale=_TO_SCALE,
            required_child_workgroup=None,  # reuse the existing thread child's
            lowered_stmts=[info.copy_evaluate],
            barriers_before=[],
            barriers_after=[],
            effects=info.effects,
        )


register_scale_expansion_template(BlockCopyExpansionTemplate())
