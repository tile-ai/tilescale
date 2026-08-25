"""Top-down scale expansion abstractions (Long-Term Plan skeleton).

This module defines the *generic*, op-agnostic data structures and template
registry for the top-down scale expansion pipeline described in
``docs/compiler_internals/scale_programming_plan.md`` ("Long-Term Plan:
Top-Down Scale Expansion"). The guiding rule is:

> Lower larger scales into smaller scales one edge at a time.

The generic normalizer (``tilelang.transform.normalize_scale_expansion``) parses
a PrimFunc body into a :class:`ScaleRegion` tree, identifies the
:class:`ScaleSegment`\\ s that need expansion, resolves a
:class:`ScaleExpansionTemplate` by ``(from_scale, op_name)``, and asks it for a
generic :class:`ExpansionPlan`. The normalizer owns region parsing, statement
ordering, compatibility checks, barrier placement, and fail-closed behavior --
it must never grow GEMM-, copy-, reduce-, or elementwise-specific decode logic.

Hard boundary (enforced by code review, not types):

- generic code may know about scale order, lexical regions, statement ordering,
  effects, dependency classes, and template registration;
- generic code may NOT know that ``tl.tileop.copy`` should move into a ``thread``
  body, or that ``tl.tileop.gemm`` uses ``BM=128``;
- template code may be highly specialized for one ``(from_scale, op_name)`` edge,
  as long as it returns a generic :class:`ExpansionPlan`.

This is the milestone-1..4 skeleton: data structures + registry + a read-only
parser and a fail-closed planner. No template performs an IR rewrite yet, so the
existing strict ``LowerScaleLaunch`` still rejects ``block -> T.copy -> thread``
and friends. The registry is intentionally separate from the legacy
``scale_template.ScaleTileOpTemplate`` whole-function registry; the GEMM device
template will be migrated behind this interface in a later milestone.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional, Union

from tvm import tirx as tir


# ---------------------------------------------------------------------------
# Memory / synchronization effect description (generic, op-agnostic).
# ---------------------------------------------------------------------------


@dataclass
class MemoryEffects:
    """Coarse memory effects of a segment or expansion plan.

    Op-agnostic: it records *which memory spaces* a chunk of work reads or
    writes, not what operation produced them. The normalizer uses this to decide
    barrier placement and whether values may cross generated child segments.
    """

    reads_global: bool = False
    writes_global: bool = False
    reads_shared: bool = False
    writes_shared: bool = False
    reads_local: bool = False
    writes_local: bool = False

    @staticmethod
    def empty() -> "MemoryEffects":
        return MemoryEffects()

    def merged_with(self, other: "MemoryEffects") -> "MemoryEffects":
        return MemoryEffects(
            reads_global=self.reads_global or other.reads_global,
            writes_global=self.writes_global or other.writes_global,
            reads_shared=self.reads_shared or other.reads_shared,
            writes_shared=self.writes_shared or other.writes_shared,
            reads_local=self.reads_local or other.reads_local,
            writes_local=self.writes_local or other.writes_local,
        )


def classify_storage_scope(scope: Optional[str]) -> str:
    """Classify a TIR buffer storage scope into a coarse memory class.

    Returns one of ``"shared"`` / ``"local"`` / ``"global"`` -- the three classes
    :class:`MemoryEffects` distinguishes. Shared-memory scopes (``"shared"``,
    ``"shared.dyn"``) -> ``"shared"``; thread-owned scopes (``"local"``,
    ``"local.fragment"``, ``"fragment"``, ``"warp"``) -> ``"local"``; everything
    else, including ``"global"``, the empty string, ``None``, and any unrecognized
    scope -> ``"global"`` (sync-conservative: ``global`` has no in-kernel ordering
    mode, so an unknown scope never fabricates a spurious block barrier).
    """
    if not scope:
        return "global"
    if "shared" in scope:
        return "shared"
    if "local" in scope or "fragment" in scope or scope == "warp":
        return "local"
    return "global"


def memory_effect_read_for_scope(scope: Optional[str]) -> MemoryEffects:
    """A :class:`MemoryEffects` recording a *read* of a buffer in ``scope``."""
    cls = classify_storage_scope(scope)
    eff = MemoryEffects()
    if cls == "shared":
        eff.reads_shared = True
    elif cls == "local":
        eff.reads_local = True
    else:
        eff.reads_global = True
    return eff


def memory_effect_write_for_scope(scope: Optional[str]) -> MemoryEffects:
    """A :class:`MemoryEffects` recording a *write* of a buffer in ``scope``."""
    cls = classify_storage_scope(scope)
    eff = MemoryEffects()
    if cls == "shared":
        eff.writes_shared = True
    elif cls == "local":
        eff.writes_local = True
    else:
        eff.writes_global = True
    return eff


@dataclass
class BarrierSpec:
    """A required synchronization point, parameterized by scale.

    ``scope`` is the *execution* scale the barrier synchronizes -- ``"block"``
    (CTA-wide), ``"cluster"``, ``"device"`` (full grid), ``"node"``. ``kind`` is
    the physical realization on that scale -- ``"sync_threads"`` (in-kernel CTA
    barrier), ``"cluster_sync"``, ``"launch_boundary"`` (split kernels around a
    full-grid ordering), ``"runtime_barrier"``. ``memory_scope`` optionally names
    the memory space the barrier orders (e.g. ``"shared"`` / ``"global"``); it is
    a hint for lowering, not a second execution scope.

    The :class:`~tilelang.tileop.scale_semantics.ScaleSemantics` registry maps a
    scale to its default ``BarrierSpec``; the normalizer lowers a ``BarrierSpec``
    to the concrete sync (block sync -> ``tvm_storage_sync``; device launch
    boundary -> stage split, skeleton only for now). Templates declare *that* a
    barrier is required and at *what scope*; they do not emit IR.
    """

    scope: str = "block"
    kind: str = "sync_threads"
    memory_scope: Optional[str] = None

    @staticmethod
    def block_sync(memory_scope: str = "shared") -> "BarrierSpec":
        """Convenience: a CTA-wide in-kernel barrier ordering ``memory_scope``."""
        return BarrierSpec(scope="block", kind="sync_threads",
                           memory_scope=memory_scope)

    @staticmethod
    def device_launch_boundary() -> "BarrierSpec":
        """Convenience: a full-grid ordering realized as a kernel launch boundary."""
        return BarrierSpec(scope="device", kind="launch_boundary",
                           memory_scope="global")


# ---------------------------------------------------------------------------
# Region / segment tree (read-only parse output).
# ---------------------------------------------------------------------------


@dataclass
class ScaleSegment:
    """An ordered run of statements that all belong to one scale level.

    A segment is a maximal sequence of sibling statements at ``scale_name`` that
    are *not* themselves child scale regions. ``op_name`` is the dominant tile-op
    name when the segment is a single ``tl.tileop.*`` evaluate (used for template
    dispatch); it is ``None`` for raw side-effect segments (e.g. a bare
    ``BufferStore``) or for *mixed* segments (a tile op grouped with siblings, or
    more than one tile op), which the planner treats as fail-closed candidates.

    ``tile_op_names`` lists *every* ``tl.tileop.*`` call in the segment, in order
    (so the planner can detect "contains a managed tile op but not as a single
    clean op" and fail-close generically, without re-walking the statements).
    """

    scale_name: str
    path: tuple[str, ...]
    stmts: list[tir.Stmt]
    op_name: Optional[str] = None
    tile_op_names: tuple[str, ...] = ()
    is_side_effect: bool = False
    effects: MemoryEffects = field(default_factory=MemoryEffects.empty)


@dataclass
class ScaleRegion:
    """A ``T.scale`` region and its ordered children (segments + child regions).

    ``loop`` is the frontend scale ``For`` loop (annotated ``tl.scale``).
    ``items`` preserves lexical order, interleaving :class:`ScaleSegment` and
    nested :class:`ScaleRegion`. The root of a parsed function is represented as a
    region with ``scale_name == "<root>"`` and ``loop is None``.
    """

    scale_name: str
    path: tuple[str, ...]
    loop: Optional[tir.For] = None
    workgroup: tuple[tir.PrimExpr, ...] = ()
    items: list[Union["ScaleSegment", "ScaleRegion"]] = field(default_factory=list)

    @property
    def is_root(self) -> bool:
        return self.loop is None

    def child_regions(self) -> list["ScaleRegion"]:
        return [it for it in self.items if isinstance(it, ScaleRegion)]

    def segments(self) -> list[ScaleSegment]:
        return [it for it in self.items if isinstance(it, ScaleSegment)]


# ---------------------------------------------------------------------------
# Expansion plan (generic output of a template).
# ---------------------------------------------------------------------------


@dataclass
class ExpansionPlan:
    """The generic result of expanding one parent segment.

    Two plan shapes, both consumed op-agnostically by the normalizer:

    - **merge** (the default): ``lowered_stmts`` are spliced into a compatible
      child region of scale ``to_scale``, with ``barriers_before`` /
      ``barriers_after`` inserted and ``effects`` recorded for dependency
      reasoning. Used by e.g. the block-copy template.
    - **replace_func**: when ``replacement_func`` is set, the whole enclosing
      PrimFunc is replaced by it. Used when a template expands a scope into a
      fully generated lower-scale kernel (e.g. a device-scope GEMM expands into a
      generated ``device -> block -> thread`` PrimFunc). The normalizer performs
      the swap generically -- it does not know what op produced the function.

    ``kind`` is derived: ``"replace_func"`` when ``replacement_func`` is set, else
    ``"merge"``.
    """

    from_scale: str
    to_scale: str
    required_child_workgroup: Optional[tuple[tir.PrimExpr, ...]] = None
    lowered_stmts: list[tir.Stmt] = field(default_factory=list)
    barriers_before: list[BarrierSpec] = field(default_factory=list)
    barriers_after: list[BarrierSpec] = field(default_factory=list)
    effects: MemoryEffects = field(default_factory=MemoryEffects.empty)
    # When set, the whole enclosing PrimFunc is replaced by this generated func
    # (a generic "expand this scope into a lower-scale kernel" plan kind).
    replacement_func: Optional[tir.PrimFunc] = None

    @property
    def kind(self) -> str:
        return "replace_func" if self.replacement_func is not None else "merge"


# ---------------------------------------------------------------------------
# Expansion context + template interface.
# ---------------------------------------------------------------------------


@dataclass
class ExpansionContext:
    """Context handed to a template during decode / validate / plan.

    Carries the enclosing :class:`ScaleRegion` (the parent being expanded), the
    function under expansion, and the lowering target. Kept deliberately small;
    templates pull what they need from the segment and this context.
    """

    region: ScaleRegion
    func: tir.PrimFunc
    target: object = None


class ScaleExpansionTemplate(abc.ABC):
    """Expand one ``(from_scale, op_name)`` segment into ``to_scale`` IR.

    The extension point of the top-down pipeline. A template claims one or more
    ``op_names`` on a single ``from_scale -> to_scale`` edge and returns a generic
    :class:`ExpansionPlan`. All op-specific knowledge lives here; the normalizer
    only calls :meth:`match` / :meth:`decode` / :meth:`validate` / :meth:`plan`.
    """

    @property
    @abc.abstractmethod
    def from_scale(self) -> str:
        """The scale this template expands *from* (e.g. ``"device"``/``"block"``)."""

    @property
    @abc.abstractmethod
    def to_scale(self) -> str:
        """The next-smaller scale this template expands *to*."""

    @property
    @abc.abstractmethod
    def op_names(self) -> tuple[str, ...]:
        """The ``tl.tileop.*`` names this template claims."""

    @abc.abstractmethod
    def match(self, segment: ScaleSegment, context: ExpansionContext) -> bool:
        """Cheap predicate: does this template apply to ``segment`` here?"""

    @abc.abstractmethod
    def decode(self, segment: ScaleSegment, context: ExpansionContext):
        """Decode the segment into an op-specific info object (or raise)."""

    @abc.abstractmethod
    def validate(self, info, context: ExpansionContext) -> None:
        """Validate the decoded info against this edge's constraints (raise on fail)."""

    @abc.abstractmethod
    def plan(self, info, context: ExpansionContext) -> ExpansionPlan:
        """Return the generic :class:`ExpansionPlan` for this segment."""


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------


_REGISTRY: list[ScaleExpansionTemplate] = []


def register_scale_expansion_template(template: ScaleExpansionTemplate) -> None:
    """Register an expansion template (idempotent by template type)."""
    for idx, existing in enumerate(_REGISTRY):
        if type(existing) is type(template):
            _REGISTRY[idx] = template
            return
    _REGISTRY.append(template)


def resolve_scale_expansion_template(
        from_scale: str, op_name: str) -> Optional[ScaleExpansionTemplate]:
    """Return the template claiming ``(from_scale, op_name)``, or ``None``."""
    for template in _REGISTRY:
        if from_scale == template.from_scale and op_name in template.op_names:
            return template
    return None


def has_scale_expansion_templates(from_scale: str) -> bool:
    """True if any registered template expands ``from_scale`` (any op)."""
    return any(from_scale == t.from_scale for t in _REGISTRY)


def registered_scale_expansion_templates() -> tuple[ScaleExpansionTemplate, ...]:
    """Snapshot of registered templates (for tests / introspection)."""
    return tuple(_REGISTRY)


_DEFAULTS_REGISTERED = False


def ensure_default_scale_expansion_templates_registered() -> None:
    """Lazily register the built-in expansion templates (idempotent).

    Imports the template modules whose import side effect registers them. Done
    lazily here -- rather than importing them at the top of this registry module
    -- to avoid init-order / import-cycle risk (this module must stay
    import-light so it can be imported early by the transform package).
    """
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return
    # Block-scope copy -> thread cooperative copy expansion.
    import tilelang.tileop.copy.scale_expansion  # noqa: F401  (registers template)
    # Device-scope GEMM -> block/thread generated kernel expansion.
    import tilelang.tileop.gemm.scale_expansion  # noqa: F401  (registers template)
    # Block-scope elementwise (fill) -> thread expansion.
    import tilelang.tileop.elementwise.scale_expansion  # noqa: F401  (registers)
    _DEFAULTS_REGISTERED = True
