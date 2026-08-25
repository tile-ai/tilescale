"""Scale-parametric dependency + barrier + stage planning (storage-driven).

The generic analysis layer that turns ordered scale segments + memory effects
into synchronization decisions, written *once* over the scale hierarchy via
:mod:`tilelang.tileop.scale_semantics`. See
``docs/compiler_internals/scale_programming_plan.md`` -- "Scale-Parametric
Analysis And Barriers".

The core input is **storage orderability**, not scale "visibility": a
producer -> consumer dependency is barrier-orderable at an expansion boundary
only if the written/read :class:`~tilelang.tileop.scale_semantics.StorageSemantics`
exposes an :class:`~tilelang.tileop.scale_semantics.OrderingMode` whose executor
is that boundary scale. So:

- ``shared`` carries a ``block_barrier`` mode (executor ``block``) -> a
  block-scope shared dependency yields ``BarrierSpec(scope="block",
  kind="sync_threads", memory_scope="shared")``;
- ``global`` carries a ``launch_boundary`` mode (executor ``device``) -> a
  device-scope global dependency yields ``BarrierSpec(scope="device",
  kind="launch_boundary", memory_scope="global")`` (stage boundary, skeleton);
- ``register`` / ``local`` / ``fragment`` carry only ``program_order`` -- no
  ``block_barrier`` mode -- so a block-scope register dependency derives NO
  block sync. A cross-thread register dependency has no automatic ordering
  mechanism and must be carried by an explicit communication primitive / template
  proof; the planner does not silently insert ``__syncthreads()`` for it.

Three pieces:

- :class:`ScaleDependencyAnalysis` -- derive barrier-needing dependencies from
  producer/consumer effects at a boundary scale, consulting storage orderability.
- :class:`BarrierPlanner` -- turn a dependency's ordering mode into a
  :class:`BarrierSpec` (fail-closed when the executor scale's sync is unmodeled).
- :class:`StagePlanner` -- decide in-kernel barrier vs. stage boundary. The
  stage-boundary path (``device`` launch boundary) is skeleton: the decision is
  returned but the normalizer loud-errors when asked to lower it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from tilelang.tileop.scale_expansion import BarrierSpec, MemoryEffects
from tilelang.tileop.scale_semantics import (
    OrderingMode,
    resolve_scale_semantics,
    resolve_storage_semantics,
)

# Map coarse MemoryEffects flags to the storage_space names the storage registry
# uses. local/register/fragment share the "local" effect flag; we report it as
# "local" (the StorageSemantics for "local" carries only program_order, like
# register/fragment, so the orderability conclusion is identical).
_SPACE_WRITE_FLAGS = (
    ("shared", "writes_shared"),
    ("global", "writes_global"),
    ("local", "writes_local"),
)
_SPACE_READ_FLAGS = (
    ("shared", "reads_shared"),
    ("global", "reads_global"),
    ("local", "reads_local"),
)


def _written_spaces(eff: MemoryEffects) -> tuple[str, ...]:
    return tuple(space for space, flag in _SPACE_WRITE_FLAGS if getattr(eff, flag))


def _read_spaces(eff: MemoryEffects) -> tuple[str, ...]:
    return tuple(space for space, flag in _SPACE_READ_FLAGS if getattr(eff, flag))


@dataclass
class ScaleDependency:
    """A producer->consumer ordering requirement orderable at one scale.

    ``ordering_mode`` is the storage's mechanism that realizes the ordering at
    ``scale_name`` (e.g. a ``block_barrier`` for shared at block scale). It is
    always a non-``program_order`` mode -- program order needs no barrier, so it
    never produces a :class:`ScaleDependency`.
    """

    scale_name: str
    memory_space: str
    ordering_mode: OrderingMode


class ScaleDependencyAnalysis:
    """Derive barrier-needing dependencies from producer/consumer effects.

    Storage-driven and op-agnostic: for each storage space the producer writes
    and the consumer reads, consult its :class:`StorageSemantics` and check
    whether it has an :class:`OrderingMode` whose ``executor_scale`` is the
    expansion boundary ``scale_name``. If so, lexical order across that boundary
    requires that barrier. Storage with only ``program_order`` at this boundary
    (``register`` / ``local`` / ``fragment``) yields no dependency -- it cannot be
    fixed by a barrier at this scale.
    """

    def required_dependencies(self, scale_name: str,
                              producer: MemoryEffects,
                              consumer: MemoryEffects) -> list[ScaleDependency]:
        deps: list[ScaleDependency] = []
        written = set(_written_spaces(producer))
        read = set(_read_spaces(consumer))
        for space in written & read:
            storage = resolve_storage_semantics(space)
            if storage is None:
                continue
            mode = self._orderable_at(storage, scale_name)
            if mode is not None:
                deps.append(ScaleDependency(scale_name=scale_name,
                                            memory_space=space,
                                            ordering_mode=mode))
        return deps

    @staticmethod
    def _orderable_at(storage, scale_name: str) -> Optional[OrderingMode]:
        """Return a non-program-order mode for ``storage`` executable at scale.

        program_order is never returned: a same-executor dependency needs no
        barrier, and a cross-thread dependency on program-order-only storage
        (register/local/fragment) is NOT orderable here -- it must be carried by
        an explicit primitive, so we surface nothing and the dependency stays
        unhandled (fail-closed at the template / lowering level).
        """
        for mode in storage.ordering_modes:
            if mode.kind == "program_order":
                continue
            if mode.executor_scale == scale_name:
                return mode
        return None


class BarrierPlanner:
    """Turn a required dependency's ordering mode into a :class:`BarrierSpec`.

    Maps the storage's :class:`OrderingMode` to a concrete barrier
    (``block_barrier`` -> ``sync_threads`` at ``block``; ``launch_boundary`` ->
    ``launch_boundary`` at ``device``), validating the executor scale's
    synchronization is modeled. Fail-closed: an ordering mode whose executor
    scale is unmodeled (``cluster`` / ``node``) loud-errors rather than silently
    dropping the barrier.
    """

    # OrderingMode.kind -> BarrierSpec.kind.
    _KIND_MAP = {
        "block_barrier": "sync_threads",
        "launch_boundary": "launch_boundary",
        "cluster_sync": "cluster_sync",
        "warp_collective": "warp_collective",
    }

    def plan_barrier(self, dep: ScaleDependency) -> BarrierSpec:
        mode = dep.ordering_mode
        executor = mode.executor_scale
        sem = resolve_scale_semantics(executor) if executor else None
        if sem is None:
            raise NotImplementedError(
                f"BarrierPlanner: no scale semantics registered for executor "
                f"scale `{executor}`; cannot plan a barrier for a "
                f"`{dep.memory_space}` dependency.")
        if not sem.sync_modeled:
            raise NotImplementedError(
                f"BarrierPlanner: synchronization at scale `{executor}` is not "
                f"modeled yet; a `{dep.memory_space}` ordering dependency at this "
                f"scale is not supported yet.")
        barrier_kind = self._KIND_MAP.get(mode.kind)
        if barrier_kind is None:
            raise NotImplementedError(
                f"BarrierPlanner: ordering mode `{mode.kind}` for a "
                f"`{dep.memory_space}` dependency is not supported yet.")
        return BarrierSpec(scope=executor, kind=barrier_kind,
                           memory_scope=mode.memory_scope or dep.memory_space)


@dataclass
class StageDecision:
    """How a planned barrier is realized."""

    barrier: BarrierSpec
    in_kernel: bool          # True -> emit an in-kernel sync at the barrier point
    stage_boundary: bool     # True -> split the program into ordered launch stages


class StagePlanner:
    """Decide in-kernel barrier vs. stage boundary for a planned barrier.

    Skeleton: it returns the decision from :class:`ScaleSemantics`
    (``supports_in_kernel_barrier`` / ``supports_stage_boundary``). The actual
    multi-kernel stage split (``device`` launch boundary) is NOT implemented yet
    -- a stage-boundary decision is returned but the normalizer loud-errors when
    asked to lower it.
    """

    def plan_stage(self, barrier: BarrierSpec) -> StageDecision:
        sem = resolve_scale_semantics(barrier.scope)
        if sem is None:
            raise NotImplementedError(
                f"StagePlanner: no scale semantics registered for barrier scope "
                f"`{barrier.scope}`.")
        if sem.supports_in_kernel_barrier:
            return StageDecision(barrier=barrier, in_kernel=True,
                                 stage_boundary=False)
        if sem.supports_stage_boundary:
            return StageDecision(barrier=barrier, in_kernel=False,
                                 stage_boundary=True)
        raise NotImplementedError(
            f"StagePlanner: scale `{barrier.scope}` supports neither an in-kernel "
            f"barrier nor a stage boundary; cannot realize barrier `{barrier.kind}`.")
