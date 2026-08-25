"""Ordered scale-program staging model (Top-Down Scale Expansion, milestone 11).

When a scale region is an *ordered program* -- more than one execution-affecting
segment that must run in lexical order at that scale (e.g. a device-scope
``T.gemm`` followed by a device-scope store, two device GEMMs, or a static
``T.serial`` loop of GEMMs) -- preserving that order across the scale requires a
**stage boundary**, not just an in-kernel barrier. For ``block`` the boundary is
a CTA sync (in-kernel); for ``device`` it is a launch boundary (split kernels).

This module defines the *generic*, scale-agnostic representation of such a
staged program. It is a **skeleton**: the structures are built and reasoned about,
but no multi-kernel lowering happens yet. The normalizer builds a
:class:`StageProgram` when it recognizes an ordered program at a stage-boundary
scale and then fail-closes (the actual kernel split / launch-ordering / module
packaging is future work, deliberately separated from this representation so the
runtime-API surface is not entangled with the analysis).

The structures mirror the barrier model in
:mod:`tilelang.tileop.scale_barrier_planner`: a :class:`StageBoundary` reuses the
:class:`~tilelang.tileop.scale_expansion.BarrierSpec` vocabulary
(``scope="device"``, ``kind="launch_boundary"``) so a stage boundary is just the
"this ordering needs a kernel split, not an in-kernel barrier" case of the same
lattice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from tilelang.tileop.scale_expansion import BarrierSpec, MemoryEffects


@dataclass(frozen=True)
class StageBoundary:
    """An ordering boundary between two stages of a scale program.

    ``barrier`` is the :class:`BarrierSpec` that realizes the boundary (e.g.
    ``BarrierSpec(scope="device", kind="launch_boundary")``). ``in_kernel`` mirrors
    the :class:`~tilelang.tileop.scale_barrier_planner.StageDecision`: True when
    the boundary is an in-kernel barrier, False when it requires a kernel/stage
    split. The skeleton only *records* this; it does not emit the split.
    """

    barrier: BarrierSpec
    in_kernel: bool


@dataclass
class StagePlan:
    """One ordered stage of a scale program.

    A stage is a maximal run of statements that execute together at the scale,
    before the next :class:`StageBoundary`. ``stmts`` are the TIR statements of
    this stage (op-agnostic -- could be a tile op, a store, a loop body, ...).
    ``boundary_after`` is the boundary that must follow this stage (``None`` for
    the last stage).
    """

    scale_name: str
    index: int
    stmts: list
    effects: MemoryEffects = field(default_factory=MemoryEffects.empty)
    boundary_after: Optional[StageBoundary] = None


@dataclass
class StageProgram:
    """An ordered sequence of :class:`StagePlan`\\ s at one scale.

    Built by the normalizer when a scale region is recognized as an ordered
    program needing staging. ``requires_stage_boundary`` is True when at least one
    boundary is *not* in-kernel (i.e. a launch/runtime stage split is needed) --
    the skeleton uses this to fail-close. ``reason`` is a human-readable summary
    (e.g. "static device loop", "device GEMM followed by a store").
    """

    scale_name: str
    stages: list[StagePlan]
    reason: str = ""

    @property
    def requires_stage_boundary(self) -> bool:
        return any(s.boundary_after is not None and not s.boundary_after.in_kernel
                   for s in self.stages)

    @property
    def num_stages(self) -> int:
        return len(self.stages)
