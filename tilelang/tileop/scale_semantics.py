"""Scale-parametric semantics + storage orderability (Top-Down Scale Expansion).

Two registries, deliberately separate, because a *scale* (an execution / semantic
level) is not the same thing as a *memory space*:

- :class:`ScaleSemantics` -- the scale hierarchy and what synchronization a scale
  can realize: ``name`` / ``parent`` / ``children``, the ``default_barrier`` for
  that scale, and whether ordering is realizable in-kernel
  (``supports_in_kernel_barrier``) or as a stage boundary
  (``supports_stage_boundary``). ``ScaleSemantics`` does *not* describe which
  memory is "visible" at a scale.
- :class:`StorageSemantics` -- a physical storage kind (``register`` / ``shared``
  / ``global`` / ...), its logical owner scale, its instance granularity, and --
  the field the barrier planner consults -- the :class:`OrderingMode`\\ s that can
  preserve a producer -> consumer dependency for that storage.

The barrier planner asks "what storage did the producer write / consumer read,
and does that storage have an ordering mechanism available at the expansion
boundary?" -- not "can this scale see this memory space?". This is the
storage/orderability cleanup (milestone 9): ``register`` / ``local`` /
``fragment`` are not block-barrier-orderable (a CTA barrier cannot make one
thread's registers visible to another), so a block-scope register dependency does
*not* yield ``__syncthreads()``; ``shared`` is orderable by a block barrier;
``global`` is orderable by a device launch boundary (skeleton). ``cluster`` /
``node`` synchronization is fail-closed until encoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from tilelang.tileop.scale_expansion import BarrierSpec


# ---------------------------------------------------------------------------
# Scale hierarchy + synchronization capability (NOT memory visibility).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScaleSemantics:
    """Hierarchy + synchronization capability of a single scale (op-agnostic).

    Describes *what synchronization a scale can realize*, not what memory it can
    see. Memory orderability lives in :class:`StorageSemantics`.
    """

    name: str
    parent: Optional[str]
    children: tuple[str, ...]
    default_barrier: Optional[BarrierSpec]
    supports_in_kernel_barrier: bool
    supports_stage_boundary: bool
    # When False, the scale's synchronization model is not encoded yet; any
    # ordering requirement at this scale must loud-error (fail-closed).
    sync_modeled: bool = True


_REGISTRY: dict[str, ScaleSemantics] = {}


def register_scale_semantics(sem: ScaleSemantics) -> None:
    """Register (or replace) the semantics for ``sem.name``."""
    _REGISTRY[sem.name] = sem


def resolve_scale_semantics(scale_name: str) -> Optional[ScaleSemantics]:
    """Return the semantics for ``scale_name``, or ``None`` if unregistered."""
    _ensure_defaults()
    return _REGISTRY.get(scale_name)


def registered_scale_semantics() -> dict[str, ScaleSemantics]:
    """Snapshot of the scale registry (for tests / introspection)."""
    _ensure_defaults()
    return dict(_REGISTRY)


# ---------------------------------------------------------------------------
# Storage / orderability model -- the barrier planner's core input.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrderingMode:
    """A mechanism that can preserve a producer -> consumer ordering.

    ``kind`` names the mechanism (``"program_order"`` -- same executor, no
    barrier needed; ``"block_barrier"`` -- a CTA sync; ``"launch_boundary"`` -- a
    kernel split; ``"warp_collective"`` -- a warp shuffle/MMA exchange; ...).
    ``executor_scale`` is the scale that must execute the mechanism (e.g.
    ``"block"`` for a CTA barrier). ``memory_scope`` is the storage space the
    mechanism orders (e.g. ``"shared"`` / ``"global"``).
    """

    kind: str
    executor_scale: Optional[str]
    memory_scope: Optional[str]


@dataclass(frozen=True)
class StorageSemantics:
    """Physical storage kind + which mechanisms can order its dependencies.

    ``logical_owner_scale`` is the algorithmic owner (e.g. a GEMM accumulator
    fragment logically belongs to a ``block`` tile even though it is physically
    stored per thread). ``storage_space`` is the physical kind (``"register"`` /
    ``"fragment"`` / ``"local"`` / ``"shared"`` / ``"global"`` / ...).
    ``storage_instance_scope`` is the instance granularity (per ``thread`` /
    ``block`` / ``device``). ``ordering_modes`` is the field the barrier planner
    consults: a producer -> consumer dependency on this storage is orderable only
    if one of these modes is available at the expansion boundary.
    """

    logical_owner_scale: Optional[str]
    storage_space: str
    storage_instance_scope: str
    ordering_modes: tuple[OrderingMode, ...]


_STORAGE_REGISTRY: dict[str, StorageSemantics] = {}


def register_storage_semantics(sem: StorageSemantics) -> None:
    """Register (or replace) the semantics for ``sem.storage_space``."""
    _STORAGE_REGISTRY[sem.storage_space] = sem


def resolve_storage_semantics(storage_space: str) -> Optional[StorageSemantics]:
    """Return the semantics for ``storage_space``, or ``None`` if unregistered."""
    _ensure_defaults()
    return _STORAGE_REGISTRY.get(storage_space)


def registered_storage_semantics() -> dict[str, StorageSemantics]:
    """Snapshot of the storage registry (for tests / introspection)."""
    _ensure_defaults()
    return dict(_STORAGE_REGISTRY)


# Program order is available for every storage: a same-executor (e.g.
# same-thread) producer/consumer pair is ordered without any barrier.
_PROGRAM_ORDER = OrderingMode(kind="program_order", executor_scale=None,
                              memory_scope=None)


_DEFAULTS_REGISTERED = False


def _ensure_defaults() -> None:
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return
    _DEFAULTS_REGISTERED = True

    # --- Scale hierarchy + sync capability -------------------------------

    # thread: the leaf scale, no wider synchronization.
    register_scale_semantics(ScaleSemantics(
        name="thread", parent="block", children=(),
        default_barrier=None,
        supports_in_kernel_barrier=False, supports_stage_boundary=False,
        sync_modeled=True))

    # warp: subdivision of a block; an in-kernel barrier is realizable (warp
    # collective / block sync), but warp has no default ordering barrier of its
    # own (a narrower optimization).
    register_scale_semantics(ScaleSemantics(
        name="warp", parent="block", children=("thread",),
        default_barrier=None,
        supports_in_kernel_barrier=True, supports_stage_boundary=False,
        sync_modeled=True))

    # block / CTA: in-kernel CTA barrier; default barrier is __syncthreads.
    register_scale_semantics(ScaleSemantics(
        name="block", parent="device", children=("warp", "thread"),
        default_barrier=BarrierSpec.block_sync("shared"),
        supports_in_kernel_barrier=True, supports_stage_boundary=False,
        sync_modeled=True))

    # cluster: cluster-level sync is target specific -- fail-closed until encoded.
    register_scale_semantics(ScaleSemantics(
        name="cluster", parent="device", children=("block",),
        default_barrier=None,
        supports_in_kernel_barrier=True, supports_stage_boundary=False,
        sync_modeled=False))

    # device / GPU: full-grid ordering is a launch boundary (split kernels), not
    # an in-kernel instruction. Stage lowering is skeleton-only.
    register_scale_semantics(ScaleSemantics(
        name="device", parent="node", children=("cluster", "block"),
        default_barrier=BarrierSpec.device_launch_boundary(),
        supports_in_kernel_barrier=False, supports_stage_boundary=True,
        sync_modeled=True))

    # node / multi-GPU: runtime ownership + barriers must be specified first.
    register_scale_semantics(ScaleSemantics(
        name="node", parent=None, children=("device",),
        default_barrier=None,
        supports_in_kernel_barrier=False, supports_stage_boundary=True,
        sync_modeled=False))

    # --- Storage orderability --------------------------------------------

    # register / local / fragment: thread-owned, stored per thread. Only program
    # order (same thread) can order a dependency on these. There is NO
    # block_barrier mode: different threads cannot read each other's registers,
    # and a CTA barrier cannot make register state visible -- so a cross-thread
    # register/local/fragment dependency has no automatic ordering mechanism and
    # the planner fails closed (it must be carried by an explicit communication
    # primitive / template proof, e.g. a warp collective or a shared transfer).
    for space in ("register", "local", "fragment"):
        register_storage_semantics(StorageSemantics(
            logical_owner_scale="thread",
            storage_space=space,
            storage_instance_scope="thread",
            ordering_modes=(_PROGRAM_ORDER,)))

    # shared: one instance per block, owned by the block tile. A block barrier
    # (__syncthreads) orders shared dependencies across threads in the CTA.
    register_storage_semantics(StorageSemantics(
        logical_owner_scale="block",
        storage_space="shared",
        storage_instance_scope="block",
        ordering_modes=(
            _PROGRAM_ORDER,
            OrderingMode(kind="block_barrier", executor_scale="block",
                         memory_scope="shared"),
        )))

    # global: one instance per device, visible across blocks. The safe ordering
    # mechanism between device-scope segments is a launch boundary (kernel split).
    register_storage_semantics(StorageSemantics(
        logical_owner_scale="device",
        storage_space="global",
        storage_instance_scope="device",
        ordering_modes=(
            _PROGRAM_ORDER,
            OrderingMode(kind="launch_boundary", executor_scale="device",
                         memory_scope="global"),
        )))

    # distributed_shared: cluster-local shared, ordered by a cluster sync. The
    # cluster sync mechanism's executor scale (`cluster`) is itself fail-closed
    # (sync_modeled=False) for now, so the planner surfaces the dependency but
    # the BarrierPlanner refuses to lower it until cluster sync is encoded.
    register_storage_semantics(StorageSemantics(
        logical_owner_scale="cluster",
        storage_space="distributed_shared",
        storage_instance_scope="cluster",
        ordering_modes=(
            _PROGRAM_ORDER,
            OrderingMode(kind="cluster_sync", executor_scale="cluster",
                         memory_scope="distributed_shared"),
        )))
