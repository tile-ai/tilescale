"""Scale-aware layout skeleton + verifier (Top-Down Scale Expansion).

Implements the read-only layout model from
``docs/compiler_internals/scale_layout_design.md``. A buffer needs two distinct
mappings, kept separate here:

1. **Scale ownership layout** (:class:`ScaleEdgeLayout` / :class:`ScaleLayout`):
   parent-scale logical indices -> child-scale coordinate + child-local logical
   indices, one edge per scale hop (device->block, block->thread, ...).
2. **Physical storage layout** (:class:`PhysicalStorageLayout`): logical indices
   within one storage instance -> physical address / register slot / fragment
   element in a chosen memory layer.

This is a **skeleton + verifier only**: it is not wired into
``NormalizeScaleExpansion`` or ``LowerScaleLaunch`` and changes no lowering
behavior. The verifier currently admits only the conservative core
(``coverage="exact"``, ``policy="one_to_one"``, static extents, no masks /
broadcast / scatter / reduction); anything else is reported invalid (fail-closed),
not silently accepted.

The edge mapping reuses ``tir.IndexMap`` directly (per the design's first-skeleton
recommendation):

    index_map(parent_indices) =
        child_coord[0..len(child_shape)) + child_local_indices[0..len(local_shape))

so the index map's output rank is ``len(child_shape) + len(local_shape)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from tvm import tirx as tir

_SUPPORTED_COVERAGE = ("exact",)
_SUPPORTED_POLICY = ("one_to_one",)
# All policy / coverage values the model names (for validation messages); only
# the subset above is admitted by the verifier today.
_KNOWN_COVERAGE = ("exact", "partial", "masked")
_KNOWN_POLICY = ("one_to_one", "replicated", "reduction", "scatter")

# Known physical storage layers / instance scopes for PhysicalStorageLayout.
_KNOWN_MEMORY_LAYER = (
    "global", "shared", "local", "fragment", "tmem", "distributed_shared")
_KNOWN_INSTANCE_SCOPE = (
    "device", "block", "warp", "thread", "cluster", "node")

# Bound on the parent-domain enumeration the verifier will brute-force. Static
# skeleton checks (bounds + injectivity) enumerate the full parent domain; refuse
# to enumerate an unreasonably large domain rather than hang.
_MAX_ENUM = 1 << 16


# ---------------------------------------------------------------------------
# Structures.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScaleEdgeLayout:
    """Ownership mapping for one scale edge (parent -> child).

    Maps parent-scale logical indices to a child workgroup coordinate plus the
    child-local logical indices. ``index_map`` output rank must equal
    ``len(child_shape) + len(local_shape)``.

    Optional partition metadata (Group I Phase 4-prep) mirrors what a ``Fragment``
    gives ``PartitionLoop``; all optional, defaults preserve the legacy shape:

    - ``inverse_index_map``: the (child_coord, child_local) -> parent-index inverse
      map. When provided it must round-trip with ``index_map`` over the static
      domain (verified). ``None`` = not supplied (the consumer would derive it).
    - ``thread_range``: ``(min, extent)`` of the thread axis for a rank-1 child
      (the Fragment ``ThreadRange``). ``None`` = unbound. When provided, requires
      a rank-1 ``child_shape`` whose extent equals the range extent.
    - ``replicate_extent``: replication factor (Fragment ``ReplicateExtent``). 1 =
      no replication (the only value admitted by the one_to_one verifier).
    """

    from_scale: str
    to_scale: str
    input_shape: tuple
    child_shape: tuple
    local_shape: tuple
    index_map: tir.IndexMap
    coverage: str = "exact"
    policy: str = "one_to_one"
    inverse_index_map: object = None
    thread_range: object = None  # (min, extent) tuple or None
    replicate_extent: object = 1


@dataclass(frozen=True)
class ScaleLayout:
    """A complete ownership layout over a scale path: a chain of edge layouts."""

    path: tuple
    input_shape: tuple
    edges: tuple


@dataclass(frozen=True)
class PhysicalStorageLayout:
    """Physical placement of a buffer's logical indices in one memory layer.

    ``memory_layer`` is the storage kind (``"global"`` / ``"shared"`` /
    ``"local"`` / ``"fragment"`` / ``"tmem"`` / ...). ``instance_scope`` is the
    allocation instance granularity (``"device"`` / ``"block"`` / ``"thread"`` /
    ...) -- e.g. shared memory is per-block, registers/fragments are per-thread.
    These two are distinct: the layer is *what kind of memory*, the instance scope
    is *how many copies and who owns each*.
    """

    buffer: object
    logical_shape: tuple
    memory_layer: str
    instance_scope: str
    index_map: tir.IndexMap


@dataclass(frozen=True)
class LayoutAccessEffect:
    """A layout-aware buffer access (precise layer above coarse MemoryEffects).

    Not consumed by barrier selection yet (that stays on ``MemoryEffects``); this
    is the future "is this access legal / race-free" descriptor.
    """

    buffer: object
    kind: str  # "read" | "write"
    scale_layout: ScaleLayout
    storage_layout: PhysicalStorageLayout
    access_scale: str
    policy: str = "one_to_one"


@dataclass(frozen=True)
class ElementwiseLayoutContract:
    """Layout requirement for a scale-aware elementwise op (skeleton).

    All inputs and the output share a logical domain; the output is written
    one_to_one. Verified by :func:`verify_elementwise_layout_contract`. Not
    consumed by any production lowering yet.
    """

    input_scale_layouts: tuple
    output_scale_layout: ScaleLayout
    input_storage_layouts: tuple
    output_storage_layout: PhysicalStorageLayout
    policy: str = "one_to_one"


@dataclass(frozen=True)
class CopyLayoutContract:
    """Layout requirement for a scale-aware copy (skeleton).

    ``copy_map`` maps destination logical indices to source logical indices.
    Verified by :func:`verify_copy_layout_contract`. Not consumed by any
    production lowering yet (the current block-copy template does not use it).
    """

    src_scale_layout: ScaleLayout
    dst_scale_layout: ScaleLayout
    src_storage_layout: PhysicalStorageLayout
    dst_storage_layout: PhysicalStorageLayout
    copy_map: tir.IndexMap


@dataclass
class LayoutVerificationResult:
    """Outcome of a layout legality check."""

    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok

    @staticmethod
    def success() -> "LayoutVerificationResult":
        return LayoutVerificationResult(ok=True, reason="")

    @staticmethod
    def failure(reason: str) -> "LayoutVerificationResult":
        return LayoutVerificationResult(ok=False, reason=reason)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _static_int(expr) -> Optional[int]:
    """Return the Python int of a static extent, or None if not a constant int."""
    if isinstance(expr, int):
        return expr
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    if hasattr(expr, "value") and isinstance(getattr(expr, "value"), int):
        return int(expr.value)
    return None


def _const_int_after_simplify(expr) -> Optional[int]:
    """Best-effort fold of a PrimExpr to a Python int (simplify if needed).

    ``IndexMap.map_indices`` usually folds floordiv/floormod of constants, but be
    robust: if the raw value is not already a constant int, run it through an
    arith analyzer once before giving up.
    """
    v = _static_int(expr)
    if v is not None:
        return v
    try:
        from tvm import arith
        simplified = arith.Analyzer().simplify(expr)
        return _static_int(simplified)
    except Exception:  # pragma: no cover - defensive
        return None


def _static_shape(shape) -> Optional[tuple]:
    """Return a tuple of python ints for a shape, or None if any dim is dynamic."""
    out = []
    for d in shape:
        v = _static_int(d)
        if v is None:
            return None
        out.append(v)
    return tuple(out)


def _iter_domain(shape):
    """Yield every index tuple in the static rectangular domain ``shape``."""
    if not shape:
        yield ()
        return
    head, *rest = shape
    for i in range(head):
        for tail in _iter_domain(rest):
            yield (i,) + tail


def _domain_size(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


# ---------------------------------------------------------------------------
# Verifier.
# ---------------------------------------------------------------------------


def verify_scale_edge_layout(edge: ScaleEdgeLayout) -> LayoutVerificationResult:
    """Verify one :class:`ScaleEdgeLayout` against the conservative core.

    Admits only ``coverage="exact"`` + ``policy="one_to_one"`` with static
    extents. Checks:

    - coverage / policy are in the supported subset (else fail-closed);
    - all shapes are static;
    - the index map's output rank == ``len(child_shape) + len(local_shape)``;
    - every parent index maps to in-bounds (child_coord, child_local) values;
    - the mapping is injective over the parent domain (one_to_one);
    - coverage is exact: the parent domain size equals
      ``|child_shape| * |local_shape|`` and every (child, local) pair is hit once.
    """
    if edge.coverage not in _KNOWN_COVERAGE:
        return LayoutVerificationResult.failure(
            f"unknown coverage {edge.coverage!r}")
    if edge.policy not in _KNOWN_POLICY:
        return LayoutVerificationResult.failure(
            f"unknown policy {edge.policy!r}")
    if edge.coverage not in _SUPPORTED_COVERAGE:
        return LayoutVerificationResult.failure(
            f"coverage {edge.coverage!r} is not supported yet "
            f"(only {_SUPPORTED_COVERAGE} admitted)")
    if edge.policy not in _SUPPORTED_POLICY:
        return LayoutVerificationResult.failure(
            f"policy {edge.policy!r} is not supported yet "
            f"(only {_SUPPORTED_POLICY} admitted)")

    in_shape = _static_shape(edge.input_shape)
    child_shape = _static_shape(edge.child_shape)
    local_shape = _static_shape(edge.local_shape)
    if in_shape is None or child_shape is None or local_shape is None:
        return LayoutVerificationResult.failure(
            "dynamic extents are not supported yet (input/child/local shapes "
            "must be static)")

    out_rank = len(child_shape) + len(local_shape)
    if len(edge.index_map.final_indices) != out_rank:
        return LayoutVerificationResult.failure(
            f"index_map output rank {len(edge.index_map.final_indices)} != "
            f"len(child_shape)+len(local_shape) = {out_rank}")
    if len(edge.index_map.initial_indices) != len(in_shape):
        return LayoutVerificationResult.failure(
            f"index_map input rank {len(edge.index_map.initial_indices)} != "
            f"len(input_shape) = {len(in_shape)}")

    parent_size = _domain_size(in_shape)
    target_size = _domain_size(child_shape) * _domain_size(local_shape)
    if parent_size != target_size:
        return LayoutVerificationResult.failure(
            f"coverage is not exact: parent domain size {parent_size} != "
            f"child*local target size {target_size}")
    if parent_size > _MAX_ENUM:
        return LayoutVerificationResult.failure(
            f"parent domain size {parent_size} exceeds the verifier enumeration "
            f"limit {_MAX_ENUM}")

    nchild = len(child_shape)
    seen = set()
    forward_pts = {}
    for idx in _iter_domain(in_shape):
        mapped = edge.index_map.map_indices(list(idx))
        vals = [_const_int_after_simplify(m) for m in mapped]
        if any(v is None for v in vals):
            return LayoutVerificationResult.failure(
                f"index_map produced a non-constant output for parent index "
                f"{idx}; only static maps are supported yet")
        child_coord = tuple(vals[:nchild])
        local_idx = tuple(vals[nchild:])
        for c, ext in zip(child_coord, child_shape):
            if c < 0 or c >= ext:
                return LayoutVerificationResult.failure(
                    f"child coordinate {child_coord} out of bounds {child_shape} "
                    f"at parent index {idx}")
        for l, ext in zip(local_idx, local_shape):
            if l < 0 or l >= ext:
                return LayoutVerificationResult.failure(
                    f"child-local index {local_idx} out of bounds {local_shape} "
                    f"at parent index {idx}")
        key = (child_coord, local_idx)
        if key in seen:
            return LayoutVerificationResult.failure(
                f"mapping is not injective (one_to_one): parent index {idx} "
                f"collides on (child={child_coord}, local={local_idx})")
        seen.add(key)
        forward_pts[key] = tuple(idx)

    # Exact coverage: injective + matching sizes => every target hit exactly once.
    if len(seen) != target_size:
        return LayoutVerificationResult.failure(
            f"coverage is not exact: {len(seen)} distinct targets hit, expected "
            f"{target_size}")

    # --- Optional partition metadata (Group I Phase 4-prep) ---
    # replicate_extent: only 1 is admitted by the one_to_one verifier.
    rep = _static_int(edge.replicate_extent)
    if rep is None:
        return LayoutVerificationResult.failure(
            f"replicate_extent must be a static int; got {edge.replicate_extent!r}")
    if rep != 1:
        return LayoutVerificationResult.failure(
            f"replicate_extent {rep} != 1 is not supported for policy "
            f"'one_to_one' (replication is not a one_to_one cover)")

    # thread_range: only valid for a rank-1 child; extent must match child_shape[0].
    if edge.thread_range is not None:
        tr = edge.thread_range
        if not (isinstance(tr, (tuple, list)) and len(tr) == 2):
            return LayoutVerificationResult.failure(
                "thread_range must be a (min, extent) pair or None")
        tr_min = _static_int(tr[0])
        tr_ext = _static_int(tr[1])
        if tr_min is None or tr_ext is None:
            return LayoutVerificationResult.failure(
                "thread_range (min, extent) must be static ints")
        if nchild != 1:
            return LayoutVerificationResult.failure(
                f"thread_range requires a rank-1 child_shape; got rank {nchild}")
        if tr_ext != child_shape[0]:
            return LayoutVerificationResult.failure(
                f"thread_range extent {tr_ext} != child_shape[0] {child_shape[0]}")

    # inverse_index_map: when provided, must round-trip with the forward map over
    # the full (child, local) target domain.
    if edge.inverse_index_map is not None:
        inv = edge.inverse_index_map
        if len(inv.initial_indices) != out_rank:
            return LayoutVerificationResult.failure(
                f"inverse_index_map input rank {len(inv.initial_indices)} != "
                f"forward output rank {out_rank}")
        if len(inv.final_indices) != len(in_shape):
            return LayoutVerificationResult.failure(
                f"inverse_index_map output rank {len(inv.final_indices)} != "
                f"len(input_shape) {len(in_shape)}")
        for (child_coord, local_idx), parent in forward_pts.items():
            out = inv.map_indices(list(child_coord) + list(local_idx))
            got = tuple(_const_int_after_simplify(x) for x in out)
            if any(v is None for v in got):
                return LayoutVerificationResult.failure(
                    f"inverse_index_map produced a non-constant output at "
                    f"(child={child_coord}, local={local_idx})")
            if got != parent:
                return LayoutVerificationResult.failure(
                    f"inverse_index_map does not round-trip: forward {parent} -> "
                    f"(child={child_coord}, local={local_idx}) but inverse -> {got}")

    return LayoutVerificationResult.success()


def verify_scale_layout(layout: ScaleLayout) -> LayoutVerificationResult:
    """Verify a :class:`ScaleLayout`: scale chain is continuous and edges compose.

    Checks, in order:

    - scale-chain continuity: each ``edges[i].to_scale == edges[i+1].from_scale``
      (the child scale of one edge is the parent scale of the next), so a
      discontinuous chain like ``device->block`` then ``warp->thread`` is rejected
      even if the shapes happen to line up;
    - path consistency: ``layout.path`` equals
      ``(edges[0].from_scale, edges[0].to_scale, edges[1].to_scale, ...)``;
    - per-edge legality (:func:`verify_scale_edge_layout`);
    - shape composition: ``edges[0].input_shape == layout.input_shape`` and each
      edge's ``local_shape`` matches the next edge's ``input_shape`` (the
      child-local domain of one scale is the parent domain of the next).
    """
    if not layout.edges:
        return LayoutVerificationResult.failure("scale layout has no edges")

    # Scale-chain continuity: to_scale of one edge must be from_scale of the next.
    for i in range(len(layout.edges) - 1):
        cur = layout.edges[i]
        nxt = layout.edges[i + 1]
        if cur.to_scale != nxt.from_scale:
            return LayoutVerificationResult.failure(
                f"scale chain is discontinuous: edge {i} to_scale "
                f"{cur.to_scale!r} != edge {i+1} from_scale {nxt.from_scale!r}; "
                f"adjacent scale edges must compose (edge[i].to_scale == "
                f"edge[i+1].from_scale)")

    # Path consistency: path must be the edges' scale chain.
    expected_path = ((layout.edges[0].from_scale,)
                     + tuple(e.to_scale for e in layout.edges))
    if tuple(layout.path) != expected_path:
        return LayoutVerificationResult.failure(
            f"path {tuple(layout.path)} does not match the edge scale chain "
            f"{expected_path}; layout.path must equal (edges[0].from_scale, "
            f"edges[0].to_scale, edges[1].to_scale, ...)")

    if tuple(layout.edges[0].input_shape) != tuple(layout.input_shape):
        return LayoutVerificationResult.failure(
            f"first edge input_shape {tuple(layout.edges[0].input_shape)} != "
            f"layout input_shape {tuple(layout.input_shape)}")

    for i, edge in enumerate(layout.edges):
        res = verify_scale_edge_layout(edge)
        if not res.ok:
            return LayoutVerificationResult.failure(
                f"edge {i} ({edge.from_scale}->{edge.to_scale}) invalid: {res.reason}")

    for i in range(len(layout.edges) - 1):
        cur = layout.edges[i]
        nxt = layout.edges[i + 1]
        if tuple(cur.local_shape) != tuple(nxt.input_shape):
            return LayoutVerificationResult.failure(
                f"edges {i}->{i+1} do not compose: edge {i} local_shape "
                f"{tuple(cur.local_shape)} != edge {i+1} input_shape "
                f"{tuple(nxt.input_shape)}")

    return LayoutVerificationResult.success()


def verify_physical_storage_layout(
        layout: PhysicalStorageLayout) -> LayoutVerificationResult:
    """Verify a :class:`PhysicalStorageLayout`'s structural legality.

    Skeleton checks (no physical-bounds check yet -- there is no physical-shape
    field; this validates the *mapping structure*, not target-address bounds):

    - ``memory_layer`` is a known layer (global / shared / local / fragment /
      tmem / distributed_shared);
    - ``instance_scope`` is a known scope (device / block / warp / thread /
      cluster / node);
    - ``logical_shape`` is static;
    - ``index_map`` input rank == ``len(logical_shape)``;
    - ``index_map`` output rank > 0;
    - over the enumerated logical domain (capped at the 2^16 limit), every mapped
      output is a static integer.

    Anything dynamic / unknown / oversized fails closed.
    """
    if layout.memory_layer not in _KNOWN_MEMORY_LAYER:
        return LayoutVerificationResult.failure(
            f"unknown memory_layer {layout.memory_layer!r} (known: "
            f"{_KNOWN_MEMORY_LAYER})")
    if layout.instance_scope not in _KNOWN_INSTANCE_SCOPE:
        return LayoutVerificationResult.failure(
            f"unknown instance_scope {layout.instance_scope!r} (known: "
            f"{_KNOWN_INSTANCE_SCOPE})")

    logical = _static_shape(layout.logical_shape)
    if logical is None:
        return LayoutVerificationResult.failure(
            "dynamic logical_shape is not supported yet (must be static)")

    if len(layout.index_map.initial_indices) != len(logical):
        return LayoutVerificationResult.failure(
            f"index_map input rank {len(layout.index_map.initial_indices)} != "
            f"len(logical_shape) = {len(logical)}")
    if len(layout.index_map.final_indices) < 1:
        return LayoutVerificationResult.failure(
            "index_map output rank must be > 0")

    size = _domain_size(logical)
    if size > _MAX_ENUM:
        return LayoutVerificationResult.failure(
            f"logical domain size {size} exceeds the verifier enumeration limit "
            f"{_MAX_ENUM}")
    for idx in _iter_domain(logical):
        mapped = layout.index_map.map_indices(list(idx))
        vals = [_const_int_after_simplify(m) for m in mapped]
        if any(v is None for v in vals):
            return LayoutVerificationResult.failure(
                f"index_map produced a non-constant output for logical index "
                f"{idx}; only static physical maps are supported yet")
    return LayoutVerificationResult.success()


def compose_scale_edges(edges) -> ScaleLayout:
    """Build a :class:`ScaleLayout` from an ordered list of edges (no verify).

    Convenience constructor: derives ``path`` from the edges' scale names and
    ``input_shape`` from the first edge. It does not verify scale continuity,
    path consistency, or shape composition -- call :func:`verify_scale_layout` on
    the result. (It builds ``path`` from each edge's ``to_scale``, so a
    discontinuous edge chain still produces a ``ScaleLayout``; the verifier is
    what rejects it.)
    """
    edges = tuple(edges)
    if not edges:
        raise ValueError("compose_scale_edges requires at least one edge")
    path = (edges[0].from_scale,) + tuple(e.to_scale for e in edges)
    return ScaleLayout(path=path, input_shape=tuple(edges[0].input_shape),
                       edges=edges)


# ---------------------------------------------------------------------------
# Fragment relationship helper.
# ---------------------------------------------------------------------------


def scale_edge_layout_from_fragment(fragment, *,
                                    from_scale: str = "block",
                                    to_scale: str = "thread",
                                    coverage: str = "exact",
                                    policy: str = "one_to_one") -> ScaleEdgeLayout:
    """Build a ``block -> thread`` :class:`ScaleEdgeLayout` from a ``Fragment``.

    A ``Fragment`` already bundles ownership (logical index -> thread id) and
    physical placement (logical index -> per-thread fragment/register slot). This
    helper extracts the *ownership* edge view:

    - ``input_shape`` = ``fragment.get_input_shape()``
    - ``child_shape`` = ``(fragment.get_thread_size(),)``
    - ``local_shape`` = ``fragment.get_output_shape()``
    - ``index_map`` output = ``[map_forward_thread(idx)] + map_forward_index(idx)``,
      so the output rank is ``1 + len(local_shape)``.

    Skeleton limits (fail-closed -- ``NotImplementedError``): only static input /
    output / thread sizes; only ``coverage="exact"`` + ``policy="one_to_one"``; a
    replicated fragment (``replicate_size != 1``) is not supported (its ownership
    is not one_to_one). The returned edge is *not* verified here -- call
    :func:`verify_scale_edge_layout`.
    """
    if coverage != "exact" or policy != "one_to_one":
        raise NotImplementedError(
            "scale_edge_layout_from_fragment: only coverage='exact' / "
            "policy='one_to_one' are supported yet.")

    in_shape = _static_shape(tuple(fragment.get_input_shape()))
    if in_shape is None:
        raise NotImplementedError(
            "scale_edge_layout_from_fragment: only static fragment input shapes "
            "are supported yet.")

    thread_size = _static_int(fragment.get_thread_size())
    if thread_size is None:
        raise NotImplementedError(
            "scale_edge_layout_from_fragment: only a static thread size is "
            "supported yet.")

    local_shape = _static_shape(tuple(fragment.get_output_shape()))
    if local_shape is None:
        raise NotImplementedError(
            "scale_edge_layout_from_fragment: only static fragment output "
            "(per-thread local) shapes are supported yet.")

    # Replicated fragments bundle a non-one_to_one ownership mapping (a value is
    # owned by multiple threads); not admitted in the skeleton.
    rep = _static_int(getattr(fragment, "replicate_size", 1))
    if rep is None:
        raise NotImplementedError(
            "scale_edge_layout_from_fragment: a dynamic replicate size is not "
            "supported yet.")
    if rep != 1:
        raise NotImplementedError(
            "scale_edge_layout_from_fragment: replicated fragments "
            f"(replicate_size={rep}) are not supported yet (ownership is not "
            "one_to_one).")

    def _edge_map(*idx):
        idx = list(idx)
        thread = fragment.map_forward_thread(idx)
        # map_forward_thread may return a 1-tuple/array; normalize to a scalar.
        if isinstance(thread, (list, tuple)):
            thread = thread[0]
        elif hasattr(thread, "__len__"):
            thread = thread[0]
        local = fragment.map_forward_index(idx)
        if isinstance(local, (list, tuple)):
            local = list(local)
        else:
            local = list(local) if hasattr(local, "__len__") else [local]
        return [thread] + local

    index_map = tir.IndexMap.from_func(_edge_map, ndim=len(in_shape),
                                       index_dtype="int32")
    # Partition metadata (Group I Phase 4-prep): the fragment's thread range and
    # replicate extent (rep==1 enforced above); derive the inverse over the
    # (child, local) target domain so it can drive PartitionLoop directly.
    thread_range = None
    tr = getattr(fragment, "thread", None)
    # The fragment thread axis spans [0, thread_size); BindThreadRange may offset
    # it, but the public Python Fragment does not expose the bound range, so use
    # the canonical [0, thread_size) for the non-offset case.
    thread_range = (0, thread_size)
    inverse_index_map = None
    try:
        # IndexMap.inverse takes the INPUT domain shape (the parent logical
        # indices) over which to validate bijectivity.
        inverse_index_map = index_map.inverse(list(in_shape))
    except Exception:  # pragma: no cover - inverse not always derivable
        inverse_index_map = None
    return ScaleEdgeLayout(
        from_scale=from_scale,
        to_scale=to_scale,
        input_shape=tuple(in_shape),
        child_shape=(thread_size,),
        local_shape=tuple(local_shape),
        index_map=index_map,
        coverage=coverage,
        policy=policy,
        inverse_index_map=inverse_index_map,
        thread_range=thread_range,
        replicate_extent=1,
    )


def fragment_from_scale_edge_layout(edge: ScaleEdgeLayout):
    """Build a :class:`tilelang.layout.Fragment` from a ``block -> thread`` edge.

    The narrow reverse of :func:`scale_edge_layout_from_fragment`: it bridges our
    ScaleEdgeLayout back onto TileLang's core layout object so a verified scale
    edge can feed the existing ``Fragment`` / ``ParallelOp`` machinery. The edge's
    ``index_map`` maps a logical index to ``(thread_coord, local_0, ...)``; the
    Fragment's ``forward_thread`` is the thread-coord component and its
    ``forward_index`` is the local component.

    Supported (else ``NotImplementedError``, fail-closed):

    - ``from_scale == "block"`` and ``to_scale == "thread"``;
    - ``coverage == "exact"`` and ``policy == "one_to_one"``;
    - static ``input_shape`` / ``child_shape`` / ``local_shape``;
    - ``child_shape`` rank 1 (a single thread axis -- the Fragment thread dim);
    - the edge first passes :func:`verify_scale_edge_layout`.

    ``local_shape`` may be any rank; its components become the Fragment's
    ``forward_index`` outputs in order (Fragment output shape == ``local_shape``).
    """
    # Import here to avoid a module-level dependency cycle (layout pulls in tir
    # FFI which this skeleton otherwise stays independent of).
    from tilelang.layout import Fragment

    if edge.from_scale != "block" or edge.to_scale != "thread":
        raise NotImplementedError(
            "fragment_from_scale_edge_layout: only a 'block' -> 'thread' edge "
            f"maps to a Fragment; got '{edge.from_scale}' -> '{edge.to_scale}'.")
    if edge.coverage != "exact" or edge.policy != "one_to_one":
        raise NotImplementedError(
            "fragment_from_scale_edge_layout: only coverage='exact' / "
            f"policy='one_to_one' edges are supported; got coverage="
            f"{edge.coverage!r} / policy={edge.policy!r}.")

    in_shape = _static_shape(tuple(edge.input_shape))
    if in_shape is None:
        raise NotImplementedError(
            "fragment_from_scale_edge_layout: only static input_shape is "
            "supported yet.")
    child_shape = _static_shape(tuple(edge.child_shape))
    if child_shape is None:
        raise NotImplementedError(
            "fragment_from_scale_edge_layout: only static child_shape is "
            "supported yet.")
    local_shape = _static_shape(tuple(edge.local_shape))
    if local_shape is None:
        raise NotImplementedError(
            "fragment_from_scale_edge_layout: only static local_shape is "
            "supported yet.")
    if len(child_shape) != 1:
        raise NotImplementedError(
            "fragment_from_scale_edge_layout: only a rank-1 child_shape (a single "
            f"thread axis) maps to a Fragment thread dim; got {edge.child_shape}.")

    # The edge must be a legal exact/one_to_one cover before we trust its map.
    res = verify_scale_edge_layout(edge)
    if not res.ok:
        raise NotImplementedError(
            f"fragment_from_scale_edge_layout: edge does not verify: {res.reason}")

    n_local = len(local_shape)

    # Split the edge index_map output into (thread_coord, local...). The Fragment
    # forward fns receive the logical loop vars positionally; ``map_indices``
    # returns an Array of scalar PrimExpr. The thread fn returns the single thread
    # component. The index fn returns a bare scalar for a rank-1 local (matching
    # how the Fragment ctor wraps a single forward_index), or a list of scalars
    # for a multi-axis local.
    def _forward_thread(*idx):
        out = edge.index_map.map_indices(list(idx))
        return out[0]

    def _forward_index(*idx):
        out = edge.index_map.map_indices(list(idx))
        if n_local == 1:
            return out[1]
        return [out[1 + k] for k in range(n_local)]

    return Fragment(list(in_shape),
                    forward_thread_fn=_forward_thread,
                    forward_index_fn=_forward_index)


# ---------------------------------------------------------------------------
# Buffer layout binding (Group E): mirror an existing producer (e.g. GEMM
# accumulator) Fragment as the new layout model, so a downstream epilogue can
# *inherit* the producer's real layout instead of guessing a generic scalar
# partition. Read-only / not wired into lowering.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BufferLayoutBinding:
    """Binds a buffer to its scale-ownership + physical-storage layout.

    Mirrors an existing producer's layout (its ``Fragment``) in the new model:

    - ``scale_layout`` -- the ``block -> thread`` ownership :class:`ScaleLayout`
      (who owns which logical element);
    - ``storage_layout`` -- the :class:`PhysicalStorageLayout` (where each owned
      element lives: for a fragment accumulator, ``memory_layer="fragment"``,
      ``instance_scope="thread"``);
    - ``source`` -- provenance tag (e.g. ``"gemm_accumulator"``);
    - ``producer_op`` -- optional op name that produced this layout.

    Not consumed by any pass; it is the descriptor a future epilogue-inheritance
    step would read to reuse the producer layout.
    """

    buffer: object
    scale_layout: ScaleLayout
    storage_layout: PhysicalStorageLayout
    source: str = "unknown"
    producer_op: Optional[str] = None


def buffer_layout_binding_from_fragment(buffer, fragment, *,
                                        source: str = "gemm_accumulator",
                                        producer_op: Optional[str] = None,
                                        logical_shape=None) -> BufferLayoutBinding:
    """Build a :class:`BufferLayoutBinding` mirroring an existing ``Fragment``.

    Splits the producer ``Fragment`` (which bundles ownership + physical slot)
    into the two explicit layouts:

    - ownership: ``scale_edge_layout_from_fragment(fragment)`` -> a ``block ->
      thread`` :class:`ScaleLayout` (verified);
    - physical: a :class:`PhysicalStorageLayout` with ``memory_layer="fragment"``,
      ``instance_scope="thread"``, ``logical_shape`` = the fragment input shape
      (or ``logical_shape`` override), and ``index_map`` = the fragment's
      per-thread local index map (``map_forward_index``).

    Both layouts are verified (`verify_scale_layout` / `verify_physical_storage_layout`);
    fail-closed (``NotImplementedError``) on replicated / dynamic / non-static
    fragments (delegated to :func:`scale_edge_layout_from_fragment` and the static
    checks here). This does NOT change the GEMM lowering -- it only re-expresses
    the layout it already produced.
    """
    edge = scale_edge_layout_from_fragment(fragment)
    scale_layout = compose_scale_edges([edge])
    sl_res = verify_scale_layout(scale_layout)
    if not sl_res.ok:
        raise NotImplementedError(
            f"buffer_layout_binding_from_fragment: ownership scale layout does "
            f"not verify: {sl_res.reason}")

    in_shape = _static_shape(tuple(fragment.get_input_shape()))
    if in_shape is None:
        raise NotImplementedError(
            "buffer_layout_binding_from_fragment: only static fragment input "
            "shapes are supported yet.")
    logical = tuple(logical_shape) if logical_shape is not None else in_shape
    if _static_shape(logical) is None:
        raise NotImplementedError(
            "buffer_layout_binding_from_fragment: logical_shape must be static.")

    n_in = len(in_shape)

    def _phys_map(*idx):
        local = fragment.map_forward_index(list(idx))
        if isinstance(local, (list, tuple)):
            return list(local)
        if hasattr(local, "__len__"):
            return list(local)
        return [local]

    storage_layout = PhysicalStorageLayout(
        buffer=buffer,
        logical_shape=tuple(logical),
        memory_layer="fragment",
        instance_scope="thread",
        index_map=tir.IndexMap.from_func(_phys_map, ndim=n_in,
                                         index_dtype="int32"))
    ps_res = verify_physical_storage_layout(storage_layout)
    if not ps_res.ok:
        raise NotImplementedError(
            f"buffer_layout_binding_from_fragment: physical storage layout does "
            f"not verify: {ps_res.reason}")

    return BufferLayoutBinding(
        buffer=buffer,
        scale_layout=scale_layout,
        storage_layout=storage_layout,
        source=source,
        producer_op=producer_op)


def choose_elementwise_layout_source(candidate_buffers, existing_bindings,
                                     fallback_edge=None):
    """Pick the layout an elementwise op should adopt (producer-reuse skeleton).

    Read-only policy skeleton for future epilogue inheritance:

    1. If EVERY buffer in ``candidate_buffers`` that has an entry in
       ``existing_bindings`` shares one consistent binding (same ownership scale
       layout over the full domain), return ``("inherited", binding)`` -- the
       epilogue should reuse the producer's layout (e.g. a GEMM accumulator),
       NOT generate a generic scalar partition.
    2. Else if ``fallback_edge`` is provided, return ``("fallback", fallback_edge)``.
    3. Else fail-closed: ``("none", None)``.

    ``existing_bindings`` maps a buffer key (the buffer's backing data Var, via
    :func:`_buffer_key`) to a :class:`BufferLayoutBinding`. ``candidate_buffers``
    is the elementwise op's buffers (inputs + output). This does not lower
    anything; it is the decision a future inheritance pass would make.
    """
    bound = []
    for buf in candidate_buffers:
        key = _buffer_key(buf)
        if key in existing_bindings:
            bound.append(existing_bindings[key])

    if bound:
        first = bound[0]
        for other in bound[1:]:
            if not _scale_layouts_equal(first.scale_layout, other.scale_layout):
                # Conflicting producer layouts among the operands -> cannot
                # inherit a single one; fall through to fallback / fail-closed.
                bound = []
                break

    if bound:
        return ("inherited", bound[0])
    if fallback_edge is not None:
        return ("fallback", fallback_edge)
    return ("none", None)


def _buffer_key(buffer):
    """A stable dict key for a buffer (its backing data Var)."""
    data = getattr(buffer, "data", None)
    return data if data is not None else buffer


def _scale_layouts_equal(a: ScaleLayout, b: ScaleLayout) -> bool:
    """Full-domain equality of two single-edge block->thread scale layouts."""
    if a.path != b.path:
        return False
    if len(a.edges) != 1 or len(b.edges) != 1:
        return False
    ea, eb = a.edges[0], b.edges[0]
    if (ea.from_scale, ea.to_scale) != (eb.from_scale, eb.to_scale):
        return False
    sa = _static_shape(tuple(ea.input_shape))
    sb = _static_shape(tuple(eb.input_shape))
    if sa is None or sa != sb:
        return False
    if (tuple(_static_shape(tuple(ea.child_shape)) or ())
            != tuple(_static_shape(tuple(eb.child_shape)) or ())):
        return False
    if (tuple(_static_shape(tuple(ea.local_shape)) or ())
            != tuple(_static_shape(tuple(eb.local_shape)) or ())):
        return False
    for idx in _iter_domain(sa):
        oa = tuple(_const_int_after_simplify(x)
                   for x in ea.index_map.map_indices(list(idx)))
        ob = tuple(_const_int_after_simplify(x)
                   for x in eb.index_map.map_indices(list(idx)))
        if oa != ob:
            return False
    return True


# ---------------------------------------------------------------------------
# Layout contract verifiers (skeleton; not consumed by lowering yet).
# ---------------------------------------------------------------------------


def verify_elementwise_layout_contract(
        contract: ElementwiseLayoutContract) -> LayoutVerificationResult:
    """Verify an :class:`ElementwiseLayoutContract` (conservative skeleton).

    Admits only the elementwise MVP shape:

    - ``policy == "one_to_one"`` (no broadcast / mask / reduction / scatter);
    - every input and the output :class:`ScaleLayout` verifies;
    - every input and the output :class:`PhysicalStorageLayout` verifies;
    - all input scale-layout ``input_shape`` match the output's exactly (no
      broadcast).

    Anything else fails closed.
    """
    if contract.policy != "one_to_one":
        return LayoutVerificationResult.failure(
            f"elementwise policy {contract.policy!r} is not supported yet "
            f"(only 'one_to_one'; no broadcast/mask/reduction/scatter)")

    out_res = verify_scale_layout(contract.output_scale_layout)
    if not out_res.ok:
        return LayoutVerificationResult.failure(
            f"output scale layout invalid: {out_res.reason}")
    out_shape = tuple(contract.output_scale_layout.input_shape)

    for i, sl in enumerate(contract.input_scale_layouts):
        res = verify_scale_layout(sl)
        if not res.ok:
            return LayoutVerificationResult.failure(
                f"input {i} scale layout invalid: {res.reason}")
        if tuple(sl.input_shape) != out_shape:
            return LayoutVerificationResult.failure(
                f"input {i} logical shape {tuple(sl.input_shape)} != output "
                f"logical shape {out_shape} (broadcast not supported yet)")

    out_storage = verify_physical_storage_layout(contract.output_storage_layout)
    if not out_storage.ok:
        return LayoutVerificationResult.failure(
            f"output storage layout invalid: {out_storage.reason}")
    for i, psl in enumerate(contract.input_storage_layouts):
        res = verify_physical_storage_layout(psl)
        if not res.ok:
            return LayoutVerificationResult.failure(
                f"input {i} storage layout invalid: {res.reason}")

    return LayoutVerificationResult.success()


def verify_copy_layout_contract(
        contract: CopyLayoutContract) -> LayoutVerificationResult:
    """Verify a :class:`CopyLayoutContract` (conservative skeleton).

    Admits only the structural copy core:

    - src / dst :class:`ScaleLayout` verify;
    - src / dst :class:`PhysicalStorageLayout` verify;
    - ``copy_map`` maps the destination logical domain to the source logical
      domain: input rank == len(dst logical domain), output rank == len(src
      logical domain).

    No layout-transform legality (bounds / injectivity / coverage of ``copy_map``)
    beyond rank/static checks yet -- that comes when copy migrates onto this
    contract. Anything dynamic / mismatched fails closed.
    """
    src_res = verify_scale_layout(contract.src_scale_layout)
    if not src_res.ok:
        return LayoutVerificationResult.failure(
            f"src scale layout invalid: {src_res.reason}")
    dst_res = verify_scale_layout(contract.dst_scale_layout)
    if not dst_res.ok:
        return LayoutVerificationResult.failure(
            f"dst scale layout invalid: {dst_res.reason}")

    src_storage = verify_physical_storage_layout(contract.src_storage_layout)
    if not src_storage.ok:
        return LayoutVerificationResult.failure(
            f"src storage layout invalid: {src_storage.reason}")
    dst_storage = verify_physical_storage_layout(contract.dst_storage_layout)
    if not dst_storage.ok:
        return LayoutVerificationResult.failure(
            f"dst storage layout invalid: {dst_storage.reason}")

    dst_rank = len(contract.dst_scale_layout.input_shape)
    src_rank = len(contract.src_scale_layout.input_shape)
    if len(contract.copy_map.initial_indices) != dst_rank:
        return LayoutVerificationResult.failure(
            f"copy_map input rank {len(contract.copy_map.initial_indices)} != "
            f"dst logical rank {dst_rank} (copy_map maps dst -> src indices)")
    if len(contract.copy_map.final_indices) != src_rank:
        return LayoutVerificationResult.failure(
            f"copy_map output rank {len(contract.copy_map.final_indices)} != "
            f"src logical rank {src_rank} (copy_map maps dst -> src indices)")

    return LayoutVerificationResult.success()
