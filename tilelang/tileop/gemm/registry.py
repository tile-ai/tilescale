from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

from tvm.target import Target


GemmTargetPredicate = Callable[[Target], bool]


@dataclass(frozen=True)
class GemmImplEntry:
    name: str
    inst_name: str
    predicate: GemmTargetPredicate
    impl_class: type


_GEMM_IMPLS: list[GemmImplEntry] = []

# Scale scopes whose innermost context still lowers through the standard GEMM
# implementations. Scopes with dedicated templates (e.g. a device-scope GEMM
# expanded by NormalizeScaleExpansion) never reach this gate with that scope.
_ALLOWED_GEMM_SCALE_SCOPES = frozenset({None, "block", "thread", "warp", "device"})


def register_gemm_impl(
    name: str,
    inst_name: str,
    predicate: GemmTargetPredicate,
    impl_class: type,
) -> None:
    """Register a backend-specific GEMM implementation class."""
    entry = GemmImplEntry(name, inst_name, predicate, impl_class)
    for idx, registered in enumerate(_GEMM_IMPLS):
        if registered.name == name:
            _GEMM_IMPLS[idx] = entry
            return
    _GEMM_IMPLS.append(entry)


def resolve_gemm_impl(gemm_inst: str, target: Target, scale_scope: str | None = None,
                      scale_path: tuple[str, ...] | None = None) -> type:
    """Resolve the registered implementation class for a GEMM instruction key.

    ``scale_scope`` is the innermost ``tl.scale_ctx.name`` attached to the GEMM
    tile op (or ``None``). ``scale_path`` is the full scale path (outer -> inner,
    ``tl.scale_ctx.path``) or ``None``. Only ``scale_scope`` gates which scopes
    GEMM lowering accepts; ``scale_path`` is ancestor metadata and does NOT gate
    (a path with an ancestor cluster / sm / die is fine as long as the innermost
    scope is supported). Implementation selection is unchanged.
    """
    # Consistency: when both are present, the innermost path element must match
    # the innermost scope, otherwise the metadata is contradictory.
    if scale_path and scale_scope is not None:
        if scale_path[-1] != scale_scope:
            raise ValueError(
                f"Inconsistent scale_ctx metadata: scale_ctx.path innermost "
                f"`{scale_path[-1]}` does not match scale_ctx.name `{scale_scope}` "
                f"(scale_ctx.path={list(scale_path)})."
            )
    if scale_scope not in _ALLOWED_GEMM_SCALE_SCOPES:
        raise NotImplementedError(
            f"T.gemm under scale_ctx scope `{scale_scope}` is not supported yet. "
            f"Allowed scopes are {sorted(s for s in _ALLOWED_GEMM_SCALE_SCOPES if s is not None)} "
            f"(or no scale context); scale_ctx scope `{scale_scope}` has no GEMM "
            f"specialization."
        )
    matches = [entry for entry in _GEMM_IMPLS if entry.inst_name == gemm_inst and entry.predicate(target)]
    if not matches:
        raise ValueError(f"No GEMM implementation registered for instruction {gemm_inst} and target {target}")
    if len(matches) > 1:
        names = ", ".join(entry.name for entry in matches)
        raise ValueError(f"Multiple GEMM implementations matched instruction {gemm_inst} and target {target}: {names}")
    return matches[0].impl_class
