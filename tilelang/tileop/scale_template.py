"""Scale-aware tile-op template registry (Phase 4 Batch 3).

A small registry that maps a (scale name, tile-op name) pair -- e.g.
``("device", "tl.tileop.gemm")`` -- to a ``ScaleTileOpTemplate`` describing how
to decode / validate / rewrite a direct ``with T.scale(<scale>): <tile op>`` into
a runnable lower-scale kernel. The generic pass
``tilelang.transform.prepare_device_scale.PrepareScaleTileOps`` resolves a
template by (scale, op) and drives it; the only registered template today is the
GEMM one (``tilelang.tileop.gemm.device_template.GemmDeviceTemplate``), which
claims ``scale_names == ("device",)``.

This registry generalizes the earlier device-specific one. Backward-compatible
``device_*`` names are provided as thin wrappers (and re-exported from
``tilelang.tileop.device_template``) so existing callers keep working. Adding a
new scale (e.g. ``node``) is just registering a template with
``scale_names=("node",)`` -- no pass change.
"""

from __future__ import annotations

import abc
from typing import Optional


class ScaleTileOpTemplate(abc.ABC):
    """Decode / validate / rewrite a direct scale-scoped tile op.

    A template is registered for one or more (scale name, tile-op name) pairs via
    ``scale_names`` x ``op_names``. The generic pass calls, in order: ``decode``
    (returns an opaque info object or raises a precise ``NotImplementedError`` for
    an unsupported form), then ``validate`` (function-level checks against the
    decoded info), then ``rewrite`` (returns the rewritten ``PrimFunc``, with the
    original function's attrs preserved).
    """

    @property
    @abc.abstractmethod
    def scale_names(self) -> tuple[str, ...]:
        """The scale names (e.g. ``("device",)``) this template claims."""

    @property
    @abc.abstractmethod
    def op_names(self) -> tuple[str, ...]:
        """The ``tl.tileop.*`` names this template claims."""

    @abc.abstractmethod
    def decode(self, call, func):
        """Decode the tile-op call into an info object, or raise NotImplementedError."""

    @abc.abstractmethod
    def validate(self, info, func) -> None:
        """Function-level validation against the decoded info (raise on failure)."""

    @abc.abstractmethod
    def rewrite(self, info, func):
        """Return the rewritten PrimFunc (original attrs preserved)."""


_REGISTRY: list[ScaleTileOpTemplate] = []


def register_scale_template(template: ScaleTileOpTemplate) -> None:
    """Register a scale tile-op template (idempotent by template type)."""
    for idx, existing in enumerate(_REGISTRY):
        if type(existing) is type(template):
            _REGISTRY[idx] = template
            return
    _REGISTRY.append(template)


def resolve_scale_template(scale_name: str,
                           op_name: str) -> Optional[ScaleTileOpTemplate]:
    """Return the template claiming ``(scale_name, op_name)``, or None."""
    for template in _REGISTRY:
        if scale_name in template.scale_names and op_name in template.op_names:
            return template
    return None


def has_scale_templates(scale_name: str) -> bool:
    """True if any registered template claims ``scale_name`` (for any op).

    The pass uses this to decide whether a scale is template-managed: a
    template-managed scale routes through the registry (fail-closed per op),
    while an unmanaged scale (e.g. ``thread`` / ``block`` today) is ignored and
    continues through the existing lowering pipeline.
    """
    for template in _REGISTRY:
        if scale_name in template.scale_names:
            return True
    return False


_DEFAULTS_REGISTERED = False


def ensure_default_scale_templates_registered() -> None:
    """Lazily register the built-in scale tile-op templates (idempotent).

    Imports the GEMM device template module (whose import side-effect registers
    ``GemmDeviceTemplate``). Done lazily here -- rather than importing gemm at the
    top of this registry module -- to avoid init-order / import-cycle risk
    (this module must stay import-light so it can be imported early).
    """
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return
    import tilelang.tileop.gemm.device_template  # noqa: F401  (registers template)
    _DEFAULTS_REGISTERED = True


# ---------------------------------------------------------------------------
# Backward-compatible device_* aliases (the registry was originally
# device-specific). These keep existing callers working unchanged.
# ---------------------------------------------------------------------------

DeviceTileOpTemplate = ScaleTileOpTemplate


def register_device_template(template: ScaleTileOpTemplate) -> None:
    """Deprecated alias for :func:`register_scale_template`."""
    register_scale_template(template)


def resolve_device_template(op_name: str) -> Optional[ScaleTileOpTemplate]:
    """Deprecated: resolve a template for the ``device`` scale by op name."""
    return resolve_scale_template("device", op_name)


def ensure_default_device_templates_registered() -> None:
    """Deprecated alias for :func:`ensure_default_scale_templates_registered`."""
    ensure_default_scale_templates_registered()
