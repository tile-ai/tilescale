"""Device-scale GEMM expansion template (Top-Down Scale Expansion, milestone 10).

The op-specific half of the ``device -> block`` GEMM expansion, on the generic
top-down path. It claims the ``("device", "tl.tileop.gemm")`` edge (plus the
non-MVP gemm op names, so they are decoded and loud-errored here rather than
slipping past) in the :mod:`tilelang.tileop.scale_expansion` registry, and is
driven by the generic normalizer
(:mod:`tilelang.transform.normalize_scale_expansion`).

It reuses the existing GEMM MVP logic in
:mod:`tilelang.tileop.gemm.device_template` -- ``_decode_device_gemm`` (per-call
MVP decode/reject), ``validate_device_gemm_func`` (function-level MVP checks),
and ``device_gemm_template_with_attrs`` (build the generated
``device -> block -> thread`` kernel) -- so the MVP boundary and the
``PrepareDeviceScaleGemm: ...`` reject messages are defined once and identical on
both the legacy registry path and this expansion path.

The plan it returns is a generic **replace_func** :class:`ExpansionPlan`
(``replacement_func`` set): a device-scope GEMM expands the *whole* PrimFunc into
the generated lower-scale kernel. The normalizer performs the swap
op-agnostically.
"""

from __future__ import annotations

from tilelang.tileop.scale_expansion import (
    ExpansionContext,
    ExpansionPlan,
    ScaleExpansionTemplate,
    ScaleSegment,
    register_scale_expansion_template,
)
from tilelang.tileop.gemm.device_template import (
    _GEMM_OP_NAMES,
    _GEMM_OP_NAME,
    _decode_device_gemm,
    validate_device_gemm_func,
    device_gemm_template_with_attrs,
)

_FROM_SCALE = "device"
_TO_SCALE = "block"


def _gemm_call(stmt):
    """Return the GEMM ``tl.tileop.*`` Call inside an Evaluate, or None."""
    from tvm import tirx as tir
    if isinstance(stmt, tir.Evaluate):
        val = stmt.value
        if (isinstance(val, tir.Call) and isinstance(val.op, tir.op.Op)
                and val.op.name in _GEMM_OP_NAMES):
            return val
    return None


class GemmDeviceExpansionTemplate(ScaleExpansionTemplate):
    """Expand a device-scope ``T.gemm`` into a generated block/thread kernel.

    ``decode`` runs the per-call MVP decode (raising for any non-MVP shape /
    dtype / op, including wgmma / tcgen05 / blockscaled); ``validate`` runs the
    function-level MVP checks; ``plan`` builds the generated
    ``device -> block -> thread`` PrimFunc and returns it as a ``replace_func``
    plan. All reuse the legacy GEMM helpers, so behavior and messages are
    unchanged from the registry path.
    """

    @property
    def from_scale(self) -> str:
        return _FROM_SCALE

    @property
    def to_scale(self) -> str:
        return _TO_SCALE

    @property
    def op_names(self) -> tuple[str, ...]:
        return _GEMM_OP_NAMES

    def match(self, segment: ScaleSegment, context: ExpansionContext) -> bool:
        return (segment.scale_name == _FROM_SCALE
                and len(segment.stmts) == 1
                and _gemm_call(segment.stmts[0]) is not None)

    def decode(self, segment: ScaleSegment, context: ExpansionContext):
        if len(segment.stmts) != 1 or _gemm_call(segment.stmts[0]) is None:
            raise NotImplementedError(
                "PrepareDeviceScaleGemm: a device-scope segment that is not a "
                "single T.gemm cannot be expanded.")
        return _decode_device_gemm(_gemm_call(segment.stmts[0]))

    def validate(self, info, context: ExpansionContext) -> None:
        validate_device_gemm_func(info, context.func)

    def plan(self, info, context: ExpansionContext) -> ExpansionPlan:
        generated = device_gemm_template_with_attrs(info, context.func)
        return ExpansionPlan(
            from_scale=_FROM_SCALE,
            to_scale=_TO_SCALE,
            replacement_func=generated,
        )


register_scale_expansion_template(GemmDeviceExpansionTemplate())
