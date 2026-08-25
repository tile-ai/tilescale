"""Pre-LowerScaleLaunch pass for scale-scoped tile ops (Phase 4 Batch 3).

Generic scale tile-op template rewrite. Runs in ``LowerAndLegalize`` right after
``BindTarget`` and *before* ``LowerScaleLaunch``. It is the insertion point for
generative scale-scoped tile ops: a ``with T.scale(<scale>): <tile op>`` is
rewritten here into lower-scale launch IR (for the device GEMM template:
``device -> block -> thread``), which then flows through the existing scope-tree
``LowerScaleLaunch`` + ``LayoutInference`` + ``LowerTileOp`` unchanged.

The pass is fully generic: it scans tile ops whose innermost scale is a
*template-managed* scale (any scale for which the registry has a template),
resolves the ``(scale_name, op_name)`` template, and delegates decode / validate
/ rewrite to it. Op-specific checks (argument decode, MVP boundary, function
signature) live in the template, not here. Scales with no registered template
(e.g. ``thread`` / ``block`` today) are ignored and continue through the existing
pipeline; a template-managed scale with no template for the op is rejected
(fail-closed).

The only registered template today is the GEMM one
(``tilelang.tileop.gemm.device_template.GemmDeviceTemplate``, scope ``device``).

``PrepareScaleTileOps`` is the main entry; ``PrepareDeviceScaleTileOps`` and
``PrepareDeviceScaleGemm`` are kept as compatibility aliases.
"""

from typing import Optional

from tvm import tirx as tir
from tvm.tirx import PrimFunc
from tvm.tirx.stmt_functor import post_order_visit
from tvm.tirx.transform import prim_func_pass

from tilelang.language.scale import SCALE_CTX_NAME_KEY, SCALE_CTX_PATH_KEY

from tilelang.tileop.scale_template import (
    resolve_scale_template,
    has_scale_templates,
    ensure_default_scale_templates_registered,
)


def _str_val(v):
    return v.value if hasattr(v, "value") else str(v)


def _path_list(v):
    return [_str_val(x) for x in v]


def _scale_name(call) -> Optional[str]:
    ann = getattr(call, "annotations", None)
    if not ann:
        return None
    name = ann.get(SCALE_CTX_NAME_KEY, None)
    return _str_val(name) if name is not None else None


def _scale_path(call):
    ann = getattr(call, "annotations", None)
    if not ann:
        return None
    path = ann.get(SCALE_CTX_PATH_KEY, None)
    return _path_list(path) if path is not None else None


def PrepareScaleTileOps():
    """Rewrite a scale-scoped tile op via its registered template, reject the rest.

    Detects a tile op whose innermost scale (``tl.scale_ctx.name``) is
    template-managed, resolves its ``ScaleTileOpTemplate`` from the registry by
    ``(scale_name, op_name)``, and delegates ``decode`` / ``validate`` /
    ``rewrite`` to it (the template owns all op-specific checks and IR
    generation, preserving the original function's attrs / global_symbol). A
    template-managed scale whose op has no template is rejected (fail-closed);
    scales with no registered template are ignored (they continue through the
    existing lowering pipeline).
    """

    def pass_fn(func: PrimFunc, mod, ctx):
        ensure_default_scale_templates_registered()

        # Collect tile ops whose innermost scale is template-managed. A scale
        # with no registered template (e.g. thread/block today) is ignored so
        # hand-written kernels are unaffected. Within a template-managed scale,
        # fail-closed below if the specific op has no template.
        managed = []

        def visit(node):
            if isinstance(node, tir.Call) and isinstance(node.op, tir.op.Op):
                if not node.op.name.startswith("tl.tileop."):
                    return
                scale_name = _scale_name(node)
                if scale_name is not None and has_scale_templates(scale_name):
                    managed.append((node, scale_name))

        post_order_visit(func.body, visit)

        if not managed:
            return func

        # Generic limitation: a single template-managed tile-op rewrite per
        # function (multi-op templating would need an interface change).
        if len(managed) > 1:
            raise NotImplementedError(
                "PrepareScaleTileOps: multiple scale-scoped tile-op template "
                "rewrites in one function are not supported yet.")

        call, scale_name = managed[0]
        template = resolve_scale_template(scale_name, call.op.name)
        if template is None:
            raise NotImplementedError(
                f"PrepareScaleTileOps: scale-scoped tile op `{call.op.name}` "
                f"under scale `{scale_name}` has no registered template and is "
                f"not supported yet.")

        # The template owns all op-specific work: decode (raises for unsupported
        # forms), validate (function-level checks), and rewrite (builds the
        # generative kernel, preserving attrs / global_symbol).
        info = template.decode(call, func)
        template.validate(info, func)
        return template.rewrite(info, func)

    return prim_func_pass(pass_fn, opt_level=0)


# Compatibility aliases. The pass was first named for GEMM, then for the
# device scale; the registry is now scale-aware. Keep the old names working.
PrepareDeviceScaleTileOps = PrepareScaleTileOps
PrepareDeviceScaleGemm = PrepareScaleTileOps
