from __future__ import annotations

from tvm import tirx
from tvm.tirx import PrimExpr, IntImm, address_of


def put_warp(
    src: PrimExpr,
    dst: PrimExpr,
    size: PrimExpr,
    dst_pe: PrimExpr | IntImm | None = -1,
    unroll_factor: int = 4,
    enable_aggressive_vectorize: bool = False,
):
    """Put to a remote buffer with unrolled loop."""
    if dst_pe is None:
        dst_pe = -1
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.put"),
        src,
        dst,
        size,
        dst_pe,
        unroll_factor,
        "warp",
        enable_aggressive_vectorize,
    )


def get_warp(
    src: PrimExpr,
    dst: PrimExpr,
    size: PrimExpr,
    src_pe: PrimExpr | IntImm | None = -1,
    unroll_factor: int = 4,
    enable_aggressive_vectorize: bool = False,
):
    """Get from a remote buffer with unrolled loop."""
    if src_pe is None:
        src_pe = -1
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.get"),
        src,
        dst,
        size,
        src_pe,
        unroll_factor,
        "warp",
        enable_aggressive_vectorize,
    )


def put_block(src: PrimExpr, dst: PrimExpr, size: PrimExpr, dst_pe: PrimExpr | IntImm | None = -1):
    """Put to a remote buffer."""
    if dst_pe is None:
        dst_pe = -1
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.tileop.put"), src, dst, size, dst_pe, 0, "block", True)


def get_block(src: PrimExpr, dst: PrimExpr, size: PrimExpr, src_pe: PrimExpr | IntImm | None = -1):
    """Get from a remote buffer."""
    if src_pe is None:
        src_pe = -1
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.tileop.get"), src, dst, size, src_pe, 0, "block", True)


def ld(
    src: PrimExpr,
    value: PrimExpr,
    scope: str = "gpu",
    sem: str = "weak",
    na: bool = False,
    nc: bool = False,
    src_pe: PrimExpr | IntImm | None = -1,
):
    """Load a value from an address with explicit PTX scope and semantic.

    ``nc=True`` lowers to ``ld.global.nc``, which reads through the non-coherent
    (read-only) cache and carries neither a memory-ordering nor a scope
    qualifier. It is therefore only valid with ``sem="weak"`` and without an
    explicit scope, and it must not be used on memory a peer may write during the
    kernel, because the read-only cache is not kept coherent.
    """
    if src_pe is None:
        src_pe = -1
    assert scope in ["cta", "gpu", "sys"], "Scope must be one of 'cta', 'gpu', or 'sys'."
    assert sem in ["weak", "volatile", "acquire", "relaxed"], "Semantic must be one of 'weak', 'volatile', 'acquire', or 'relaxed'."
    if nc:
        # Accepting either of these would silently drop it: ld.global.nc has no
        # slot for a semantic or a scope.
        assert sem == "weak", (
            f"nc=True requires sem='weak', got sem={sem!r}: ld.global.nc has no memory ordering. Use nc=False for an ordered load."
        )
        assert scope == "gpu", (
            f"nc=True does not accept an explicit scope, got scope={scope!r}: ld.global.nc is unscoped. Use nc=False for a scoped load."
        )
    scope_id = {"cta": 0, "gpu": 1, "sys": 2}[scope]
    sem_id = {"weak": 0, "volatile": 1, "acquire": 2, "release": 3, "relaxed": 4}[sem]
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.tileop.ld"), address_of(src), value, sem_id, scope_id, int(na), int(nc), src_pe)


def st(
    dst: PrimExpr,
    value: PrimExpr,
    scope: str = "gpu",
    sem: str = "weak",
    na: bool = False,
    dst_pe: PrimExpr | IntImm | None = -1,
):
    """Store a value to an address with explicit PTX scope and semantic."""
    if dst_pe is None:
        dst_pe = -1
    assert scope in ["cta", "gpu", "sys"], "Scope must be one of 'cta', 'gpu', or 'sys'."
    assert sem in ["weak", "volatile", "release", "relaxed"], "Semantic must be one of 'weak', 'volatile', 'release', or 'relaxed'."
    scope_id = {"cta": 0, "gpu": 1, "sys": 2}[scope]
    sem_id = {"weak": 0, "volatile": 1, "acquire": 2, "release": 3, "relaxed": 4}[sem]
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.tileop.st"), address_of(dst), value, sem_id, scope_id, int(na), dst_pe)


def atom_add(target: PrimExpr, value: PrimExpr, scope: str = "gpu", sem: str = "relaxed"):
    """Perform a scoped uint32 atomic add and return the previous value."""
    assert scope in ["gpu", "sys"], "Scope must be one of 'gpu', or 'sys'."
    assert sem in ["relaxed", "acquire", "release", "acq_rel"], "Semantic must be one of 'relaxed', 'acquire', 'release', or 'acq_rel'."
    return tirx.call_intrin("uint32", tirx.op.Op.get("tl.atom_add"), address_of(target), value, sem, scope)
