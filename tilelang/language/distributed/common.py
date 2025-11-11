"""The language interface for tl programs."""
from __future__ import annotations

from tvm import tir
from tvm.tir import PrimExpr


def get_rank():
    """Get the rank of the current process.
    """
    return tir.call_intrin("uint64", tir.op.Op.get("tl.get_rank"))


def get_num_ranks():
    """Get the number of processes.
    """
    return tir.call_intrin("uint64", tir.op.Op.get("tl.get_num_ranks"))


def put_thread(src: PrimExpr,
             dst: PrimExpr,
             size: PrimExpr,
             mbar: PrimExpr | None = None,
             dst_pe: PrimExpr | None = None,
             unroll_factor: int = 4,
             scope: str = "gpu"):
    """Put to a remote buffer with unrolled loop.

    Args:
        src: PrimExpr
            The source address.
        dst: PrimExpr
            The destination address.
        size: PrimExpr
            The size of the put in elements.
        dst_pe: PrimExpr | None
            The PE index of the destination.
            If provided, the dst is a symmetric address, otherwise it is a UVA address.
            If not provided, the dst is a UVA address and dst_pe is None.
        unroll_factor: int
            The unroll factor
        scope: str
            The copy scopy, can be gpu or cluser.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.put"), src, dst, size, dst_pe, unroll_factor, scope,
                           "thread", mbar)
    
def put_warp(src: PrimExpr,
             dst: PrimExpr,
             size: PrimExpr,
             dst_pe: PrimExpr | None = None,
             unroll_factor: int = 4,
             scope: str = "gpu"):
    """Put to a remote buffer with unrolled loop.

    Args:
        src: PrimExpr
            The source address.
        dst: PrimExpr
            The destination address.
        size: PrimExpr
            The size of the put in elements.
        dst_pe: PrimExpr | None
            The PE index of the destination.
            If provided, the dst is a symmetric address, otherwise it is a UVA address.
            If not provided, the dst is a UVA address and dst_pe is None.
        unroll_factor: int
            The unroll factor
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.put"), src, dst, size, dst_pe, unroll_factor, scope,
                           "warp")


def get_warp(src: PrimExpr,
             dst: PrimExpr,
             size: PrimExpr,
             src_pe: PrimExpr | None = None,
             unroll_factor: int = 4):
    """Get from a remote buffer with unrolled loop.

    Args:
        src: PrimExpr
            The source address.
        dst: PrimExpr
            The destination address.
        size: PrimExpr
            The size of the get in elements.
        src_pe: PrimExpr | None
            The PE index of the source.
            If provided, the src is a symmetric address, otherwise it is a UVA address.
            If not provided, the src is a UVA address and src_pe is None.
        unroll_factor: int
            The unroll factor
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.get"), src, dst, size, src_pe, unroll_factor,
                           "warp")


def put_block(src: PrimExpr, dst: PrimExpr, size: PrimExpr, dst_pe: PrimExpr | None = None, scope: str = "gpu"):
    """Put to a remote buffer.

    Args:
        src: PrimExpr
            The source address.
        dst: PrimExpr
            The destination address.
        size: PrimExpr
            The size of the put in elements.
        dst_pe: PrimExpr | None
            The PE index of the destination.
            If provided, the dst is a symmetric address, otherwise it is a UVA address.
            If not provided, the dst is a UVA address and dst_pe is None.
        scope: str
            The copy scopy, can be gpu or cluser.
    """
    return tir.call_intrin(
        "handle", tir.op.Op.get("tl.put"), src, dst, size, dst_pe, 0, scope, "block"
    )  # NOTE(wt): unroll_factor is not needed because currently we implement block-level comm based on NVSHMEM-style copy


def get_block(src: PrimExpr, dst: PrimExpr, size: PrimExpr, src_pe: PrimExpr | None = None):
    """Get from a remote buffer.

    Args:
        src: PrimExpr
            The source address.
        dst: PrimExpr
            The destination address.
        size: PrimExpr
            The size of the get in elements.
        src_pe: PrimExpr | None
            The PE index of the source.
            If provided, the src is a symmetric address, otherwise it is a UVA address.
            If not provided, the src is a UVA address and src_pe is None.
    """
    return tir.call_intrin(
        "handle", tir.op.Op.get("tl.get"), src, dst, size, src_pe, 0, "block"
    )  # NOTE(wt): unroll_factor is not needed because currently we implement block-level comm based on NVSHMEM-style copy
