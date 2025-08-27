"""The language interface for tl programs."""

from tvm import tir
from typing import Optional
from tvm.tir import PrimExpr


def push_warp(src: PrimExpr,
              dst: PrimExpr,
              size: PrimExpr,
              dst_pe: Optional[PrimExpr] = None,
              unroll_factor: int = 4):
    """Push from a remote buffer with unrolled loop.

    Args:
        src: PrimExpr
            The source address.
        dst: PrimExpr
            The destination address.
        size: PrimExpr
            The size of the copy.
        dst_pe: Optional[PrimExpr]
            The PE index of the destination.
            If provided, the dst is a symmetric address, otherwise it is a UVA address.
            If not provided, the dst is a UVA address and dst_pe is None.
        unroll_factor: int
            The unroll factor
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.push_warp"), src, dst, size, dst_pe,
                           unroll_factor)


def pull_warp(src: PrimExpr,
              dst: PrimExpr,
              size: PrimExpr,
              src_pe: Optional[PrimExpr] = None,
              unroll_factor: int = 4):
    """Pull from a remote buffer with unrolled loop.

    Args:
        src: PrimExpr
            The source address.
        dst: PrimExpr
            The destination address.
        size: PrimExpr
            The size of the pull.
        src_pe: Optional[PrimExpr]
            The PE index of the source.
            If provided, the src is a symmetric address, otherwise it is a UVA address.
            If not provided, the src is a UVA address and src_pe is None.
        unroll_factor: int
            The unroll factor
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.pull_warp"), src, dst, size, src_pe,
                           unroll_factor)
