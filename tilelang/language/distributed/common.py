"""The language interface for tl programs."""

from tvm import tir
from typing import Optional
from tvm.tir import PrimExpr


def get_rank():
    """Get the rank of the current process.
    """
    return tir.call_intrin("uint64", tir.op.Op.get("tl.get_rank"))


def get_num_ranks():
    """Get the number of processes.
    """
    return tir.call_intrin("uint64", tir.op.Op.get("tl.get_num_ranks"))


def remote_copy(src: PrimExpr,
                dst: PrimExpr,
                size: PrimExpr,
                dst_pe: Optional[PrimExpr] = None,
                unroll_factor: int = 4):
    """Copy between two global memory buffers with unrolled loop.

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
    return tir.call_intrin("handle", tir.op.Op.get("tl.remote_copy"), src, dst, size, dst_pe,
                           unroll_factor)
