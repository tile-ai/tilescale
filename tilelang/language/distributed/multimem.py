"""Multimem operations (NVSwitch SHARP multicast) using layout-aware lowering.

Direct operations use T.copy's ParallelOp + InferLayout + VectorizeLoop pipeline
to handle fragment layouts before emitting multimem instructions. Bulk operations
validate and lower one contiguous shared-to-multicast region directly.
"""

from __future__ import annotations
from enum import Enum
from tvm import tirx
from tvm.tirx import PrimExpr, address_of
from tilelang.utils.language import to_buffer_region


class MultimemReduceOp(Enum):
    ADD = 0
    MIN = 1
    MAX = 2
    NONE = -1  # plain store (no reduction), for multimem_tma_store


class _MultimemMode(Enum):
    LD_REDUCE = 0
    ST = 1
    RED = 2
    TMA_STORE = 3
    TMA_RED_STORE = 4


def _multimem_impl(src, dst, mode: _MultimemMode, reduce_op: MultimemReduceOp = MultimemReduceOp.NONE):
    """Shared implementation for all multimem operations.

    Converts src/dst to buffer regions and emits the tl.tileop.multimem intrinsic.

    Args:
        src: Source (Buffer, BufferLoad with slice, or BufferRegion)
        dst: Destination (Buffer, BufferLoad with slice, or BufferRegion)
        mode: 0=kLdReduce, 1=kSt, 2=kRed
        reduce_op: 0=ADD, 1=MIN, 2=MAX
    """
    src_region = to_buffer_region(src, access_type="r")
    dst_region = to_buffer_region(dst, access_type="w")
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.multimem"),
        src_region,
        dst_region,
        mode.value,
        reduce_op.value,
    )


def multimem_ld_reduce(src, dst, reduce_op: MultimemReduceOp = MultimemReduceOp.ADD):
    """Load-reduce from multicast address into local buffer.

    The direct CUDA path uses scalar, 64-bit, or 128-bit multimem instructions
    according to alignment and tail predicates. Source and destination regions
    must have matching rank, extent, and scalar dtype. Float16 and bfloat16 use
    packed x2 instructions, so their local region must cover the full fragment,
    their last extent must be even, multicast rows must be pair-aligned, and the
    fragment layout must preserve the packed lowering's contiguous-pair thread
    ownership. Direct multimem requires SM90+ and CUDA Toolkit 12.1+. The packed
    float16/bfloat16 load-reduce form requires CUDA Toolkit 12.2+ for PTX 8.2
    ``.acc::f32`` support.

    Args:
        src: Multicast source (Buffer, BufferLoad with slice, or BufferRegion)
        dst: Local destination (Buffer, BufferLoad with slice, or BufferRegion)
        reduce_op: Reduction operation. The direct CUDA path currently supports
            ADD; unsupported PTX type/operation combinations are rejected
            during lowering.
    """
    return _multimem_impl(src, dst, mode=_MultimemMode.LD_REDUCE, reduce_op=reduce_op)


def multimem_st(src, dst):
    """Store to multicast address (broadcast to all ranks).

    Region, dtype, and packed x2 constraints match ``multimem_ld_reduce``.
    The CUDA implementation requires SM90+ and CUDA Toolkit 12.1+.

    Args:
        src: Local source (Buffer, BufferLoad with slice, or BufferRegion)
        dst: Multicast destination (Buffer, BufferLoad with slice, or BufferRegion)
    """
    return _multimem_impl(src, dst, mode=_MultimemMode.ST)


def multimem_red(src, dst, reduce_op: MultimemReduceOp = MultimemReduceOp.ADD):
    """Reduce into multicast address (accumulate without read-back).

    Region, dtype, and packed x2 constraints match ``multimem_ld_reduce``.
    The CUDA implementation requires SM90+ and CUDA Toolkit 12.1+.

    Args:
        src: Local source (Buffer, BufferLoad with slice, or BufferRegion)
        dst: Multicast destination (Buffer, BufferLoad with slice, or BufferRegion)
        reduce_op: Reduction operation. The direct CUDA path currently supports
            ADD; unsupported PTX type/operation combinations are rejected
            during lowering.
    """
    return _multimem_impl(src, dst, mode=_MultimemMode.RED, reduce_op=reduce_op)


def multimem_tma_store(src, dst, reduce_op: MultimemReduceOp | None = None):
    """Async bulk TMA store from shared memory to multicast global address.

    A single thread emits one PTX instruction per call. Synchronization is
    caller-managed: every shared-memory producer must first issue
    ``T.fence_proxy_async()``, then the CTA must synchronize. One elected
    thread then issues this operation, ``T.tma_store_arrive()``, and
    ``T.tma_store_wait()``.

    Source and destination regions must have matching rank, extents, and dtype.
    Each region must be in bounds, physically contiguous, byte-addressable, and
    start at a provably 16-byte-aligned address. The positive transfer size must
    be divisible by 16 bytes and fit in an unsigned 32-bit byte count. Layout-
    remapped buffers are rejected because their physical contiguity cannot be
    inferred from the logical region.

    Plain stores support matching byte-addressable scalar or vector dtypes. Bulk
    reductions support ADD for float32, float16, and bfloat16, plus MIN/MAX for
    float16 and bfloat16. Other dtype/operation combinations are rejected during
    lowering.

    Args:
        src: Shared memory source (Buffer, BufferLoad or BufferRegion, shared scope)
        dst: Multicast global destination (Buffer, BufferLoad or BufferRegion, global scope)
        reduce_op: None for plain store (broadcast), MultimemReduceOp.ADD/MIN/MAX for reduce-accumulate

    NOTE: The current CUDA implementation requires SM90+ and CUDA toolkit 13.1+.
    Older toolkits can use an ordinary TMA store to the multicast virtual
    address where that path has been validated.
    """
    if reduce_op is None:
        return _multimem_impl(src, dst, mode=_MultimemMode.TMA_STORE)
    return _multimem_impl(src, dst, mode=_MultimemMode.TMA_RED_STORE, reduce_op=reduce_op)


def _signal_dtype_tag(addr, *, allow_signed: bool = False) -> str:
    dtype = getattr(addr, "dtype", None)
    if dtype is None:
        raise TypeError("multimem_signal requires an address expression with a dtype")
    if dtype == "uint32" or dtype == "uint32_t":
        return "uint32_t"
    if dtype == "uint64" or dtype == "uint64_t":
        return "uint64_t"
    if allow_signed and (dtype == "int32" or dtype == "int32_t"):
        return "int32_t"
    supported = "uint32/int32/uint64" if allow_signed else "uint32/uint64"
    raise TypeError(f"multimem_signal only supports {supported} signal dtypes, got {dtype}")


def multimem_signal(addr, value: PrimExpr):
    return tirx.call_extern("handle", f"tl::multimem::Signal<{_signal_dtype_tag(addr)}>::run", address_of(addr), value)


def multimem_signal_add(addr, value: PrimExpr):
    return tirx.call_extern(
        "handle",
        f"tl::multimem::SignalAdd<{_signal_dtype_tag(addr, allow_signed=True)}>::run",
        address_of(addr),
        value,
    )
