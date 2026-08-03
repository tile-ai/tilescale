"""Inter-node communication primitives backed by the NCCL GIN device API.

These ops address memory as ``(window, offset)`` rather than as a peer pointer,
which is what makes them work across nodes: a remote node's allocation has no
local virtual address, so the intra-node ``peer_base + offset`` arithmetic in
:mod:`comm` cannot name it. TileScale registers the whole allocator arena as one
NCCL window, so a local arena pointer converts to a remote address by
subtracting the arena base -- the same subtraction the intra-node path performs,
paired with a window handle instead of a peer base.

Both buffers must live in the allocator arena. A local ``T.alloc_shared`` or
``T.alloc_fragment`` buffer is not in a registered window and cannot be a GIN
source or destination.

``peer`` is a **global** rank, since GIN puts are issued against the
communicator-wide team. Passing a peer on the local node is allowed and works,
but it routes through the network stack rather than NVLink; prefer
:func:`~tilelang.language.distributed.comm.put_block` for intra-node traffic.

Requires NCCL >= 2.28.7 and a devcomm published by the allocator. When GIN was
not compiled in, the generated kernel will not contain these calls at all --
build-time detection gates the device header.
"""

from __future__ import annotations

from tvm import tirx
from tvm.tirx import PrimExpr, IntImm, address_of

# GIN 2.28.9 exposes no remote-read operation: the device class has put,
# putValue, signal, and the signal/counter waits, but nothing that pulls bytes
# from a peer. A `get` therefore cannot be one-sided the way intra-node
# `get_block` is -- it needs the data's owner to issue a put. Expressing that as
# a `get` would hide a required remote-side call behind a local-looking op, so it
# is deliberately absent. Model a pull as the owner putting plus a signal.

__all__ = [
    "put",
    "put_signal",
    "signal",
    "wait_signal",
    "flush",
]


def _coop(scope: str) -> str:
    """Validate a cooperation scope and return it.

    The scope becomes an ``ncclCoop*`` template argument, so an unknown value
    would surface as an nvcc template error in generated code rather than here.
    """
    if scope not in ("thread", "warp", "block"):
        raise ValueError(f"scope must be one of 'thread', 'warp', or 'block', got {scope!r}")
    return scope


def put(
    src: PrimExpr,
    dst: PrimExpr,
    size: PrimExpr,
    peer: PrimExpr | IntImm,
    scope: str = "block",
):
    """Write ``size`` elements from local ``src`` into ``dst`` on ``peer``.

    ``size`` counts elements, not bytes, matching
    :func:`~tilelang.language.distributed.comm.put_block`. ``src`` and ``dst``
    must have the same dtype.

    One-sided and asynchronous: the call returns before the data has landed.
    Nothing tells the peer the write happened -- pair with :func:`signal`, or use
    :func:`put_signal` to fuse the notification into the put. Reusing ``src``
    requires a :func:`flush` first.

    ``dst`` is indexed with the *peer's* view of the buffer, which is the same
    index the local rank would use because the arena is symmetric.
    """
    # Validate before touching the buffers so a bad scope reports the scope
    # rather than whatever address_of makes of the arguments.
    coop = _coop(scope)
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.gin_put"),
        address_of(src),
        address_of(dst),
        size,
        peer,
        0,  # signal id, unused without a remote action
        0,  # no remote signal
        coop,
    )


def put_signal(
    src: PrimExpr,
    dst: PrimExpr,
    size: PrimExpr,
    peer: PrimExpr | IntImm,
    signal_id: int = 0,
    scope: str = "block",
):
    """Like :func:`put`, and increment ``signal_id`` on ``peer`` once it lands.

    The increment is ordered after this put's payload and after any preceding
    puts to the same peer on the same context, so a peer released by
    :func:`wait_signal` is guaranteed to observe the bytes. This ordering is why
    a fused put+signal is preferred over a separate :func:`signal`.
    """
    coop = _coop(scope)
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.gin_put"),
        address_of(src),
        address_of(dst),
        size,
        peer,
        int(signal_id),
        1,  # increment signal on arrival
        coop,
    )


def signal(peer: PrimExpr | IntImm, signal_id: int = 0, scope: str = "block"):
    """Increment ``signal_id`` on ``peer`` without moving payload.

    Ordered after this context's preceding puts to that peer, so it can act as a
    completion marker for a batch of :func:`put` calls.
    """
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.gin_signal"),
        peer,
        int(signal_id),
        _coop(scope),
    )


def wait_signal(least: PrimExpr, signal_id: int = 0, scope: str = "block"):
    """Block until ``signal_id`` has been incremented at least ``least`` times.

    Signals are cumulative running totals compared with rolling arithmetic, not
    flags -- they are not consumed by a wait. A kernel that waits repeatedly
    tracks an increasing expected total rather than resetting between phases.
    """
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.gin_wait_signal"),
        least,
        int(signal_id),
        _coop(scope),
    )


def flush(scope: str = "block"):
    """Wait until this coop's put source buffers are safe to overwrite.

    This says nothing about remote visibility; only a signal does. Use it before
    rewriting a send buffer, not to establish that a peer can read the data.
    """
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.gin_flush"),
        _coop(scope),
    )
