"""NCCL window registration for GIN (GPU-Initiated Networking).

GIN addresses remote memory as an ``(ncclWindow_t, byte offset)`` pair rather
than a raw pointer, because a remote rank's allocation has no local virtual
address. ``ncclCommWindowRegister`` is collective and returns one local handle
that is *not* per-peer: that handle plus a peer index names any rank's bytes.

TileScale therefore registers the whole allocator arena once, at allocator init,
instead of per tensor. Registration is collective and 4096-byte aligned, so a
per-tensor scheme would turn every ``allocate_tensor`` into a world-wide barrier.
Since the arena is symmetric, ``local_ptr - arena_base`` yields the offset valid
on every rank -- the same subtraction the intra-node path already performs.

Requires the NCCL Device API (>= 2.28.7). Everything here degrades to "no
window" when that is unavailable, so single-node paths keep working untouched.
"""

from __future__ import annotations

import ctypes
import os

import torch
import torch.distributed as dist

__all__ = [
    "NCCL_WIN_DEFAULT",
    "NCCL_WIN_COLL_SYMMETRIC",
    "NCCL_WIN_REQUIRED_ALIGNMENT",
    "DEV_COMM_STORAGE_BYTES",
    "DevComm",
    "GIN_CONTEXT_COUNT",
    "GIN_SIGNAL_COUNT",
    "GIN_COUNTER_COUNT",
    "nccl_version",
    "supports_device_api",
    "unavailable_reason",
    "get_comm_ptr",
    "register_window",
    "deregister_window",
    "create_dev_comm",
    "destroy_dev_comm",
]

NCCL_WIN_DEFAULT = 0x00
NCCL_WIN_COLL_SYMMETRIC = 0x01
# NCCL_WIN_REQUIRED_ALIGNMENT from nccl.h; both base and size must respect it.
NCCL_WIN_REQUIRED_ALIGNMENT = 4096

# The Device API (nccl_device.h, ncclDevCommCreate, GIN) first ships in 2.28.7.
# Encoded as NCCL_VERSION(x, y, z) = x * 10000 + y * 100 + z.
_MIN_DEVICE_API_VERSION = 2 * 10000 + 28 * 100 + 7

# sizeof(ncclDevComm), measured from the 2.28.9 headers on node071. The struct is
# opaque here: kernels only ever receive a pointer to it, so its size and 8-byte
# alignment are all the host side needs. Over-allocating a page is deliberate --
# a later NCCL may grow the struct, and a short buffer would be a silent
# out-of-bounds write inside ncclDevCommCreate rather than an error.
_DEV_COMM_SIZEOF = 200
DEV_COMM_STORAGE_BYTES = 4096

# GIN resource counts requested when creating the devcomm.
#
# Contexts are independent network channels; kernels rotate over them
# (blockIdx.x % count) so concurrent CTAs do not serialize on one channel.
# Signals and counters are guaranteed to start at id 0, so a kernel can address
# signal i directly for i < GIN_SIGNAL_COUNT.
GIN_CONTEXT_COUNT = 8
GIN_SIGNAL_COUNT = 32
GIN_COUNTER_COUNT = 32


class _DevCommRequirements(ctypes.Structure):
    """Mirror of ``struct ncclDevCommRequirements`` from nccl_device/core.h.

    Field order and types are taken from the 2.28.9 header; the layout was
    verified against ``offsetof``/``sizeof`` on node071 (56 bytes, 8-byte
    aligned, ``bool`` members padded to the following 4-byte field). ctypes
    reproduces that natural layout, so no explicit padding is declared.
    """

    _fields_ = [
        ("resourceRequirementsList", ctypes.c_void_p),
        ("teamRequirementsList", ctypes.c_void_p),
        ("lsaMultimem", ctypes.c_bool),
        ("barrierCount", ctypes.c_int),
        ("lsaBarrierCount", ctypes.c_int),
        ("railGinBarrierCount", ctypes.c_int),
        ("lsaLLA2ABlockCount", ctypes.c_int),
        ("lsaLLA2ASlotCount", ctypes.c_int),
        ("ginForceEnable", ctypes.c_bool),
        ("ginContextCount", ctypes.c_int),
        ("ginSignalCount", ctypes.c_int),
        ("ginCounterCount", ctypes.c_int),
    ]


# Guard against a silently different layout: ctypes computing a size other than
# the measured 56 would mean requirements land in the wrong fields, which NCCL
# would read as a garbage resource request rather than reject.
assert ctypes.sizeof(_DevCommRequirements) == 56, (
    f"ncclDevCommRequirements mirror is {ctypes.sizeof(_DevCommRequirements)} bytes, expected 56")

_lib = None
_lib_error: str | None = None


def _load_libnccl():
    """Return the libnccl already mapped into this process, or None.

    dlopen by soname returns the existing mapping, so this resolves to the same
    copy torch links against rather than loading a second one. A second copy
    would hand back handles from a different NCCL state than the communicator
    the process group owns.
    """
    global _lib, _lib_error
    if _lib is not None or _lib_error is not None:
        return _lib

    candidates = []
    override = os.environ.get("TILESCALE_NCCL_LIB")
    if override:
        candidates.append(override)
    candidates += ["libnccl.so.2", "libnccl.so"]

    errors = []
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
        except OSError as exc:
            errors.append(f"{name}: {exc}")
            continue
        _lib = lib
        return _lib

    _lib_error = "; ".join(errors)
    return None


def nccl_version() -> int | None:
    """Return the integer NCCL version, or None when libnccl is unavailable."""
    lib = _load_libnccl()
    if lib is None:
        return None
    try:
        fn = lib.ncclGetVersion
    except AttributeError:
        return None
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
    version = ctypes.c_int(0)
    if fn(ctypes.byref(version)) != 0:
        return None
    return int(version.value)


def supports_device_api() -> bool:
    """True when the loaded NCCL exposes the Device API used by GIN."""
    lib = _load_libnccl()
    if lib is None:
        return False
    version = nccl_version()
    if version is None or version < _MIN_DEVICE_API_VERSION:
        return False
    # Version alone is not proof: some builds omit the device symbols.
    return all(hasattr(lib, sym) for sym in ("ncclCommWindowRegister", "ncclDevCommCreate"))


def unavailable_reason() -> str:
    """Human-readable explanation for why windows cannot be registered."""
    lib = _load_libnccl()
    if lib is None:
        return f"libnccl could not be loaded ({_lib_error})"
    version = nccl_version()
    if version is None:
        return "ncclGetVersion failed; cannot confirm Device API support"
    if version < _MIN_DEVICE_API_VERSION:
        return (
            f"NCCL {version // 10000}.{version // 100 % 100}.{version % 100} predates the "
            "Device API; GIN needs >= 2.28.7 (set TILESCALE_NCCL_LIB to a newer libnccl)"
        )
    missing = [s for s in ("ncclCommWindowRegister", "ncclDevCommCreate") if not hasattr(lib, s)]
    if missing:
        return f"libnccl is missing Device API symbols: {', '.join(missing)}"
    return "NCCL Device API is available"


def get_comm_ptr(group: dist.ProcessGroup) -> int:
    """Return the raw ``ncclComm_t`` backing ``group``, or 0 if unobtainable.

    NCCL communicators are created lazily, so the pointer only exists after the
    group has run at least one collective on the current device. The caller is
    expected to have done so (allocator init is collective throughout).
    """
    try:
        backend = group._get_backend(torch.device("cuda", torch.cuda.current_device()))
    except Exception:  # noqa: BLE001 - non-NCCL backend or unsupported torch
        return 0

    comm_ptr = getattr(backend, "_comm_ptr", None)
    if comm_ptr is None:
        return 0
    try:
        value = comm_ptr()
    except Exception:  # noqa: BLE001 - comm not yet initialized
        return 0
    return int(value) if value else 0


def _check(lib, rc: int, what: str) -> None:
    if rc == 0:
        return
    detail = ""
    try:
        lib.ncclGetErrorString.restype = ctypes.c_char_p
        lib.ncclGetErrorString.argtypes = [ctypes.c_int]
        msg = lib.ncclGetErrorString(rc)
        if msg:
            detail = f" ({msg.decode()})"
    except Exception:  # noqa: BLE001 - error string is best effort
        pass
    raise RuntimeError(f"{what} failed: rc={rc}{detail}")


def register_window(comm_ptr: int, base_ptr: int, size: int, flags: int = NCCL_WIN_COLL_SYMMETRIC) -> int:
    """Register ``[base_ptr, base_ptr + size)`` as an NCCL window.

    Collective over the communicator: every rank must call this at the same
    point in its init sequence with a matching size. ``NCCL_WIN_COLL_SYMMETRIC``
    asserts matching layouts across ranks, and a mismatch hangs rather than
    erroring, which is why the arena registration lives in a single place.

    Returns the ``ncclWindow_t`` as an integer handle.
    """
    lib = _load_libnccl()
    if lib is None:
        raise RuntimeError(f"cannot register an NCCL window: {unavailable_reason()}")
    if not comm_ptr:
        raise ValueError("comm_ptr must be a non-null ncclComm_t")
    if base_ptr % NCCL_WIN_REQUIRED_ALIGNMENT:
        raise ValueError(
            f"window base {base_ptr:#x} is not {NCCL_WIN_REQUIRED_ALIGNMENT}-byte aligned "
            "as NCCL_WIN_REQUIRED_ALIGNMENT demands"
        )
    if size <= 0 or size % NCCL_WIN_REQUIRED_ALIGNMENT:
        raise ValueError(f"window size {size} must be positive and a multiple of {NCCL_WIN_REQUIRED_ALIGNMENT}")

    fn = lib.ncclCommWindowRegister
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
    ]
    win = ctypes.c_void_p(0)
    _check(
        lib,
        fn(ctypes.c_void_p(comm_ptr), ctypes.c_void_p(base_ptr), ctypes.c_size_t(size), ctypes.byref(win), ctypes.c_int(flags)),
        "ncclCommWindowRegister",
    )
    if not win.value:
        raise RuntimeError("ncclCommWindowRegister returned success but a null window handle")
    return int(win.value)


class DevComm:
    """A live GIN devcomm: the host struct NCCL owns, plus its device copy.

    Both halves must be kept: ``ncclDevCommDestroy`` takes the *host* pointer it
    was created with, while kernels dereference the *device* pointer published in
    the metadata table. ``device_ptr`` is only valid while this object is alive.
    """

    __slots__ = (
        "_host_buf",
        "_storage",
        "device_ptr",
        "_comm_ptr",
        "_destroyed",
        "context_count",
    )

    def __init__(self, comm_ptr: int, host_buf, storage: "torch.Tensor", context_count: int):
        self._comm_ptr = comm_ptr
        self._host_buf = host_buf
        self._storage = storage
        self.device_ptr = int(storage.data_ptr())
        # Carried so callers publish the count this devcomm was actually built
        # with. Reading the module default instead would silently lie to kernels
        # whenever a caller overrides context_count.
        self.context_count = context_count
        self._destroyed = False

    def destroy(self) -> None:
        """Release the devcomm. Collective; idempotent."""
        if self._destroyed:
            return
        self._destroyed = True
        lib = _load_libnccl()
        if lib is None or not self._comm_ptr:
            return
        fn = lib.ncclDevCommDestroy
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _check(
            lib,
            fn(ctypes.c_void_p(self._comm_ptr), ctypes.byref(self._host_buf)),
            "ncclDevCommDestroy",
        )


def create_dev_comm(
    comm_ptr: int,
    *,
    context_count: int = GIN_CONTEXT_COUNT,
    signal_count: int = GIN_SIGNAL_COUNT,
    counter_count: int = GIN_COUNTER_COUNT,
    rail_gin_barrier_count: int = 1,
) -> DevComm:
    """Create a GIN-enabled ``ncclDevComm``.

    Collective over the communicator: every rank must call this with matching
    requirements, and a mismatch hangs rather than erroring.

    ``ncclDevCommCreate`` writes the devcomm into caller-provided **host**
    storage. Measured on node071 with NCCL 2.28.9: passing a ``cudaMalloc``
    pointer segfaults inside the call, while a host struct succeeds. NCCL's own
    examples then pass the result to kernels by value in a ``__grid_constant__``
    parameter, so the struct is trivially copyable.

    TileScale instead publishes a pointer through the metadata table, so kernel
    signatures stay unchanged. That pointer is dereferenced on the device, so the
    host struct is created first and then copied into a device tensor.

    ``ginForceEnable`` is set because GIN is otherwise enabled only when NCCL
    decides the topology warrants it; TileScale needs it deterministically, since
    a kernel compiled for the GIN path cannot fall back at runtime.

    The returned :class:`DevComm` owns both buffers and must outlive every kernel
    that uses it: dropping it frees the memory its device pointer names.
    """
    lib = _load_libnccl()
    if lib is None:
        raise RuntimeError(f"cannot create an NCCL devcomm: {unavailable_reason()}")
    if not comm_ptr:
        raise ValueError("comm_ptr must be a non-null ncclComm_t")
    if signal_count <= 0 or counter_count <= 0 or context_count <= 0:
        raise ValueError("GIN context, signal, and counter counts must all be positive")

    reqs = _DevCommRequirements()
    ctypes.memset(ctypes.byref(reqs), 0, ctypes.sizeof(reqs))
    reqs.ginForceEnable = True
    reqs.ginContextCount = context_count
    reqs.ginSignalCount = signal_count
    reqs.ginCounterCount = counter_count
    reqs.railGinBarrierCount = rail_gin_barrier_count

    host_buf = (ctypes.c_uint8 * DEV_COMM_STORAGE_BYTES)()

    fn = lib.ncclDevCommCreate
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DevCommRequirements), ctypes.c_void_p]
    _check(
        lib,
        fn(ctypes.c_void_p(comm_ptr), ctypes.byref(reqs), ctypes.byref(host_buf)),
        "ncclDevCommCreate",
    )

    # uint8 so numel() == bytes. Only the struct itself is copied; the rest of the
    # padding stays zero.
    storage = torch.zeros(DEV_COMM_STORAGE_BYTES, dtype=torch.uint8, device="cuda")
    staged = torch.frombuffer(
        memoryview(host_buf)[:_DEV_COMM_SIZEOF], dtype=torch.uint8).clone()
    storage[:_DEV_COMM_SIZEOF].copy_(staged)
    torch.cuda.synchronize()
    return DevComm(comm_ptr, host_buf, storage, context_count)


def destroy_dev_comm(dev_comm: DevComm | None) -> None:
    """Release a devcomm. Collective, and must precede freeing its storage."""
    if dev_comm is not None:
        dev_comm.destroy()


def deregister_window(comm_ptr: int, window: int) -> None:
    """Release a window handle. Collective, and must precede freeing the arena."""
    lib = _load_libnccl()
    if lib is None or not window or not comm_ptr:
        return
    fn = lib.ncclCommWindowDeregister
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _check(lib, fn(ctypes.c_void_p(comm_ptr), ctypes.c_void_p(window)), "ncclCommWindowDeregister")
