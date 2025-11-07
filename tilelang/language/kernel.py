"""The language interface for tl programs."""
from __future__ import annotations

from collections import deque
from tvm import tir
from tvm.tir import Var
from tvm.script.ir_builder.tir.frame import TIRFrame, BlockFrame
from tvm.ffi import register_object
from tilelang import _ffi_api
import threading
from typing import List, Tuple, Optional

# Ensure single-dimension kernel bindings can be unpacked like iterables.
# especially for issue https://github.com/tile-ai/tilelang/issues/830
if not hasattr(Var, "__iter__"):

    def _var_iter(self):
        yield self

    Var.__iter__ = _var_iter  # type: ignore[attr-defined]

if not hasattr(Var, "__len__"):
    Var.__len__ = lambda self: 1  # type: ignore[attr-defined]


class FrameStack:
    """
    A simple stack-like wrapper around a deque that provides
    push, pop, and top methods for convenience.
    """

    def __init__(self):
        self._stack = deque()

    def push(self, item):
        """Pushes an item onto the top of the stack."""
        self._stack.append(item)

    def pop(self):
        """
        Pops and returns the top of the stack, or returns None
        if the stack is empty.
        """
        if self._stack:
            return self._stack.pop()
        raise IndexError(f"{self.__class__.__name__} is empty")

    def top(self):
        """
        Returns the item on the top of the stack without removing it,
        or None if the stack is empty.
        """
        if self._stack:
            return self._stack[-1]
        raise IndexError(f"{self.__class__.__name__} is empty")

    def size(self):
        """Returns the number of items in the stack."""
        return len(self._stack)

    def __len__(self):
        """Returns the number of items in the stack."""
        return len(self._stack)

    def __bool__(self):
        """
        Allows truthy checks on the stack object itself,
        e.g., 'if stack: ...'
        """
        return bool(self._stack)


# Use thread local to store the stack
# This is to avoid the cross-thread interference
_local = threading.local()


def _get_current_stack() -> FrameStack:
    if not hasattr(_local, "kernel_launch_frame_stack"):
        _local.kernel_launch_frame_stack = FrameStack()
    return _local.kernel_launch_frame_stack


def _normalize_bindings(bindings: list[Var]) -> Var | list[Var]:
    """
    Return a bare Var when we only have a single binding so that users may write either
    `with T.Kernel(...) as pid:` or `with T.Kernel(...) as (pid,)`.
    Otherwise, keep the list semantics for multi-dimensional launches.
    """
    if len(bindings) == 1:
        return bindings[0]
    return bindings


@register_object("tl.KernelLaunchFrame")
class KernelLaunchFrame(TIRFrame):
    """
    KernelLaunchFrame is a custom TIRFrame that manages block/thread indices
    and handles the entry and exit of the kernel launch scope.
    """

    def __enter__(self) -> Var | list[Var]:
        """
        Enters the KernelLaunchFrame scope and pushes this frame onto the stack.
        Returns one Var if we detect exactly 5 frames (meaning there is a single
        block dimension), or a list of Vars otherwise.
        """
        super().__enter__()
        _get_current_stack().push(self)

        last_block_frame = self.frames[-1]
        assert isinstance(last_block_frame,
                          BlockFrame), f"Last frame must be a block frame, got {last_block_frame}"

        maybe_cpu = last_block_frame.annotations.get("tilelang.is_cpu_kernel_frame", False)

        if maybe_cpu:
            # CPU kernel frame, return a list of for frame items.
            return _normalize_bindings([frame.vars[0] for frame in self.frames[0:-1]])
        else:
            # Otherwise, return blockIdx.* bindings only (exclude cluster and thread frames).
            binds: list[Var] = []
            for fr in self.frames[0:-4]:
                iv = getattr(fr, "iter_var", None)
                if iv is None:
                    continue
                tag = getattr(iv, "thread_tag", None)
                if isinstance(tag, str) and tag.startswith("blockIdx."):
                    binds.append(iv.var)
            return _normalize_bindings(binds)

    def __exit__(self, ptype, value, trace):
        """
        Exits the KernelLaunchFrame scope and pops this frame from the stack,
        but only if it's indeed the topmost frame.
        """
        stack = _get_current_stack()
        if stack.top() is self:
            stack.pop()
        super().__exit__(ptype, value, trace)

    @classmethod
    def Current(cls) -> KernelLaunchFrame | None:
        """
        Returns the topmost (current) KernelLaunchFrame from the stack if it exists,
        or None if the stack is empty.
        """
        stack = _get_current_stack()
        return stack.top() if stack else None

    def _collect_block_iters(self) -> list:
        frames = []
        for fr in self.frames[:-4]:  # exclude thread dims and attr block
            iv = getattr(fr, "iter_var", None)
            if iv is None:
                continue
            tag = getattr(iv, "thread_tag", None)
            if isinstance(tag, str) and tag.startswith("blockIdx."):
                frames.append(iv)
        return frames

    def get_block_extent(self, dim: int) -> int:
        """
        Returns the block extent for the given dimension.
        dim=0 corresponds to blockIdx.x, dim=1 to blockIdx.y, and dim=2 to blockIdx.z.
        """
        iters = self._collect_block_iters()
        return int(iters[dim].dom.extent)

    def get_block_extents(self) -> list[int]:
        """
        Returns the block extents for all three dimensions.
        """
        return [self.get_block_extent(dim) for dim in range(3)]

    def get_thread_extent(self, dim: int) -> int:
        """
        Returns the thread extent for the given dimension.
        dim=0 corresponds to threadIdx.x, dim=1 to threadIdx.y, and dim=2 to threadIdx.z.
        """
        iter_var = self.frames[-4 + dim].iter_var
        return int(iter_var.dom.extent)

    def get_thread_extents(self) -> list[int]:
        """
        Returns the thread extents for all three dimensions.
        """
        return [self.get_thread_extent(dim) for dim in range(3)]

    def get_thread_binding(self, dim: int = 0) -> Var:
        """
        Returns the thread binding for the given dimension.
        dim=0 corresponds to threadIdx.x, dim=1 to threadIdx.y, and dim=2 to threadIdx.z.
        """
        return self.frames[-4 + dim].iter_var.var

    def get_thread_bindings(self) -> list[Var]:
        """
        Returns the thread binding for the given dimension.
        dim=0 corresponds to threadIdx.x, dim=1 to threadIdx.y, and dim=2 to threadIdx.z.
        """
        return [frame.iter_var.var for frame in self.frames[-4:-1]]

    def get_num_threads(self) -> int:
        """
        Returns the thread indices from the topmost frame.
        """
        num_threads: int = 1
        for thread_dim in range(3):
            num_threads *= self.get_thread_extent(thread_dim)
        return num_threads

    def get_block_binding(self, dim: int = 0) -> Var:
        """
        Returns the block binding for the given dimension.
        dim=0 corresponds to blockIdx.x, dim=1 to blockIdx.y, and dim=2 to blockIdx.z.
        """
        iters = self._collect_block_iters()
        return iters[dim].var

    def get_block_bindings(self) -> list[Var]:
        """
        Returns all three block bindings.
        """
        return [iv.var for iv in self._collect_block_iters()]

    @property
    def blocks(self) -> list[Var]:
        """
        Returns the block indices from the topmost frame.
        """
        return [iv.var for iv in self._collect_block_iters()]

    @property
    def threads(self) -> list[Var]:
        """
        Returns the thread indices from the topmost frame.
        """
        return [frame.iter_var.var for frame in self.frames[-4:]]

    @property
    def num_threads(self) -> int:
        """
        Returns the total number of threads.
        """
        return self.get_num_threads()


def Kernel(
    *blocks: list[tir.PrimExpr],
    threads: int | list[int] | tuple | None = None,
    is_cpu: bool = False,
    prelude: str | None = None,
):
    """Tools to quickly construct a GPU kernel launch frame.

    Parameters
    ----------
    blocks : List[int]
        A list of extent, can be 1-3 dimension, representing gridDim.(x|y|z)
    threads : int
        A integer representing blockDim.x
        Or a list of integers representing blockDim.(x|y|z)
        if the value is -1, we skip the threadIdx.x binding.
    is_cpu : bool
        Whether the kernel is running on CPU.
        Thus we will not bind threadIdx.x, threadIdx.y, threadIdx.z.
        and blockIdx.x, blockIdx.y, blockIdx.z.
    prelude : str
        The import c code of the kernel,
        will be injected before the generated kernel code.

    Returns
    -------
    res : Tuple[frame.LaunchThreadFrame]
        The result LaunchThreadFrame.

    Examples
    --------
    Create a 1-D CUDA kernel launch and unpack the single block index:

    .. code-block:: python

        with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
            # bx is the blockIdx.x binding (also iterable as (bx,))
            ...

    Launch a 2-D grid while requesting two thread dimensions:

    .. code-block:: python

        with T.Kernel(grid_x, grid_y, threads=(64, 2)) as (bx, by):
            tx, ty = T.get_thread_bindings()
            ...

    Emit a CPU kernel where thread bindings are skipped:

    .. code-block:: python

        with T.Kernel(loop_extent, is_cpu=True) as (i,):
            ...
    """
    attrs: dict = {}

    if not is_cpu and threads is None:
        threads = 128  # default thread number

    if isinstance(threads, int):
        threads = [threads, 1, 1]
    elif isinstance(threads, list):
        threads = threads + [1] * (3 - len(threads))
    elif isinstance(threads, tuple):
        threads = list(threads) + [1] * (3 - len(threads))
    else:
        assert is_cpu, "threads must be an integer or a list of integers"

    if is_cpu:
        attrs["tilelang.is_cpu_kernel_frame"] = True

    if prelude is not None:
        attrs["pragma_import_c"] = prelude

    return _ffi_api.KernelLaunch(blocks, threads, attrs)


def ScopeKernel(
    *,
    grid: Tuple | List,
    cluster: Optional[Tuple | List] = None,
    threads: int | List[int] | Tuple | None = None,
    is_cpu: bool = False,
    prelude: str | None = None,
):
    """Launch a kernel with explicit logical grid and cluster shapes.

    This mimics T.Kernel, but accepts a `grid` and a `cluster` shape.
    Under the hood it launches blocks = grid[i] * cluster[i].
    """
    attrs: dict = {}

    if not is_cpu and threads is None:
        threads = 128

    # Normalize threads to 3D if provided
    if isinstance(threads, int):
        threads = [threads, 1, 1]
    elif isinstance(threads, list):
        threads = threads + [1] * (3 - len(threads))
    elif isinstance(threads, tuple):
        threads = list(threads) + [1] * (3 - len(threads))
    else:
        assert is_cpu, "threads must be an integer or a list/tuple of integers"

    if is_cpu:
        attrs["tilelang.is_cpu_kernel_frame"] = True

    if prelude is not None:
        attrs["pragma_import_c"] = prelude

    # Normalize grid/cluster to up to 3 dims
    def _norm_dims(x, fill=1):
        if x is None:
            return [fill, fill, fill]
        if isinstance(x, (int, tir.PrimExpr)):
            return [x, fill, fill]
        if isinstance(x, tuple):
            x = list(x)
        if isinstance(x, list):
            return x + [fill] * (3 - len(x))
        raise TypeError("grid/cluster must be int, list or tuple")

    grid3 = _norm_dims(grid)
    cluster_opt = None if cluster is None else _norm_dims(cluster, 1)

    return _ffi_api.ScopeKernelLaunch(grid3, cluster_opt, threads, attrs)


def get_thread_binding(dim: int = 0) -> Var:
    """Returns the thread binding for the given dimension.
    """
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_thread_binding(dim)


def get_thread_bindings() -> list[Var]:
    """Returns all three thread bindings.
    """
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_thread_bindings()


def get_block_binding(dim: int = 0) -> Var:
    """Returns the block binding for the given dimension.
    """
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_block_binding(dim)


def get_block_bindings() -> list[Var]:
    """Returns all three block bindings.
    """
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_block_bindings()


def _current_cluster_shape_or_none():
    cur = KernelLaunchFrame.Current()
    if cur is None:
        return None
    try:
        last_block_frame = cur.frames[-1]
        if not isinstance(last_block_frame, BlockFrame):
            return None
        cluster = last_block_frame.annotations.get("tilelang.cluster_shape")
        return cluster
    except Exception:
        return None


def _collect_tag_bindings(prefix: str) -> list[Var]:
    cur = KernelLaunchFrame.Current()
    assert cur is not None, "KernelLaunchFrame is not initialized"
    order = {"x": 0, "y": 1, "z": 2}
    slots: list[Optional[Var]] = [None, None, None]
    for fr in cur.frames[:-1]:  # skip trailing attribute block
        iv = getattr(fr, "iter_var", None)
        if iv is None:
            continue
        tag = getattr(iv, "thread_tag", None)
        if not isinstance(tag, str):
            continue
        if tag.startswith(prefix):
            axis = tag.split(".")[-1]
            idx = order.get(axis)
            if idx is not None:
                slots[idx] = iv.var
    return [v for v in slots if v is not None]


def _collect_tag_extents(prefix: str) -> list:
    cur = KernelLaunchFrame.Current()
    assert cur is not None, "KernelLaunchFrame is not initialized"
    order = {"x": 0, "y": 1, "z": 2}
    slots: list[Optional[int]] = [None, None, None]
    for fr in cur.frames[:-1]:
        iv = getattr(fr, "iter_var", None)
        if iv is None:
            continue
        tag = getattr(iv, "thread_tag", None)
        if not isinstance(tag, str):
            continue
        if tag.startswith(prefix):
            axis = tag.split(".")[-1]
            idx = order.get(axis)
            if idx is not None:
                slots[idx] = int(iv.dom.extent)
    return [e for e in slots if e is not None]


def get_cluster_binding(dim: Optional[int] = None):
    """
    Returns the cluster-local binding for the given dimension.
    This equals blockIdx.{dim} % cluster[{dim}] inside ScopeKernel.
    """
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    binds = _collect_tag_bindings("clusterIdx.")
    assert len(binds) > 0, "get_cluster_binding must be used inside T.ScopeKernel with cluster dims"
    if dim is None:
        return binds
    return binds[dim]


def get_cluster_bindings() -> List:
    """Returns all three cluster-local bindings (see get_cluster_binding)."""
    return [get_cluster_binding(i) for i in range(3)]


def get_grid_extent(dim: int = 0):
    """Returns the logical grid extent for the given dimension (inside ScopeKernel)."""
    cur = KernelLaunchFrame.Current()
    assert cur is not None, "KernelLaunchFrame is not initialized"
    # Prefer annotated grid_shape if available
    try:
        last_block_frame = cur.frames[-1]
        if isinstance(last_block_frame, BlockFrame):
            grid_shape = last_block_frame.annotations.get("tilelang.grid_shape")
            if grid_shape is not None:
                return grid_shape[dim]
    except Exception:
        pass
    # Fallback: try to read extents by tag
    ext = _collect_tag_extents("blockIdx.")
    if dim < len(ext):
        return ext[dim]
    return cur.get_block_extent(dim)


def get_grid_extents() -> List:
    """Returns all three logical grid extents (inside ScopeKernel)."""
    return [get_grid_extent(i) for i in range(3)]


def get_cluster_extent(dim: int = 0):
    """Returns the cluster extent for the given dimension (inside ScopeKernel)."""
    ext = _collect_tag_extents("clusterIdx.")
    assert len(ext) > 0, "get_cluster_extent must be used inside T.ScopeKernel"
    return ext[dim]


def get_cluster_extents() -> List:
    """Returns all three cluster extents (inside ScopeKernel)."""
    cluster = _current_cluster_shape_or_none()
    assert cluster is not None, "get_cluster_extents must be used inside T.ScopeKernel"
    return [cluster[i] for i in range(3)]


def get_thread_extent(dim: int = 0) -> int:
    """Returns the thread extent for the given dimension.
    """
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_thread_extent(dim)


def get_thread_extents() -> list[int]:
    """Returns all three thread extents.
    """
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_thread_extents()


def get_block_extent(dim: int = 0) -> int:
    """Returns the block extent for the given dimension.
    """
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_block_extent(dim)


def get_block_extents() -> list[int]:
    """Returns all three block extents.
    """
    assert KernelLaunchFrame.Current() is not None, "KernelLaunchFrame is not initialized"
    return KernelLaunchFrame.Current().get_block_extents()
