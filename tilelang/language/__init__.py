"""The language interface for tl programs."""

from typing import Optional, Callable, Dict
# from .parser import *
# now is fully compatible with the upstream
# tir script
# TODO(lei): remove this import once the
# upstream tir script is fully compatible
from tvm.script.parser.tir import *
from .tir import (
    prim_func,  # noqa: F401
)
from .tir.ir import *  # noqa: F401
from tilelang.layout import Layout, Fragment  # noqa: F401
from .proxy import (
    ptr,  # noqa: F401
    make_tensor,  # noqa: F401
    Buffer,  # noqa: F401
    Tensor,  # noqa: F401
    FragmentBuffer,  # noqa: F401
    SharedBuffer,  # noqa: F401
    LocalBuffer,  # noqa: F401
)
from .parallel import Parallel  # noqa: F401
from .pipeline import Pipelined  # noqa: F401
from .persistent import Persistent  # noqa: F401
from .frame import has_let_value, get_let_value  # noqa: F401
from .kernel import (
    Kernel,  # noqa: F401
    KernelLaunchFrame,  # noqa: F401
    get_thread_binding,  # noqa: F401
    get_thread_bindings,  # noqa: F401
    get_block_binding,  # noqa: F401
    get_block_bindings,  # noqa: F401
)
from .warpgroup import ws  # noqa: F401
from .allocate import (
    alloc_var,  # noqa: F401
    alloc_local,  # noqa: F401
    alloc_shared,  # noqa: F401
    alloc_fragment,  # noqa: F401
    alloc_barrier,  # noqa: F401
)
from .copy import copy, c2d_im2col  # noqa: F401
from .gemm import GemmWarpPolicy, gemm  # noqa: F401
from .experimental.gemm_sp import gemm_sp  # noqa: F401
from .fill import fill, clear  # noqa: F401
from .reduce import (
    reduce,  # noqa: F401
    reduce_max,  # noqa: F401
    reduce_min,  # noqa: F401
    reduce_sum,  # noqa: F401
    reduce_abssum,  # noqa: F401
    reduce_absmax,  # noqa: F401
    cumsum,  # noqa: F401
)
from .print import print  # noqa: F401
from .customize import (
    atomic_add,  # noqa: F401
    atomic_addx2,  # noqa: F401
    atomic_addx4,  # noqa: F401
    dp4a,  # noqa: F401
    clamp,  # noqa: F401
    reshape,  # noqa: F401
    view,  # noqa: F401
)
from .logical import any_of, all_of  # noqa: F401
from .builtin import *  # noqa: F401
from .distributed.multi_device.nvshmem import *  # noqa: F401
from .distributed.multi_device.cpengine import *  # noqa: F401
from .distributed.common import *  # noqa: F401

from .memscope import *  # noqa: F401


def symbolic(name: str, dtype: str = "int32"):
    return tir.Var(name, dtype)


def use_swizzle(panel_size: int, order: str = "row", offset: int = 0, enable: bool = True):
    # If order is row, use rasterization2DRow, otherwise use rasterization2DColumn
    # The panel size is the number of threads in a warp
    # Use to improve the L2 Cache Locality
    device_func = ("rasterization2DRow" if order == "row" else "rasterization2DColumn")
    return attr(None, "threadblock_swizzle_pattern",
                f"tl::{device_func}<{panel_size}, {offset}>") if enable else None


def annotate_layout(layout_map: Dict):
    """Annotate the layout of the buffer

    Args:
        layout_map (Dict): a dictionary of buffer to layout

    Returns:
        block_attr: a block attribute
    
    Example:
        @T.prim_func
        def main(
                A: T.Tensor((M, N), dtype),
                B: T.Tensor((M, N), dtype),
        ):
            # Initialize Kernel Context
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_N), dtype)

                T.annotate_layout({A_shared: layout})
                for i, j in T.Parallel(block_M, block_N):
                    A_shared[i, j] = A[by * block_M + i, bx * block_N + j]

                for i, j in T.Parallel(block_M, block_N):
                    B[by * block_M + i, bx * block_N + j] = A_shared[i, j]

        return main
    """
    # layout_map is a dictionary of buffer to layout
    _layout_map = {}
    for buffer, layout in layout_map.items():
        if isinstance(layout, Layout):
            _layout_map[buffer.data] = layout
        elif isinstance(layout, Callable):
            _layout_map[buffer.data] = Layout(buffer.shape, layout)
        else:
            raise ValueError(f"Invalid layout: {layout}")

    return block_attr({"layout_map": _layout_map})


def annotate_padding(padding_map: Dict):
    """Annotate the padding of the buffer

    Args:
        padding_map (dict): a dictionary of buffer to padding value

    Returns:
        block_attr: a block attribute
    
    Example:
        @T.prim_func
        def main(
                A: T.Tensor((M, N), dtype),
                B: T.Tensor((M, N), dtype),
        ):
            # Initialize Kernel Context
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_N), dtype)

                T.annotate_padding({A_shared: pad_value})
                for i, j in T.Parallel(block_M, block_N):
                    A_shared[i, j] = A[by * block_M + i - 10, bx * block_N + j]

                for i, j in T.Parallel(block_M, block_N):
                    B[by * block_M + i, bx * block_N + j] = A_shared[i, j]

        return main
    """
    # padding_map is a dictionary of buffer to padding value
    _padding_map = {}
    for buffer, padding_value in padding_map.items():
        # assert not global
        assert buffer.scope() != "global", "padding can not be applied to global buffers"
        _padding_map[buffer.data] = padding_value
    return block_attr({"padding_map": _padding_map})


def annotate_l2_hit_ratio(l2_hit_ratio_map: Dict):
    """Annotate the L2 hit ratio of the buffer, detailed explanation please refer to:
    https://docs.nvidia.com/cuda/cuda-c-programming-guide/#l2-policy-for-persisting-accesses

    Args:
        l2_hit_ratio_map (dict): a dictionary of buffer to L2 hit ratio value
    Example:
        # 0.5 is the hit ratio
        T.annotate_l2_hit_ratio({A: 0.5})
    """
    _l2_hit_ratio_map = {}
    for buffer, hit_ratio in l2_hit_ratio_map.items():
        assert buffer.scope() == "global", "persistent L2 can only be applied to global buffers"
        _l2_hit_ratio_map[buffer.data] = float(hit_ratio)
    return block_attr({"l2_hit_ratio_map": _l2_hit_ratio_map})


def import_source(source: Optional[str] = None):
    # source is the source code to be imported
    return block_attr({"pragma_import_c": source}) if source is not None else None
