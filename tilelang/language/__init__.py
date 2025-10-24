"""The language interface for tl programs."""
from __future__ import annotations

# from .parser import *
# now is fully compatible with the upstream
# tir script
# TODO(lei): remove this import once the
# upstream tir script is fully compatible
from tvm.script.parser.tir import *
from . import overrides as _overrides  # noqa: F401
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
    StridedTensor,  # noqa: F401
    FragmentBuffer,  # noqa: F401
    SharedBuffer,  # noqa: F401
    LocalBuffer,  # noqa: F401
)
from .parallel import Parallel  # noqa: F401
from .pipeline import Pipelined  # noqa: F401
from .persistent import Persistent  # noqa: F401
from .frame import has_let_value, get_let_value  # noqa: F401
from .math_intrinsics import *  # noqa: F401
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
    alloc_tmem,  # noqa: F401
    alloc_reducer,  # noqa: F401
    alloc_descriptor,  # noqa: F401
)
from .copy import copy, c2d_im2col  # noqa: F401
from .gemm import GemmWarpPolicy, gemm, gemm_v2  # noqa: F401
from .experimental.gemm_sp import gemm_sp  # noqa: F401
from .fill import fill, clear  # noqa: F401
from .reduce import (
    reduce,  # noqa: F401
    reduce_max,  # noqa: F401
    reduce_min,  # noqa: F401
    reduce_sum,  # noqa: F401
    reduce_abssum,  # noqa: F401
    reduce_absmax,  # noqa: F401
    reduce_bitand,  # noqa: F401
    reduce_bitor,  # noqa: F401
    reduce_bitxor,  # noqa: F401
    cumsum,  # noqa: F401
    finalize_reducer,  # noqa: F401
)
from .print import print  # noqa: F401
from .customize import (
    atomic_max,  # noqa: F401
    atomic_min,  # noqa: F401
    atomic_add,  # noqa: F401
    atomic_addx2,  # noqa: F401
    atomic_addx4,  # noqa: F401
    dp4a,  # noqa: F401
    clamp,  # noqa: F401
    reshape,  # noqa: F401
    view,  # noqa: F401
    atomic_load,  # noqa: F401
    atomic_store,  # noqa: F401
    loop_break,  # noqa: F401
)
from .logical import any_of, all_of  # noqa: F401
from .builtin import *  # noqa: F401
from .distributed.multi_device.nvshmem import *  # noqa: F401
from .distributed.multi_device.cpengine import *  # noqa: F401
from .distributed.common import *  # noqa: F401

from .utils import index_to_coordinates  # noqa: F401

from .symbolics import dynamic, symbolic  # noqa: F401
from .annotations import (  # noqa: F401
    use_swizzle, annotate_layout, annotate_safe_value, annotate_l2_hit_ratio,
)


def import_source(source: str | None = None):
    # source is the source code to be imported
    return block_attr({"pragma_import_c": source}) if source is not None else None
