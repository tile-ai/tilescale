"""The language interface for tl programs."""

from tilelang.primitives.gemm.base import GemmWarpPolicy
import tilelang.language as T
from tvm import tir
from typing import Union, List


def gemm(
    A: Union[tir.Buffer, tir.Var],
    B: Union[tir.Buffer, tir.Var],
    C: Union[tir.Buffer, tir.Var],
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
):
    """Perform a General Matrix Multiplication (GEMM) operation.

    This function computes C = A @ B where A and B can optionally be transposed.
    The operation supports various warp policies and accumulation modes.

    Args:
        A (Union[tir.Buffer, tir.Var]): First input matrix
        B (Union[tir.Buffer, tir.Var]): Second input matrix
        C (Union[tir.Buffer, tir.Var]): Output matrix for results
        transpose_A (bool, optional): Whether to transpose matrix A. Defaults to False.
        transpose_B (bool, optional): Whether to transpose matrix B. Defaults to False.
        policy (GemmWarpPolicy, optional): Warp execution policy. Defaults to GemmWarpPolicy.Square.
        clear_accum (bool, optional): Whether to clear accumulator before computation. Defaults to False.
        k_pack (int, optional): Number of k dimensions packed into a single warp. Defaults to 1.
        wg_wait (int, optional): Warp group wait count. Defaults to 0.

    Returns:
        tir.Call: A handle to the GEMM operation

    Raises:
        AssertionError: If the K dimensions of matrices A and B don't match
    """

    def legalize_arguments(arg: Union[tir.Buffer, tir.Var]):
        """Convert let-bound variables to their corresponding buffers.

        Args:
            arg (Union[tir.Buffer, tir.Var]): Input argument to legalize

        Returns:
            Union[tir.Buffer, tir.Var]: The legalized argument
        """
        if isinstance(arg, tir.Var) and T.has_let_value(arg):
            return T.get_let_value(arg).buffer
        return arg

    A = legalize_arguments(A)
    B = legalize_arguments(B)
    C = legalize_arguments(C)

    def retrieve_shape(object: Union[tir.Buffer, tir.BufferRegion]) -> List[int]:
        if isinstance(object, tir.Buffer):
            return object.shape
        elif isinstance(object, tir.BufferRegion):
            region = object.region
            shape = []
            for r in region:
                shape.append(r.extent)
            return shape
        else:
            raise ValueError(f"Unsupported argument type: {type(object)} for buffer {object}")

    def retrieve_stride(object: Union[tir.Buffer, tir.BufferRegion]) -> List[int]:
        if isinstance(object, tir.Buffer):
            strides = []
            stride = 1
            for s in reversed(object.shape):
                strides.insert(0, stride)
                stride *= s
            return strides
        elif isinstance(object, tir.BufferRegion):
            buffer, _ = object.buffer, object.region
            strides = []
            stride = 1
            for s in reversed(buffer.shape):
                strides.insert(0, stride)
                stride *= s
            return strides
        else:
            raise ValueError(f"Unsupported argument type: {type(object)} for buffer {object}")

    A_shape = retrieve_shape(A)
    B_shape = retrieve_shape(B)
    C_shape = retrieve_shape(C)

    A_stride = retrieve_stride(A)
    B_stride = retrieve_stride(B)

    assert len(C_shape) == 2, "current only support C as a 2D tensor"
    assert len(A_shape) >= 2, "current only support A as a 2D or higher-order tensor"
    assert len(B_shape) >= 2, "current only support B as a 2D or higher-order tensor"
    if len(A_shape) > 2:
        for i in range(len(A_shape) - 2):
            assert A_shape[i] == 1, \
                "current only support A as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"
    if len(B_shape) > 2:
        for i in range(len(B_shape) - 2):
            assert B_shape[i] == 1, \
                "current only support B as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"

    M, N = C_shape
    K = A_shape[-2] if transpose_A else A_shape[-1]
    K_B = B_shape[-1] if transpose_B else B_shape[-2]
    assert K == K_B, f"T.gemm K shape check failed: K_A = {K}, K_B = {K_B}"

    stride_a = A_stride[-2]
    stride_b = B_stride[-2]

    def retrieve_ptr(object: Union[tir.Buffer, tir.BufferRegion],
                     access_type: str = "r") -> tir.PrimExpr:
        if isinstance(object, tir.Buffer):
            return object.access_ptr(access_type)
        elif isinstance(object, tir.BufferRegion):
            buffer, region = object.buffer, object.region
            indices = []
            for r in region:
                indices.append(r.min)
            strides = []
            stride = 1
            for s in reversed(buffer.shape):
                strides.insert(0, stride)
                stride *= s
            offset = 0
            # not offset the last two dimension
            for i in range(len(indices) - 2):
                offset += indices[i] * strides[i]
            return buffer.access_ptr(access_mask=access_type, offset=offset)
        else:
            raise ValueError(f"Unsupported argument type: {type(object)} for buffer {object}")

    def retrieve_offset(object: Union[tir.Buffer, tir.BufferRegion]) -> tir.PrimExpr:
        """Retrieve the offset of the buffer or buffer region."""
        if isinstance(object, tir.Buffer):
            return [0] * len(object.shape)
        elif isinstance(object, tir.BufferRegion):
            _, region = object.buffer, object.region
            indices = []
            for r in region:
                indices.append(r.min)
            return indices
        else:
            raise ValueError(f"Unsupported argument type: {type(object)} for buffer {object}")

    A_offset = retrieve_offset(A)
    B_offset = retrieve_offset(B)
    assert A_offset[-2] == 0, "The offset of the first dimension of A must be 0"
    assert B_offset[-2] == 0, "The offset of the first dimension of B must be 0"
    offset_a = A_offset[-1]
    offset_b = B_offset[-1]

    Aptr = retrieve_ptr(A, "r")
    Bptr = retrieve_ptr(B, "r")
    Cptr = retrieve_ptr(C, "rw")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.gemm"),
        Aptr,
        Bptr,
        Cptr,
        transpose_A,
        transpose_B,
        M,
        N,
        K,
        policy,
        clear_accum,
        stride_a,
        stride_b,
        offset_a,
        offset_b,
        k_pack,
        wg_wait,
    )
