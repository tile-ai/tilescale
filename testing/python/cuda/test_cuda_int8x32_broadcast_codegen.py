import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _make_int8x32_broadcast_kernel(dtype):
    @T.prim_func
    def main(A: T.Tensor((1,), dtype), B: T.Tensor((32,), dtype)):
        with T.Kernel(1, threads=1):
            value = A[0]
            for i in T.vectorized(32):
                B[i] = value

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "dtype,torch_dtype,value,constructor",
    [
        (T.int8, torch.int8, -3, "make_longlong4("),
        (T.uint8, torch.uint8, 201, "make_ulonglong4("),
    ],
)
def test_int8x32_broadcast_uses_packed_constructor(dtype, torch_dtype, value, constructor):
    kernel = tilelang.compile(_make_int8x32_broadcast_kernel(dtype), out_idx=[1], target="cuda")
    source = kernel.get_kernel_source()

    assert constructor in source
    assert source.count("tl_pack_int8x8(") == 4

    input_value = torch.tensor([value], dtype=torch_dtype, device="cuda")
    output = kernel(input_value)
    torch.testing.assert_close(output, torch.full((32,), value, dtype=torch_dtype, device="cuda"))
