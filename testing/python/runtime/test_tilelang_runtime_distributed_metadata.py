import ctypes

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _plain_kernel():
    @T.prim_func
    def main(output: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=1):
            output[0] = 7

    return main


def _rank_kernel():
    @T.prim_func
    def main(output: T.Tensor((1,), T.int64)):
        with T.Kernel(1, threads=1):
            output[0] = T.Cast(T.int64, T.get_rank())

    return main


def _init_cython_table(kernel, table) -> int:
    return kernel.adapter.lib.init_table(
        ctypes.c_void_p(ctypes.addressof(table)),
        len(table),
        ctypes.c_void_p(0),
    )


@tilelang.testing.requires_cuda
def test_plain_cython_kernel_does_not_require_distributed_metadata():
    kernel = tilelang.compile(_plain_kernel(), execution_backend="cython")
    source = kernel.get_kernel_source()

    assert "tl_templates/cuda/distributed/" not in source
    assert "meta_data" not in source

    assert kernel.adapter.lib.init_table(ctypes.c_void_p(), 0, ctypes.c_void_p()) == 0
    table = (ctypes.c_uint64 * 2)(0, 1)
    assert _init_cython_table(kernel, table) == 0

    output = torch.empty((1,), dtype=torch.int32, device="cuda")
    kernel(output)
    assert output.item() == 7


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("execution_backend", ["cython", "tvm_ffi"])
def test_distributed_metadata_symbol_can_be_initialized(execution_backend):
    kernel = tilelang.compile(_rank_kernel(), execution_backend=execution_backend)
    source = kernel.get_kernel_source()
    assert "tl_templates/cuda/distributed/distributed.h" in source

    expected_rank = 17
    table = (ctypes.c_uint64 * 3)(expected_rank, 23, 0)
    if execution_backend == "cython":
        result = _init_cython_table(kernel, table)
    else:
        result = kernel.adapter.init_table(ctypes.addressof(table), len(table), 0)
    assert result == 0

    output = torch.empty((1,), dtype=torch.int64, device="cuda")
    kernel(output)
    assert output.item() == expected_rank


if __name__ == "__main__":
    tilelang.testing.main()
