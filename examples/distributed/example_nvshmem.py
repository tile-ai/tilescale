# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T

import tvm


@tvm.register_func("tilelang_callback_cuda_postproc", override=True)
def tilelang_callback_cuda_postproc(code, _):
    # from code_replace import rc
    # return rc
    return code


def dist_test(M, N, block_M, block_N, dtype="float16"):

    @T.prim_func
    def main(
            A: T.Buffer((M, N), dtype),
            B: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(A_shared, B[by * block_M, bx * block_N])

    return main


func = dist_test(1024, 1024, 128, 128)

kernel = tilelang.compile(func, out_idx=-1)

# Get CUDA Source
print(kernel.get_kernel_source())

profiler = kernel.get_profiler()
out = profiler.run_once()

print(out)
