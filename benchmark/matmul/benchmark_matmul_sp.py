import argparse
import itertools
import logging
import torch
from triton.testing import do_bench

import tilelang.language as T
from tilelang.autotuner import autotune
from tilelang import jit
from tilelang.layout import make_metadata_layout
# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def ref_program(A, B):
    """
    A reference matrix multiplication program, used to compare performance.

    Parameters
    ----------
    A : numpy.ndarray
        The matrix with shape (M, K).
    B : numpy.ndarray
        The matrix with shape (N, K).

    Returns
    -------
    np.ndarray
        The result of A @ B.T, shape (M, N).
    """
    return A @ B.T


def get_configs(M, N, K):
    """
    Generate a list of configuration dictionaries that will be used for tuning.
    
    Parameters
    ----------
    with_roller : bool
        Whether to enable bitblas roller to deduce search spaces

    Returns
    -------
    list of dict
        Each configuration dict includes various block sizes, pipeline stages,
        thread numbers, and other parameters to explore during autotuning.
    """
    block_M = [64, 128, 256]
    block_N = [64, 128, 256]
    block_K = [64, 128]
    num_stages = [0, 1, 2, 3]
    thread_num = [128, 256]
    enable_rasterization = [True, False]
    policy = [T.GemmWarpPolicy.Square]
    _configs = list(
        itertools.product(
            block_M,
            block_N,
            block_K,
            num_stages,
            thread_num,
            policy,
            enable_rasterization,
        ))

    configs = [
        {
            "block_M": c[0],
            "block_N": c[1],
            "block_K": c[2],
            "num_stages": c[3],
            "thread_num": c[4],
            "policy": c[5],
            "enable_rasterization": c[6],  # keep param name for backward-compat
        } for c in _configs
    ]
    return configs


def matmul_sp(M, N, K):
    """
    Create an autotuned matrix multiplication kernel for matrices of shape:
      - A: (M, K)
      - B: (N, K)
      - C: (M, N)

    Parameters
    ----------
    M : int
        The dimension M of the matrix multiplication.
    N : int
        The dimension N of the matrix multiplication.
    K : int
        The dimension K of the matrix multiplication.

    Returns
    -------
    (best_latency, best_config, ref_latency)
        best_latency : float
            The best latency found among the tuned configurations.
        best_config : dict
            The parameter configuration that yielded best_latency.
        ref_latency : float
            The baseline latency of the reference program (for computing speedup).
    """

    # Decorate the kernel with autotune & jit, specifying:
    #  - Tuning config list
    #  - Profiling keys
    #  - Warmup and repetition counts for better measurement
    #  - A reference program for correctness verification
    #  - The "tvm" profiler backend
    #  - HIP as the compilation target (modify as needed for your hardware)

    @autotune(
        configs=get_configs(M, N, K),
        warmup=3,
        rep=20,
    )
    @jit(out_idx=[2],)
    def kernel(
        block_M=None,
        block_N=None,
        block_K=None,
        num_stages=None,
        thread_num=None,
        policy=None,
        enable_rasterization=None,
    ):
        """
        The actual kernel to compute C = A @ B^T.

        Parameters
        ----------
        block_M : int
            Block size in M dimension.
        block_N : int
            Block size in N dimension.
        block_K : int
            Block size in K dimension.
        num_stages : int
            Number of pipelined stages (for asynchronous load).
        thread_num : int
            Number of threads to use per block.
        k_pack : int
            K dimension packing factor to improve memory coalescing.

        Returns
        -------
        Function
            A TVM Tensor Language function (T.prim_func) that computes matmul.
        """
        # Use half-precision for input data to reduce memory bandwidth,
        # accumulate in float for better numerical accuracy
        dtype = "float16"
        accum_dtype = "float"

        @T.prim_func
        def main(
                A_sparse: T.Tensor((M, K // 2), dtype),
                E: T.Tensor((M, K // 8), 'uint8'),
                B: T.Tensor((N, K), dtype),
                C: T.Tensor((M, N), dtype),
        ):
            """
            The compiled TVM function for block-level matrix multiplication.

            - We divide the entire (M, N) domain into blocks of shape
              (block_M, block_N).
            - Each block has its own allocated shared memory for sub-blocks
              of A and B.
            - The partial results go into C_local, and then we copy them back
              to global memory C.
            """
            # Bind x-dimension to block index in N,
            #     y-dimension to block index in M.
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):

                # Allocate shared memory for A sub-block of shape (block_M, block_K)
                A_shared = T.alloc_shared((block_M, block_K // 2), dtype)
                # Allocate shared memory for B sub-block of shape (block_N, block_K)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                # Allocate shared memory for E sub-block of shape (block_M, block_K // E_factor)
                E_shared = T.alloc_shared((block_M, block_K // 8), 'uint8')
                # Allocate a local fragment for intermediate accumulation
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                # Allocate a shared memory for C sub-block of shape (block_M, block_N)
                C_shared = T.alloc_shared((block_M, block_N), dtype)

                # Clear out the accumulation buffer
                T.clear(C_local)
                T.disable_warp_group_reg_alloc()

                T.use_swizzle(panel_size=10, enable=enable_rasterization)
                T.annotate_layout({
                    E:
                        make_metadata_layout(
                            E, mma_dtype="float16", arch="sm90", backend="cutlass",
                            block_k=block_K),
                    E_shared:
                        make_metadata_layout(
                            E_shared,
                            mma_dtype="float16",
                            arch="sm90",
                            backend="cutlass",
                            block_k=block_K),
                })
                # Loop over sub-blocks in K dimension, pipelined by num_stages
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    # Load a sub-block of A from global memory into A_shared
                    T.copy(A_sparse[by * block_M, k * block_K], A_shared)
                    # Load a sub-block of E from global memory into E_shared
                    T.copy(E[by * block_M, k * block_K // 8], E_shared)
                    # Load a sub-block of B from global memory into B_shared
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    # Perform a partial matrix multiplication:
                    #   C_local += A_shared @ B_shared^T
                    T.gemm_sp(
                        A_shared,
                        E_shared,
                        B_shared,
                        C_local,
                        transpose_B=True,
                        policy=policy,
                    )
                # Write back the results from C_local to the global memory C
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return main

    return kernel()


if __name__ == "__main__":
    # Parse command-line arguments for matrix dimensions
    parser = argparse.ArgumentParser(description="Autotuned MatMul Benchmark")
    parser.add_argument("--m", type=int, default=16384, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=16384, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=16384, help="Matrix dimension K")
    args = parser.parse_args()

    M, N, K = args.m, args.n, args.k

    # Compute total floating-point operations to measure throughput
    total_flops = 2 * M * N * K

    # matmul(...) returns (best_latency, best_config, ref_latency)
    best_result = matmul_sp(M, N, K)
    best_latency = best_result.latency
    best_config = best_result.config
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")
    ref_latency = do_bench(lambda: A @ B.T)

    # Print out the benchmark results
    print(f"Best latency (s): {best_latency}")
    print(f"Best TFlops: {total_flops / best_latency * 1e-9:.3f}")
    print(f"Best config: {best_config}")

    print(f"Reference TFlops: {total_flops / ref_latency * 1e-9:.3f}")
