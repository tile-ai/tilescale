import tilelang
import tilelang.language as T
import argparse

tilelang.disable_cache()


@tilelang.jit(out_idx=[-1])
def matmul_cluster(M,
                   N,
                   K,
                   block_M,
                   block_N,
                   block_K,
                   threads,
                   cluster,
                   num_stages,
                   dtype="float16",
                   accum_dtype="float"):

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.ScopeKernel(
                grid=(T.ceildiv(M, block_M), T.ceildiv(N, block_N), 1),
                cluster=cluster,
                threads=threads):

            bx, by, _ = T.get_block_bindings()

            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[bx * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, by * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[bx * block_M, by * block_N])

    return main


def ref_program(A, B):
    return A @ B


def main(M=4096, N=4096, K=4096, cluster=None):
    total_flops = 2 * M * N * K

    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    threads = 256
    num_stages = 3

    kernel = matmul_cluster(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, threads, cluster, num_stages)
    print(kernel.get_kernel_source())
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)
    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("All check passed.")
    latency = profiler.do_bench(warmup=500)
    print(f"GEMM Latency: {latency} ms")
    print(f"GEMM TFlops: {total_flops / latency * 1e-9} TFlops")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=8192, help='M dimension')
    parser.add_argument('--N', type=int, default=8192, help='N dimension')
    parser.add_argument('--K', type=int, default=8192, help='K dimension')
    parser.add_argument("--cluster", type=int, nargs='+', default=[2, 1, 1], help="cluster size")
    args = parser.parse_args()
    M, N, K, cluster = args.M, args.N, args.K, tuple(args.cluster)
    main(M, N, K, cluster)
