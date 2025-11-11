import tilelang
import tilelang.language as T
import argparse
import torch

tilelang.disable_cache()


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    })
def copy_cluster(M, N, block_M, block_N, threads, cluster, dtype="float16"):

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.ScopeKernel(
                grid=(T.ceildiv(M, block_M), T.ceildiv(N, block_N), 1),
                cluster=cluster,
                threads=threads):

            bx, by, _ = T.get_block_bindings()
            cx, cy, cz = T.get_cluster_bindings()
            tx = T.get_thread_binding(0)

            A_local = T.alloc_fragment((block_M, block_N), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)
            bar_ready = T.alloc_barrier(arrive_count=1)

            if (tx == 0):
                T.ptx_arrive_barrier_expect_tx(bar_ready[0], 2048)

            T.copy(A[bx * block_M, by * block_N], A_local)
            for i in T.serial(4):
                T.put_thread(
                    src=T.address_of(A_local[(i * 256 + tx * 8) // block_N,
                                             (i * 256 + tx * 8) % block_N]),
                    dst=T.address_of(B_shared[(i * 256 + tx * 8) // block_N,
                                              (i * 256 + tx * 8) % block_N]),
                    size=0,
                    mbar=T.address_of(bar_ready),
                    dst_pe=(cx + 1) % cluster[0],
                    scope="cluster")
            T.barrier_wait(bar_ready, 0)
            T.copy(B_shared, B[(bx ^ 1) * block_M, by * block_N])
            T.sync_cluster()

    return main


def ref_program(A):
    return A


def main(M=4096, N=4096, cluster=None):

    BLOCK_M = 32
    BLOCK_N = 32
    threads = 32

    kernel = copy_cluster(M, N, BLOCK_M, BLOCK_N, threads, cluster)
    print(kernel.get_kernel_source())
    A = torch.randn((M, N), dtype=torch.float16).cuda()
    B = kernel(A)
    # print(A)
    # print(B)
    assert torch.allclose(A, B, rtol=0.01, atol=0.01)
    print("All checks passed.✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=8192, help='M dimension')
    parser.add_argument('--N', type=int, default=8192, help='N dimension')
    parser.add_argument("--cluster", type=int, nargs='+', default=[2, 1, 1], help="cluster size")
    args = parser.parse_args()
    M, N, cluster = args.M, args.N, tuple(args.cluster)
    main(M, N, cluster)
