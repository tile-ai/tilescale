import tilelang
import tilelang.language as T
import torch

# tilelang.disable_cache()


@tilelang.jit(out_idx=-1)
def get_test_barrier_blocks_kernel(num_blocks: int, threads: int):

    @T.prim_func
    def main(A: T.Tensor([threads], "int32"), bar: T.Tensor([1], "int32"),
             B: T.Tensor([num_blocks, threads], "int32")):
        with T.Kernel(num_blocks, threads=threads) as bid:
            tid = T.get_thread_binding()
            b = T.alloc_shared([threads], "int32")
            T.atomic_add(A[tid], 1)

            T.barrier_blocks(bar, num_blocks)

            T.copy(A, b)
            T.copy(b, B[bid, :])

    return main


def test_barrier_blocks(num_blocks: int = 256):
    threads = 128
    kernel = get_test_barrier_blocks_kernel(num_blocks, threads)
    bar = torch.tensor([0], dtype=torch.int32, device='cuda')
    input = torch.zeros(threads, dtype=torch.int32, device='cuda')

    print('Compilation done..')
    output = kernel(input, bar)  # shift 1
    assert torch.all(output == num_blocks)
    print('Check passed✅')


if __name__ == "__main__":
    test_barrier_blocks()
