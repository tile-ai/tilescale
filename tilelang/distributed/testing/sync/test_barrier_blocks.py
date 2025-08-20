import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
import torch
import argparse

tilelang.disable_cache()


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


def test_barrier_blocks(num_blocks: int = 64, threads: int = 128, print_source: bool = False):
    kernel = get_test_barrier_blocks_kernel(num_blocks, threads)
    bar = torch.tensor([0], dtype=torch.int32, device='cuda')
    input = torch.zeros(threads, dtype=torch.int32, device='cuda')
    if print_source:
        print(kernel.get_kernel_source())
    print('Compilation done, start running...')

    output = kernel(input, bar)

    assert torch.all(output == num_blocks)
    print('Check passed✅')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blocks', type=int, default=64)
    parser.add_argument('--threads', type=int, default=128)
    parser.add_argument('--print_source', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.blocks <= driver.get_num_sms(
    ), f'Launched {args.blocks} blocks, which is larger than the number of SM ({driver.get_num_sms()}) on the current device and may cause deadlock!'
    test_barrier_blocks(args.blocks, args.threads, args.print_source)
