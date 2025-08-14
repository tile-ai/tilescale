import torch
import tilelang
import tilelang.language as T
import pytest

tilelang.disable_cache()


def remote_copy(M, N, dtype="float", unroll_factor=4, blocks=1, threads=128):

    @T.prim_func
    def main(
            src: T.Tensor((M, N), dtype),
            dst: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(blocks, threads=threads) as (bx):
            T.copy_unrolled(
                T.address_of(dst[T.ceildiv(M, blocks) * bx, 0]),
                T.address_of(src[T.ceildiv(M, blocks) * bx, 0]),
                T.ceildiv(M, blocks) * N, unroll_factor)

    return main


@pytest.mark.parametrize("M", [1, 16, 256])
@pytest.mark.parametrize("N", [64, 1024])
@pytest.mark.parametrize("dtype", ["float32", "int32", "int64"])
@pytest.mark.parametrize("unroll_factor", [2, 3, 4, 5])
@pytest.mark.parametrize("blocks", [1, 4, 16])
@pytest.mark.parametrize("threads", [128, 256])
def test_copy_unrolled(M, N, dtype, unroll_factor, blocks, threads, get_kernel_source=False):

    def dtype_to_torch(dtype):
        if dtype == "float32":
            return torch.float32
        elif dtype == "int32":
            return torch.int32
        elif dtype == "int64":
            return torch.int64
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    torch_dtype = dtype_to_torch(dtype)
    is_float = torch.zeros((), dtype=torch_dtype).is_floating_point()
    if is_float:
        src = torch.empty(M, N, dtype=torch_dtype, device="cuda").normal_()
        dst = torch.empty(M, N, dtype=torch_dtype, device="cuda")
    else:
        info = torch.iinfo(torch_dtype)
        low = max(info.min, -32768)
        high = min(info.max, 32767) + 1
        src = torch.randint(low=low, high=high, size=(M, N), dtype=torch_dtype, device="cuda")
        dst = torch.empty(M, N, dtype=torch_dtype, device="cuda")

    func = remote_copy(M, N, dtype, unroll_factor, blocks, threads)
    kernel = tilelang.compile(func)
    kernel(src, dst)
    if get_kernel_source:
        print(kernel.get_kernel_source())
    assert torch.allclose(src, dst)


if __name__ == "__main__":
    test_copy_unrolled(16, 1024, "int64", 4, 1, 128, get_kernel_source=True)