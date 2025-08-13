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
@pytest.mark.parametrize("dtype", ["float16", "float32", "int32", "int8"])
@pytest.mark.parametrize("unroll_factor", [2, 3, 4, 5])
@pytest.mark.parametrize("blocks", [1, 4, 16])
@pytest.mark.parametrize("threads", [128, 256])
def test_copy_unrolled(M, N, dtype, unroll_factor, blocks, threads):

    def dtype_to_torch(dtype):
        if dtype == "float16":
            return torch.float16
        elif dtype == "float32":
            return torch.float32
        elif dtype == "int32":
            return torch.int32
        elif dtype == "int8":
            return torch.int8
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

    func = remote_copy(M, N, dtype, unroll_factor, blocks)
    kernel = tilelang.compile(func)
    kernel(src, dst)
    if torch.allclose(src, dst):
        print('Allclose passed!✅')
    else:
        print('Allclose failed!❌')
