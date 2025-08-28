from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ABI = 0  # from torch._C._GLIBCXX_USE_CXX11_ABI == False

setup(
    name="alloc_cuda",
    ext_modules=[
        CUDAExtension(
            name="alloc_cuda",
            sources=[
                "tensor_from_ptr.cpp",
            ],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-std=c++17",
                    "-fPIC",
                    f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"
                ],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-Xcompiler",
                    "-fPIC",
                    "-Xcompiler",
                    f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
