from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="alloc_cuda",
    ext_modules=[
        CUDAExtension(
            name="alloc_cuda",
            sources=[
                "tensor_from_ptr.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-fPIC"],
                "nvcc": [
                    "-O3", "-std=c++17", "-Xcompiler", "-fPIC"
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
