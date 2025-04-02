from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import torch
import os
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

torch_include_paths = torch.utils.cpp_extension.include_paths()

cuda_include_path = "/usr/local/cuda/include"
nvshmem_include_path = os.environ["NVSHMEM_PATH"] + "/build/src/include"
nvshmem_lib_path = os.environ["NVSHMEM_PATH"] + "/build/src/lib"


extensions = [
    CUDAExtension(
        "tilelang.jit.nvshmem_utils",
        sources=[
            "tilelang/jit/nvshmem_utils.pyx",
            "tilelang/jit/nvshmem_utils_impl.cu"
        ],
        include_dirs=[np.get_include(), nvshmem_include_path, cuda_include_path] + torch_include_paths,
        library_dirs=[nvshmem_lib_path],
        language="c++",
        extra_link_args=[
            f"-L{nvshmem_lib_path}",
            "--enable-new-dtags",
            "-lcudart",
            "-lnvToolsExt"
        ],
        libraries=["nvshmem_host", "nvshmem_device", "cudart"],
        extra_compile_args={
            "cxx": ["-std=c++17", f"-I{nvshmem_include_path}", "-fPIC"],
            "nvcc": [
                "-std=c++17",
                "-rdc=true",
                f"-I{nvshmem_include_path}",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
            ],
        },
    )
]

setup(
    name="nvshmem_utils",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "embedsignature": True,
        },
    ),
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)