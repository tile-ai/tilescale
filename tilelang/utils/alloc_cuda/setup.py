from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os, torch

torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
try:
    from torch.utils.cpp_extension import _get_torch_lib_dir
    torch_lib_dir = _get_torch_lib_dir()
except Exception:
    pass

extra_compile_args = {
    "cxx":  ["-O3", "-std=c++17", "-fPIC"],
    "nvcc": ["-O3", "-std=c++17", "-Xcompiler", "-fPIC"],
}

extra_link_args = []
runtime_library_dirs = []

if os.name == "posix":
    runtime_library_dirs = [torch_lib_dir]
    extra_link_args = [f"-Wl,-rpath,{torch_lib_dir}"]

setup(
    name="alloc_cuda",
    packages=["alloc_cuda"],
    ext_modules=[
        CUDAExtension(
            name="alloc_cuda._C",
            sources=["alloc_cuda/tensor_from_ptr.cpp"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            runtime_library_dirs=runtime_library_dirs,
            library_dirs=[torch_lib_dir],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
