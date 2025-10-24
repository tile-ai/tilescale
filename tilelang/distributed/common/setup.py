from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME
import os
import torch

# --- Torch lib dir for rpath/loader path ---
try:
    from torch.utils.cpp_extension import _get_torch_lib_dir
    torch_lib_dir = _get_torch_lib_dir()
except Exception:
    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")

# --- Optional CUDA include dir (only if CUDA exists) ---
include_dirs = []
if CUDA_HOME is not None:
    # e.g. /usr/local/cuda or conda's CUDA toolkit
    cuda_inc = os.path.join(CUDA_HOME, "include")
    if os.path.isdir(cuda_inc):
        include_dirs.append(cuda_inc)

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17", "-fPIC"]
}

extra_link_args = []
runtime_library_dirs = []

if os.name == "posix":
    # Linux/macOS: embed search path to PyTorch .so (libc10, libtorch_cpu, etc.)
    runtime_library_dirs = [torch_lib_dir]
    extra_link_args = [f"-Wl,-rpath,{torch_lib_dir}"]

setup(
    name="ipc_ext",
    packages=["ipc_ext"],
    ext_modules=[
        CppExtension(
            name="ipc_ext._C",
            sources=["ipc_ext/ipc_ext.cpp"],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            runtime_library_dirs=runtime_library_dirs,
            library_dirs=[torch_lib_dir],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
