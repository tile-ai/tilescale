from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import setuptools
from torch.utils.cpp_extension import BuildExtension

from tilelang.env import env

# Project directory root
root_path: Path = Path(__file__).resolve().parent
PACKAGE_NAME = "pynvshmem"


def cuda_version() -> tuple[int, ...]:
    """CUDA Toolkit version as a (major, minor) by nvcc --version"""

    # Try finding NVCC
    nvcc_bin: Path | None = None
    if nvcc_bin is None and os.getenv("CUDA_HOME"):
        # Check in CUDA_HOME
        cuda_home = Path(os.getenv("CUDA_HOME"))
        nvcc_bin = cuda_home / "bin" / "nvcc"
    if nvcc_bin is None:
        # Check if nvcc is in path
        nvcc_bin = shutil.which("nvcc")
        if nvcc_bin is not None:
            nvcc_bin = Path(nvcc_bin)
    if nvcc_bin is None:
        # Last-ditch guess in /usr/local/cuda
        cuda_home = Path("/usr/local/cuda")
        nvcc_bin = cuda_home / "bin" / "nvcc"
    if not nvcc_bin.is_file():
        raise FileNotFoundError(f"Could not find NVCC at {nvcc_bin}")

    # Query NVCC for version info
    output = subprocess.run(
        [nvcc_bin, "-V"],
        capture_output=True,
        check=True,
        text=True,
    )
    match = re.search(r"release\s*([\d.]+)", output.stdout)
    version = match.group(1).split(".")
    return tuple(int(v) for v in version)


def get_package_version():
    return "0.0.1"


def pathlib_wrapper(func):

    def wrapper(*kargs, **kwargs):
        include_dirs, library_dirs, libraries = func(*kargs, **kwargs)
        return map(str, include_dirs), map(str, library_dirs), map(str, libraries)

    return wrapper


def ensure_nvshmem_symlinks():
    """
    Ensure symbolic links exist for NVSHMEM libraries.

    The nvidia-nvshmem-cu12 wheel provides versioned libraries (e.g., libnvshmem_host.so.3),
    but the linker expects unversioned names (e.g., libnvshmem_host.so).
    This function creates the necessary symlinks automatically during build.
    """
    if env.NVSHMEM_LIB_PATH is None:
        return

    lib_path = Path(env.NVSHMEM_LIB_PATH)
    if not lib_path.exists():
        return

    # Map of expected symlink name to the versioned library file pattern
    symlink_map = {
        "libnvshmem_host.so": "libnvshmem_host.so.*",
        "libnvshmem_device.a": "libnvshmem_device.a",  # This one might already be correct
    }

    for symlink_name, pattern in symlink_map.items():
        symlink_path = lib_path / symlink_name

        # Skip if symlink already exists and is valid
        if symlink_path.exists() or symlink_path.is_symlink():
            continue

        # Find the versioned library file
        versioned_libs = list(lib_path.glob(pattern))
        if not versioned_libs:
            continue

        # Use the first match (or latest if multiple)
        target = versioned_libs[0].name

        try:
            # Create the symlink
            symlink_path.symlink_to(target)
            print(f"Created symlink: {symlink_path} -> {target}")
        except Exception as e:
            print(f"Warning: Could not create symlink {symlink_path}: {e}")


@pathlib_wrapper
def nvshmem_deps():
    # Ensure symlinks exist before returning dependencies
    ensure_nvshmem_symlinks()

    include_dirs = [env.NVSHMEM_INCLUDE_DIR]
    library_dirs = [env.NVSHMEM_LIB_PATH]
    libraries = ["nvshmem_host", "nvshmem_device"]
    return include_dirs, library_dirs, libraries


@pathlib_wrapper
def cuda_deps():
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    include_dirs = [cuda_home / "include"]
    library_dirs = [cuda_home / "lib64", cuda_home / "lib64/stubs"]
    libraries = ["cuda", "cudart", "nvidia-ml"]
    return include_dirs, library_dirs, libraries


def setup_pytorch_extension() -> setuptools.Extension:
    """Setup CppExtension for PyTorch support"""
    include_dirs, library_dirs, libraries = [], [], []

    deps = [nvshmem_deps(), cuda_deps()]

    for include_dir, library_dir, library in deps:
        include_dirs += include_dir
        library_dirs += library_dir
        libraries += library

    # Compiler flags
    # too much warning from CUDA /usr/local/cuda/include/cusparse.h: "-Wdeprecated-declarations"
    cxx_flags = [
        "-O3",
        "-DTORCH_CUDA=1",
        "-fvisibility=hidden",
        "-Wno-deprecated-declarations",
        "-fdiagnostics-color=always",
    ]
    ld_flags = ["-Wl,--exclude-libs=libnccl_static"]

    from torch.utils.cpp_extension import CUDAExtension

    return CUDAExtension(
        name="_pynvshmem",
        sources=["src/pynvshmem.cc"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        dlink=True,
        dlink_libraries=["nvshmem_device", "cudart_static"],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": ["-rdc=true"]
        },
        extra_link_args=ld_flags,
    )


def main():
    packages = setuptools.find_packages(
        where="python",
        include=[
            "pynvshmem",
            "_pynvshmem",
        ],
    )
    # Configure package
    setuptools.setup(
        name=PACKAGE_NAME,
        version=get_package_version(),
        package_dir={"": "python"},
        packages=packages,
        description="tilelang-distributed pynvshmem",
        ext_modules=[setup_pytorch_extension()],
        cmdclass={"build_ext": BuildExtension},
        setup_requires=["cmake", "packaging"],
        install_requires=[],
        extras_require={"test": ["numpy"]},
        license_files=("LICENSE",),
        package_data={
            "python/pynvshmem/lib": ["*.so"],
        },  # only works for bdist_wheel under package
        python_requires=">=3.8",
        include_package_data=True,
    )


if __name__ == "__main__":
    main()
