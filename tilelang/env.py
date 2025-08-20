import sys
import os
import pathlib
import logging
import shutil
import glob

logger = logging.getLogger(__name__)


def _find_cuda_home() -> str:
    """Find the CUDA install path.

    Adapted from https://github.com/pytorch/pytorch/blob/main/torch/utils/cpp_extension.py
    """
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is not None and "cuda" in nvcc_path.lower():
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        else:
            # Guess #3
            if sys.platform == 'win32':
                cuda_homes = glob.glob('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                cuda_home = '' if len(cuda_homes) == 0 else cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    return cuda_home if cuda_home is not None else ""


def _find_rocm_home() -> str:
    """Find the ROCM install path."""
    rocm_home = os.environ.get('ROCM_PATH') or os.environ.get('ROCM_HOME')
    if rocm_home is None:
        rocmcc_path = shutil.which("hipcc")
        if rocmcc_path is not None:
            rocm_home = os.path.dirname(os.path.dirname(rocmcc_path))
        else:
            rocm_home = '/opt/rocm'
            if not os.path.exists(rocm_home):
                rocm_home = None
    return rocm_home if rocm_home is not None else ""


def _initialize_torch_cuda_arch_flags():
    import os
    from tilelang.contrib import nvcc
    from tilelang.utils.target import determine_target

    target = determine_target(return_object=True)
    # create tmp source file for torch cpp extension
    compute_version = "".join(nvcc.get_target_compute_version(target).split("."))
    # set TORCH_CUDA_ARCH_LIST
    major = compute_version[0]
    minor = compute_version[1]

    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"


CUDA_HOME = _find_cuda_home()
ROCM_HOME = _find_rocm_home()

CUTLASS_INCLUDE_DIR: str = os.environ.get("TL_CUTLASS_PATH", None)
COMPOSABLE_KERNEL_INCLUDE_DIR: str = os.environ.get("TL_COMPOSABLE_KERNEL_PATH", None)
TVM_PYTHON_PATH: str = os.environ.get("TVM_IMPORT_PYTHON_PATH", None)
TVM_LIBRARY_PATH: str = os.environ.get("TVM_LIBRARY_PATH", None)
TILELANG_TEMPLATE_PATH: str = os.environ.get("TL_TEMPLATE_PATH", None)
TILELANG_PACKAGE_PATH: str = pathlib.Path(__file__).resolve().parents[0]
USE_DISTRIBUTED = os.getenv("TILELANG_USE_DISTRIBUTED", "0").lower() in ("1", "true", "on")
if USE_DISTRIBUTED:
    if os.environ.get("NVSHMEM_SRC", None) is not None:
        NVSHMEM_SRC = os.environ.get("NVSHMEM_SRC")
    else:
        NVSHMEM_SRC = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),"..", "3rdparty", "nvshmem_src")
    NVSHMEM_INCLUDE_DIR: str = NVSHMEM_SRC + "/build/src/include"
    NVSHMEM_LIB_PATH: str = NVSHMEM_SRC + "/build/src/lib"
else:
    NVSHMEM_INCLUDE_DIR = None
    NVSHMEM_LIB_PATH = None

TILELANG_CACHE_DIR: str = os.environ.get("TILELANG_CACHE_DIR",
                                         os.path.expanduser("~/.tilelang/cache"))

# Auto-clear cache if environment variable is set
TILELANG_CLEAR_CACHE = os.environ.get("TILELANG_CLEAR_CACHE", "0")

# CPU Utilizations for Auto-Tuning, default is 0.9
TILELANG_AUTO_TUNING_CPU_UTILITIES: str = os.environ.get("TILELANG_AUTO_TUNING_CPU_UTILITIES",
                                                         "0.9")

# CPU COUNTS for Auto-Tuning, default is -1,
# which will use TILELNAG_AUTO_TUNING_CPU_UTILITIES * get_available_cpu_count()
TILELANG_AUTO_TUNING_CPU_COUNTS: str = os.environ.get("TILELANG_AUTO_TUNING_CPU_COUNTS", "-1")

# Max CPU Count for Auto-Tuning, default is 100
TILELANG_AUTO_TUNING_MAX_CPU_COUNT: str = os.environ.get("TILELANG_AUTO_TUNING_MAX_CPU_COUNT", "-1")

# SETUP ENVIRONMENT VARIABLES
CUTLASS_NOT_FOUND_MESSAGE = ("CUTLASS is not installed or found in the expected path")
", which may lead to compilation bugs when utilize tilelang backend."
COMPOSABLE_KERNEL_NOT_FOUND_MESSAGE = (
    "Composable Kernel is not installed or found in the expected path")
", which may lead to compilation bugs when utilize tilelang backend."
TL_TEMPLATE_NOT_FOUND_MESSAGE = ("TileLang is not installed or found in the expected path")
", which may lead to compilation bugs when utilize tilelang backend."
TVM_LIBRARY_NOT_FOUND_MESSAGE = ("TVM is not installed or found in the expected path")

SKIP_LOADING_TILELANG_SO = os.environ.get("SKIP_LOADING_TILELANG_SO", "0")

# Handle TVM_IMPORT_PYTHON_PATH to import tvm from the specified path
TVM_IMPORT_PYTHON_PATH = os.environ.get("TVM_IMPORT_PYTHON_PATH", None)

if TVM_IMPORT_PYTHON_PATH is not None:
    os.environ["PYTHONPATH"] = TVM_IMPORT_PYTHON_PATH + ":" + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, TVM_IMPORT_PYTHON_PATH)
else:
    install_tvm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3rdparty", "tvm")
    if os.path.exists(install_tvm_path) and install_tvm_path not in sys.path:
        os.environ["PYTHONPATH"] = (
            install_tvm_path + "/python:" + os.environ.get("PYTHONPATH", ""))
        sys.path.insert(0, install_tvm_path + "/python")
        TVM_IMPORT_PYTHON_PATH = install_tvm_path + "/python"

    develop_tvm_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "3rdparty", "tvm")
    if os.path.exists(develop_tvm_path) and develop_tvm_path not in sys.path:
        os.environ["PYTHONPATH"] = (
            develop_tvm_path + "/python:" + os.environ.get("PYTHONPATH", ""))
        sys.path.insert(0, develop_tvm_path + "/python")
        TVM_IMPORT_PYTHON_PATH = develop_tvm_path + "/python"

    develop_tvm_library_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "build", "tvm")
    install_tvm_library_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
    if os.environ.get("TVM_LIBRARY_PATH") is None:
        if os.path.exists(develop_tvm_library_path):
            os.environ["TVM_LIBRARY_PATH"] = develop_tvm_library_path
        elif os.path.exists(install_tvm_library_path):
            os.environ["TVM_LIBRARY_PATH"] = install_tvm_library_path
        else:
            logger.warning(TVM_LIBRARY_NOT_FOUND_MESSAGE)
        # pip install build library path
        lib_path = os.path.join(TILELANG_PACKAGE_PATH, "lib")
        existing_path = os.environ.get("TVM_LIBRARY_PATH")
        if existing_path:
            os.environ["TVM_LIBRARY_PATH"] = f"{existing_path}:{lib_path}"
        else:
            os.environ["TVM_LIBRARY_PATH"] = lib_path
        TVM_LIBRARY_PATH = os.environ.get("TVM_LIBRARY_PATH", None)

if os.environ.get("TL_CUTLASS_PATH", None) is None:
    install_cutlass_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "3rdparty", "cutlass")
    develop_cutlass_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "3rdparty", "cutlass")
    if os.path.exists(install_cutlass_path):
        os.environ["TL_CUTLASS_PATH"] = install_cutlass_path + "/include"
        CUTLASS_INCLUDE_DIR = install_cutlass_path + "/include"
    elif (os.path.exists(develop_cutlass_path) and develop_cutlass_path not in sys.path):
        os.environ["TL_CUTLASS_PATH"] = develop_cutlass_path + "/include"
        CUTLASS_INCLUDE_DIR = develop_cutlass_path + "/include"
    else:
        logger.warning(CUTLASS_NOT_FOUND_MESSAGE)

if os.environ.get("TL_COMPOSABLE_KERNEL_PATH", None) is None:
    install_ck_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "3rdparty", "composable_kernel")
    develop_ck_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "3rdparty", "composable_kernel")
    if os.path.exists(install_ck_path):
        os.environ["TL_COMPOSABLE_KERNEL_PATH"] = install_ck_path + "/include"
        COMPOSABLE_KERNEL_INCLUDE_DIR = install_ck_path + "/include"
    elif (os.path.exists(develop_ck_path) and develop_ck_path not in sys.path):
        os.environ["TL_COMPOSABLE_KERNEL_PATH"] = develop_ck_path + "/include"
        COMPOSABLE_KERNEL_INCLUDE_DIR = develop_ck_path + "/include"
    else:
        logger.warning(COMPOSABLE_KERNEL_NOT_FOUND_MESSAGE)

if os.environ.get("TL_TEMPLATE_PATH", None) is None:
    install_tl_template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    develop_tl_template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    if os.path.exists(install_tl_template_path):
        os.environ["TL_TEMPLATE_PATH"] = install_tl_template_path
        TILELANG_TEMPLATE_PATH = install_tl_template_path
    elif (os.path.exists(develop_tl_template_path) and develop_tl_template_path not in sys.path):
        os.environ["TL_TEMPLATE_PATH"] = develop_tl_template_path
        TILELANG_TEMPLATE_PATH = develop_tl_template_path
    else:
        logger.warning(TL_TEMPLATE_NOT_FOUND_MESSAGE)


# Cache control
class CacheState:
    """Class to manage global kernel caching state."""
    _enabled = True

    @classmethod
    def enable(cls):
        """Enable kernel caching globally."""
        cls._enabled = True

    @classmethod
    def disable(cls):
        """Disable kernel caching globally."""
        cls._enabled = False

    @classmethod
    def is_enabled(cls) -> bool:
        """Return current cache state."""
        return cls._enabled


# Replace the old functions with class methods
enable_cache = CacheState.enable
disable_cache = CacheState.disable
is_cache_enabled = CacheState.is_enabled

__all__ = [
    "CUTLASS_INCLUDE_DIR",
    "COMPOSABLE_KERNEL_INCLUDE_DIR",
    "TVM_PYTHON_PATH",
    "TVM_LIBRARY_PATH",
    "TILELANG_TEMPLATE_PATH",
    "CUDA_HOME",
    "ROCM_HOME",
    "TILELANG_CACHE_DIR",
    "enable_cache",
    "disable_cache",
    "is_cache_enabled",
    "_initialize_torch_cuda_arch_flags",
]
