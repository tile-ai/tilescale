import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[5]
_HEADER = _REPO_ROOT / "src/tl_templates/cuda/distributed/multimem.h"
_CUTLASS_INCLUDE = _REPO_ROOT / "3rdparty/cutlass/include"
_ACC_F32_INSTRUCTIONS = (
    "multimem.ld_reduce.weak.global.add.acc::f32.f16x2",
    "multimem.ld_reduce.weak.global.add.acc::f32.bf16x2",
)


def _cuda_root() -> Path:
    configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if configured:
        return Path(configured)
    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc).resolve().parent.parent
    pytest.skip("CUDA Toolkit is required to preprocess the multimem header")


def _cccl_include(cuda_root: Path) -> Path:
    candidates = [cuda_root / "include/cccl", *sorted(cuda_root.glob("targets/*/include/cccl"))]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    pytest.skip("CUDA Toolkit CCCL headers are unavailable")


@pytest.mark.parametrize(
    ("cuda_minor", "expect_direct", "expect_acc_f32"),
    ((0, False, False), (1, True, False), (2, True, True)),
)
def test_multimem_cuda_version_guards(cuda_minor, expect_direct, expect_acc_f32):
    cuda_root = _cuda_root()
    cxx = shlex.split(os.environ.get("CXX", "c++"))
    if not cxx or shutil.which(cxx[0]) is None:
        pytest.skip("A C++ preprocessor is required")

    command = [
        *cxx,
        "-E",
        "-P",
        "-x",
        "c++",
        "-D__CUDA_ARCH__=900",
        "-D__CUDACC_VER_MAJOR__=12",
        f"-D__CUDACC_VER_MINOR__={cuda_minor}",
        f"-I{cuda_root / 'include'}",
        f"-I{_cccl_include(cuda_root)}",
        f"-I{_REPO_ROOT / 'src/tl_templates/cuda'}",
        f"-I{_CUTLASS_INCLUDE}",
        str(_HEADER),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)

    assert ("namespace multimem" in result.stdout) is expect_direct
    for instruction in _ACC_F32_INSTRUCTIONS:
        assert (instruction in result.stdout) is expect_acc_f32
