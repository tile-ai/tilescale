import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[5]
_HEADER = _REPO_ROOT / "src/tl_templates/cuda/distributed/multimem.h"
_CUTLASS_INCLUDE = _REPO_ROOT / "3rdparty/cutlass/include"
_ACC_F32_INSTRUCTIONS = (
    "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.f16x2",
    "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.bf16x2",
)
_BULK_INSTRUCTIONS = (
    "multimem.cp.async.bulk.global.shared::cta.bulk_group",
    "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32",
    "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16",
    "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16",
    "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.min.f16",
    "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.max.f16",
    "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.min.bf16",
    "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.max.bf16",
)
_BULK_UNSUPPORTED_DIAGNOSTIC = "tl::multimem bulk operations require SM90+ and CUDA Toolkit 13.1+ (PTX 9.1)"


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


def _preprocess_multimem_header(*, cuda_arch: int, cuda_major: int, cuda_minor: int) -> str:
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
        f"-D__CUDA_ARCH__={cuda_arch}",
        f"-D__CUDACC_VER_MAJOR__={cuda_major}",
        f"-D__CUDACC_VER_MINOR__={cuda_minor}",
        f"-I{cuda_root / 'include'}",
        f"-I{_cccl_include(cuda_root)}",
        f"-I{_REPO_ROOT / 'src/tl_templates/cuda'}",
        f"-I{_CUTLASS_INCLUDE}",
        str(_HEADER),
    ]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


@pytest.mark.parametrize(
    ("cuda_arch", "cuda_major", "cuda_minor", "expect_direct", "expect_acc_f32", "expect_bulk"),
    (
        (800, 13, 1, False, False, False),
        (900, 12, 0, False, False, False),
        (900, 12, 1, True, False, False),
        (900, 12, 2, True, True, False),
        (900, 13, 0, True, True, False),
        (900, 13, 1, True, True, True),
    ),
)
def test_multimem_cuda_version_guards(cuda_arch, cuda_major, cuda_minor, expect_direct, expect_acc_f32, expect_bulk):
    output = _preprocess_multimem_header(cuda_arch=cuda_arch, cuda_major=cuda_major, cuda_minor=cuda_minor)
    output = re.sub(r'"\s*"', "", output)

    assert ("enum class ReduceOp" in output) is expect_direct
    for instruction in _ACC_F32_INSTRUCTIONS:
        assert (instruction in output) is expect_acc_f32
    for instruction in _BULK_INSTRUCTIONS:
        assert (instruction in output) is expect_bulk
    assert (_BULK_UNSUPPORTED_DIAGNOSTIC in output) is not expect_bulk
    assert 'asm("trap;")' not in output


def _nvcc() -> Path:
    nvcc = _cuda_root() / "bin/nvcc"
    if not nvcc.is_file():
        pytest.skip("nvcc is required to compile the multimem header")
    return nvcc


def _compile_cuda(source: str, tmp_path: Path, *, arch: str) -> subprocess.CompletedProcess[str]:
    source_path = tmp_path / f"multimem_{arch}.cu"
    object_path = tmp_path / f"multimem_{arch}.o"
    source_path.write_text(source)
    return subprocess.run(
        [
            str(_nvcc()),
            "-std=c++17",
            f"-arch={arch}",
            "-c",
            f"-I{_REPO_ROOT / 'src'}",
            f"-I{_CUTLASS_INCLUDE}",
            str(source_path),
            "-o",
            str(object_path),
        ],
        capture_output=True,
        text=True,
    )


def test_unsupported_arch_only_rejects_explicit_bulk_use(tmp_path):
    # Exercise the externally valid reverse include order as well. These
    # headers must not share a namespace-dependent fallback trait.
    include = "#include <tl_templates/cuda/distributed/multimem.h>\n#include <tl_templates/cuda/distributed/ldst.h>\n"
    ordinary = _compile_cuda(include + "__global__ void kernel() {}\n", tmp_path, arch="sm_80")
    assert ordinary.returncode == 0, ordinary.stderr

    explicit_bulk = _compile_cuda(
        include
        + "__global__ void kernel(void *dst, void *src) {\n"
        + "#if defined(__CUDA_ARCH__)\n"
        + "  tl::multimem::cp_async_bulk(dst, src, 16);\n"
        + "#endif\n"
        + "}\n",
        tmp_path,
        arch="sm_80",
    )
    assert explicit_bulk.returncode != 0
    assert _BULK_UNSUPPORTED_DIAGNOSTIC in explicit_bulk.stderr


def test_supported_bulk_instruction_matrix_compiles(tmp_path):
    version = subprocess.run([str(_nvcc()), "--version"], check=True, capture_output=True, text=True)
    match = re.search(r"release (\d+)\.(\d+)", version.stdout)
    if match is None or tuple(map(int, match.groups())) < (13, 1):
        pytest.skip("CUDA Toolkit 13.1+ is required for multimem bulk instructions")

    calls = "\n".join(
        f"  tl::multimem::{name}(dst, src, 16);"
        for name in (
            "cp_async_bulk",
            "cp_reduce_async_bulk_add_f32",
            "cp_reduce_async_bulk_add_f16",
            "cp_reduce_async_bulk_add_bf16",
            "cp_reduce_async_bulk_min_f16",
            "cp_reduce_async_bulk_max_f16",
            "cp_reduce_async_bulk_min_bf16",
            "cp_reduce_async_bulk_max_bf16",
        )
    )
    result = _compile_cuda(
        "#include <tl_templates/cuda/distributed/multimem.h>\n"
        + "__global__ void kernel(void *dst, void *src) {\n"
        + "#if defined(__CUDA_ARCH__)\n"
        + calls
        + "\n#endif\n"
        + "\n}\n",
        tmp_path,
        arch="sm_90",
    )
    assert result.returncode == 0, result.stderr
