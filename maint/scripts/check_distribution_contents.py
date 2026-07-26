#!/usr/bin/env python3
"""Verify that release artifacts contain required notices and no restricted files."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path
from pathlib import PurePosixPath


REQUIRED_LICENSE_FILES = (
    "LICENSE",
    "THIRDPARTYNOTICES.txt",
    "LICENSES/1bitLLM-BitNet.txt",
    "LICENSES/AMD-Matrix-Instruction-Calculator.txt",
    "LICENSES/BitBLAS.txt",
    "LICENSES/DeepGEMM.txt",
    "LICENSES/Flash-Linear-Attention.txt",
    "LICENSES/FlashMLA.txt",
    "LICENSES/MInference.txt",
    "LICENSES/Triton.txt",
    "3rdparty/tvm/LICENSE",
    "3rdparty/tvm/NOTICE",
    "3rdparty/tvm/3rdparty/tvm-ffi/LICENSE",
    "3rdparty/tvm/3rdparty/tvm-ffi/NOTICE",
    "3rdparty/tvm/3rdparty/tvm-ffi/3rdparty/dlpack/LICENSE",
    "3rdparty/tvm/3rdparty/tvm-ffi/3rdparty/libbacktrace/LICENSE",
    "3rdparty/tvm/3rdparty/tvm-ffi/licenses/LICENSE.pytorch.txt",
    "3rdparty/tvm/3rdparty/tvm-ffi/licenses/NOTICE.pytorch.txt",
    "3rdparty/tvm/3rdparty/OpenCL-Headers/LICENSE",
    "3rdparty/tvm/3rdparty/cutlass_fpA_intB_gemm/LICENSE",
    "3rdparty/tvm/3rdparty/libflash_attn/LICENSE",
    "3rdparty/tvm/licenses/LICENSE.blockingconcurrentqueue.txt",
    "3rdparty/tvm/licenses/LICENSE.builtin_fp16.txt",
    "3rdparty/tvm/licenses/LICENSE.concurrentqueue.txt",
    "3rdparty/tvm/licenses/LICENSE.cutlass.txt",
    "3rdparty/tvm/licenses/LICENSE.cutlass_fpA_intB_gemm.txt",
    "3rdparty/tvm/licenses/LICENSE.l2_cache_flush.txt",
    "3rdparty/tvm/licenses/LICENSE.libflash_attn.txt",
    "3rdparty/tvm/licenses/LICENSE.rang.txt",
    "3rdparty/tvm/licenses/LICENSE.tensorrt_llm.txt",
    "3rdparty/tvm/licenses/LICENSE.vllm.txt",
    "3rdparty/cutlass/LICENSE.txt",
    "3rdparty/composable_kernel/LICENSE",
)

FORBIDDEN_PATH_FRAGMENTS = (
    ".agents/",
    "3rdparty/tvm/3rdparty/cutlass/EULA.txt",
    "3rdparty/tvm/3rdparty/cutlass/python/CuTeDSL/",
    "3rdparty/tvm/3rdparty/cutlass/examples/python/CuTeDSL/",
    "3rdparty/tvm/3rdparty/cutlass/docs/",
    "3rdparty/tvm/3rdparty/cutlass/media/",
    "3rdparty/tvm/3rdparty/cutlass_fpA_intB_gemm/cutlass/docs/",
    "3rdparty/tvm/3rdparty/cutlass_fpA_intB_gemm/cutlass/media/",
    "3rdparty/tvm/3rdparty/libflash_attn/cutlass/docs/",
    "3rdparty/tvm/3rdparty/libflash_attn/cutlass/media/",
    "3rdparty/cutlass/EULA.txt",
    "3rdparty/cutlass/python/CuTeDSL/",
    "3rdparty/cutlass/examples/python/CuTeDSL/",
    "3rdparty/cutlass/docs/",
    "3rdparty/cutlass/media/",
    "src/cuda/stubs/vendor/cuda.h",
)


def _forbidden_reason(name: str) -> str | None:
    parts = PurePosixPath(name).parts
    if ".git" in parts:
        return "nested Git metadata"
    if "__pycache__" in parts or name.endswith((".pyc", ".pyo")):
        return "Python bytecode/cache"
    if any(fragment in name for fragment in FORBIDDEN_PATH_FRAGMENTS):
        return "restricted or non-release source"
    return None


def _wheel_members(path: Path) -> tuple[set[str], set[str]]:
    with zipfile.ZipFile(path) as archive:
        members = {name.lstrip("./") for name in archive.namelist() if not name.endswith("/")}

    license_roots = {
        name.partition(".dist-info/licenses/")[0] + ".dist-info/licenses/" for name in members if ".dist-info/licenses/" in name
    }
    if len(license_roots) != 1:
        raise ValueError(f"expected one .dist-info/licenses directory, found {sorted(license_roots)}")
    license_root = license_roots.pop()
    required = {license_root + name for name in REQUIRED_LICENSE_FILES}
    return members, required


def _sdist_members(path: Path) -> tuple[set[str], set[str]]:
    with tarfile.open(path, mode="r:*") as archive:
        members = {member.name.lstrip("./") for member in archive if member.isfile()}

    roots = {name.split("/", 1)[0] for name in members if "/" in name}
    if len(roots) != 1:
        raise ValueError(f"expected one sdist root directory, found {sorted(roots)}")
    root = roots.pop() + "/"
    required = {root + name for name in REQUIRED_LICENSE_FILES}
    return members, required


def check_artifact(path: Path) -> list[str]:
    if path.suffix == ".whl":
        members, required = _wheel_members(path)
    elif path.name.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
        members, required = _sdist_members(path)
    else:
        raise ValueError(f"unsupported artifact type: {path}")

    problems = [f"missing required file: {name}" for name in sorted(required - members)]
    problems.extend(
        f"contains forbidden file ({reason}): {name}" for name in sorted(members) if (reason := _forbidden_reason(name)) is not None
    )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()

    failed = False
    for artifact in args.artifacts:
        if not artifact.is_file():
            print(f"FAIL {artifact}: artifact does not exist")
            failed = True
            continue
        try:
            problems = check_artifact(artifact)
        except (OSError, tarfile.TarError, ValueError, zipfile.BadZipFile) as error:
            print(f"FAIL {artifact}: {error}")
            failed = True
            continue
        if problems:
            print(f"FAIL {artifact}")
            for problem in problems:
                print(f"  - {problem}")
            failed = True
        else:
            print(f"PASS {artifact}")

    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
