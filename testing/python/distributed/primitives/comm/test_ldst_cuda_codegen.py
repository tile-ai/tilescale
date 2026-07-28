from pathlib import Path
import shutil
import subprocess

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[5]
_CUTLASS_INCLUDE = _REPO_ROOT / "3rdparty/cutlass/include"


def test_weak_load_specializations_compile_with_ptxas(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is required to compile distributed load helpers")

    source_path = tmp_path / "weak_loads.cu"
    object_path = tmp_path / "weak_loads.o"
    source_path.write_text(
        """
#include <cstdint>
#include <tl_templates/cuda/distributed/ldst.h>

__global__ void weak_loads(const uint32_t *src, uint32_t *dst) {
  uint32_t cta, gpu, sys, gpu_na, sys_na;
  tl::ld<Semantic::WEAK, Scope::CTA, false, false>(src, cta);
  tl::ld<Semantic::WEAK, Scope::GPU, false, false>(src, gpu);
  tl::ld<Semantic::WEAK, Scope::SYS, false, false>(src, sys);
  tl::ld<Semantic::WEAK, Scope::GPU, false, true>(src, gpu_na);
  tl::ld<Semantic::WEAK, Scope::SYS, false, true>(src, sys_na);
  dst[0] = cta + gpu + sys + gpu_na + sys_na;
}
"""
    )
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-arch=sm_90",
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

    assert result.returncode == 0, result.stderr
