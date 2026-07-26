from __future__ import annotations

import os
import platform
import subprocess
import time
from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).parent

base_version = (ROOT / "VERSION").read_text().strip()
# When installing a sdist,
# the installed version needs to match the sdist version,
# so pip will complain when we install `tilescale-0.0.2+gitxxxx.tar.gz`.
# To workaround that, when building sdist,
# we do not add version label and use a file to store the git hash instead.
git_pin = ROOT / ".git_commit.txt"


def _read_cmake_bool(i: str | None, default=False):
    if i is None:
        return default
    return i.lower() not in ("0", "false", "off", "no", "n", "")


def _is_git_checkout() -> bool:
    """Return whether ROOT owns Git metadata, including worktree markers."""
    marker = ROOT / ".git"
    return marker.is_dir() or marker.is_file()


@lru_cache(maxsize=1)
def get_git_commit_id() -> str | None:
    """Get the current git commit hash by running git in the current file's directory."""

    if _is_git_checkout():
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, encoding="utf-8")
        if result.returncode == 0:
            commit = result.stdout.strip()
            git_pin.write_text(commit)
            return commit
    if git_pin.exists():
        return git_pin.read_text().strip()
    return None


def dynamic_metadata(field: str, settings: dict[str, object] | None = None) -> str:
    assert field == "version"

    version = base_version

    # generate git version for sdist
    get_git_commit_id()

    # An sdist or source archive must rebuild to the exact public version even
    # when release-only environment variables are absent. The pin remains
    # available for provenance but never becomes local version metadata.
    if not _is_git_checkout():
        return version

    if not _read_cmake_bool(os.environ.get("NO_VERSION_LABEL")):
        exts = []
        backend = None
        if _read_cmake_bool(os.environ.get("NO_TOOLCHAIN_VERSION")):
            pass
        elif platform.system() == "Darwin":
            # only on macosx_11_0_arm64, not necessary
            # backend = 'metal'
            pass
        elif _read_cmake_bool(os.environ.get("USE_ROCM", "")) and not _read_cmake_bool(os.environ.get("USE_CUDA", "")):
            # ROCm-only build. When USE_CUDA is also on (fat wheel), fall through and label as the CUDA backend so the wheel keeps using cuda naming
            backend = "rocm"
        elif "USE_CUDA" in os.environ and not _read_cmake_bool(os.environ.get("USE_CUDA")):
            backend = "cpu"
        else:  # cuda
            # Read nvcc version from env.
            # This is not exactly how it should be,
            # but works for now if building in a nvidia/cuda image.
            if cuda_version := os.environ.get("CUDA_VERSION"):
                major, minor, *_ = cuda_version.split(".")
                backend = f"cu{major}{minor}"
            else:
                backend = "cuda"
        if backend:
            exts.append(backend)

        # Add build date if TILELANG_BUILD_WHEEL_WITH_DATE is set
        if _read_cmake_bool(os.environ.get("TILELANG_BUILD_WHEEL_WITH_DATE")):
            exts.append(f"d{time.strftime('%Y%m%d')}")

        if _read_cmake_bool(os.environ.get("NO_GIT_VERSION")):
            pass
        elif git_hash := get_git_commit_id():
            exts.append(f"git{git_hash[:8]}")
        else:
            exts.append("gitunknown")

        if exts:
            version += "+" + ".".join(exts)

    return version


__all__ = ["dynamic_metadata"]
