from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

import version_provider


def _run_isolated(code: str) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    import_paths = [str(repo_root), *(path for path in sys.path if "site-packages" in path)]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(import_paths))
    subprocess.run([sys.executable, "-S", "-c", code], cwd=repo_root, env=env, check=True)


def test_top_level_import_keeps_distributed_runtime_lazy():
    code = textwrap.dedent(
        """
        import inspect
        import sys
        import tilelang

        assert callable(tilelang.tensor)
        assert callable(tilelang.get_allocator)
        assert list(inspect.signature(tilelang.tensor).parameters) == [
            "shape", "dtype", "device", "allocator", "return_peers"
        ]
        assert list(inspect.signature(tilelang.get_allocator).parameters) == [
            "size", "device", "is_distributed", "local_rank",
            "num_local_ranks", "group", "use_vmm", "mcast_size"
        ]
        assert "tilelang.distributed.allocator" not in sys.modules
        assert "tilelang.distributed.host" not in sys.modules
        assert "tilelang.distributed.shared_memory" not in sys.modules
        """
    )
    _run_isolated(code)


def test_source_version_uses_project_provider():
    code = textwrap.dedent(
        """
        import tilelang
        from version_provider import dynamic_metadata

        assert tilelang.__version__ == dynamic_metadata("version")
        """
    )
    _run_isolated(code)


@pytest.fixture
def isolated_version_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(version_provider, "ROOT", tmp_path)
    monkeypatch.setattr(version_provider, "base_version", "0.0.2")
    monkeypatch.setattr(version_provider, "git_pin", tmp_path / ".git_commit.txt")
    version_provider.get_git_commit_id.cache_clear()
    yield version_provider
    version_provider.get_git_commit_id.cache_clear()


def test_source_archive_version_is_stable_without_git(monkeypatch, isolated_version_provider):
    commit = "1" * 40
    isolated_version_provider.git_pin.write_text(commit)
    monkeypatch.setattr(
        isolated_version_provider.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("git must not run outside a checkout"),
    )
    monkeypatch.delenv("NO_VERSION_LABEL", raising=False)
    monkeypatch.delenv("NO_TOOLCHAIN_VERSION", raising=False)
    monkeypatch.delenv("NO_GIT_VERSION", raising=False)

    assert isolated_version_provider.get_git_commit_id() == commit
    assert isolated_version_provider.dynamic_metadata("version") == "0.0.2"


def test_worktree_git_file_keeps_checkout_version_labels(monkeypatch, isolated_version_provider):
    commit = "1234567890abcdef" * 2 + "12345678"
    (isolated_version_provider.ROOT / ".git").write_text("gitdir: /tmp/example-worktree\n")
    completed = subprocess.CompletedProcess(["git", "rev-parse", "HEAD"], 0, stdout=f"{commit}\n", stderr="")
    monkeypatch.setattr(isolated_version_provider.subprocess, "run", lambda *_args, **_kwargs: completed)
    monkeypatch.delenv("NO_VERSION_LABEL", raising=False)
    monkeypatch.setenv("NO_TOOLCHAIN_VERSION", "ON")
    monkeypatch.delenv("NO_GIT_VERSION", raising=False)
    monkeypatch.delenv("TILELANG_BUILD_WHEEL_WITH_DATE", raising=False)

    assert isolated_version_provider.dynamic_metadata("version") == "0.0.2+git12345678"
    assert isolated_version_provider.git_pin.read_text() == commit
