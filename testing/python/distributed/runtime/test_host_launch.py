from __future__ import annotations

import pytest
import torch

import tilelang.testing
from tilelang.distributed import host


def _mock_process_group(monkeypatch):
    captured = {}

    def init_process_group(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(host.dist, "init_process_group", init_process_group)
    monkeypatch.setattr(host.dist, "get_rank", lambda: captured["rank"])
    monkeypatch.setattr(host.dist, "get_world_size", lambda: captured["world_size"])
    monkeypatch.setattr(host.torch.cuda, "set_device", lambda device: captured.setdefault("device", device))
    return captured


def _clear_launcher_env(monkeypatch):
    for name in (
        "GROUP_RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "NNODES",
        "NODE_RANK",
        "RANK",
        "WORLD_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_legacy_node_count_environment_is_rejected(monkeypatch):
    _clear_launcher_env(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "7")

    with pytest.raises(NotImplementedError, match="LOCAL_WORLD_SIZE"):
        host.init_dist(local_rank=1, num_local_ranks=2)


def test_local_spawn_uses_explicit_local_rank_and_size(monkeypatch):
    _clear_launcher_env(monkeypatch)
    captured = _mock_process_group(monkeypatch)

    rank, world_size, group = host.init_dist(local_rank=1, num_local_ranks=2)

    assert captured["rank"] == rank == 1
    assert captured["world_size"] == world_size == 2
    assert captured["device"] == 1
    assert group is host.dist.group.WORLD


def test_single_node_torchrun_contract(monkeypatch):
    _clear_launcher_env(monkeypatch)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("GROUP_RANK", "0")
    captured = _mock_process_group(monkeypatch)

    rank, world_size, _ = host.init_dist(local_rank=1, num_local_ranks=2)

    assert captured["rank"] == rank == 1
    assert captured["world_size"] == world_size == 2


def test_multi_node_launch_is_rejected(monkeypatch):
    _clear_launcher_env(monkeypatch)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("GROUP_RANK", "1")

    with pytest.raises(NotImplementedError, match="single-node"):
        host.init_dist(local_rank=0, num_local_ranks=2)


def test_signal_helpers_reject_invalid_host_inputs():
    with pytest.raises(ValueError, match="CUDA tensor"):
        host.set_signal(torch.zeros(1, dtype=torch.int32), 1)
    with pytest.raises(TypeError, match="dtype"):
        host.wait_eq(torch.zeros(1, dtype=torch.float32), 1)
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        host.set_signal(torch.zeros(1, dtype=torch.int32), -1)


@tilelang.testing.requires_cuda
def test_signal_helpers_validate_and_dispatch_without_driver_access(monkeypatch):
    writes = []
    waits = []

    class FakeStream:
        device = torch.device("cuda:0")
        cuda_stream = 123

    tensor32 = torch.zeros(1, dtype=torch.uint32, device="cuda:0")
    tensor64 = torch.zeros(1, dtype=torch.int64, device="cuda:0")
    monkeypatch.setattr(host.cuda, "cuStreamWriteValue32", lambda *args: (writes.append(args),))
    monkeypatch.setattr(host.cuda, "cuStreamWaitValue64", lambda *args: (waits.append(args),))
    monkeypatch.setattr(host, "CUDA_CHECK", lambda _err: None)

    host.set_signal(tensor32, 7, FakeStream())
    host.wait_eq(tensor64, 9, FakeStream(), require_i64=True)

    assert writes[0][0:3] == (123, tensor32.data_ptr(), 7)
    assert waits[0][0:3] == (123, tensor64.data_ptr(), 9)

    with pytest.raises(ValueError, match="at least one element"):
        host.set_signal(torch.empty(0, dtype=torch.int32, device="cuda:0"), 1, FakeStream())
    with pytest.raises(ValueError, match="contiguous"):
        host.wait_eq(torch.zeros((2, 2), dtype=torch.int32, device="cuda:0").T, 1, FakeStream())
