from __future__ import annotations

import io
import subprocess
from types import SimpleNamespace

import pynvml

from tilelang.distributed import topology


def _properties(uuid: str, bus: int, device: int = 0, domain: int = 0):
    return SimpleNamespace(uuid=uuid, pci_bus_id=bus, pci_device_id=device, pci_domain_id=domain)


def _mock_visible_devices(monkeypatch, properties):
    monkeypatch.setattr(topology.torch.cuda, "device_count", lambda: len(properties))
    monkeypatch.setattr(topology.torch.cuda, "get_device_properties", properties.__getitem__)
    topology.has_fullmesh_nvlink.cache_clear()
    topology.has_fullmesh_nvlink_pynvml.cache_clear()


def test_nvidia_smi_topology_parser(monkeypatch):
    inventory = """\
0, GPU-00000000-0000-0000-0000-000000000000, 00000000:01:00.0
1, GPU-11111111-1111-1111-1111-111111111111, 00000000:02:00.0
2, GPU-22222222-2222-2222-2222-222222222222, 00000000:03:00.0
"""
    topology_output = """\
            \x1b[4mGPU0    GPU1    GPU2    CPU Affinity\x1b[0m
    GPU0     X      NV18    SYS     0-31
    GPU1    NV18     X      NV18    0-31
    GPU2    SYS     NV18     X      32-63
    """
    _mock_visible_devices(
        monkeypatch,
        [
            _properties("00000000-0000-0000-0000-000000000000", 1),
            _properties("22222222-2222-2222-2222-222222222222", 3),
        ],
    )

    def check_output(command, **_kwargs):
        return topology_output if command[1:3] == ["topo", "-m"] else inventory

    monkeypatch.setattr(topology.subprocess, "check_output", check_output)

    assert topology.NvidiaSmiUtil.get_nvlink_adjacency_matrix() == [
        [-1, -1],
        [-1, -1],
    ]


def test_single_visible_device_is_fullmesh(monkeypatch):
    _mock_visible_devices(monkeypatch, [_properties("0" * 32, 1)])

    assert topology.has_fullmesh_nvlink_pynvml()
    assert topology.has_fullmesh_nvlink()


def test_no_visible_device_is_not_fullmesh(monkeypatch):
    _mock_visible_devices(monkeypatch, [])

    assert not topology.has_fullmesh_nvlink_pynvml()
    assert not topology.has_fullmesh_nvlink()


def test_nvml_uses_visible_uuid_instead_of_physical_index(monkeypatch):
    uuids = ["44444444-4444-4444-4444-444444444444", "55555555-5555-5555-5555-555555555555"]
    _mock_visible_devices(monkeypatch, [_properties(uuids[0], 4), _properties(uuids[1], 5)])
    requested = []

    monkeypatch.setattr(topology, "ensure_nvml_initialized", lambda: None)
    monkeypatch.setattr(
        pynvml,
        "nvmlDeviceGetHandleByUUID",
        lambda uuid: requested.append(uuid) or f"handle:{uuid}",
    )
    monkeypatch.setattr(pynvml, "nvmlDeviceGetP2PStatus", lambda *_args: pynvml.NVML_P2P_STATUS_OK)

    assert topology.has_fullmesh_nvlink_pynvml()
    assert requested == [f"GPU-{uuid}" for uuid in uuids]


def test_nvml_falls_back_to_visible_pci_bus_id(monkeypatch):
    _mock_visible_devices(monkeypatch, [_properties("first", 4), _properties("second", 5)])
    requested = []

    monkeypatch.setattr(topology, "ensure_nvml_initialized", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByUUID", lambda _uuid: (_ for _ in ()).throw(RuntimeError("no UUID")))
    monkeypatch.setattr(
        pynvml,
        "nvmlDeviceGetHandleByPciBusId",
        lambda pci: requested.append(pci) or f"handle:{pci}",
    )
    monkeypatch.setattr(pynvml, "nvmlDeviceGetP2PStatus", lambda *_args: pynvml.NVML_P2P_STATUS_OK)

    assert topology.has_fullmesh_nvlink_pynvml()
    assert requested == ["0000:04:00.0", "0000:05:00.0"]


def test_fallback_fullmesh_uses_only_visible_devices(monkeypatch):
    inventory = """\
0, GPU-00000000-0000-0000-0000-000000000000, 00000000:01:00.0
1, GPU-11111111-1111-1111-1111-111111111111, 00000000:02:00.0
2, GPU-22222222-2222-2222-2222-222222222222, 00000000:03:00.0
"""
    topology_output = """\
            GPU0    GPU1    GPU2    CPU Affinity
    GPU0     X      SYS     NV18    0-31
    GPU1    SYS     X      SYS     0-31
    GPU2    NV18    SYS     X      32-63
    """
    _mock_visible_devices(
        monkeypatch,
        [
            _properties("22222222-2222-2222-2222-222222222222", 3),
            _properties("00000000-0000-0000-0000-000000000000", 1),
        ],
    )
    monkeypatch.setattr(topology, "has_fullmesh_nvlink_pynvml", lambda: (_ for _ in ()).throw(RuntimeError("NVML unavailable")))

    def check_output(command, **_kwargs):
        return topology_output if command[1:3] == ["topo", "-m"] else inventory

    monkeypatch.setattr(topology.subprocess, "check_output", check_output)

    assert topology.NvidiaSmiUtil.get_nvlink_adjacency_matrix() == [[-1, 1], [1, -1]]
    assert topology.has_fullmesh_nvlink()


def test_empty_or_malformed_fallback_is_not_fullmesh(monkeypatch):
    _mock_visible_devices(
        monkeypatch,
        [_properties("00000000-0000-0000-0000-000000000000", 1), _properties("11111111-1111-1111-1111-111111111111", 2)],
    )
    monkeypatch.setattr(topology, "has_fullmesh_nvlink_pynvml", lambda: (_ for _ in ()).throw(RuntimeError("NVML unavailable")))
    monkeypatch.setattr(topology.NvidiaSmiUtil, "get_nvlink_adjacency_matrix", lambda: [])

    assert not topology.has_fullmesh_nvlink()


def test_fallback_exception_is_not_fullmesh(monkeypatch):
    _mock_visible_devices(
        monkeypatch,
        [_properties("00000000-0000-0000-0000-000000000000", 1), _properties("11111111-1111-1111-1111-111111111111", 2)],
    )
    monkeypatch.setattr(topology, "has_fullmesh_nvlink_pynvml", lambda: (_ for _ in ()).throw(RuntimeError("NVML unavailable")))
    monkeypatch.setattr(
        topology.NvidiaSmiUtil,
        "get_nvlink_adjacency_matrix",
        lambda: (_ for _ in ()).throw(subprocess.SubprocessError("nvidia-smi failed")),
    )

    assert not topology.has_fullmesh_nvlink()


def test_get_gpu_numa_node_uses_visible_pci_identity(monkeypatch):
    _mock_visible_devices(monkeypatch, [_properties("44444444-4444-4444-4444-444444444444", 0x9A)])
    opened = []

    def open_file(path):
        opened.append(path)
        return io.StringIO("3\n")

    monkeypatch.setattr("builtins.open", open_file)

    assert topology.NvidiaSmiUtil.get_gpu_numa_node(0) == 3
    assert opened == ["/sys/bus/pci/devices/0000:9a:00.0/numa_node"]
