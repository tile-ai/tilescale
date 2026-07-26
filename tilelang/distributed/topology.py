from __future__ import annotations

import functools
import re
import subprocess
import warnings
from threading import Lock

import torch

__all__ = [
    "NvidiaSmiUtil",
    "ensure_nvml_initialized",
    "has_fullmesh_nvlink",
    "has_fullmesh_nvlink_pynvml",
]

_pynvml_initialized = False
_lock = Lock()
_GPU_LABEL_RE = re.compile(r"^GPU(?P<index>\d+)$")
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _normalize_gpu_uuid(value) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized.startswith("gpu-"):
        normalized = normalized[4:]
    return normalized


def _normalize_pci_bus_id(value) -> str | None:
    if value is None:
        return None
    try:
        domain, bus, device = str(value).strip().lower().rsplit(":", 2)
    except ValueError:
        return None
    if "." not in device:
        device = f"{device}.0"
    return f"{domain[-4:].zfill(4)}:{bus.zfill(2)}:{device}"


def _visible_device_identity(device_index: int) -> tuple[str | None, str | None]:
    device_count = torch.cuda.device_count()
    if not 0 <= device_index < device_count:
        raise IndexError(f"CUDA device index {device_index} is outside the visible range [0, {device_count})")

    properties = torch.cuda.get_device_properties(device_index)
    uuid = _normalize_gpu_uuid(getattr(properties, "uuid", None))

    pci_value = getattr(properties, "pci_bus_id", None)
    if isinstance(pci_value, str) and ":" in pci_value:
        pci_bus_id = _normalize_pci_bus_id(pci_value)
    else:
        domain = getattr(properties, "pci_domain_id", None)
        device = getattr(properties, "pci_device_id", None)
        if all(isinstance(value, int) for value in (domain, pci_value, device)):
            pci_bus_id = _normalize_pci_bus_id(f"{domain:08x}:{pci_value:02x}:{device:02x}.0")
        else:
            pci_bus_id = None

    if uuid is None and pci_bus_id is None:
        raise RuntimeError(f"CUDA device {device_index} exposes neither a UUID nor a PCI bus ID")
    return uuid, pci_bus_id


def _nvml_handle_for_visible_device(pynvml, device_index: int):
    uuid, pci_bus_id = _visible_device_identity(device_index)
    errors = []

    if uuid is not None and hasattr(pynvml, "nvmlDeviceGetHandleByUUID"):
        selector = uuid if uuid.startswith("mig-") else f"GPU-{uuid}"
        try:
            return pynvml.nvmlDeviceGetHandleByUUID(selector)
        except Exception as exc:
            errors.append(exc)

    if pci_bus_id is not None and hasattr(pynvml, "nvmlDeviceGetHandleByPciBusId"):
        try:
            return pynvml.nvmlDeviceGetHandleByPciBusId(pci_bus_id)
        except Exception as exc:
            errors.append(exc)

    if errors:
        raise errors[-1]
    raise RuntimeError(f"Unable to map visible CUDA device {device_index} to an NVML handle")


def _parse_nvidia_smi_inventory(output: str) -> tuple[dict[str, str], dict[str, str]]:
    labels_by_uuid: dict[str, str] = {}
    labels_by_pci: dict[str, str] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        index, uuid, pci_bus_id = fields
        if not index.isdigit():
            continue
        label = f"GPU{int(index)}"
        normalized_uuid = _normalize_gpu_uuid(uuid)
        normalized_pci = _normalize_pci_bus_id(pci_bus_id)
        if normalized_uuid is not None:
            labels_by_uuid[normalized_uuid] = label
        if normalized_pci is not None:
            labels_by_pci[normalized_pci] = label
    return labels_by_uuid, labels_by_pci


def _visible_nvidia_smi_labels() -> list[str]:
    identities = [_visible_device_identity(index) for index in range(torch.cuda.device_count())]
    if not identities:
        return []

    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    labels_by_uuid, labels_by_pci = _parse_nvidia_smi_inventory(output)

    labels = []
    for uuid, pci_bus_id in identities:
        label = labels_by_uuid.get(uuid) if uuid is not None else None
        if label is None and pci_bus_id is not None:
            label = labels_by_pci.get(pci_bus_id)
        if label is None or label in labels:
            return []
        labels.append(label)
    return labels


def ensure_nvml_initialized() -> None:
    global _pynvml_initialized
    if _pynvml_initialized:
        return
    with _lock:
        if not _pynvml_initialized:
            import pynvml

            pynvml.nvmlInit()
            _pynvml_initialized = True


@functools.lru_cache(maxsize=1)
def has_fullmesh_nvlink_pynvml() -> bool:
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        return False
    if num_devices == 1:
        return True

    ensure_nvml_initialized()
    import pynvml

    try:
        handles = [_nvml_handle_for_visible_device(pynvml, index) for index in range(num_devices)]
        for current in range(num_devices):
            for remote in range(num_devices):
                if current == remote:
                    continue
                status = pynvml.nvmlDeviceGetP2PStatus(
                    handles[current],
                    handles[remote],
                    pynvml.NVML_P2P_CAPS_INDEX_NVLINK,
                )
                if status != pynvml.NVML_P2P_STATUS_OK:
                    return False
        return True
    except pynvml.NVMLError_NotSupported:
        return False


class NvidiaSmiUtil:
    @staticmethod
    def get_nvlink_adjacency_matrix() -> list[list[int]]:
        visible_labels = _visible_nvidia_smi_labels()
        if not visible_labels:
            return []

        output = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
        column_labels = None
        rows: dict[str, dict[str, str]] = {}
        for line in output.splitlines():
            parts = _ANSI_ESCAPE_RE.sub("", line).split()
            if not parts:
                continue
            if column_labels is None:
                candidates = [part for part in parts if _GPU_LABEL_RE.fullmatch(part)]
                if candidates and "X" not in parts[1:]:
                    column_labels = candidates
                continue
            if column_labels is None or not _GPU_LABEL_RE.fullmatch(parts[0]):
                continue
            links = parts[1 : len(column_labels) + 1]
            if len(links) == len(column_labels):
                rows[parts[0]] = dict(zip(column_labels, links))

        if column_labels is None or any(label not in column_labels or label not in rows for label in visible_labels):
            return []

        matrix = [[-1 for _ in visible_labels] for _ in visible_labels]
        for row, row_label in enumerate(visible_labels):
            for column, column_label in enumerate(visible_labels):
                link = rows[row_label].get(column_label)
                if link is None or (row == column and link != "X"):
                    return []
                if row != column and link.startswith("NV"):
                    matrix[row][column] = 1
        return matrix

    @staticmethod
    def get_gpu_numa_node(gpu_index: int = 0) -> int:
        try:
            uuid, pci_id = _visible_device_identity(gpu_index)
            if pci_id is None:
                selector = uuid if uuid.startswith("mig-") else f"GPU-{uuid}"
                pci_id = _normalize_pci_bus_id(
                    subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=pci.bus_id",
                            "--format=csv,noheader,nounits",
                            "-i",
                            selector,
                        ],
                        text=True,
                    ).strip()
                )
            if pci_id is None:
                return -1
            domain, bus, device = pci_id.rsplit(":", 2)
            domain = domain[-4:]
            numa_path = f"/sys/bus/pci/devices/{domain}:{bus.lower()}:{device.lower()}/numa_node"
            with open(numa_path) as numa_file:
                numa_node = int(numa_file.read().strip())
            return numa_node if numa_node >= 0 else 0
        except (IndexError, OSError, RuntimeError, ValueError, subprocess.SubprocessError):
            return -1


@functools.lru_cache(maxsize=1)
def has_fullmesh_nvlink() -> bool:
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        return False
    if num_devices == 1:
        return True

    try:
        return has_fullmesh_nvlink_pynvml()
    except Exception:
        try:
            matrix = NvidiaSmiUtil.get_nvlink_adjacency_matrix()
        except Exception:
            return False
        if len(matrix) != num_devices or any(len(row) != num_devices for row in matrix):
            return False
        has_nvlink = any(value == 1 for row in matrix for value in row)
        is_fullmesh = all(row == column or value == 1 for row, values in enumerate(matrix) for column, value in enumerate(values))
        if has_nvlink and not is_fullmesh:
            warnings.warn(
                "NVLink is present but the visible GPUs are not fully connected; full-mesh-only distributed kernels are unsupported.",
                RuntimeWarning,
                stacklevel=2,
            )
        return is_fullmesh
