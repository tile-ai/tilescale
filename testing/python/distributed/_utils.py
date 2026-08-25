from __future__ import annotations

from collections.abc import Callable
import importlib
import os
import socket

import pytest
import torch
import torch.multiprocessing

import tilelang.testing

_DISTRIBUTED_WORKERS: dict[tuple[str, str], Callable[[int, int], None]] = {}


def distributed_test(
    *,
    nprocs: int | None = 2,
    require_fabric: bool = False,
    require_multicast: bool = False,
):
    """Decorate a worker(local_rank, num_ranks) as a distributed pytest case."""

    def decorator(worker: Callable[[int, int], None]):
        worker_key = (worker.__module__, worker.__qualname__)
        _DISTRIBUTED_WORKERS[worker_key] = worker

        @tilelang.testing.requires_cuda
        def test_wrapper():
            resolved_nprocs = _resolve_nprocs(nprocs)
            _skip_if_not_enough_gpus(resolved_nprocs)
            if require_fabric:
                _skip_if_no_fabric()
            if require_multicast:
                _skip_if_no_multicast()
            master_port = None if "MASTER_PORT" in os.environ or "TILESCALE_MASTER_PORT" in os.environ else _find_free_port()
            torch.multiprocessing.spawn(
                _distributed_worker_entry,
                args=(worker_key, resolved_nprocs, master_port),
                nprocs=resolved_nprocs,
            )

        test_wrapper.__name__ = worker.__name__
        test_wrapper.__doc__ = worker.__doc__
        return test_wrapper

    return decorator


def _distributed_worker_entry(local_rank: int, worker_key: tuple[str, str], num_ranks: int, master_port: int | None):
    if master_port is not None:
        os.environ["TILESCALE_MASTER_PORT"] = str(master_port)
    module_name, _ = worker_key
    importlib.import_module(module_name)
    worker = _DISTRIBUTED_WORKERS[worker_key]
    worker(local_rank, num_ranks)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _resolve_nprocs(nprocs: int | None) -> int:
    if nprocs is None:
        return torch.cuda.device_count()
    return nprocs


def _skip_if_not_enough_gpus(nprocs: int):
    num_gpus = torch.cuda.device_count()
    if num_gpus < nprocs:
        pytest.skip(f"Need >= {nprocs} GPUs, found {num_gpus}")


def _skip_if_no_fabric():
    from tilelang.distributed.shared_memory import _supports_vmm_fabric

    if not _supports_vmm_fabric():
        pytest.skip("VMM fabric unavailable; check GPU, driver, and IMEX configuration")


def _skip_if_no_multicast():
    from tilelang.distributed.shared_memory import _supports_multicast, _supports_multicast_posix

    if not (_supports_multicast() or _supports_multicast_posix()):
        pytest.skip("NVSwitch multicast unavailable; check GPU, driver, fabric, and IMEX configuration")
