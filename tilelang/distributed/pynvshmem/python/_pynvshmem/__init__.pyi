from __future__ import annotations

from typing import Sequence
import numpy as np
import torch


def nvshmemx_cumodule_init(module: np.intp) -> None:
    ...


def nvshmemx_cumodule_finalize(module: np.intp) -> None:
    ...


def nvshmem_my_pe() -> np.int32:
    ...


def nvshmem_n_pes() -> np.int32:
    ...


def nvshmem_team_my_pe(team: np.int32) -> np.int32:
    ...


def nvshmem_team_n_pes(team: np.int32) -> np.int32:
    ...


def nvshmem_malloc(size: np.uint) -> np.intp:
    ...


def nvshmem_ptr(ptr, peer):
    ...


def nvshmemx_mc_ptr(team, ptr):
    """ DON'T CALL this function if NVLS is not used on NVSHMEM 3.2.5!!!
    even nvshmem official doc say that it returns a nullptr(https://docs.nvidia.com/nvshmem/api/gen/api/setup.html?highlight=nvshmemx_mc_ptr#nvshmemx-mc-ptr), it actually core dump without any notice. use this function only when you are sure NVLS is used.
    here is an issue: https://forums.developer.nvidia.com/t/how-do-i-query-if-nvshmemx-mc-ptr-is-supported-nvshmemx-mc-ptr-core-dump-if-nvls-is-not-used/328986
    """
    ...


def nvshmemx_get_uniqueid() -> bytes:
    ...


def nvshmemx_init_attr_with_uniqueid(rank: np.int32, nranks: np.int32, unique_id: bytes) -> None:
    ...


def nvshmem_int_p(ptr: np.intp, src: np.int32, dst: np.int32) -> None:
    ...


def nvshmem_barrier_all():
    ...


def nvshmemx_barrier_all_on_stream(stream: np.intp):
    ...


def nvshmem_getmem(dest: np.intp, source: np.intp, nelems: int, pe: int):
    ...


def nvshmem_putmem(dest: np.intp, source: np.intp, nelems: int, pe: int):
    ...


def nvshmemx_getmem_on_stream(dest: np.intp, source: np.intp, nelems: int, pe: int,
                              stream: np.intp):
    ...


def nvshmemx_putmem_on_stream(dest: np.intp, source: np.intp, nelems: int, pe: int,
                              stream: np.intp):
    ...


def nvshmemx_putmem_signal_on_stream(dest: np.intp, source: np.intp, nelems: int, sig_add: np.intp,
                                     signal: int, sig_op: int, pe: int, stream: np.intp):
    ...


# torch related
def nvshmem_create_tensor(shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    ...


def nvshmem_create_tensor_list_intra_node(shape: Sequence[int],
                                          dtype: torch.dtype) -> list[torch.Tensor]:
    ...
