import argparse
import tilelang.testing
import torch
import torch.multiprocessing

import example_tilescale_copy


@tilelang.testing.requires_cuda
def test_example_tilescale_copy_simt_push_tile():
    args = argparse.Namespace(M=1024, N=1024, kernel='simt_push_tile')
    torch.multiprocessing.spawn(example_tilescale_copy.main, args=(2, args), nprocs=2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_example_tilescale_copy_tma_load_tile():
    args = argparse.Namespace(M=1024, N=1024, kernel='tma_load_tile')
    torch.multiprocessing.spawn(example_tilescale_copy.main, args=(2, args), nprocs=2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_example_tilescale_copy_tma_store_tile():
    args = argparse.Namespace(M=1024, N=1024, kernel='tma_store_tile')
    torch.multiprocessing.spawn(example_tilescale_copy.main, args=(2, args), nprocs=2)


if __name__ == "__main__":
    tilelang.testing.main()
