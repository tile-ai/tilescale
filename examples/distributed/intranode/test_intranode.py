import torch
import tilelang
import tilelang.language as T
import tilelang.testing

import example_allgather_gemm_overlapped
import example_gemm_rs_overlapped
import example_sp_ag_attention_intra_node


@tilelang.testing.requires_distributed
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_allgather_gemm_overlapped():
    torch.multiprocessing.spawn(example_allgather_gemm_overlapped.main, args=(2, None), nprocs=2)


@tilelang.testing.requires_distributed
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_gemm_rs_overlapped():
    torch.multiprocessing.spawn(example_gemm_rs_overlapped.main, args=(2, None), nprocs=2)



@tilelang.testing.requires_distributed
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_sp_ag_attention_intra_node():
    torch.multiprocessing.spawn(example_sp_ag_attention_intra_node.main, args=(2, None), nprocs=2)
