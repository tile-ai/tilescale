import torch
import tilelang
import tilelang.testing

import example_allgather_gemm_overlapped
import example_reduce_scatter
import example_gemm_rs_overlapped
import example_sp_ag_attention_intra_node
import example_pre_attn_all2all_intranode
import example_pre_attn_all2all_transpose_intranode
import example_post_attn_all2all_transpose_intranode


@tilelang.testing.requires_distributed
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_allgather_gemm_overlapped():
    torch.multiprocessing.spawn(example_allgather_gemm_overlapped.main, args=(2, None), nprocs=2)


@tilelang.testing.requires_distributed
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_reduce_scatter():
    torch.multiprocessing.spawn(example_reduce_scatter.main, args=(2, None), nprocs=2)


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


@tilelang.testing.requires_distributed
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_pre_attn_all2all_intranode():
    torch.multiprocessing.spawn(example_pre_attn_all2all_intranode.main, args=(2, None), nprocs=2)


@tilelang.testing.requires_distributed
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_pre_attn_all2all_transpose_intranode():
    torch.multiprocessing.spawn(example_pre_attn_all2all_transpose_intranode.main, args=(2, None), nprocs=2)


@tilelang.testing.requires_distributed
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_post_attn_all2all_transpose_intranode():
    torch.multiprocessing.spawn(example_post_attn_all2all_transpose_intranode.main, args=(2, None), nprocs=2)
