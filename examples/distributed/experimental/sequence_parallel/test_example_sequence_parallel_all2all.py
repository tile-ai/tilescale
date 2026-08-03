from __future__ import annotations

import argparse

import tilelang.testing
from testing.python.distributed._utils import distributed_test

import example_post_attn_all2all_transpose
import example_pre_attn_all2all
import example_pre_attn_all2all_transpose


def _args(num_ranks: int) -> argparse.Namespace:
    return argparse.Namespace(
        num_processes=num_ranks,
        batch_size=1,
        num_heads=8,
        seq_len=64,
        head_dim=16,
        target_ctas=256,
        dtype="fp16",
        warmup=1,
        rep=1,
        atol=1e-3,
        rtol=1e-3,
        print_source=False,
    )


@distributed_test(nprocs=4)
def test_example_pre_attn_all2all(local_rank: int, num_ranks: int):
    example_pre_attn_all2all.main(local_rank, num_ranks, _args(num_ranks))


@distributed_test(nprocs=4)
def test_example_pre_attn_all2all_transpose(local_rank: int, num_ranks: int):
    example_pre_attn_all2all_transpose.main(local_rank, num_ranks, _args(num_ranks))


@distributed_test(nprocs=4)
def test_example_post_attn_all2all_transpose(local_rank: int, num_ranks: int):
    example_post_attn_all2all_transpose.main(local_rank, num_ranks, _args(num_ranks))


if __name__ == "__main__":
    tilelang.testing.main()
