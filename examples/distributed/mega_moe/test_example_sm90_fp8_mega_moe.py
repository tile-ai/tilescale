from __future__ import annotations

import argparse

import tilelang.testing
from testing.python.distributed._utils import distributed_test

import example_sm90_fp8_mega_moe


@distributed_test(nprocs=4, require_fabric=True)
def test_example_sm90_fp8_mega_moe(local_rank: int, num_ranks: int):
    args = argparse.Namespace(
        num_processes=num_ranks,
        model_config="smoke",
        num_tokens=32,
        capacity=64,
        activation_clamp=10.0,
        seed=0,
        diff_tol=0.01,
        warmup=0,
        rep=0,
        check=True,
        print_source=False,
    )
    example_sm90_fp8_mega_moe.main(local_rank, num_ranks, args)


if __name__ == "__main__":
    tilelang.testing.main()
