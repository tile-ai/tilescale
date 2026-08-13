from __future__ import annotations

import argparse

import tilelang.testing
from testing.python.distributed._utils import distributed_test

import example_sm90_fp8_mega_moe


def test_custom_model_config_and_schedule():
    args = argparse.Namespace(
        model_config="smoke",
        hidden=2560,
        intermediate_hidden=1536,
        num_experts=64,
        num_topk=4,
    )
    model_name, model = example_sm90_fp8_mega_moe.resolve_model_config(args)
    assert model_name == "custom"
    assert model == {
        "hidden": 2560,
        "intermediate_hidden": 1536,
        "num_experts": 64,
        "num_topk": 4,
    }

    family, l1, l2 = example_sm90_fp8_mega_moe.select_manual_warp_configs(
        model["hidden"],
        model["intermediate_hidden"],
        num_tokens=64,
        num_topk=model["num_topk"],
        num_experts_per_rank=16,
        num_sms=132,
    )
    assert family == "generic"
    assert l1 == {
        "block_m": 64,
        "block_n": 256,
        "block_k": 128,
        "threads": 384,
        "pipeline_stages": 5,
        "num_experts_per_wave": 16,
    }
    assert l2 == {**l1, "pipeline_stages": 3}

    family, l1, l2 = example_sm90_fp8_mega_moe.select_manual_warp_configs(
        4096, 2048, num_tokens=128, num_topk=6, num_experts_per_rank=32, num_sms=132
    )
    assert family == "compact"
    assert l1["pipeline_stages"] == l2["pipeline_stages"] == 3
    assert l1["num_experts_per_wave"] == l2["num_experts_per_wave"] == 4

    family, l1, l2 = example_sm90_fp8_mega_moe.select_manual_warp_configs(
        7168, 3072, num_tokens=128, num_topk=6, num_experts_per_rank=48, num_sms=132
    )
    assert family == "wide"
    assert l1["pipeline_stages"] == 4
    assert l2["pipeline_stages"] == 3
    assert l1["num_experts_per_wave"] == l2["num_experts_per_wave"] == 4
    assert example_sm90_fp8_mega_moe.normalize_experts_per_wave(50, 16) == 25


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
