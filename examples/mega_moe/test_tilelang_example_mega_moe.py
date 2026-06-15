import os
import sys

import pytest
import torch
import tilelang.testing
from testing.python.distributed._utils import distributed_test

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import reference
import tilelang_mega_moe_sm100 as mega_moe


def _has_usable_cuda_devices(num_devices: int = 1) -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < num_devices:
        return False
    try:
        for device_idx in range(num_devices):
            with torch.cuda.device(device_idx):
                probe = torch.tensor([1.0], device=f"cuda:{device_idx}")
                if probe.cpu().item() != 1.0:
                    return False
                torch.cuda.synchronize(device_idx)
        return True
    except Exception:
        return False


def _run_public_api(inputs, *, activation_clamp=10.0, group=None, use_vmm=None, num_experts=None):
    if num_experts is None:
        num_experts = inputs.l1_weights[0].size(0)
    sym_buffer = mega_moe.get_symm_buffer_for_mega_moe(
        group,
        num_experts,
        inputs.x_fp8.size(0),
        inputs.topk_idx.size(1),
        inputs.x_fp8.size(1),
        inputs.l2_weights[0].size(2) * 2,
        device=inputs.x_fp8.device,
        use_vmm=use_vmm,
    )
    try:
        mega_moe._copy_inputs_to_buffer(sym_buffer, inputs)
        y = torch.empty((inputs.x_fp8.size(0), inputs.x_fp8.size(1)), dtype=torch.bfloat16, device=inputs.x_fp8.device)
        mega_moe.fp8_fp4_mega_moe(
            y,
            inputs.l1_weights,
            inputs.l2_weights,
            sym_buffer,
            activation_clamp=activation_clamp,
        )
        return y
    finally:
        sym_buffer.destroy()


def _assert_public_api_matches_reference(
    *,
    num_tokens=16,
    hidden=256,
    intermediate_hidden=128,
    num_experts=4,
    num_topk=2,
    activation_clamp=10.0,
    masked_ratio=0.0,
    seed=0,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = reference.make_random_mega_moe_inputs(
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_experts=num_experts,
        num_topk=num_topk,
        device=device,
        seed=19,
        masked_ratio=masked_ratio,
    )
    got = _run_public_api(inputs, activation_clamp=activation_clamp)
    ref = reference.mega_moe_reference(
        inputs.x_fp8,
        inputs.x_sf,
        inputs.topk_idx,
        inputs.topk_weights,
        inputs.plain_l1_weights[0],
        inputs.plain_l1_weights[1],
        inputs.plain_l2_weights[0],
        inputs.plain_l2_weights[1],
        hidden,
        intermediate_hidden,
        num_experts,
        activation_clamp=activation_clamp,
    )
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_weight_transform_round_trip_cpu():
    inputs = reference.make_random_mega_moe_inputs(
        num_tokens=2,
        hidden=128,
        intermediate_hidden=128,
        num_experts=4,
        num_topk=2,
        device="cpu",
        seed=1,
    )
    plain_l1, plain_l2 = reference.inverse_transform_weights_for_mega_moe(inputs.l1_weights, inputs.l2_weights)
    assert torch.equal(plain_l1[0], inputs.plain_l1_weights[0])
    assert torch.equal(plain_l1[1], inputs.plain_l1_weights[1])
    assert torch.equal(plain_l2[0], inputs.plain_l2_weights[0])
    assert torch.equal(plain_l2[1], inputs.plain_l2_weights[1])


def test_functional_mega_moe_cpu():
    _assert_public_api_matches_reference(
        num_tokens=4,
        hidden=128,
        intermediate_hidden=128,
        num_experts=4,
        num_topk=2,
        activation_clamp=10.0,
        device="cpu",
        seed=2,
    )


def test_functional_mega_moe_masked_cpu():
    _assert_public_api_matches_reference(
        num_tokens=4,
        hidden=128,
        intermediate_hidden=128,
        num_experts=4,
        num_topk=2,
        activation_clamp=None,
        masked_ratio=0.5,
        device="cpu",
        seed=3,
    )


def test_cumulative_local_expert_recv_stats_cpu():
    inputs = reference.make_random_mega_moe_inputs(
        num_tokens=5,
        hidden=128,
        intermediate_hidden=128,
        num_experts=4,
        num_topk=3,
        masked_ratio=0.35,
        device="cpu",
        seed=5,
    )
    sym_buffer = mega_moe.get_symm_buffer_for_mega_moe(
        None,
        4,
        inputs.x_fp8.size(0),
        inputs.topk_idx.size(1),
        inputs.x_fp8.size(1),
        inputs.l2_weights[0].size(2) * 2,
        device="cpu",
    )
    try:
        mega_moe._copy_inputs_to_buffer(sym_buffer, inputs)
        stats = torch.tensor([7, 11, 13, 17], dtype=torch.int32)
        initial_stats = stats.clone()
        y = torch.empty((inputs.x_fp8.size(0), inputs.x_fp8.size(1)), dtype=torch.bfloat16)
        mega_moe.fp8_fp4_mega_moe(
            y,
            inputs.l1_weights,
            inputs.l2_weights,
            sym_buffer,
            cumulative_local_expert_recv_stats=stats,
            activation_clamp=10.0,
        )
        expected_counts = torch.tensor(
            [(inputs.topk_idx == expert).sum().item() for expert in range(4)],
            dtype=torch.int32,
        )
        assert torch.equal(stats, initial_stats + expected_counts)
    finally:
        sym_buffer.destroy()


def test_update_cumulative_local_expert_recv_stats_from_counts_cpu():
    local_map = mega_moe.LocalDispatchMap(
        expert_counts=torch.tensor([2, 0, 3], dtype=torch.int32),
        expert_offsets=torch.tensor([0, 2, 2, 5], dtype=torch.int32),
        token_indices=torch.empty((5,), dtype=torch.int32),
        topk_slots=torch.empty((5,), dtype=torch.int32),
        local_expert_indices=torch.empty((5,), dtype=torch.int32),
    )
    stats = torch.tensor([7, 11, 13], dtype=torch.int32)
    mega_moe._update_cumulative_local_expert_recv_stats_from_counts(local_map, stats)
    assert torch.equal(stats, torch.tensor([9, 11, 16], dtype=torch.int32))


def _assert_dispatch_map_matches_topk(local_map, topk_idx, num_experts_per_rank, local_expert_start):
    counts = local_map.expert_counts.cpu()
    offsets = local_map.expert_offsets.cpu()
    token_indices = local_map.token_indices.cpu()
    topk_slots = local_map.topk_slots.cpu()
    local_expert_indices = local_map.local_expert_indices.cpu()
    topk_idx_cpu = topk_idx.cpu()

    expected_counts = torch.tensor(
        [(topk_idx_cpu == local_expert_start + expert).sum().item() for expert in range(num_experts_per_rank)],
        dtype=torch.int32,
    )
    assert torch.equal(counts, expected_counts)
    assert offsets[0].item() == 0
    assert torch.equal(offsets[1:] - offsets[:-1], counts)

    total = offsets[-1].item()
    assert torch.all(token_indices[total:] == -1)
    assert torch.all(topk_slots[total:] == -1)
    assert torch.all(local_expert_indices[total:] == -1)
    for local_expert in range(num_experts_per_rank):
        begin = offsets[local_expert].item()
        end = offsets[local_expert + 1].item()
        for idx in range(begin, end):
            token = token_indices[idx].item()
            slot = topk_slots[idx].item()
            assert local_expert_indices[idx].item() == local_expert
            assert topk_idx_cpu[token, slot].item() == local_expert_start + local_expert


def _assert_distributed_l1_dispatch_matches_inputs(sym_buffer, distributed_map):
    local_map = distributed_map.local_map
    valid = torch.nonzero(local_map.token_indices >= 0, as_tuple=False).flatten().cpu()
    token_indices = local_map.token_indices.cpu()
    topk_slots = local_map.topk_slots.cpu()
    source_ranks = distributed_map.source_ranks.cpu()
    for pool_idx in valid.tolist():
        token = token_indices[pool_idx].item()
        slot = topk_slots[pool_idx].item()
        src_rank = source_ranks[pool_idx].item()
        peer_x = sym_buffer.peer_tensor(src_rank, "x")
        peer_x_sf = sym_buffer.peer_tensor(src_rank, "x_sf")
        peer_topk_weights = sym_buffer.peer_tensor(src_rank, "topk_weights")
        assert torch.equal(sym_buffer.l1_acts[pool_idx].cpu(), peer_x[token].cpu())
        assert torch.equal(sym_buffer.l1_acts_sf[pool_idx].cpu(), peer_x_sf[token].cpu())
        assert sym_buffer.l1_topk_weights[pool_idx].cpu().item() == peer_topk_weights[token, slot].cpu().item()


def test_build_local_dispatch_map_cpu():
    topk_idx = torch.tensor(
        [
            [2, 3, -1],
            [4, 2, 5],
            [3, 0, 1],
            [-1, 3, 4],
        ],
        dtype=torch.int64,
    )
    local_map = mega_moe.build_local_dispatch_map(
        topk_idx,
        num_experts_per_rank=3,
        local_expert_start=2,
    )
    _assert_dispatch_map_matches_topk(local_map, topk_idx, 3, 2)


def test_build_local_dispatch_map_respects_active_tokens_cpu():
    topk_idx = torch.tensor(
        [
            [0, 1],
            [2, 0],
            [1, -1],
            [3, 2],
            [0, 2],
        ],
        dtype=torch.int64,
    )
    local_map = mega_moe.build_local_dispatch_map(
        topk_idx,
        num_experts_per_rank=4,
        local_expert_start=0,
        num_tokens=3,
    )
    _assert_dispatch_map_matches_topk(local_map, topk_idx[:3], 4, 0)
    assert (local_map.token_indices >= 3).sum().item() == 0


def test_symm_buffer_exposes_deepgemm_workspace_views_cpu():
    num_ranks = 1
    num_experts = 4
    max_tokens = 384
    num_topk = 2
    hidden = 128
    intermediate_hidden = 128
    sym_buffer = mega_moe.get_symm_buffer_for_mega_moe(
        None,
        num_experts,
        max_tokens,
        num_topk,
        hidden,
        intermediate_hidden,
        device="cpu",
    )
    try:
        layout = sym_buffer.layout
        assert sym_buffer.workspace.numel() == layout.workspace_bytes
        assert sym_buffer.workspace_barrier.numel() == mega_moe.MEGAMOE_WORKSPACE_BARRIER_SLOTS * 4
        assert sym_buffer.workspace_barrier.view(torch.int32).numel() == mega_moe.MEGAMOE_WORKSPACE_BARRIER_SLOTS
        assert sym_buffer.peer_tensor(0, "workspace").data_ptr() == sym_buffer.workspace.data_ptr()
    finally:
        sym_buffer.destroy()


def test_public_local_path_prefers_megakernel_before_reference_fallback_cpu(monkeypatch):
    sym_buffer = mega_moe.get_symm_buffer_for_mega_moe(
        None,
        2,
        4,
        1,
        256,
        128,
        device="cpu",
    )
    sym_buffer.device = torch.device("cuda")
    y = torch.empty((4, 256), dtype=torch.bfloat16)
    stats = torch.zeros((2,), dtype=torch.int32)
    seen = {}
    megakernel_map = mega_moe.LocalDispatchMap(
        expert_counts=torch.tensor([2, 1], dtype=torch.int32),
        expert_offsets=torch.tensor([0, 128, 129], dtype=torch.int32),
        token_indices=torch.empty((sym_buffer.l1_acts.shape[0],), dtype=torch.int32),
        topk_slots=torch.empty((sym_buffer.l1_acts.shape[0],), dtype=torch.int32),
        local_expert_indices=torch.empty((sym_buffer.l1_acts.shape[0],), dtype=torch.int32),
    )

    def fake_megakernel(*args, **kwargs):
        seen["megakernel"] = kwargs
        return megakernel_map

    monkeypatch.setattr(mega_moe, "_run_local_dispatch_l1_l2_combine_sm100_megakernel", fake_megakernel)
    monkeypatch.setattr(mega_moe, "_is_sm100_cuda_device", lambda *args, **kwargs: True)
    try:
        result = mega_moe.fp8_fp4_mega_moe(
            y,
            (torch.empty((2, 256, 128), dtype=torch.int8), torch.empty((2, 256), dtype=torch.int32)),
            (torch.empty((2, 256, 64), dtype=torch.int8), torch.empty((2, 256), dtype=torch.int32)),
            sym_buffer,
            cumulative_local_expert_recv_stats=stats,
        )
    finally:
        sym_buffer.destroy()
    assert result is y
    assert "megakernel" in seen
    assert seen["megakernel"]["num_tokens"] == 4
    assert torch.equal(stats, torch.tensor([2, 1], dtype=torch.int32))


def test_public_distributed_path_prefers_megakernel_before_reference_fallback_cpu(monkeypatch):
    sym_buffer = mega_moe.get_symm_buffer_for_mega_moe(
        None,
        4,
        4,
        1,
        256,
        128,
        device="cpu",
    )
    sym_buffer.num_ranks = 2
    sym_buffer.rank = 0
    sym_buffer.group = object()
    sym_buffer.device = torch.device("cuda")
    y = torch.empty((4, 256), dtype=torch.bfloat16)
    stats = torch.zeros((2,), dtype=torch.int32)
    seen = {}
    megakernel_map = mega_moe.DistributedDispatchMap(
        local_map=mega_moe.LocalDispatchMap(
            expert_counts=torch.tensor([2, 1], dtype=torch.int32),
            expert_offsets=torch.tensor([0, 128, 129], dtype=torch.int32),
            token_indices=torch.empty((sym_buffer.l1_acts.shape[0],), dtype=torch.int32),
            topk_slots=torch.empty((sym_buffer.l1_acts.shape[0],), dtype=torch.int32),
            local_expert_indices=torch.empty((sym_buffer.l1_acts.shape[0],), dtype=torch.int32),
        ),
        source_ranks=torch.empty((sym_buffer.l1_acts.shape[0],), dtype=torch.int32),
    )

    def fake_megakernel(*args, **kwargs):
        seen["megakernel"] = kwargs
        return megakernel_map

    monkeypatch.setattr(
        mega_moe,
        "_run_distributed_dispatch_l1_l2_remote_combine_sm100_megakernel",
        fake_megakernel,
    )
    monkeypatch.setattr(mega_moe, "_uses_distributed_symmetric_path", lambda *args, **kwargs: True)
    monkeypatch.setattr(mega_moe.torch.cuda, "synchronize", lambda *args, **kwargs: None)
    try:
        result = mega_moe.fp8_fp4_mega_moe(
            y,
            (torch.empty((2, 256, 128), dtype=torch.int8), torch.empty((2, 256), dtype=torch.int32)),
            (torch.empty((2, 256, 64), dtype=torch.int8), torch.empty((2, 256), dtype=torch.int32)),
            sym_buffer,
            cumulative_local_expert_recv_stats=stats,
        )
    finally:
        sym_buffer.destroy()
    assert result is y
    assert "megakernel" in seen
    assert seen["megakernel"]["num_tokens"] == 4
    assert torch.equal(stats, torch.tensor([2, 1], dtype=torch.int32))


def _requires_opt_in_dist(fn):
    fn = pytest.mark.skipif(
        os.environ.get("TILESCALE_RUN_MEGAMOE_DIST_TEST") != "1",
        reason="Set TILESCALE_RUN_MEGAMOE_DIST_TEST=1 to run the distributed MegaMoE contract test",
    )(fn)
    fn = pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Need two visible CUDA devices for distributed MegaMoE contract test",
    )(fn)
    return fn


def _assert_distributed_combine_slots_written(sym_buffer, topk_idx: torch.Tensor, num_tokens: int) -> None:
    valid = topk_idx[:num_tokens] >= 0
    for token in range(num_tokens):
        for slot in range(topk_idx.size(1)):
            if bool(valid[token, slot].item()):
                assert torch.isfinite(sym_buffer.combine_acts[slot, token].float()).all()


def _run_functional_mega_moe_distributed_contract(
    local_rank: int,
    num_ranks: int,
    *,
    hidden: int,
    num_topk: int,
    seed: int,
):
    import torch.distributed as dist

    from tilelang.distributed.host import init_dist

    rank, world_size, group = init_dist(local_rank, num_ranks)
    assert rank == local_rank
    assert world_size == num_ranks

    intermediate_hidden = 128
    num_tokens = 4
    num_experts_per_rank = 2
    num_experts = num_experts_per_rank * num_ranks
    device = torch.device(f"cuda:{local_rank}")

    global_inputs = reference.make_random_mega_moe_inputs(
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_experts=num_experts,
        num_topk=num_topk,
        device=device,
        seed=seed,
        masked_ratio=0.25,
    )
    local_slice = slice(rank * num_experts_per_rank, (rank + 1) * num_experts_per_rank)
    sym_buffer = mega_moe.get_symm_buffer_for_mega_moe(
        group,
        num_experts,
        num_tokens,
        num_topk,
        hidden,
        intermediate_hidden,
        device=device,
    )
    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=device)
    try:
        mega_moe._copy_inputs_to_buffer(sym_buffer, global_inputs)
        sym_buffer.combine_acts.fill_(float("nan"))
        dist.barrier(group)
        mega_moe.fp8_fp4_mega_moe(
            y,
            (
                global_inputs.l1_weights[0][local_slice].contiguous(),
                global_inputs.l1_weights[1][local_slice].contiguous(),
            ),
            (
                global_inputs.l2_weights[0][local_slice].contiguous(),
                global_inputs.l2_weights[1][local_slice].contiguous(),
            ),
            sym_buffer,
            activation_clamp=10.0,
        )
        torch.cuda.synchronize(device)
        _assert_distributed_combine_slots_written(sym_buffer, global_inputs.topk_idx, num_tokens)
    finally:
        sym_buffer.destroy()
    ref = reference.mega_moe_reference(
        global_inputs.x_fp8,
        global_inputs.x_sf,
        global_inputs.topk_idx,
        global_inputs.topk_weights,
        global_inputs.plain_l1_weights[0],
        global_inputs.plain_l1_weights[1],
        global_inputs.plain_l2_weights[0],
        global_inputs.plain_l2_weights[1],
        hidden,
        intermediate_hidden,
        num_experts,
        activation_clamp=10.0,
    )
    diff = 1.0 - torch.nn.functional.cosine_similarity(y.float().flatten(), ref.float().flatten(), dim=0)
    max_abs = (y.float() - ref.float()).abs().max()
    assert diff.item() < 1e-4, f"diff={diff.item()} max_abs={max_abs.item()}"
    assert max_abs.item() < 2.0, f"diff={diff.item()} max_abs={max_abs.item()}"
    dist.barrier(group)
    dist.destroy_process_group()


@_requires_opt_in_dist
@distributed_test(nprocs=2)
def test_functional_mega_moe_distributed_contract(local_rank: int, num_ranks: int):
    _run_functional_mega_moe_distributed_contract(local_rank, num_ranks, hidden=128, num_topk=2, seed=11)


@_requires_opt_in_dist
@distributed_test(nprocs=2)
def test_functional_mega_moe_distributed_fused_hidden256_contract(local_rank: int, num_ranks: int):
    _run_functional_mega_moe_distributed_contract(local_rank, num_ranks, hidden=256, num_topk=2, seed=19)


@_requires_opt_in_dist
@distributed_test(nprocs=2)
def test_functional_mega_moe_distributed_top1_contract(local_rank: int, num_ranks: int):
    _run_functional_mega_moe_distributed_contract(local_rank, num_ranks, hidden=128, num_topk=1, seed=31)


def _run_distributed_single_launch_dispatch_stage_contract(local_rank: int, num_ranks: int, hidden: int, seed: int):
    import torch.distributed as dist

    from tilelang.distributed.host import init_dist

    rank, world_size, group = init_dist(local_rank, num_ranks)
    assert rank == local_rank
    assert world_size == num_ranks

    intermediate_hidden = 128
    num_tokens = 4
    num_topk = 2
    num_experts_per_rank = 2
    num_experts = num_experts_per_rank * num_ranks
    device = torch.device(f"cuda:{local_rank}")
    global_inputs = reference.make_random_mega_moe_inputs(
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_experts=num_experts,
        num_topk=num_topk,
        device=device,
        seed=seed,
        masked_ratio=0.25,
    )
    local_slice = slice(rank * num_experts_per_rank, (rank + 1) * num_experts_per_rank)
    sym_buffer = mega_moe.get_symm_buffer_for_mega_moe(
        group,
        num_experts,
        num_tokens,
        num_topk,
        hidden,
        intermediate_hidden,
        device=device,
    )
    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=device)
    try:
        mega_moe._copy_inputs_to_buffer(sym_buffer, global_inputs)
        sym_buffer.combine_acts.fill_(0)
        dist.barrier(group)
        distributed_map = mega_moe._run_distributed_dispatch_l1_l2_remote_combine_sm100_megakernel(
            y,
            sym_buffer,
            (
                global_inputs.l1_weights[0][local_slice].contiguous(),
                global_inputs.l1_weights[1][local_slice].contiguous(),
            ),
            (
                global_inputs.l2_weights[0][local_slice].contiguous(),
                global_inputs.l2_weights[1][local_slice].contiguous(),
            ),
            True,
            10.0,
            num_tokens=num_tokens,
        )
        assert distributed_map is not None
        torch.cuda.synchronize(device)
        _assert_distributed_l1_dispatch_matches_inputs(sym_buffer, distributed_map)
    finally:
        sym_buffer.destroy()
        dist.destroy_process_group()


@_requires_opt_in_dist
@distributed_test(nprocs=2)
def test_distributed_single_launch_dispatch_stage_hidden128_contract(local_rank: int, num_ranks: int):
    _run_distributed_single_launch_dispatch_stage_contract(local_rank, num_ranks, hidden=128, seed=23)


@_requires_opt_in_dist
@distributed_test(nprocs=2)
def test_distributed_single_launch_dispatch_stage_contract(local_rank: int, num_ranks: int):
    _run_distributed_single_launch_dispatch_stage_contract(local_rank, num_ranks, hidden=256, seed=19)


requires_sm100 = tilelang.testing.requires_cuda_compute_version_eq(10, 0)


@requires_sm100
def test_functional_mega_moe_sm100_reference_fallback_cuda(monkeypatch):
    if not _has_usable_cuda_devices(1):
        pytest.skip("Need one usable CUDA device")
    inputs = reference.make_random_mega_moe_inputs(
        num_tokens=8,
        hidden=256,
        intermediate_hidden=128,
        num_experts=8,
        num_topk=2,
        device="cuda",
        seed=4,
    )
    sym_buffer = mega_moe.get_symm_buffer_for_mega_moe(
        None,
        8,
        inputs.x_fp8.size(0),
        inputs.topk_idx.size(1),
        inputs.x_fp8.size(1),
        inputs.l2_weights[0].size(2) * 2,
        device="cuda",
    )
    try:
        mega_moe._copy_inputs_to_buffer(sym_buffer, inputs)
        y = torch.empty((inputs.x_fp8.size(0), inputs.x_fp8.size(1)), dtype=torch.bfloat16, device="cuda")
        mega_moe.fp8_fp4_mega_moe(
            y,
            inputs.l1_weights,
            inputs.l2_weights,
            sym_buffer,
            activation_clamp=10.0,
            all_l1_weights=inputs.l1_weights,
            all_l2_weights=inputs.l2_weights,
        )
        ref = reference.mega_moe_reference(
            inputs.x_fp8,
            inputs.x_sf,
            inputs.topk_idx,
            inputs.topk_weights,
            inputs.plain_l1_weights[0],
            inputs.plain_l1_weights[1],
            inputs.plain_l2_weights[0],
            inputs.plain_l2_weights[1],
            inputs.x_fp8.size(1),
            inputs.l2_weights[0].size(2) * 2,
            inputs.plain_l1_weights[0].size(0),
            activation_clamp=10.0,
        )
        torch.testing.assert_close(y, ref, rtol=0, atol=0)
    finally:
        sym_buffer.destroy()


@requires_sm100
def test_functional_mega_moe_megakernel_sm100_cuda(monkeypatch):
    if not _has_usable_cuda_devices(1):
        pytest.skip("Need one usable CUDA device")
    inputs = reference.make_random_mega_moe_inputs(
        num_tokens=8,
        hidden=256,
        intermediate_hidden=128,
        num_experts=8,
        num_topk=2,
        device="cuda",
        seed=4,
    )
    got = _run_public_api(inputs, activation_clamp=10.0)
    ref = reference.mega_moe_reference(
        inputs.x_fp8,
        inputs.x_sf,
        inputs.topk_idx,
        inputs.topk_weights,
        inputs.plain_l1_weights[0],
        inputs.plain_l1_weights[1],
        inputs.plain_l2_weights[0],
        inputs.plain_l2_weights[1],
        inputs.x_fp8.size(1),
        inputs.l2_weights[0].size(2) * 2,
        inputs.plain_l1_weights[0].size(0),
        activation_clamp=10.0,
    )
    torch.cuda.synchronize()
    max_abs = (got.float() - ref.float()).abs().max()
    diff = 1.0 - torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0)
    assert diff.item() < 1e-3
    assert max_abs.item() < 2.0


@requires_sm100
def test_functional_mega_moe_megakernel_hidden128_sm100_cuda(monkeypatch):
    if not _has_usable_cuda_devices(1):
        pytest.skip("Need one usable CUDA device")
    inputs = reference.make_random_mega_moe_inputs(
        num_tokens=8,
        hidden=128,
        intermediate_hidden=128,
        num_experts=8,
        num_topk=2,
        device="cuda",
        seed=44,
    )
    got = _run_public_api(inputs, activation_clamp=10.0)
    ref = reference.mega_moe_reference(
        inputs.x_fp8,
        inputs.x_sf,
        inputs.topk_idx,
        inputs.topk_weights,
        inputs.plain_l1_weights[0],
        inputs.plain_l1_weights[1],
        inputs.plain_l2_weights[0],
        inputs.plain_l2_weights[1],
        inputs.x_fp8.size(1),
        inputs.l2_weights[0].size(2) * 2,
        inputs.plain_l1_weights[0].size(0),
        activation_clamp=10.0,
    )
    torch.cuda.synchronize()
    max_abs = (got.float() - ref.float()).abs().max()
    diff = 1.0 - torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0)
    assert diff.item() < 1e-3
    assert max_abs.item() < 2.0


@requires_sm100
def test_functional_mega_moe_megakernel_top1_sm100_cuda(monkeypatch):
    if not _has_usable_cuda_devices(1):
        pytest.skip("Need one usable CUDA device")
    inputs = reference.make_random_mega_moe_inputs(
        num_tokens=8,
        hidden=256,
        intermediate_hidden=128,
        num_experts=8,
        num_topk=1,
        device="cuda",
        seed=41,
    )
    got = _run_public_api(inputs, activation_clamp=10.0)
    ref = reference.mega_moe_reference(
        inputs.x_fp8,
        inputs.x_sf,
        inputs.topk_idx,
        inputs.topk_weights,
        inputs.plain_l1_weights[0],
        inputs.plain_l1_weights[1],
        inputs.plain_l2_weights[0],
        inputs.plain_l2_weights[1],
        inputs.x_fp8.size(1),
        inputs.l2_weights[0].size(2) * 2,
        inputs.plain_l1_weights[0].size(0),
        activation_clamp=10.0,
    )
    torch.cuda.synchronize()
    max_abs = (got.float() - ref.float()).abs().max()
    diff = 1.0 - torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0)
    assert diff.item() < 1e-3
    assert max_abs.item() < 2.0


@requires_sm100
def test_functional_mega_moe_megakernel_multi_wave_sm100_cuda(monkeypatch):
    if not _has_usable_cuda_devices(1):
        pytest.skip("Need one usable CUDA device")
    num_experts = 200
    inputs = reference.make_random_mega_moe_inputs(
        num_tokens=num_experts,
        hidden=256,
        intermediate_hidden=128,
        num_experts=num_experts,
        num_topk=1,
        device="cuda",
        seed=53,
    )
    inputs.topk_idx.copy_(torch.arange(num_experts, device="cuda", dtype=torch.int64).reshape(num_experts, 1))
    inputs.topk_weights.fill_(1.0)
    got = _run_public_api(inputs, activation_clamp=10.0)
    ref = reference.mega_moe_reference(
        inputs.x_fp8,
        inputs.x_sf,
        inputs.topk_idx,
        inputs.topk_weights,
        inputs.plain_l1_weights[0],
        inputs.plain_l1_weights[1],
        inputs.plain_l2_weights[0],
        inputs.plain_l2_weights[1],
        inputs.x_fp8.size(1),
        inputs.l2_weights[0].size(2) * 2,
        inputs.plain_l1_weights[0].size(0),
        activation_clamp=10.0,
    )
    torch.cuda.synchronize()
    max_abs = (got.float() - ref.float()).abs().max()
    diff = 1.0 - torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0)
    assert diff.item() < 1e-3
    assert max_abs.item() < 2.0


if __name__ == "__main__":
    tilelang.testing.main()
