"""
Multicast (NVSwitch) operations test.

Usage:
  # Single-GPU unit tests (no torchrun needed):
  python testing/python/distributed/test_multicast_ops.py

  # Multi-GPU integration test:
  torchrun --nproc_per_node=8 testing/python/distributed/test_multicast_ops.py --distributed
"""

import argparse
import os

import torch
import torch.distributed as dist


def test_supports_multicast():
    """Test multicast support detection."""
    from tilelang.distributed.shared_memory import _supports_multicast

    result = _supports_multicast()
    print(f"\033[32m[PASS]\033[0m _supports_multicast() = {result}")
    return result


def test_distributed_multicast_allocator(rank, world_size):
    """Multi-GPU integration test: MulticastAllocator with fabric VMM."""
    from tilelang.utils.allocator import MulticastAllocator

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    group = dist.new_group(list(range(world_size)))

    allocator = MulticastAllocator(
        size=1024 * 1024,  # 1 MB
        device="cuda",
        local_rank=local_rank,
        num_local_ranks=world_size,
        group=group,
    )

    assert allocator._initialized
    assert allocator.ptr != 0
    assert len(allocator.peer_ptrs) == world_size

    # Get local tensor and write data
    t = allocator.get_local_tensor((256,), torch.bfloat16)
    t.fill_(float(rank + 1))
    torch.cuda.synchronize()

    dist.barrier()

    # Verify we can read peer data
    peer_rank = (local_rank + 1) % world_size
    peer_t = allocator.get_peer_tensor(peer_rank, (256,), torch.bfloat16)
    peer_val = peer_t[0].item()
    expected_val = float(peer_rank + 1)
    assert abs(peer_val - expected_val) < 1e-2, \
        f"rank {local_rank}: peer[{peer_rank}] = {peer_val}, expected {expected_val}"

    dist.barrier()

    if rank == 0:
        print(f"\033[32m[PASS]\033[0m test_distributed_multicast_allocator (world_size={world_size})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distributed", action="store_true", help="Run multi-GPU tests (requires torchrun)"
    )
    args = parser.parse_args()

    if args.distributed:
        import datetime

        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(seconds=60),
        )

        test_distributed_multicast_allocator(rank, world_size)

        dist.destroy_process_group()
    else:
        # Single-GPU unit tests
        torch.cuda.set_device(0)
        test_supports_multicast()
        print("\n=== Single-GPU tests done ===")


if __name__ == "__main__":
    main()
