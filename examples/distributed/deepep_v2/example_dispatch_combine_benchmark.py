"""Benchmark: dispatch/combine GB/s for the DeepEP-EPv2-aligned port.

    python example_dispatch_combine_benchmark.py --num-sms 24
    python example_dispatch_combine_benchmark.py --num-sms 64

**Warm the clocks or the numbers are noise.** These GPUs idle at 120 MHz against
a 1965 MHz boost ceiling, and `do_bench`'s `warmup` is a count of *iterations*,
not milliseconds -- the default handful of ~1ms iterations is nowhere near
enough to get off the idle clock, and each rep's `torch.cuda._sleep` spin draws
little power, so the clock can sag again mid-measurement. Runs taken without
`--clock-warmup-sec` varied by up to 1.5x on identical binaries and configs,
with dispatch and combine moving together (the signature of a clock effect, not
a kernel one). `_warm_clocks` runs real bf16 GEMMs on every rank first; locking
the clock outright (`nvidia-smi -lgc`) would be better but needs root.
"""

import argparse
import time

import torch
import torch.distributed as dist
import torch.multiprocessing

from tilelang.distributed.host import init_dist
from tilelang.distributed.bench import do_bench

from buffer import Buffer
import reference


def _warm_clocks(seconds: float, device: str) -> None:
    """Drive the SMs hard enough, for long enough, to reach boost clocks."""
    if seconds <= 0:
        return
    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=device)
    b = torch.randn(8192, 8192, dtype=torch.bfloat16, device=device)
    c = torch.empty(8192, 8192, dtype=torch.bfloat16, device=device)
    deadline = time.time() + seconds
    while time.time() < deadline:
        for _ in range(20):
            torch.matmul(a, b, out=c)
        torch.cuda.synchronize()


class _ClockProbe:
    """Sample this rank's SM clock in a thread, so a measurement can report the
    clock it actually ran at.

    Without this there is no way to tell a kernel regression from the GPU
    having been at 120 MHz for half the run.
    """

    def __init__(self, device_index: int, period: float = 0.02):
        self.period, self.samples, self._stop = period, [], None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml, self._handle = pynvml, pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception:
            self._nvml = None

    def __enter__(self):
        if self._nvml is None:
            return self
        import threading

        self._stop = threading.Event()

        def poll():
            while not self._stop.wait(self.period):
                self.samples.append(self._nvml.nvmlDeviceGetClockInfo(self._handle, self._nvml.NVML_CLOCK_SM))

        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._stop is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
        return False

    def summary(self) -> str:
        if not self.samples:
            return "clock n/a"
        s = sorted(self.samples)
        return f"SM clock min/median/max {s[0]}/{s[len(s) // 2]}/{s[-1]} MHz over {len(s)} samples"


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    torch.manual_seed(1234 + rank)
    device = f"cuda:{local_rank}"
    x = torch.randn(args.tokens, args.hidden, dtype=torch.bfloat16, device=device)
    topk_idx, topk_weights = reference.make_topk(args.tokens, args.topk, args.experts, device, args.masked_ratio)

    dtype = torch.float8_e4m3fn if args.fp8 else torch.bfloat16
    # Quantising is the caller's job (see buffer.py's `dispatch` docstring);
    # only the dispatch call itself sees fp8, everything downstream of the
    # cast-back (expert compute, combine) stays bf16.
    dispatch_x = reference.per_token_cast_to_fp8(x) if args.fp8 else x

    buf = Buffer(
        group=group,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        num_max_tokens_per_rank=args.tokens,
        hidden=args.hidden,
        num_topk=args.topk,
        num_experts=args.experts,
        dtype=dtype,
        num_sms=args.num_sms,
        dispatch_threads=args.dispatch_threads,
        combine_threads=args.combine_threads,
    )

    itemsize = 1 if args.fp8 else 2  # fp8 payload byte, not counting the small per-128 scale
    recv, recv_topk_idx, recv_topk_weights, handle, _ = buf.dispatch(dispatch_x, topk_idx, topk_weights)
    # Outside every timed region: this is the one host read of the count.
    num_recv_tokens = handle.num_recv_tokens
    recv_topk_idx = recv_topk_idx[:num_recv_tokens]
    recv_topk_weights = recv_topk_weights[:num_recv_tokens]
    recv_x = reference.per_token_cast_back(recv[:num_recv_tokens], args.hidden) if args.fp8 else recv[:num_recv_tokens]
    dispatch_bytes = num_recv_tokens * args.hidden * itemsize
    # Combine always moves bf16 (see buffer.py) regardless of dispatch's dtype.
    combine_bytes = num_recv_tokens * args.hidden * 2

    expert_stats = torch.zeros(args.experts // num_local_ranks, dtype=torch.uint32, device=device) if args.expert_stats else None

    def run_dispatch():
        buf.dispatch(dispatch_x, topk_idx, topk_weights, cumulative_local_expert_recv_stats=expert_stats)

    _warm_clocks(args.clock_warmup_sec, device)
    dist.barrier(group)
    with _ClockProbe(local_rank) as probe:
        dispatch_ms = do_bench(run_dispatch, warmup=args.warmup, rep=args.rep, group=group)
    if rank == 0:
        print(
            f"dispatch: {dispatch_ms * 1000:.1f} us, {dispatch_bytes / (dispatch_ms * 1e-3) / 1e9:.1f} GB/s (recv-side, this rank)  [{probe.summary()}]"
        )

    expert_out = reference.simulate_expert_compute(recv_x, recv_topk_idx, recv_topk_weights)

    def run_combine():
        buf.combine(expert_out, handle)

    _warm_clocks(args.clock_warmup_sec, device)
    dist.barrier(group)
    with _ClockProbe(local_rank) as probe:
        combine_ms = do_bench(run_combine, warmup=args.warmup, rep=args.rep, group=group)
    if rank == 0:
        print(
            f"combine: {combine_ms * 1000:.1f} us, {combine_bytes / (combine_ms * 1e-3) / 1e9:.1f} GB/s (send-side, this rank)  [{probe.summary()}]"
        )

    buf.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    # Fraction of top-k selections marked unselected (-1), DeepEP's marker.
    parser.add_argument("--masked-ratio", type=float, default=0.0)
    # Dispatch payload dtype; combine is always bf16 (see buffer.py).
    parser.add_argument("--fp8", action="store_true")
    # Accumulate DeepEP's per-local-expert receive counts during dispatch.
    parser.add_argument("--expert-stats", action="store_true")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--num-sms", type=int, default=64)
    # Dispatch no longer stages rows through shared memory, so warps per block
    # is now a pure occupancy knob rather than something bounded by a
    # shared-memory budget; 512 measures fastest at this shape.
    parser.add_argument("--dispatch-threads", type=int, default=512)
    parser.add_argument("--combine-threads", type=int, default=256)
    # Iteration counts, not milliseconds.
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=50)
    # Seconds of real GEMM load before each timed section -- see the module
    # docstring. Set to 0 only if the clock is externally locked.
    parser.add_argument("--clock-warmup-sec", type=float, default=5.0)
    args = parser.parse_args()
    torch.multiprocessing.spawn(main, args=(args.num_processes, args), nprocs=args.num_processes, join=True)
