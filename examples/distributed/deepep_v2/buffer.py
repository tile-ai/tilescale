"""Buffer/EPHandle: DeepEP-EPv2-aligned API over the TileScale intranode port.

Names and call shape mirror DeepEP EPv2's real ``ElasticBuffer``/``EPHandle``
(see ``deep_ep/buffers/elastic.py`` in the DeepEP submodule) as closely as this
port's scope allows. Deliberately dropped: RDMA/hybrid mode, low-latency
decode path, Engram/PP/AGRS, FP8/scale-factor dispatch, expert-alignment and
expand-layout (no fused grouped GEMM in scope, so there's nothing to align or
expand for), and handle-cached "skip renotify" reuse (DeepEP's decode-replay
optimization -- a real simplification, not attempted here yet).

What *is* aligned with DeepEP's real design (see ``kernels/dispatch.py`` and
``kernels/combine.py`` for the detailed mapping): per-(token, destination-rank)
dedup, GPU-side notify with a real cross-rank count exchange, local (not
remote) atomic slot claiming into a fixed per-sender receive slice sized
exactly `num_max_tokens_per_rank * num_ranks` (a hard capacity bound once
deduped, not a statistical headroom guess), a PDL-chained compaction epilogue,
and a combine that stores back into unique per-(rank, token) slots and reduces
locally instead of pushing remote atomics. `dispatch_threads`/`combine_threads`
control warps per block independently; several warps per SM, like DeepEP
itself, is what actually uses NVLink's per-SM bandwidth.

Dispatch writes straight into the compact output, so it needs no staging buffer
at all (see ``kernels/dispatch.py``); ``comm_x`` is combine's, holding one slot
per (contributing rank, source token) on the way back.
"""

from collections import deque

import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T
from tilelang.distributed.allocator import get_allocator

from kernels.dispatch import dispatch_kernel
from kernels.combine import combine_kernel

_TL_DTYPES = {
    torch.bfloat16: T.bfloat16,
    torch.float16: T.float16,
    torch.float32: T.float32,
    torch.float8_e4m3fn: T.float8_e4m3fn,
}

# FP8 is quantised per token over groups of this many elements, and carries one
# fp32 scale per group -- DeepEP's `per_token_cast_to_fp8` layout, which the
# caller is expected to produce.
FP8_GROUP = 128


class EPHandle:
    """Communication handle returned by ``Buffer.dispatch``, consumed by ``Buffer.combine``.

    Attributes (named to match DeepEP's real ``EPHandle`` where a concept applies):
        num_experts, num_max_tokens_per_rank, num_sms: as passed to dispatch.
        topk_idx: this rank's own top-k expert indices, `[num_tokens, num_topk]`.
        num_recv: total compacted tokens received by this rank, as a
            one-element *device* tensor. `num_recv_tokens` reads it back to the
            host, which synchronises -- call it outside a timed region.
        finish_event: for `async_finish` dispatches, the event the caller must
            wait on before reading anything the call returned; `None` otherwise.
        psum_recv_count: inclusive prefix sum of deduplicated received counts per
            sender rank, `[num_ranks]` -- DeepEP's `psum_num_recv_tokens_per_scaleup_rank`.
        recv_src_rank, recv_src_token: per-compact-row source (rank, token) --
            a simplified, unpacked form of DeepEP's single encoded `recv_src_metadata`.
        num_tokens: how many tokens this rank dispatched, i.e. how many rows
            `combine` reduces back into.
    """

    def __init__(
        self,
        num_experts,
        num_max_tokens_per_rank,
        num_sms,
        topk_idx,
        num_recv,
        psum_recv_count,
        recv_src_rank,
        recv_src_token,
        num_tokens,
        finish_event=None,
    ):
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_sms = num_sms
        self.topk_idx = topk_idx
        self.num_recv = num_recv
        self.psum_recv_count = psum_recv_count
        self.recv_src_rank = recv_src_rank
        self.recv_src_token = recv_src_token
        self.num_tokens = num_tokens
        # Set only for `async_finish` dispatches: nothing the call returned may
        # be read until this is waited on.
        self.finish_event = finish_event
        self._num_recv_tokens = None

    @property
    def num_recv_tokens(self) -> int:
        """The received row count, on the host. Synchronises on first read.

        `dispatch` deliberately does not do this itself: a device-to-host read
        of `num_recv` cost ~33us of its ~970us, and nothing in the pipeline
        needs the value -- `combine` takes the device tensor straight through to
        its kernel. Callers that genuinely want the number (a benchmark's byte
        count, an assertion, slicing for a reference implementation) pay for it
        here, once, and outside whatever they are timing.
        """
        if self._num_recv_tokens is None:
            if self.finish_event is not None:
                self.finish_event.synchronize()
            self._num_recv_tokens = int(self.num_recv[0].item())
        return self._num_recv_tokens


class Buffer:
    """Intranode (NVLink-only) MoE dispatch/combine buffer, DeepEP-EPv2-style."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        local_rank: int,
        num_local_ranks: int,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_topk: int,
        num_experts: int,
        dtype: torch.dtype = torch.bfloat16,
        num_sms: int = 0,
        dispatch_threads: int = 1024,
        combine_threads: int = 1024,
        reduce_threads: int = 256,
        pipeline_depth: int = 2,
    ):
        self.group = group
        self.rank_idx = local_rank
        self.num_ranks = num_local_ranks
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.hidden = hidden
        self.num_topk = num_topk
        self.num_experts = num_experts
        # `dtype` is the *dispatch payload* type. Combine always moves bf16:
        # what it carries is the expert output, which is not quantised.
        self.dtype = dtype
        self.tl_dtype = _TL_DTYPES[dtype]
        self.is_fp8 = dtype == torch.float8_e4m3fn
        if self.is_fp8:
            assert hidden % FP8_GROUP == 0, f"hidden={hidden} is not a multiple of {FP8_GROUP}"
        self.scale_dim = hidden // FP8_GROUP if self.is_fp8 else 0
        self.combine_dtype = torch.bfloat16 if self.is_fp8 else dtype
        self.tl_combine_dtype = _TL_DTYPES[self.combine_dtype]
        # Wide blocks at both ends of the SM range: at 64 SMs 1024 threads
        # measured 688.0 GB/s against 680.8 for 512 (four readings each, back
        # to back), and at 24 SMs the two are within noise. Fewer, wider blocks
        # never lost here, so there is no SM-dependent rule to write.
        self.dispatch_threads = dispatch_threads
        self.combine_threads = combine_threads
        # Separate from `combine_threads` because the reduce wants
        # `hidden / reduce_threads` to be a whole number of 128-bit loads: at
        # hidden=7168 that is 28 per thread at 256 against an awkward 14 at 512,
        # worth 633.7 GB/s against 625.9.
        self.reduce_threads = reduce_threads

        from tilelang.carver.arch import driver

        device_sms = driver.get_num_sms()
        if num_sms == 0:
            num_sms = device_sms
        # Dispatch is a single persistent grid whose phases rendezvous through
        # global counters, so every block has to be resident simultaneously.
        # The kernel is compiled with `__launch_bounds__(threads, 1)`, which
        # guarantees one block per SM fits, but nothing guarantees a grid larger
        # than the device: that would spin forever instead of failing.
        if num_sms > device_sms:
            raise ValueError(
                f"num_sms={num_sms} exceeds the device's {device_sms} SMs; dispatch's grid-wide "
                "rendezvous requires every block to be co-resident and would deadlock"
            )
        self.num_sms = num_sms

        self.cap = num_max_tokens_per_rank
        self.total_capacity = self.cap * self.num_ranks

        itemsize = torch.empty((), dtype=dtype).element_size()
        comm_bytes = self.num_ranks * self.cap * hidden * 2  # combine is always bf16
        scale_bytes = (hidden // FP8_GROUP) * 4 if dtype == torch.float8_e4m3fn else 0
        compact_bytes = self.total_capacity * (hidden * itemsize + scale_bytes + 4 + 4 + num_topk * 4 + num_topk * 4)
        combined_bytes = num_max_tokens_per_rank * (hidden * 2 + 4) + self.num_ranks * self.cap * 4
        total = comm_bytes + compact_bytes + combined_bytes
        self.allocator = get_allocator(
            # 5% of slack: every tensor is padded for alignment and the sum
            # above does not model that, so a fixed margin does not scale.
            size=max(int(total * 1.05) + (1 << 20), 1 << 24),
            device=f"cuda:{local_rank}",
            is_distributed=True,
            local_rank=local_rank,
            num_local_ranks=num_local_ranks,
            group=group,
        )

        # One `num_ranks`-wide slot per barrier site: dispatch entry/exit,
        # combine entry/exit. They must not share -- `tl::barrier_blocks`
        # settles by having each peer's +TAG cancelled by its -TAG, so a second
        # site on the same slot pushes a still-waiting block back above zero.
        # That converges rather than deadlocks, which is why sharing cost 20x
        # instead of hanging. DeepEP separates its tags for the same reason.
        self.barrier = tilelang.tensor((4 * self.num_ranks,), torch.int32, allocator=self.allocator)
        # uint32: atom_add's PTX intrinsic requires an unsigned target.
        self.send_count = tilelang.tensor((self.num_ranks,), torch.uint32, allocator=self.allocator)
        # Grid-wide rendezvous counters for dispatch's three phases.
        self.notify_done = tilelang.tensor((1,), torch.uint32, allocator=self.allocator)
        self.exchange_done = tilelang.tensor((1,), torch.uint32, allocator=self.allocator)
        self.slot_counter = tilelang.tensor((self.num_ranks,), torch.uint32, allocator=self.allocator)
        # int32 (signed): -1 is the "not yet published" sentinel every rank
        # spins on while the count matrix fills in.
        self.count_matrix = tilelang.tensor((self.num_ranks * self.num_ranks,), torch.int32, allocator=self.allocator)
        self.send_base = tilelang.tensor((self.num_ranks,), torch.int32, allocator=self.allocator)
        self.psum_recv_count = tilelang.tensor((self.num_ranks,), torch.int32, allocator=self.allocator)
        self.num_recv = tilelang.tensor((1,), torch.int32, allocator=self.allocator)
        self.send_rank_mask = tilelang.tensor((num_max_tokens_per_rank,), torch.int32, allocator=self.allocator)

        # The symmetric allocator has no FP8 dtype, so the payload is held as
        # bytes and viewed. Nothing in the kernel cares -- `put_warp` moves
        # 16-byte vectors either way -- and the view is what the caller sees.
        self._recv_x_storage = tilelang.tensor(
            (self.total_capacity, hidden), torch.uint8 if self.is_fp8 else dtype, allocator=self.allocator
        )
        self.recv_x = self._recv_x_storage.view(dtype) if self.is_fp8 else self._recv_x_storage
        self.recv_x_flat = self.recv_x.view(-1)
        # Only the FP8 path has scales; on BF16 both of these are 1x1
        # stand-ins matching the kernel's degenerate argument shapes.
        self.recv_x_scales = tilelang.tensor(
            (self.total_capacity, self.scale_dim) if self.is_fp8 else (1, 1), torch.float32, allocator=self.allocator
        )
        self._no_scales = torch.zeros((1, 1), dtype=torch.float32, device=f"cuda:{local_rank}")
        self.recv_src_rank = tilelang.tensor((self.total_capacity,), torch.int32, allocator=self.allocator)
        self.recv_src_token = tilelang.tensor((self.total_capacity,), torch.int32, allocator=self.allocator)
        self.recv_topk_idx = tilelang.tensor((self.total_capacity, num_topk), torch.int32, allocator=self.allocator)
        self.recv_topk_weights = tilelang.tensor((self.total_capacity, num_topk), torch.float32, allocator=self.allocator)

        # Combine's staging buffer, one slot per (contributing rank, source
        # token) -- the equivalent of DeepEP's `recv_buffer` on the way back.
        self.comm_x = tilelang.tensor((self.num_ranks * self.cap * hidden,), self.combine_dtype, allocator=self.allocator)

        self.combined = tilelang.tensor((num_max_tokens_per_rank, hidden), self.combine_dtype, allocator=self.allocator)

        # `combine` only ever reads slots dispatch actually wrote, but zeroing
        # once keeps a first-use read of never-written memory from producing
        # NaNs in the (unused) tail of the compact output.
        self._recv_x_storage.zero_()
        self.send_rank_mask.zero_()
        # The kernel resets these at the end of every call; this is only about
        # the first call finding them defined. `barrier` is the exception --
        # it returns to zero by itself, and re-zeroing it from the host is
        # exactly what breaks a peer still inside it.
        self.barrier.zero_()
        self.send_count.zero_()
        self.notify_done.zero_()
        self.exchange_done.zero_()
        self.slot_counter.zero_()
        self.count_matrix.fill_(-1)

        # Communication runs on its own stream, DeepEP's `comm_stream`. On the
        # caller's stream, everything this class does -- the topk conversions,
        # the launch, the handle bookkeeping -- sits between GPU kernels with
        # the GPU idle through it. On a private stream it overlaps with whatever
        # the caller already has queued instead.
        self.comm_stream = torch.cuda.Stream()

        # How far the CPU may run ahead, in calls -- not about ordering
        # (`wait_stream` handles that) but about skew between ranks: the kernel
        # ends in a cross-rank barrier, so a rank queued N calls behind stalls
        # everyone inside it. 1, 2 and 4 all measure ~680-685 GB/s; 2 had no
        # low outlier.
        self.pipeline_depth = pipeline_depth
        self._in_flight: deque = deque()

        self._dispatch_kernels = {}
        self._combine_kernels = {}

    def _get_dispatch_kernel(self, num_tokens: int):
        if num_tokens not in self._dispatch_kernels:
            kernel = dispatch_kernel(
                num_tokens,
                self.num_ranks,
                self.num_experts,
                self.num_topk,
                self.hidden,
                self.num_max_tokens_per_rank,
                self.num_sms,
                self.dispatch_threads,
                self.tl_dtype,
                self.scale_dim,
            )
            kernel.compile_group = self.group
            kernel.initialize(allocator=self.allocator)
            self._dispatch_kernels[num_tokens] = kernel
        return self._dispatch_kernels[num_tokens]

    def _get_combine_kernel(self, num_tokens: int):
        if num_tokens not in self._combine_kernels:
            kernel = combine_kernel(
                num_tokens,
                self.num_ranks,
                self.hidden,
                self.num_max_tokens_per_rank,
                self.total_capacity,
                self.num_sms,
                self.combine_threads,
                self.reduce_threads,
                self.tl_combine_dtype,
            )
            kernel.compile_group = self.group
            kernel.initialize(allocator=self.allocator)
            self._combine_kernels[num_tokens] = kernel
        return self._combine_kernels[num_tokens]

    def dispatch(self, x, topk_idx: torch.Tensor, topk_weights: torch.Tensor, num_sms: int = 0, async_finish: bool = False):
        """Scatter `x` to the ranks owning each token's top-k experts.

        `x` is `(values, scales)` when the buffer's dtype is FP8 and a plain
        tensor otherwise; the first return value mirrors that.

        With `async_finish`, the returned handle carries a `finish_event` and
        the caller's stream is *not* joined to the communication stream: nothing
        the call returns may be read until that event is waited on. The default
        joins the streams before returning, so the result is usable immediately.
        """
        # FP8 arrives already quantised, as `(values, scales)` -- the same
        # shape DeepEP's `dispatch` takes. Quantising is the caller's job
        # because it is fused into the previous layer's epilogue in practice.
        if self.is_fp8:
            x, x_scales = x
            assert x_scales.shape[1] == self.scale_dim, f"expected {self.scale_dim} scales per token, got {x_scales.shape[1]}"
            x_scales = x_scales.float().contiguous()
        else:
            x_scales = self._no_scales
        num_tokens = x.shape[0]
        num_sms = self.num_sms if num_sms == 0 else num_sms
        compute_stream = torch.cuda.current_stream()
        self.comm_stream.wait_stream(compute_stream)

        with torch.cuda.stream(self.comm_stream):
            # Nothing is reset here: the kernel does it at the end of every
            # call, which measured 633 GB/s against 626 for six host-side
            # `zero_()` launches -- consistently ahead across four interleaved
            # rounds. It needs a `T.sync_grid()` to be correct; see
            # kernels/dispatch.py.
            topk_idx_i32 = topk_idx.to(torch.int32).contiguous()
            topk_weights_f32 = topk_weights.to(torch.float32).contiguous()

            kernel = self._get_dispatch_kernel(num_tokens)
            # No `dist.barrier` on either side. The reset above is peer-visible
            # state, so it does need ordering against peers -- but the kernel's
            # own entry `barrier_blocks` provides it, which is why no collective
            # is needed here. See kernels/dispatch.py.
            kernel(
                x,
                x_scales,
                topk_idx_i32,
                topk_weights_f32,
                self.notify_done,
                self.exchange_done,
                self.send_count,
                self.count_matrix,
                self.send_base,
                self.psum_recv_count,
                self.num_recv,
                self.slot_counter,
                self.send_rank_mask[:num_tokens],
                self.barrier,
                self.recv_x_flat,
                self.recv_x_scales.view(-1),
                self.recv_src_rank,
                self.recv_src_token,
                self.recv_topk_idx.view(-1),
                self.recv_topk_weights.view(-1),
            )

            # No device-to-host read of `num_recv`: it cost ~33us and nothing
            # here needs the value. Overflow cannot happen by construction --
            # every peer sends at most `num_max_tokens_per_rank` rows and
            # capacity is `num_ranks` times that.
            num_recv = self.num_recv.clone()
            psum_recv_count = self.psum_recv_count.clone()

        finish_event = torch.cuda.Event()
        finish_event.record(self.comm_stream)

        # Keep the CPU from running arbitrarily far ahead. See `pipeline_depth`.
        self._in_flight.append(finish_event)
        while len(self._in_flight) > self.pipeline_depth:
            self._in_flight.popleft().synchronize()

        if async_finish:
            # These were allocated on the communication stream; the caller will
            # read them on its own, so the allocator has to know both.
            for t in (topk_idx_i32, topk_weights_f32, num_recv, psum_recv_count):
                t.record_stream(compute_stream)
        else:
            compute_stream.wait_stream(self.comm_stream)

        handle = EPHandle(
            self.num_experts,
            self.num_max_tokens_per_rank,
            num_sms,
            topk_idx_i32,
            num_recv,
            psum_recv_count,
            self.recv_src_rank,
            self.recv_src_token,
            num_tokens,
            finish_event if async_finish else None,
        )
        recv = (self.recv_x, self.recv_x_scales) if self.is_fp8 else self.recv_x
        return (recv, self.recv_topk_idx, self.recv_topk_weights, handle)

    def combine(self, x: torch.Tensor, handle: EPHandle, num_sms: int = 0):
        # Read the caller's contribution (see kernels/combine.py) in place --
        # `combine_kernel` takes its length as a symbolic extent, so no copy
        # into a fixed-shape buffer is needed.
        x_flat = x.reshape(-1) if x.is_contiguous() else x.contiguous().reshape(-1)
        num_sms = handle.num_sms if num_sms == 0 else num_sms
        num_tokens = handle.num_tokens
        compute_stream = torch.cuda.current_stream()
        self.comm_stream.wait_stream(compute_stream)
        with torch.cuda.stream(self.comm_stream):
            kernel = self._get_combine_kernel(num_tokens)
            kernel(
                x_flat,
                self.recv_src_rank,
                self.recv_src_token,
                handle.num_recv,
                self.send_rank_mask[:num_tokens],
                self.barrier,
                self.comm_x,
                self.combined[:num_tokens],
            )
        compute_stream.wait_stream(self.comm_stream)
        # No pipeline bound here, unlike `dispatch`, and the asymmetry is not
        # understood: bounding dispatch is worth 634 -> 530 -> 460 GB/s over
        # successive runs, bounding this one costs 626 -> 610. Measured, not
        # reasoned; re-measure over several runs before changing either.
        return self.combined[:num_tokens]

    def close(self):
        self.allocator.close()
