"""Buffer/EPHandle: DeepEP-EPv2-aligned API over the TileScale intranode port.

Names and call shape mirror DeepEP EPv2's real ``ElasticBuffer``/``EPHandle``
(see ``deep_ep/buffers/elastic.py`` in the DeepEP submodule) as closely as this
port's scope allows. Deliberately dropped: RDMA/hybrid mode, low-latency
decode path, Engram/PP/AGRS, expert-alignment and expand-layout (no fused
grouped GEMM in scope, so there's nothing to align or expand for), and
handle-cached "skip renotify" reuse (DeepEP's decode-replay optimization --
a real simplification, not attempted here yet).

What *is* aligned with DeepEP's real design (see ``kernels/dispatch.py`` and
``kernels/combine.py`` for the detailed mapping): per-(token, destination-rank)
dedup, GPU-side notify with a real cross-rank count exchange, local (not
remote) atomic slot claiming into a fixed per-sender receive slice sized
exactly `num_max_tokens_per_rank * num_ranks` (a hard capacity bound once
deduped, not a statistical headroom guess), and a combine that stores back
into unique per-(rank, token) slots and reduces locally instead of pushing
remote atomics. `dispatch_threads`/`combine_threads` control warps per block
independently; several warps per SM, like DeepEP itself, is what actually
uses NVLink's per-SM bandwidth.

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
from reference import packed_row_bytes

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


class EventOverlap:
    """DeepEP's `EventOverlap`: a comm-stream event, plus what it must outlive.

    `dispatch` and `combine` return one whether or not they were asked to run
    asynchronously; synchronously it wraps `None`, so `with event:` is a no-op
    and callers do not branch.

    `extra_tensors` is DeepEP's mechanism and its reason is worth repeating:
    the obvious way to keep a comm-stream tensor alive until the compute stream
    is done with it is `Tensor.record_stream`, but that is incompatible with
    CUDA graph capture. Holding a reference here instead ties the tensors'
    lifetime to the event object, which the caller drops after waiting.
    """

    def __init__(self, event: torch.cuda.Event | None = None, extra_tensors: tuple = ()):
        self.event = event
        self.extra_tensors = extra_tensors
        self._release_handle_by_call = False

    def current_stream_wait(self, release_handle: bool = False) -> None:
        """Make the current stream wait for the comm kernels, without blocking the host."""
        assert self.event is not None, "no event: this call was not made with async_finish=True"
        torch.cuda.current_stream().wait_event(self.event)
        if release_handle:
            self.event = None
            self.extra_tensors = ()

    def __call__(self, release_handle: bool = False) -> "EventOverlap":
        self._release_handle_by_call = release_handle
        return self

    def __enter__(self) -> "EventOverlap":
        """Overlap whatever the block enqueues with the communication.

        ```python
        recv_x, _, _, handle, event = buf.dispatch(x, topk_idx, topk_weights, async_finish=True)
        with event:
            unrelated_work_on_the_current_stream()
        # leaving the block, the current stream waits: `recv_x` is now readable
        ```
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.event is not None:
            self.current_stream_wait(release_handle=self._release_handle_by_call)
        self._release_handle_by_call = False


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
            `None` in the expanded layout, which groups by expert rather than by
            sender and so never computes it: use `expert_count`/`expert_offset`.
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
        expert_count=None,
        expert_offset=None,
        expand_overflow=None,
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
        # Expanded layout only, `None` otherwise: local expert `e` owns rows
        # `[expert_offset[e], expert_offset[e + 1])` of the received tensor, of
        # which the first `expert_count[e]` are real and the rest is alignment
        # padding, zeroed unless `zero_padding=False`. See kernels/dispatch.py.
        self.expert_count = expert_count
        self.expert_offset = expert_offset
        self._expand_overflow = expand_overflow
        self._num_recv_tokens = None

    @property
    def expand_overflow(self) -> int:
        """Rows the expanded layout needed beyond capacity, 0 if it fit.

        Non-zero means dispatch *skipped* this rank rather than writing past
        its buffer, so everything the call returned is meaningless: re-create
        the `Buffer` with `expand_factor` at least this over `recv_capacity`.
        Reads back to the host, like `num_recv_tokens`.
        """
        if self._expand_overflow is None:
            return 0
        if self.finish_event is not None:
            self.finish_event.synchronize()
        return int(self._expand_overflow[0].item())

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
        do_expand: bool = False,
        expert_alignment: int = 1,
        zero_padding: bool = True,
        expand_factor: float = 1.0,
    ):
        self.group = group
        self.rank_idx = local_rank
        self.num_ranks = num_local_ranks
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.hidden = hidden
        self.num_topk = num_topk
        self.num_experts = num_experts
        assert num_experts % num_local_ranks == 0, f"num_experts={num_experts} is not a multiple of {num_local_ranks} ranks"
        self.experts_per_rank = num_experts // num_local_ranks
        # `dtype` is the *dispatch payload* type. Combine always moves bf16:
        # what it carries is the expert output, which is not quantised.
        self.dtype = dtype
        self.tl_dtype = _TL_DTYPES[dtype]
        self.is_fp8 = dtype == torch.float8_e4m3fn
        if self.is_fp8:
            assert hidden % FP8_GROUP == 0, f"hidden={hidden} is not a multiple of {FP8_GROUP}"
        self.scale_dim = hidden // FP8_GROUP if self.is_fp8 else 0
        # The row dispatch actually moves: for FP8, payload bytes followed by
        # the per-group fp32 scale packed right after -- so the scatter needs
        # one remote store per token-destination pair instead of two -- padded
        # to `put_warp`'s preferred boundary. `reference.packed_row_bytes` owns
        # the formula and the reasoning; kernels/dispatch.py mirrors it.
        self.row_bytes = packed_row_bytes(hidden, FP8_GROUP) if self.is_fp8 else None
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
        self.device_sms = device_sms
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

        # DeepEP's `do_expand`: one received row per (token, expert), grouped
        # by local expert, which is the layout a grouped GEMM wants. See
        # kernels/dispatch.py for how the sender computes the index and
        # `reference.expanded_layout` for what the result should look like.
        self.do_expand = do_expand
        self.expert_alignment = expert_alignment if do_expand else 1
        self.zero_padding = zero_padding
        # Deduplicated, `total_capacity` cannot be exceeded. Expanded it can:
        # the hard bound is `min(num_topk, experts_per_rank)` times higher, for
        # routing that puts every one of a token's experts on one rank, which
        # at the V3 shape is 7 GiB of receive buffer against 0.88. Balanced
        # routing needs `expand_factor=1`, so that is the default and dispatch
        # raises `expand_overflow` rather than corrupting memory if a call
        # exceeds it. Raise the factor (up to the hard bound) for skewed
        # routing.
        self.expand_factor = expand_factor
        if do_expand:
            hard_bound = self.total_capacity * min(num_topk, self.experts_per_rank)
            # Plus the alignment padding, which is not routing-dependent and so
            # sits outside the bound rather than inside it.
            aligned_slack = self.experts_per_rank * (self.expert_alignment - 1)
            self.recv_capacity = min(int(self.total_capacity * expand_factor), hard_bound) + aligned_slack
        else:
            self.recv_capacity = self.total_capacity

        itemsize = torch.empty((), dtype=dtype).element_size()
        comm_bytes = self.num_ranks * self.cap * hidden * 2  # combine is always bf16
        row_bytes = self.row_bytes if self.is_fp8 else hidden * itemsize
        compact_bytes = self.recv_capacity * (row_bytes + 4 + 4 + num_topk * 4 + num_topk * 4)
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
        n_dst = num_experts if do_expand else self.num_ranks
        self.send_count = tilelang.tensor((n_dst,), torch.uint32, allocator=self.allocator)
        # Grid-wide rendezvous counters for dispatch's three phases.
        self.notify_done = tilelang.tensor((1,), torch.uint32, allocator=self.allocator)
        self.exchange_done = tilelang.tensor((1,), torch.uint32, allocator=self.allocator)
        self.slot_counter = tilelang.tensor((n_dst,), torch.uint32, allocator=self.allocator)
        # Stand-in for the degenerate `recv_expert_stats` argument when the
        # caller did not ask for per-expert stats; see `dispatch`.
        self._no_expert_stats = tilelang.tensor((1,), torch.uint32, allocator=self.allocator)
        # Expanded-layout outputs. One element each unless expanding, matching
        # the kernel's degenerate argument shapes.
        self.expert_count = tilelang.tensor((self.experts_per_rank if self.do_expand else 1,), torch.int32, allocator=self.allocator)
        self.expert_offset = tilelang.tensor((self.experts_per_rank + 1 if self.do_expand else 1,), torch.int32, allocator=self.allocator)
        self.expand_overflow = tilelang.tensor((1,), torch.int32, allocator=self.allocator)
        # Likewise for combine's unused bias arguments; see `combine`.
        self._no_bias = tilelang.tensor((1, hidden), self.combine_dtype, allocator=self.allocator)
        # int32 (signed): -1 is the "not yet published" sentinel every rank
        # spins on while the count matrix fills in.
        self.count_matrix = tilelang.tensor((self.num_ranks * n_dst,), torch.int32, allocator=self.allocator)
        self.send_base = tilelang.tensor((n_dst,), torch.int32, allocator=self.allocator)
        self.psum_recv_count = tilelang.tensor((self.num_ranks,), torch.int32, allocator=self.allocator)
        self.num_recv = tilelang.tensor((1,), torch.int32, allocator=self.allocator)
        self.send_rank_mask = tilelang.tensor((num_max_tokens_per_rank,), torch.int32, allocator=self.allocator)

        # FP8: raw `row_bytes` per slot (payload followed by scale, packed --
        # see `row_bytes` above), opaque to the caller until unpacked with
        # `reference.per_token_cast_back`. BF16: `hidden` elements of `dtype`,
        # unchanged.
        self.recv_x = tilelang.tensor(
            (self.recv_capacity, self.row_bytes if self.is_fp8 else hidden),
            torch.uint8 if self.is_fp8 else dtype,
            allocator=self.allocator,
        )
        self.recv_x_flat = self.recv_x.view(-1)
        self.recv_src_rank = tilelang.tensor((self.recv_capacity,), torch.int32, allocator=self.allocator)
        self.recv_src_token = tilelang.tensor((self.recv_capacity,), torch.int32, allocator=self.allocator)
        self.recv_topk_idx = tilelang.tensor((self.recv_capacity, num_topk), torch.int32, allocator=self.allocator)
        self.recv_topk_weights = tilelang.tensor((self.recv_capacity, num_topk), torch.float32, allocator=self.allocator)

        # Combine's staging buffer, one slot per (contributing rank, source
        # token) -- the equivalent of DeepEP's `recv_buffer` on the way back.
        self.comm_x = tilelang.tensor((self.num_ranks * self.cap * hidden,), self.combine_dtype, allocator=self.allocator)

        self.combined = tilelang.tensor((num_max_tokens_per_rank, hidden), self.combine_dtype, allocator=self.allocator)

        # `combine` only ever reads slots dispatch actually wrote, but zeroing
        # once keeps a first-use read of never-written memory from producing
        # NaNs in the (unused) tail of the compact output.
        self.recv_x.zero_()
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
        # 0 disables the throttle, which is what an `async_finish` caller
        # driving its own overlap usually wants: the `synchronize` it does
        # blocks the *host*, so a bounded run-ahead and a fully asynchronous
        # call are not the same thing.
        self.pipeline_depth = pipeline_depth
        self._in_flight: deque = deque()

        self._dispatch_kernels = {}
        self._combine_kernels = {}

    def _resolve_num_sms(self, num_sms: int) -> int:
        """Per-call grid size, validated. Both kernels rendezvous grid-wide, so
        every block has to be resident and the count cannot exceed the device."""
        num_sms = self.num_sms if num_sms == 0 else num_sms
        assert 0 < num_sms <= self.device_sms, f"num_sms={num_sms} exceeds the device's {self.device_sms} SMs"
        return num_sms

    def _get_dispatch_kernel(self, num_tokens: int, collect_expert_stats: bool = False, num_sms: int = 0):
        num_sms = self._resolve_num_sms(num_sms)
        key = (num_tokens, collect_expert_stats, num_sms)
        if key not in self._dispatch_kernels:
            kernel = dispatch_kernel(
                num_tokens,
                self.num_ranks,
                self.num_experts,
                self.num_topk,
                self.hidden,
                self.num_max_tokens_per_rank,
                num_sms,
                self.dispatch_threads,
                self.tl_dtype,
                self.scale_dim,
                self.row_bytes or 0,
                collect_expert_stats,
                self.do_expand,
                self.expert_alignment,
                self.zero_padding,
                self.recv_capacity,
            )
            kernel.compile_group = self.group
            kernel.initialize(allocator=self.allocator)
            self._dispatch_kernels[key] = kernel
        return self._dispatch_kernels[key]

    def _get_combine_kernel(self, num_tokens: int, num_bias: int = 0, num_sms: int = 0):
        num_sms = self._resolve_num_sms(num_sms)
        key = (num_tokens, num_bias, num_sms)
        if key not in self._combine_kernels:
            kernel = combine_kernel(
                num_tokens,
                self.num_ranks,
                self.hidden,
                self.num_max_tokens_per_rank,
                self.total_capacity,
                num_sms,
                self.combine_threads,
                self.reduce_threads,
                self.tl_combine_dtype,
                num_bias,
            )
            kernel.compile_group = self.group
            kernel.initialize(allocator=self.allocator)
            self._combine_kernels[key] = kernel
        return self._combine_kernels[key]

    def dispatch(
        self,
        x,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_sms: int = 0,
        previous_event: EventOverlap = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        cumulative_local_expert_recv_stats: torch.Tensor = None,
    ):
        """Scatter `x` to the ranks owning each token's top-k experts.

        `x` is the packed `(values, scale)` buffer `reference.per_token_cast_to_fp8`
        produces when the buffer's dtype is FP8 (payload bytes followed by the
        per-group fp32 scale, so the scatter moves both in one remote store per
        token-destination pair -- see kernels/dispatch.py), and a plain tensor
        otherwise. The first return value mirrors that.

        Returns `(recv_x, recv_topk_idx, recv_topk_weights, handle, event)`.
        The `event` is an `EventOverlap`, as in DeepEP, and is returned either
        way -- synchronously it wraps `None`, so callers need not branch.

        With `async_finish` the caller's stream is *not* joined to the
        communication stream and nothing the call returns may be read until the
        event is waited on (`with event:` or `event.current_stream_wait()`).
        EPv2 spells this argument `async_with_compute_stream`; the name here
        matches its `combine` and DeepEP's own legacy buffer.

        `previous_event` starts the communication after one specific event
        rather than after everything queued on the caller's stream, and
        `allocate_on_comm_stream` leaves this call's temporaries owned by the
        communication stream -- keeping them alive through the returned event
        instead of `record_stream`, which CUDA graph capture does not allow. As
        in DeepEP the first requires the second.

        `cumulative_local_expert_recv_stats` is DeepEP's load-balance counter:
        a `[num_experts // num_ranks]` uint32 tensor this rank's received token
        count per local expert is *added into*. The caller owns it and decides
        when to zero it, so it can accumulate over a step, a batch or a whole
        run. Passing it compiles a separate kernel variant -- the tally is
        absent from the default one, not merely skipped.
        """
        collect_expert_stats = cumulative_local_expert_recv_stats is not None
        if collect_expert_stats:
            stats = cumulative_local_expert_recv_stats
            assert stats.dtype == torch.uint32 and stats.shape == (self.experts_per_rank,), (
                f"expected a ({self.experts_per_rank},) uint32 tensor, got {tuple(stats.shape)} {stats.dtype}"
            )
        else:
            stats = self._no_expert_stats
        if self.is_fp8:
            assert x.dtype == torch.uint8 and x.shape[1] == self.row_bytes, (
                f"expected a packed (*, {self.row_bytes}) uint8 buffer from reference.per_token_cast_to_fp8, got {tuple(x.shape)} {x.dtype}"
            )
        num_tokens = x.shape[0]
        num_sms = self._resolve_num_sms(num_sms)
        compute_stream = torch.cuda.current_stream()
        if previous_event is not None:
            assert allocate_on_comm_stream, "previous_event requires allocate_on_comm_stream"
            self.comm_stream.wait_event(previous_event.event)
        else:
            self.comm_stream.wait_stream(compute_stream)

        with torch.cuda.stream(self.comm_stream):
            # Nothing is reset here: the kernel does it at the end of every
            # call, which measured 633 GB/s against 626 for six host-side
            # `zero_()` launches -- consistently ahead across four interleaved
            # rounds. It needs a `T.sync_grid()` to be correct; see
            # kernels/dispatch.py.
            topk_idx_i32 = topk_idx.to(torch.int32).contiguous()
            topk_weights_f32 = topk_weights.to(torch.float32).contiguous()

            kernel = self._get_dispatch_kernel(num_tokens, collect_expert_stats, num_sms)
            # No `dist.barrier` on either side. The reset above is peer-visible
            # state, so it does need ordering against peers -- but the kernel's
            # own entry `barrier_blocks` provides it, which is why no collective
            # is needed here. See kernels/dispatch.py.
            kernel(
                x,
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
                self.recv_src_rank,
                self.recv_src_token,
                self.recv_topk_idx.view(-1),
                self.recv_topk_weights.view(-1),
                stats,
                self.expert_count,
                self.expert_offset,
                self.expand_overflow,
            )

            # No device-to-host read of `num_recv`: it cost ~33us and nothing
            # here needs the value. Overflow cannot happen by construction --
            # every peer sends at most `num_max_tokens_per_rank` rows and
            # capacity is `num_ranks` times that.
            num_recv = self.num_recv.clone()
            psum_recv_count = self.psum_recv_count.clone()
            # Only the expanded layout produces these, so deduplicated there is
            # nothing to snapshot.
            if self.do_expand:
                expert_count = self.expert_count.clone()
                expert_offset = self.expert_offset.clone()
                expand_overflow = self.expand_overflow.clone()
            else:
                expert_count = expert_offset = expand_overflow = None

        finish_event = torch.cuda.Event()
        finish_event.record(self.comm_stream)

        # Keep the CPU from running arbitrarily far ahead. See `pipeline_depth`.
        if self.pipeline_depth:
            self._in_flight.append(finish_event)
            while len(self._in_flight) > self.pipeline_depth:
                self._in_flight.popleft().synchronize()

        # Allocated on the communication stream, read by the caller on its
        # own: something has to keep them alive across the handover.
        temporaries = tuple(
            t
            for t in (topk_idx_i32, topk_weights_f32, num_recv, psum_recv_count, expert_count, expert_offset, expand_overflow)
            if t is not None
        )
        if async_finish:
            if not allocate_on_comm_stream:
                for t in temporaries:
                    t.record_stream(compute_stream)
        else:
            compute_stream.wait_stream(self.comm_stream)

        handle = EPHandle(
            self.num_experts,
            self.num_max_tokens_per_rank,
            num_sms,
            topk_idx_i32,
            num_recv,
            None if self.do_expand else psum_recv_count,
            self.recv_src_rank,
            self.recv_src_token,
            num_tokens,
            finish_event if async_finish else None,
            expert_count,
            expert_offset,
            expand_overflow,
        )
        event = EventOverlap(
            finish_event if async_finish else None,
            temporaries if async_finish and allocate_on_comm_stream else (),
        )
        # FP8: the packed uint8 buffer, unpacked with `reference.per_token_cast_back`.
        return (self.recv_x, self.recv_topk_idx, self.recv_topk_weights, handle, event)

    def combine(
        self,
        x: torch.Tensor,
        handle: EPHandle,
        num_sms: int = 0,
        bias=None,
        previous_event: EventOverlap = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ):
        """Reduce every rank's contribution back into this rank's token order.

        Returns `(combined, event)`. DeepEP's third element,
        `combined_topk_weights`, has no counterpart here -- see the README on
        why carrying the gate weights back would return the caller its own
        input. `previous_event`, `async_finish` and `allocate_on_comm_stream`
        mean exactly what they do on `dispatch`.

        `bias` is DeepEP's `bias_0`/`bias_1`: `None`, one `[num_tokens, hidden]`
        tensor, or a pair of them, added to the output. As in DeepEP they are
        added once per output token rather than once per contribution, and a
        token with no contributions still receives them. Each distinct count
        compiles its own kernel variant.
        """
        assert not self.do_expand, (
            "combine does not support the expanded layout yet: its store-back slot is "
            "`comm_x[rank][src_token]`, unique only because dispatch deduplicated. Expanded, "
            "a token with two experts on one rank produces two rows that collide there. "
            "DeepEP's answer is `kDoExpandedSend` -- sum a token's local-expert rows before "
            "sending -- which needs the (src_rank, src_token) -> rows inversion dispatch does "
            "not currently record."
        )
        if bias is None:
            biases = ()
        elif torch.is_tensor(bias):
            biases = (bias,)
        else:
            biases = tuple(bias)
        assert len(biases) <= 2, f"DeepEP takes at most two bias tensors, got {len(biases)}"
        for b in biases:
            assert b.shape == (handle.num_tokens, self.hidden) and b.dtype == self.combine_dtype, (
                f"expected a ({handle.num_tokens}, {self.hidden}) {self.combine_dtype} bias, got {tuple(b.shape)} {b.dtype}"
            )
        # The kernel always takes both arguments; unused ones are one-row
        # stand-ins nothing in the generated code reads. See kernels/combine.py.
        bias_args = tuple(biases) + tuple(self._no_bias for _ in range(2 - len(biases)))

        # Read the caller's contribution (see kernels/combine.py) in place --
        # `combine_kernel` takes its length as a symbolic extent, so no copy
        # into a fixed-shape buffer is needed.
        x_contig = x if x.is_contiguous() else x.contiguous()
        x_flat = x_contig.reshape(-1)
        num_sms = self._resolve_num_sms(num_sms or handle.num_sms)
        num_tokens = handle.num_tokens
        compute_stream = torch.cuda.current_stream()
        if previous_event is not None:
            assert allocate_on_comm_stream, "previous_event requires allocate_on_comm_stream"
            self.comm_stream.wait_event(previous_event.event)
        else:
            self.comm_stream.wait_stream(compute_stream)
        with torch.cuda.stream(self.comm_stream):
            kernel = self._get_combine_kernel(num_tokens, len(biases), num_sms)
            kernel(
                x_flat,
                self.recv_src_rank,
                self.recv_src_token,
                handle.num_recv,
                self.send_rank_mask[:num_tokens],
                self.barrier,
                self.comm_x,
                *bias_args,
                self.combined[:num_tokens],
            )
        finish_event = torch.cuda.Event()
        finish_event.record(self.comm_stream)
        # Only a `contiguous()` copy belongs to this call; a reshape view of
        # the caller's own tensor does not, and the output is buffer-owned and
        # outlives any single call.
        temporaries = () if x_contig is x else (x_contig,)
        if async_finish:
            if not allocate_on_comm_stream:
                for t in temporaries:
                    t.record_stream(compute_stream)
        else:
            compute_stream.wait_stream(self.comm_stream)
        # No pipeline bound here, unlike `dispatch`, and the asymmetry is not
        # understood: bounding dispatch is worth 634 -> 530 -> 460 GB/s over
        # successive runs, bounding this one costs 626 -> 610. Measured, not
        # reasoned; re-measure over several runs before changing either.
        event = EventOverlap(
            finish_event if async_finish else None,
            temporaries if async_finish and allocate_on_comm_stream else (),
        )
        return self.combined[:num_tokens], event

    def close(self):
        self.allocator.close()
