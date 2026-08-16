# DeepEP EPv2 port — intranode dispatch / combine

A TileScale-native MoE all-to-all, following DeepEP EPv2's `impls/dispatch.cuh`
and `impls/combine.cuh`. Scoped to **intranode NVLink** on Blackwell (SM100):
no RDMA/scaleout, no low-latency decode path, no expert-alignment or expand
layout.

Two collectives, both written in TileLang with no inline PTX:

- **`dispatch`** — send each token to the ranks owning its top-k experts, and
  land it directly at its final index in the receiver's compact buffer.
- **`combine`** — send each expert's contribution back and reduce, per source
  token, into the original row order.

## When to use this

Suited to intranode expert parallelism where a whole MoE layer fits inside one
NVLink domain (EP ≤ 8 on a B200 node). Dispatch payload may be bf16 or fp8;
combine is always bf16, since it carries expert output.

Not a drop-in for DeepEP: there is no RDMA path, so multi-node EP is out of
scope, and the low-latency decode kernels are not ported.

## Design

### dispatch — three phases in one kernel

1. **Count.** Every warp scans this rank's `topk_idx` and deduplicates per
   (token, destination rank) with `T.match_any_sync` — a token whose top-k
   picks two experts on one rank is one row, not two. Tallies land in a
   per-warp slice of shared memory; dedup makes the recording lanes hold
   pairwise-distinct destinations, so the tally needs no atomics.
2. **Exchange.** One warp publishes this rank's count vector to every peer, so
   all ranks hold the full `count_matrix[sender][destination]` and can derive
   `send_base[d]` — where their rows begin in destination `d`'s output.
3. **Scatter.** One warp per token. Destinations claim slots with one round of
   `atom_add` on a local counter, then the warp pushes the row, its scales (fp8
   only) and its metadata to `send_base[d] + slot` on peer `d`.

**No copy epilogue.** DeepEP stages rows per sender and compacts them in a
second kernel, because its expand layout hides the final position at send time.
Here the full count matrix makes `send_base[d] + slot` the final compact index,
so rows land in place — saving a whole local read+write of the payload, at the
cost of the scatter having to wait for the exchange (~19µs).

### combine — store-back, then local reduce

One warp per compact row stores into `comm_x[my_rank][src_token]` on the source
rank; that slot is unique per (contributing rank, source token), so nothing
needs to be atomic. A second kernel then sums, per source token, the slots named
by the destination mask dispatch recorded.

### Synchronisation

Both kernels open and close with `tl::barrier_blocks` on private slots, so
neither needs a `dist.barrier` around it. Note that `barrier_blocks`
rendezvouses *ranks*, not a rank's own blocks — any single-block epilogue behind
it also needs `T.sync_grid()`.

Communication runs on a private CUDA stream. `Buffer.pipeline_depth` bounds how
far the CPU may run ahead, which matters because a rank queued several calls
behind stalls every other rank inside the cross-rank barrier.

## API

```python
from buffer import Buffer

buf = Buffer(
    group=group, local_rank=local_rank, num_local_ranks=8,
    num_max_tokens_per_rank=8192, hidden=7168, num_topk=8, num_experts=256,
    dtype=torch.bfloat16,        # or torch.float8_e4m3fn for fp8 dispatch
    num_sms=64,
)

# bf16: x is [num_tokens, hidden]
# fp8:  x is (values, scales) from reference.per_token_cast_to_fp8
recv_x, recv_topk_idx, recv_topk_weights, handle, event = buf.dispatch(x, topk_idx, topk_weights)

n = handle.num_recv_tokens          # device-to-host read; call outside timed regions
expert_out = my_expert_compute(recv_x[:n], recv_topk_idx[:n], recv_topk_weights[:n])

combined, event = buf.combine(expert_out, handle)   # [num_tokens, hidden] bf16
```

`dispatch` returns views over the **full receive capacity**; slice with
`handle.num_recv_tokens` when you need the compact rows. On the fp8 path the
first return value is `(values, scales)` and the caller casts back before the
expert computation — `reference.per_token_cast_back` does this.

### Overlapping

Both collectives return an `EventOverlap` as their last value, DeepEP's wrapper
around the communication-stream event. It comes back either way -- synchronously
it wraps `None` -- so a caller can write `with event:` without knowing which
mode it asked for.

```python
recv_x, recv_topk_idx, recv_topk_weights, handle, event = buf.dispatch(
    x, topk_idx, topk_weights, async_finish=True, allocate_on_comm_stream=True
)
with event:                       # runs on the compute stream, overlapping the dispatch
    something_else()
# leaving the block, the current stream waits: recv_x is readable

combined, event = buf.combine(
    expert_out, handle, previous_event=event, async_finish=True, allocate_on_comm_stream=True
)
event.current_stream_wait()
```

`async_finish` leaves the caller's stream unjoined from the communication
stream; nothing returned may be read until the event is waited on. EPv2 spells
this `async_with_compute_stream` on dispatch; the name here follows its
`combine` and DeepEP's legacy buffer.

`previous_event` starts the communication after one specific event instead of
after everything queued on the caller's stream. `allocate_on_comm_stream` keeps
this call's temporaries owned by the communication stream and alive through the
returned event, rather than `Tensor.record_stream`, which CUDA graph capture
does not permit -- the reason DeepEP carries `extra_tensors` on its event. As in
DeepEP, `previous_event` requires `allocate_on_comm_stream`.

One asymmetry with DeepEP worth knowing: `Buffer.pipeline_depth` (default 2)
bounds how far the CPU may run ahead by blocking the *host* on an event a few
calls back. It exists because a rank queued several calls behind stalls every
other rank inside the kernel's cross-rank barrier, and it is orthogonal to
`async_finish` -- so an asynchronous call can still block the host. Pass
`pipeline_depth=0` to turn it off when driving the overlap yourself.

DeepEP's `combine` returns a third value, `combined_topk_weights`, which has no
counterpart here for the reason given under *Not implemented*.

`dispatch(x, handle=h)` reuses the layout `h` was built with and skips the
notify kernel outright -- DeepEP's cached dispatch. Phases 1-2 depend only on
the routing, so a call whose `topk_idx` has not changed is recomputing
something it already has. As in DeepEP, `topk_idx` and `topk_weights` must be
`None`; the handle replays its own copies.

Measured, three clean samples each: fp8 dispatch 514 -> 497 µs (3.3%), bf16 888
-> 871 µs (2.0%). Less than the 36 µs the notify kernel costs, because the
cached scatter runs about 10 µs slower than the uncached one: the entry barrier
moves into it rather than disappearing, being about peers not overwriting data
this rank is still reading, which holds however the layout was obtained.

A handle is only good until the next layout-computing dispatch. `send_base` and
`send_rank_mask` are updated in place, so a stale handle would not fail, it
would route to the wrong slots and return plausible numbers -- dispatch
therefore tracks a layout generation and rejects a handle that no longer
matches.

`dispatch(..., cumulative_local_expert_recv_stats=t)` adds this rank's received
token count per local expert into `t`, a `[num_experts // num_ranks]` uint32
tensor -- DeepEP's load-balance counter. It accumulates rather than overwrites,
so the caller decides the window by choosing when to zero it. Costs ~25 µs of
dispatch's ~896 (2.8%) and compiles its own kernel variant, so the default path
does not pay for it. DeepEP gets the same number for free because its expanded
layout already exchanges per-expert counts; this port has no such exchange and
counts locally instead, which is why it is not free here.

`dispatch` produces DeepEP's **expanded layout** when the buffer is built with
`do_expand=True`: one received row per (token, expert) instead of one per
(token, rank), and rows grouped by local expert, so each expert's rows are the
contiguous block a grouped GEMM wants. `handle.expert_offset` gives the segment
bounds and `handle.expert_count` how many rows in each are real;
`expert_alignment=n` rounds each segment up to a multiple of `n` and the gap is
zeroed unless `zero_padding=False`.

DeepEP expands in a receiver-side copy epilogue. This port has none -- rows land
at their final index straight from the sender -- so instead the count exchange
runs at expert granularity and the sender derives the index itself. That also
makes the capacity check free: every rank computes the same layout from the same
count matrix, so an overflow is known before any payload moves. Deduplicated,
capacity cannot be exceeded; expanded it can, so `expand_factor` sizes the
receive buffer (default 1.0, right for balanced routing) and `handle.expand_overflow`
reports how many rows a call needed if it did not fit -- dispatch skips the rank
rather than writing past it.

**`combine` cannot consume an expanded dispatch yet** and asserts rather than
returning a wrong answer. Its store-back slot is `comm_x[rank][src_token]`,
unique only because dispatch deduplicated; expanded, a token with two experts on
one rank has two rows that collide there. The fix is DeepEP's `kDoExpandedSend`
-- sum a token's local-expert rows before sending -- which needs a
(src_rank, src_token) -> rows inversion dispatch does not currently record.

`combine(..., bias=b)` adds one tensor, or `bias=(b0, b1)` two, to the output --
DeepEP's `bias_0`/`bias_1`, each `[num_tokens, hidden]`. They seed the reduce
accumulator instead of being added after it, so they cost nothing measurable,
and a token whose every selection was masked off still comes back as its bias.

Knobs worth tuning: `num_sms`, `dispatch_threads`, `combine_threads`, and
`reduce_threads` (separate because the reduce wants `hidden / reduce_threads` to
be a whole number of 128-bit loads). The thread defaults are wide (1024) because
that measured at least as fast at every SM count tried.

## Performance

8× B200, full NVLink mesh, 8192 tokens/rank, hidden 7168, top-8, 256 experts,
64 SMs. Bandwidth is the bottleneck rank's, over payload bytes that cross
NVLink. Three to four samples per row, each one gated on whether any process
that is not ours *used the SMs* at any point during the run, sampled throughout
with `nvidia-smi pmon`. Presence is not the test: an 8-way inference server
holding 167 GB/GPU at 0% utilisation blocks a presence-based gate forever while
disturbing nothing, and a job under this same account is invisible to a
by-other-user check while pinning a GPU at 100%. Both happened; both produced
numbers 30–60% off.

| | dispatch | combine |
|---|---|---|
| bf16, whole call | **686–687 GB/s** (897–899 µs) | **623–624 GB/s** (988–989 µs) |
| bf16, kernel only | 694 GB/s | — |
| fp8, whole call | **590–592 GB/s** (520–523 µs) | 623–625 GB/s (986–989 µs) |
| bf16, `scatter_sms=128` | **703–716 GB/s** (861–877 µs) | unchanged |
| fp8, `scatter_sms=128` | **613–617 GB/s** (499–503 µs) | unchanged |

The last two rows are the same kernels on a wider scatter grid. Dispatch is two
launches, and only the first needs a persistent grid, so the scatter is free to
use more of the device than `num_sms` allows -- worth 3.4% on bf16 and 4.0% on
fp8. Opt-in, because a caller who capped `num_sms` to leave room for expert
compute did not ask for it back. See `kernels/dispatch.py`.

Combine is bf16 whichever dtype dispatch used, and measures the same either
way, as it should.

FP8's rate is below bf16's partly as an accounting artifact: it counts the
7168 payload bytes a row carries, but the row that crosses NVLink is 7680 --
the per-128 fp32 scales packed in alongside, plus alignment padding (see
`reference.packed_row_bytes`). On the wire that is ~632 GB/s. The rest of the
gap is the kernel's fixed phases -- notify, dedup, count exchange, metadata
stores -- costing the same in absolute terms against a payload half the size.

Where the time goes in a bf16 dispatch (887µs kernel, 897µs call):

| | |
|---|---|
| scatter | ~822 µs (749 GB/s — at the `put_warp` roofline of 731–739) |
| entry + exit barriers | ~44 µs |
| count exchange | ~19 µs |
| host | ~9 µs |

combine splits into ~850µs of store-back (725 GB/s, likewise at the roofline)
and ~125µs of local reduce (733MB at ~5.9 TB/s).

Fewer SMs, same shape:

| #SMs | dispatch | combine |
|---|---|---|
| 64 | 686–688 GB/s | 623–625 GB/s |
| 24 | 598 GB/s | 545–546 GB/s |

That is −13% / −13% against DeepEP's −11% / −9% over the same range. At 24 SMs
each block already carries 16 warps, past the point where `put_warp` saturates,
so widening blocks does not help (1024 threads against 512 is ~1% at either
end): what runs out is SM count itself.

### Against DeepEP

DeepEP's own `tests/elastic/test_ep.py`, same machine, same shape, 64 SMs.
Its headline numbers cover `dispatch_impl` / `combine_impl` **only** -- the copy
and reduce epilogues are timed and reported separately. This port has no
dispatch epilogue at all, so the honest comparison is the sum.

**dispatch** (fp8 payload both sides):

| | DeepEP | this port |
|---|---|---|
| main kernel | 442 µs (735 GB/s) | — |
| epilogue | 110–129 µs | none |
| **end to end** | **~562 µs** | **522–524 µs** |

**combine** (bf16 both sides):

| | DeepEP | this port |
|---|---|---|
| store-back | 839 µs (745 GB/s) | 850 µs (723 GB/s) |
| reduce | 155 µs | 125 µs |
| **end to end** | **994 µs** | **985–989 µs** |

This port's two component rows come from a separate profiling pass and do not
re-add to its end-to-end row exactly; the end-to-end figure is the measured
whole call and is the one to trust.

**A whole layer's collectives** -- what a plain dispatch → expert → combine
loop actually pays, since neither epilogue has anything to hide behind there:

| | DeepEP | this port |
|---|---|---|
| dispatch | 442 + 129 µs | 522–524 µs |
| combine | 839 + 156 µs | 985–989 µs |
| **total** | **~1566 µs** | **1507–1513 µs** |

DeepEP is ahead on the cross-rank movement itself -- 745 against 723 GB/s on
combine's store-back, which is this port's roofline for `put_warp`. It gives
that back to the epilogues: writing rows straight into their final compact index
costs a wait for the count exchange (~19 µs) and saves a whole extra pass over
the payload.

Which accounting matters depends on the caller. DeepEP's epilogue is a separate
kernel with a stream hook (`previous_event_before_epilogue`) so a schedule with
independent work can hide it; in a plain dispatch → expert → combine loop there
is nothing to hide it behind, and the end-to-end column is what you get. Note
also that its epilogue *produces* `recv_x`, so it cannot overlap the expert
computation that consumes it.

DeepEP's figures above are GPU time and this port's are whole Python calls,
which is not a like-for-like comparison. Measured the same way -- per-kernel
duration from the profiler, which is what DeepEP's `bench_kineto` reports, both
on the same idle machine at the same shape -- fp8 dispatch comes out:

| | DeepEP | this port |
|---|---|---|
| layout / notify | *(inside the main kernel)* | 35 µs |
| cross-rank movement | 439 µs | 472 µs |
| compaction epilogue | 113–130 µs | none |
| **total** | **~559 µs** | **~508 µs** |

So DeepEP is ahead on the movement itself by about 7%, and this port is ahead
overall by about 9% because it has no compaction pass to pay for. Host overhead
is the difference between those kernel times and the whole-call figures above:
8–10 µs.

**What `async_finish` buys.** Measured against an 8192-square bf16 GEMM sized
to match the collective, on an idle machine:

| | compute | dispatch | serial | overlapped | hidden |
|---|---|---|---|---|---|
| before | 672.4 us | 891.8 | 1568.9 | 1538.4 | 29.7 us (4%) |
| + stream priority | 671.0 us | 891.5 | 1566.2 | 1279.9 | 286.3 us (43%) |
| + cast hoisted | 675.8 us | 890.9 | 1566.7 | **1135.8** | **430.9 us (64%)** |

Two one-line fixes, and the reason each is worth what it is worth is worth
writing down, because six plausible theories were measured and rejected before
the first one.

A private stream buys *eligibility*, not *admission*. The trace shows the
collective becoming eligible the instant the previous scatter ends, and then
waiting 624 us anyway. What it is waiting for is an SM. A GEMM large enough to
be worth hiding behind is also large enough to hold the whole device: 2048
blocks, each needing a full SM's registers and 213 KB of shared memory, so 148
are resident and fourteen waves are pending. Block admission is greedy, so
every SM that frees goes to the next GEMM block, and a collective that arrives
even microseconds later gets in only as the last wave drains -- a fixed ~48 us
window, which is why the amount hidden was a constant ~30 us regardless of
dispatch length, compute length or SM count.

The experiment that settled it: `torch.cuda._sleep` calibrated to the same
674 us as the GEMM occupies the compute stream just as long but uses **one
block**, and it is hidden **100%**. Same streams, same events, same
dependencies, 147 SMs free instead of none. Not ordering -- occupancy.

Stream priority biases precisely that admission decision. Any raised priority
works (-1, -2 and -3 all measure ~285 us), so the buffer takes whatever the
device offers. `num_sms` matters now for the first time, since it is how many
blocks the collective is trying to get admitted: 158 us hidden at 16 SMs,
223.6 at 32, 285.3 at 64.

The timeline confirms the mechanism rather than just the number. Before, the
GEMM started 9-10 us *after* the scatter ended, every iteration, and the two
were never resident together. After:

```
scatter [2916.8 -> 3770.6]   GEMM [2605.8 -> 3505.6]   588.8 us concurrent
scatter [4057.4 -> 4892.2]   GEMM [3781.8 -> 4704.4]   647.0 us concurrent
```

`num_sms` is now the knob that matters, and 64 is already the peak: 158 us
hidden at 16, 223.6 at 32, 285.3 at 64, 267.7 at 96, and 96.9 at 128, where so
many blocks are queued for admission that the collective starves itself.

**The second fix: do not spend the free admission slot on a dtype cast.** The
admission stall above is not paid uniformly. There is exactly one moment per
iteration when it is free -- the ~2.5 us window after the previous scatter
releases its SMs and before the GEMM's next wave refills them -- and whichever
operation is queued first on the communication stream gets it. Everything
behind it pays ~90-140 us.

That slot was going to `topk_idx.to(int32)`, a 3 us elementwise kernel that the
buffer issued inside the communication-stream block, 0.4 us ahead of the GEMM.
The 850 us collective behind it then paid the full 129.9 us. Converting on the
caller's stream instead -- before `wait_stream`, so the dependency still covers
it -- puts the collective's own kernel first in the queue. Worth 144.6 us, or
43% to 64% hidden, for moving two lines above a `wait`.

Isolated first from the caller side, by pre-converting the tensors so the
buffer's cast becomes a no-op: 285.0 us hidden with the cast, 425.7 without.

DeepEP is at 81%, and the gap left is one more instance of the same thing. Both
sides pay the admission tax, but this port pays it twice per iteration, once
per launch (mean 267.6 us), where DeepEP pays it at most once (mean 81.5 us):
its epilogue is pre-admitted through programmatic dependent launch, so the seam
between its two kernels is *negative*, -30.9 us. Ours is a full stall. That is
the next thing to try.

Standalone performance is unchanged -- 898-900 us bf16 dispatch, 984-985
combine, 522 fp8, all within the spread of the numbers above -- so this costs
nothing when there is nothing to overlap with.

Rejected along the way, each with numbers: SM starvation as a *count* problem
(8, 16 and 32 SMs all hid the same ~35 us, and at 8 SMs there are 140 free),
proportionality (99 us and 673 us of compute hid the same absolute amount), the
scatter and barrier spinning (a notify-only dispatch, 63 us and almost entirely
barriers, hid 28.2 us where the full 892 us one hid 29.7), the cooperative
launch (moving padding, stats and reset into a third kernel removes every
`sync_grid`, correct on 8 ranks, and changed nothing -- and note that the
motivation given in the commit for that test was itself wrong: EPv2's intranode
path *does* use `this_grid().sync()`, at `common/comm.cuh:233,251,269` from
`impls/dispatch.cuh:74,398`, so a cooperative launch was never the structural
difference it was claimed to be), grids beyond the SM
count (450 us at 128 blocks against 621 at 256), host run-ahead (both sides
queue thousands of microseconds ahead), and the stream dependency itself
(DeepEP's `stream_control_prologue` takes the same `wait_stream(comm, compute)`
when no `previous_event` is given, and overlaps anyway).

Note that `async_finish` changes neither column. It moves who waits, not how
long the work takes: measured, the same dispatch is 901.2 µs synchronous and
901.4 µs asynchronous with the event waited on. Timing an asynchronous call
*without* waiting reports 3.0 µs, which is the launch and nothing else.

## Not implemented

DeepEP intranode features this port does not cover. None of them is blocked by
a design decision here -- each would be an additive parameter or output:

| | |
|---|---|
| `kAllowMultipleReduction` | combine-side local sum across several experts on one rank -- what combine needs before it can consume an expanded dispatch, see below |
| `deterministic` mode | DeepEP has a separate prologue for it |
| `use_tma_aligned_col_major_sf` | column-major scale-factor layout for a downstream GEMM |

Out of scope by design: RDMA/scaleout (`num_qps`), the low-latency decode path,
and Engram/PP/CP.

Also deliberately absent: combine's `topk_weights` side output. DeepEP carries
the gate weights back through combine because its *expanded* layout reorders
them, so the source rank cannot reconstruct which weight went with which slot.
In this port's rank layout the source rank is the one that produced
`topk_weights` and still holds it in its original order, so the output would be
a collective that returns its own input.

## Running

```bash
python example_dispatch_combine_correctness.py --num-sms 64
python example_dispatch_combine_benchmark.py  --num-sms 64
pytest test_example_deepep_v2.py
```

Both examples take `--tokens/--hidden/--topk/--experts` and the thread knobs.
Measurements need a quiet machine and warm clocks: an idle B200 sits at
1.3 GHz against a 1.965 GHz boost, and `do_bench` idles the GPU ~10ms per
iteration, so warm before every timed section rather than once per run.
