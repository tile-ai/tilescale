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
recv_x, recv_topk_idx, recv_topk_weights, handle = buf.dispatch(x, topk_idx, topk_weights)

n = handle.num_recv_tokens          # device-to-host read; call outside timed regions
expert_out = my_expert_compute(recv_x[:n], recv_topk_idx[:n], recv_topk_weights[:n])

combined = buf.combine(expert_out, handle)   # [num_tokens, hidden] bf16
```

`dispatch` returns views over the **full receive capacity**; slice with
`handle.num_recv_tokens` when you need the compact rows. On the fp8 path the
first return value is `(values, scales)` and the caller casts back before the
expert computation — `reference.per_token_cast_back` does this.

`dispatch(..., async_finish=True)` skips joining the communication stream and
puts a `finish_event` on the handle; nothing returned may be read until that
event is waited on.

Knobs worth tuning: `num_sms`, `dispatch_threads`, `combine_threads`, and
`reduce_threads` (separate because the reduce wants `hidden / reduce_threads` to
be a whole number of 128-bit loads). The thread defaults are wide (1024) because
that measured at least as fast at every SM count tried.

## Performance

8× B200, full NVLink mesh, 8192 tokens/rank, hidden 7168, top-8, 256 experts,
64 SMs. Bandwidth is the bottleneck rank's, over bytes that cross NVLink.

| | dispatch | combine |
|---|---|---|
| bf16, whole call | **680–685 GB/s** | **631–637 GB/s** |
| bf16, kernel only | 694 GB/s | — |
| fp8, whole call | **618–625 GB/s** | 629–633 GB/s |

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
| 64 | 681–692 GB/s | 635–642 GB/s |
| 24 | 589–599 GB/s | 557–562 GB/s |

That is −13% / −12% against DeepEP's −11% / −9% over the same range. At 24 SMs
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
| **end to end** | **~562 µs** | **517 µs** |

**combine** (bf16 both sides):

| | DeepEP | this port |
|---|---|---|
| store-back | 839 µs (745 GB/s) | 850 µs (723 GB/s) |
| reduce | 155 µs | 125 µs |
| **end to end** | **994 µs** | **968–977 µs** |

**A whole layer's collectives** -- what a plain dispatch → expert → combine
loop actually pays, since neither epilogue has anything to hide behind there:

| | DeepEP | this port |
|---|---|---|
| dispatch | 442 + 129 µs | 510–518 µs |
| combine | 839 + 156 µs | 965–974 µs |
| **total** | **~1566 µs** | **1475–1492 µs** |

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

DeepEP's figures are GPU time only; this port's are whole Python calls,
including ~9 µs of host work.

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
