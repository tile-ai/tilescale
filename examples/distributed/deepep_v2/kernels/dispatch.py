"""Dispatch: whole-grid notify, then a deduplicated scatter into the compact output.

Follows DeepEP EPv2's ``impls/dispatch.cuh``, intranode/NVLink only: no RDMA,
no expert alignment, no expand layout. Payload may be bf16 or fp8 (per-token,
per-128-element scales, DeepEP's ``per_token_cast_to_fp8`` layout, packed
right after the payload -- see below).

Three phases in one kernel, no host round trip between them:

1. **Count** -- every warp scans this rank's ``topk_idx``, deduplicates per
   (token, destination rank) with ``T.match_any_sync``, and tallies into a
   per-warp slice of shared memory. Dedup makes the recording lanes hold
   pairwise-distinct destinations, so the tally needs no atomics; one global
   ``atom_add`` per (block, destination) folds the slices into ``send_count``.
2. **Exchange** -- one warp publishes this rank's count vector to *every* peer,
   so all ranks hold the full ``count_matrix[sender][destination]`` and can each
   derive ``send_base[d]``, where their rows begin in destination ``d``'s output.
3. **Scatter** -- one warp per token. A token's deduplicated destinations claim
   slots in one round of ``atom_add`` on a *local* counter, and the warp pushes
   the row plus metadata to ``send_base[d] + slot`` on peer ``d``.

**No copy epilogue.** DeepEP stages rows per sender and compacts them in a
second kernel, because expand layout and expert alignment hide the final
position at send time. Here the full count matrix makes ``send_base[d] + slot``
the final compact index, so rows land in place -- saving a whole local
read+write of the payload, at the cost of the scatter having to wait for the
exchange.

**Requires** the `address_of` fix in `src/transform/legalize_safe_memory_access.cc`:
without it, bounds checking rewrites `slot_counter[dst_rank]` into an
`if_then_else` and the `address_of` that `atom_add`/`st` build around it is
rejected at codegen. The workaround was an inline clamp on every such index
plus disabling `LoopUnswitching`; both are gone.

**Why ``put_warp`` and not TMA.** The TMA path was built and never won (572.6
against 563.5 GB/s at 512 threads, 518.9 against 520.4 at 256 -- inside the
noise). ``cp.async.bulk`` has no global-to-global form, so a TMA store must
stage through shared memory, trading that round trip for the issue slots
``put_warp`` spends.

**Why FP8's scale is packed into the same row as the payload, not a second
buffer.** It used to be a second ``put_warp`` per token-destination pair.
Measured: 566.9us with it, 493.1us without -- the payload copy alone already
scales cleanly with bytes (493.1us is almost exactly half of bf16's ~895us,
matching fp8's half-size payload), and a *large* transfer split into two
``put_warp`` calls costs nothing extra (901.8us against a 894.8us single-call
baseline, same total bytes). So the ~74us is not "a second call" in general --
it is a second call for something this small. Whatever fixed cost a remote
store pays regardless of size (peer-address translation, warp-level setup, an
NVLink round trip) is noise against a 7168-byte payload and dominates a
224-byte scale. Packing scale bytes right after the payload -- what
``reference.per_token_cast_to_fp8`` produces -- turns two stores into one and
costs nothing extra upstream: quantisation already has to write its output
somewhere.

Fused, the real number is ~522us against the two-store 547-580us -- a win,
but not the full ~74us the isolated single-call-vs-two-call comparison above
suggested, because the fused call also moves the 224 scale bytes the no-scale
test did not (7392 against 7168) and then the row padding on top of that.
How wide to pad is its own measured tradeoff, and `reference.packed_row_bytes`
owns it; `row_bytes` arrives here already decided.

**The expanded layout** (``do_expand``, DeepEP's ``kDoExpand``). Off, a token
that picks two experts on one rank is deduplicated into a single received row,
and rows are grouped by *sender*. On, it becomes one row per (token, expert),
and rows are grouped by *local expert* -- what a grouped GEMM needs, since each
expert's rows are then contiguous.

DeepEP expands in a receiver-side copy epilogue, which can afford an atomic
bump per expert because it is already reading every staged row. This port has
no epilogue at all -- rows land at their final index straight from the sender
-- so the sender has to know that index, and the only thing standing in the way
is the *granularity of the count exchange*. Everything else is unchanged: phase
1 tallies per destination, phase 2 turns the tallies into a base, phase 3 adds
a locally-claimed slot. Expanding only redefines "destination" from rank to
expert, so ``n_dst`` is ``num_experts`` instead of ``num_ranks`` and the same
three phases produce the expert-major layout.

The one genuinely new term is the segment base: the destination groups its
output by local expert, so this rank's rows for expert ``e`` start at
``segment_base[e] + sender_base[e]``, where ``segment_base`` is the exclusive
prefix sum over the destination's local experts of their *aligned* received
counts, and ``sender_base`` is what phase 2 already computed. Every rank holds
the whole count matrix, so every rank computes the same segment bases without
talking to anyone.

``expert_alignment`` rounds each segment up. The gap between an expert's real
count and its aligned one is never written by the scatter, so it holds
whatever the previous call left; ``zero_padding`` clears it. Turn it off only
if the consumer honours the unaligned counts.

**Capacity.** Deduplicated, a peer can send at most ``num_max_tokens_per_rank``
rows, so ``num_ranks`` times that is a bound no routing can exceed. Expanded,
the bound is ``min(topk, experts_per_rank)`` times higher -- every token
picking every one of its experts on the same rank -- which at the V3 shape is
7 GiB against 0.88. Balanced routing needs only the 0.88, so capacity is the
caller's ``expand_factor`` and phase 2 checks it: every rank derives the same
total from the same count matrix, so an overflow is known *before* any payload
moves, and the destination is skipped rather than corrupted. See
``buffer.py``'s ``expand_overflow``.

**Reusing a layout** (``cached``, DeepEP's ``handle=``). Phases 1-2 depend only
on the routing, so a second dispatch with the same ``topk_idx`` recomputes a
layout it already has. ``cached`` traces the notify kernel away entirely and
leaves only the scatter, which reads ``send_base``/``send_rank_mask``/
``num_recv`` exactly as the previous call left them -- none of them is touched
by the end-of-call reset, which clears only what phase 1-2 consume.

Splitting the kernel is what makes this worth anything. Fused, notify was
welded to the scatter and skipping it meant skipping the payload too; DeepEP
has the same problem from the other direction and its cached dispatch measures
441-444us against 437-439 uncached -- i.e. it saves host work, not GPU time.
Here the whole 34us kernel disappears.

The entry barrier moves with it. It stops a peer's next round from overwriting
data this rank has not finished reading, which is needed whether or not the
layout was recomputed, so when ``cached`` it opens the scatter instead.

**Two launches, not one** (``scatter_sms``). Phases 1-2 compute a layout and
phase 3 moves the payload, and they are separate kernels: the launch boundary
gives phase 3 everything phase 2 wrote, which is exactly what the in-kernel
spin on ``exchange_done`` used to buy. Splitting is free -- 881.7us against
883.7 fused at the V3 shape, the extra launch absorbed by the spinner it
replaces -- and it removes a constraint. The fused kernel ends in
``T.sync_grid()``, so every block has to be resident and the grid is pinned to
``num_sms``; that pinned the *scatter* too, even though the scatter itself
needs no grid-wide rendezvous until the reset at the end. Now the scatter can
be wider: at ``scatter_sms=128`` dispatch is 849.1us bf16 and 487.0 fp8,
against 883.7 and 507.4 fused, about 4%.

It is not DeepEP's split. DeepEP cuts *after* the movement, into a copy
epilogue that compacts rows staged per sender; this port's scatter already
writes rows at their final index, so there is nothing there to extract. The cut
here is before the movement instead, and the piece it isolates -- 34.7us of
layout computation, constant across dtypes -- is the piece a cached ``handle=``
dispatch would skip outright.

``scatter_sms`` defaults to ``num_sms``. A caller who set ``num_sms`` low to
leave the rest of the device for expert compute did not thereby ask for a
scatter that takes it back, so widening is opt-in.
"""

import tilelang
import tilelang.language as T


def _dedup_leader(value, lane):
    """1 on the lowest-indexed lane holding `value` -- DeepEP's `ptx::deduplicate`.

    `T.match_any_sync` is warp-collective, so callers must evaluate this
    unconditionally, never behind a lane-divergent guard or a short-circuiting
    `and`. The int32 spelling of "bits below my lane" is deliberate: the uint32
    `(1u << lane) - 1` trips TVM's bounds analysis.
    """
    match_mask = T.Cast("int32", T.match_any_sync(value))
    return T.if_then_else((match_mask & (~(-1 << lane))) == 0, 1, 0)


@tilelang.jit(compile_once=True)
def dispatch_kernel(
    num_tokens: int,
    num_ranks: int,
    num_experts: int,
    topk: int,
    hidden: int,
    num_max_tokens_per_rank: int,
    num_sms: int,
    threads: int = 256,
    dtype=T.bfloat16,
    scale_dim: int = 0,
    row_bytes: int = 0,
    collect_expert_stats: bool = False,
    do_expand: bool = False,
    expert_alignment: int = 1,
    zero_padding: bool = True,
    expand_capacity: int = 0,
    scatter_sms: int = 0,
    cached: bool = False,
):
    assert threads % 32 == 0
    assert num_experts % num_ranks == 0
    assert topk <= 32, "one lane per top-k entry"
    assert expert_alignment >= 1
    assert do_expand or expert_alignment == 1, "expert_alignment only means anything in the expanded layout"
    experts_per_rank = num_experts // num_ranks
    # Rank or expert: the only thing `do_expand` really changes. See the
    # module docstring.
    n_dst = num_experts if do_expand else num_ranks
    warps_per_cta = threads // 32
    total_warps = num_sms * warps_per_cta
    # The scatter is its own launch, so it is free to be wider than the
    # notify grid. See the module docstring.
    scatter_blocks = scatter_sms or num_sms
    scatter_warps = scatter_blocks * warps_per_cta
    cap = num_max_tokens_per_rank
    total_capacity = cap * num_ranks
    # Deduplicated, `total_capacity` is a bound no routing can exceed. Expanded
    # it is not, so the caller sizes it and phase 2 checks. See the module
    # docstring.
    recv_capacity = expand_capacity if do_expand else total_capacity
    assert not do_expand or recv_capacity > 0, "the expanded layout needs an explicit expand_capacity"

    # `scale_dim` is a Python int, so this branch is resolved while tracing.
    # FP8's `x`/`recv_x` is the *packed* row `reference.per_token_cast_to_fp8`
    # produces: `hidden` payload bytes followed by the per-group fp32 scale,
    # then padding, moved as one opaque `uint8` region so the scatter needs
    # one `put_warp` per token-destination pair instead of two -- see the
    # module docstring. `row_bytes` is `reference.packed_row_bytes`'s result,
    # passed in by `buffer.py` rather than recomputed here: which padding is
    # fastest is a measured, size-dependent call and wants one owner (see
    # that function's docstring for the numbers).
    # bf16 (`scale_dim == 0`) is untouched: `row_width`/`row_dtype` degenerate
    # to exactly what they were before this existed.
    if scale_dim:
        assert row_bytes >= hidden * (dtype.bits // 8) + scale_dim * 4, (
            f"row_bytes={row_bytes} cannot hold {hidden} payload + {scale_dim} scales"
        )
        row_width = row_bytes
        row_dtype = T.uint8
    else:
        row_width = hidden
        row_dtype = dtype

    # Another trace-time branch: off, the tally below is absent from the
    # generated code and the argument degenerates to a one-element stand-in.
    stats_dim = experts_per_rank if collect_expert_stats else 1
    # Likewise for the expanded layout's own outputs. Gated on
    # `do_expand`, not on `collect_expert_stats` -- different features that
    # happen to be per-local-expert.
    count_dim = experts_per_rank if do_expand else 1
    seg_dim = experts_per_rank + 1 if do_expand else 1

    @T.prim_func
    def main(
        x: T.Tensor((num_tokens, row_width), row_dtype),
        topk_idx: T.Tensor((num_tokens, topk), T.int32),
        topk_weights: T.Tensor((num_tokens, topk), T.float32),
        # `notify_done` counts arriving blocks in phase 1; `exchange_done`
        # releases every block once phase 2 has published the offsets.
        notify_done: T.Tensor((1,), T.uint32),
        exchange_done: T.Tensor((1,), T.uint32),
        send_count: T.Tensor((n_dst,), T.uint32),
        # `count_matrix[sender * n_dst + destination]`, -1 until published.
        # Symmetric: every rank writes its own row into every peer's copy.
        # "Destination" is a rank, or an expert when expanding.
        count_matrix: T.Tensor((num_ranks * n_dst,), T.int32),
        send_base: T.Tensor((n_dst,), T.int32),
        psum_recv_count: T.Tensor((num_ranks,), T.int32),
        num_recv: T.Tensor((1,), T.int32),
        slot_counter: T.Tensor((n_dst,), T.uint32),
        # Bit r set iff this token went to rank r. Free to produce here, and it
        # is what lets `kernels/combine.py`'s reduce pass know which slots are
        # live without redoing the dedup (DeepEP recomputes it there instead).
        send_rank_mask: T.Tensor((num_tokens,), T.int32),
        barrier: T.Tensor((4 * num_ranks,), T.int32),
        # Flat 1D: `put_warp` takes a raw address, and `T.address_of(buf[row, 0])`
        # on a buffer with a large row stride trips "Can't fetch the lanes of a
        # scalable vector" in `StorageRewrite`; `st` only accepts single-index
        # buffer loads at all.
        recv_x: T.Tensor((recv_capacity * row_width,), row_dtype),
        recv_src_rank: T.Tensor((recv_capacity,), T.int32),
        recv_src_token: T.Tensor((recv_capacity,), T.int32),
        recv_topk_idx: T.Tensor((recv_capacity * topk,), T.int32),
        recv_topk_weights: T.Tensor((recv_capacity * topk,), T.float32),
        # DeepEP's `cumulative_local_expert_recv_stats`: tokens received per
        # *local* expert, accumulated across calls. uint32 for `atom_add`, as
        # with `send_count`. Never reset here -- the caller owns the window
        # it is accumulating over.
        recv_expert_stats: T.Tensor((stats_dim,), T.uint32),
        # Expanded layout only. `expert_offset` is the exclusive prefix sum of
        # this rank's *aligned* per-expert counts, so local expert `e` owns
        # `[expert_offset[e], expert_offset[e + 1])` and `expert_count[e]` of
        # those rows are real. `expand_overflow` is set if the aligned total
        # does not fit the capacity the caller sized for.
        expert_count: T.Tensor((count_dim,), T.int32),
        expert_offset: T.Tensor((seg_dim,), T.int32),
        expand_overflow: T.Tensor((1,), T.int32),
    ):
        # A single persistent grid: the phases rendezvous through global
        # counters, so every block has to be resident at once. `num_sms`
        # defaults to the device's SM count (see buffer.py) and one block per SM
        # always fits.
        if not cached:
            with T.Kernel(num_sms, threads=threads) as bx:
                tid = T.get_thread_binding()
                lane = tid % 32
                local_warp = tid // 32
                warp = bx * warps_per_cta + local_warp
                # Through a variable, not `T.get_rank()` inline: inline leaves the
                # index range unknown, so bounds checking wraps every
                # `buf[my_rank ...]` in an `if_then_else` that `address_of` rejects.
                my_rank = T.alloc_var(T.int32, init=T.get_rank())

                # Entry barrier, DeepEP's `kDispatchTag0`: stops a peer's *next*
                # round from writing this rank's `count_matrix` before this round's
                # reset has landed. The exit barrier cannot -- it only says a peer
                # reached it, after which the peer may finish and relaunch. Its own
                # slot, disjoint from the exit barrier's; see buffer.py.
                T.barrier_blocks(barrier[0])

                # ---------------- Phase 1: count ----------------
                blk_count = T.alloc_shared((warps_per_cta * n_dst,), "int32")
                # Flat, one row per warp, and deliberately untouched by any tile-level
                # op: keeping `T.copy`/`T.atomic_add` away from it is what keeps the
                # compiler from inserting block-wide barriers into warp-scoped code.
                for i in T.serial(tid, warps_per_cta * n_dst, threads):
                    blk_count[i] = 0
                T.sync_threads()

                for token in T.serial(warp, num_tokens, total_warps):
                    expert = T.alloc_var(T.int32)
                    dst_rank = T.alloc_var(T.int32)
                    expert = -1
                    if lane < topk:
                        expert = topk_idx[token, lane]
                    dst_rank = -1
                    if expert >= 0:
                        dst_rank = expert // experts_per_rank
                    # Expanding, the destination *is* the expert, and the dedup
                    # below is a no-op: DeepEP asserts a token's top-k entries are
                    # distinct experts, so the lanes are already pairwise-distinct.
                    # The counter stays atomic-free for the same reason either way.
                    dst = expert if do_expand else dst_rank
                    leader = T.alloc_var(T.int32, init=_dedup_leader(dst, lane))
                    if leader == 1 and dst >= 0:
                        # No atomic: after dedup the lanes reaching here hold
                        # pairwise-distinct destinations, and each warp owns its own
                        # slice, so no two threads ever touch the same counter.
                        at = local_warp * n_dst + dst
                        blk_count[at] = blk_count[at] + 1
                    # Record the destination set here rather than in the scatter:
                    # combine needs it, and gathering it costs a round of shuffles
                    # that has no business being on the data path.
                    # Rank granularity even when expanding: combine reduces over
                    # ranks either way. Expanding, `leader` is per-expert and this
                    # needs a dedup of its own; not expanding, `dst` *is* `dst_rank`,
                    # so reuse it rather than issue a second warp-collective
                    # `T.match_any_sync` per token for the same answer.
                    if do_expand:
                        rank_leader = T.alloc_var(T.int32, init=_dedup_leader(dst_rank, lane))
                    else:
                        rank_leader = leader
                    rank_mask = T.alloc_var(T.int32, init=0)
                    for k in range(topk):
                        dst_k = T.alloc_var(T.int32)
                        lead_k = T.alloc_var(T.int32)
                        dst_k = T.shfl_sync(dst_rank, k)
                        lead_k = T.shfl_sync(rank_leader, k)
                        if lead_k == 1 and dst_k >= 0:
                            rank_mask = rank_mask + (1 << dst_k)
                    if lane == 0:
                        send_rank_mask[token] = rank_mask
                T.sync_threads()

                for d in T.serial(tid, n_dst, threads):
                    folded = T.alloc_var(T.int32, init=0)
                    for w in range(warps_per_cta):
                        folded = folded + blk_count[w * n_dst + d]
                    if folded > 0:
                        T.atom_add(send_count[d], folded, scope="gpu")
                T.sync_threads()
                if tid == 0:
                    T.atom_add(notify_done[0], 1, scope="gpu", sem="release")

                # ---------------- Phase 2: exchange ----------------
                if bx == 0 and local_warp == 0:
                    if lane == 0:
                        T.wait_ge(notify_done[0], num_sms, scope=T.WaitScope.GPU, semantics=T.WaitSemantics.ACQUIRE)
                    T.sync_warp()

                    # Publish this rank's count vector to every peer, so all ranks
                    # hold the same `count_matrix`.
                    #
                    # One fence, then relaxed stores. Per-store `sem="release"` cost
                    # 4.4%: each carries a `fence.release.sys` that waits for prior
                    # writes to cross NVLink, serialising eight round trips to
                    # publish 64 integers. Peers read `count_matrix` directly and
                    # infer nothing else from it, so the stores need no ordering
                    # against each other.
                    T.fence_sys()
                    # `n_dst` is `num_ranks` unless expanding, where it is
                    # `num_experts` and the row no longer fits one lane each.
                    for p in range(num_ranks):
                        for c in T.serial(lane, n_dst, 32):
                            T.st(count_matrix[my_rank * n_dst + c], send_count[c], scope="sys", sem="relaxed", dst_pe=p)
                    T.sync_warp()
                    for s in range(num_ranks):
                        for c in T.serial(lane, n_dst, 32):
                            T.wait_ge(count_matrix[s * n_dst + c], 0, scope=T.WaitScope.SYS, semantics=T.WaitSemantics.ACQUIRE)
                    T.sync_warp()

                    # Lane d: where my rows start inside destination d's output.
                    # Expanding, `d` is an expert and this is only the offset
                    # *within* that expert's segment; the segment base is added
                    # below, once every destination's aligned layout is known.
                    for d in T.serial(lane, n_dst, 32):
                        base = T.alloc_var(T.int32, init=0)
                        for s in T.serial(my_rank):
                            base = base + count_matrix[s * n_dst + d]
                        send_base[d] = base
                    T.sync_warp()

                    if do_expand:
                        # Every destination lays its output out the same way, from
                        # the same count matrix, so each rank can reconstruct all of
                        # them and nobody has to be told. Lane p handles peer p.
                        if lane < num_ranks:
                            seg = T.alloc_var(T.int32, init=0)
                            for e in range(experts_per_rank):
                                total = T.alloc_var(T.int32, init=0)
                                for s in range(num_ranks):
                                    total = total + count_matrix[s * n_dst + lane * experts_per_rank + e]
                                # My rows for this expert sit at the segment base
                                # plus the offset among earlier senders.
                                send_base[lane * experts_per_rank + e] = seg + send_base[lane * experts_per_rank + e]
                                # Aligned, so the next segment starts on a boundary
                                # a grouped GEMM can consume.
                                seg = seg + T.ceildiv(total, expert_alignment) * expert_alignment
                                if lane == my_rank:
                                    expert_count[e] = total
                                    expert_offset[e + 1] = seg
                            # `seg` is now this destination's aligned total. Known
                            # to every rank before a single payload byte has moved,
                            # which is what makes the capacity check free.
                            if lane == my_rank:
                                expert_offset[0] = 0
                                num_recv[0] = seg
                                if seg > recv_capacity:
                                    expand_overflow[0] = seg
                            if seg > recv_capacity:
                                # Skip this destination entirely rather than write
                                # past its buffer. It raises its own flag.
                                for e in range(experts_per_rank):
                                    send_base[lane * experts_per_rank + e] = -1
                    else:
                        # Lane 0: my own receive prefix sum (DeepEP's
                        # `psum_num_recv_tokens_per_scaleup_rank`) and total.
                        if lane == 0:
                            psum = T.alloc_var(T.int32, init=0)
                            for s in range(num_ranks):
                                psum = psum + count_matrix[s * num_ranks + my_rank]
                                psum_recv_count[s] = psum
                            num_recv[0] = psum
                    T.sync_warp()
                    if lane == 0:
                        T.atom_add(exchange_done[0], 1, scope="gpu", sem="release")

                # Everything phase 2 published is visible to the next kernel
                # across the launch boundary, which is what the spin on
                # `exchange_done` used to be for.

        with T.Kernel(scatter_blocks, threads=threads) as bx:
            tid = T.get_thread_binding()
            lane = tid % 32
            local_warp = tid // 32
            warp = bx * warps_per_cta + local_warp
            my_rank = T.alloc_var(T.int32, init=T.get_rank())

            # With no notify kernel ahead of it, the scatter owns the entry
            # barrier. It is not about the layout -- it stops a peer's next
            # round from landing on data this rank is still reading -- so it is
            # needed either way, just in whichever kernel goes first.
            if cached:
                T.barrier_blocks(barrier[0])

            # ---------------- Phase 3: scatter ----------------
            for token in T.serial(warp, num_tokens, scatter_warps):
                expert = T.alloc_var(T.int32)
                dst_rank = T.alloc_var(T.int32)
                slot = T.alloc_var(T.int32)
                expert = -1
                if lane < topk:
                    expert = topk_idx[token, lane]
                dst_rank = -1
                if expert >= 0:
                    dst_rank = expert // experts_per_rank
                dst = expert if do_expand else dst_rank
                leader = T.alloc_var(T.int32, init=_dedup_leader(dst, lane))
                # Every destination of this token claims its slot at once, one
                # atomic per owning lane. `send_base[dst] < 0` marks a
                # destination phase 2 found would overflow, which only the
                # expanded layout can hit -- deduplicated, capacity is a hard
                # bound -- but the condition is the same either way and
                # `send_base` is eight hot integers.
                slot = -1
                if leader == 1 and dst >= 0 and send_base[dst] >= 0:
                    slot = T.atom_add(slot_counter[dst], 1, scope="gpu")

                for k in range(topk):
                    dst_k = T.alloc_var(T.int32)
                    slot_k = T.alloc_var(T.int32)
                    dst_k = T.shfl_sync(dst, k)
                    # Expanding, the destination is an expert and the peer that
                    # owns it has to be shuffled separately; otherwise they are
                    # the same value and the second shuffle is pure cost.
                    if do_expand:
                        pe_k = T.shfl_sync(dst_rank, k)
                    else:
                        pe_k = dst_k
                    slot_k = T.shfl_sync(slot, k)
                    if slot_k >= 0:
                        # The final index on the destination -- known here only
                        # because phase 2 handed every rank the full count
                        # matrix. Expanding, it is already inside the right
                        # expert's segment.
                        idx_k = T.alloc_var(T.int32, init=send_base[dst_k] + slot_k)
                        T.put_warp(
                            src=T.address_of(x[token, 0]),
                            dst=T.address_of(recv_x[idx_k * row_width]),
                            size=row_width,
                            dst_pe=pe_k,
                        )
                        if lane == 0:
                            T.st(recv_src_rank[idx_k], my_rank, scope="sys", sem="relaxed", dst_pe=pe_k)
                            T.st(recv_src_token[idx_k], token, scope="sys", sem="relaxed", dst_pe=pe_k)
                        if lane < topk:
                            local_expert = T.alloc_var(T.int32, init=-1)
                            if pe_k * experts_per_rank <= expert and expert < (pe_k + 1) * experts_per_rank:
                                local_expert = expert - pe_k * experts_per_rank
                            T.st(recv_topk_idx[idx_k * topk + lane], local_expert, scope="sys", sem="relaxed", dst_pe=pe_k)
                            T.st(recv_topk_weights[idx_k * topk + lane], topk_weights[token, lane], scope="sys", sem="relaxed", dst_pe=pe_k)

            T.barrier_blocks(barrier[num_ranks])

            # Reset for the next call. The `T.sync_grid()` is load-bearing:
            # `tl::barrier_blocks` rendezvouses *ranks*, not this rank's own
            # blocks, so behind it alone block 0 zeroes `slot_counter` while
            # other blocks are still claiming slots -- which silently produces
            # duplicate (src_rank, src_token) rows. DeepEP's `gpu_barrier` uses
            # `this_grid().sync()` for exactly this.
            T.sync_grid()

            # ---------------- Alignment padding ----------------
            # The scatter writes only real rows, so the gap between an expert's
            # count and its aligned segment end still holds whatever the last
            # call put there. A grouped GEMM reading the aligned segment would
            # consume it, so clear it -- DeepEP's `kDoZeroPadding`, and for the
            # same reason. Rows, not bytes: one warp per padding row, strided
            # over the whole grid.
            if do_expand and zero_padding:
                for e in range(experts_per_rank):
                    pad_begin = T.alloc_var(T.int32, init=expert_offset[e] + expert_count[e])
                    pad_end = T.alloc_var(T.int32, init=expert_offset[e + 1])
                    for row in T.serial(pad_begin + warp, pad_end, scatter_warps):
                        for i in T.serial(lane, row_width, 32):
                            recv_x[row * row_width + i] = T.Cast(row_dtype, 0)
                        if lane == 0:
                            # A padding row belongs to no sender and no token;
                            # combine skips anything marked this way.
                            recv_src_rank[row] = -1
                            recv_src_token[row] = -1
                        if lane < topk:
                            recv_topk_idx[row * topk + lane] = -1
                            recv_topk_weights[row * topk + lane] = T.Cast(T.float32, 0)

            # ---------------- Per-expert receive stats (optional) ----------------
            # Counted here, on the receiver, rather than exchanged like the rank
            # counts: `recv_topk_idx` already holds the local expert for every
            # received (row, top-k slot) and -1 elsewhere, so the answer is a
            # local scan of something this rank was going to be handed anyway.
            # DeepEP derives it in notify instead, from a per-expert count
            # vector it already exchanges -- it has one because the expanded
            # layout needs per-expert offsets. This port's layout does not, and
            # widening the exchange from `num_ranks` to `num_experts` entries
            # per rank to avoid a scan that costs microseconds is a bad trade.
            #
            # Behind the exit barrier and `sync_grid`, so every peer's stores
            # have landed and this rank's own blocks are past the scatter.
            if collect_expert_stats:
                n_recv = T.alloc_var(T.int32, init=num_recv[0])
                for i in T.serial(bx * threads + tid, n_recv * topk, scatter_blocks * threads):
                    local_expert = T.alloc_var(T.int32, init=recv_topk_idx[i])
                    if local_expert >= 0:
                        T.atom_add(recv_expert_stats[local_expert], 1, scope="gpu")

            if bx == 0:
                for d in T.serial(tid, n_dst, threads):
                    send_count[d] = 0
                    slot_counter[d] = 0
                for c in T.serial(tid, num_ranks * n_dst, threads):
                    count_matrix[c] = -1
                if tid == 0:
                    notify_done[0] = 0
                    exchange_done[0] = 0

    return main
