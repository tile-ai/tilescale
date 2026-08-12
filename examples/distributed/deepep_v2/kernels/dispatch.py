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
):
    assert threads % 32 == 0
    assert num_experts % num_ranks == 0
    assert topk <= 32, "one lane per top-k entry"
    experts_per_rank = num_experts // num_ranks
    warps_per_cta = threads // 32
    total_warps = num_sms * warps_per_cta
    cap = num_max_tokens_per_rank
    total_capacity = cap * num_ranks

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

    @T.prim_func
    def main(
        x: T.Tensor((num_tokens, row_width), row_dtype),
        topk_idx: T.Tensor((num_tokens, topk), T.int32),
        topk_weights: T.Tensor((num_tokens, topk), T.float32),
        # `notify_done` counts arriving blocks in phase 1; `exchange_done`
        # releases every block once phase 2 has published the offsets.
        notify_done: T.Tensor((1,), T.uint32),
        exchange_done: T.Tensor((1,), T.uint32),
        send_count: T.Tensor((num_ranks,), T.uint32),
        # `count_matrix[sender * num_ranks + destination]`, -1 until published.
        # Symmetric: every rank writes its own row into every peer's copy.
        count_matrix: T.Tensor((num_ranks * num_ranks,), T.int32),
        send_base: T.Tensor((num_ranks,), T.int32),
        psum_recv_count: T.Tensor((num_ranks,), T.int32),
        num_recv: T.Tensor((1,), T.int32),
        slot_counter: T.Tensor((num_ranks,), T.uint32),
        # Bit r set iff this token went to rank r. Free to produce here, and it
        # is what lets `kernels/combine.py`'s reduce pass know which slots are
        # live without redoing the dedup (DeepEP recomputes it there instead).
        send_rank_mask: T.Tensor((num_tokens,), T.int32),
        barrier: T.Tensor((4 * num_ranks,), T.int32),
        # Flat 1D: `put_warp` takes a raw address, and `T.address_of(buf[row, 0])`
        # on a buffer with a large row stride trips "Can't fetch the lanes of a
        # scalable vector" in `StorageRewrite`; `st` only accepts single-index
        # buffer loads at all.
        recv_x: T.Tensor((total_capacity * row_width,), row_dtype),
        recv_src_rank: T.Tensor((total_capacity,), T.int32),
        recv_src_token: T.Tensor((total_capacity,), T.int32),
        recv_topk_idx: T.Tensor((total_capacity * topk,), T.int32),
        recv_topk_weights: T.Tensor((total_capacity * topk,), T.float32),
    ):
        # A single persistent grid: the phases rendezvous through global
        # counters, so every block has to be resident at once. `num_sms`
        # defaults to the device's SM count (see buffer.py) and one block per SM
        # always fits.
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
            blk_count = T.alloc_shared((warps_per_cta * num_ranks,), "int32")
            # Flat, one row per warp, and deliberately untouched by any tile-level
            # op: keeping `T.copy`/`T.atomic_add` away from it is what keeps the
            # compiler from inserting block-wide barriers into warp-scoped code.
            for i in T.serial(tid, warps_per_cta * num_ranks, threads):
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
                leader = T.alloc_var(T.int32, init=_dedup_leader(dst_rank, lane))
                if leader == 1 and dst_rank >= 0:
                    # No atomic: after dedup the lanes reaching here hold
                    # pairwise-distinct destinations, and each warp owns its own
                    # slice, so no two threads ever touch the same counter.
                    at = local_warp * num_ranks + dst_rank
                    blk_count[at] = blk_count[at] + 1
                # Record the destination set here rather than in the scatter:
                # combine needs it, and gathering it costs a round of shuffles
                # that has no business being on the data path.
                rank_mask = T.alloc_var(T.int32, init=0)
                for k in range(topk):
                    dst_k = T.alloc_var(T.int32)
                    lead_k = T.alloc_var(T.int32)
                    dst_k = T.shfl_sync(dst_rank, k)
                    lead_k = T.shfl_sync(leader, k)
                    if lead_k == 1 and dst_k >= 0:
                        rank_mask = rank_mask + (1 << dst_k)
                if lane == 0:
                    send_rank_mask[token] = rank_mask
            T.sync_threads()

            if tid < num_ranks:
                folded = T.alloc_var(T.int32, init=0)
                for w in range(warps_per_cta):
                    folded = folded + blk_count[w * num_ranks + tid]
                if folded > 0:
                    T.atom_add(send_count[tid], folded, scope="gpu")
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
                for p in range(num_ranks):
                    if lane < num_ranks:
                        T.st(count_matrix[my_rank * num_ranks + lane], send_count[lane], scope="sys", sem="relaxed", dst_pe=p)
                T.sync_warp()
                for s in range(num_ranks):
                    if lane < num_ranks:
                        T.wait_ge(count_matrix[s * num_ranks + lane], 0, scope=T.WaitScope.SYS, semantics=T.WaitSemantics.ACQUIRE)
                T.sync_warp()

                # Lane d: where my rows start inside destination d's output.
                if lane < num_ranks:
                    base = T.alloc_var(T.int32, init=0)
                    for s in T.serial(my_rank):
                        base = base + count_matrix[s * num_ranks + lane]
                    send_base[lane] = base
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

            # One spinner per block, not one per warp: every polling lane is
            # hitting the same line, and `total_warps` of them contend with the
            # remote traffic phase 2 is trying to get through.
            if tid == 0:
                T.wait_ge(exchange_done[0], 1, scope=T.WaitScope.GPU, semantics=T.WaitSemantics.ACQUIRE)
            T.sync_threads()

            # ---------------- Phase 3: scatter ----------------
            for token in T.serial(warp, num_tokens, total_warps):
                expert = T.alloc_var(T.int32)
                dst_rank = T.alloc_var(T.int32)
                slot = T.alloc_var(T.int32)
                expert = -1
                if lane < topk:
                    expert = topk_idx[token, lane]
                dst_rank = -1
                if expert >= 0:
                    dst_rank = expert // experts_per_rank
                leader = T.alloc_var(T.int32, init=_dedup_leader(dst_rank, lane))
                # Every destination of this token claims its slot at once, one
                # atomic per owning lane.
                slot = -1
                if leader == 1 and dst_rank >= 0:
                    slot = T.atom_add(slot_counter[dst_rank], 1, scope="gpu")

                for k in range(topk):
                    dst_k = T.alloc_var(T.int32)
                    slot_k = T.alloc_var(T.int32)
                    dst_k = T.shfl_sync(dst_rank, k)
                    slot_k = T.shfl_sync(slot, k)
                    if slot_k >= 0:
                        # The final compact index on the destination -- known
                        # here only because phase 2 handed every rank the full
                        # count matrix.
                        idx_k = T.alloc_var(T.int32, init=send_base[dst_k] + slot_k)
                        T.put_warp(
                            src=T.address_of(x[token, 0]),
                            dst=T.address_of(recv_x[idx_k * row_width]),
                            size=row_width,
                            dst_pe=dst_k,
                        )
                        if lane == 0:
                            T.st(recv_src_rank[idx_k], my_rank, scope="sys", sem="relaxed", dst_pe=dst_k)
                            T.st(recv_src_token[idx_k], token, scope="sys", sem="relaxed", dst_pe=dst_k)
                        if lane < topk:
                            local_expert = T.alloc_var(T.int32, init=-1)
                            if dst_k * experts_per_rank <= expert and expert < (dst_k + 1) * experts_per_rank:
                                local_expert = expert - dst_k * experts_per_rank
                            T.st(recv_topk_idx[idx_k * topk + lane], local_expert, scope="sys", sem="relaxed", dst_pe=dst_k)
                            T.st(
                                recv_topk_weights[idx_k * topk + lane], topk_weights[token, lane], scope="sys", sem="relaxed", dst_pe=dst_k
                            )

            T.barrier_blocks(barrier[num_ranks])

            # Reset for the next call. The `T.sync_grid()` is load-bearing:
            # `tl::barrier_blocks` rendezvouses *ranks*, not this rank's own
            # blocks, so behind it alone block 0 zeroes `slot_counter` while
            # other blocks are still claiming slots -- which silently produces
            # duplicate (src_rank, src_token) rows. DeepEP's `gpu_barrier` uses
            # `this_grid().sync()` for exactly this.
            T.sync_grid()
            if bx == 0:
                if tid < num_ranks:
                    send_count[tid] = 0
                    slot_counter[tid] = 0
                if tid < num_ranks * num_ranks:
                    count_matrix[tid] = -1
                if tid == 0:
                    notify_done[0] = 0
                    exchange_done[0] = 0

    return main
