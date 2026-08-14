"""Combine: remote store-back, then a local reduce.

Mirrors DeepEP's ``combine_impl`` + ``combine_reduce_epilogue_impl`` as two
kernel launches:

- **Store-back** -- one warp per compact row, ``put_warp`` into
  ``comm_x[my_rank][src_token]`` on the source rank. That slot is unique per
  (contributing rank, source token), so nothing needs to be atomic.
- **Reduce** -- one block per source token, summing the slots named by
  ``send_rank_mask[token]``, the deduplicated destination set ``dispatch``
  recorded.

Gate weights are *not* applied here. Like DeepEP, weighting and the local sum
across several experts on one rank belong to the expert epilogue; ``x`` is
already the per-compact-row contribution.

**The expanded layout** (``do_expand``). The store-back slot
``comm_x[rank][src_token]`` is unique only because a deduplicated dispatch
gives a token one row per destination rank. Expanded, a token with two experts
here has two rows and they would collide.

DeepEP's answer is ``kDoExpandedSend``: sum a token's local-expert rows before
sending, so one row per (rank, token) still crosses NVLink and ``comm_x`` and
the reduce stay exactly as they are. That needs the inverse of what dispatch
recorded -- dispatch gives row -> (src_rank, src_token), and this needs
(src_rank, src_token) -> rows -- so an extra kernel builds it, bucketing each
received row under its source. It touches metadata only, never the payload.

The store-back then runs one warp per *group* rather than per row, elected by
the group's first row, and the common case is unchanged: with routing spread
over many ranks most groups hold a single row, which is sent straight from
``x`` with no summing and no staging. Only groups of two or more accumulate
into ``reduce_scratch`` first, one row of it per warp.

Up to two bias tensors may be added to the output, DeepEP's ``bias_0``/
``bias_1``. They seed the reduce accumulator rather than being added to it
afterwards, which costs nothing: the accumulator had to be written once
either way, and seeding replaces the clear. A token with no contributions at
all still gets its bias, which is what makes bias-only tokens behave.
"""

import tilelang
import tilelang.language as T


@tilelang.jit(compile_once=True)
def combine_kernel(
    num_tokens: int,
    num_ranks: int,
    hidden: int,
    num_max_tokens_per_rank: int,
    total_capacity: int,
    num_sms: int,
    threads: int = 256,
    reduce_threads: int = 0,
    dtype=T.bfloat16,
    num_bias: int = 0,
    do_expand: bool = False,
    max_rows_per_token: int = 1,
    recv_capacity: int = 0,
):
    assert threads % 32 == 0
    assert 0 <= num_bias <= 2, f"DeepEP takes at most two bias tensors, got {num_bias}"
    reduce_threads = reduce_threads or threads
    assert reduce_threads % 32 == 0
    warps_per_cta = threads // 32
    total_warps = num_sms * warps_per_cta
    cap = num_max_tokens_per_rank

    # Symbolic so the caller's contribution tensor can be read in place at
    # whatever row count the last dispatch produced.
    num_elems = T.symbolic("num_elems")

    # Trace-time, like dispatch's `scale_dim`: an unused bias degenerates to a
    # one-row stand-in and every reference to it is absent from the generated
    # code, not predicated in it.
    n_bias_0 = num_tokens if num_bias >= 1 else 1
    n_bias_1 = num_tokens if num_bias >= 2 else 1
    # Deduplicated there is one row per (rank, token) and nothing to group, so
    # the grouping buffers degenerate.
    # Expanded, the receive buffers are wider than `total_capacity`; see
    # buffer.py's `recv_capacity`.
    n_recv_slots = recv_capacity or total_capacity
    max_k = max_rows_per_token if do_expand else 1
    n_pairs = num_ranks * cap if do_expand else 1
    n_scratch = total_warps * hidden if do_expand else 1

    @T.prim_func
    def main(
        x: T.Tensor((num_elems,), dtype),
        recv_src_rank: T.Tensor((n_recv_slots,), T.int32),
        recv_src_token: T.Tensor((n_recv_slots,), T.int32),
        num_recv: T.Tensor((1,), T.int32),
        send_rank_mask: T.Tensor((num_tokens,), T.int32),
        barrier: T.Tensor((4 * num_ranks,), T.int32),
        comm_x: T.Tensor((num_ranks * cap * hidden,), dtype),
        # Expanded layout only: the (src_rank, src_token) -> rows inversion,
        # and one scratch row per warp for groups that need summing.
        group_count: T.Tensor((n_pairs,), T.uint32),
        group_rows: T.Tensor((n_pairs * max_k,), T.int32),
        reduce_scratch: T.Tensor((n_scratch,), dtype),
        bias_0: T.Tensor((n_bias_0, hidden), dtype),
        bias_1: T.Tensor((n_bias_1, hidden), dtype),
        combined: T.Tensor((num_tokens, hidden), dtype),
    ):
        # ---------------- Bucket rows by source (expanded only) ----------------
        if do_expand:
            with T.Kernel(num_sms, threads=threads) as bx:
                tid = T.get_thread_binding()
                n_recv = T.alloc_var(T.int32, init=num_recv[0])
                for i in T.serial(bx * threads + tid, n_pairs, num_sms * threads):
                    group_count[i] = 0
                # The fill below reads counters the zeroing above writes, and
                # any block may touch any counter.
                T.sync_grid()
                # Names distinct from the store-back kernel's below: the
                # tracer binds an `alloc_var` name for the whole traced
                # function, so reusing one across the two kernels reads as the
                # same immutable variable escaping its region.
                for i in T.serial(bx * threads + tid, n_recv, num_sms * threads):
                    bucket_src = T.alloc_var(T.int32, init=recv_src_rank[i])
                    # Alignment padding is marked -1 by dispatch and belongs to
                    # no token.
                    if bucket_src >= 0:
                        bucket_pair = T.alloc_var(T.int32, init=bucket_src * cap + recv_src_token[i])
                        bucket_slot = T.alloc_var(T.int32, init=T.atom_add(group_count[bucket_pair], 1, scope="gpu"))
                        if bucket_slot < max_k:
                            group_rows[bucket_pair * max_k + bucket_slot] = i

        with T.Kernel(num_sms, threads=threads) as bx:
            tid = T.get_thread_binding()
            # Through a variable, not inline: see kernels/dispatch.py.
            my_rank = T.alloc_var(T.int32, init=T.get_rank())

            # Entry barrier. The exit barrier only says every store-back landed,
            # not that every rank's *reduce* has read it, so without this a fast
            # rank overwrites slots a slow one is still reducing. Buying that
            # here is what lets `combine()` run with no collective around it.
            T.barrier_blocks(barrier[2 * num_ranks])

            # A contiguous chunk per warp, indexed warp-major -- DeepEP's
            # `global_warp_idx`. Compact rows are grouped by source rank, so the
            # obvious `bx * warps_per_cta + warp` gives one block a few hundred
            # consecutive rows, all bound for one peer, and funnels the block
            # down a single NVLink. Warp-major spreads a block's warps across
            # the whole compact range instead; rotating the start by `my_rank`
            # keeps the ranks from walking the peers in lockstep. Together worth
            # 512 -> 609 GB/s, of which the rotation is 598 -> 609.
            warp = ((tid // 32 + my_rank) % warps_per_cta) * num_sms + bx
            # From device memory: as a scalar argument it would force `dispatch`
            # to read the count back to the host, worth ~33us there.
            n_recv = T.alloc_var(T.int32, init=num_recv[0])
            per_warp = T.alloc_var(T.int32, init=T.ceildiv(n_recv, total_warps))
            lane = tid % 32
            # A scratch row per warp, indexed by the *unrotated* block/warp
            # pair. `warp` below folds in `my_rank`, whose range the compiler
            # cannot prove, so using it here would bounds-wrap the index in an
            # `if_then_else` that `address_of` rejects -- the same thing
            # kernels/dispatch.py hit with `T.get_rank()` inline.
            scratch_warp = bx * warps_per_cta + tid // 32
            for i in T.serial(warp * per_warp, T.min((warp + 1) * per_warp, n_recv)):
                src_rank = T.alloc_var(T.int32, init=recv_src_rank[i])
                # Padding rows belong to nobody; only the expanded layout has any.
                if src_rank >= 0:
                    slot = my_rank * cap + recv_src_token[i]
                    if do_expand:
                        pair = T.alloc_var(T.int32, init=src_rank * cap + recv_src_token[i])
                        cnt = T.alloc_var(T.int32, init=group_count[pair])
                        # One warp per group, not per row: the group's first row
                        # elects itself, the rest of the group does nothing.
                        if group_rows[pair * max_k] == i:
                            if cnt == 1:
                                # The common case with routing spread across
                                # ranks -- nothing to sum, send it where it lies.
                                T.put_warp(
                                    src=T.address_of(x[i * hidden]),
                                    dst=T.address_of(comm_x[slot * hidden]),
                                    size=hidden,
                                    dst_pe=src_rank,
                                )
                            else:
                                # Lane-strided rather than `T.Parallel`: this is
                                # warp-scoped code, and a tile-level op here
                                # would have the compiler insert a block-wide
                                # barrier into it.
                                for e in T.serial(lane, hidden, 32):
                                    acc = T.alloc_var(T.float32, init=0.0)
                                    for j in range(max_k):
                                        if j < cnt:
                                            acc = acc + T.Cast(T.float32, x[group_rows[pair * max_k + j] * hidden + e])
                                    reduce_scratch[scratch_warp * hidden + e] = T.Cast(dtype, acc)
                                T.sync_warp()
                                T.put_warp(
                                    src=T.address_of(reduce_scratch[scratch_warp * hidden]),
                                    dst=T.address_of(comm_x[slot * hidden]),
                                    size=hidden,
                                    dst_pe=src_rank,
                                )
                    else:
                        T.put_warp(
                            src=T.address_of(x[i * hidden]),
                            dst=T.address_of(comm_x[slot * hidden]),
                            size=hidden,
                            dst_pe=src_rank,
                        )

            T.barrier_blocks(barrier[3 * num_ranks])

            # No PDL between the two kernels. PDL relaxes exactly the visibility
            # an ordinary launch boundary provides, and what the reduce reads was
            # written by *peer* GPUs, which PDL says nothing about; once the
            # reduce got 3x faster, one rank in eight read stale slots. It saved
            # no measurable time either.

        # One block per source token, not a persistent `num_sms` grid: the
        # reduce has no cross-block state, and inheriting `num_sms=64` left most
        # of the GPU's 148 SMs idle (551us for 733MB, 1.33 TB/s).
        with T.Kernel(num_tokens, threads=reduce_threads) as token:
            # Straight into a fragment, contribution by contribution. Two
            # alternatives measured slower: register-staging every contribution
            # first (1477us against 507us), and replacing the exit barrier with
            # per-token arrival flags to overlap the reduce with the store-back
            # (346-350 GB/s against 366-372) -- the chunks all finish together,
            # so there is no slack to overlap into.
            acc = T.alloc_fragment((hidden,), T.float32)
            mask = send_rank_mask[token]
            # Seed with the biases instead of clearing -- same number of
            # writes to `acc`, and it keeps a token whose every selection was
            # masked off from losing its bias.
            if num_bias >= 1:
                for i in T.Parallel(hidden):
                    acc[i] = bias_0[token, i]
            else:
                T.clear(acc)
            if num_bias >= 2:
                for i in T.Parallel(hidden):
                    acc[i] += bias_1[token, i]
            for r in range(num_ranks):
                if ((mask >> r) & 1) == 1:
                    base = (r * cap + token) * hidden
                    for i in T.Parallel(hidden):
                        acc[i] += comm_x[base + i]
            T.copy(acc, combined[token, :])

    return main
