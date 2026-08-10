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
):
    assert threads % 32 == 0
    reduce_threads = reduce_threads or threads
    assert reduce_threads % 32 == 0
    warps_per_cta = threads // 32
    total_warps = num_sms * warps_per_cta
    cap = num_max_tokens_per_rank

    # Symbolic so the caller's contribution tensor can be read in place at
    # whatever row count the last dispatch produced.
    num_elems = T.symbolic("num_elems")

    @T.prim_func
    def main(
        x: T.Tensor((num_elems,), dtype),
        recv_src_rank: T.Tensor((total_capacity,), T.int32),
        recv_src_token: T.Tensor((total_capacity,), T.int32),
        num_recv: T.Tensor((1,), T.int32),
        send_rank_mask: T.Tensor((num_tokens,), T.int32),
        barrier: T.Tensor((4 * num_ranks,), T.int32),
        comm_x: T.Tensor((num_ranks * cap * hidden,), dtype),
        combined: T.Tensor((num_tokens, hidden), dtype),
    ):
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
            for i in T.serial(warp * per_warp, T.min((warp + 1) * per_warp, n_recv)):
                slot = my_rank * cap + recv_src_token[i]
                T.put_warp(
                    src=T.address_of(x[i * hidden]),
                    dst=T.address_of(comm_x[slot * hidden]),
                    size=hidden,
                    dst_pe=recv_src_rank[i],
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
            T.clear(acc)
            for r in range(num_ranks):
                if ((mask >> r) & 1) == 1:
                    base = (r * cap + token) * hidden
                    for i in T.Parallel(hidden):
                        acc[i] += comm_x[base + i]
            T.copy(acc, combined[token, :])

    return main
