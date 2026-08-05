# Multi-process autotuning for the 2D inter-node collectives

Design notes, not an implementation. The subject is the tuning surface of
`examples/distributed/internode/internode_2d.py` — `Allgather2D`, `ReduceScatter2D`,
`Allreduce2D` and the two fused GEMM wrappers — at 16 GPUs over two nodes. The question is
what a tuner for them has to look like, given that every candidate evaluation is a 16-rank
collective and that the measurement is noisy enough to lie. Everything numeric below is
quoted from measurements already recorded in `CLAUDE.md` and the module docstrings, or is
arithmetic over per-unit costs recorded there; nothing here is a new measurement.

## 1. Why the hand-tuned constants have to go

The knobs are not independent of each other, of the collective, or of the size.

**`--mc-tiles` splits the collectives into two families.** Allgather wants 32 tiles per
multicast CTA at every size measured: 0.164 ms against 0.233 at 4 tiles for 48 MiB, 0.501
against 0.585 for 240 MiB. Allreduce wants the *opposite* at small sizes — 0.282 ms at 4
tiles against 0.361 at 32 for 48 MiB — then agrees at large ones, 0.944 at 32 against 1.086
at 4 for 240 MiB. Reduce-scatter behaves like allreduce at 48 MiB (0.138 ms at the
size-scaled value against 0.194 at 32) and like allgather at 120 and 240. The explanation in
`internode_2d` is convincing — allgather's intra half is two publishes and no reduce, so it
is switch-bound and wants few fat CTAs, while the reduce-carrying collectives are
occupancy-bound once the shard is small — but it explains a *shape*, not a formula, and the
crossover is not derivable from it.

**`--rail-groups` and `--gin-contexts` are one knob, not two.** Allgather is fastest at 2
groups on 4 contexts (0.500 ms) and slower at 8 groups on 1 context (0.561); allreduce
inverts it, 0.944 ms at 8 groups against 1.015 at 2, because it has two fabric hops to
overlap. They are also structurally coupled: a group's grid must be a whole number of chunks
*and* a multiple of the context count, because `wait_signal` divides the grid-wide target by
the granted context count and a narrower grid rounds it down. So "more groups" is often only
purchasable by spending contexts, and that trade is collective-specific.

**Every closed form tried regressed something.** `pick_mc_tiles` scales one tile per 480 KB
of shard; refining it to 240 KB regressed allreduce at 120 MiB, 0.747 ms against 0.521.
`MIN_GROUP_BYTES = 1_500_000` is the same kind of artefact — allreduce at 8 groups is 0.95 ms
on a 15.7 MB shard and 0.61 ms on a 3.1 MB shard where the fabric alone needs 0.13, i.e. half
a millisecond of pure launch overhead. Both are fits to three sizes on one cluster.

**And the noise would let a naive sweep enshrine itself.** Allreduce at 240 MiB, one config,
six runs: 0.944, 0.951, 1.015, 1.028, 1.034, 1.049 ms — about 10% spread. The torch baseline
drifts as much or more over the same window (240 MiB: 0.764, 0.783, 0.993; 120 MiB: 0.559,
0.622, 0.710), because the cluster is shared and other tenants use the same NICs. Several
differences quoted above — 0.944 against 1.015, say — are barely outside that band from a
single pair of runs.

Three consequences, and they are the constraints that matter most:

1. **Repeat and take a robust statistic.** Take the **minimum** across repeats of the
   per-candidate summary, and within one `do_bench` call the **median** of reps rather than
   the mean it currently returns. Minimum across repeats because the noise is one-sided
   contention: nothing makes a collective faster than an uncontended fabric allows, so the
   fastest observation best estimates the config's own cost while a mean mixes in other
   tenants' traffic. Median within a call because one rep can be perturbed by a stray
   barrier. The counter-argument is real — if a config is *intermittently* bad the minimum
   hides it — so the spread must be reported alongside, never discarded.
2. **Re-time the baseline next to each candidate**, not once per sweep. `bench_vs_torch`
   already times torch before *and* after each run for this reason and `report_tuning`
   warns above 15% drift. Compare ratios, not milliseconds: over those six allreduce runs
   the millisecond column moved 11% while the ratio stayed inside 0.96–1.01x.
3. **Treat any decision under ~5% as undecided** and break ties toward the simpler config
   (fewer groups, fewer launches). A tuner that reports a winner without a tie band gives a
   different answer every run, which is worse than a constant.

## 2. Why neither existing tuner transfers

### 2.1 The `--tune` helpers in `internode_common.py`

`tune_grid` / `report_tuning` / `per_launch_signals` are a real in-process sweep and they
got the flat collectives tuned, so they are the right ancestor. But they are wired only
into the three *flat* examples; the 2D examples inherit `--tune`, `--tune-chunks` and
`--tune-contexts` from `add_common_args` and never read them, so those flags are silently
inert on the 2D path today. Lifting them is not just a matter of calling them.

**The grid is the wrong shape.** `tune_grid` knows `chunks`, `gin_contexts`, `threads`. The
2D surface adds `rail_groups`, `mc_tiles`, `mc_threads`, `intra_chunks`, and its single
validity rule (`chunks % contexts == 0`) is one of six.

**The signal budget does not survive.** `tune_grid` gives each candidate virgin signal slots,
which is correct and is the only reason sweeping contexts in one process is safe: signals are
cumulative, nothing resets them, and the device divides the accumulated total by its *current*
context span, so a slot reused under a different span yields a wrong target from then on. But
a 2D candidate needs one signal per rail group, and a composed allreduce one per group in each
half — at `--rail-groups 8` that is 16 of the 32 provisioned signals, so virgin slots allow
**two candidates per process** against a 19 s bootstrap. Either (i) the driver carries a
running total per slot forward and only reuses a slot for candidates with the same
`gin_contexts`, or (ii) `TILESCALE_GIN_SIGNALS` is raised (the override exists in
`nccl_window.py`, default 32). I could not determine whether a larger request is granted:
contexts are documented as a hint and clamped (8 asked, 4 granted) and the device exposes
`probe.nContexts`, but nothing reports a granted *signal* count. Option (i) works today.

**Buffers cannot be reallocated per candidate.** Each collective allocates its arena tensors
in `__init__`, and `BaseAllocator._allocate_tensor_locked` is a bump allocator with no free —
only `close()` releases anything. Worse, the multicast buffer is sized *exactly* at
`Context(mcast_bytes=numel*itemsize)`, so a second `ctx.mcast_tensor()` fails outright. A
driver must allocate once and rebuild only kernels between candidates, which means separating
buffer allocation from kernel construction in `_Base` — small, but required.

**`args` is mutated, and the effective config differs from the requested one.**
`_Base.__init__` does `args.mc_tiles = pick_mc_tiles(...)`, freezing the auto-scaling after
the first candidate, so each candidate needs a copy. And three things rewrite the config
during construction — `workable_chunks()` halves `chunks`, `MIN_GROUP_BYTES` backs
`rail_groups` off, `pick_mc_tiles` fills `mc_tiles` in — so the driver must record what the
object actually built, or the report attributes a time to a config that never ran and two
"different" candidates are silently the same one measured twice.

### 2.2 TileLang's `AutoTuner` (`tilelang/autotuner/tuner.py`)

A well-built single-process tuner in which essentially every mechanism is hostile to a
collective evaluation. It compiles on a `ThreadPoolExecutor` and benchmarks from worker
threads, optionally across several local GPUs, so candidate order is nondeterministic; it
enforces a per-candidate timeout via `SIGALRM` or an async exception injected into the
benchmark thread and on timeout or error **skips that candidate**; it picks a winner from
rank-local latencies; and it caches to disk under a key derived from the function source and
the config list. Each of those is a deadlock at 16 ranks:

- **Every evaluation is collective at three levels.** `ctx.compile` passes
  `compile_once=True, compile_group=ctx.group`, and `_maybe_compile_once` does a
  `dist.all_gather_object` **per compile** — even on a `KernelCache` memory-cache hit.
  `do_bench` barriers before and after and, with `barrier_comm_profiling` on, does a
  `torch.cuda._sleep` plus a `dist.all_reduce` **per rep**. `check` does an `all_reduce`.
  So ranks must agree on the candidate list, the order, how many kernels each candidate
  compiles, and the rep count. A rank that skips, reorders or times out and moves on does
  not lose performance — it desynchronises the collective stream and the job hangs.
- **Rank-local winners are incoherent, not merely suboptimal.** A config is a property of
  the group: `chunks`, `gin_contexts` and `rail_groups` all appear in the sender's grid
  *and* the receiver's wait target. Two ranks on different configs wait for signal totals
  nobody will post.
- **Timeouts cannot be per-rank.** An async exception in one rank's benchmark thread leaves
  15 peers in a barrier. The only safe deadline is an agreed one: decided up front,
  uniformly, aborting the sweep rather than a candidate.
- **Its cache key is wrong here** — function source and config list, not world size, node
  shape or fabric. See §3(c).

Extending it would also put a distributed policy into core TileLang, which the project's
own layering rule keeps out (distributed work lives in `tilelang/distributed/` and the
examples tree so the tree stays rebaseable on upstream).

One correction to `workable_chunks`'s comment, which overstates the symmetry of the compile
hazard. In `_maybe_compile_once` the compile root catches its own exception and ships the
traceback through the `all_gather_object`, so every rank raises a `RuntimeError` carrying
it — a lowering failure **on the root** fails the job cleanly rather than hanging. The
dangerous case is asymmetric: a rank raising *outside* that window (a non-root rank's own
`cached()` call after the gather, an `expect=` assertion, divergent control flow before it)
leaves peers blocked. Not academic here — `run_internode.sh` deliberately uses a different
interpreter and NCCL directory on <node> than on the other nodes, so "the same source lowers
the same way on every rank" is an assumption about two toolchains. Hence the rule: **probe
locally with a plain `tilelang.compile`, agree on the verdict collectively, then compile.**

### 2.3 Configs that are silently wrong, not slow

Why every candidate needs verifying rather than just timing. All these failure modes are
"fast and plausible":

- A rail grid narrower than the granted context count rounds the wait target down — to zero
  in the worst case — and the wait becomes a no-op. This is the mechanism behind the
  historical 332 GB/s reading that exceeded the PCIe egress ceiling.
- Overlapping signal *ranges* (not merely reusing an id) let one collective's wait be
  satisfied by another's arrivals; this corrupted exactly the second half of the allreduce
  output when the allgather started at `SIGNAL_PHASE2` instead of
  `SIGNAL_DATA + rs.signals_used`.
- A reused slot whose device total is stale while the host's running total restarts at zero
  satisfies the first wait instantly — exactly the hazard slot reuse (§2.1) introduces if
  the totals are not carried.
- Two grid computations truncate rather than raise. `mc_bcast_kernel` uses
  `ctas = (span // block_N) // tiles_per_cta` with `span = shard/groups`, while `_Base`
  only checks `shard_numel % (2*mc_threads*mc_tiles)` — the *unsliced* shard — so with
  `rail_groups > 1` a candidate can publish less than its span, or get `ctas == 0` and
  publish nothing. Similarly `rs_sum_kernel`'s chunk width is
  `span // (intra_chunks // groups)`, which drops a tail unless `intra_chunks` divides the
  shard and `groups` divides `intra_chunks`; neither is checked on the multimem path. Both
  become pruning rules below, and arguably should also become assertions.

A check is cheap next to a timing run — one launch, one torch reference, one `all_reduce` —
so there is no reason to make it optional.

### 2.4 Bootstrap forces one process for the whole sweep

A rank costs ~19 s to bring up: 5.7 s `init_dist`, 12 s allocator of which 6.4 s is
`ncclDevCommCreate`. That 6.4 s is flat against the requested resource counts (6.53 / 6.90 /
6.36 s at 8 / 4 / 1 contexts), so it is DOCA/IBGDA setup inside NCCL and not reducible from
here. Fork-per-candidate pays it per candidate; in-process pays it once. That settles the
shape: **one process lifetime, many candidates, all ranks in lockstep.** Two corollaries:
`TL_PG_TIMEOUT_SEC` (180 in `run_internode.sh`) and `RANK_TIMEOUT` (900) both need raising,
since 15 ranks sit inside a collective while one compiles; and the JIT cache must be warm,
since 22 kernels cost 1.5 s as cache hits (~0.08 s each) and "minutes" cold, and one cold
compile past the group timeout aborts the run.

## 3. Design options

**(a) Extend `tilelang.autotuner` with a collective evaluation hook** — deterministic
iteration, a group barrier around evaluation, rank 0 decides and broadcasts; in exchange the
sweep inherits progress bars, the result dataclass and the disk cache. But the pieces that
must be disabled (thread pool, multi-GPU workers, per-candidate timeout, skip-on-error,
rank-local winner, cache key) are most of the class, and what survives is a loop and a table.
Worth revisiting *after* (b) exists and the policy has stopped moving, as an upstreamable
"collective evaluation" strategy.

**(b) A lockstep sweep driver in `internode_common.py`** — a `sweep()` that all ranks call
and walk identically. Rank 0 builds the candidate list and broadcasts it with
`dist.broadcast_object_list`; every rank then iterates it in the same order. Per candidate:
probe validity locally with plain `tilelang.compile`, `all_reduce` the verdict so the skip
is unanimous, build kernels against pre-allocated buffers, verify with `check()`, time with
`do_bench` sandwiched between two baseline timings. Rank 0 accumulates the table and
broadcasts the winner. This is the natural extension of what already works, every
constraint in §2 maps onto an explicit line of it rather than a disabled feature of
something else, and it can be debugged under `run_2d_proxy.sh` — useless for timings, but
it catches every correctness and lockstep bug a real two-node run can.

**(c) An offline cache consulted at startup** — persist the winner as JSON keyed by problem
and topology, and have `_Base.__init__` consult it instead of
`pick_mc_tiles`/`MIN_GROUP_BYTES` when an entry matches. This is the actual goal; (b) is the
machine that fills it. The key must carry
`(collective, algo, intra mode, world_size, local_world_size, num_nodes, dtype, numel)` plus
a fabric fingerprint — the granted GIN context count (4 here, not the 8 requested), the NCCL
version, and whether multicast came up on fabric or POSIX-FD handles. Node identity matters
too, since a NIC shared with another tenant changes not just the numbers but which knob
appears to matter; I would store hostnames as metadata and warn on them rather than key on
them, since keying makes every entry single-use. Staleness must be detectable, and the repo
documents why to worry: `KernelCache._generate_key` does not hash
`src/tl_templates/cuda/distributed/*.h`, so editing a device template leaves cached binaries
stale under an identical key. A tuning cache inherits that and adds one — a config that won
against one `internode_2d.py` need not win against the next. So an entry should carry the
tilelang version, a hash of `internode_2d.py`, a hash of the device templates, the date, and
the ms *and* baseline ms it won with: hash mismatch is a hard invalidation, and a large
divergence between the recorded ratio and the observed one is a warning that the fit is stale.

**Recommendation: build (b), then (c) on top of it, and leave (a) for later.** In order:
separate buffer allocation from kernel construction in `_Base`; add driver-owned per-slot
signal running totals; write `sweep()` next to `tune_grid` and wire `--tune` into the three
2D examples; debug under `run_2d_proxy.sh`; then add the JSON cache and the `_Base` lookup.
The first four are the day of work; (c) is a few hours more and is where the value lands,
because it is what removes the constants from the source.

## 4. Practicalities

**Search space.** `--chunks` over {4, 8} (16 and above hit the put-size lowering bug on the
2D path, and `workable_chunks` halves down anyway). `--gin-contexts` over {1, 2, 4}, since
4 is what the devcomm grants. `--rail-groups` over {1, 2, 4, 8}. `--mc-threads` ×
`--mc-tiles` over {256, 512} × {4, 8, 16, 32} **as a pair, never as two independent axes**:
the climb is non-monotonic at low thread counts (at 256 threads, 8 tiles gives 337 GB/s and
16 gives 317, while 512/32 gives 397). `--intra-chunks` over {512, 1024, 2048} — it looks
pull-path-only but `ReduceScatter2D` and `Allreduce2D` use it for their sum-kernel grids on
the multimem path too. `--no-overlap` is a diagnostic, not a candidate.

**Pruned by construction, before any compile:** `chunks % groups == 0` and
`(chunks // groups) % gin_contexts == 0`; `shard_numel % (2*mc_threads*mc_tiles*groups) == 0`
with `(span // block_N) // mc_tiles >= 1` (the span-level rules from §2.3, stronger than
what `_Base` checks); `shard_numel % intra_chunks == 0` and `intra_chunks % groups == 0`.
The put size must lower, which can only be probed, not predicted — the bad set is not
contiguous (30720 lowers, 32768–61440 fail, 65536 and 81920 lower, 98304 fails) — and the
verdict depends only on `(shard_numel, chunks)`, so it is shared across the other knobs and
worth caching in-process. `MIN_GROUP_BYTES` should be a *default, not a filter*: it is one
of the fits the tuner exists to replace, so a sweep that enforces it can never discover it
is wrong. Let small slices in and let the measurement reject them.

**Sweep cost.** At 240 MiB bf16 on 16 ranks (shard 7 864 320 elements) the divisibility
rules leave 15 valid `(chunks, groups, contexts)` triples, all 8 `(mc_threads, mc_tiles)`
pairs and all 3 `intra_chunks` values — 360 combinations, too many for a repeated protocol.
So stage it: the 15 fabric triples at default `mc_*`, then the 8 multicast pairs at the
winning triple, then the 3 `intra_chunks`, then re-confirm the top few of stage 1 at the
winning `mc_*`. About 26 candidates plus a confirmation pass. This is coordinate descent and
can miss a genuine interaction; the confirmation pass is the cheap partial defence, and the
full 360 stays available for an occasional overnight run. Per candidate, warm cache: the
allgather half compiles roughly `4G+1` kernels (9 at 2 groups, 33 at 8) and the
reduce-scatter half `3G+2` (8 at 2, 26 at 8), so a composed allreduce is up to ~59 — at
~0.08 s per cache hit, 0.7–4.7 s of compile, each still paying one `all_gather_object`.
Timing is three `do_bench` calls (baseline, candidate, baseline) at warmup 20 / rep 50, and
each rep carries a `torch.cuda._sleep(2e7)` cycles, a 256 MB L2 flush and a barrier
`all_reduce`, so a call is dominated by its own overhead rather than the ~1 ms collective —
order half a second to a second each. Call it 5–10 s per candidate: the staged sweep is
3–5 min per repeat, 10–15 at three repeats, plus the 19 s bootstrap; three collectives at
three sizes is an hour or two, a held-nodes window rather than a research project. The cost
is dominated by measurement repetition rather than compilation, which is where it belongs.

**Measurement protocol.** Per candidate: verify, time the baseline, time the candidate, time
the baseline again; report the ratio against `min(pre, post)` and the drift between them.
Repeat the whole candidate loop *R* times (default 3) as an outer pass rather than repeating
each candidate three times in a row, so a bad ten minutes on the fabric penalises all
candidates rather than whichever three landed in it. Summarise by the minimum of the
per-pass ratios and report the spread. Discard the *pass*, not the candidate, if its two
baseline readings disagree by more than ~15% (`report_tuning` already warns there). Never
compare across passes in milliseconds.

**Reporting.** One row per candidate: the **effective** config (post `workable_chunks` /
`MIN_GROUP_BYTES` / `pick_mc_tiles`), PASS/FAIL, per-pass ratios, summary ratio, spread,
baseline drift. Then the winner *and every candidate within 5% of it*, so a human sees the
plateau rather than a single number, plus the count of pruned combinations with the reason
for each. Print the proposed cache entry verbatim so it can be diffed against the tree — a
sweep that recommends the current defaults is a useful result and should be legible as one.

## 5. What I could not determine from the code

- Whether `TILESCALE_GIN_SIGNALS` above 32 is granted. Contexts are a hint and get clamped;
  nothing reports a granted signal count. Needs a probe run with `TL_GIN_DEBUG=1`.
- Whether the two per-node toolchains ever disagree about which put sizes lower. Collective
  agreement on the verdict makes it safe either way, but it would be worth knowing whether
  the `all_reduce` ever actually vetoes anything.
- How the answer moves with node count. Everything above is 2×8. `--algo merged` measures
  1.296 ms against the composed path's 1.086 at two nodes and is expected to win at more,
  so the algorithm itself belongs in the search space once a third node exists — and the
  key's `num_nodes` field is load-bearing, not decorative.
- Whether the `mc_bcast` / `rs_sum` truncation cases in §2.3 are reachable at the sizes
  actually in use, or only at ones the current defaults never produce. Either way a tuner
  will reach them, since it explores exactly the corners the defaults avoid.
