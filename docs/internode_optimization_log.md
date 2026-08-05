# Inter-node kernel optimisation log

One section per kernel, worked one at a time. Each records what was tried, what it measured,
and what was concluded — including the attempts that lost, because those are the expensive
knowledge and the reason not to retry them.

## How to read the numbers

Absolute times on this cluster drift ~8% with other tenants on the same NICs: one allreduce
configuration measured 0.944 / 0.951 / 1.015 / 1.028 / 1.034 / 1.049 ms unchanged, while the
torch baseline over those same runs moved 0.950 → 1.021. **Ratios are the only stable
quantity**, because torch is timed immediately before and after every candidate. Any
difference under ~5% needs repetition before it means anything, and several early
conclusions in this file's history were noise.

Runs only happen when two nodes are *fully* idle; a `PreToolUse` guard enforces that and the
`gpu-window` skill queues work for the next window.

## Status

| # | kernel | vs torch | vs triton-dist | state |
|---|--------|----------|----------------|-------|
| 1 | allgather | 1.49–1.78x | 1.30x @64 MB, 0.93x @32 MB | round 1 done |
| 2 | reduce_scatter | 1.34–1.45x | no comparison yet | not started |
| 3 | allreduce | 1.01–1.12x | no comparison yet | not started |
| 4 | ag_gemm | 1.08–1.34x | their test fails on our launcher | not started |
| 5 | gemm_rs | 1.10–1.24x | their test fails on our launcher | not started |

Baselines are torch NCCL, timed in-run, and triton-dist where its kernel runs inter-node.
Their pull-mode (`dl.symm_at`) kernels are P2P-only by construction and have no inter-node
number; their push kernels do.

## The one idea not yet tried: SM specialisation

Every overlap attempt so far has lost, and always the same way — a host `dist.barrier` costs
30–50 µs and a launch 5–10 µs against phases of 0.1–0.2 ms, so adding either exceeds the
fabric time being hidden:

| attempt | result |
|---|--------|
| merged allreduce, one fabric hop carrying all node partials | 1.296 ms vs composed 1.044 |
| ag_gemm `--mode pipeline`, after the collective got faster | 0.441 vs serial 0.373 |
| gemm_rs `--mode pipeline`, swizzle-adapted remote-first | 0.476 vs serial 0.429 |

The one overlap that *did* pay — rail-group pipelining, 403 → 471 GB/s — adds no barrier at
all, ordering everything with device-side GIN signals.

Triton-distributed's answer is **SM specialisation**: a persistent kernel launched with
`grid = min(NUM_SMS, total_tiles)`, with a `gemm_sm` parameter reserving part of the SMs, so
some CTAs communicate while others compute inside one launch. That avoids both costs at once
— no host barrier, and no extra launch — and it is safe from the wait-inside-a-large-grid
deadlock precisely because the grid is capped at one CTA per SM, making every participant
resident.

This is the missing prerequisite alternative to a device-side barrier (which does not exist
here: `T.barrier_blocks` is lowered with global ranks and takes `get_remote_base_ptr` of each
participant, returning 0 for inter-node peers). Either mechanism would unblock the three
attempts above; SM specialisation needs no new primitive.

Caveat from an earlier attempt: a persistent GEMM capped at 132 CTAs fixed a hang and was
*slower than serial*, so the cap itself costs something. The open question is whether a
proper split — comm CTAs sized to the fabric, compute CTAs taking the rest — recovers more
than the cap costs. That has never been measured.

---

## 1. allgather

`example_internode_allgather_2d.py`, rail-aligned GIN + NVSwitch multicast.

### Where it stands

```
                 32 MB      64 MB     128 MB     240 MB
ours            203.6      316.3      400.4      471.5   GB/s
torch           121.8      188.5      257.7      309.4
triton-dist     234.3      243.3         --         --
```

Beats torch everywhere by 1.5–1.68x, and beats triton-dist from 64 MB up (1.30x). **Loses at
32 MB, 0.87x.** Their curve is nearly flat across 32→64 MB (234 → 243) while ours scales
steeply (204 → 316), which is the signature of a latency-dominated regime: their single-kernel
design pays less fixed cost than our multi-launch one. Crossover is near 48 MB.

### Settled, do not retry

- **The 32 MB config is already optimal among the existing knobs.** `--rail-groups 2` gives
  0.171 ms and `--mc-tiles 4` gives 0.167 against the default's 0.154. An earlier reading of
  158.2 GB/s that suggested a configuration deficit was contention — torch read 86.6 GB/s in
  that same session against 122 on an idle pair.
- `--rail-groups 2` with 4 contexts is the optimum at 240 MB (0.500 ms) and 8 groups on 1
  context is worse (0.561); the reverse holds for allreduce, so the depth is per collective.
- `mc_tiles` must be 32 here at every size measured, unlike the reduce-carrying collectives
  which want it scaled: allgather's intra half is two publishes and no reduce, so it is
  switch-bound rather than occupancy-bound and prefers few fat CTAs (48 MB: 0.164 ms at 32
  against 0.233 at 4).

### Round 1: knob sweep at 32 MB (2026-08-05)

Eleven configurations, torch steady at 121.9-123.1 GB/s throughout, so these are comparable.

| config | GB/s | vs default |
|--------|------|-----------|
| **`--mc-tiles 16`** | **216.9** | **+11.5%** |
| `--mc-tiles 8` | 210.4 | +8.2% |
| `--chunks 4` | 209.4 | +7.7% |
| `--gin-contexts 2` | 208.4 | +7.1% |
| `--mc-threads 256` | 206.6 | +6.2% |
| `--mc-threads 1024` | 202.5 | +4.1% |
| `--gin-contexts 1` | 199.4 | +2.5% |
| default (`--mc-tiles 32`) | 194.5 | -- |
| `--threads 512` | 194.4 | 0% |
| `--mc-tiles 2` | 183.7 | -5.6% |

**The knobs do not compose.** Every combination of the individual winners came out worse than
the best single change:

| combination | GB/s |
|-------------|------|
| `--chunks 4 --gin-contexts 2 --mc-tiles 8` | 206.9 |
| `--chunks 4 --gin-contexts 2 --mc-threads 256` | 194.0 |
| `--chunks 4 --mc-tiles 8` | 185.2 |
| `--chunks 4 --gin-contexts 2 --mc-tiles 8 --mc-threads 256` | 183.6 |

That matters beyond this kernel: it rules out coordinate descent for the autotuner, since each
knob's optimum moves when another changes. The search has to sweep tuples, which is a larger
budget than the design notes assumed.

Result: 194.5 -> **216.9 GB/s at 32 MB, 1.78x torch**, from one flag. Against triton-dist's
234.3 that closes 0.87x -> 0.93x, though their figure is from an earlier session; a
same-session re-run has twice produced no output, its sweep apparently exceeding the run
timeout at 500 iterations.

`mc_tiles` is therefore not "32 everywhere" for allgather as previously concluded -- that was
inferred from 48 MB, where only 4 and 32 had been compared. The default is now a threshold on
shard size, fitted to three points and no finer.

The small-size settings do **not** transfer upward: at 240 MB the combination gives 401.0 GB/s
against the default's 460.3. At 64 MB it gives 327.2 against 316.3, inside noise.

### Open

The remaining 32 MB gap is fixed cost, so the candidates reduce launches rather than bandwidth:

1. Fold `publish_own` and `publish_remote` into one kernel with a slot-indexed grid (-1 launch).
2. Fold the rail put+wait and the publish into one persistent SM-specialised kernel (-2
   launches and the closing barrier).
3. A node-local device barrier, which needs the primitive written first.

A `--phases` run at 32 MB should apportion the fixed cost first -- without it all three are
guesses about where the 0.145 ms goes.
