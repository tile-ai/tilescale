# TileScale 0.0.2 Release Validation

This page records the 2026-07-26 release-candidate handoff and the IMEX-enabled
follow-up completed on 2026-07-27. It describes the source baseline and the
validation completed on the candidate tree; it is not a claim that the release
has been published. The release manager must complete the sign-off record
before publication.

## Baseline and Package Identity

- TileScale 0.0.2 is based on upstream TileLang commit
  `550e25d493a93729cb087e4ecb587c19028d3cea` from 2026-06-04. That baseline
  includes the upstream block-scaled TCGEN5 refactor and `.f8f6f4` and
  `.mxf8f6f4` support.
- The Python distribution name is `tilescale`. The compatibility import
  namespace remains `tilelang`, so existing TileLang imports and kernel source
  do not need a namespace migration.
- Upstream `tilelang` and TileScale must not be installed in the same Python
  environment because both distributions provide the `tilelang` namespace.
- The retained upstream surface includes the Python kernel language, compiler,
  JIT, cache, CUDA/CuTeDSL paths, and the source-level ROCm, CPU, Metal, and
  WebGPU backend structure. The validation table below states exactly which
  paths ran on the release host; inheritance is not a claim that every backend
  ran on B200.

## Functional and Architectural Changes

The distributed extension is intentionally limited to one host with one
process per local GPU. Multi-node execution, NVSHMEM, DeepEP, and a general
distributed runtime are outside the 0.0.2 scope.

- Distributed modules are imported lazily, preserving normal `tilelang`
  compiler and CPU-only imports when optional CUDA host bindings are absent.
- The distributed allocator exchanges peer mappings and exposes zero-copy peer
  tensor views. It selects VMM fabric handles only after a successful runtime
  probe and otherwise falls back to CUDA IPC. `TILESCALE_USE_VMM=0` and
  `TILESCALE_USE_VMM=1` provide explicit diagnostic overrides.
- Rank queries, peer load/store/copy/TMA operations, system-scope signals,
  waits, atomics, barriers, and compile-once multi-process coordination are
  available through the existing `tilelang` language and JIT interfaces.
- Multicast allocation and multimem operations remain experimental and
  capability-gated. They require compatible GPUs, NVSwitch fabric, driver
  support, and an accessible NVIDIA IMEX channel.
- Allocator teardown, imported-handle ownership, process-group validation, rank
  environment handling, launch validation, and shared-memory FFI registration
  were hardened for deterministic multi-process use.
- CUDA 13 and Blackwell fixes cover metadata symbols, TMA alignment, vector
  broadcast/code generation, TCGEN05 fence handling, CuTeDSL/PDL compatibility,
  rank-1 descriptor TMA store layout inference, and reduction tolerances.
  CPU-only configuration and build paths were also repaired without replacing
  the upstream backend layout.
- Packaging now emits the `tilescale` distribution, verifies required license
  and notice files, rejects restricted CuTeDSL/EULA content and generated
  CUTLASS documentation or media that are not build inputs, and smoke-tests the
  installed compatibility namespace and distributed FFI surface. The
  `apache-tvm-ffi` runtime is pinned to `0.1.11`, matching the vendored headers
  used to build the native libraries. `z3-solver` is pinned to `4.15.4.0` for
  both isolated builds and runtime installation so the linked Z3 SONAME remains
  consistent.

## Upgrade

The historical `tilescale==0.0.0.dev1/dev2` placeholder depended on the
separate upstream `tilelang` distribution. A normal in-place upgrade can leave
two conflicting owners of the same import namespace. Use a fresh virtual
environment, or remove both distributions before installing 0.0.2:

```bash
python -m pip uninstall -y tilescale tilelang
python -m pip install "tilescale[distributed]==0.0.2"
python -c "import tilelang; import importlib.metadata as m; print(m.version('tilescale'))"
```

## Validation Host

| Component | Candidate validation environment |
|-----------|----------------------------------|
| GPUs | 8 x NVIDIA B200, compute capability 10.0, 183,359 MiB per GPU |
| CUDA driver | 590.48.01 |
| CUDA toolkit | 13.2, `nvcc` 13.2.78 |
| Interconnect | Full 8-GPU NVSwitch topology reported as NV18 between peers |
| Fabric service | NVIDIA Fabric Manager active |
| IMEX device | Initially absent; `/dev/nvidia-caps-imex-channels/channel0` was configured at 2026-07-26 23:39 local time and made accessible for follow-up validation |

## Test Evidence

The rows below are independent validation slices and may overlap; their counts
must not be added into a synthetic total. Capability skips are reported
separately from passes.

| Validation slice | Result | JUnit record and SHA-256 |
|------------------|--------|--------------------------|
| Broad upstream-compatible group A | 283 passed, 3 hardware skips | `/tmp/tl-release-broad-a.xml`, `93334dc87c6822e2f5f21bdbda1cc691205029b3c5876dde8a8fa98fc641fa83` |
| Broad upstream-compatible group B, isolated from editable installs | 1,072 passed, 19 hardware/backend skips | `/tmp/tilescale-broad-b-clean-final-0726.xml`, `52f8c726e73a527ae849c2f817a85c01f0291cb1ee0a1766db1acde594304f1a` |
| Broad upstream-compatible group C | 149 passed, 125 hardware/backend skips, 1 performance case deselected | `/tmp/tl-release-broad-c.xml`, `34a70f3c63d3da59caa2abe2d53137c41a58a797cae19309c2ccf5bdbbfe475c` |
| Reduction regressions | 53 passed | `/tmp/tl-release-reduce-final.xml`, `0dc8516152930f2ba63bb1ff92bd0e55a7504e89153b3b7a78da304bef419af7` |
| CuTeDSL and PDL focused regressions | 7 passed | `/tmp/tl-release-cutedsl-pdl-final.xml`, `84d647cb34c965807e663ad83088144b8a4be386290342b5aeeb01537ed0087d` |
| Fence-proxy transform and TMA-store regressions | 23 passed with a fresh cache | `/tmp/tilescale-fence-tma-final.xml`, `5a3f7308a6e273221bc66edc325068357f5285890ad82d9b5c7bc2c43b5a1080` |
| Two-GPU distributed suite, forced CUDA IPC | 97 passed, 5 fabric/multicast capability skips | `/tmp/tilescale-distributed-final2-0726.xml`, `7ee9a2b26ae357eead449141510a048e8cb38fe3e1f95b82996d947c104ce42a` |
| Eight-GPU executable examples, forced CUDA IPC | 3 passed on all 8 ranks | `/tmp/tilescale-8gpu-final2-0726.2AFSaz/`; per-case logs below |
| Eight-GPU multicast-dependent example gates, before IMEX configuration | 3 IMEX/multicast capability skips | `/tmp/tilescale-8gpu-capability-final-0726.xml`, `c059d9b4bc495c13568c010d6356bf98dba026ffd283d62ff2f77b01d4c078d1` |
| Two-GPU distributed suite, automatic VMM selection after IMEX configuration | 103 passed | `/tmp/tilescale-vmm-auto-full-final2.xml`, `e9a4adb00d353d4b7e27c56149cb9007f8303a51a667b42454858bf9daca0e6e` |
| Eight-GPU VMM/multicast executable examples after IMEX configuration | 4 passed on all 8 ranks | `/tmp/tilescale-release-8gpu-final.xml`, `57802a4abb76322442dab347b30fb48fb3c2307e00bd2f2a75b8ebe518a0fe34` |
| Two-GPU VMM, multicast, and multimem primitives, forced VMM | 58 passed, 1 mutually exclusive IPC fallback test deselected | `/tmp/tilescale-vmm-explicit-final3.xml`, `18305ec0c51f7111c083d753b7a431a13238b78797d5dccf4b8468fe42225149` |
| Final four-GPU distributed suite on B200 with IMEX configured | 143 passed | `/tmp/tilescale-pr61-distributed-no-cluster-edit-4gpu.xml`, `f3cd1599515d4bf77368fe798831569f22e99d6139bf28a7f47c42e09077080f` |
| Native multimem TMA broadcast and ADD on B200 (SM100), forced VMM | 1 passed with exact checks of both physical backings | `/tmp/tilescale-vmm-multimem-tma-final.xml`, `b68fb4556df7c7c6fc91a5e95bf0d8786d6a43faa835fce4dad3f376ac3c78a0` |
| Rank-1 descriptor TMA store and neighboring TMA-copy regressions | 12 passed | `/tmp/tilescale-release-tma-copy-final.xml`, `87d4e871afdb76316976f9f03eca36a55ff503088498113d403efc09cd134bc1` |

The forced-IPC two-GPU record contains 17 real two-rank distributed test nodes
and 34 workers that exited cleanly. It covers peer put/get, remote copy, remote
load/store and TMA, signals and barriers, peer tensor views, IPC fallback, and
compile-once coordination, including collective allocator fault recovery. The
post-IMEX automatic-VMM record contains 21 real two-rank distributed nodes and
42 workers; it adds the VMM, multicast allocator, and multimem nodes.

The pre-IMEX forced-IPC candidate directory contains these three executable
records. Every example used a separate disabled cache and process-group port.

| Eight-GPU case | Result | Wall time | Log SHA-256 |
|----------------|--------|-----------|-------------|
| `example_allgather_gemm_overlapped.py` | Passed; all 8 ranks reported successful checks | 29.92 s | `e327f7f3d29480eab5dfc0f454a15690a0cf99acfc24a6f56539a140dceb0a99` |
| `example_gemm_rs_overlapped.py` | Passed; all 8 ranks reported successful checks | 33.54 s | `92af3d507fac7aacbf1836f9e581a0124722397f310d65602e9a0d27a9863d31` |
| `example_gemm_rs_specialized.py` | Passed; all 8 ranks reported successful checks; maximum reported difference 0.1875 | 26.43 s | `16790bc94d8707d08d76bec5bd92757416b4f9697461a541a6fe4c69859bb7ea` |

After IMEX configuration, a fresh-cache four-node suite exercised specialized
all-gather GEMM, specialized GEMM all-reduce, multimem one-shot/two-shot
all-reduce, and ordinary TMA store to multicast VA. It completed in 80.73 s.
All eight ranks passed every correctness check. The two-GPU suite selected VMM
without an override and executed its VMM, multicast, and multimem tests.

The focused forced-VMM record isolates the low-level VMM allocation and handle
exchange, multicast allocator, and multimem tests. Its one deselected node is
`test_distributed_ipc_fallback`, which intentionally asserts that VMM is
disabled and is therefore incompatible with `TILESCALE_USE_VMM=1`. The native
multimem TMA node separately exercised both `multimem.cp.async.bulk` broadcast
and `multimem.cp.reduce.async.bulk.add.f32` on two B200 GPUs. Every rank checked
its physical multicast backing exactly, including ADD behavior with nonzero and
zero initial values.

The release manager should archive the `/tmp` JUnit and log records outside the
validation host before signing the release.

## IMEX Validation Follow-up

The first validation stage accurately recorded an environment boundary: the
IMEX device node was absent, direct CUDA driver probes for fabric-handle
`cuMemCreate` and `cuMulticastCreate` returned `CUDA_ERROR_NOT_PERMITTED (800)`,
and multicast-dependent tests skipped instead of executing. Fabric Manager and
full NVSwitch topology alone did not satisfy the runtime requirements.

An administrator subsequently configured an IMEX channel accessible to the
release user. The runtime VMM-fabric and multicast probes then both returned
true. Fresh-cache reruns selected VMM automatically and completed the two-GPU
distributed suite and all four named eight-GPU capability examples. A later
forced-VMM rerun also completed the focused VMM/multicast/multimem suite and the
native multimem TMA broadcast and ADD test on B200 (SM100). These follow-up results
validate the paths on the stated B200/CUDA 13.2/IMEX configuration; they do not
remove the runtime capability gates or imply support on systems without an
accessible IMEX channel.

## Publication Gates

### Public Git History

The development ancestry contains deleted non-redistributable sources. Removing
them from the current tree does not remove them from Git history. Do not push
`v0-release-0726`, its ancestry, development branches, or unreviewed tags to a
public remote.

Two publication topologies are maintained:

- The review and merge path uses
  `release/v0.0.2-tilelang-550e25d`. Its commits were reconstructed directly on
  the existing public `main` baseline; the private development commits are not
  ancestors of the branch, and every reconstructed tree uses the public CUDA
  Toolkit declarations.
- A new repository or standalone source publication should use the parentless
  `v0-release-0726-public` snapshot. It has the same final source tree without
  inheriting any existing repository history.

Before merging the review branch, verify its public base, reconstruction fork,
absence of merge/private/CI ancestry, final-tree equivalence, restricted-path
history, and unchanged CI workflow:

```bash
release_ref=release/v0.0.2-tilelang-550e25d
public_base=4704282a16fd0e7ff2c2c13f87772b42e4dc6163
private_fork=8205791dfb65272b4d5bcb812f88456cf918b895
excluded_ci=2035db5636f1f09476a0311c255ab1955e4ef769
ci_parent=c3fefac1101bc05b786dd4a5784ce313cfbd273b

test "$(git merge-base "$public_base" "$release_ref")" = "$public_base"
test "$(git merge-base v0-release-0726 "$release_ref")" = "$private_fork"
test -z "$(git rev-list --merges "${public_base}..${release_ref}")"
! git merge-base --is-ancestor "$excluded_ci" "$release_ref"
test "$(git rev-parse "${release_ref}^{tree}")" = \
  "$(git rev-parse 'v0-release-0726^{tree}')"
test "$(git rev-parse "${release_ref}^{tree}")" = \
  "$(git rev-parse 'v0-release-0726-public^{tree}')"
test "$(git rev-parse "${release_ref}:.github/workflows/ci.yml")" = \
  "$(git rev-parse "${ci_parent}:.github/workflows/ci.yml")"
test -z "$(git log --format='%H' "$release_ref" -- \
  src/cuda/stubs/vendor/cuda.h examples/mega_moe/reference.py)"
```

For the standalone path, verify that the snapshot contains exactly one commit
and that the commit has no parent:

```bash
git rev-list --count v0-release-0726-public
git show -s --format='%H %P' v0-release-0726-public
git diff --exit-code release/v0.0.2-tilelang-550e25d v0-release-0726-public
```

The first command must print `1`; the second must print only the snapshot
commit hash with no parent hash. A safe release branch does not sanitize other
remote refs. Before declaring an existing repository publication-clean, audit
all advertised branches, tags, and persistent pull-request refs. Use a clean
repository or coordinate sensitive-object purging with the hosting provider if
any unsafe ref has already reached the remote.

### License Confirmation

The root `LICENSE` contains MIT terms and also states that work from December 1,
2024 through March 14, 2025 is subject to additional collaboration terms with
Microsoft Corporation. Package metadata therefore uses the conservative custom
expression `LicenseRef-TileScale-Distribution` rather than claiming that the
aggregate archive is MIT-only. The rights holder and legal reviewer must
confirm that the proposed public snapshot can be released under the declared
terms. The third-party notices, bundled license files, and custom metadata
expression do not replace that confirmation.

### PyPI Trusted Publishing

The repository workflow is prepared for OIDC trusted publishing, but the
external PyPI project configuration has not been verified during this
validation. Before publishing `tilescale==0.0.2`, confirm that the PyPI Trusted
Publisher entry matches the public repository, workflow filename, owner, and
release environment. A successful package build or smoke test is not evidence
that the external OIDC binding is configured.

### Release Manager Sign-Off

These values are intentionally left for the release manager to record at
signing time:

- [ ] Final private release commit
- [ ] Sanitized release-PR head, public base, and final source-tree hash
- [ ] Parentless `v0-release-0726-public` snapshot commit, if using the standalone path
- [ ] Remote branch, tag, and pull-ref audit or clean-repository confirmation
- [ ] Final wheel filenames and SHA-256 values
- [ ] Final source-distribution filename and SHA-256 value
- [ ] Archived validation-record location
- [ ] Rights-holder and legal confirmation reference
- [ ] PyPI Trusted Publisher verification result
