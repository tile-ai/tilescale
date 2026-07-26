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
  used to build the native libraries.

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
| Fence-proxy transform regressions | 18 passed with a fresh disabled cache | `/tmp/tl-release-fence-proxy-final.xml`, `6daef2294d769e49f86e9ec87ef7fc120af88829d9ee9c1324c9e5b02eeac4a7` |
| Two-GPU distributed suite, forced CUDA IPC | 97 passed, 5 fabric/multicast capability skips | `/tmp/tilescale-distributed-final2-0726.xml`, `7ee9a2b26ae357eead449141510a048e8cb38fe3e1f95b82996d947c104ce42a` |
| Eight-GPU executable examples, forced CUDA IPC | 3 passed on all 8 ranks | `/tmp/tilescale-8gpu-final2-0726.2AFSaz/`; per-case logs below |
| Eight-GPU multicast-dependent example gates, before IMEX configuration | 3 IMEX/multicast capability skips | `/tmp/tilescale-8gpu-capability-final-0726.xml`, `c059d9b4bc495c13568c010d6356bf98dba026ffd283d62ff2f77b01d4c078d1` |
| Two-GPU distributed suite, automatic VMM selection after IMEX configuration | 102 passed | `/tmp/tilescale-v0-release-2gpu-vmm-final.xml`, `30e30a054cb8208f8460c5a668c13d3037299099b423f1f96d071735f3de4c71` |
| Eight-GPU VMM/multicast executable examples after IMEX configuration | 4 passed on all 8 ranks | `/tmp/tilescale-v0-release-8gpu-final.xml`, `57802a4abb76322442dab347b30fb48fb3c2307e00bd2f2a75b8ebe518a0fe34` |
| Rank-1 descriptor TMA store and neighboring TMA-copy regressions | 12 passed | `/tmp/tilescale-v0-release-tma-copy-final.xml`, `87d4e871afdb76316976f9f03eca36a55ff503088498113d403efc09cd134bc1` |

The forced-IPC two-GPU record contains 17 real two-rank distributed test nodes
and 34 workers that exited cleanly. It covers peer put/get, remote copy, remote
load/store and TMA, signals and barriers, peer tensor views, IPC fallback, and
compile-once coordination, including collective allocator fault recovery. The
post-IMEX VMM record adds the VMM, multicast allocator, and multimem nodes.

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
distributed suite and all four named eight-GPU capability examples. These
follow-up results validate the paths on the stated B200/CUDA 13.2/IMEX
configuration; they do not remove the runtime capability gates or imply support
on systems without an accessible IMEX channel.

## Publication Gates

### Public Git History

The development ancestry contains a deleted proprietary CUDA header. Removing
that file from the current tree does not remove it from Git history. Public
publication must therefore push only the parentless snapshot branch
`v0-release-0726-public`. Do not push `v0-release-0726`, its ancestry, or the
repository's existing tags to the public remote.

Before pushing, verify that the public branch contains exactly one commit and
that its commit has no parent:

```bash
git rev-list --count v0-release-0726-public
git show -s --format='%H %P' v0-release-0726-public
git push <public-remote> v0-release-0726-public
```

The first command must print `1`; the second must print only the snapshot
commit hash with no parent hash.

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
- [ ] Parentless `v0-release-0726-public` snapshot commit
- [ ] Final wheel filenames and SHA-256 values
- [ ] Final source-distribution filename and SHA-256 value
- [ ] Archived validation-record location
- [ ] Rights-holder and legal confirmation reference
- [ ] PyPI Trusted Publisher verification result
