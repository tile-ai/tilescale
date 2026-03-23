# Skill: Sync Upstream TileLang into TileScale

## Overview

TileScale (`tilescale`) is a distributed-computing DSL forked from [TileLang](https://github.com/tile-ai/tilelang).
TileScale extends TileLang with multi-device primitives (NVSHMEM, IPC-based communication, distributed operators) while keeping the entire upstream TileLang core as its foundation.

Periodically, upstream TileLang ships new language features, bug fixes, hardware support, and compiler improvements that TileScale must absorb.
This document is the authoritative guide for an LLM agent performing that sync.

---

## 1. Repository Topology

| Repo | URL | Branch |
|------|-----|--------|
| TileScale (fork) | `https://github.com/tile-ai/tilescale` | `main` |
| TileLang (upstream) | `https://github.com/tile-ai/tilelang` | `main` |

The Python package in both repos is named `tilelang`; the fork adds `tilescale_ext` (a C extension) and `tilelang/distributed/` on top.

---

## 2. TileScale-Exclusive Files — Never Overwrite

The following paths exist **only in TileScale** or carry TileScale-specific logic. They must **never** be blindly overwritten with upstream content. When conflicts arise in these paths, always preserve the TileScale version and integrate upstream changes surgically (see Section 5).

### 2.1 Distributed Runtime

```
tilelang/distributed/                  # IPC tensor allocation, pynvshmem binding,
                                       # distributed utils, launch scripts
tilelang/distributed/pynvshmem/        # NVSHMEM Python bindings (independent CMake/setup)
tilelang/distributed/testing/          # distributed-specific unit tests
```

### 2.2 Distributed Language Primitives (IR Level)

```
tilelang/language/distributed/                    # put/get warp/block, signal primitives
tilelang/language/distributed/common.py           # rank/pe queries, wait_eq, barrier
tilelang/language/distributed/multi_device/       # NVSHMEM and CPEngine IR-level ops
```

### 2.3 TileScale C++ Transforms

```
src/transform/                         # TileScale-specific pass infrastructure
src/transform/common/                  # shared headers used by TileScale transforms
src/tl_templates/                      # CUDA/HIP/CPU template headers extended by TileScale
```

### 2.4 Distributed Examples and Benchmarks

```
examples/distributed/                  # all multi-device examples
benchmark/distributed/                 # distributed performance benchmarks
```

### 2.5 Project-Level Identity Files

```
README.md                              # TileScale branding; do NOT replace with TileLang's
docs/index.md                          # TileScale docs homepage
images/                                # TileScale architecture diagrams
```

---

## 3. What to Merge — Decision Table

### 3.1 Always Merge

| Category | Typical File Paths | Reason |
|----------|--------------------|--------|
| TVM submodule bump | `3rdparty/tvm` | TileScale builds on TVM; staying current is required |
| Core language new features | `tilelang/language/*.py` (non-distributed) | New intrinsics, operators, type annotations used by upper-layer code |
| Compiler / code-generation fixes | `src/` (non-`transform/`), `tilelang/jit/`, `tilelang/carver/` | Correctness and performance of the compilation pipeline |
| Bug fixes in shared modules | Any file **not** in Section 2 | Stability |
| Autotuner improvements | `tilelang/autotuner/` | Directly used by TileScale kernels |
| Build system changes | `CMakeLists.txt`, `cmake/`, `pyproject.toml`, `setup.py` | Must track upstream to build correctly; see Section 5.3 |
| Shared tests | `testing/python/` (non-distributed subdirs) | Validate that merged code works |
| Hardware support (new GPU arch, new dtype) | `src/`, `tilelang/intrinsics/`, `tilelang/math/` | TileScale kernels run on the same hardware |
| Dependency version bumps | `requirements*.txt`, `pyproject.toml` | Avoid dependency drift |
| Critical documentation for shared API | `docs/` pages describing public API behavior | Users of TileScale also read these |

### 3.2 Merge with Adaptation Required

These files exist in both TileLang and TileScale with divergent content. Take upstream changes but **do not drop TileScale additions**.

| File | What upstream may change | What TileScale adds | Merge strategy |
|------|--------------------------|---------------------|----------------|
| `tilelang/__init__.py` | Version string, new re-exports | `tilescale` version, `tilelang.distributed` imports | Keep TileScale version logic; merge new upstream re-exports |
| `tilelang/language/__init__.py` | New public symbols | Distributed primitive re-exports | Append upstream additions; keep distributed re-exports |
| `CMakeLists.txt` | New source files, new targets, dependency changes | `tilescale_ext` target, distributed build targets | Integrate upstream additions into TileScale's CMake; never delete `tilescale_ext` targets |
| `pyproject.toml` / `setup.py` | New dependencies, build wheel config | `tilescale_ext` C extension, extra distributed deps | Merge deps lists; keep TileScale extension definitions |
| `tilelang/jit/adapter/*.py` | Execution backend changes | May reference TileScale-extended kernels | Merge carefully; test distributed kernel compilation after merge |
| `testing/python/` subdirs | New shared test cases | TileScale may have patched existing tests | Merge new tests; investigate any conflict in existing tests before overwriting |

### 3.3 Skip — Do Not Merge

| Category | Typical Paths | Reason |
|----------|---------------|--------|
| TileLang homepage / landing page | `README.md`, `docs/index.md` | TileScale has its own branding |
| CI/CD workflows | `.github/workflows/` | TileScale maintains independent CI that tests distributed features |
| TileLang release artifacts | `VERSION`, `CHANGELOG*`, `docs/release_notes/` | TileScale manages its own versioning |
| Marketing images and logos | `images/` (root), `docs/_static/img/logo*` | Different project identity |
| TileLang-only examples | `examples/` that don't exercise shared APIs | Clutters TileScale's example tree; import selectively only if they demonstrate an API that TileScale also uses |
| Internal TileLang CI helpers | `.pre-commit-config.yaml`, `.clang-tidy` | May conflict with TileScale's linting setup; evaluate case by case |
| TileLang docs pages specific to TileLang tooling | e.g., `docs/get_started/Installation.md` references TileLang pip install | Evaluate; do not overwrite TileScale-specific install instructions |

---

## 4. Commit Classification Heuristic

When iterating over upstream commits to decide what to cherry-pick:

1. **Commit touches only Section 2 paths** → Skip entirely.
2. **Commit touches Section 3.1 paths only** → Cherry-pick as-is.
3. **Commit touches Section 3.2 paths** → Cherry-pick, then open a conflict-resolution step.
4. **Commit touches Section 3.3 paths only** → Skip entirely.
5. **Commit is a mix** → Cherry-pick, then revert changes to Section 2 and 3.3 paths, keep 3.1 and 3.2.
6. **Commit updates `3rdparty/tvm`** → Always merge; run full build + test after.

### Keyword signals in commit messages that indicate "merge":

- `[Feat]`, `[Feature]` — new language primitive, new hardware support
- `[BugFix]`, `[Fix]` — correctness fix in shared code
- `[Enhancement]` — performance or usability improvement
- `[Refactor]` — internal cleanup (check for API-breaking changes)
- `[FFI]`, `[TVM]`, `[Build]` — infrastructure updates
- `[Dependency]` — library version changes

### Keyword signals that indicate "skip or evaluate":

- `[Docs]` — evaluate whether the doc covers a shared API or is TileLang-specific
- `[CI]` — almost always skip
- `[Release]` — skip version bumps; evaluate any associated code changes
- `[AMD]` / `[ROCm]` — merge if it's a hardware backend fix in shared code; skip if it only changes CI docker files

---

## 5. Step-by-Step Merge Procedure

### 5.1 Setup

```bash
# Ensure upstream remote exists
git remote add tilelang https://github.com/tile-ai/tilelang.git 2>/dev/null || true
git fetch tilelang

# Identify the last sync point (look for the previous sync commit message or tag)
# Example: the last sync was tilelang commit <LAST_SYNC_SHA>
LAST_SYNC_SHA=<SHA of last upstream commit included in TileScale>
NEW_UPSTREAM_SHA=tilelang/main
```

### 5.2 Generate Candidate Commit List

```bash
# List upstream commits since last sync, oldest first
git log --oneline --no-merges ${LAST_SYNC_SHA}..${NEW_UPSTREAM_SHA} --reverse \
    > /tmp/upstream_candidates.txt
```

For each commit, apply the classification heuristic in Section 4.

### 5.3 Cherry-pick Workflow

```bash
# For each classified commit SHA:
git cherry-pick --no-commit <SHA>

# Inspect what changed
git diff --cached --name-only

# For any file in Section 2 (TileScale-exclusive), revert it:
git checkout HEAD -- tilelang/distributed/
git checkout HEAD -- tilelang/language/distributed/
git checkout HEAD -- src/transform/
git checkout HEAD -- examples/distributed/
git checkout HEAD -- benchmark/distributed/
git checkout HEAD -- README.md
git checkout HEAD -- docs/index.md
git checkout HEAD -- images/

# For Section 3.2 files, resolve conflicts manually (see 5.4)

# Commit
git commit -m "[Sync] <upstream commit title> (tilelang/<short-sha>)"
```

### 5.4 Handling CMakeLists.txt Conflicts

`CMakeLists.txt` is the highest-risk merge target because both upstream and TileScale add new source files and targets independently.

1. Take upstream's new `add_library` / `target_sources` additions.
2. **Never remove** the `tilescale_ext` target or any `pynvshmem`-related build targets.
3. If upstream changes the TVM subproject loading (`cmake/load_tvm.cmake`), apply it, then verify TileScale's custom TVM patches still apply cleanly.
4. After any `CMakeLists.txt` merge, run a full CMake configure + build before committing.

### 5.5 Handling `tilelang/__init__.py` and `tilelang/language/__init__.py`

TileLang may add new top-level re-exports. The TileScale versions additionally export:
- `tilelang.distributed` namespace
- `tilelang.language.distributed` primitives

Merge strategy:
1. Identify new symbol additions from upstream diff.
2. Insert them in TileScale's file at the same logical position.
3. Do not touch existing TileScale-specific import blocks.
4. Run `python -c "import tilelang; import tilelang.distributed"` to verify both namespaces load.

### 5.6 TVM Submodule Update

```bash
# Update submodule pointer
cd 3rdparty/tvm
git fetch origin
git checkout <upstream-tvm-sha>
cd ../..
git add 3rdparty/tvm

# Rebuild and run tests
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
python -m pytest testing/python/ -x -q
```

TVM submodule updates are the most impactful change. Always do a full build and run both shared tests and at least one distributed example after a TVM bump.

---

## 6. Post-Merge Validation

After each non-trivial sync PR, the following must pass:

### 6.1 Build

```bash
cmake -B build && cmake --build build -j$(nproc)
# Verify tilescale_ext also builds:
python -c "import tilescale_ext"
```

### 6.2 Shared Unit Tests

```bash
python -m pytest testing/python/ -x -q --ignore=testing/python/distributed
```

### 6.3 Distributed Smoke Test

Run at least one intra-node distributed example to verify the distributed layer was not broken:

```bash
torchrun --nproc_per_node=2 examples/distributed/example_allgather_gemm_overlapped.py
```

Or an NVSHMEM example if the environment supports it:

```bash
examples/distributed/nvshmem/example_allgather.py
```

### 6.4 API Compatibility Check

If upstream changed any of the following, check all callers in TileScale's codebase and update them:

- `tilelang.language` public functions
- `tilelang.jit.compile` / `tilelang.jit.kernel` signatures
- `tilelang.autotuner` API
- TVM FFI registration names (search for `tl.` prefixed op names)

```bash
# Find all usages of changed APIs in TileScale-specific code
grep -r "tl\.\(get_rank\|get_num_ranks\|put\|get\|BarrierAll\|SyncAll\)" \
    tilelang/language/distributed/ tilelang/distributed/ examples/distributed/
```

---

## 7. Adaptation Rules: When Upstream Breaks TileScale Layers

### 7.1 TVM FFI Op Registration Name Changes

TileScale distributed primitives are registered as TVM intrinsics under `tl.*` names (e.g., `tl.get_rank`, `tl.BarrierAll`, `tl.tileop.put`). If upstream refactors op registration in a way that alters naming conventions, TileScale's `tilelang/language/distributed/` must be updated to match the new registration approach.

### 7.2 JIT / Compilation Pipeline Changes

If upstream restructures `tilelang/jit/` (e.g., new execution backend, new `KernelCache` interface), TileScale's distributed kernel compilation path must be verified. The distributed layer invokes the same JIT pipeline; any interface change propagates.

### 7.3 `T.` Namespace Language API Changes

TileScale's distributed primitives call into TVM TIR builders. If upstream changes how TIR builder helpers work (e.g., new `tir.call_intrin` signatures, new `ForFrame` step parameter), update `tilelang/language/distributed/common.py` and `tilelang/language/distributed/multi_device/` accordingly.

### 7.4 C++ Header Reorganization

If upstream moves or renames headers in `src/` that TileScale's `src/transform/` includes, update the include paths in TileScale's transform headers. Key shared headers used by TileScale transforms:

```
src/transform/common/attr.h
src/transform/common/collector.h
src/transform/common/loop_fusion_utils.h
src/transform/common/loop_vectorization_utils.h
src/tl_templates/cuda/gemm_sm70.h
src/tl_templates/hip/
src/tl_templates/cpp/
```

### 7.5 dtype System Changes

Upstream periodically adds new data types (e.g., `float8_e4m3fn`, `float4`). TileScale's distributed kernels may pass these types over the network. After a dtype addition upstream:

1. Verify new types are handled in `tilelang/distributed/utils.py` (tensor creation).
2. Verify `tilescale_ext` C extension's tensor allocation handles new dtypes.
3. Add a smoke test in `tilelang/distributed/testing/` if needed.

---

## 8. Reference: PR #50 as a Canonical Example

PR [#50](https://github.com/tile-ai/tilescale/pull/50) ("Sync mainstream TileLang TVM-FFI features into TileScale") is the canonical example of a large upstream sync. Key decisions made in that PR:

| Upstream feature | Action taken |
|------------------|--------------|
| TVM-FFI as default execution backend (`[FFI]` commits) | Merged fully; updated `tilelang/jit/` and build system |
| Warp reduce operators | Merged into shared language layer |
| `T.view` / `T.reshape` shape checks | Merged |
| `T.Ref` annotation, `T.Var` reference passing | Merged |
| Dynamic shared memory size | Merged |
| CUDA read-only param annotation (`__restrict__` / `const`) | Merged; benefits both single-device and distributed kernels |
| Layout inference improvements | Merged |
| TVM submodule bump (multiple commits) | Merged; full rebuild required |
| `T.print` improvements, `T.assume` | Merged |
| `CMakeLists.txt` restructuring | Merged with conflict resolution; `tilescale_ext` preserved |
| Upstream version bump to `0.1.7` | **Skipped**; TileScale manages its own version |
| `.github/workflows/ci.yml` changes | **Skipped** |
| `README.md` rebranding | **Skipped**; TileScale README preserved |
| `docs/index.md` changes | **Skipped**; TileScale homepage preserved |
| TileLang-only installation docs | **Selectively skipped**; shared API docs merged |
| AMD CI Docker changes | **Skipped** (CI-only) |
| New non-distributed examples (GQA, FlashAttn, etc.) | **Selectively merged** when they demonstrated shared API usage patterns useful to TileScale users |

---

## 9. Checklist for Each Sync PR

Before opening the PR:

- [ ] All cherry-picked commits classified and documented in PR description
- [ ] Section 2 files verified not overwritten (run `git diff main -- tilelang/distributed/ src/transform/`)
- [ ] `CMakeLists.txt` conflict resolved; `tilescale_ext` target intact
- [ ] `tilelang/__init__.py` still exports distributed namespace
- [ ] Full build passes
- [ ] Shared `testing/python/` tests pass
- [ ] At least one distributed example runs end-to-end
- [ ] API-breaking upstream changes reflected in TileScale distributed layer if applicable
- [ ] PR title follows: `[Sync] Merge upstream TileLang <date or version range>`
- [ ] PR description lists: last-synced upstream SHA, new upstream SHA, major features included, any skipped items with justification

