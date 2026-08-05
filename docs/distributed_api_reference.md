# TileScale Distributed API Reference

TileScale extends TileLang with experimental multi-GPU CUDA primitives. The
supported process model is one process per local GPU, using an NCCL process
group for host-side coordination.

Multi-node execution is supported through the NCCL Device API ("GIN",
GPU-Initiated Networking), which exposes one-sided RDMA callable from inside a
kernel. Intra-node peers keep using IPC or VMM peer pointers; inter-node peers
are reached through a registered NCCL window instead. NVSHMEM and a
general-purpose distributed runtime remain outside the current scope.

The Python distribution is `tilescale`, while the import namespace remains
`tilelang`. Install the optional host dependencies with
`pip install "tilescale[distributed]"`.

## Runtime Requirements

The available memory path depends on the host and visible GPUs:

| Path | Requirements | Selection |
|------|--------------|-----------|
| CUDA IPC | Same host, CUDA peer access between participating GPUs | Used when VMM is disabled or unavailable |
| VMM fabric | CUDA driver API 12.4 or newer, fabric-handle support on every visible GPU, and an accessible NVIDIA IMEX channel | Auto-selected for distributed allocators when the runtime probe succeeds |
| Multicast | A VMM allocator plus multicast-capable GPUs, normally an NVSwitch system. Fabric handles are *not* required: without an IMEX channel the multicast object is shared as a POSIX file descriptor | Enabled only when `mcast_size` is explicitly requested |
| GIN (inter-node) | NCCL 2.28.7 or newer with `nccl_device/gin.h` and `ncclDevCommCreate`, a VMM-backed arena, and a working RDMA fabric | Attempted automatically when more than one node is detected; `TILESCALE_USE_GIN=1` makes an unavailable Device API a hard error |

`_supports_vmm_fabric()` and `_supports_multicast()` are runtime probes, not
portable feature guarantees. Both inspect all CUDA-visible devices, so set
`CUDA_VISIBLE_DEVICES` to the exact local rank set before starting processes.

Both probes reach `cuCtxGetDevice` and therefore report `False` with no current
CUDA context. `cuInit` plus a primary-context retain is not sufficient; force a
context first (for example `torch.zeros(1, device="cuda")`) or a capable machine
will be misreported as incapable.

Fabric handles require an NVIDIA IMEX channel that is accessible inside the
host or container. On a compatible Linux driver installation, an administrator
can run:

```bash
tilelang/distributed/scripts/conf_vmm.sh [channel_id]
```

The script requires `sudo`, creates
`/dev/nvidia-caps-imex-channels/channel<id>`, and grants the invoking user
read/write access. It does not install the NVIDIA driver or make unsupported
hardware fabric-capable. For containers, the configured device must also be
passed through to the container.

## Process Group Initialization

```python
from tilelang.distributed.host import init_dist

rank, world_size, group = init_dist(
    local_rank,
    num_local_ranks,
    master_port=None,
)

rank, world_size, group, node_info = init_dist(
    local_rank,
    num_local_ranks,
    return_node_info=True,
)
```

`init_dist` sets `cuda:local_rank`, creates an NCCL process group, and returns
its rank, size, and `dist.group.WORLD`. With `return_node_info=True` it also
returns a `NodeTopology` describing `num_nodes`, `node_rank`, `local_rank` and
`local_world_size`, which the allocator needs to decide whether to set up GIN.
Pass it on as `get_allocator(..., node_info=node_info)`.

Topology is read from torchrun-style `LOCAL_WORLD_SIZE`/`GROUP_RANK` when those
are present, and otherwise from `NNODES`/`NODE_RANK`.

`init_dist` sets `NCCL_IB_DISABLE=1` by default. **Clear it for any inter-node
run**, or NCCL will refuse to use the RDMA fabric that GIN depends on.

The current implementation requires a contiguous rank-to-device mapping:
process rank `i` uses CUDA ordinal `i`, and participating devices must appear
as ordinals `0..N-1` in every process. Restrict `CUDA_VISIBLE_DEVICES` to the
participating GPUs before launching. A non-contiguous subgroup inside a larger
visible device set is not currently supported by multicast VA access setup.

Two launch modes are accepted:

- A local `torch.multiprocessing.spawn`-style launch with no torchrun rank
  variables. The function arguments define the rank and world size.
- Single-node `torchrun`, identified by `LOCAL_WORLD_SIZE`. Its
  `LOCAL_WORLD_SIZE`, `WORLD_SIZE`, `RANK`, and `LOCAL_RANK` values must agree
  with the function arguments.

Multi-node values (`WORLD_SIZE > LOCAL_WORLD_SIZE`, `NNODES > 1`, or a nonzero
node rank) are rejected. A `WORLD_SIZE`/`RANK` environment without
`LOCAL_WORLD_SIZE` is also rejected unless it is the neutral `1`/`0` pair.

### Launch Environment

Defaults below are applied by `init_dist` only when the caller has not already
set the variable.

| Variable | Effective default | Behavior |
|----------|-------------------|----------|
| `MASTER_ADDR` | `127.0.0.1` | TCP rendezvous address |
| `TILESCALE_MASTER_PORT` | unset | Preferred rendezvous port override |
| `MASTER_PORT` | unset | Used when `TILESCALE_MASTER_PORT` is unset |
| rendezvous port | `8361` | Used when neither port variable nor `master_port` argument is provided |
| `NCCL_IB_DISABLE` | `1` | Preserves a caller-provided value |
| `NCCL_DEBUG` | `ERROR` | Preserves a caller-provided value |

Port precedence is: `master_port` argument, `TILESCALE_MASTER_PORT`,
`MASTER_PORT`, then `8361`.

Multi-node launches additionally need `WORLD_SIZE`, `LOCAL_WORLD_SIZE`, and
either `NNODES`/`NODE_RANK` or torchrun's `GROUP_RANK`, with `MASTER_ADDR` set to
an address the other nodes can reach. `NCCL_IB_DISABLE` must be cleared.

Variables read outside `init_dist`:

| Variable | Effective default | Behavior |
|----------|-------------------|----------|
| `TILESCALE_USE_VMM` | unset | `1` forces VMM, any other value forces CUDA IPC |
| `TILESCALE_USE_GIN` | unset | `1` requests arena window registration even on a single node, and makes an unavailable Device API a hard error rather than a warning |
| `TILESCALE_NCCL_LIB` | unset | Path to a GIN-capable `libnccl.so.2`, for when the ambient NCCL predates the Device API |
| `TILESCALE_GIN_CONTEXTS` | `8` | Requested `ginContextCount`. One context is one QP per peer; the count is a hint and may be granted in part |
| `TILESCALE_GIN_SIGNALS` | `32` | Requested `ginSignalCount`, i.e. how many independent signals a kernel may use |
| `TILESCALE_GIN_COUNTERS` | `32` | Requested `ginCounterCount` |

The GIN resource counts affect device memory but not start-up latency:
`ncclDevCommCreate` measures the same at 1, 4 and 8 contexts, because its cost is
transport initialisation rather than per-context setup.

## Distributed Allocator

```python
allocator = tilelang.get_allocator(
    size=2**30,
    device="cuda",
    is_distributed=True,
    local_rank=local_rank,
    num_local_ranks=num_local_ranks,
    group=group,
    use_vmm=None,
    mcast_size=None,
)
```

`BaseAllocator` is a bump allocator over one base allocation per rank, with a
default alignment of 256 bytes. Distributed ranks exchange handles and build a
CPU `uint64` metadata table with this layout:

```text
[local_rank, group_size, rank_0_base_ptr, ..., rank_n_base_ptr]
```

All ranks must perform allocator-backed allocations in the same order and with
matching sizes. Remote addressing uses the local allocation offset with the
selected peer base pointer; it is not a named-object or arbitrary-pointer
lookup service.

### VMM Selection

The selection order is:

1. If `TILESCALE_USE_VMM` is present, exactly `1` selects VMM; any other value
   selects CUDA IPC. This environment override takes precedence over the
   function argument.
2. Otherwise, an explicit `use_vmm=True` or `False` is honored.
3. With `use_vmm=None`, a distributed allocator uses VMM only when
   `_supports_vmm_fabric()` succeeds; otherwise it uses CUDA IPC.

Forcing VMM does not provide an IPC fallback if fabric allocation fails.

GIN requires a VMM allocator: `ncclCommWindowRegister` rejects `cudaMalloc`
memory, so window registration fails with the IPC backend. When fabric handles
are unavailable the arena is still VMM-backed, created with
`CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` and shared between local ranks by
duplicating that descriptor with `pidfd_getfd`. That path needs ptrace-level
access to sibling ranks (true for one job under one user) and a kernel with
`pidfd_getfd`, i.e. Linux 5.6 or newer.

### Multicast Allocation

Passing `mcast_size` requires a distributed VMM allocator and a successful
multicast capability probe. The internal
`allocator._allocate_mcast_tensor(shape, dtype)` method returns
`(multicast_view, local_physical_view)`. The multicast view is used by multimem
instructions; each rank writes its own contribution through the local physical
view.

The multicast buffer is a **separate allocation from the arena**, and only the
arena is registered as an NCCL window. Anything a GIN put reads or writes must
therefore come from `tilelang.tensor(..., allocator=...)`, not from
`_allocate_mcast_tensor`. A kernel that reduces through the switch and then
sends over the fabric needs one buffer of each.

`mcast_size` is the total multicast capacity and is consumed by a bump pointer
with no free, so size it for every multicast tensor the process will allocate.
A caller that allocates two (an allreduce reducing its input and broadcasting
its output, for instance) must request both up front.

Sharing the multicast object across processes uses fabric handles when an IMEX
channel is available and POSIX file descriptors otherwise; `_multicast_uses_fd()`
reports which route the C++ side selected, so the allocator does not re-probe
and risk disagreeing with it.

`allocator.close()` must be called collectively before destroying the process
group, especially for multicast allocations. The allocator is also a context
manager.

### Tensor Allocation

```python
tilelang.tensor(shape, dtype, device=None, allocator=None, return_peers=None)
```

Without an allocator this delegates to `torch.empty`. With an initialized
`BaseAllocator`, it returns a zero-copy tensor over the next allocator region.
`return_peers=True` returns one tensor view per rank; all views are represented
on the current CUDA device because peer mappings live in the current process's
virtual address space.

Ownership transfer is not implemented. Allocator-backed tensors must not
outlive their allocator.

## Kernel Metadata Initialization

```python
kernel.initialize(allocator=allocator, stream=None)
```

This copies the allocator metadata table into the generated module. It must run
before kernels that call rank queries, remap peer addresses, or create remote
TMA descriptors. The table is limited to 1024 `uint64` entries. `stream` is a
raw non-negative CUDA stream handle; `None` uses stream handle `0`.

## Kernel-Side APIs

These APIs are exported through `tilelang.language` and are normally used as
`T.*` inside a `T.prim_func`.

### Rank Queries

| API | Declared return type | Description |
|-----|----------------------|-------------|
| `T.get_rank()` | `int32` | Current local process rank |
| `T.get_num_ranks()` | `int32` | Size of the initialized local process group |

### Peer Memory Operations

The peer operations use CUDA IPC or VMM mappings from the allocator metadata
table. They do not call NVSHMEM.

```python
T.put_warp(src, dst, size, dst_pe=-1, unroll_factor=4,
           enable_aggressive_vectorize=False)
T.get_warp(src, dst, size, src_pe=-1, unroll_factor=4,
           enable_aggressive_vectorize=False)
T.put_block(src, dst, size, dst_pe=-1)
T.get_block(src, dst, size, src_pe=-1)

T.copy(src, dst, src_pe=None, dst_pe=None, ...)
T.tma_copy(src, dst, src_pe=None, dst_pe=None, ...)
```

A PE value of `-1`, or an omitted `src_pe`/`dst_pe`, keeps the local address.
A non-negative PE remaps the corresponding address by allocator-relative
offset. A copy cannot make both sides remote in the same operation. Remote TMA
also requires the normal TMA hardware and lowering constraints.

Remote TMA needs one compile-time descriptor per peer, so the descriptor
dispatch covers PEs `0` through `7`. A remote TMA copy or TMA reduce whose PE
falls outside that range issues no operation, which stalls a load waiting on
its mbarrier. Non-TMA remote paths (`T.ld`, `T.st`, `T.put_*`, `T.get_*`,
non-TMA `T.copy`) have no such limit.

Scoped scalar operations are:

```python
T.ld(src, value, scope="gpu", sem="weak", na=False, nc=False, src_pe=-1)
T.st(dst, value, scope="gpu", sem="weak", na=False, dst_pe=-1)
T.atom_add(target, value, scope="gpu", sem="relaxed")
```

`T.atom_add` is the distributed `uint32` atomic-add intrinsic and supports
`gpu` or `sys` scope with `relaxed`, `acquire`, `release`, or `acq_rel`
semantics.

### Wait and Barrier Operations

```python
T.wait_eq(value, expected, peer=-1, scope="sys", semantics="acquire")
T.wait_ne(value, expected, peer=-1, scope="sys", semantics="acquire")
T.wait_ge(value, expected, peer=-1, scope="sys", semantics="acquire")
T.wait_le(value, expected, peer=-1, scope="sys", semantics="acquire")
T.wait_gt(value, expected, peer=-1, scope="sys", semantics="acquire")
T.wait_lt(value, expected, peer=-1, scope="sys", semantics="acquire")
```

Valid wait scopes are `sys` and `gpu`; valid semantics are `acquire` and
`volatile`. A non-negative `peer` remaps the polled address to that peer.

GPU-local and cross-block barrier helpers include `T.init_barrier_gpu`,
`T.arrive_barrier_gpu`, `T.wait_barrier_gpu`, `T.sync_barrier_gpu`,
`T.barrier_blocks`, and `T.sync_blocks`. Cross-GPU correctness still depends on
using peer-visible storage and the appropriate system-scope ordering.

### GIN Inter-Node Operations

These lower to the NCCL Device API and are the only kernel-side path to a peer
on another node. They live under `T.nccl_gin.*` and require the allocator to
have registered its arena as an NCCL window.

| API | Purpose |
|-----|---------|
| `T.nccl_gin.put(src, dst, size, peer, scope="block")` | One-sided RDMA write into `peer`'s symmetric buffer |
| `T.nccl_gin.put_signal(src, dst, size, peer, signal_id, scope="block")` | Same, plus increment `signal_id` on the destination once the payload has landed |
| `T.nccl_gin.signal(peer, signal_id, scope="block")` | Increment a remote signal with no payload |
| `T.nccl_gin.wait_signal(least, signal_id, scope="block")` | Block until the cumulative count for `signal_id` reaches `least` |
| `T.nccl_gin.flush(scope="block")` | Wait until this rank's source buffers are reusable |

`src` and `dst` are ordinary buffer element references; the lowering converts
them to `(window, offset)` pairs using the symmetric arena, so the offset that
names bytes locally names the same bytes on the peer. `peer` is a **global**
rank.

Three properties of the signal mechanism decide how these are used:

- **Signals are cumulative and a wait does not consume them.** A compile-time
  `least` is therefore satisfied on every launch after the first, which silently
  turns the wait into a no-op. Pass the target as a kernel argument that the host
  advances per launch.
- **Signal state is per context.** A put issued on sender context *i* increments
  the receiver's signal through context *i*, so a CTA sees only `1/contexts` of
  the arrivals. `wait_signal` divides `least` by the device-side
  `context_span()` for this reason. A wait grid narrower than the sender's rounds
  the target down, to zero in the worst case; a wider one parks CTAs on contexts
  nothing signals.
- **The requested context count is a hint.** `ncclDevCommRequirements.ginContextCount`
  may be granted in part, so the divisor must come from the device rather than
  from the host's request.

Do not wait on a signal from inside a large-grid kernel. A waiting CTA occupies
an SM, so if the CTAs that would issue the matching puts are still queued behind
it, nothing progresses. Use separate streams, or confine the wait to a
small-grid kernel.

### Multimem Operations

Multimem operations require a valid multicast allocation and compatible
NVSwitch/multicast hardware. `MultimemReduceOp` provides `ADD`, `MIN`, `MAX`,
and `NONE`; `NONE` is retained for compatibility and is equivalent to `None`
for a plain `multimem_tma_store`.

| API | Purpose |
|-----|---------|
| `T.multimem_ld_reduce(src, dst, reduce_op=ADD)` | Reduce-load a multicast region into a local region |
| `T.multimem_st(src, dst)` | Broadcast a local region into a multicast region |
| `T.multimem_red(src, dst, reduce_op=ADD)` | Reduce-store into a multicast region |
| `T.multimem_tma_store(src, dst, reduce_op=None)` | Shared-memory to multicast TMA store or reduce-store |
| `T.multimem_signal(addr, value)` | Store a `uint32` or `uint64` signal through a multicast address |
| `T.multimem_signal_add(addr, value)` | Add a `uint32`, `int32`, or `uint64` signal through a multicast address |

The signal type is inferred from `addr`; there is no `dtype_tag` argument. The
direct `multimem_ld_reduce` and `multimem_red` paths currently expose ADD;
unsupported PTX dtype/operation combinations are rejected during lowering.

Two constraints shape how these are called in practice:

- **A float16/bfloat16 region must be exactly one contiguous pair per thread**,
  because those dtypes lower to packed x2 instructions. In tile terms the region
  width must equal `2 * threads`; any other width lets a neighbouring `T.copy`
  infer a wider vectorisation, and layout inference then fails with *"requires
  the local fragment layout to preserve canonical pair ownership"*. Work per
  thread therefore has to come from looping over tiles, not from widening one.
- **The multicast region must be provably in bounds at compile time.** A runtime
  offset into it is rejected with *"multimem packed multicast region must be
  provably in bounds or use a tile-aligned all-or-none dynamic partition"*, so a
  kernel that publishes a varying slice needs the offset as a compile-time
  constant, i.e. one specialisation per slice.
Direct multimem operations and signals require SM90+ and CUDA Toolkit 12.1+
(PTX 8.1+). Packed `float16`/`bfloat16` load-reduce additionally requires CUDA
Toolkit 12.2+ because its `.acc::f32` form was introduced in PTX 8.2. Packed
regions require an even last extent, pair-aligned multicast rows, a full local
fragment region, and a fragment layout that preserves contiguous-pair thread
ownership. Dynamic packed regions must be tile-aligned, all-or-none partitions;
a region that can partially overlap the buffer is rejected during lowering.

Bulk `multimem_tma_store` additionally requires CUDA Toolkit 13.1+. Its source and
destination regions must have matching rank, extents, and dtype. Both regions
must be in bounds, physically contiguous, byte-addressable, and start at a
provably 16-byte-aligned address. The positive transfer size must be divisible
by 16 bytes and fit in an unsigned 32-bit byte count. Layout-remapped buffers
are rejected. Plain bulk stores accept matching byte-addressable scalar or
vector dtypes. Bulk reductions accept ADD for `float32`, `float16`, and
`bfloat16`, and MIN/MAX for `float16` and `bfloat16`. Dynamic bulk regions obey
the same tile-aligned, all-or-none rule. Explicit Bulk use on an unsupported
architecture or pre-13.1 toolkit fails during device compilation instead of
emitting a runtime trap.

Shared-memory producer synchronization and async bulk-group completion are
managed by the caller. Every producer thread must make its shared-memory writes
visible to the async proxy before the CTA synchronizes. A single elected thread
then issues the store, commit, and wait sequence:

```python
T.fence_proxy_async()
T.sync_threads()
if T.get_thread_binding() == 0:
    T.multimem_tma_store(shared, multicast)
    T.tma_store_arrive()
    T.tma_store_wait(0, False)
```

## Host Utilities

| API | Current behavior |
|-----|------------------|
| `set_signal(tensor, value, stream=None)` | Enqueues `cuStreamWriteValue32`; accepts `int32` or `uint32` tensors and an unsigned 32-bit value |
| `wait_eq(tensor, value, stream=None, require_i64=False)` | Enqueues `cuStreamWaitValue32` for `int32`/`uint32`; with `require_i64=True`, enqueues `cuStreamWaitValue64` for `int64`/`uint64` and an unsigned 64-bit value |
| `supports_p2p_native_atomic()` | Checks native P2P atomic support only for CUDA devices 0 and 1 and requires at least two visible GPUs |
| `cuda_stream_max_priority()` | Returns the CUDA runtime's highest stream priority value |
| `do_bench(...)` / `perf_fn(...)` | CUDA-event benchmark helper; optionally aggregates rank timings when a process group is initialized |

Signal tensors must be nonempty, contiguous CUDA tensors. An explicitly
provided stream must belong to the same CUDA device as the tensor. These checks
run before a raw address or stream handle is passed to the CUDA driver.

Topology helpers try NVML first and fall back to `nvidia-smi topo -m`:

- `has_fullmesh_nvlink_pynvml()` performs the NVML pairwise check.
- `has_fullmesh_nvlink()` uses NVML when available and otherwise parses
  `nvidia-smi` output.
- `NvidiaSmiUtil.get_gpu_numa_node(index)` reads the Linux PCI NUMA mapping.

The full-mesh helpers return `True` for zero or one visible GPU. They describe
topology and must not be used alone as proof that a multi-GPU test ran.

## Shared-Memory FFI

The underscore-prefixed bindings are internal APIs. Their current signatures
are listed here because the allocator depends on them; they may change between
releases. Every registered FFI name below is prefixed with
`tl.shared_memory.`.

### IPC and VMM

| Python binding | Registered FFI call |
|----------------|---------------------|
| `_create_ipc_handle(ptr) -> bytes` | `create_ipc_handle(ptr: int64) -> Bytes` |
| `_open_ipc_handle(handle) -> int` | `open_ipc_handle(handle: Bytes) -> int64` |
| `_close_ipc_handle(ptr) -> None` | `close_ipc_handle(ptr: int64) -> void` |
| `_sync_ipc_handles(rank, device_ids, buffer_ptrs_gpu_addr, handles, root_unique_id_opt=None) -> None` | `sync_ipc_handles(rank: int64, num_ranks: int64, buffer_ptrs_gpu_addr: int64, packed_handles: Bytes) -> void` |
| `_vmm_malloc(size) -> int` | `vmm_malloc(size: int64) -> int64` |
| `_vmm_free(ptr) -> None` | `vmm_free(ptr: int64) -> void` |
| `_create_vmm_handle(ptr) -> bytes` | `create_vmm_handle(ptr: int64) -> Bytes` |
| `_open_vmm_handle(handle) -> int` | `open_vmm_handle(handle: Bytes) -> int64` |
| `_close_vmm_handle(ptr) -> None` | `close_vmm_handle(ptr: int64) -> void` |
| `_sync_vmm_handles(rank, device_ids, buffer_ptrs_gpu_addr, handles) -> None` | `sync_vmm_handles(rank: int64, num_ranks: int64, buffer_ptrs_gpu_addr: int64, packed_handles: Bytes) -> void` |
| `_supports_vmm_fabric() -> bool` | `supports_vmm_fabric() -> bool` |
| `_supports_multicast() -> bool` | `supports_multicast() -> bool` |
| `_create_vmm_fd_handle(ptr) -> bytes` | `create_vmm_fd_handle(ptr: int64) -> Bytes` |
| `_open_vmm_fd_handle(handle) -> int` | `open_vmm_fd_handle(handle: Bytes) -> int64` |

`create_vmm_fd_handle` exports the allocation as a POSIX file descriptor and
returns `size | pid | fd`; `open_vmm_fd_handle` duplicates that descriptor into
the caller with `pidfd_open` plus `pidfd_getfd`, then imports and maps it. This
is the route used when fabric handles are unavailable, and it avoids the
unix-socket rendezvous an `SCM_RIGHTS` exchange would need. The exporting
process keeps its descriptor open for its lifetime, because a peer may import at
any point before teardown.

The Python sync wrappers pack one handle per rank and derive `num_ranks` from
`len(device_ids)`. `buffer_ptrs_gpu_addr` is the integer device address of a
CUDA `uint64` pointer table. The optional IPC `root_unique_id_opt` parameter is
currently ignored.

### Multicast Internals

| Python binding | Registered FFI signature |
|----------------|--------------------------|
| `_mc_create(size, num_devices) -> int` | `(int64, int64) -> int64` |
| `_mc_export_handle(handle) -> bytes` | `(int64) -> Bytes` |
| `_mc_import_handle(handle_bytes) -> int` | `(Bytes) -> int64` |
| `_mc_add_device(handle, device_id) -> None` | `(int64, int64) -> void` |
| `_mc_bind_mem(handle, ptr, size) -> None` | `(int64, int64, int64) -> void` |
| `_mc_map(handle, size, num_devices) -> int` | `(int64, int64, int64) -> int64` |
| `_mc_release_handle(handle) -> None` | `(int64) -> void` |
| `_mc_unmap(ptr, size, num_devices) -> None` | `(int64, int64, int64) -> void` |
| `_mc_get_aligned_size(size, num_devices) -> int` | `(int64, int64) -> int64` |
| `_mc_export_fd_handle(handle) -> bytes` | `(int64) -> Bytes` |
| `_mc_open_fd_handle(handle_bytes) -> int` | `(Bytes) -> int64` |
| `_multicast_uses_fd() -> bool` | `multicast_uses_fd() -> bool` |

`mc_create` requests whichever handle type `multicast_uses_fd()` reports, and the
matching export/import pair must be used with it. The fd pair carries `pid | fd`
and is duplicated with `pidfd_getfd`, as for the arena.

### Tensor From Pointer

```python
tensor_from_ptr(
    ptr_val,
    shape,
    dtype_str="float32",
    device=0,
    take_ownership=False,
) -> torch.Tensor
```

This creates a zero-copy CUDA tensor view. Supported dtype strings are
`float32`/`float`, `float16`/`half`, `bfloat16`, `float64`/`double`,
`int32`/`int`, `int64`/`long`, `uint8`/`byte`, `int8`, `bool`, `uint32`, and
`uint64`. `take_ownership=True` is not implemented.

## Validation Boundaries

Most maintained cross-rank compatibility cases request four spawned processes
and skip when fewer than four GPUs are visible. A small number of tests retain
two or eight processes when the primitive or algorithm requires that topology.
Fabric and multicast cases additionally skip when their runtime capability
probes fail. A skipped capability test is not evidence that the corresponding
path works on that host; release validation should record the GPU count and
which capability-gated cases actually ran.

The maintained examples and tests are under `examples/distributed` and
`testing/python/distributed`. This reference intentionally does not label every
example as universally working because results depend on GPU architecture,
topology, CUDA toolkit, driver, and IMEX configuration.

Two inter-node limitations are known and will be met in normal use rather than
at the edges:

- **Some `put_signal` transfer sizes fail to lower**, with
  `Can't fetch the lanes of a scalable vector at a compile time`. The trigger is
  the element count, not the byte count, and the failing set is not contiguous.
  Callers that derive a transfer size from a buffer length should be prepared to
  adjust it; halving the chunk count doubles the transfer and generally moves off
  a bad value.
- **`T.barrier_blocks` is single-node** despite documenting a rendezvous across
  "every rank". It is lowered with the global rank and world size, and the device
  side takes `get_remote_base_ptr` of each participant, which returns 0 for a peer
  it considers inter-node. In a multi-node job the inter-node slots become null
  system atomics. Inter-node ordering should come from GIN signals, which need no
  barrier; a node-local barrier variant does not exist yet.
