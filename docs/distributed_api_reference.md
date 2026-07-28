# TileScale Distributed API Reference

TileScale extends TileLang with experimental single-node, multi-GPU CUDA
primitives. The supported process model is one process per local GPU on one
host, using an NCCL process group for host-side coordination. Multi-node
execution, NVSHMEM, and a general-purpose distributed runtime are outside the
current scope.

The Python distribution is `tilescale`, while the import namespace remains
`tilelang`. Install the optional host dependencies with
`pip install "tilescale[distributed]"`.

## Runtime Requirements

The available memory path depends on the host and visible GPUs:

| Path | Requirements | Selection |
|------|--------------|-----------|
| CUDA IPC | Same host, CUDA peer access between participating GPUs | Used when VMM is disabled or unavailable |
| VMM fabric | CUDA driver API 12.4 or newer, fabric-handle support on every visible GPU, and an accessible NVIDIA IMEX channel | Auto-selected for distributed allocators when the runtime probe succeeds |
| Multicast | VMM fabric requirements plus multicast-capable GPUs and fabric, normally an NVSwitch system | Enabled only when `mcast_size` is explicitly requested |

`_supports_vmm_fabric()` and `_supports_multicast()` are runtime probes, not
portable feature guarantees. Both inspect all CUDA-visible devices, so set
`CUDA_VISIBLE_DEVICES` to the exact local rank set before starting processes.

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
```

`init_dist` sets `cuda:local_rank`, creates an NCCL process group, and returns
its rank, size, and `dist.group.WORLD`.

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

### Multicast Allocation

Passing `mcast_size` requires a distributed VMM allocator and a successful
multicast capability probe. The internal
`allocator._allocate_mcast_tensor(shape, dtype)` method returns
`(multicast_view, local_physical_view)`. The multicast view is used by multimem
instructions; each rank writes its own contribution through the local physical
view.

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
