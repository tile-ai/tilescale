#pragma once

// Layout of the distributed metadata table.
//
// The table is produced on the host by BaseAllocator._init_table
// (tilelang/distributed/allocator.py), copied into the device __constant__
// `meta_data` symbol by __tilescale_init_table
// (src/runtime/tilescale_cuda_module.cc), and additionally cached on the host
// for remote TMA descriptor encoding by SetRemoteTensorMapMetaData
// (src/cuda/runtime.cc).
//
// All three readers must agree, so the offsets live here and nowhere else.
// This header is intentionally free of CUDA constructs so it can be included
// from plain host translation units.
//
//   [0] global_rank
//   [1] global_world_size
//   [2] node_rank
//   [3] num_nodes
//   [4] local_rank
//   [5] local_world_size
//   [6] global NCCL device-comm handle (0 when unused)
//   [7] inter-node NCCL device-comm handle (0 when unused)
//   [8] ncclWindow_t for the whole allocator arena (0 when unregistered)
//   [9] local arena base address, for pointer -> window offset conversion
//   [10 .. 10 + local_world_size) intra-node peer base pointers, by local rank
//
// The GIN context count is deliberately absent. The devcomm may grant fewer
// contexts than the allocator requested, and only the device can read the
// granted number back (tl::gin::context_span in nccl_gin.h). Publishing the
// host's request here would let a kernel index past the end of the context
// array, which hangs rather than failing.
//
// Peer base pointers are node-local by construction: inter-node peers are not
// mappable into this address space, so their bases are never present here.
//
// Inter-node peers are reached instead through the arena window. GIN addresses
// memory as an (ncclWindow_t, byte offset) pair rather than a raw pointer,
// because a remote rank's allocation has no local virtual address. The window
// handle returned by ncclCommWindowRegister is not per-peer: one local handle
// plus a peer index names any rank's bytes, so a single arena registration
// covers the whole communicator. Since the arena is symmetric, a local pointer
// converts to the offset valid on every rank by subtracting ARENA_BASE -- the
// same subtraction the intra-node path already performs, just paired with a
// window handle instead of a peer base pointer.

#define TL_META_GLOBAL_RANK 0
#define TL_META_GLOBAL_WORLD_SIZE 1
#define TL_META_NODE_RANK 2
#define TL_META_NUM_NODES 3
#define TL_META_LOCAL_RANK 4
#define TL_META_LOCAL_WORLD_SIZE 5
#define TL_META_GLOBAL_DEV_COMM 6
#define TL_META_INTERNODE_DEV_COMM 7
#define TL_META_ARENA_WINDOW 8
#define TL_META_ARENA_BASE 9
#define TL_META_PEER_BASE 10

// Number of scalar header entries preceding the peer base pointer array.
#define TL_META_HEADER_SIZE TL_META_PEER_BASE
