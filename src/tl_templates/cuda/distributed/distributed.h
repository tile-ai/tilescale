#pragma once

#include "../common.h"
#include "meta_layout.h"

#define TL_ENABLE_DISTRIBUTED_METADATA 1
extern "C" {
__constant__ uint64_t meta_data[1024];
}
namespace tl {

// See meta_layout.h for the table layout shared with the host runtime.

TL_DEVICE uint64_t get_global_rank() { return meta_data[TL_META_GLOBAL_RANK]; }

TL_DEVICE uint64_t get_global_world_size() {
  return meta_data[TL_META_GLOBAL_WORLD_SIZE];
}

TL_DEVICE uint64_t get_node_rank() { return meta_data[TL_META_NODE_RANK]; }

TL_DEVICE uint64_t get_num_nodes() { return meta_data[TL_META_NUM_NODES]; }

TL_DEVICE uint64_t get_local_rank() { return meta_data[TL_META_LOCAL_RANK]; }

TL_DEVICE uint64_t get_local_world_size() {
  return meta_data[TL_META_LOCAL_WORLD_SIZE];
}

// Backward compatibility aliases
TL_DEVICE uint64_t get_rank() { return get_global_rank(); }

TL_DEVICE uint64_t get_num_ranks() { return get_global_world_size(); }

// Check if target rank is on same node
TL_DEVICE bool is_local_peer(uint64_t target_global_rank) {
  uint64_t target_node = target_global_rank / get_local_world_size();
  return target_node == get_node_rank();
}

// Get intra-node base pointer (local peers only)
TL_DEVICE uint64_t get_local_peer_base_ptr(uint64_t local_rank) {
  return meta_data[TL_META_PEER_BASE + local_rank];
}

// Get remote base pointer by global rank. Single-node ranks are their own local
// ranks, so this indexes the peer array directly. Returns 0 for an inter-node
// rank, whose memory is not mappable here: route those through GIN instead.
TL_DEVICE uint64_t get_remote_base_ptr(uint64_t rank) {
  if (get_num_nodes() == 1) {
    return get_local_peer_base_ptr(rank);
  }
  if (!is_local_peer(rank)) {
    return 0;
  }
  return get_local_peer_base_ptr(rank % get_local_world_size());
}

template <typename dtype_t> TL_DEVICE uint64_t get_uintptr_t(dtype_t *ptr) {
  return reinterpret_cast<uint64_t>(ptr);
}

} // namespace tl

TL_DEVICE void print_table() {
  printf("Table base address: %p\n", static_cast<const void *>(meta_data));
  for (int i = 0; i < 10; i++) {
    printf("meta_data[%d] = %llu\n", i,
           static_cast<unsigned long long>(meta_data[i]));
  }
}
