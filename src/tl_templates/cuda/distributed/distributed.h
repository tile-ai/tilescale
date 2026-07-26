#pragma once

#include "../common.h"

#define TL_ENABLE_DISTRIBUTED_METADATA 1
extern "C" {
__constant__ uint64_t meta_data[1024];
}
namespace tl {

TL_DEVICE uint64_t get_rank() { return meta_data[0]; }

TL_DEVICE uint64_t get_num_ranks() { return meta_data[1]; }

TL_DEVICE uint64_t get_remote_base_ptr(uint64_t rank) {
  return meta_data[2 + rank];
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
