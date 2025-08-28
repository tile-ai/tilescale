#pragma once

#include "common.h"

namespace tl {

extern "C" extern __device__ uint64_t meta_data[1024];

TL_DEVICE uint64_t get_rank() {
  return meta_data[0];
}

TL_DEVICE uint64_t get_num_ranks() {
  return meta_data[1];
}

TL_DEVICE uint64_t get_remote_base_ptr(uint64_t rank) {
  return meta_data[2 + rank];
}

template <typename dtype_t>
TL_DEVICE uint64_t get_uintptr_t(dtype_t* ptr) {
  return reinterpret_cast<uint64_t>(ptr);
}

} // namespace tl
