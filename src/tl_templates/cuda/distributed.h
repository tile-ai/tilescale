#pragma once

#include "common.h"
#include <cstdint>

namespace tl {

extern "C" __device__ uint64_t meta_data[1024];
extern "C" uint64_t *host_meta_data;

TL_HOST_DEVICE uint64_t get_rank() {
#ifdef __CUDA_ARCH__
  return meta_data[0];
#else
  return host_meta_data[0];
#endif
}

TL_HOST_DEVICE uint64_t get_num_ranks() {
#ifdef __CUDA_ARCH__
  return meta_data[1];
#else
  return host_meta_data[1];
#endif
}

TL_HOST_DEVICE void *get_remote_base_ptr(uint64_t rank) {
#ifdef __CUDA_ARCH__
  return (void *)meta_data[2 + rank];
#else
  return (void *)host_meta_data[2 + rank];
#endif
}

// NOTE(wt): Be careful about the return types here!
// get_local_base() returns u64 since I could not find a way cast u64 to ptr in
// tir
TL_HOST_DEVICE uint64_t get_local_base() {
#ifdef __CUDA_ARCH__
  return meta_data[2 + get_rank()];
#else
  return host_meta_data[2 + get_rank()];
#endif
}

template <typename dtype_t>
TL_HOST_DEVICE uint64_t get_uintptr_t(dtype_t *ptr) {
  return reinterpret_cast<uint64_t>(ptr);
}

} // namespace tl
