#pragma once

#include "common.h"

extern "C" extern __constant__ uint64_t meta_data[1024];
namespace tl {

TL_DEVICE uint64_t get_rank() { return meta_data[0]; }

TL_DEVICE uint64_t get_num_ranks() { return meta_data[1]; }

TL_DEVICE uint64_t get_remote_base_ptr(uint64_t rank) {
  return meta_data[2 + rank];
}

template <typename dtype_t> TL_DEVICE uint64_t get_uintptr_t(dtype_t *ptr) {
  return reinterpret_cast<uint64_t>(ptr);
}

// Block-level remote copy: copies N*sizeof(float) bytes from src to dst
template <int N> TL_DEVICE void cp_block(uint64_t dst_addr, uint64_t src_addr) {
  using CopyT = uint64_t;
  constexpr int num_elements =
      (N * (int)sizeof(float) + (int)sizeof(CopyT) - 1) / (int)sizeof(CopyT);
  auto *dst = reinterpret_cast<CopyT *>(dst_addr);
  const auto *src = reinterpret_cast<const CopyT *>(src_addr);
#pragma unroll
  for (int i = 0; i < num_elements; i++) {
    dst[i] = src[i];
  }
}

// Warp-level remote copy
template <int N, int UNROLL_FACTOR, bool AGGRESSIVE_VECTORIZE>
TL_DEVICE void cp_warp(uint64_t dst_addr, uint64_t src_addr) {
  cp_block<N>(dst_addr, src_addr);
}

} // namespace tl

TL_DEVICE void print_table() {
  std::printf("Table base address: %llu\n", meta_data);
  for (int i = 0; i < 10; i++) {
    std::printf("meta_data[%d] = %llu\n", i, meta_data[i]);
  }
}
