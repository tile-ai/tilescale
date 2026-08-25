#pragma once
#include <cstdint>

namespace tilescale_numa {

constexpr int kPageSize = 4 * 1024;  // 4KB per tile (32 rows × 64 cols × 2 bytes)

int cudaMallocAndGetPageId(void **devPtr, size_t size);

template <typename T>
int cudaMallocAndGetPageId(T **devPtr, size_t size) {
    return cudaMallocAndGetPageId((void **)devPtr, size);
}

void pack_page(void *dst, const void *src, int tileK, int tileM, int K, int total_MN, int elem_bytes);
void unpack_page(void *dst, const void *src, int tileK, int tileM, int K, int total_MN, int elem_bytes);
void pack_remap(void *dst, const void *src, int page_id, uint64_t total_elems, uint64_t remote_group_offset, int elem_bytes);
void unpack_remap(void *dst, const void *src, int page_id, uint64_t total_elems, uint64_t remote_group_offset, int elem_bytes);

}  // namespace tilescale_numa
