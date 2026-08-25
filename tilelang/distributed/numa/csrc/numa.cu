/**
 * NUMA-aware memory allocation and data packing for B200 dual-die GPUs.
 *
 * Provides:
 * - cudaMallocAndGetPageId: allocate + detect which die pages landed on
 * - pack_page / unpack_page: linear <-> tiled layout conversion
 * - pack_remap / unpack_remap: tiled <-> NUMA-interleaved layout
 */

#include "numa.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace tilescale_numa {

// ---- Device helpers ----

__device__ __forceinline__ int smid() {
    int sm;
    asm volatile("mov.b32 %0, %%smid;" : "=r"(sm));
    return sm;
}

__host__ __device__ __forceinline__ int popc32(uint32_t value) {
#ifdef __CUDA_ARCH__
    return __popc(value);
#else
    return __builtin_popcount(value);
#endif
}

__host__ __device__ __forceinline__ int isRemoteTile(int pageId, uint64_t tileId) {
    tileId += pageId * 1024;
    tileId %= 4194304;
    return popc32(static_cast<uint32_t>(tileId & 0x2AD3EF)) & 1;
}

// ---- Page ID detection via latency probing ----

__device__ __forceinline__ uint32_t ld_global_cg(uint32_t *gmem) {
    uint32_t result;
    asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(result) : "l"(gmem));
    return result;
}

__device__ __forceinline__ void st_global_cg(uint32_t *gmem, uint32_t value) {
    asm volatile("st.global.cg.b32 [%0], %1;" : : "l"(gmem), "r"(value) : "memory");
}

__global__ void latencyTestKernel(int target_sm, int numPages, uint32_t *pool, uint16_t *lats) {
    __shared__ uint32_t values[8192];
    __shared__ uint16_t clocks[8192];

    if (smid() != target_sm) return;
    if (threadIdx.x != 0) return;

    for (int pageId = 0; pageId < numPages; ++pageId) {
        uint32_t *gmem = pool + pageId * 1024 * 1024;
        uint32_t value = ld_global_cg(gmem);
        value += 1;
        st_global_cg(gmem, value);
        __threadfence();

        int start = clock();
        value = ld_global_cg(gmem);
        value += 1;
        int end = clock();
        values[pageId] = value;
        clocks[pageId] = (uint16_t)(end - start);
    }

    for (int pageId = 0; pageId < numPages; ++pageId)
        (pool + pageId * 1024 * 1024)[0] = values[pageId];

    for (int pageId = 0; pageId < numPages; ++pageId)
        lats[pageId] = clocks[pageId];
}

int getDieId(int pageId) {
    return __builtin_popcount(pageId & 0xAB4) & 1;
}

int cudaMallocAndGetPageId(void **devPtr, size_t size) {
    cudaMalloc(devPtr, size);

    int numPages = size / (4 * 1024 * 1024);

    int *pageIdToDieId = new int[8192];
    uint16_t *lats;
    cudaMalloc(&lats, numPages * sizeof(uint16_t));

    uint16_t *temp = new uint16_t[8192];
    latencyTestKernel<<<148, 32>>>(0, numPages, (uint32_t *)*devPtr, lats);
    cudaDeviceSynchronize();
    cudaMemcpy(temp, lats, numPages * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    for (int pageId = 0; pageId < numPages; ++pageId)
        pageIdToDieId[pageId] = temp[pageId] > 500;

    cudaFree(lats);
    delete[] temp;

    for (int baseId = 0; baseId < 4096; ++baseId) {
        bool flag = true;
        for (int pageId = 0; pageId < numPages; ++pageId)
            flag &= getDieId((baseId + pageId) % 4096) == pageIdToDieId[pageId];
        if (flag) {
            delete[] pageIdToDieId;
            return baseId;
        }
    }

    delete[] pageIdToDieId;
    cudaFree(*devPtr);
    *devPtr = nullptr;
    return -1;
}

// ---- Pack/Unpack: linear <-> tiled layout ----

template <typename T>
__global__ void pack_page_kernel(T *dst, const T *src, int tileK, int numtileK,
                                  int tileMN, int numtileMN, int64_t total_elements) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_elements) return;
    auto k = numtileK * tileK;
    dst[i / (tileMN * k) * tileMN * k +
        i / tileK % (k / tileK) * tileMN * tileK +
        i / k % tileMN * tileK +
        i % tileK] = src[i];
}

template <typename T>
__global__ void unpack_page_kernel(T *dst, const T *src, int tileK, int numtileK,
                                    int tileMN, int numtileMN, int64_t total_elements) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_elements) return;
    auto k = numtileK * tileK;
    dst[i] = src[i / (tileMN * k) * tileMN * k +
                 i / tileK % (k / tileK) * tileMN * tileK +
                 i / k % tileMN * tileK +
                 i % tileK];
}

void pack_page(void *dst, const void *src, int tileK, int tileM, int K, int total_MN, int elem_bytes) {
    int64_t total_elements = (int64_t)K * total_MN;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    if (elem_bytes == 2) {
        pack_page_kernel<<<grid_size, block_size>>>((uint16_t *)dst, (const uint16_t *)src,
                                                     tileK, K / tileK, tileM, total_MN / tileM, total_elements);
    } else {
        pack_page_kernel<<<grid_size, block_size>>>((uint8_t *)dst, (const uint8_t *)src,
                                                     tileK, K / tileK, tileM, total_MN / tileM, total_elements);
    }
}

void unpack_page(void *dst, const void *src, int tileK, int tileM, int K, int total_MN, int elem_bytes) {
    int64_t total_elements = (int64_t)K * total_MN;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    if (elem_bytes == 2) {
        unpack_page_kernel<<<grid_size, block_size>>>((uint16_t *)dst, (const uint16_t *)src,
                                                       tileK, K / tileK, tileM, total_MN / tileM, total_elements);
    } else {
        unpack_page_kernel<<<grid_size, block_size>>>((uint8_t *)dst, (const uint8_t *)src,
                                                       tileK, K / tileK, tileM, total_MN / tileM, total_elements);
    }
}

// ---- Pack/Unpack: tiled <-> NUMA-interleaved ----

__device__ __forceinline__ uint64_t countSameTypeTilesBefore(int page_id, uint64_t tile_id, bool is_remote) {
    uint64_t count = 0;
    for (uint64_t i = 0; i < tile_id; ++i) {
        if (isRemoteTile(page_id, i) == is_remote) count++;
    }
    return count;
}

template <typename T>
__global__ void pack_remap_kernel(T *dst, const T *src, int page_id,
                                   uint64_t num_tiles, uint64_t elem_per_tile,
                                   uint64_t remote_group_offset, uint64_t total_elems) {
    uint64_t tile_id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_id >= num_tiles) return;

    uint64_t phys_start = tile_id * elem_per_tile;
    bool is_remote = isRemoteTile(page_id, tile_id);
    uint64_t rank = countSameTypeTilesBefore(page_id, tile_id, is_remote);

    uint64_t log_start;
    if (is_remote) {
        if (remote_group_offset >= total_elems) {
            for (uint64_t i = 0; i < elem_per_tile; ++i) dst[phys_start + i] = T(0);
            return;
        }
        log_start = remote_group_offset + rank * elem_per_tile;
    } else {
        if (remote_group_offset == 0) {
            for (uint64_t i = 0; i < elem_per_tile; ++i) dst[phys_start + i] = T(0);
            return;
        }
        log_start = rank * elem_per_tile;
    }

    if (log_start + elem_per_tile <= total_elems) {
        for (uint64_t i = 0; i < elem_per_tile; ++i)
            dst[phys_start + i] = src[log_start + i];
    }
}

template <typename T>
__global__ void unpack_remap_kernel(T *dst, const T *src, int page_id,
                                     uint64_t num_tiles, uint64_t elem_per_tile,
                                     uint64_t remote_group_offset, uint64_t total_elems) {
    uint64_t tile_id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_id >= num_tiles) return;

    uint64_t phys_start = tile_id * elem_per_tile;
    bool is_remote = isRemoteTile(page_id, tile_id);
    uint64_t rank = countSameTypeTilesBefore(page_id, tile_id, is_remote);

    uint64_t log_start;
    if (is_remote) {
        if (remote_group_offset >= total_elems) return;
        log_start = remote_group_offset + rank * elem_per_tile;
    } else {
        if (remote_group_offset == 0) return;
        log_start = rank * elem_per_tile;
    }

    if (log_start + elem_per_tile <= total_elems) {
        for (uint64_t i = 0; i < elem_per_tile; ++i)
            dst[log_start + i] = src[phys_start + i];
    }
}

uint64_t div_up(uint64_t value, uint64_t divisor) {
    return (value + divisor - 1) / divisor;
}

uint64_t remap_num_tiles(uint64_t total_elems, int elem_bytes) {
    uint64_t elem_per_tile = kPageSize / elem_bytes;
    return div_up(total_elems, elem_per_tile);
}

uint64_t remap_padded_elems(uint64_t total_elems, int elem_bytes) {
    uint64_t elem_per_tile = kPageSize / elem_bytes;
    return remap_num_tiles(total_elems, elem_bytes) * elem_per_tile;
}

uint64_t remap_remote_group_offset(int page_id, uint64_t total_elems, int elem_bytes) {
    uint64_t elem_per_tile = kPageSize / elem_bytes;
    uint64_t num_tiles = remap_num_tiles(total_elems, elem_bytes);
    uint64_t local_tiles = 0;
    for (uint64_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        if (!isRemoteTile(page_id, tile_id)) local_tiles++;
    }
    return local_tiles * elem_per_tile;
}

void pack_remap(void *dst, const void *src, int page_id, uint64_t total_elems,
                uint64_t remote_group_offset, int elem_bytes) {
    uint64_t elem_per_tile = kPageSize / elem_bytes;
    uint64_t num_tiles = total_elems / elem_per_tile;
    if (num_tiles == 0) return;
    int grid_size = (num_tiles + 255) / 256;
    if (elem_bytes == 2) {
        pack_remap_kernel<<<grid_size, 256>>>((uint16_t *)dst, (const uint16_t *)src,
                                               page_id, num_tiles, elem_per_tile, remote_group_offset, total_elems);
    } else {
        pack_remap_kernel<<<grid_size, 256>>>((uint8_t *)dst, (const uint8_t *)src,
                                               page_id, num_tiles, elem_per_tile, remote_group_offset, total_elems);
    }
}

void unpack_remap(void *dst, const void *src, int page_id, uint64_t total_elems,
                  uint64_t remote_group_offset, int elem_bytes) {
    uint64_t elem_per_tile = kPageSize / elem_bytes;
    uint64_t num_tiles = total_elems / elem_per_tile;
    if (num_tiles == 0) return;
    int grid_size = (num_tiles + 255) / 256;
    if (elem_bytes == 2) {
        unpack_remap_kernel<<<grid_size, 256>>>((uint16_t *)dst, (const uint16_t *)src,
                                                 page_id, num_tiles, elem_per_tile, remote_group_offset, total_elems);
    } else {
        unpack_remap_kernel<<<grid_size, 256>>>((uint8_t *)dst, (const uint8_t *)src,
                                                 page_id, num_tiles, elem_per_tile, remote_group_offset, total_elems);
    }
}

}  // namespace tilescale_numa

// ---- TVM FFI Bindings ----

#include <tvm/ffi/tvm_ffi.h>
#include <unordered_map>

struct NUMAHandle {
    void *ptr;
    int page_id;
    int64_t total_elems;
    int elem_bytes;
};

static std::unordered_map<int64_t, NUMAHandle> &get_handles() {
    static std::unordered_map<int64_t, NUMAHandle> handles;
    return handles;
}

namespace ffi_numa {

int64_t alloc(int64_t size_bytes, int64_t elem_bytes) {
    constexpr int64_t PAGE_SIZE_4M = 4LL * 1024 * 1024;
    int64_t alloc_size = ((size_bytes + PAGE_SIZE_4M - 1) / PAGE_SIZE_4M) * PAGE_SIZE_4M;

    void *ptr = nullptr;
    int page_id = tilescale_numa::cudaMallocAndGetPageId(&ptr, (size_t)alloc_size);
    if (page_id == -1) return 0;

    int64_t handle = reinterpret_cast<int64_t>(ptr);
    int64_t total_elems = size_bytes / elem_bytes;
    get_handles()[handle] = {ptr, page_id, total_elems, (int)elem_bytes};
    return handle;
}

void free(int64_t handle) {
    auto &handles = get_handles();
    auto it = handles.find(handle);
    if (it == handles.end()) return;
    cudaFree(it->second.ptr);
    handles.erase(it);
}

int32_t get_page_id(int64_t handle) {
    auto it = get_handles().find(handle);
    return it != get_handles().end() ? it->second.page_id : -1;
}

void pack(tvm::ffi::TensorView src, int64_t dst_handle, int32_t tileK, int32_t tileMN, int32_t K, int32_t total_MN) {
    auto it = get_handles().find(dst_handle);
    if (it == get_handles().end()) return;
    auto &info = it->second;

    uint64_t total_elems = (uint64_t)K * (uint64_t)total_MN;
    uint64_t padded_elems = tilescale_numa::remap_padded_elems(total_elems, info.elem_bytes);
    uint64_t remote_group_offset =
        tilescale_numa::remap_remote_group_offset(info.page_id, total_elems, info.elem_bytes);
    size_t size_bytes = padded_elems * info.elem_bytes;

    void *paged = nullptr;
    cudaMalloc(&paged, size_bytes);
    cudaMemset(paged, 0, size_bytes);

    tilescale_numa::pack_page(paged, src.data_ptr(), tileK, tileMN, K, total_MN, info.elem_bytes);
    tilescale_numa::pack_remap(info.ptr, paged, info.page_id, padded_elems,
                                remote_group_offset, info.elem_bytes);
    cudaDeviceSynchronize();
    cudaFree(paged);
}

void unpack(int64_t src_handle, tvm::ffi::TensorView dst, int32_t tileK, int32_t tileMN, int32_t K, int32_t total_MN) {
    auto it = get_handles().find(src_handle);
    if (it == get_handles().end()) return;
    auto &info = it->second;

    uint64_t total_elems = (uint64_t)K * (uint64_t)total_MN;
    uint64_t padded_elems = tilescale_numa::remap_padded_elems(total_elems, info.elem_bytes);
    uint64_t remote_group_offset =
        tilescale_numa::remap_remote_group_offset(info.page_id, total_elems, info.elem_bytes);
    size_t size_bytes = padded_elems * info.elem_bytes;

    void *paged = nullptr;
    cudaMalloc(&paged, size_bytes);
    cudaMemset(paged, 0, size_bytes);

    tilescale_numa::unpack_remap(paged, info.ptr, info.page_id, padded_elems,
                                  remote_group_offset, info.elem_bytes);
    tilescale_numa::unpack_page(dst.data_ptr(), paged, tileK, tileMN, K, total_MN, info.elem_bytes);
    cudaDeviceSynchronize();
    cudaFree(paged);
}

}  // namespace ffi_numa

TVM_FFI_DLL_EXPORT_TYPED_FUNC(tilescale_numa_alloc, ffi_numa::alloc);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(tilescale_numa_free, ffi_numa::free);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(tilescale_numa_get_page_id, ffi_numa::get_page_id);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(tilescale_numa_pack, ffi_numa::pack);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(tilescale_numa_unpack, ffi_numa::unpack);
