#pragma once

#include "common.h"

#define IS_MASTER_THREAD() (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
#define IS_MASTER_BLOCK() (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)

#define BARRIER_MAGIC 0x80000000

namespace tl {

// Triggers a GPU trap for debugging
TL_DEVICE void trap() {
    asm("trap;\n");
}

// CTA-level memory fence
TL_DEVICE void memory_fence_cta() {
    asm volatile("fence.acq_rel.cta;\n"::: "memory");
}

// GPU-level memory fence
TL_DEVICE void memory_fence_gpu() {
    asm volatile("fence.acq_rel.gpu;\n"::: "memory");
}

// System-level memory fence
TL_DEVICE void memory_fence_sys() {
    asm volatile("fence.acq_rel.sys;\n"::: "memory");
}

// GPU-level load with acquire semantics
TL_DEVICE uint32_t ld_acquire_gpu_u32(const uint32_t *ptr) {
    uint32_t ret;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(ret) : "l"(ptr));
    return ret;
}

// GPU-level atomic add with release semantics
TL_DEVICE uint32_t atomic_add_release_gpu_u32(const uint32_t* ptr, uint32_t value) {
    uint32_t ret;
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;\n" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}

// Initialize a GPU barrier
template <const uint32_t kExpected>
TL_DEVICE void init_barrier_gpu(uint32_t* barrier) {
    if (IS_MASTER_BLOCK() && IS_MASTER_THREAD()) {
        *barrier = BARRIER_MAGIC - kExpected;
    }
    memory_fence_gpu();  // TODO: Is fence or sync needed here?
}

// Arrive at a GPU barrier (atomic increment)
TL_DEVICE void arrive_barrier_gpu(uint32_t* barrier) {
    __syncthreads();
    if (IS_MASTER_THREAD()) {
        atomic_add_release_gpu_u32(barrier, 1);
    }
}

// Wait at a GPU barrier until all expected blocks have arrived
TL_DEVICE void wait_barrier_gpu(uint32_t* barrier) {
    if (IS_MASTER_THREAD()) {
        uint32_t arrive = ld_acquire_gpu_u32(barrier);
        while (!(arrive & BARRIER_MAGIC)) {
            arrive = ld_acquire_gpu_u32(barrier);
        }
    }
    __syncthreads();
}

// Synchronize at a GPU barrier (arrive + wait)
TL_DEVICE void sync_barrier_gpu(uint32_t* barrier) {
    __syncthreads();
    if (IS_MASTER_THREAD()) {
        atomic_add_release_gpu_u32(barrier, 1);
        uint32_t arrive = ld_acquire_gpu_u32(barrier);
        while (arrive < BARRIER_MAGIC) {
            arrive = ld_acquire_gpu_u32(barrier);
        }
    }
    __syncthreads();
}

} // namespace tl