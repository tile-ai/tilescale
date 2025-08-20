#pragma once

#include "common.h"

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
TL_DEVICE int ld_acquire_gpu(const int *ptr) {
    int ret;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(ret) : "l"(ptr));
    return ret;
}

// System-level load with acquire semantics
TL_DEVICE uint64_t ld_acquire_sys(const uint64_t *ptr) {
    uint64_t ret;  // ? Why uint64_t?
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];\n" : "=l"(ret) : "l"(ptr));
    return ret;
}

// GPU-level atomic add with release semantics
TL_DEVICE int atomic_add_release_gpu(const int* ptr, int value) {
    int ret;
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;\n" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}

// System-level atomic add with acquire semantics
TL_DEVICE int atomic_add_acquire_sys(const int* ptr, int value) {
    int ret;
    asm volatile("atom.add.acquire.sys.global.s32 %0, [%1], %2;\n" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}

// Barrier blocks in a grid until expected number of blocks arrive
template <const int kExpected>
TL_DEVICE void barrier_blocks(int* bar) {
    __syncthreads();
    if (threadIdx.x + threadIdx.y + threadIdx.z == 0) {
        atomic_add_release_gpu(bar, 1);
        int arrive = ld_acquire_gpu(bar);
        while (arrive < kExpected) {
            arrive = ld_acquire_gpu(bar);
        }
    }
    __syncthreads();
}

} // namespace tl