#pragma once

#include "common.h"

#define IS_MASTER_THREAD()                                                     \
  (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
#define IS_MASTER_BLOCK()                                                      \
  (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)

#define BARRIER_MAGIC 0x80000000

namespace tl {

// Triggers a GPU trap for debugging
TL_DEVICE void trap() { asm("trap;\n"); }

// CTA-level memory fence
TL_DEVICE void memory_fence_cta() {
  asm volatile("fence.acq_rel.cta;\n" ::: "memory");
}

// GPU-level memory fence
TL_DEVICE void memory_fence_gpu() {
  asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
}

// System-level memory fence
TL_DEVICE void memory_fence_sys() {
  asm volatile("fence.acq_rel.sys;\n" ::: "memory");
}

// GPU-level load with acquire semantics
TL_DEVICE uint32_t ld_acquire_gpu_u32(const uint32_t *ptr) {
  uint32_t ret;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(ret) : "l"(ptr));
  return ret;
}

// GPU-level atomic add with release semantics
TL_DEVICE uint32_t atomic_add_release_gpu_u32(const uint32_t *ptr,
                                              uint32_t value) {
  uint32_t ret;
  asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;\n"
               : "=r"(ret)
               : "l"(ptr), "r"(value));
  return ret;
}

// System-level atomic load with acquire semantics
TL_DEVICE int atomic_load_acquire_sys_s32(const int *ptr) {
  int ret;
  asm volatile("atom.load.acquire.sys.global.s32 %0, [%1];\n"
               : "=r"(ret)
               : "l"(ptr));
  return ret;
}

TL_DEVICE int ld_volatile_global_s32(const int *ptr) {
  int ret;
  asm volatile("ld.volatile.global.s32 %0, [%1];\n" : "=r"(ret) : "l"(ptr));
  return ret;
}

// Initialize a GPU barrier
template <const uint32_t kExpected>
TL_DEVICE void init_barrier_gpu(uint32_t *barrier) {
  if (IS_MASTER_BLOCK() && IS_MASTER_THREAD()) {
    *barrier = BARRIER_MAGIC - kExpected;
  }
  memory_fence_gpu(); // TODO: Is fence or sync needed here?
}

// Arrive at a GPU barrier (atomic increment)
TL_DEVICE void arrive_barrier_gpu(uint32_t *barrier) {
  memory_fence_gpu();
  if (IS_MASTER_THREAD()) {
    atomic_add_release_gpu_u32(barrier, 1);
  }
}

// Wait at a GPU barrier until all expected blocks have arrived
TL_DEVICE void wait_barrier_gpu(uint32_t *barrier) {
  if (IS_MASTER_THREAD()) {
    uint32_t arrive = ld_acquire_gpu_u32(barrier);
    while (!(arrive & BARRIER_MAGIC)) {
      arrive = ld_acquire_gpu_u32(barrier);
    }
  }
  __syncthreads();
}

// Synchronize at a GPU barrier (arrive + wait)
TL_DEVICE void sync_barrier_gpu(uint32_t *barrier) {
  memory_fence_gpu();
  if (IS_MASTER_THREAD()) {
    atomic_add_release_gpu_u32(barrier, 1);
    uint32_t arrive = ld_acquire_gpu_u32(barrier);
    while (arrive < BARRIER_MAGIC) {
      arrive = ld_acquire_gpu_u32(barrier);
    }
  }
  __syncthreads();
}

// Synchronize all blocks at a system-level barrier
// TODO(wt): Add sync-only option and timeout handling
TL_DEVICE void barrier_all_blocks_sys(int offset, // &barrier - base
                                      int rank, int num_ranks) {
  // Gather ptrs to barriers on all ranks via metadata
  uint64_t barrier_ptrs[num_ranks];
  for (int i = 0; i < num_ranks; i++) {
    barrier_ptrs[i] = get_remote_base_ptr(i) + offset;
  }

  __syncthreads();
  memory_fence_sys();

  int tid = threadIdx.x;
  if (tid < num_ranks) {
    atomicAdd_system(reinterpret_cast<int32_t *>(barrier_ptrs[rank]) + tid, 1);
    atomicAdd_system(reinterpret_cast<int32_t *>(barrier_ptrs[tid]) + rank, -1);
  }

  while (true) {
    int value =
        tid < num_ranks ? ld_volatile_global_s32(reinterpret_cast<int32_t *>(barrier_ptrs[rank]) + tid) : 0;
    if (__all_sync(0xffffffff, value <= 0)) {
      break;
    }
  }
  __syncthreads();
}

} // namespace tl