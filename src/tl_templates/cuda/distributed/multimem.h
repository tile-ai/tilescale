#pragma once

#include "../common.h"

// Direct multimem instructions require SM90+ and PTX 8.1+ (CUDA 12.1+).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 &&                          \
    ((__CUDACC_VER_MAJOR__ > 12) ||                                            \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 1))

// PTX 8.2 added .acc::f32 for packed f16/bf16 load-reduce instructions.
#if (__CUDACC_VER_MAJOR__ > 12) ||                                             \
    (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 2)
#define TL_MULTIMEM_HAS_ACC_F32
#endif

#ifndef TL_ALWAYS_FALSE_V_DEFINED
#define TL_ALWAYS_FALSE_V_DEFINED
template <class> inline constexpr bool always_false_v = false;
#endif

namespace tl {
namespace multimem {

enum class ReduceOp { ADD = 0, MIN = 1, MAX = 2 };

// === Per-instruction primitives (used by MultimemRewriter post-process) ===

// --- V1 scalar forms (used for predicated and unaligned tails) ---

template <ReduceOp op, typename DType> struct LdReduceV1 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::LdReduceV1: unsupported dtype/op combination");
  }
};

template <> struct LdReduceV1<ReduceOp::ADD, float> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    uint32_t ret;
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];"
                 : "=r"(ret)
                 : "l"(mcast_ptr)
                 : "memory");
    *reinterpret_cast<uint32_t *>(dst) = ret;
  }
};

template <typename DType> struct StV1 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::StV1: unsupported dtype");
  }
};

template <> struct StV1<float> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    uint32_t val = *reinterpret_cast<const uint32_t *>(src);
    asm volatile("multimem.st.relaxed.sys.global.b32 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "r"(val)
                 : "memory");
  }
};

template <ReduceOp op, typename DType> struct RedV1 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::RedV1: unsupported dtype/op combination");
  }
};

template <> struct RedV1<ReduceOp::ADD, float> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    uint32_t val = *reinterpret_cast<const uint32_t *>(src);
    asm volatile("multimem.red.relaxed.sys.global.add.f32 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "r"(val)
                 : "memory");
  }
};

// --- LdReduceV4: 128-bit load-reduce from multicast address ---

template <ReduceOp op, typename DType> struct LdReduceV4 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::LdReduceV4: unsupported dtype/op/toolkit "
                  "combination");
  }
};

template <> struct LdReduceV4<ReduceOp::ADD, float> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    int4 ret;
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0, %1, %2, %3}, "
        "[%4];"
        : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
        : "l"(mcast_ptr)
        : "memory");
    *reinterpret_cast<int4 *>(dst) = ret;
  }
};

#ifdef TL_MULTIMEM_HAS_ACC_F32

template <> struct LdReduceV4<ReduceOp::ADD, half_t> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    uint32_t *dst_u32 = reinterpret_cast<uint32_t *>(dst);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 2; i++) {
      uint32_t ret;
      asm volatile("multimem.ld_reduce.weak.global.add.acc::f32.f16x2 %0, [%1];"
                   : "=r"(ret)
                   : "l"(mc + i * 4)
                   : "memory");
      dst_u32[i] = ret;
    }
  }
};

template <> struct LdReduceV4<ReduceOp::ADD, bfloat16_t> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    uint32_t *dst_u32 = reinterpret_cast<uint32_t *>(dst);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 2; i++) {
      uint32_t ret;
      asm volatile(
          "multimem.ld_reduce.weak.global.add.acc::f32.bf16x2 %0, [%1];"
          : "=r"(ret)
          : "l"(mc + i * 4)
          : "memory");
      dst_u32[i] = ret;
    }
  }
};

#endif // TL_MULTIMEM_HAS_ACC_F32

// --- StV4: 128-bit store to multicast address ---

template <typename DType> struct StV4 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::StV4: unsupported dtype");
  }
};

template <> struct StV4<float> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    float4 val = *reinterpret_cast<const float4 *>(src);
    asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(mcast_ptr), "f"(val.x), "f"(val.y), "f"(val.z),
                   "f"(val.w)
                 : "memory");
  }
};

template <> struct StV4<half_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const uint32_t *src_u32 = reinterpret_cast<const uint32_t *>(src);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 2; i++) {
      asm volatile("multimem.st.weak.global.f16x2 [%0], %1;"
                   :
                   : "l"(mc + i * 4), "r"(src_u32[i])
                   : "memory");
    }
  }
};

template <> struct StV4<bfloat16_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const uint32_t *src_u32 = reinterpret_cast<const uint32_t *>(src);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 2; i++) {
      asm volatile("multimem.st.weak.global.bf16x2 [%0], %1;"
                   :
                   : "l"(mc + i * 4), "r"(src_u32[i])
                   : "memory");
    }
  }
};

// --- RedV4: 128-bit reduce into multicast address ---

template <ReduceOp op, typename DType> struct RedV4 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::RedV4: unsupported dtype/op combination");
  }
};

template <> struct RedV4<ReduceOp::ADD, float> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    int4 val = *reinterpret_cast<const int4 *>(src);
    asm volatile(
        "multimem.red.relaxed.sys.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(mcast_ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
        : "memory");
  }
};

template <> struct RedV4<ReduceOp::ADD, half_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const uint32_t *src_u32 = reinterpret_cast<const uint32_t *>(src);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 2; i++) {
      asm volatile("multimem.red.release.sys.global.add.f16x2 [%0], %1;"
                   :
                   : "l"(mc + i * 4), "r"(src_u32[i])
                   : "memory");
    }
  }
};

template <> struct RedV4<ReduceOp::ADD, bfloat16_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const uint32_t *src_u32 = reinterpret_cast<const uint32_t *>(src);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 2; i++) {
      asm volatile("multimem.red.release.sys.global.add.bf16x2 [%0], %1;"
                   :
                   : "l"(mc + i * 4), "r"(src_u32[i])
                   : "memory");
    }
  }
};

// === V2 variants (64-bit = 2×f32, implemented as 2 scalar ops) ===

template <ReduceOp op, typename DType> struct LdReduceV2 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::LdReduceV2: unsupported dtype/op/toolkit "
                  "combination");
  }
};

template <> struct LdReduceV2<ReduceOp::ADD, float> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    int2 ret;
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v2.f32 {%0, %1}, [%2];"
        : "=r"(ret.x), "=r"(ret.y)
        : "l"(mcast_ptr)
        : "memory");
    *reinterpret_cast<int2 *>(dst) = ret;
  }
};

#ifdef TL_MULTIMEM_HAS_ACC_F32

template <> struct LdReduceV2<ReduceOp::ADD, half_t> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    uint32_t ret;
    asm volatile("multimem.ld_reduce.weak.global.add.acc::f32.f16x2 %0, [%1];"
                 : "=r"(ret)
                 : "l"(mcast_ptr)
                 : "memory");
    *reinterpret_cast<uint32_t *>(dst) = ret;
  }
};

template <> struct LdReduceV2<ReduceOp::ADD, bfloat16_t> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    uint32_t ret;
    asm volatile("multimem.ld_reduce.weak.global.add.acc::f32.bf16x2 %0, [%1];"
                 : "=r"(ret)
                 : "l"(mcast_ptr)
                 : "memory");
    *reinterpret_cast<uint32_t *>(dst) = ret;
  }
};

#endif // TL_MULTIMEM_HAS_ACC_F32

template <typename DType> struct StV2 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::StV2: unsupported dtype");
  }
};

template <> struct StV2<float> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    float2 val = *reinterpret_cast<const float2 *>(src);
    asm volatile("multimem.st.relaxed.sys.global.v2.f32 [%0], {%1, %2};"
                 :
                 : "l"(mcast_ptr), "f"(val.x), "f"(val.y)
                 : "memory");
  }
};

template <> struct StV2<half_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    uint32_t val = *reinterpret_cast<const uint32_t *>(src);
    asm volatile("multimem.st.weak.global.f16x2 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "r"(val)
                 : "memory");
  }
};

template <> struct StV2<bfloat16_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    uint32_t val = *reinterpret_cast<const uint32_t *>(src);
    asm volatile("multimem.st.weak.global.bf16x2 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "r"(val)
                 : "memory");
  }
};

template <ReduceOp op, typename DType> struct RedV2 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::RedV2: unsupported dtype/op");
  }
};

template <> struct RedV2<ReduceOp::ADD, float> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const float *src_f = reinterpret_cast<const float *>(src);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 2; i++) {
      unsigned val = __float_as_uint(src_f[i]);
      asm volatile("multimem.red.relaxed.sys.global.add.f32 [%0], %1;"
                   :
                   : "l"(mc + i * 4), "r"(val)
                   : "memory");
    }
  }
};

template <> struct RedV2<ReduceOp::ADD, half_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    uint32_t val = *reinterpret_cast<const uint32_t *>(src);
    asm volatile("multimem.red.release.sys.global.add.f16x2 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "r"(val)
                 : "memory");
  }
};

template <> struct RedV2<ReduceOp::ADD, bfloat16_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    uint32_t val = *reinterpret_cast<const uint32_t *>(src);
    asm volatile("multimem.red.release.sys.global.add.bf16x2 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "r"(val)
                 : "memory");
  }
};

// === V8 variants (128-bit = 8×fp16/bf16, implemented as 4 packed x2 ops) ===

template <ReduceOp op, typename DType> struct LdReduceV8 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::LdReduceV8: unsupported dtype/op/toolkit "
                  "combination");
  }
};

#ifdef TL_MULTIMEM_HAS_ACC_F32

template <> struct LdReduceV8<ReduceOp::ADD, half_t> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    uint32_t *dst_u32 = reinterpret_cast<uint32_t *>(dst);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      uint32_t ret;
      asm volatile("multimem.ld_reduce.weak.global.add.acc::f32.f16x2 %0, [%1];"
                   : "=r"(ret)
                   : "l"(mc + i * 4)
                   : "memory");
      dst_u32[i] = ret;
    }
  }
};

template <> struct LdReduceV8<ReduceOp::ADD, bfloat16_t> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    uint32_t *dst_u32 = reinterpret_cast<uint32_t *>(dst);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      uint32_t ret;
      asm volatile(
          "multimem.ld_reduce.weak.global.add.acc::f32.bf16x2 %0, [%1];"
          : "=r"(ret)
          : "l"(mc + i * 4)
          : "memory");
      dst_u32[i] = ret;
    }
  }
};

#endif // TL_MULTIMEM_HAS_ACC_F32

template <typename DType> struct StV8 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::StV8: unsupported dtype");
  }
};

template <> struct StV8<half_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const uint32_t *src_u32 = reinterpret_cast<const uint32_t *>(src);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      asm volatile("multimem.st.weak.global.f16x2 [%0], %1;"
                   :
                   : "l"(mc + i * 4), "r"(src_u32[i])
                   : "memory");
    }
  }
};

template <> struct StV8<bfloat16_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const uint32_t *src_u32 = reinterpret_cast<const uint32_t *>(src);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      asm volatile("multimem.st.weak.global.bf16x2 [%0], %1;"
                   :
                   : "l"(mc + i * 4), "r"(src_u32[i])
                   : "memory");
    }
  }
};

template <ReduceOp op, typename DType> struct RedV8 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::RedV8: unsupported dtype/op");
  }
};

template <> struct RedV8<ReduceOp::ADD, half_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const uint32_t *src_u32 = reinterpret_cast<const uint32_t *>(src);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      asm volatile("multimem.red.release.sys.global.add.f16x2 [%0], %1;"
                   :
                   : "l"(mc + i * 4), "r"(src_u32[i])
                   : "memory");
    }
  }
};

template <> struct RedV8<ReduceOp::ADD, bfloat16_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const uint32_t *src_u32 = reinterpret_cast<const uint32_t *>(src);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      asm volatile("multimem.red.release.sys.global.add.bf16x2 [%0], %1;"
                   :
                   : "l"(mc + i * 4), "r"(src_u32[i])
                   : "memory");
    }
  }
};

// === Thread-level signal write to multicast address ===

template <typename T> struct Signal {
  TL_DEVICE static void run(void *, T) {
    static_assert(always_false_v<T>, "tl::multimem::Signal: unsupported type");
  }
};
template <> struct Signal<uint32_t> {
  TL_DEVICE static void run(void *mcast_ptr, uint32_t val) {
    asm volatile("multimem.st.release.sys.global.u32 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "r"(val)
                 : "memory");
  }
};
template <> struct Signal<uint64_t> {
  TL_DEVICE static void run(void *mcast_ptr, uint64_t val) {
    asm volatile("multimem.st.release.sys.global.u64 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "l"(val)
                 : "memory");
  }
};

template <typename T> struct SignalAdd {
  TL_DEVICE static void run(void *, T) {
    static_assert(always_false_v<T>,
                  "tl::multimem::SignalAdd: unsupported type");
  }
};
template <> struct SignalAdd<uint32_t> {
  TL_DEVICE static void run(void *mcast_ptr, uint32_t val) {
    asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "r"(val)
                 : "memory");
  }
};
template <> struct SignalAdd<int32_t> {
  TL_DEVICE static void run(void *mcast_ptr, int32_t val) {
    asm volatile("multimem.red.release.sys.global.add.s32 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "r"(val)
                 : "memory");
  }
};
template <> struct SignalAdd<uint64_t> {
  TL_DEVICE static void run(void *mcast_ptr, uint64_t val) {
    asm volatile("multimem.red.release.sys.global.add.u64 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "l"(val)
                 : "memory");
  }
};

// === Bulk async TMA-to-multicast (SM90+ / PTX 9.1+ / CUDA 13.1+) ===
// Both: shared::cta → global(mcast), bulk_group completion

#if __CUDA_ARCH__ >= 900 &&                                                    \
    (__CUDACC_VER_MAJOR__ > 13 ||                                              \
     (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1))

TL_DEVICE void cp_async_bulk(void *mcast_global, void *smem, uint32_t size) {
  uint32_t smem_int = smem_ptr_to_uint(smem);
  asm volatile(
      "multimem.cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
      :
      : "l"(mcast_global), "r"(smem_int), "r"(size)
      : "memory");
}

TL_DEVICE void cp_reduce_async_bulk_add_f32(void *mcast_global, void *smem,
                                            uint32_t size) {
  uint32_t smem_int = smem_ptr_to_uint(smem);
  asm volatile(
      "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 "
      "[%0], [%1], %2;\n"
      :
      : "l"(mcast_global), "r"(smem_int), "r"(size)
      : "memory");
}

TL_DEVICE void cp_reduce_async_bulk_add_f16(void *mcast_global, void *smem,
                                            uint32_t size) {
  uint32_t smem_int = smem_ptr_to_uint(smem);
  asm volatile("multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group."
               "add.noftz.f16 "
               "[%0], [%1], %2;\n"
               :
               : "l"(mcast_global), "r"(smem_int), "r"(size)
               : "memory");
}

TL_DEVICE void cp_reduce_async_bulk_add_bf16(void *mcast_global, void *smem,
                                             uint32_t size) {
  uint32_t smem_int = smem_ptr_to_uint(smem);
  asm volatile("multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group."
               "add.noftz.bf16 "
               "[%0], [%1], %2;\n"
               :
               : "l"(mcast_global), "r"(smem_int), "r"(size)
               : "memory");
}

TL_DEVICE void cp_reduce_async_bulk_min_f16(void *mcast_global, void *smem,
                                            uint32_t size) {
  uint32_t smem_int = smem_ptr_to_uint(smem);
  asm volatile(
      "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.min.f16 "
      "[%0], [%1], %2;\n"
      :
      : "l"(mcast_global), "r"(smem_int), "r"(size)
      : "memory");
}

TL_DEVICE void cp_reduce_async_bulk_max_f16(void *mcast_global, void *smem,
                                            uint32_t size) {
  uint32_t smem_int = smem_ptr_to_uint(smem);
  asm volatile(
      "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.max.f16 "
      "[%0], [%1], %2;\n"
      :
      : "l"(mcast_global), "r"(smem_int), "r"(size)
      : "memory");
}

TL_DEVICE void cp_reduce_async_bulk_min_bf16(void *mcast_global, void *smem,
                                             uint32_t size) {
  uint32_t smem_int = smem_ptr_to_uint(smem);
  asm volatile(
      "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.min.bf16 "
      "[%0], [%1], %2;\n"
      :
      : "l"(mcast_global), "r"(smem_int), "r"(size)
      : "memory");
}

TL_DEVICE void cp_reduce_async_bulk_max_bf16(void *mcast_global, void *smem,
                                             uint32_t size) {
  uint32_t smem_int = smem_ptr_to_uint(smem);
  asm volatile(
      "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.max.bf16 "
      "[%0], [%1], %2;\n"
      :
      : "l"(mcast_global), "r"(smem_int), "r"(size)
      : "memory");
}

#else // PTX 9.1 not available — unconditional trap

TL_DEVICE void cp_async_bulk(void *mcast_global, void *smem, uint32_t size) {
  (void)mcast_global;
  (void)smem;
  (void)size;
  asm("trap;");
}
TL_DEVICE void cp_reduce_async_bulk_add_f32(void *mcast_global, void *smem,
                                            uint32_t size) {
  (void)mcast_global;
  (void)smem;
  (void)size;
  asm("trap;");
}
TL_DEVICE void cp_reduce_async_bulk_add_f16(void *mcast_global, void *smem,
                                            uint32_t size) {
  (void)mcast_global;
  (void)smem;
  (void)size;
  asm("trap;");
}
TL_DEVICE void cp_reduce_async_bulk_min_f16(void *mcast_global, void *smem,
                                            uint32_t size) {
  (void)mcast_global;
  (void)smem;
  (void)size;
  asm("trap;");
}
TL_DEVICE void cp_reduce_async_bulk_max_f16(void *mcast_global, void *smem,
                                            uint32_t size) {
  (void)mcast_global;
  (void)smem;
  (void)size;
  asm("trap;");
}
TL_DEVICE void cp_reduce_async_bulk_min_bf16(void *mcast_global, void *smem,
                                             uint32_t size) {
  (void)mcast_global;
  (void)smem;
  (void)size;
  asm("trap;");
}
TL_DEVICE void cp_reduce_async_bulk_max_bf16(void *mcast_global, void *smem,
                                             uint32_t size) {
  (void)mcast_global;
  (void)smem;
  (void)size;
  asm("trap;");
}
TL_DEVICE void cp_reduce_async_bulk_add_bf16(void *mcast_global, void *smem,
                                             uint32_t size) {
  (void)mcast_global;
  (void)smem;
  (void)size;
  asm("trap;");
}

#endif // SM90+ and CUDA 13.1+

#undef TL_MULTIMEM_HAS_ACC_F32

} // namespace multimem
} // namespace tl

#endif // SM90+ and CUDA 12.1+
