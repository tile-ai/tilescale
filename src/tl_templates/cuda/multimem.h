#pragma once

#include "common.h"

// multimem instructions require SM 90+ (Hopper) and CUDA 12.0+
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && __CUDACC_VER_MAJOR__ >= 12

#ifndef TL_ALWAYS_FALSE_V_DEFINED
#define TL_ALWAYS_FALSE_V_DEFINED
template <class> inline constexpr bool always_false_v = false;
#endif

namespace tl {
namespace multimem {

enum class ReduceOp { ADD = 0, MIN = 1, MAX = 2 };

// === Per-instruction primitives (used by MultimemRewriter post-process) ===

// --- LdReduceV4: 128-bit load-reduce from multicast address ---

template <ReduceOp op, typename DType> struct LdReduceV4 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::LdReduceV4: unsupported dtype/op combination");
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

template <> struct LdReduceV4<ReduceOp::MIN, float> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    int4 ret;
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.min.v4.f32 {%0, %1, %2, %3}, "
        "[%4];"
        : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
        : "l"(mcast_ptr)
        : "memory");
    *reinterpret_cast<int4 *>(dst) = ret;
  }
};

template <> struct LdReduceV4<ReduceOp::MAX, float> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    int4 ret;
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.max.v4.f32 {%0, %1, %2, %3}, "
        "[%4];"
        : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
        : "l"(mcast_ptr)
        : "memory");
    *reinterpret_cast<int4 *>(dst) = ret;
  }
};

// --- StV4: 128-bit store to multicast address ---

template <typename DType> struct StV4 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::StV4: unsupported dtype");
  }
};

template <> struct StV4<float> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    int4 val = *reinterpret_cast<const int4 *>(src);
    asm volatile("multimem.st.relaxed.sys.global.v4.b32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(mcast_ptr), "r"(val.x), "r"(val.y), "r"(val.z),
                   "r"(val.w)
                 : "memory");
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

template <> struct RedV4<ReduceOp::MIN, float> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    // multimem.red min not directly available as v4; use scalar fallback
    const float *src_f = reinterpret_cast<const float *>(src);
    const char *mc_bytes = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      unsigned val = __float_as_uint(src_f[i]);
      asm volatile("multimem.red.relaxed.sys.global.min.f32 [%0], %1;"
                   :
                   : "l"(mc_bytes + i * 4), "r"(val)
                   : "memory");
    }
  }
};

template <> struct RedV4<ReduceOp::MAX, float> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const float *src_f = reinterpret_cast<const float *>(src);
    const char *mc_bytes = reinterpret_cast<const char *>(mcast_ptr);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      unsigned val = __float_as_uint(src_f[i]);
      asm volatile("multimem.red.relaxed.sys.global.max.f32 [%0], %1;"
                   :
                   : "l"(mc_bytes + i * 4), "r"(val)
                   : "memory");
    }
  }
};

// === V2 variants (64-bit = 2×f32, implemented as 2 scalar ops) ===

template <ReduceOp op, typename DType> struct LdReduceV2 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>,
                  "tl::multimem::LdReduceV2: unsupported dtype/op");
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

template <> struct LdReduceV2<ReduceOp::MIN, float> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    int2 ret;
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.min.v2.f32 {%0, %1}, [%2];"
        : "=r"(ret.x), "=r"(ret.y)
        : "l"(mcast_ptr)
        : "memory");
    *reinterpret_cast<int2 *>(dst) = ret;
  }
};

template <> struct LdReduceV2<ReduceOp::MAX, float> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    int2 ret;
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.max.v2.f32 {%0, %1}, [%2];"
        : "=r"(ret.x), "=r"(ret.y)
        : "l"(mcast_ptr)
        : "memory");
    *reinterpret_cast<int2 *>(dst) = ret;
  }
};

template <typename DType> struct StV2 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(always_false_v<DType>, "tl::multimem::StV2: unsupported dtype");
  }
};

template <> struct StV2<float> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    int2 val = *reinterpret_cast<const int2 *>(src);
    asm volatile("multimem.st.relaxed.sys.global.v2.b32 [%0], {%1, %2};"
                 :
                 : "l"(mcast_ptr), "r"(val.x), "r"(val.y)
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

template <> struct RedV2<ReduceOp::MIN, float> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const float *src_f = reinterpret_cast<const float *>(src);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
    #pragma unroll
    for (int i = 0; i < 2; i++) {
      unsigned val = __float_as_uint(src_f[i]);
      asm volatile("multimem.red.relaxed.sys.global.min.f32 [%0], %1;"
                   :
                   : "l"(mc + i * 4), "r"(val)
                   : "memory");
    }
  }
};

template <> struct RedV2<ReduceOp::MAX, float> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    const float *src_f = reinterpret_cast<const float *>(src);
    const char *mc = reinterpret_cast<const char *>(mcast_ptr);
    #pragma unroll
    for (int i = 0; i < 2; i++) {
      unsigned val = __float_as_uint(src_f[i]);
      asm volatile("multimem.red.relaxed.sys.global.max.f32 [%0], %1;"
                   :
                   : "l"(mc + i * 4), "r"(val)
                   : "memory");
    }
  }
};

} // namespace multimem
} // namespace tl

#endif // __CUDA_ARCH__ >= 900 && __CUDACC_VER_MAJOR__ >= 12
