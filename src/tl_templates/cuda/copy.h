#pragma once

#include "common.h"

#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 900))
#include "copy_sm90.h"
#endif

namespace tl {

TL_DEVICE void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N> TL_DEVICE void cp_async_wait() {
  if constexpr (N == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
  }
}

template <int N>
TL_DEVICE void cp_async_gs(void const *const smem_addr, void *global_ptr) {
  static_assert(N == 16 || N == 8 || N == 4);
  unsigned int addr = smem_ptr_to_uint(smem_addr);
  if constexpr (N == 16) {
    asm volatile(
#if TL_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
        "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
        ::"r"(addr),
        "l"((void *)(global_ptr)), "n"(N));
  } else {
    asm volatile(
#if TL_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;"
#else
        "cp.async.ca.shared.global [%0], [%1], %2;"
#endif
        ::"r"(addr),
        "l"((void *)(global_ptr)), "n"(N));
  }
}

template <int N>
TL_DEVICE void cp_async_gs_conditional(void const *const smem_addr,
                                       void *global_ptr, bool cond) {
  static_assert(N == 16 || N == 8 || N == 4);
  int bytes = cond ? N : 0;
  unsigned int addr = smem_ptr_to_uint(smem_addr);
  if constexpr (N == 16) {
    asm volatile(
#if TL_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
#else
        "cp.async.cg.shared.global [%0], [%1], %2, %3;"
#endif
        ::"r"(addr),
        "l"((void *)(global_ptr)), "n"(N), "r"(bytes));
  } else {
    asm volatile(
#if TL_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;"
#else
        "cp.async.ca.shared.global [%0], [%1], %2, %3;"
#endif
        ::"r"(addr),
        "l"((void *)(global_ptr)), "n"(N), "r"(bytes));
  }
}

template <int kBytes>
struct VecInt {};
template<> struct VecInt<1> { using vec_t = int8_t; };
template<> struct VecInt<2> { using vec_t = int16_t; };
template<> struct VecInt<4> { using vec_t = int; };
template<> struct VecInt<8> { using vec_t = int64_t; };
template<> struct VecInt<16> { using vec_t = int4; };

#define LD_NC_FUNC "ld.volatile.global"
#define ST_NA_FUNC "st.global"

template <typename dtype_t>
TL_DEVICE dtype_t ld_nc_global(const dtype_t *ptr) {
    auto ret = ld_nc_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr));
    return *reinterpret_cast<dtype_t*>(&ret);
}

template <>
TL_DEVICE uint8_t ld_nc_global(const uint8_t *ptr) {
    uint16_t ret;
    // NOTES: we must use `uint16_t` as inline ASM does not support 8-bit constraint letter (`h` below means unsigned 16-bit)
    asm volatile(LD_NC_FUNC ".u8 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return static_cast<uint8_t>(ret);
}

template <>
TL_DEVICE int ld_nc_global(const int *ptr) {
    int ret;
    asm volatile(LD_NC_FUNC ".s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

template <>
TL_DEVICE int64_t ld_nc_global(const int64_t *ptr) {
    int64_t ret;
    asm volatile(LD_NC_FUNC ".s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

template <>
TL_DEVICE float ld_nc_global(const float *ptr) {
    float ret;
    asm volatile(LD_NC_FUNC ".f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

template <>
TL_DEVICE int2 ld_nc_global(const int2 *ptr) {
    int2 ret;
    asm volatile(LD_NC_FUNC ".v2.s32 {%0, %1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l"(ptr));
    return ret;
}

template <>
TL_DEVICE int4 ld_nc_global(const int4 *ptr) {
    int4 ret;
    asm volatile(LD_NC_FUNC ".v4.s32 {%0, %1, %2, %3}, [%4];"
            : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
    return ret;
}

template <typename dtype_t>
TL_DEVICE void st_na_global(const dtype_t *ptr, const dtype_t& value) {
    st_na_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr),
                 *reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(&value));
}

template <>
TL_DEVICE void st_na_global(const int *ptr, const int& value) {
    asm volatile(ST_NA_FUNC ".s32 [%0], %1;" ::"l"(ptr), "r"(value));
}

template <>
TL_DEVICE void st_na_global(const int64_t *ptr, const int64_t& value) {
    asm volatile(ST_NA_FUNC ".s64 [%0], %1;" ::"l"(ptr), "l"(value));
}

template <>
TL_DEVICE void st_na_global(const float *ptr, const float& value) {
    asm volatile(ST_NA_FUNC ".f32 [%0], %1;" ::"l"(ptr), "f"(value));
}

template <>
TL_DEVICE void st_na_global(const int4 *ptr, const int4& value) {
    asm volatile(ST_NA_FUNC ".v4.s32 [%0], {%1, %2, %3, %4};"
            ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
}

#define LD_FUNC(ptr) ld_nc_global(ptr)
#define ST_FUNC(prt, value) st_na_global(prt, value)

template <int N, int UNROLL_FACTOR, typename dtype_t>
TL_DEVICE void cp_unrolled(dtype_t const* const dst_addr, dtype_t const* const src_addr) {
  int lane_id;
  asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
  constexpr int kLoopStride = 32 * (UNROLL_FACTOR);
  typename std::remove_reference<decltype(LD_FUNC((src_addr) + 0))>::type unrolled_values[(UNROLL_FACTOR)];
  auto __src = (src_addr);
  auto __dst = (dst_addr);
  for (int __i = (lane_id); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) {
    _Pragma("unroll")
    for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j)
      unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32);
    _Pragma("unroll")
    for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j)
      ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]);
  }
  for (int __i = ((N) / kLoopStride) * kLoopStride + (lane_id); __i < (N); __i += 32)
    ST_FUNC(__dst + __i, LD_FUNC(__src + __i));
}

} // namespace tl
