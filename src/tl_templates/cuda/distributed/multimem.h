#pragma once

#include "../common.h"

#if defined(__CUDA_ARCH__)

namespace tl {
namespace multimem {

template <class> inline constexpr bool dependent_false_v = false;

// Direct multimem instructions require SM90+ and PTX 8.1+ (CUDA 12.1+).
#if __CUDA_ARCH__ >= 900 &&                                                    \
    ((__CUDACC_VER_MAJOR__ > 12) ||                                            \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 1))

// PTX 8.2 added .acc::f32 for packed f16/bf16 load-reduce instructions.
#if (__CUDACC_VER_MAJOR__ > 12) ||                                             \
    (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 2)
#define TL_MULTIMEM_HAS_ACC_F32
#endif

enum class ReduceOp { ADD = 0, MIN = 1, MAX = 2 };

// Memory-ordering convention for this header:
//   * Data movement (LdReduce*, St*, Red*) uses `.relaxed.sys` for every dtype.
//     Cross-rank ordering comes from the barrier, i.e. the `Signal`/`SignalAdd`
//     release below paired with the acquire load in the wait path, so the data
//     instructions themselves only need system-scope coherence.
//   * Synchronization (Signal, SignalAdd) uses `.release.sys`.
// Keep new specializations on the same semantics as their f32 counterpart: a
// per-dtype mismatch here is silent, and the packed 16-bit paths are the ones
// the multimem all-reduce example exercises by default.

// === Per-instruction primitives (used by MultimemRewriter post-process) ===

// --- V1 scalar forms (used for predicated and unaligned tails) ---

template <ReduceOp op, typename DType> struct LdReduceV1 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(dependent_false_v<DType>,
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
    static_assert(dependent_false_v<DType>,
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
    static_assert(dependent_false_v<DType>,
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
    static_assert(dependent_false_v<DType>,
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

// --- StV4: 128-bit store to multicast address ---

template <typename DType> struct StV4 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(dependent_false_v<DType>,
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

// --- RedV4: 128-bit reduce into multicast address ---

template <ReduceOp op, typename DType> struct RedV4 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(dependent_false_v<DType>,
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

// === V2 variants (64-bit f32 vectors or one packed 16-bit pair) ===

template <ReduceOp op, typename DType> struct LdReduceV2 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(dependent_false_v<DType>,
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
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.acc::f32.f16x2 %0, "
                 "[%1];"
                 : "=r"(ret)
                 : "l"(mcast_ptr)
                 : "memory");
    *reinterpret_cast<uint32_t *>(dst) = ret;
  }
};

template <> struct LdReduceV2<ReduceOp::ADD, bfloat16_t> {
  TL_DEVICE static void run(void *dst, const void *mcast_ptr) {
    uint32_t ret;
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.acc::f32.bf16x2 "
                 "%0, [%1];"
                 : "=r"(ret)
                 : "l"(mcast_ptr)
                 : "memory");
    *reinterpret_cast<uint32_t *>(dst) = ret;
  }
};

#endif // TL_MULTIMEM_HAS_ACC_F32

template <typename DType> struct StV2 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(dependent_false_v<DType>,
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
    asm volatile("multimem.st.relaxed.sys.global.f16x2 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "r"(val)
                 : "memory");
  }
};

template <> struct StV2<bfloat16_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    uint32_t val = *reinterpret_cast<const uint32_t *>(src);
    asm volatile("multimem.st.relaxed.sys.global.bf16x2 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "r"(val)
                 : "memory");
  }
};

template <ReduceOp op, typename DType> struct RedV2 {
  TL_DEVICE static void run(void *, const void *) {
    static_assert(dependent_false_v<DType>,
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
    asm volatile("multimem.red.relaxed.sys.global.add.f16x2 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "r"(val)
                 : "memory");
  }
};

template <> struct RedV2<ReduceOp::ADD, bfloat16_t> {
  TL_DEVICE static void run(void *mcast_ptr, const void *src) {
    uint32_t val = *reinterpret_cast<const uint32_t *>(src);
    asm volatile("multimem.red.relaxed.sys.global.add.bf16x2 [%0], %1;"
                 :
                 : "l"(mcast_ptr), "r"(val)
                 : "memory");
  }
};

// === Thread-level signal write to multicast address ===

template <typename T> struct Signal {
  TL_DEVICE static void run(void *, T) {
    static_assert(dependent_false_v<T>,
                  "tl::multimem::Signal: unsupported type");
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
    static_assert(dependent_false_v<T>,
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

#undef TL_MULTIMEM_HAS_ACC_F32

#endif // SM90+ and CUDA 12.1+

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

#else // SM90+ and PTX 9.1 are unavailable

// Keep unsupported entry points parseable so ordinary direct-multimem kernels
// still compile. The dependent assertion fires only if lowering emits a bulk
// call for an unsupported target.
#define TL_MULTIMEM_UNSUPPORTED_BULK(name)                                     \
  template <typename Target = void>                                            \
  TL_DEVICE void name(void *, void *, uint32_t) {                              \
    static_assert(dependent_false_v<Target>,                                   \
                  "tl::multimem bulk operations require SM90+ and CUDA "       \
                  "Toolkit 13.1+ (PTX 9.1)");                                  \
  }

TL_MULTIMEM_UNSUPPORTED_BULK(cp_async_bulk)
TL_MULTIMEM_UNSUPPORTED_BULK(cp_reduce_async_bulk_add_f32)
TL_MULTIMEM_UNSUPPORTED_BULK(cp_reduce_async_bulk_add_f16)
TL_MULTIMEM_UNSUPPORTED_BULK(cp_reduce_async_bulk_add_bf16)
TL_MULTIMEM_UNSUPPORTED_BULK(cp_reduce_async_bulk_min_f16)
TL_MULTIMEM_UNSUPPORTED_BULK(cp_reduce_async_bulk_max_f16)
TL_MULTIMEM_UNSUPPORTED_BULK(cp_reduce_async_bulk_min_bf16)
TL_MULTIMEM_UNSUPPORTED_BULK(cp_reduce_async_bulk_max_bf16)

#undef TL_MULTIMEM_UNSUPPORTED_BULK

#endif // SM90+ and CUDA 13.1+

} // namespace multimem
} // namespace tl

#endif // defined(__CUDA_ARCH__)
