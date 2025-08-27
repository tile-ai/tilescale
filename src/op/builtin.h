/*!
 * \file tl/op/builtin.h
 * \brief Builtin intrinsics.
 *
 */

#ifndef TVM_TL_OP_BUILTIN_H_
#define TVM_TL_OP_BUILTIN_H_

#include "op.h"
#include <tvm/ir/transform.h>

namespace tvm {
namespace tl {

namespace attr {
static constexpr const char *kPaddingMap = "padding_map";
static constexpr const char *kWarpSpecializationScope =
    "kWarpSpecializationScope";
} // namespace attr

static constexpr const char *kDebugMergeSharedMemoryAllocations =
    "tl.debug_merge_shared_memory_allocations";
static constexpr const char *kDisableTMALower = "tl.disable_tma_lower";
static constexpr const char *kDisableSafeMemoryLegalize =
    "tl.disable_safe_memory_legalize";
static constexpr const char *kDisableWarpSpecialized =
    "tl.disable_warp_specialized";
static constexpr const char *kConfigIndexBitwidth = "tl.config_index_bitwidth";
static constexpr const char *kEnableAggressiveSharedMemoryMerge =
    "tl.enable_aggressive_shared_memory_merge";
static constexpr const char *kDisableRDC = "tl.disable_rdc";
static constexpr const char *kDisableFastMath = "tl.disable_fast_math";
static constexpr const char *kPtxasRegisterUsageLevel =
    "tl.ptxas_register_usage_level";
static constexpr const char *kEnablePTXASVerboseOutput =
    "tl.enable_ptxas_verbose_output";
static constexpr const char *kDisableShuffleElect = "tl.disable_shuffle_elect";
/*!
 * \brief Whether to disable dynamic tail split
 *
 * kDisableDynamicTailSplit = "tl.disable_dynamic_tail_split"
 *
 */
static constexpr const char *kDisableDynamicTailSplit =
    "tl.disable_dynamic_tail_split";

/*!
 * \brief The size of the vectorized dimension in buffer, designed by user
 *
 * For example, if the vectorized dimension is 128 bits and the dtype of buffer
 * A[m, k] is float16, the size of the vectorized dimension (i.e. k) in buffer A
 * should be divisible by 8 (8 = 128 / 16).
 *
 * kDynamicAlignment = "tl.dynamic_alignment"
 *
 */
static constexpr const char *kDynamicAlignment = "tl.dynamic_alignment";

/*!
 * \brief Get the type of the CUDA tensor map
 *
 * DataType cuTensorMapType()
 *
 */
DataType cuTensorMapType();

/*!
 * \brief tvm intrinsics for TMADescriptor creation for tiled load
 *
 * CuTensorMap* create_tma_descriptor(data_type, rank, global_addr,
 * global_shape..., global_stride..., smem_box..., smem_stride..., interleave,
 * swizzle, l2_promotion, oob_fill)
 *
 */
TVM_DLL const Op &create_tma_descriptor();

/*!
 * \brief tvm intrinsics for TMADescriptor creation for image to column load
 *
 * CuTensorMap* create_tma_im2col_descriptor(data_type, rank, global_addr,
 * global_shape..., global_stride..., elem_stride..., lower_corner...,
 * upper_corner..., smme_box_pixel, smem_box_channel, interleave, swizzle,
 * l2_promotion, oob_fill)
 *
 */
TVM_DLL const Op &create_tma_im2col_descriptor();

/*!
 * \brief Create a list of mbarrier with num_threads
 *
 * create_list_of_mbarrier(num_threads0, num_threads1, ...)
 *
 */
TVM_DLL const Op &create_list_of_mbarrier();

/*!
 * \brief Get the mbarrier with barrier_id
 *
 * int64_t* GetMBarrier(barrier_id)
 *
 */
TVM_DLL const Op &get_mbarrier();

/*!
 * \brief tvm intrinsics for loading data from global tensor descriptor to
 * shared memory
 *
 * tma_load(descriptor, mbarrier, smem_data, coord_0, coord_1, ...)
 *
 */
TVM_DLL const Op &tma_load();

/*!
 * \brief tvm intrinsics for loading image from global tensor to columns in
 * shared memory
 *
 * tma_load(descriptor, mbarrier, smem_data, coord_0, coord_1, ...,
 * image_offset, ...)
 *
 */
TVM_DLL const Op &tma_load_im2col();

/*!
 * \brief tvm intrinsics for storing data from shared memory to global tensor
 * descriptor
 *
 * tma_store(descriptor, smem_data, coord_0, coord_1, ...)
 *
 */
TVM_DLL const Op &tma_store();

/*!
 * \brief tvm intrinsics for mbarrier wait with parity bit
 *
 * mbarrier_wait_parity(mbarrier, parity)
 *
 */
TVM_DLL const Op &mbarrier_wait_parity();

/*!
 * \brief tvm intrinsics for mbarrier expect tx
 *
 * mbarrier_expect_tx(mbarrier, transaction_bytes)
 *
 */
TVM_DLL const Op &mbarrier_expect_tx();

/*!
 * \brief tvm intrinsics for ldmatrix
 *
 * ptx_ldmatrix(transposed, num, shared_addr, local_addr)
 *
 */
TVM_DLL const Op &ptx_ldmatrix();

/*!
 * \brief tvm intrinsics for stmatrix
 *
 * ptx_ldmatrix(transposed, num, shared_addr, int32_values...)
 *
 */
TVM_DLL const Op &ptx_stmatrix();

/*!
 * \brief tvm intrinsics for sync threads partial
 *
 * sync_thread_partial()
 *
 */
TVM_DLL const Op &sync_thread_partial();

/*!
 * \brief tvm intrinsics for copy unrolled
 *
 * copy_unrolled(dst, src, size, unroll_factor)
 *
 */
TVM_DLL const Op &copy_unrolled();

/*!
 * \brief Pack two b16 value into a b32 value
 *
 * int32 pack_b16(b16_value, b16_value)
 *
 */
TVM_DLL const Op &pack_b16();

/*!
 * \brief Issue a shared memory fence for async operations
 *
 * FenceProxyAsync()
 *
 */
TVM_DLL const Op &fence_proxy_async();

/*!
 * \brief Indicate arrival of warp issuing TMA_STORE
 *
 * tma_store_arrive()
 *
 */
TVM_DLL const Op &tma_store_arrive();

/*!
 * \brief Wait for TMA_STORE to finish
 *
 * tma_store_wait()
 *
 */
TVM_DLL const Op &tma_store_wait();

/*!
 * \brief Set reg hint for warp-specialized branched
 *
 * SetMaxNRegInc(num_reg, is_inc)
 *
 */
TVM_DLL const Op &set_max_nreg();

/*!
 * \brief No set reg hint for warp-specialized branched
 *
 * no_set_max_nreg()
 *
 */
TVM_DLL const Op &no_set_max_nreg();

/*!
 * \brief Wait the previous wgmma to finish
 *
 * wait_wgmma(num_mma)
 *
 */
TVM_DLL const Op &wait_wgmma();

/*!
 * \brief Synchronize all threads in a grid
 *
 * sync_grid()
 *
 */
TVM_DLL const Op &sync_grid();

/*!
 * \brief tvm intrinsic for loop continue
 *
 * loop_break()
 *
 */
TVM_DLL const Op &loop_break();

/*!
 * \brief tvm intrinsic for amd matrix core mfma instructions.
 *
 *  void tvm_mfma(StringImm shape, StringImm A_layout, StringImm B_layout,
 *               StringImm A_dtype, StringImm B_dtype, StringImm C_dtype,
 *               Var multiplicand_a, Expr a_index,
 *               Var multiplicand_b, Expr b_index,
 *               Var accumulator, Expr c_index);
 */
TVM_DLL const Op &tvm_mfma();

/*!
 * \brief tvm intrinsic for storing the result of AMD MFMA into a destination
 * pointer.
 *
 *        There is no real instruction that does that, but we want to hide
 * details of complex index manipulation behind this intrinsic to simplify TIR
 * lowering passes (e.g. LowerWarpMemory) like cuda ptx backend does.
 *
 * void tvm_mfma_store(IntImm m, IntImm n, Var dst_ptr, Var src_ptr, Expr
 * src_offset, Var dst_stride);
 */
TVM_DLL const Op &tvm_mfma_store();

/*!
 * \brief tvm intrinsic for amd rdna matrix core instructions.
 *
 *  void tvm_rdna_wmma(StringImm shape, StringImm A_layout, StringImm B_layout,
 *               StringImm A_dtype, StringImm B_dtype, StringImm C_dtype,
 *               Var multiplicand_a, Expr a_index,
 *               Var multiplicand_b, Expr b_index,
 *               Var accumulator, Expr c_index);
 */
TVM_DLL const Op &tvm_rdna_wmma();

/*!
 * \brief tvm intrinsic for storing the result of AMD RDNA WMMA into a
 * destination pointer.
 *
 *        There is no real instruction that does that, but we want to hide
 * details of complex index manipulation behind this intrinsic to simplify TIR
 * lowering passes (e.g. LowerWarpMemory) like cuda ptx backend does.
 *
 * void tvm_rdna_wmma_store(IntImm m, IntImm n, Var dst_ptr, Var src_ptr, Expr
 * src_offset, Var dst_stride);
 */
TVM_DLL const Op &tvm_rdna_wmma_store();

/*!
 * \brief tilelang intrinsic for general matrix multiplication (GEMM).
 *
 *  This op is used to represent a generic GEMM operation in tilelang.
 */
TVM_DLL const Op &tl_gemm();

/*!
 * \brief tilelang intrinsic for sparse matrix multiplication (GEMM with
 * sparsity).
 *
 *  This op is used to represent a sparse GEMM operation in tilelang.
 */
TVM_DLL const Op &tl_gemm_sp();

/*!
 * \brief tilelang intrinsic for shuffle elect.
 *
 *  This op is used to represent a shuffle elect operation in tilelang.
 */
TVM_DLL const Op &tl_shuffle_elect();

/*!
 * \brief Initialize a barrier for GPU-level synchronization
 *
 * void init_barrier_gpu(barrier, expected)
 */
TVM_DLL const Op init_barrier_gpu();

/*!
 * \brief Arrive at a barrier for GPU-level synchronization
 *
 * void arrive_barrier_gpu(barrier)
 */
TVM_DLL const Op &arrive_barrier_gpu();

/*!
 * \brief Wait at a barrier for GPU-level synchronization
 *
 * void wait_barrier_gpu(barrier)
 */
TVM_DLL const Op &wait_barrier_gpu();

/*!
 * \brief Synchronize at a barrier for GPU-level synchronization
 *
 * void sync_barrier_gpu(barrier)
 */
TVM_DLL const Op &sync_barrier_gpu();

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_BUILTIN_H_