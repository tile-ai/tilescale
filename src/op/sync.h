/*!
 * \file tl/op/sync.h
 * \brief Synchronization intrinsics.
 *
 */

#ifndef TVM_TL_OP_SYNC_H_
#define TVM_TL_OP_SYNC_H_

#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>

#include "op.h"

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Initialize a barrier for GPU-level synchronization
 *
 * void init_barrier_gpu(barrier, expected)
 */
TVM_DLL const Op &init_barrier_gpu();

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

/*!
 * \brief Synchronize all blocks at a system-level barrier
 *
 * void barrier_all_blocks_sys(barrier, rank, num_ranks)
 *
 */
class BarrierAllBlocksSysOp : public Operator {
public:
  BarrierAllBlocksSysOp(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  static const Op &Get();

  std::unique_ptr<Operator> Clone() const final {
    return std::make_unique<BarrierAllBlocksSysOp>(*this);
  }

  PrimExpr get_offset(const BufferLoadNode *load);

private:
  PrimExpr local_bar_addr;
  PrimExpr offset;
  Buffer local_bar;
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_SYNC_H_