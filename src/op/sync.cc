/*!
 * \file tl/op/sync.cc
 * \brief Synchronization intrinsics.
 *
 */

#include "sync.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "distributed.h"

namespace tvm {
namespace tl {

using namespace tir;

PrimExpr BarrierAllBlocksSysOp::get_offset(const BufferLoadNode *load) {
  PrimExpr offset = 0;
  PrimExpr stride = 1;
  auto buffer_shape = load->buffer->shape;
  for (int i = load->indices.size() - 1; i >= 0; i--) {
    offset += load->indices[i] * stride;
    stride *= buffer_shape[i];
  }
  return div(offset * load->dtype.bits(), 8);
}

#define TIR_DEFINE_TL_BUILTIN(OpName)                                          \
  const Op &OpName() {                                                         \
    static const Op &op = Op::Get("tl." #OpName);                              \
    return op;                                                                 \
  }                                                                            \
  TVM_REGISTER_OP("tl." #OpName)                                               \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", #OpName)

TIR_DEFINE_TL_BUILTIN(init_barrier_gpu)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(arrive_barrier_gpu)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(wait_barrier_gpu)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(sync_barrier_gpu)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

BarrierAllBlocksSysOp::BarrierAllBlocksSysOp(Array<PrimExpr> args,
                                             BufferMap vmap) {
  local_bar_addr = args[0];
  ICHECK(local_bar_addr.as<CallNode>()) << "local_bar_addr must be a call node";
  ICHECK(local_bar_addr.as<CallNode>()->op.same_as(builtin::address_of()))
      << "local_bar_addr must be address_of op";

  offset = this->get_offset(
      local_bar_addr.as<CallNode>()->args[0].as<BufferLoadNode>());
  local_bar =
      local_bar_addr.as<CallNode>()->args[0].as<BufferLoadNode>()->buffer;
}

Stmt BarrierAllBlocksSysOp::Lower(const LowerArgs &T,
                                  arith::Analyzer *analyzer) const {
  Array<PrimExpr> new_args;
  std::stringstream ss;
  ss << "tl::barrier_all_blocks_sys";
  new_args.push_back(StringImm(ss.str()));

  PrimExpr rank = Call(DataType::Int(64), tl::get_rank(), {});
  PrimExpr num_ranks = Call(DataType::Int(64), tl::get_num_ranks(), {});
  PrimExpr local_base_ptr =
      Call(DataType::Handle(), tl::get_remote_base_ptr(), {rank});
  PrimExpr offset_to_base =
      Sub(Call(DataType::Handle(), tl::get_uintptr_t(), {local_bar_addr}),
          local_base_ptr);

  new_args.push_back(offset_to_base);
  new_args.push_back(rank);
  new_args.push_back(num_ranks);

  auto barrier_all_blocks_sys =
      Call(DataType::Handle(), builtin::call_extern(), new_args);
  return Evaluate(barrier_all_blocks_sys);
}

TIR_REGISTER_TL_OP(BarrierAllBlocksSysOp, barrier_all_blocks_sys)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

} // namespace tl
} // namespace tvm
