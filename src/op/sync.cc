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

PrimExpr
BarrierAllBlocksSysOpNode::get_offset(const BufferLoadNode *load) const {
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

TIR_DEFINE_TL_BUILTIN(wait_eq).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(sync_barrier_gpu)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(sync_grid).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

BarrierAllBlocksSysOp::BarrierAllBlocksSysOp(Array<PrimExpr> args,
                                             BufferMap vmap) {
  ObjectPtr<BarrierAllBlocksSysOpNode> node =
      make_object<BarrierAllBlocksSysOpNode>();
  node->local_bar_addr = args[0];
  const auto *call = node->local_bar_addr.as<CallNode>();
  ICHECK(call) << "local_bar_addr must be a call node";
  ICHECK(call->op.same_as(builtin::address_of()))
      << "local_bar_addr must be address_of op";

  const auto *load = call->args[0].as<BufferLoadNode>();
  ICHECK(load) << "address_of argument must be a BufferLoad";
  node->offset = node->get_offset(load);
  node->local_bar = load->buffer;
  node->local_indices = load->indices;
  data_ = std::move(node);
  (void)vmap;
}

Stmt BarrierAllBlocksSysOpNode::Lower(const LowerArgs &T,
                                      arith::Analyzer *analyzer) const {
  (void)analyzer;
  Array<PrimExpr> new_args;
  std::stringstream ss;
  ss << "tl::barrier_all_blocks_sys";
  new_args.push_back(StringImm(ss.str()));

  PrimExpr bar_addr = MakeLocalBarAddr(T);
  PrimExpr rank = Call(DataType::Int(64), tl::get_rank(), {});
  PrimExpr num_ranks = Call(DataType::Int(64), tl::get_num_ranks(), {});
  PrimExpr local_base_ptr =
      Call(DataType::Handle(), tl::get_remote_base_ptr(), {rank});
  PrimExpr offset_to_base =
      Sub(Call(DataType::Handle(), tl::get_uintptr_t(), {bar_addr}),
          local_base_ptr);

  new_args.push_back(offset_to_base);
  new_args.push_back(rank);
  new_args.push_back(num_ranks);

  auto barrier_all_blocks_sys =
      Call(DataType::Handle(), builtin::call_extern(), new_args);
  return Evaluate(barrier_all_blocks_sys);
}

LayoutMap BarrierAllBlocksSysOpNode::InferLayout(const LayoutInferArgs &T,
                                                 InferLevel level) const {
  (void)T;
  (void)level;
  return {};
}

TileOperator BarrierAllBlocksSysOpNode::Clone() const {
  auto node = make_object<BarrierAllBlocksSysOpNode>(*this);
  return BarrierAllBlocksSysOp(node);
}

PrimExpr BarrierAllBlocksSysOpNode::MakeLocalBarAddr(const LowerArgs &T) const {
  const auto *call = local_bar_addr.as<CallNode>();
  ICHECK(call && call->op.same_as(builtin::address_of()))
      << "local_bar_addr must remain an address_of call";
  const auto *load = call->args[0].as<BufferLoadNode>();
  ICHECK(load) << "address_of must wrap a BufferLoad";
  Buffer buffer = load->buffer;
  if (T.buffer_remap.count(buffer)) {
    buffer = T.buffer_remap[buffer];
  }
  return Call(DataType::Handle(), builtin::address_of(),
              {BufferLoad(buffer, local_indices)});
}

TIR_REGISTER_TL_OP(BarrierAllBlocksSysOp, barrier_all_blocks_sys)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(fence_cta).set_num_inputs(0).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(fence_gpu).set_num_inputs(0).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(fence_sys).set_num_inputs(0).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK({ BarrierAllBlocksSysOpNode::RegisterReflection(); });

} // namespace tl
} // namespace tvm
