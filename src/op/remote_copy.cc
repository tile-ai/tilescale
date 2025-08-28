/*!
 * \file tl/op/remote_copy.cc
 * \brief Remote copy operator.
 *
 */

#include "remote_copy.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "../target/cuda.h"
#include "../target/utils.h"
#include "builtin.h"
#include "distributed.h"

namespace tvm {
namespace tl {

using namespace tir;

PrimExpr get_offset(const BufferLoadNode *load) {
  PrimExpr offset = 0;
  PrimExpr stride = 1;
  auto buffer_shape = load->buffer->shape;
  for (int i = load->indices.size() - 1; i >= 0; i--) {
    offset += load->indices[i] * stride;
    stride *= buffer_shape[i];
  }
  return div(offset * load->dtype.bits(), 8);
}

RemoteCopyOp::RemoteCopyOp(Array<PrimExpr> args, BufferMap vmap) {
  src_addr = args[0];
  dst_addr = args[1];
  ICHECK(src_addr.as<CallNode>()) << "src_addr must be a call node";
  ICHECK(src_addr.as<CallNode>()->op.same_as(builtin::address_of()))
      << "src_addr must be address_of op";
  ICHECK(dst_addr.as<CallNode>()) << "dst_addr must be a call node";
  ICHECK(dst_addr.as<CallNode>()->op.same_as(builtin::address_of()))
      << "dst_addr must be address_of op";

  src_offset =
      get_offset(src_addr.as<CallNode>()->args[0].as<BufferLoadNode>());
  dst_offset =
      get_offset(dst_addr.as<CallNode>()->args[0].as<BufferLoadNode>());
  src_buffer = src_addr.as<CallNode>()->args[0].as<BufferLoadNode>()->buffer;
  dst_buffer = dst_addr.as<CallNode>()->args[0].as<BufferLoadNode>()->buffer;

  copy_size = args[2];
  dst_pe = args[3];
  unroll_factor = args[4].as<IntImm>().value()->value;
  if (dst_pe.defined()) {
    is_symmetric = true;
  }
}

Stmt RemoteCopyOp::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  Array<PrimExpr> new_args;
  std::stringstream ss;
  ss << "tl::cp_unrolled<" << copy_size << ", " << unroll_factor << ">";
  new_args.push_back(StringImm(ss.str()));
  if (is_symmetric) {
    PrimExpr local_rank = Call(DataType::Int(64), tl::get_rank(), {});
    PrimExpr local_base_ptr =
        Call(DataType::Handle(), tl::get_remote_base_ptr(), {local_rank});
    PrimExpr offset_to_base =
        Sub(Call(DataType::Handle(), tl::get_uintptr_t(), {dst_addr}),
            local_base_ptr);
    new_args.push_back(
        Call(DataType::Handle(), tl::get_remote_base_ptr(), {dst_pe}) +
        offset_to_base);
  } else {
    new_args.push_back(dst_addr);
  }
  new_args.push_back(src_addr);
  auto unrolled_copy =
      Call(DataType::Handle(), builtin::call_extern(), new_args);
  return Evaluate(unrolled_copy);
}

TIR_REGISTER_TL_OP(RemoteCopyOp, remote_copy)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

} // namespace tl
} // namespace tvm
