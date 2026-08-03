/*!
 * \file tl/op/nccl_gin.cc
 * \brief Lowering for the NCCL GIN inter-node operators.
 */

#include "nccl_gin.h"

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

#include <sstream>

#include "builtin.h"
#include "distributed.h"
#include "distributed_utils.h"
#include "operator.h"

namespace tvm {
namespace tl {

using namespace tirx;

// Map the DSL scope name onto an NCCL coop *type*. These are tag types in the
// device API, not enum values, and they appear here as a template argument, so
// this must name the type -- `ncclCoopCta()` would be a value and fail to
// substitute. The device wrapper default-constructs it internally.
static std::string CoopType(const std::string &scope) {
  if (scope == "thread") {
    return "ncclCoopThread";
  }
  if (scope == "warp") {
    return "ncclCoopWarp";
  }
  if (scope == "block") {
    return "ncclCoopCta";
  }
  LOG(FATAL) << "invalid GIN cooperation scope: " << scope;
  return "";
}

// Unlike the intra-node copies, the transfer size is a runtime argument to
// ncclGin::put rather than a template parameter, so a dynamic size is fine here
// and no constant-folding check is needed.

// `size` counts elements at the DSL surface, matching T.put_block, whose
// cp_block<N> takes N elements of a typed pointer. ncclGin::put takes bytes, so
// the conversion happens here rather than being pushed onto the user -- a GIN op
// whose size meant something different from the intra-node op it sits beside
// would be silently wrong by a factor of the element width.
static PrimExpr CopySizeInBytes(const PrimExpr &copy_size, const Buffer &buffer) {
  const int bits = buffer->dtype.bits() * buffer->dtype.lanes();
  ICHECK(bits % 8 == 0) << "GIN put requires a byte-addressable element type, got "
                        << buffer->dtype;
  return cast(DataType::UInt(64), copy_size) *
         make_const(DataType::UInt(64), bits / 8);
}

GinPutOp::GinPutOp(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<GinPutOpNode> node = tvm::ffi::make_object<GinPutOpNode>();
  node->src_addr = args[0];
  node->dst_addr = args[1];
  ICHECK(node->src_addr.as<CallNode>() &&
         node->src_addr.as<CallNode>()->op.same_as(builtin::address_of()))
      << "GIN put src must be address_of(...)";
  ICHECK(node->dst_addr.as<CallNode>() &&
         node->dst_addr.as<CallNode>()->op.same_as(builtin::address_of()))
      << "GIN put dst must be address_of(...)";

  const auto *src_load =
      node->src_addr.as<CallNode>()->args[0].as<BufferLoadNode>();
  const auto *dst_load =
      node->dst_addr.as<CallNode>()->args[0].as<BufferLoadNode>();
  ICHECK(src_load && dst_load) << "address_of must wrap BufferLoad nodes";

  node->src_buffer = src_load->buffer;
  node->dst_buffer = dst_load->buffer;
  node->src_indices = src_load->indices;
  node->dst_indices = dst_load->indices;

  // `size` is an element count, so the two sides must agree on what an element
  // is; otherwise the byte count computed from the source would under- or
  // over-write the destination.
  ICHECK_EQ(node->src_buffer->dtype, node->dst_buffer->dtype)
      << "GIN put requires matching src/dst dtypes, got " << node->src_buffer->dtype
      << " and " << node->dst_buffer->dtype;

  node->copy_size = args[2];
  node->peer = args[3];
  node->signal_id = args[4].as<IntImm>().value()->value;
  node->with_signal = bool(args[5].as<IntImm>().value()->value);
  node->scope = args[6].as<StringImm>().value()->value;
  data_ = std::move(node);
}

Stmt GinPutOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  (void)analyzer;
  Array<PrimExpr> new_args;
  std::stringstream ss;

  // Both offsets are computed device-side by tl::gin::arena_offset, which
  // subtracts the arena base published in the metadata table. Doing it here
  // instead would require the arena base as a compile-time value, which it is
  // not -- it differs per rank and is only known once the allocator has run.
  ss << (with_signal ? "tl::gin::put_signal_addr<" : "tl::gin::put_addr<")
     << CoopType(scope) << ">";
  new_args.push_back(StringImm(ss.str()));

  // Peer is a global rank: GIN puts go through the communicator-wide team, whose
  // rank space is global, unlike the node-local peer index the IPC path uses.
  new_args.push_back(peer);
  new_args.push_back(MakeRemappedAddress(T, dst_buffer, dst_indices));
  new_args.push_back(MakeRemappedAddress(T, src_buffer, src_indices));
  new_args.push_back(CopySizeInBytes(copy_size, src_buffer));
  if (with_signal) {
    new_args.push_back(IntImm(DataType::Int(32), signal_id));
  }

  return Evaluate(
      Call(DataType::Handle(), builtin::call_extern(), new_args));
}

LayoutMap GinPutOpNode::InferLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  (void)T;
  (void)level;
  return {};
}

TileOperator GinPutOpNode::Clone() const {
  auto node = tvm::ffi::make_object<GinPutOpNode>(*this);
  return GinPutOp(node);
}

GinSignalOp::GinSignalOp(Array<PrimExpr> args,
                         Map<String, ObjectRef> annotations) {
  ObjectPtr<GinSignalOpNode> node = tvm::ffi::make_object<GinSignalOpNode>();
  node->peer = args[0];
  node->signal_id = args[1].as<IntImm>().value()->value;
  node->scope = args[2].as<StringImm>().value()->value;
  data_ = std::move(node);
}

Stmt GinSignalOpNode::Lower(const LowerArgs &T,
                            arith::Analyzer *analyzer) const {
  (void)T;
  (void)analyzer;
  std::stringstream ss;
  ss << "tl::gin::signal_peer<" << CoopType(scope) << ">";
  Array<PrimExpr> new_args{StringImm(ss.str()), peer,
                           IntImm(DataType::Int(32), signal_id)};
  return Evaluate(Call(DataType::Handle(), builtin::call_extern(), new_args));
}

LayoutMap GinSignalOpNode::InferLayout(const LayoutInferArgs &T,
                                       InferLevel level) const {
  (void)T;
  (void)level;
  return {};
}

TileOperator GinSignalOpNode::Clone() const {
  auto node = tvm::ffi::make_object<GinSignalOpNode>(*this);
  return GinSignalOp(node);
}

GinWaitSignalOp::GinWaitSignalOp(Array<PrimExpr> args,
                                 Map<String, ObjectRef> annotations) {
  ObjectPtr<GinWaitSignalOpNode> node =
      tvm::ffi::make_object<GinWaitSignalOpNode>();
  node->least = args[0];
  node->signal_id = args[1].as<IntImm>().value()->value;
  node->scope = args[2].as<StringImm>().value()->value;
  data_ = std::move(node);
}

Stmt GinWaitSignalOpNode::Lower(const LowerArgs &T,
                                arith::Analyzer *analyzer) const {
  (void)T;
  (void)analyzer;
  std::stringstream ss;
  ss << "tl::gin::wait_signal<" << CoopType(scope) << ">";
  Array<PrimExpr> new_args{StringImm(ss.str()),
                           IntImm(DataType::Int(32), signal_id),
                           cast(DataType::UInt(64), least)};
  return Evaluate(Call(DataType::Handle(), builtin::call_extern(), new_args));
}

LayoutMap GinWaitSignalOpNode::InferLayout(const LayoutInferArgs &T,
                                           InferLevel level) const {
  (void)T;
  (void)level;
  return {};
}

TileOperator GinWaitSignalOpNode::Clone() const {
  auto node = tvm::ffi::make_object<GinWaitSignalOpNode>(*this);
  return GinWaitSignalOp(node);
}

GinFlushOp::GinFlushOp(Array<PrimExpr> args,
                       Map<String, ObjectRef> annotations) {
  ObjectPtr<GinFlushOpNode> node = tvm::ffi::make_object<GinFlushOpNode>();
  node->scope = args[0].as<StringImm>().value()->value;
  data_ = std::move(node);
}

Stmt GinFlushOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  (void)T;
  (void)analyzer;
  std::stringstream ss;
  ss << "tl::gin::flush<" << CoopType(scope) << ">";
  Array<PrimExpr> new_args{StringImm(ss.str())};
  return Evaluate(Call(DataType::Handle(), builtin::call_extern(), new_args));
}

LayoutMap GinFlushOpNode::InferLayout(const LayoutInferArgs &T,
                                      InferLevel level) const {
  (void)T;
  (void)level;
  return {};
}

TileOperator GinFlushOpNode::Clone() const {
  auto node = tvm::ffi::make_object<GinFlushOpNode>(*this);
  return GinFlushOp(node);
}

// kUpdateState, matching the intra-node put/get: these mutate memory (locally or
// remotely) and must not be reordered or elided.
TIR_REGISTER_TL_TILE_OP(GinPutOp, gin_put)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kUpdateState));

TIR_REGISTER_TL_TILE_OP(GinSignalOp, gin_signal)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kUpdateState));

TIR_REGISTER_TL_TILE_OP(GinWaitSignalOp, gin_wait_signal)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kUpdateState));

TIR_REGISTER_TL_TILE_OP(GinFlushOp, gin_flush)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kUpdateState));

TVM_FFI_STATIC_INIT_BLOCK() {
  GinPutOpNode::RegisterReflection();
  GinSignalOpNode::RegisterReflection();
  GinWaitSignalOpNode::RegisterReflection();
  GinFlushOpNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
