/*!
 * \file tl/op/nccl_gin.h
 * \brief Inter-node put/signal operators backed by the NCCL GIN device API.
 *
 * These are kept apart from the put/get in remote_copy.h because they use a
 * different addressing model rather than a different transport for the same one.
 * remote_copy computes `peer_base + (local_ptr - local_base)`, which needs the
 * peer's memory mapped locally; GIN names memory as an (ncclWindow_t, offset)
 * pair, which is what allows it to reach a rank on another node whose allocation
 * has no local virtual address. Sharing an op between the two would mean a
 * single node carrying both meanings of "peer" and both address computations.
 */

#ifndef TVM_TL_OP_NCCL_GIN_H_
#define TVM_TL_OP_NCCL_GIN_H_

#include <tvm/target/target.h>
#include <tvm/tirx/stmt_functor.h>

#include "../layout/layout.h"
#include "operator.h"

namespace tvm {
namespace tl {

using namespace tirx;

/*!
 * \brief One-sided inter-node put, optionally incrementing a remote signal.
 */
class GinPutOpNode : public TileOperatorNode {
public:
  PrimExpr src_addr;   ///< address_of the local source buffer element
  PrimExpr dst_addr;   ///< address_of the destination buffer element
  PrimExpr copy_size;  ///< Bytes to transfer
  PrimExpr peer;       ///< Destination *global* rank within the world team
  int signal_id;       ///< Signal to increment on arrival, when with_signal
  bool with_signal;    ///< Whether completion increments a remote signal
  std::string scope;   ///< Cooperation scope: {thread, warp, block}
  Buffer src_buffer;   ///< Source buffer, for arena offset computation
  Buffer dst_buffer;   ///< Destination buffer
  Array<PrimExpr> src_indices; ///< Source indices
  Array<PrimExpr> dst_indices; ///< Destination indices

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.GinPutOp", GinPutOpNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GinPutOpNode>()
        .def_ro("src_addr", &GinPutOpNode::src_addr)
        .def_ro("dst_addr", &GinPutOpNode::dst_addr)
        .def_ro("copy_size", &GinPutOpNode::copy_size)
        .def_ro("peer", &GinPutOpNode::peer)
        .def_ro("signal_id", &GinPutOpNode::signal_id)
        .def_ro("with_signal", &GinPutOpNode::with_signal)
        .def_ro("scope", &GinPutOpNode::scope)
        .def_ro("src_buffer", &GinPutOpNode::src_buffer)
        .def_ro("dst_buffer", &GinPutOpNode::dst_buffer)
        .def_ro("src_indices", &GinPutOpNode::src_indices)
        .def_ro("dst_indices", &GinPutOpNode::dst_indices);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const override;
};

class GinPutOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GinPutOp, TileOperator,
                                             GinPutOpNode);
  TVM_DLL GinPutOp(Array<PrimExpr> args, Map<String, ObjectRef> annotations =
                                             Map<String, ObjectRef>());
  static const Op &Get();
};

/*!
 * \brief Increment a signal on a peer without moving payload.
 */
class GinSignalOpNode : public TileOperatorNode {
public:
  PrimExpr peer;     ///< Destination global rank
  int signal_id;     ///< Signal to increment
  std::string scope; ///< Cooperation scope

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.GinSignalOp", GinSignalOpNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GinSignalOpNode>()
        .def_ro("peer", &GinSignalOpNode::peer)
        .def_ro("signal_id", &GinSignalOpNode::signal_id)
        .def_ro("scope", &GinSignalOpNode::scope);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const override;
};

class GinSignalOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GinSignalOp, TileOperator,
                                             GinSignalOpNode);
  TVM_DLL GinSignalOp(Array<PrimExpr> args, Map<String, ObjectRef> annotations =
                                                Map<String, ObjectRef>());
  static const Op &Get();
};

/*!
 * \brief Block until a signal reaches a cumulative threshold.
 */
class GinWaitSignalOpNode : public TileOperatorNode {
public:
  PrimExpr least;    ///< Cumulative count to wait for
  int signal_id;     ///< Signal to observe
  std::string scope; ///< Cooperation scope

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.GinWaitSignalOp", GinWaitSignalOpNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GinWaitSignalOpNode>()
        .def_ro("least", &GinWaitSignalOpNode::least)
        .def_ro("signal_id", &GinWaitSignalOpNode::signal_id)
        .def_ro("scope", &GinWaitSignalOpNode::scope);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const override;
};

class GinWaitSignalOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GinWaitSignalOp, TileOperator,
                                             GinWaitSignalOpNode);
  TVM_DLL GinWaitSignalOp(Array<PrimExpr> args,
                          Map<String, ObjectRef> annotations =
                              Map<String, ObjectRef>());
  static const Op &Get();
};

/*!
 * \brief Make put source buffers reusable. Implies nothing about remote arrival.
 */
class GinFlushOpNode : public TileOperatorNode {
public:
  std::string scope; ///< Cooperation scope

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.GinFlushOp", GinFlushOpNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GinFlushOpNode>().def_ro("scope", &GinFlushOpNode::scope);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const override;
};

class GinFlushOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GinFlushOp, TileOperator,
                                             GinFlushOpNode);
  TVM_DLL GinFlushOp(Array<PrimExpr> args, Map<String, ObjectRef> annotations =
                                               Map<String, ObjectRef>());
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_NCCL_GIN_H_
