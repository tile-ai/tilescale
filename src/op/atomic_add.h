/*!
 * \file tl/op/atomic_add.h
 * \brief Atomic addition operations for concurrent memory updates
 */

#ifndef TVM_TL_OP_ATOMIC_ADD_H_
#define TVM_TL_OP_ATOMIC_ADD_H_

#include "operator.h"
#include "parallel.h"

namespace tvm {
namespace tl {

using namespace tir;

/// Node class for atomic addition operations
class AtomicAddNode : public TileOperatorNode {
public:
  Buffer src, dst; ///< Source and destination buffers
  Array<Range> src_range,
      dst_range;          ///< Access ranges for source and destination
  IntImm use_tma;         ///< Whether to use TMA for memory operations
  IntImm coalesced_width; ///< Width for memory coalescing optimization
  IntImm memory_order;    ///< Memory order for atomic operations

  mutable ParallelOp par_op_; ///< Associated parallel operation
  static constexpr const char *_type_key = "tl.AtomicAdd";
  TVM_DECLARE_FINAL_OBJECT_INFO(AtomicAddNode, TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) const;

  static const Op &Get();
  TileOperator Clone() const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AtomicAddNode>()
        .def_ro("src", &AtomicAddNode::src)
        .def_ro("dst", &AtomicAddNode::dst)
        .def_ro("src_range", &AtomicAddNode::src_range)
        .def_ro("dst_range", &AtomicAddNode::dst_range)
        .def_ro("use_tma", &AtomicAddNode::use_tma)
        .def_ro("coalesced_width", &AtomicAddNode::coalesced_width)
        .def_ro("memory_order", &AtomicAddNode::memory_order);
  }

  bool SEqualReduce(const AtomicAddNode *other, SEqualReducer equal) const {
    return equal(src, other->src) && equal(dst, other->dst) &&
           equal(src_range, other->src_range) &&
           equal(dst_range, other->dst_range) &&
           equal(use_tma, other->use_tma) &&
           equal(coalesced_width, other->coalesced_width) &&
           equal(memory_order, other->memory_order);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(src);
    hash_reduce(dst);
    hash_reduce(src_range);
    hash_reduce(dst_range);
    hash_reduce(use_tma);
    hash_reduce(coalesced_width);
    hash_reduce(memory_order);
  }

  static constexpr bool _type_has_method_sequal_reduce = true;
  static constexpr bool _type_has_method_shash_reduce = true;

protected:
  /// Create SIMT-style parallel loop structure
  For MakeSIMTLoop(arith::Analyzer *analyzer) const;
  /// Generate iteration variables for loop nest
  Array<IterVar> MakeIterVars() const;
  /// Generate buffer indices from iteration variables
  Array<PrimExpr> MakeIndices(const Array<IterVar> &ivs, int src_dst) const;
  /// Return buffer indices and size
  std::pair<Array<PrimExpr>, PrimExpr> ReturnIndicesAndSize(int src_dst) const;
  /// Create boundary predicate for memory safety
  PrimExpr MakePredicate(arith::Analyzer *analyzer, const Array<IterVar> &ivs,
                         Array<PrimExpr> extents, int src_dst) const;
};

/// Wrapper class for atomic addition operations
class AtomicAdd : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(AtomicAdd, TileOperator, AtomicAddNode);
  TVM_DLL AtomicAdd(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_ATOMIC_ADD_H_