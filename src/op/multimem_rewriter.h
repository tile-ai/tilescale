/*!
 * \file tl/op/multimem_rewriter.h
 * \brief Post-process IR to replace vectorized BufferLoad/Store on mcast
 * buffers with multimem call_extern instructions.
 */

#ifndef TVM_TL_OP_MULTIMEM_REWRITER_H_
#define TVM_TL_OP_MULTIMEM_REWRITER_H_

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr_functor.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include "multimem.h"

namespace tvm {
namespace tl {

using namespace tirx;

class PlainMulticastAccessFinder : public StmtExprVisitor {
public:
  explicit PlainMulticastAccessFinder(Buffer target)
      : target_(std::move(target)) {}

  bool Find(const Stmt &stmt) {
    VisitStmt(stmt);
    return found_;
  }

private:
  Buffer target_;
  bool found_{false};

  void VisitStmt_(const BufferStoreNode *op) override {
    if (op->buffer.same_as(target_)) {
      found_ = true;
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) override {
    if (op->buffer.same_as(target_)) {
      found_ = true;
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) override {
    // BufferLoad under address_of denotes an address; it is not a plain load.
    if (op->op.same_as(builtin::address_of())) {
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }
};

/*!
 * \brief Rewrites BufferLoad/BufferStore involving a multicast buffer
 *        into multimem call_extern instructions.
 *
 * After ParallelOp + VectorizeLoop, the IR contains ForKind::kVectorized loops
 * with scalar loop variables (Ramp is not materialized until codegen).
 * This rewriter detects two patterns:
 *
 * 1. ForKind::kVectorized loop containing mcast buffer access:
 *    for (vec: kVectorized, extent=N) { dst[base+vec] = mcast[base+vec] }
 *    → call_extern("LdReduceVN<...>::run", &dst[base], &mcast[base])
 *
 * 2. Scalar BufferStore with Ramp indices (if vectorization produced Ramp):
 *    dst[Ramp(base,1,N)] = mcast[Ramp(base,1,N)]
 *    → call_extern("LdReduceVN<...>::run", &dst[base], &mcast[base])
 */
class MultimemRewriter : public StmtExprMutator {
public:
  MultimemRewriter(Buffer mcast_buf, MultimemMode mode, int reduce_op)
      : mcast_buf_(std::move(mcast_buf)), mode_(mode), reduce_op_(reduce_op) {}

  Stmt Rewrite(Stmt stmt) {
    Stmt result = VisitStmt(std::move(stmt));
    ICHECK(!PlainMulticastAccessFinder(mcast_buf_).Find(result))
        << "multimem lowering left a plain multicast BufferLoad/BufferStore; "
           "this access pattern cannot be represented safely:\n"
        << result;
    return result;
  }

protected:
  /*!
   * \brief Handle ForKind::kVectorized loops.
   * If the loop body is a single BufferStore involving the mcast buffer,
   * replace the entire loop with a single vectorized multimem call.
   */
  Stmt VisitStmt_(const ForNode *op) override {
    if (op->kind == ForKind::kVectorized) {
      auto extent_ptr = op->extent.as<IntImmNode>();
      if (extent_ptr && extent_ptr->value > 0) {
        int lanes = static_cast<int>(extent_ptr->value);
        auto result = TryRewriteVectorizedLoop(op, lanes);
        if (result.defined()) {
          return result;
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  /*!
   * \brief Handle scalar BufferStore and Ramp-indexed vector accesses.
   */
  Stmt VisitStmt_(const BufferStoreNode *op) override {
    MatchedAccess access;
    if (MatchBufferStore(op, &access) && NormalizeRampAccess(&access)) {
      return EmitMatchedAccess(access);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

private:
  struct MatchedAccess {
    Buffer local_buf;
    Array<PrimExpr> local_indices;
    Buffer mcast_buf;
    Array<PrimExpr> mcast_indices;
    int lanes{1};
    Optional<Var> loop_var;
    PrimExpr loop_min;
    Array<PrimExpr> statement_predicates;
    Optional<PrimExpr> source_predicate;
  };

  Buffer mcast_buf_;
  MultimemMode mode_;
  int reduce_op_;
  arith::Analyzer analyzer_;

  bool IsZeroValue(const PrimExpr &value) const {
    if (is_zero(value)) {
      return true;
    }
    if (auto *float_imm = value.as<FloatImmNode>()) {
      return float_imm->value == 0.0;
    }
    return false;
  }

  bool MatchLdReduceValue(const PrimExpr &value, const BufferLoadNode **load,
                          Optional<PrimExpr> *predicate) const {
    if (auto *direct_load = value.as<BufferLoadNode>()) {
      *load = direct_load;
      *predicate = std::nullopt;
      return true;
    }
    if (auto *call = value.as<CallNode>()) {
      if (call->op.same_as(builtin::if_then_else()) && call->args.size() == 3 &&
          IsZeroValue(call->args[2])) {
        if (auto *guarded_load = call->args[1].as<BufferLoadNode>()) {
          *load = guarded_load;
          *predicate = call->args[0];
          return true;
        }
      }
    }
    return false;
  }

  // Normalize the three direct modes to local/multicast buffers.  The value is
  // deliberately restricted to the exact forms emitted by MakeSIMTLoop.
  bool MatchBufferStore(const BufferStoreNode *store,
                        MatchedAccess *access) const {
    const BufferLoadNode *load = nullptr;
    Optional<PrimExpr> source_predicate;
    if (mode_ == MultimemMode::kLdReduce) {
      if (!MatchLdReduceValue(store->value, &load, &source_predicate) ||
          !load->buffer.same_as(mcast_buf_)) {
        return false;
      }
      access->local_buf = store->buffer;
      access->local_indices = store->indices;
      access->mcast_buf = load->buffer;
      access->mcast_indices = load->indices;
      access->source_predicate = source_predicate;
      return true;
    }

    load = store->value.as<BufferLoadNode>();
    if (!load || !store->buffer.same_as(mcast_buf_)) {
      return false;
    }
    access->local_buf = load->buffer;
    access->local_indices = load->indices;
    access->mcast_buf = store->buffer;
    access->mcast_indices = store->indices;
    return true;
  }

  bool NormalizeRampAccess(MatchedAccess *access) const {
    int local_lanes = 1;
    int mcast_lanes = 1;
    if (!TryGetLanes(access->local_indices, &local_lanes) ||
        !TryGetLanes(access->mcast_indices, &mcast_lanes) ||
        local_lanes != mcast_lanes) {
      return false;
    }
    if (access->source_predicate.defined() &&
        !HasCompatibleLanes(access->source_predicate.value(), local_lanes)) {
      return false;
    }
    access->lanes = local_lanes;
    return true;
  }

  /*!
   * \brief Try to rewrite a kVectorized for-loop containing a mcast
   * BufferStore. Returns the replacement Stmt, or undefined if the pattern
   * doesn't match.
   */
  Stmt TryRewriteVectorizedLoop(const ForNode *op, int lanes) {
    // The body should be a single BufferStore, possibly wrapped in one or more
    // predicates without else branches.
    Stmt body = op->body;
    Array<PrimExpr> predicates;
    while (auto *ite = body.as<IfThenElseNode>()) {
      if (ite->else_case.defined()) {
        return Stmt();
      }
      predicates.push_back(ite->condition);
      body = ite->then_case;
    }
    const BufferStoreNode *store = body.as<BufferStoreNode>();
    if (!store)
      return Stmt();

    MatchedAccess access;
    if (!MatchBufferStore(store, &access))
      return Stmt();

    int local_lanes = 1;
    int mcast_lanes = 1;
    if (!TryGetLanes(access.local_indices, &local_lanes) ||
        !TryGetLanes(access.mcast_indices, &mcast_lanes) || local_lanes != 1 ||
        mcast_lanes != 1 ||
        (access.source_predicate.defined() &&
         !HasCompatibleLanes(access.source_predicate.value(), 1))) {
      return Stmt();
    }
    for (const auto &predicate : predicates) {
      if (!HasCompatibleLanes(predicate, 1)) {
        return Stmt();
      }
    }

    access.lanes = lanes;
    access.loop_var = op->loop_var;
    access.loop_min = op->min;
    access.statement_predicates = std::move(predicates);
    return EmitMatchedAccess(access);
  }

  // Both vectorized-loop and Ramp forms reach this emitter.  Wide calls are
  // used only for physically adjacent lanes; all other shapes become V1.
  Stmt EmitMatchedAccess(const MatchedAccess &access) {
    Stmt vector_path = EmitChunks(access);
    if (access.statement_predicates.empty() &&
        !access.source_predicate.defined()) {
      return vector_path;
    }

    // A wide call is legal only when every original predicate holds.  The
    // scalar path preserves statement guards and load zero-fill lane by lane.
    PrimExpr all_lanes_valid = make_const(DataType::Bool(), true);
    Array<Stmt> scalar_calls;
    for (int i = 0; i < access.lanes; ++i) {
      PrimExpr statement_valid = make_const(DataType::Bool(), true);
      for (const auto &predicate : access.statement_predicates) {
        statement_valid =
            And(statement_valid, MaterializeLane(predicate, access, i));
      }
      PrimExpr source_valid = make_const(DataType::Bool(), true);
      if (access.source_predicate.defined()) {
        source_valid =
            MaterializeLane(access.source_predicate.value(), access, i);
      }
      all_lanes_valid =
          And(all_lanes_valid, And(statement_valid, source_valid));

      Array<PrimExpr> local_indices =
          MaterializeIndices(access.local_indices, access, i);
      Stmt scalar_call = MakeMultimemCall(
          access.local_buf, local_indices, access.mcast_buf,
          MaterializeIndices(access.mcast_indices, access, i), 1);
      Stmt lane_action = scalar_call;
      if (access.source_predicate.defined()) {
        Stmt zero =
            BufferStore(access.local_buf, make_zero(access.local_buf->dtype),
                        local_indices);
        lane_action = IfThenElse(source_valid, scalar_call, zero);
      }
      if (!access.statement_predicates.empty()) {
        lane_action = IfThenElse(statement_valid, lane_action);
      }
      scalar_calls.push_back(lane_action);
    }
    Stmt scalar_path = scalar_calls.size() == 1
                           ? scalar_calls[0]
                           : SeqStmt(std::move(scalar_calls));
    return IfThenElse(all_lanes_valid, vector_path, scalar_path);
  }

  Stmt EmitChunks(const MatchedAccess &access) {
    Array<Stmt> calls;
    for (int lane = 0; lane < access.lanes;) {
      int width = SelectChunkWidth(access, lane);
      calls.push_back(MakeMultimemCall(
          access.local_buf,
          MaterializeIndices(access.local_indices, access, lane),
          access.mcast_buf,
          MaterializeIndices(access.mcast_indices, access, lane), width));
      lane += width;
    }
    return calls.size() == 1 ? calls[0] : SeqStmt(std::move(calls));
  }

  int SelectChunkWidth(const MatchedAccess &access, int lane) {
    for (int width : {4, 2}) {
      if (lane + width <= access.lanes &&
          IsUnitContiguous(access, lane, width) &&
          IsNaturallyAligned(access.local_buf, access.local_indices, access,
                             lane, width) &&
          IsNaturallyAligned(access.mcast_buf, access.mcast_indices, access,
                             lane, width)) {
        return width;
      }
    }
    return 1;
  }

  bool TryGetLanes(const Array<PrimExpr> &indices, int *lanes) const {
    *lanes = 1;
    for (const auto &index : indices) {
      int index_lanes = index.dtype().lanes();
      if (index_lanes > 1) {
        if (*lanes != 1 && *lanes != index_lanes) {
          return false;
        }
        *lanes = index_lanes;
      }
    }
    return true;
  }

  bool HasCompatibleLanes(const PrimExpr &value, int lanes) const {
    return value.dtype().lanes() == 1 || value.dtype().lanes() == lanes;
  }

  Array<PrimExpr> MaterializeIndices(const Array<PrimExpr> &indices,
                                     const MatchedAccess &access,
                                     int lane) const {
    if (access.loop_var.defined()) {
      Map<Var, PrimExpr> vmap;
      const Var &var = access.loop_var.value();
      vmap.Set(var, access.loop_min + IntImm(var.dtype(), lane));
      return indices.Map(
          [&vmap](const PrimExpr &index) { return Substitute(index, vmap); });
    }

    Array<PrimExpr> result;
    for (const auto &index : indices) {
      if (auto *ramp = index.as<RampNode>()) {
        result.push_back(ramp->base + ramp->stride * lane);
      } else if (auto *broadcast = index.as<BroadcastNode>()) {
        result.push_back(broadcast->value);
      } else if (index.dtype().lanes() > 1) {
        result.push_back(Shuffle::ExtractElement(index, lane));
      } else {
        result.push_back(index);
      }
    }
    return result;
  }

  PrimExpr MaterializeLane(const PrimExpr &value, const MatchedAccess &access,
                           int lane) const {
    PrimExpr lane_value = value;
    if (access.loop_var.defined()) {
      const Var &var = access.loop_var.value();
      Map<Var, PrimExpr> vmap;
      vmap.Set(var, access.loop_min + IntImm(var.dtype(), lane));
      lane_value = Substitute(lane_value, vmap);
    }
    return ExtractLane(lane_value, lane);
  }

  PrimExpr ExtractLane(const PrimExpr &value, int lane) const {
    if (auto *broadcast = value.as<BroadcastNode>()) {
      return broadcast->value;
    }
    if (value.dtype().lanes() > 1) {
      return Shuffle::ExtractElement(value, lane);
    }
    return value;
  }

  bool IsUnitContiguous(const MatchedAccess &access, int lane, int width) {
    for (int i = lane; i + 1 < lane + width; ++i) {
      if (!HasUnitPhysicalStride(access.local_buf, access.local_indices, access,
                                 i) ||
          !HasUnitPhysicalStride(access.mcast_buf, access.mcast_indices, access,
                                 i)) {
        return false;
      }
    }
    return true;
  }

  bool HasUnitPhysicalStride(const Buffer &buffer,
                             const Array<PrimExpr> &indices,
                             const MatchedAccess &access, int lane) {
    Array<PrimExpr> current =
        buffer->ElemOffset(MaterializeIndices(indices, access, lane));
    Array<PrimExpr> next =
        buffer->ElemOffset(MaterializeIndices(indices, access, lane + 1));
    if (current.size() != 1 || next.size() != 1) {
      return false;
    }
    PrimExpr stride = analyzer_.Simplify(next[0] - current[0]);
    return analyzer_.CanProveEqual(stride, make_const(stride.dtype(), 1));
  }

  bool IsNaturallyAligned(const Buffer &buffer, const Array<PrimExpr> &indices,
                          const MatchedAccess &access, int lane, int width) {
    int element_bytes = buffer->dtype.bytes() * buffer->dtype.lanes();
    int required_alignment = width * element_bytes;
    if (buffer->data_alignment < required_alignment ||
        buffer->data_alignment % required_alignment != 0) {
      return false;
    }
    Array<PrimExpr> offsets =
        buffer->ElemOffset(MaterializeIndices(indices, access, lane));
    if (offsets.size() != 1) {
      return false;
    }
    PrimExpr offset = analyzer_.Simplify(offsets[0]);
    return analyzer_.CanProveEqual(
        FloorMod(offset, make_const(offset.dtype(), width)), 0);
  }

  /*!
   * \brief Create the call_extern for a multimem instruction.
   */
  Stmt MakeMultimemCall(const Buffer &local_buf,
                        const Array<PrimExpr> &local_indices,
                        const Buffer &mc_buf, const Array<PrimExpr> &mc_indices,
                        int lanes) const {
    std::string func_name =
        multimem_detail::FuncName(mode_, reduce_op_, lanes, local_buf->dtype);

    Array<PrimExpr> args;
    args.push_back(StringImm(func_name));

    if (mode_ == MultimemMode::kLdReduce) {
      args.push_back(multimem_detail::MakeAddress(local_buf, local_indices));
      args.push_back(multimem_detail::MakeAddress(mc_buf, mc_indices));
    } else {
      args.push_back(multimem_detail::MakeAddress(mc_buf, mc_indices));
      args.push_back(multimem_detail::MakeAddress(local_buf, local_indices));
    }

    auto call = Call(DataType::Handle(), builtin::call_extern(), args);
    return Evaluate(call);
  }
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_MULTIMEM_REWRITER_H_
